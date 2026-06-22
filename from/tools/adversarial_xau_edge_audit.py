#!/usr/bin/env python3
"""Adversarial audit of the saved XAU cross-market edge artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_selected_model_portfolio as bp
import cross_market_edge_search as cm
import forward_holdout_search as fh
import long_history_yahoo_forward_search as lh


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
MASTER = r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet"
FORWARD_XAU = str(DATA / "derived" / "xauusd_dukascopy_5m_forward_20260522_20260618.csv")


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def save_json(name: str, obj: dict) -> None:
    (REPORTS / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def selected(kind: str) -> list[dict]:
    if kind == "5m":
        return load_json("selected_10_models_oldrank_forwardgate.json")["models"]
    return load_json("selected_10_models_60m_730d_oldrank_forwardgate.json")["models"]


def load_px(kind: str, period: str = "forward") -> pd.DataFrame:
    models = selected(kind)
    symbols = sorted({m["leader"] for m in models})
    ns = argparse.Namespace(
        selected="",
        out="",
        trades_out="",
        capital=5_000_000.0,
        cap_gross_exposure=False,
        period=period,
        forward_xau=FORWARD_XAU,
        master=MASTER,
        cache_dir=str(DATA / "cache" / "yahoo"),
        interval="5m" if kind == "5m" else "60m",
        range="60d" if kind == "5m" else "730d",
    )
    px, errors = bp.load_px(ns, symbols)
    if errors:
        print("leader errors", kind, errors)
    return px


def model_trades(px: pd.DataFrame, model: dict, allocation: float = 500_000.0, *, reverse: bool = False, cost_mult: float | None = None, latency: int = 0, override: dict | None = None) -> pd.DataFrame:
    m = dict(model)
    if override:
        m.update(override)
    leader = str(m["leader"])
    d = px.dropna(subset=[leader]).copy()
    values = fh.feature_values(d, leader, int(m["window"]), str(m["feature"]), 48)
    sig = np.where(values > float(m["threshold"]), 1.0, np.where(values < -float(m["threshold"]), -1.0, 0.0))
    if str(m["mode"]) == "fade":
        sig = -sig
    if reverse:
        sig = -sig
    horizon = int(m["horizon"])
    entry_shift = int(latency)
    entry_xau = d["xau"].shift(-entry_shift)
    exit_xau = d["xau"].shift(-(entry_shift + horizon))
    future = exit_xau - entry_xau
    smask = bp.session_mask(d.index, str(m["session"]))
    mask = np.isfinite(sig) & np.isfinite(future.values) & np.isfinite(entry_xau.values) & (sig != 0.0) & smask
    spread_mult = float(cost_mult if cost_mult is not None else m["cost"])
    spread = d["spread"].fillna(d["spread"].median()).clip(lower=0.0)
    entry = entry_xau.values[mask]
    gross_per_oz = sig[mask] * future.values[mask]
    cost_per_oz = spread.values[mask] * spread_mult
    ounces = allocation / entry
    out = pd.DataFrame(
        {
            "model": str(m["name"]),
            "leader": leader,
            "entry_time": d.index.to_series().shift(-entry_shift).values[mask],
            "signal_time": d.index[mask],
            "exit_time": d.index.to_series().shift(-(entry_shift + horizon)).values[mask],
            "direction": sig[mask].astype(int),
            "xau_entry": entry,
            "xau_exit": exit_xau.values[mask],
            "gross_per_oz": gross_per_oz,
            "cost_per_oz": cost_per_oz,
            "pnl_per_dollar": (gross_per_oz - cost_per_oz) / entry,
            "ounces": ounces,
            "notional": allocation,
            "pnl": (gross_per_oz - cost_per_oz) * ounces,
        }
    )
    out = out.dropna(subset=["entry_time", "exit_time", "pnl"]).sort_values("exit_time")
    return out


def stat_from_trades(trades: pd.DataFrame, capital: float = 5_000_000.0) -> dict:
    if len(trades) == 0:
        return {"trades": 0, "pnl": 0.0, "return_pct": 0.0, "pf": 0.0, "max_dd": 0.0}
    p = trades["pnl"].values
    win = float(p[p > 0].sum())
    loss = float(-p[p < 0].sum())
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(p)),
        "pnl": float(p.sum()),
        "return_pct": float(p.sum() / capital * 100),
        "win_rate": float((p > 0).mean()),
        "pf": float(win / loss) if loss > 0 else 0.0,
        "max_dd": float((peak - eq).max()) if len(eq) else 0.0,
        "max_dd_pct": float(((peak - eq).max() if len(eq) else 0.0) / capital * 100),
    }


def candidate_rows(kind: str) -> list[dict]:
    rows = []
    files = [f"fifty_symbol_batch{i}_yahoo_5m_forward_search_2x.json" for i in range(1, 5)] if kind == "5m" else [f"long_history_60m_batch{i}_search_2x.json" for i in range(1, 5)]
    for f in files:
        d = load_json(f)
        for r in d.get("top", []):
            if r.get("cost", 0) >= 2 and all(r[s]["pnl"] > 0 for s in ["train", "val", "old_test"]):
                rows.append(r)
    return rows


def attack_multiple_comparisons(kind: str, draws: int = 20_000) -> dict:
    rows = candidate_rows(kind)
    actual = load_json("rerun_5m_forward_portfolio_capped.json" if kind == "5m" else "rerun_60m_forward_portfolio_capped.json")["portfolio"]
    eligible = [r for r in rows if r["forward"]["pnl"] > 0]
    rng = np.random.default_rng(20260618)
    sums = []
    pfs = []
    for _ in range(draws):
        sample = rng.choice(eligible, size=min(10, len(eligible)), replace=False)
        pnl = sum(float(r["forward"]["pnl"]) for r in sample)
        wins = sum(float(r["forward"]["pnl"]) for r in sample if r["forward"]["pnl"] > 0)
        # This is candidate-level PF proxy, not trade-level PF.
        losses = sum(-float(r["forward"]["pnl"]) for r in sample if r["forward"]["pnl"] < 0)
        sums.append(pnl)
        pfs.append(wins / losses if losses > 0 else np.inf)
    arr = np.array(sums)
    return {
        "kind": kind,
        "status": "INCONCLUSIVE",
        "why_inconclusive": "Saved reports do not contain every evaluated grid combination or capped portfolio trades for random candidate portfolios. This is a weaker promoted-candidate raw-forward-PnL null, not a full multiple-comparisons correction.",
        "candidate_rows_after_old_gate": len(rows),
        "forward_positive_candidates": len(eligible),
        "draws": draws,
        "actual_capped_return_pct": actual["return_pct"],
        "actual_capped_pnl": actual["pnl"],
        "null_raw_forward_pnl_mean": float(np.mean(arr)),
        "null_raw_forward_pnl_p50": float(np.quantile(arr, 0.50)),
        "null_raw_forward_p95": float(np.quantile(arr, 0.95)),
        "null_raw_forward_p99": float(np.quantile(arr, 0.99)),
    }


def attack_reverse(kind: str) -> dict:
    px = load_px(kind)
    rows = []
    for m in selected(kind):
        fwd = stat_from_trades(model_trades(px, m))
        rev = stat_from_trades(model_trades(px, m, reverse=True))
        rows.append({"model": m["name"], "forward_pnl": fwd["pnl"], "forward_pf": fwd["pf"], "reverse_pnl": rev["pnl"], "reverse_pf": rev["pf"], "reverse_ratio": rev["pnl"] / abs(fwd["pnl"]) if fwd["pnl"] else 0.0})
    weak = [r for r in rows if r["reverse_pnl"] > -0.25 * abs(r["forward_pnl"])]
    return {"status": "FAIL" if weak else "PASS", "threshold": "PASS requires reverse PnL <= -25% of forward PnL for every model", "weak_models": weak, "models": rows}


def block_bootstrap(values: np.ndarray, block: int, trials: int = 5000, capital: float = 5_000_000.0) -> dict:
    rng = np.random.default_rng(20260618)
    n = len(values)
    totals, pfs, dds = [], [], []
    if n == 0:
        return {}
    for _ in range(trials):
        out = []
        while len(out) < n:
            start = int(rng.integers(0, n))
            idx = [(start + j) % n for j in range(block)]
            out.extend(values[idx])
        sample = np.array(out[:n])
        win = sample[sample > 0].sum()
        loss = -sample[sample < 0].sum()
        eq = sample.cumsum()
        peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
        totals.append(sample.sum() / capital * 100)
        pfs.append(win / loss if loss > 0 else np.inf)
        dds.append((peak - eq).max() / capital * 100)
    return {
        "trials": trials,
        "block_length": block,
        "return_pct_ci95": [float(np.quantile(totals, 0.025)), float(np.quantile(totals, 0.975))],
        "pf_ci95": [float(np.quantile(pfs, 0.025)), float(np.quantile(pfs, 0.975))],
        "max_dd_pct_ci95": [float(np.quantile(dds, 0.025)), float(np.quantile(dds, 0.975))],
    }


def attack_bootstrap(kind: str) -> dict:
    file = "rerun_5m_forward_portfolio_capped_trades.csv" if kind == "5m" else "rerun_60m_forward_portfolio_capped_trades.csv"
    trades = pd.read_csv(REPORTS / file)
    block = 30 if kind == "5m" else 8
    port = block_bootstrap(trades["pnl"].values, block)
    px = load_px(kind)
    model_rows = []
    for m in selected(kind):
        tr = model_trades(px, m)
        b = block_bootstrap(tr["pnl"].values, max(4, int(m["horizon"])))
        model_rows.append({"model": m["name"], **b})
    return {"status": "FAIL" if port["return_pct_ci95"][0] <= 0 else "PASS", "threshold": "PASS requires portfolio 95% block-bootstrap return CI lower bound > 0", "portfolio": port, "models": model_rows}


def attack_sensitivity(kind: str) -> dict:
    px = load_px(kind)
    rows = []
    for m in selected(kind):
        base = stat_from_trades(model_trades(px, m))
        variants = []
        for param, factors in {"window": [0.8, 1.2], "quantile": [0.9, 1.1], "horizon": [0.8, 1.2]}.items():
            for factor in factors:
                override = {}
                if param == "window":
                    override["window"] = max(2, int(round(int(m["window"]) * factor)))
                elif param == "horizon":
                    override["horizon"] = max(1, int(round(int(m["horizon"]) * factor)))
                else:
                    q = min(0.95, max(0.05, float(m["quantile"]) * factor))
                    # Approximation: saved artifacts do not include train feature vectors, so
                    # scale threshold by quantile ratio as a local perturbation proxy.
                    override["threshold"] = float(m["threshold"]) * (q / float(m["quantile"]))
                st = stat_from_trades(model_trades(px, m, override=override))
                variants.append({"param": param, "factor": factor, "override": override, "pnl": st["pnl"], "pf": st["pf"], "return_pct": st["return_pct"]})
        cliffs = [v for v in variants if v["pnl"] <= 0 or v["pnl"] < 0.5 * base["pnl"]]
        rows.append({"model": m["name"], "base_pnl": base["pnl"], "base_pf": base["pf"], "cliff_count": len(cliffs), "variants": variants})
    bad = [r for r in rows if r["cliff_count"] > 0]
    return {"status": "FAIL" if bad else "PASS", "threshold": "PASS requires all +/- perturbation variants to remain positive and above 50% of base PnL", "fragile_models": [r["model"] for r in bad], "models": rows}


def attack_independence(kind: str) -> dict:
    px = load_px(kind)
    parts = []
    for m in selected(kind):
        tr = model_trades(px, m)
        tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True)
        daily = tr.set_index("exit_time")["pnl"].resample("1D").sum().rename(m["name"])
        parts.append(daily)
    mat = pd.concat(parts, axis=1).fillna(0.0)
    corr = mat.corr().fillna(0.0)
    vals = np.linalg.eigvalsh(corr.values)
    vals = np.maximum(vals, 0)
    participation = float((vals.sum() ** 2) / (np.square(vals).sum())) if np.square(vals).sum() else 0.0
    kaiser = int((vals > 1.0).sum())
    return {"status": "FAIL" if participation < 5 else "PASS", "threshold": "PASS requires participation-ratio effective bets >= 5 of 10", "participation_ratio_effective_bets": participation, "kaiser_components_gt_1": kaiser, "eigenvalues": vals.tolist(), "correlation_csv": corr.to_csv()}


def all_uncapped_trades(kind: str) -> pd.DataFrame:
    px = load_px(kind)
    frames = [model_trades(px, m) for m in selected(kind)]
    return pd.concat(frames, ignore_index=True)


def cap_with_policy(trades: pd.DataFrame, policy: str, capital: float = 5_000_000.0) -> pd.DataFrame:
    t = trades.copy()
    if policy == "reverse_model":
        t["_model_order"] = t["model"].rank(method="dense", ascending=False)
    elif policy == "edge_rank":
        avg = t.groupby("model")["pnl_per_dollar"].mean().rank(method="dense", ascending=False)
        t["_model_order"] = t["model"].map(avg)
    else:
        t["_model_order"] = t["model"].rank(method="dense", ascending=True)
    t = t.sort_values(["entry_time", "_model_order", "exit_time"]).reset_index(drop=True)
    events = []
    for i, row in t.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_ = pd.Timestamp(row["exit_time"])
        entry = entry.tz_convert("UTC") if entry.tzinfo else entry.tz_localize("UTC")
        exit_ = exit_.tz_convert("UTC") if exit_.tzinfo else exit_.tz_localize("UTC")
        events.append((entry, 1, i))
        events.append((exit_, 0, i))
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0.0
    assigned = np.zeros(len(t))
    for _, typ, i in events:
        if typ == 0:
            active -= assigned[i]
            active = max(active, 0.0)
        else:
            fill = min(float(t.at[i, "notional"]), max(0.0, capital - active))
            assigned[i] = fill
            active += fill
    t["notional"] = assigned
    t = t[t["notional"] > 0].copy()
    t["pnl"] = t["pnl_per_dollar"] * t["notional"]
    return t.sort_values("exit_time")


def attack_allocator(kind: str) -> dict:
    tr = all_uncapped_trades(kind)
    rows = {}
    for policy in ["existing", "reverse_model", "edge_rank"]:
        rows[policy] = stat_from_trades(cap_with_policy(tr, policy))
    pnls = [v["pnl"] for v in rows.values()]
    swing = (max(pnls) - min(pnls)) / max(1.0, abs(rows["existing"]["pnl"]))
    return {"status": "FAIL" if swing > 0.3 else "PASS", "threshold": "PASS requires allocator policy PnL swing <= 30% of existing PnL", "pnl_swing_ratio": float(swing), "policies": rows}


def attack_cost_latency(kind: str) -> dict:
    px = load_px(kind)
    rows = {}
    for cost in [2.0, 3.0, 5.0]:
        frames = [model_trades(px, m, cost_mult=cost) for m in selected(kind)]
        rows[f"cost_{cost}x_uncapped"] = stat_from_trades(pd.concat(frames, ignore_index=True))
    for latency in [1, 2]:
        frames = [model_trades(px, m, latency=latency) for m in selected(kind)]
        rows[f"latency_{latency}_bar_uncapped"] = stat_from_trades(pd.concat(frames, ignore_index=True))
    killed = rows["cost_5.0x_uncapped"]["pnl"] <= 0 or rows["latency_1_bar_uncapped"]["pnl"] <= 0
    return {"status": "FAIL" if killed else "PASS", "threshold": "PASS requires positive uncapped PnL at 5x cost and 1-bar latency", "results": rows}


def attack_subperiod(kind: str) -> dict:
    px = load_px(kind)
    model_rows = []
    all_tr = []
    for m in selected(kind):
        tr = model_trades(px, m)
        all_tr.append(tr)
        n = len(tr)
        qs = [float(tr["pnl"].iloc[i * n // 4 : (i + 1) * n // 4].sum()) for i in range(4)]
        model_rows.append({"model": m["name"], "quarters": qs, "total": sum(qs), "max_quarter_share": max(qs) / sum(qs) if sum(qs) > 0 else None})
    trp = pd.concat(all_tr, ignore_index=True).sort_values("exit_time")
    n = len(trp)
    pqs = [float(trp["pnl"].iloc[i * n // 4 : (i + 1) * n // 4].sum()) for i in range(4)]
    bad = [r for r in model_rows if any(q <= 0 for q in r["quarters"]) or (r["max_quarter_share"] is not None and r["max_quarter_share"] > 0.75)]
    return {"status": "FAIL" if bad or any(q <= 0 for q in pqs) else "PASS", "threshold": "PASS requires all four portfolio quarters positive and no individual model with negative quarter or >75% PnL from one quarter", "portfolio_quarters": pqs, "bad_models": [r["model"] for r in bad], "models": model_rows}


def attack_leave_one_out(kind: str) -> dict:
    tr = all_uncapped_trades(kind)
    base = stat_from_trades(cap_with_policy(tr, "existing"))
    rows = []
    for model in sorted(tr["model"].unique()):
        st = stat_from_trades(cap_with_policy(tr[tr["model"] != model], "existing"))
        rows.append({"removed": model, "pnl": st["pnl"], "return_pct": st["return_pct"], "max_dd": st["max_dd"], "pnl_change_pct_of_base": (st["pnl"] - base["pnl"]) / abs(base["pnl"]) * 100 if base["pnl"] else 0, "dd_change_pct_of_base": (st["max_dd"] - base["max_dd"]) / base["max_dd"] * 100 if base["max_dd"] else 0})
    bad = [r for r in rows if r["pnl"] <= 0 or abs(r["dd_change_pct_of_base"]) > 30]
    return {"status": "FAIL" if bad else "PASS", "threshold": "PASS requires no leave-one-out portfolio to flip negative and no max-DD change >30%", "base": base, "bad": bad, "leave_one_out": rows}


def leakage_audit() -> dict:
    # Static audit from actual scripts.
    return {
        "status": "FAIL",
        "threshold": "PASS requires no material leakage/alignment/availability caveats",
        "findings": [
            {"item": "spread_t", "verdict": "INCONCLUSIVE/WEAK", "detail": "Cost uses observed average spread of the same execution bar at signal entry. This is conservative as a cost subtraction but not a tradable known-at-signal quote. It is not alpha leakage, but it is not a broker-real quote model."},
            {"item": "threshold", "verdict": "PASS for saved models", "detail": "Thresholds in search scripts are fitted from the first half of old data only; selected JSON stores fixed thresholds used in forward reruns."},
            {"item": "timestamp alignment", "verdict": "INCONCLUSIVE", "detail": "5m Yahoo leader and restored XAU are inner-joined on timestamps. A prior bug in 60m forward aggregation collapsed timestamps to 1970 and was fixed before final 60m reruns. No independent timezone audit against exchange calendars was done."},
            {"item": "rolling windows", "verdict": "PASS", "detail": "Leader momentum and vol use rolling values shifted by one bar before signal generation in the search functions."},
            {"item": "universe availability", "verdict": "FAIL/BIAS RISK", "detail": "The 50-symbol universe was manually selected for liquid available Yahoo symbols with usable data. This can bias toward instruments that happen to have clean data and macro relevance in the tested window."},
        ],
    }


def main() -> int:
    result = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "attacks": {
            "multiple_comparisons": {"5m": attack_multiple_comparisons("5m"), "60m": attack_multiple_comparisons("60m")},
            "leakage_timestamp_audit": leakage_audit(),
            "reverse_rule": {"5m": attack_reverse("5m"), "60m": attack_reverse("60m")},
            "block_bootstrap": {"5m": attack_bootstrap("5m"), "60m": attack_bootstrap("60m")},
            "parameter_sensitivity": {"5m": attack_sensitivity("5m"), "60m": attack_sensitivity("60m")},
            "independence": {"5m": attack_independence("5m"), "60m": attack_independence("60m")},
            "allocator_order": {"5m": attack_allocator("5m"), "60m": attack_allocator("60m")},
            "cost_latency": {"5m": attack_cost_latency("5m"), "60m": attack_cost_latency("60m")},
            "subperiod": {"5m": attack_subperiod("5m"), "60m": attack_subperiod("60m")},
            "leave_one_out": {"5m": attack_leave_one_out("5m"), "60m": attack_leave_one_out("60m")},
        },
    }
    save_json("adversarial_xau_edge_audit_results.json", result)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) and "status" in v else "nested") for k, v in result["attacks"].items()}, indent=2))
    print("wrote", REPORTS / "adversarial_xau_edge_audit_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
