#!/usr/bin/env python3
"""Strict XAU cross-market screening pipeline with full-grid persistence.

This is intentionally more conservative than the earlier search scripts:
every evaluated grid row is persisted, rejection reasons are explicit, and
portfolio construction is checked for allocator dependence and independence.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

SYMBOLS_5M = [
    "ACWI", "BTC-USD", "CL=F", "DBC", "DIA", "DX-Y.NYB", "EEM", "EFA", "ETH-USD", "EURUSD=X",
    "EWG", "EWJ", "EWU", "EWZ", "FXE", "FXI", "FXY", "GC=F", "GDX", "GDXJ", "GLD", "HYG",
    "IAU", "IEF", "IWM", "LQD", "QQQ", "SHY", "SI=F", "SLV", "SPY", "TLT", "UNG", "USDJPY=X",
    "USO", "UUP", "VXX", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "ZB=F", "ZN=F",
]

SYMBOLS_60M = sorted(set(SYMBOLS_5M + ["FXA", "FXB"]))

FEATURES = ["volz", "divergence"]
WINDOWS = [2, 3, 4, 6, 9, 12, 18, 24]
QUANTILES = [0.55, 0.65, 0.75, 0.85]
SESSIONS = ["all", "london", "ny"]
MODES = ["follow", "fade"]
HORIZONS_5M = [6, 12, 18, 24, 30, 36]
HORIZONS_60M = [1, 2, 3, 4, 6, 8]
COSTS = [2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", choices=["5m", "60m"], required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--master", default=MASTER)
    p.add_argument("--forward-xau", default=FORWARD_XAU)
    p.add_argument("--cache-dir", default=str(DATA / "cache" / "yahoo"))
    p.add_argument("--bootstrap-trials", type=int, default=2000)
    p.add_argument("--max-portfolio-models", type=int, default=10)
    p.add_argument("--limit-symbols", nargs="*", default=None)
    return p.parse_args()


def stat(pnl: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float | int]:
    if len(pnl) == 0:
        return {"trades": 0, "tpd": 0.0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "sh": 0.0, "dd": 0.0}
    days = max(1e-9, (times[-1] - times[0]).total_seconds() / 86400.0)
    win = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    avg = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    eq = pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(pnl)),
        "tpd": float(len(pnl) / days),
        "pnl": float(pnl.sum()),
        "wr": float((pnl > 0).mean()),
        "pf": float(win / loss) if loss > 0 else 0.0,
        "sh": float(avg / sd * math.sqrt(len(pnl))) if sd > 0 else 0.0,
        "dd": float((peak - eq).max()) if len(eq) else 0.0,
    }


def session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    if session == "london":
        return np.array([(7 <= t.hour < 16) for t in index], dtype=bool)
    if session == "ny":
        return np.array([(13 <= t.hour < 21) for t in index], dtype=bool)
    return np.ones(len(index), dtype=bool)


def split_old(pnl: np.ndarray, times: pd.DatetimeIndex) -> tuple[dict, dict, dict]:
    a = len(pnl) // 2
    b = 3 * len(pnl) // 4
    return stat(pnl[:a], times[:a]), stat(pnl[a:b], times[a:b]), stat(pnl[b:], times[b:])


def trade_sequence(
    df: pd.DataFrame,
    leader: str,
    feature: str,
    window: int,
    threshold: float,
    mode: str,
    session: str,
    horizon: int,
    cost: float,
    vol_window: int,
    *,
    reverse: bool = False,
) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    d = df.dropna(subset=[leader, "xau", "spread"]).copy()
    vals = fh.feature_values(d, leader, window, feature, vol_window)
    sig = np.asarray(np.where(vals > threshold, 1.0, np.where(vals < -threshold, -1.0, 0.0)), dtype=float)
    if mode == "fade":
        sig = -sig
    if reverse:
        sig = -sig
    future = (d["xau"].shift(-horizon) - d["xau"]).astype(float).values
    smask = session_mask(pd.DatetimeIndex(d.index), session)
    spread = d["spread"].fillna(d["spread"].median()).clip(lower=0.0).astype(float).values
    mask = np.isfinite(sig) & np.isfinite(future) & np.isfinite(spread) & (sig != 0.0) & smask
    gross = sig[mask] * future[mask]
    cost_values = spread[mask] * cost
    pnl = gross - cost_values
    times = pd.DatetimeIndex(d.index[mask])
    details = pd.DataFrame(
        {
            "entry_time": times,
            "exit_time": pd.Series(d.index, index=d.index).shift(-horizon).values[mask],
            "direction": sig[mask].astype(int),
            "xau_entry": d["xau"].astype(float).values[mask],
            "pnl_per_oz": pnl,
            "gross_per_oz": gross,
            "cost_per_oz": cost_values,
        }
    )
    details = details.dropna(subset=["exit_time", "xau_entry", "pnl_per_oz"])
    return pnl[: len(details)], pd.DatetimeIndex(details["entry_time"]), details


def block_bootstrap_lower(pnl: np.ndarray, block: int, trials: int, rng: np.random.Generator) -> dict[str, float | int]:
    if len(pnl) < max(20, block * 4):
        return {"trials": trials, "block": block, "ci_low": float("-inf"), "ci_high": float("nan")}
    if pnl.sum() <= 0:
        return {"trials": trials, "block": block, "ci_low": float(pnl.sum()), "ci_high": float(pnl.sum())}
    n = len(pnl)
    totals = np.empty(trials, dtype=float)
    starts_needed = int(math.ceil(n / block))
    for i in range(trials):
        starts = rng.integers(0, n, size=starts_needed)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        totals[i] = float(pnl[idx].sum())
    return {"trials": trials, "block": block, "ci_low": float(np.quantile(totals, 0.025)), "ci_high": float(np.quantile(totals, 0.975))}


def subperiod_check(pnl: np.ndarray, periods: int = 4) -> dict[str, object]:
    if len(pnl) < periods:
        return {"periods": [], "negative_periods": periods, "max_share": None, "pass": False}
    chunks = [float(pnl[i * len(pnl) // periods : (i + 1) * len(pnl) // periods].sum()) for i in range(periods)]
    total = float(sum(chunks))
    neg = int(sum(1 for x in chunks if x < 0))
    max_share = max(chunks) / total if total > 0 else None
    ok = total > 0 and neg <= 1 and (max_share is not None and max_share <= 0.75)
    return {"periods": chunks, "negative_periods": neg, "max_share": max_share, "pass": bool(ok)}


def normal_lower_bound(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return float("-inf")
    total = float(pnl.sum())
    sd = float(pnl.std(ddof=1))
    return total - 1.96 * sd * math.sqrt(len(pnl))


def old_gate(train: dict, val: dict, old_test: dict, fwd: dict, timeframe: str) -> bool:
    if timeframe == "5m":
        lo, hi = 15.0, 35.0
        min_fwd = 50
    else:
        lo, hi = 0.5, 12.0
        min_fwd = 25
    return bool(
        train["pnl"] > 0
        and val["pnl"] > 0
        and old_test["pnl"] > 0
        and fwd["pnl"] > 0
        and lo <= train["tpd"] <= hi
        and lo <= val["tpd"] <= hi
        and lo <= old_test["tpd"] <= hi
        and fwd["trades"] >= min_fwd
    )


def parameter_check(old: pd.DataFrame, forward: pd.DataFrame, row: dict, vol_window: int) -> dict[str, object]:
    base = float(row["forward"]["pnl"])
    variants = []
    specs = [
        ("window", 0.8), ("window", 1.2),
        ("quantile", 0.9), ("quantile", 1.1),
        ("horizon", 0.8), ("horizon", 1.2),
    ]
    leader = row["leader"]
    old_leader = old.dropna(subset=[leader])
    for field, factor in specs:
        window = int(row["window"])
        quantile = float(row["quantile"])
        horizon = int(row["horizon"])
        threshold = float(row["threshold"])
        if field == "window":
            window = max(2, int(round(window * factor)))
            vals = fh.feature_values(old_leader, leader, window, row["feature"], vol_window)
            sample = np.abs(vals[: len(vals) // 2])
            sample = sample[np.isfinite(sample)]
            if len(sample) >= 100:
                threshold = float(np.quantile(sample, quantile))
        elif field == "quantile":
            quantile = min(0.95, max(0.05, quantile * factor))
            vals = fh.feature_values(old_leader, leader, window, row["feature"], vol_window)
            sample = np.abs(vals[: len(vals) // 2])
            sample = sample[np.isfinite(sample)]
            if len(sample) >= 100:
                threshold = float(np.quantile(sample, quantile))
        else:
            horizon = max(1, int(round(horizon * factor)))
        pnl, times, _ = trade_sequence(forward, leader, row["feature"], window, threshold, row["mode"], row["session"], horizon, row["cost"], vol_window)
        st = stat(pnl, times)
        variants.append({"field": field, "factor": factor, "pnl": st["pnl"], "pf": st["pf"], "trades": st["trades"]})
    failures = [v for v in variants if v["pnl"] <= 0 or v["pnl"] < 0.5 * base]
    return {"pass": len(failures) == 0, "failures": len(failures), "variants": variants}


def flatten_stat(prefix: str, st: dict) -> dict[str, object]:
    return {f"{prefix}_{k}": v for k, v in st.items()}


def json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


def load_data(args: argparse.Namespace, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    interval = args.timeframe
    range_ = "60d" if interval == "5m" else "730d"
    leaders, errors = cm.load_leaders(symbols, interval, range_, Path(args.cache_dir))
    xau_old = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), interval)
    old = leaders.join(xau_old[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    old = old.dropna(subset=["xau", "spread"])
    old = old[old["gap_fill"] == 0].copy()
    if interval == "5m":
        forward_xau = pd.read_csv(args.forward_xau, parse_dates=["time"]).rename(columns={"close": "xau"}).set_index("time").sort_index()
    else:
        forward_xau = lh.aggregate_forward_xau(args.forward_xau, interval)
    forward = leaders.join(forward_xau[["xau", "spread", "ticks"]], how="inner").sort_index().dropna(subset=["xau", "spread"])
    snapshot = {
        "snapshot_created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "availability_asof_policy": "Uses local Yahoo cache files already present before this rebuild run; not a true pre-forward vendor snapshot.",
        "cache_dir": str(Path(args.cache_dir)),
        "requested_symbols": symbols,
        "included_symbols": [s for s in symbols if s in leaders.columns and leaders[s].notna().sum() >= 500],
        "excluded_symbols": {s: errors.get(s, "insufficient rows or missing loaded column") for s in symbols if s not in leaders.columns or leaders[s].notna().sum() < 500},
        "leader_errors": errors,
    }
    meta = {
        "old_rows": int(len(old)),
        "forward_rows": int(len(forward)),
        "old_start": str(old.index.min()),
        "old_end": str(old.index.max()),
        "forward_start": str(forward.index.min()),
        "forward_end": str(forward.index.max()),
        "underpowered_forward_warning": "Forward window is the same short 2026-05-22 to 2026-06-18 window; any survivor is in-sample under the new gate and requires rerun on longer history.",
    }
    return old, forward, snapshot, meta


def model_trades_for_portfolio(forward: pd.DataFrame, candidates: list[dict], allocation: float) -> pd.DataFrame:
    frames = []
    for c in candidates:
        _, _, details = trade_sequence(
            forward,
            c["leader"],
            c["feature"],
            int(c["window"]),
            float(c["threshold"]),
            c["mode"],
            c["session"],
            int(c["horizon"]),
            float(c["cost"]),
            48,
        )
        if len(details) == 0:
            continue
        details["model"] = c["name"]
        details["leader"] = c["leader"]
        details["notional"] = allocation
        details["pnl_per_dollar"] = details["pnl_per_oz"] / details["xau_entry"]
        details["pnl"] = details["pnl_per_dollar"] * allocation
        details["vol_key"] = float(c["forward"]["dd"]) / max(1.0, float(c["forward"]["pnl"]))
        details["edge_key"] = float(c["forward"]["pnl"]) / max(1, int(c["forward"]["trades"]))
        frames.append(details)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cap_policy(trades: pd.DataFrame, policy: str, capital: float = 5_000_000.0) -> pd.DataFrame:
    if len(trades) == 0:
        return trades
    t = trades.copy()
    if policy == "edge_rank":
        order = t.groupby("model")["edge_key"].mean().rank(method="dense", ascending=False)
        t["_order"] = t["model"].map(order)
    elif policy == "inverse_vol":
        order = t.groupby("model")["vol_key"].mean().rank(method="dense", ascending=True)
        t["_order"] = t["model"].map(order)
    else:
        t["_order"] = t["model"].rank(method="dense", ascending=True)
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.sort_values(["entry_time", "_order", "exit_time"]).reset_index(drop=True)
    events = []
    for i, row in t.iterrows():
        events.append((row["entry_time"], 1, i))
        events.append((row["exit_time"], 0, i))
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0.0
    assigned = np.zeros(len(t), dtype=float)
    for _, typ, i in events:
        if typ == 0:
            active = max(0.0, active - assigned[i])
        else:
            fill = min(float(t.at[i, "notional"]), max(0.0, capital - active))
            assigned[i] = fill
            active += fill
    t["notional"] = assigned
    t = t[t["notional"] > 0].copy()
    t["pnl"] = t["pnl_per_dollar"] * t["notional"]
    return t.sort_values("exit_time")


def portfolio_stats(trades: pd.DataFrame, capital: float = 5_000_000.0) -> dict[str, object]:
    if len(trades) == 0:
        return {"trades": 0, "pnl": 0.0, "return_pct": 0.0, "pf": 0.0, "max_dd": 0.0}
    pnl = trades["pnl"].to_numpy(float)
    win = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    eq = pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(pnl)),
        "pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / capital * 100),
        "pf": float(win / loss) if loss > 0 else 0.0,
        "max_dd": float((peak - eq).max()) if len(eq) else 0.0,
        "max_dd_pct": float(((peak - eq).max() if len(eq) else 0.0) / capital * 100),
    }


def effective_bets(trades: pd.DataFrame) -> dict[str, object]:
    if len(trades) == 0 or "model" not in trades:
        return {"participation_ratio": 0.0, "kaiser_components_gt_1": 0, "eigenvalues": []}
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    parts = []
    for model, g in t.groupby("model"):
        parts.append(g.set_index("exit_time")["pnl"].resample("1D").sum().rename(model))
    mat = pd.concat(parts, axis=1).fillna(0.0)
    if mat.shape[1] < 2:
        return {"participation_ratio": float(mat.shape[1]), "kaiser_components_gt_1": int(mat.shape[1]), "eigenvalues": [1.0] * mat.shape[1]}
    corr = mat.corr().fillna(0.0)
    vals = np.maximum(np.linalg.eigvalsh(corr.values), 0.0)
    pr = float((vals.sum() ** 2) / np.square(vals).sum()) if np.square(vals).sum() else 0.0
    return {"participation_ratio": pr, "kaiser_components_gt_1": int((vals > 1.0).sum()), "eigenvalues": vals.tolist()}


def select_portfolio(candidates: list[dict], forward: pd.DataFrame, max_models: int) -> dict[str, object]:
    picked: list[dict] = []
    for c in sorted(candidates, key=lambda r: (r["forward"]["pnl"], r["old_test"]["pnl"]), reverse=True):
        if any(p["leader"] == c["leader"] for p in picked):
            continue
        trial = picked + [c]
        trades = model_trades_for_portfolio(forward, trial, 5_000_000.0 / max(1, len(trial)))
        eb = effective_bets(trades)
        if len(trial) <= 5 or eb["participation_ratio"] >= min(5.0, len(trial) * 0.5):
            picked = trial
        if len(picked) >= max_models:
            break
    trades = model_trades_for_portfolio(forward, picked, 5_000_000.0 / max(1, len(picked)))
    policies = {name: portfolio_stats(cap_policy(trades, name)) for name in ["edge_rank", "inverse_vol", "name_order"]}
    pnls = [v["pnl"] for v in policies.values()]
    base = policies["edge_rank"]["pnl"]
    swing = float((max(pnls) - min(pnls)) / max(1.0, abs(base))) if pnls else 0.0
    return {
        "selected_count": len(picked),
        "models": picked,
        "allocator_policies": policies,
        "allocator_swing_ratio": swing,
        "allocator_pass": bool(swing <= 0.30),
        "effective_bets": effective_bets(trades),
    }


def main() -> int:
    args = parse_args()
    symbols = args.limit_symbols or (SYMBOLS_5M if args.timeframe == "5m" else SYMBOLS_60M)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = out_prefix.with_name(out_prefix.name + "_full_grid.jsonl")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")
    snapshot_path = out_prefix.with_name(out_prefix.name + "_universe_snapshot.json")

    old, forward, snapshot, meta = load_data(args, symbols)
    snapshot_path.write_text(json.dumps(json_sanitize(snapshot), indent=2), encoding="utf-8")

    horizons = HORIZONS_5M if args.timeframe == "5m" else HORIZONS_60M
    rng = np.random.default_rng(20260618)
    rows_written = 0
    survivors: list[dict] = []
    breakdown = {
        "evaluated": 0,
        "failed_old_gate": 0,
        "failed_bootstrap": 0,
        "failed_subperiod": 0,
        "failed_parameter": 0,
        "failed_reverse": 0,
        "survived_all": 0,
    }
    sample_rows = []
    vol_window = 48
    with grid_path.open("w", encoding="utf-8") as f:
        for leader in symbols:
            if leader not in old.columns or leader not in forward.columns:
                continue
            old_leader = old.dropna(subset=[leader]).copy()
            if len(old_leader) < 500:
                continue
            for feature in FEATURES:
                for window in WINDOWS:
                    vals = fh.feature_values(old_leader, leader, window, feature, vol_window)
                    train_sample = np.abs(vals[: len(vals) // 2])
                    train_sample = train_sample[np.isfinite(train_sample)]
                    if len(train_sample) < 100:
                        continue
                    for quantile in QUANTILES:
                        threshold = float(np.quantile(train_sample, quantile))
                        for mode in MODES:
                            for session in SESSIONS:
                                for horizon in horizons:
                                    for cost in COSTS:
                                        old_pnl, old_times, _ = trade_sequence(old_leader, leader, feature, window, threshold, mode, session, horizon, cost, vol_window)
                                        fwd_pnl, fwd_times, _ = trade_sequence(forward, leader, feature, window, threshold, mode, session, horizon, cost, vol_window)
                                        train, val, old_test = split_old(old_pnl, old_times)
                                        fwd = stat(fwd_pnl, fwd_times)
                                        og = old_gate(train, val, old_test, fwd, args.timeframe)
                                        if og and fwd["pnl"] > 0:
                                            rev_pnl, rev_times, _ = trade_sequence(forward, leader, feature, window, threshold, mode, session, horizon, cost, vol_window, reverse=True)
                                            rev = stat(rev_pnl, rev_times)
                                        else:
                                            rev = {"trades": 0, "tpd": 0.0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "sh": 0.0, "dd": 0.0, "not_run_reason": "old_gate_failed_or_forward_pnl_nonpositive"}
                                        block = max(4, int(horizon) * 3)
                                        approx_low = normal_lower_bound(fwd_pnl)
                                        if og and fwd["trades"] and approx_low > 0:
                                            boot = block_bootstrap_lower(fwd_pnl, block, args.bootstrap_trials, rng)
                                        else:
                                            boot = {
                                                "trials": args.bootstrap_trials,
                                                "block": block,
                                                "ci_low": float("-inf"),
                                                "ci_high": None,
                                                "normal_approx_ci_low": approx_low,
                                                "not_run_reason": "old_gate_failed_no_forward_trades_or_normal_lower_bound_nonpositive",
                                            }
                                        sub = subperiod_check(fwd_pnl)
                                        # Parameter perturbations are costly and only meaningful for candidates that
                                        # have a positive base forward result. Others fail the promotion gate already.
                                        if og and boot["ci_low"] > 0 and sub["pass"] and rev["pnl"] <= -0.25 * abs(fwd["pnl"]):
                                            param = parameter_check(old, forward, {
                                                "leader": leader, "feature": feature, "window": window, "quantile": quantile,
                                                "threshold": threshold, "mode": mode, "session": session, "horizon": horizon,
                                                "cost": cost, "forward": fwd,
                                            }, vol_window)
                                        else:
                                            param = {"pass": False, "failures": None, "variants": [], "not_run_reason": "short_circuited_before_parameter_gate"}

                                        reverse_pass = bool(rev["pnl"] <= -0.25 * abs(fwd["pnl"])) if fwd["pnl"] > 0 else False
                                        failures = []
                                        if not og:
                                            failures.append("old_gate")
                                            breakdown["failed_old_gate"] += 1
                                        if not (boot["ci_low"] is not None and boot["ci_low"] > 0):
                                            failures.append("bootstrap")
                                            breakdown["failed_bootstrap"] += 1
                                        if not sub["pass"]:
                                            failures.append("subperiod")
                                            breakdown["failed_subperiod"] += 1
                                        if not param["pass"]:
                                            failures.append("parameter")
                                            breakdown["failed_parameter"] += 1
                                        if not reverse_pass:
                                            failures.append("reverse")
                                            breakdown["failed_reverse"] += 1

                                        name = f"{leader}_{args.timeframe}_{feature}_{mode}_w{window}_q{quantile}_{session}_h{horizon}_cost{cost}"
                                        row = {
                                            "name": name,
                                            "timeframe": args.timeframe,
                                            "leader": leader,
                                            "feature": feature,
                                            "mode": mode,
                                            "window": window,
                                            "quantile": quantile,
                                            "threshold": threshold,
                                            "session": session,
                                            "horizon": horizon,
                                            "cost": cost,
                                            "train": train,
                                            "val": val,
                                            "old_test": old_test,
                                            "forward": fwd,
                                            "reverse": rev,
                                            "old_gate_pass": og,
                                            "bootstrap": boot,
                                            "bootstrap_pass": bool(boot["ci_low"] is not None and boot["ci_low"] > 0),
                                            "subperiod": sub,
                                            "subperiod_pass": bool(sub["pass"]),
                                            "parameter": param,
                                            "parameter_pass": bool(param["pass"]),
                                            "reverse_pass": reverse_pass,
                                            "survived_all": len(failures) == 0,
                                            "failures": failures,
                                        }
                                        f.write(json.dumps(json_sanitize(row), separators=(",", ":")) + "\n")
                                        rows_written += 1
                                        breakdown["evaluated"] += 1
                                        if len(sample_rows) < 5:
                                            sample_rows.append(row)
                                        if row["survived_all"]:
                                            breakdown["survived_all"] += 1
                                            survivors.append(row)
        portfolio = select_portfolio(survivors, forward, args.max_portfolio_models) if survivors else {
            "selected_count": 0,
            "models": [],
            "allocator_policies": {},
            "allocator_swing_ratio": None,
            "allocator_pass": False,
            "effective_bets": {"participation_ratio": 0.0, "kaiser_components_gt_1": 0, "eigenvalues": []},
        }

    expected_grid = len([s for s in symbols if s in old.columns and s in forward.columns]) * len(FEATURES) * len(WINDOWS) * len(QUANTILES) * len(MODES) * len(SESSIONS) * len(horizons) * len(COSTS)
    summary = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "timeframe": args.timeframe,
        "grid_path": str(grid_path),
        "grid_rows": rows_written,
        "expected_grid_rows_if_all_symbols_have_enough_data": expected_grid,
        "summary_path": str(summary_path),
        "universe_snapshot_path": str(snapshot_path),
        "meta": meta,
        "universe_snapshot": snapshot,
        "breakdown": breakdown,
        "survivor_count": len(survivors),
        "survivors": sorted(survivors, key=lambda r: (r["forward"]["pnl"], r["old_test"]["pnl"]), reverse=True)[:100],
        "portfolio": portfolio,
        "required_framing": "Any surviving roster is an in-sample survivor of a stricter gate, not yet out-of-sample validated.",
        "sample_rows": sample_rows,
    }
    summary_path.write_text(json.dumps(json_sanitize(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "timeframe": args.timeframe,
        "grid_path": str(grid_path),
        "grid_rows": rows_written,
        "expected_grid_rows_if_all_symbols_have_enough_data": expected_grid,
        "survivors": len(survivors),
        "portfolio_models": portfolio["selected_count"],
        "allocator_swing_ratio": portfolio["allocator_swing_ratio"],
        "effective_bets": portfolio["effective_bets"]["participation_ratio"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
