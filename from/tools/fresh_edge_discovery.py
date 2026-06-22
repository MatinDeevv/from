#!/usr/bin/env python3
"""Fresh mechanism-tagged XAU edge discovery screen."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cross_market_edge_search as cm
import long_history_yahoo_forward_search as lh
import strict_screening_pipeline as ss


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
MASTER = r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet"
FORWARD_XAU = str(DATA / "derived" / "xauusd_dukascopy_5m_forward_20260522_20260618.csv")

MECHANISMS = {
    "energy_commodity_follow": ["UNG", "USO", "CL=F"],
    "us_sector_risk_fade": ["XLF", "XLV", "XLY", "XLK"],
    "silver_leads_gold": ["SI=F"],
    "long_duration_rates_fade": ["TLT", "IEF", "ZB=F"],
    "fx_safe_haven_divergence": ["USDJPY=X", "DX-Y.NYB", "EURUSD=X"],
    "crypto_liquidity_impulse": ["BTC-USD", "ETH-USD"],
}
SYMBOL_TO_MECH = {s: k for k, vals in MECHANISMS.items() for s in vals}
SYMBOLS = [s for vals in MECHANISMS.values() for s in vals]

FEATURES = ["volz", "divergence", "momentum_ratio", "ew_zspread", "session_volz"]
WINDOWS = [2, 3, 4, 6, 9, 12, 18, 24]
QUANTILES = [0.55, 0.65, 0.75, 0.85]
MODES = ["follow", "fade"]
SESSIONS = ["all", "london", "ny"]
HORIZONS_5M = [6, 12, 18, 24, 30, 36, 42, 48]
HORIZONS_60M = [1, 2, 3, 4, 6, 8]
COSTS = [2.0]

MECHANISM_MODES = {
    "energy_commodity_follow": ["follow"],
    "us_sector_risk_fade": ["fade"],
    "silver_leads_gold": ["follow"],
    "long_duration_rates_fade": ["fade"],
    "fx_safe_haven_divergence": ["follow", "fade"],
    "crypto_liquidity_impulse": ["follow", "fade"],
}

MECHANISM_SESSIONS = {
    "energy_commodity_follow": ["london"],
    "us_sector_risk_fade": ["london", "ny"],
    "silver_leads_gold": ["all"],
    "long_duration_rates_fade": ["all"],
    "fx_safe_haven_divergence": ["all", "london", "ny"],
    "crypto_liquidity_impulse": ["all", "london", "ny"],
}

SPLITS = {
    "5m": {
        "train": ("2026-03-24 08:00:00+00:00", "2026-04-23 23:59:59+00:00"),
        "val": ("2026-04-24 00:00:00+00:00", "2026-05-08 23:59:59+00:00"),
        "old_test": ("2026-05-09 00:00:00+00:00", "2026-05-22 13:50:00+00:00"),
        "forward": ("2026-05-22 13:55:00+00:00", "2026-06-17 23:55:00+00:00"),
    },
    "60m": {
        "train": ("2023-07-21 08:00:00+00:00", "2025-07-01 00:00:00+00:00"),
        "val": ("2025-07-01 00:00:00+00:00", "2026-01-01 00:00:00+00:00"),
        "old_test": ("2026-01-01 00:00:00+00:00", "2026-05-22 12:59:59+00:00"),
        "forward": ("2026-05-22 13:00:00+00:00", "2026-06-17 23:00:00+00:00"),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", choices=["5m", "60m"], required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--bootstrap-trials", type=int, default=2000)
    p.add_argument("--cache-dir", default=str(DATA / "cache" / "yahoo"))
    p.add_argument("--master", default=MASTER)
    p.add_argument("--forward-xau", default=FORWARD_XAU)
    p.add_argument("--limit-symbols", nargs="*", default=None)
    return p.parse_args()


def between(df: pd.DataFrame, tf: str, split: str) -> pd.DataFrame:
    start, end = [pd.Timestamp(x) for x in SPLITS[tf][split]]
    return df[(df.index >= start) & (df.index <= end)].copy()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    range_ = "60d" if args.timeframe == "5m" else "730d"
    symbols = args.limit_symbols or SYMBOLS
    leaders, errors = cm.load_leaders(symbols, args.timeframe, range_, Path(args.cache_dir))
    xau_old = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), args.timeframe)
    old = leaders.join(xau_old[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    old = old.dropna(subset=["xau", "spread"])
    old = old[old["gap_fill"] == 0].copy()
    if args.timeframe == "5m":
        fwd_xau = pd.read_csv(args.forward_xau, parse_dates=["time"]).rename(columns={"close": "xau"}).set_index("time").sort_index()
    else:
        fwd_xau = lh.aggregate_forward_xau(args.forward_xau, args.timeframe)
    forward = leaders.join(fwd_xau[["xau", "spread", "ticks"]], how="inner").sort_index().dropna(subset=["xau", "spread"])
    forward = between(forward, args.timeframe, "forward")
    profile = pd.read_csv(REPORTS / f"fresh_edge_{args.timeframe}_mechanism_data_profile.csv")
    excluded = profile[profile["exclude"].astype(bool)]["symbol"].tolist()
    old = old[[c for c in old.columns if c not in excluded or c in ["xau", "spread", "ticks", "exact_all", "gap_fill"]]]
    forward = forward[[c for c in forward.columns if c not in excluded or c in ["xau", "spread", "ticks"]]]
    meta = {
        "timeframe": args.timeframe,
        "symbols": [s for s in symbols if s not in excluded],
        "excluded_symbols": excluded,
        "leader_errors": errors,
        "old_rows": int(len(old)),
        "forward_rows": int(len(forward)),
        "old_start": str(old.index.min()),
        "old_end": str(old.index.max()),
        "forward_start": str(forward.index.min()),
        "forward_end": str(forward.index.max()),
        "splits": SPLITS[args.timeframe],
    }
    return old, forward, meta


def session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    return ss.session_mask(index, session)


def feature_values(df: pd.DataFrame, leader: str, window: int, feature: str, session: str) -> np.ndarray:
    lr = np.log(df[leader]).diff()
    xr = np.log(df["xau"]).diff()
    lmom = lr.rolling(window).sum().shift(1)
    xmom = xr.rolling(window).sum().shift(1)
    if feature == "volz":
        vol = lr.rolling(48).std().shift(1) * math.sqrt(window)
        out = lmom / vol.replace(0.0, np.nan)
    elif feature == "divergence":
        lv = lr.rolling(48).std().shift(1) * math.sqrt(window)
        xv = xr.rolling(48).std().shift(1) * math.sqrt(window)
        out = (lmom / lv.replace(0.0, np.nan)) - (xmom / xv.replace(0.0, np.nan))
    elif feature == "momentum_ratio":
        out = lmom / xmom.abs().replace(0.0, np.nan)
    elif feature == "ew_zspread":
        lv = lr.ewm(span=48, adjust=False, min_periods=12).std().shift(1) * math.sqrt(window)
        xv = xr.ewm(span=48, adjust=False, min_periods=12).std().shift(1) * math.sqrt(window)
        out = (lmom / lv.replace(0.0, np.nan)) - (xmom / xv.replace(0.0, np.nan))
    elif feature == "session_volz":
        vol = lr.rolling(48).std().shift(1) * math.sqrt(window)
        out = lmom / vol.replace(0.0, np.nan)
        out = out.mask(~session_mask(pd.DatetimeIndex(df.index), session), 0.0)
    else:
        raise ValueError(feature)
    return out.to_numpy(float)


def trade_sequence(df: pd.DataFrame, leader: str, feature: str, window: int, threshold: float, mode: str, session: str, horizon: int, cost: float, *, reverse: bool = False) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    d = df.dropna(subset=[leader, "xau", "spread"]).copy()
    vals = feature_values(d, leader, window, feature, session)
    sig = np.where(vals > threshold, 1.0, np.where(vals < -threshold, -1.0, 0.0))
    if mode == "fade":
        sig = -sig
    if reverse:
        sig = -sig
    smask = session_mask(pd.DatetimeIndex(d.index), session)
    future = (d["xau"].shift(-horizon) - d["xau"]).astype(float).to_numpy()
    spread = d["spread"].fillna(d["spread"].median()).clip(lower=0.0).astype(float).to_numpy()
    mask = np.isfinite(sig) & np.isfinite(future) & np.isfinite(spread) & (sig != 0.0) & smask
    pnl = sig[mask] * future[mask] - spread[mask] * cost
    times = pd.DatetimeIndex(d.index[mask])
    details = pd.DataFrame({
        "entry_time": times,
        "exit_time": pd.Series(d.index, index=d.index).shift(-horizon).to_numpy()[mask],
        "direction": sig[mask].astype(int),
        "xau_entry": d["xau"].astype(float).to_numpy()[mask],
        "pnl_per_oz": pnl,
        "pnl_per_dollar": pnl / d["xau"].astype(float).to_numpy()[mask],
    }).dropna(subset=["exit_time", "pnl_per_dollar"])
    return pnl[: len(details)], pd.DatetimeIndex(details["entry_time"]), details


def period_stats(old: pd.DataFrame, leader: str, feature: str, window: int, threshold: float, mode: str, session: str, horizon: int, cost: float, tf: str) -> dict[str, dict]:
    out = {}
    for split in ["train", "val", "old_test"]:
        pnl, times, _ = trade_sequence(between(old, tf, split), leader, feature, window, threshold, mode, session, horizon, cost)
        out[split] = ss.stat(pnl, times)
    return out


def old_gate(stats: dict, fwd: dict, tf: str) -> bool:
    lo, hi = (15.0, 35.0) if tf == "5m" else (0.5, 12.0)
    return bool(
        stats["train"]["pnl"] > 0 and stats["val"]["pnl"] > 0 and stats["old_test"]["pnl"] > 0
        and fwd["pnl"] > 0
        and lo <= stats["train"]["tpd"] <= hi
        and lo <= stats["val"]["tpd"] <= hi
        and lo <= stats["old_test"]["tpd"] <= hi
    )


def parameter_check(old: pd.DataFrame, forward: pd.DataFrame, row: dict, tf: str) -> dict:
    base = row["forward"]["pnl"]
    variants = []
    train_df = between(old, tf, "train").dropna(subset=[row["leader"]])
    for field, factor in [("window", 0.8), ("window", 1.2), ("quantile", 0.9), ("quantile", 1.1), ("horizon", 0.8), ("horizon", 1.2)]:
        window, q, horizon = int(row["window"]), float(row["quantile"]), int(row["horizon"])
        if field == "window":
            window = max(2, int(round(window * factor)))
        elif field == "quantile":
            q = min(0.95, max(0.05, q * factor))
        else:
            horizon = max(1, int(round(horizon * factor)))
        vals = feature_values(train_df, row["leader"], window, row["feature"], row["session"])
        sample = np.abs(vals[np.isfinite(vals)])
        threshold = float(np.quantile(sample, q)) if len(sample) >= 100 else float(row["threshold"])
        pnl, times, _ = trade_sequence(forward, row["leader"], row["feature"], window, threshold, row["mode"], row["session"], horizon, row["cost"])
        st = ss.stat(pnl, times)
        variants.append({"field": field, "factor": factor, "pnl": st["pnl"], "pf": st["pf"], "trades": st["trades"]})
    failures = [v for v in variants if v["pnl"] <= 0 or v["pnl"] < 0.5 * base]
    return {"pass": len(failures) == 0, "failures": len(failures), "variants": variants}


def portfolio_trades(forward: pd.DataFrame, models: list[dict], allocation: float) -> pd.DataFrame:
    frames = []
    for m in models:
        _, _, details = trade_sequence(forward, m["leader"], m["feature"], int(m["window"]), float(m["threshold"]), m["mode"], m["session"], int(m["horizon"]), float(m["cost"]))
        if len(details) == 0:
            continue
        details["model"] = m["name"]
        details["leader"] = m["leader"]
        details["mechanism_family"] = m["mechanism_family"]
        details["notional"] = allocation
        details["pnl"] = details["pnl_per_dollar"] * allocation
        details["edge_key"] = float(m["forward"]["pnl"]) / max(1, int(m["forward"]["trades"]))
        details["vol_key"] = float(m["forward"]["dd"]) / max(1.0, float(m["forward"]["pnl"]))
        frames.append(details)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_portfolio(survivors: list[dict], forward: pd.DataFrame) -> dict:
    picked = []
    families = set()
    for s in sorted(survivors, key=lambda r: (r["forward"]["pnl"], r["old_test"]["pnl"]), reverse=True):
        if any(p["leader"] == s["leader"] and p["mechanism_family"] == s["mechanism_family"] for p in picked):
            continue
        trial_fams = families | {s["mechanism_family"]}
        picked.append(s)
        families = trial_fams
        if len(picked) >= 10:
            break
    trades = portfolio_trades(forward, picked, 5_000_000.0 / max(1, len(picked)))
    policies = {p: ss.portfolio_stats(ss.cap_policy(trades, p)) for p in ["edge_rank", "inverse_vol", "name_order"]} if len(trades) else {}
    pnls = [v["pnl"] for v in policies.values()]
    swing = (max(pnls) - min(pnls)) / max(1.0, abs(policies["edge_rank"]["pnl"])) if pnls else None
    eb = ss.effective_bets(trades) if len(trades) else {"participation_ratio": 0.0, "kaiser_components_gt_1": 0, "eigenvalues": []}
    family_count = len({m["mechanism_family"] for m in picked})
    pass_ = bool(len(picked) >= 10 and family_count >= 3 and swing is not None and swing <= 0.30 and eb["participation_ratio"] >= 5)
    eq_path = None
    if len(trades):
        capped = ss.cap_policy(trades, "edge_rank")
        capped["exit_date"] = pd.to_datetime(capped["exit_time"], utc=True).dt.date.astype(str)
        eq = capped.groupby("exit_date")["pnl"].sum().cumsum().reset_index(name="cum_pnl")
        eq_path = None
    return {"selected_count": len(picked), "mechanism_family_count": family_count, "models": picked, "allocator_policies": policies, "allocator_swing_ratio": swing, "effective_bets": eb, "portfolio_pass": pass_, "equity_curve": eq.to_dict("records") if len(trades) else [], "equity_curve_path": eq_path}


def main() -> int:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = out_prefix.with_name(out_prefix.name + "_full_grid.jsonl")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")
    old, forward, meta = load_data(args)
    horizons = HORIZONS_5M if args.timeframe == "5m" else HORIZONS_60M
    rng = np.random.default_rng(20260618)
    breakdown = {"evaluated": 0, "insufficient_forward_trades": 0, "failed_old_gate": 0, "failed_bootstrap": 0, "failed_subperiod": 0, "failed_parameter": 0, "failed_reverse": 0, "survived_all": 0}
    by_mech = {m: dict(breakdown) for m in MECHANISMS}
    survivors, top_by_mech = [], {m: [] for m in MECHANISMS}
    samples = []
    with grid_path.open("w", encoding="utf-8") as f:
        for leader in meta["symbols"]:
            if leader not in old.columns or leader not in forward.columns:
                continue
            mech = SYMBOL_TO_MECH[leader]
            train_df = between(old, args.timeframe, "train").dropna(subset=[leader])
            for feature in FEATURES:
                for window in WINDOWS:
                    vals = feature_values(train_df, leader, window, feature, "all")
                    sample = np.abs(vals[np.isfinite(vals)])
                    if len(sample) < 100:
                        continue
                    for q in QUANTILES:
                        threshold = float(np.quantile(sample, q))
                        for mode in MECHANISM_MODES[mech]:
                            for session in MECHANISM_SESSIONS[mech]:
                                if feature == "session_volz":
                                    vals2 = feature_values(train_df, leader, window, feature, session)
                                    sample2 = np.abs(vals2[np.isfinite(vals2)])
                                    if len(sample2) < 100:
                                        continue
                                    threshold2 = float(np.quantile(sample2, q))
                                else:
                                    threshold2 = threshold
                                for horizon in horizons:
                                    for cost in COSTS:
                                        pnl_f, times_f, _ = trade_sequence(forward, leader, feature, window, threshold2, mode, session, horizon, cost)
                                        fwd = ss.stat(pnl_f, times_f)
                                        stats = period_stats(old, leader, feature, window, threshold2, mode, session, horizon, cost, args.timeframe)
                                        og = old_gate(stats, fwd, args.timeframe)
                                        failures = []
                                        breakdown["evaluated"] += 1
                                        by_mech[mech]["evaluated"] += 1
                                        insufficient = fwd["trades"] < 100
                                        if insufficient:
                                            failures.append("insufficient_forward_trades")
                                            breakdown["insufficient_forward_trades"] += 1
                                            by_mech[mech]["insufficient_forward_trades"] += 1
                                        if not og:
                                            failures.append("old_gate")
                                            breakdown["failed_old_gate"] += 1
                                            by_mech[mech]["failed_old_gate"] += 1
                                        block = max(4, horizon * 3)
                                        approx = ss.normal_lower_bound(pnl_f)
                                        if (not insufficient) and og and approx > 0:
                                            boot = ss.block_bootstrap_lower(pnl_f, block, args.bootstrap_trials, rng)
                                        else:
                                            boot = {"trials": args.bootstrap_trials, "block": block, "ci_low": float("-inf"), "ci_high": None, "normal_approx_ci_low": approx, "not_run_reason": "insufficient_trades_old_gate_failed_or_normal_lower_bound_nonpositive"}
                                        if not (boot["ci_low"] is not None and boot["ci_low"] > 0):
                                            failures.append("bootstrap")
                                            breakdown["failed_bootstrap"] += 1
                                            by_mech[mech]["failed_bootstrap"] += 1
                                        sub = ss.subperiod_check(pnl_f)
                                        if not sub["pass"]:
                                            failures.append("subperiod")
                                            breakdown["failed_subperiod"] += 1
                                            by_mech[mech]["failed_subperiod"] += 1
                                        if (not insufficient) and og and boot["ci_low"] > 0 and sub["pass"]:
                                            rev_pnl, rev_times, _ = trade_sequence(forward, leader, feature, window, threshold2, mode, session, horizon, cost, reverse=True)
                                            rev = ss.stat(rev_pnl, rev_times)
                                        else:
                                            rev = {"trades": 0, "tpd": 0.0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "sh": 0.0, "dd": 0.0, "not_run_reason": "short_circuited"}
                                        reverse_pass = bool(fwd["pnl"] > 0 and rev["pnl"] <= -0.25 * abs(fwd["pnl"]))
                                        if not reverse_pass:
                                            failures.append("reverse")
                                            breakdown["failed_reverse"] += 1
                                            by_mech[mech]["failed_reverse"] += 1
                                        if (not insufficient) and og and boot["ci_low"] > 0 and sub["pass"] and reverse_pass:
                                            param = parameter_check(old, forward, {"leader": leader, "feature": feature, "window": window, "quantile": q, "threshold": threshold2, "mode": mode, "session": session, "horizon": horizon, "cost": cost, "forward": fwd}, args.timeframe)
                                        else:
                                            param = {"pass": False, "failures": None, "variants": [], "not_run_reason": "short_circuited"}
                                        if not param["pass"]:
                                            failures.append("parameter")
                                            breakdown["failed_parameter"] += 1
                                            by_mech[mech]["failed_parameter"] += 1
                                        name = f"{leader}_{args.timeframe}_{feature}_{mode}_w{window}_q{q}_{session}_h{horizon}_cost{cost}"
                                        row = {
                                            "name": name, "timeframe": args.timeframe, "mechanism_family": mech, "leader": leader, "feature": feature, "mode": mode, "window": window, "quantile": q, "threshold": threshold2, "session": session, "horizon": horizon, "cost": cost,
                                            **stats, "forward": fwd, "bootstrap": boot, "subperiod": sub, "reverse": rev, "parameter": param,
                                            "insufficient_forward_trades": insufficient, "old_gate_pass": og, "bootstrap_pass": boot["ci_low"] is not None and boot["ci_low"] > 0, "subperiod_pass": sub["pass"], "reverse_pass": reverse_pass, "parameter_pass": param["pass"], "failures": failures, "survived_all": len(failures) == 0,
                                        }
                                        if row["survived_all"]:
                                            breakdown["survived_all"] += 1
                                            by_mech[mech]["survived_all"] += 1
                                            survivors.append(row)
                                        top_by_mech[mech].append(row)
                                        top_by_mech[mech] = sorted(top_by_mech[mech], key=lambda r: r["forward"]["pnl"], reverse=True)[:20]
                                        if len(samples) < 5:
                                            samples.append(row)
                                        f.write(json.dumps(ss.json_sanitize(row), separators=(",", ":")) + "\n")
    portfolio = select_portfolio(survivors, forward) if survivors else {"selected_count": 0, "mechanism_family_count": 0, "models": [], "allocator_policies": {}, "allocator_swing_ratio": None, "effective_bets": {"participation_ratio": 0.0}, "portfolio_pass": False, "equity_curve": []}
    eq_path = out_prefix.with_name(out_prefix.name + "_equity_curve.csv")
    pd.DataFrame(portfolio.get("equity_curve", [])).to_csv(eq_path, index=False)
    portfolio["equity_curve_path"] = str(eq_path)
    summary = {"created_utc": dt.datetime.now(dt.UTC).isoformat(), "phase0_phase1_path": str(REPORTS / "fresh_edge_phase0_phase1.md"), "grid_path": str(grid_path), "grid_rows": breakdown["evaluated"], "summary_path": str(summary_path), "meta": meta, "features": FEATURES, "mechanism_modes": MECHANISM_MODES, "mechanism_sessions": MECHANISM_SESSIONS, "breakdown": breakdown, "breakdown_by_mechanism": by_mech, "survivor_count": len(survivors), "survivors": sorted(survivors, key=lambda r: r["forward"]["pnl"], reverse=True)[:200], "portfolio": portfolio, "top20_by_mechanism": top_by_mech, "sample_rows": samples, "required_framing": "Any surviving portfolio is an in-sample survivor of strict gate, not yet out-of-sample validated."}
    summary_path.write_text(json.dumps(ss.json_sanitize(summary), indent=2), encoding="utf-8")
    print(json.dumps({"timeframe": args.timeframe, "grid_rows": breakdown["evaluated"], "survivors": len(survivors), "portfolio_pass": portfolio["portfolio_pass"], "mechanism_family_count": portfolio["mechanism_family_count"], "effective_bets": portfolio["effective_bets"]["participation_ratio"], "allocator_swing": portfolio["allocator_swing_ratio"], "grid_path": str(grid_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
