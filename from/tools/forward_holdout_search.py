#!/usr/bin/env python3
"""Select cross-market rules on pre-forward data and score a forward XAU holdout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cross_market_edge_search as cm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    p.add_argument("--forward-xau", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    p.add_argument("--symbols", nargs="*", default=["SPY", "GC=F", "SI=F", "DX-Y.NYB", "ZN=F", "ZB=F", "CL=F"])
    p.add_argument("--windows", nargs="*", type=int, default=[2, 3, 4, 6, 9, 12])
    p.add_argument("--quantiles", nargs="*", type=float, default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    p.add_argument("--horizons", nargs="*", type=int, default=[6, 9, 12])
    p.add_argument("--costs", nargs="*", type=float, default=[1.0, 1.5, 2.0])
    p.add_argument("--sessions", nargs="*", default=["all", "london", "ny"])
    p.add_argument("--features", nargs="*", default=["raw"], choices=["raw", "volz", "divergence"])
    p.add_argument("--vol-window", type=int, default=48)
    p.add_argument("--exact-only", action="store_true", default=True)
    p.add_argument("--allow-non-exact", action="store_false", dest="exact_only")
    p.add_argument("--no-gap-fill", action="store_true", default=True)
    p.add_argument("--allow-gap-fill", action="store_false", dest="no_gap_fill")
    return p.parse_args()


def stat(p: np.ndarray, t: pd.DatetimeIndex) -> dict[str, float | int]:
    if len(p) == 0:
        return {"trades": 0, "tpd": 0.0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "sh": 0.0, "dd": 0.0}
    days = max(1e-9, (t[-1] - t[0]).total_seconds() / 86400.0)
    avg = float(p.mean())
    sd = float(p.std(ddof=1)) if len(p) > 1 else 0.0
    win = float(p[p > 0].sum())
    loss = float(-p[p < 0].sum())
    eq = p.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(p)),
        "tpd": float(len(p) / days),
        "pnl": float(p.sum()),
        "wr": float((p > 0).mean()),
        "pf": float(win / loss) if loss > 0 else 0.0,
        "sh": float(avg / sd * math.sqrt(len(p))) if sd > 0 else 0.0,
        "dd": float((peak - eq).max()) if len(p) else 0.0,
    }


def feature_values(df: pd.DataFrame, leader: str, window: int, feature: str, vol_window: int) -> np.ndarray:
    leader_ret = np.log(df[leader]).diff()
    momentum = leader_ret.rolling(window).sum().shift(1)
    if feature == "raw":
        return momentum.values
    if feature == "volz":
        vol = leader_ret.rolling(vol_window).std().shift(1) * math.sqrt(window)
        return (momentum / vol.replace(0.0, np.nan)).values
    if feature == "divergence":
        xau_ret = np.log(df["xau"]).diff().rolling(window).sum().shift(1)
        leader_vol = leader_ret.rolling(vol_window).std().shift(1) * math.sqrt(window)
        xau_vol = np.log(df["xau"]).diff().rolling(vol_window).std().shift(1) * math.sqrt(window)
        leader_z = momentum / leader_vol.replace(0.0, np.nan)
        xau_z = xau_ret / xau_vol.replace(0.0, np.nan)
        return (leader_z - xau_z).values
    raise ValueError(feature)


def trade_stats(df: pd.DataFrame, leader: str, threshold: float, window: int, horizon: int, cost: float, mode: str, session: str, feature: str, vol_window: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    d = df.dropna(subset=[leader]).copy()
    idx = d.index
    values = feature_values(d, leader, window, feature, vol_window)
    signal = np.where(values > threshold, 1.0, np.where(values < -threshold, -1.0, 0.0))
    if mode == "fade":
        signal = -signal
    if session == "london":
        session_mask = np.array([(7 <= t.hour < 16) for t in idx])
    elif session == "ny":
        session_mask = np.array([(13 <= t.hour < 21) for t in idx])
    else:
        session_mask = np.ones(len(d), dtype=bool)
    future = d.xau.shift(-horizon) - d.xau
    valid = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0) & session_mask
    pnl = signal[valid] * future.values[valid] - d.spread.fillna(d.spread.median()).values[valid] * cost
    return pnl, idx[valid]


def main() -> int:
    args = parse_args()
    leaders, errors = cm.load_leaders(args.symbols, "5m", "60d", Path(args.cache_dir))
    xau_old = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), "5m")
    old = leaders.join(xau_old[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    old = old.dropna(subset=["xau", "spread"])
    if args.exact_only:
        old = old[old.exact_all == 1]
    if args.no_gap_fill:
        old = old[old.gap_fill == 0]
    old = old.copy()

    forward_xau = pd.read_csv(args.forward_xau, parse_dates=["time"]).rename(columns={"close": "xau"}).set_index("time").sort_index()
    forward = leaders.join(forward_xau[["xau", "spread", "ticks"]], how="inner").sort_index().dropna(subset=["xau", "spread"])

    rows = []
    for leader in args.symbols:
        if leader not in old.columns or leader not in forward.columns:
            continue
        old_leader = old.dropna(subset=[leader])
        if len(old_leader) < 500:
            continue
        for window in args.windows:
            for feature in args.features:
                train_values = feature_values(old_leader, leader, window, feature, args.vol_window)
                train_mom = np.abs(train_values[: len(old_leader) // 2])
                train_mom = train_mom[np.isfinite(train_mom)]
                if len(train_mom) < 100:
                                    continue
                for quantile in args.quantiles:
                    threshold = float(np.quantile(train_mom, quantile))
                    for mode in ["fade", "follow"]:
                        for session in args.sessions:
                            for horizon in args.horizons:
                                for cost in args.costs:
                                    p, t = trade_stats(old_leader, leader, threshold, window, horizon, cost, mode, session, feature, args.vol_window)
                                    if len(p) < 160:
                                        continue
                                    a = len(p) // 2
                                    b = 3 * len(p) // 4
                                    train = stat(p[:a], t[:a])
                                    val = stat(p[a:b], t[a:b])
                                    old_test = stat(p[b:], t[b:])
                                    pfwd, tfwd = trade_stats(forward, leader, threshold, window, horizon, cost, mode, session, feature, args.vol_window)
                                    forward_stat = stat(pfwd, tfwd)
                                    if train["pnl"] > 0 and val["pnl"] > 0 and 15 <= train["tpd"] <= 35 and 15 <= val["tpd"] <= 35 and forward_stat["trades"] >= 50:
                                        rows.append(
                                            {
                                                "name": f"{leader}_{feature}_{mode}_w{window}_q{quantile}_{session}_h{horizon}_cost{cost}",
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
                                                "forward": forward_stat,
                                            }
                                        )
    rows.sort(key=lambda r: (r["forward"]["pnl"], r["old_test"]["pnl"]), reverse=True)
    forward_band = [r for r in rows if r["forward"]["pnl"] > 0 and 15 <= r["forward"]["tpd"] <= 35]
    forward_band_2x = [r for r in forward_band if r["cost"] >= 2.0]
    report = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "old_rows": int(len(old)),
        "forward_rows": int(len(forward)),
        "old_start": str(old.index.min()),
        "old_end": str(old.index.max()),
        "forward_start": str(forward.index.min()),
        "forward_end": str(forward.index.max()),
        "leader_errors": errors,
        "old_exact_only": bool(args.exact_only),
        "old_no_gap_fill": bool(args.no_gap_fill),
        "candidate_count": len(rows),
        "forward_positive_15_35_count": len(forward_band),
        "forward_positive_15_35_2x_count": len(forward_band_2x),
        "top": rows,
        "top_forward_15_35": forward_band[:100],
        "top_forward_15_35_2x": forward_band_2x[:100],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["old_rows", "forward_rows", "candidate_count", "forward_positive_15_35_count", "forward_positive_15_35_2x_count"]}, indent=2))
    for row in forward_band[:20]:
        print(row["name"], "old", round(row["train"]["pnl"], 1), round(row["val"]["pnl"], 1), round(row["old_test"]["pnl"], 1), "FWD", round(row["forward"]["pnl"], 1), round(row["forward"]["tpd"], 1), round(row["forward"]["wr"], 3), round(row["forward"]["pf"], 2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
