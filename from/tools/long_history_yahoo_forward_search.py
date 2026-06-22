#!/usr/bin/env python3
"""Search longer Yahoo intraday leader history and score a forward XAU holdout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cross_market_edge_search as cm
import forward_holdout_search as fh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    p.add_argument("--forward-xau", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    p.add_argument("--interval", choices=["15m", "60m"], default="60m")
    p.add_argument("--range", default="730d")
    p.add_argument("--symbols", nargs="*", required=True)
    p.add_argument("--features", nargs="*", default=["volz", "divergence"], choices=["raw", "volz", "divergence"])
    p.add_argument("--windows", nargs="*", type=int, default=[2, 3, 4, 6, 9, 12, 18, 24])
    p.add_argument("--quantiles", nargs="*", type=float, default=[0.55, 0.65, 0.75, 0.85])
    p.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--costs", nargs="*", type=float, default=[2.0])
    p.add_argument("--sessions", nargs="*", default=["all", "london", "ny"])
    p.add_argument("--vol-window", type=int, default=48)
    p.add_argument("--min-tpd", type=float, default=0.5)
    p.add_argument("--max-tpd", type=float, default=12.0)
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--no-gap-fill", action="store_true", default=True)
    p.add_argument("--allow-gap-fill", action="store_false", dest="no_gap_fill")
    return p.parse_args()


def interval_ms(interval: str) -> int:
    return cm.interval_ms(interval)


def aggregate_forward_xau(path: str, interval: str) -> pd.DataFrame:
    bucket = interval_ms(interval)
    x = pd.read_csv(path).sort_values("time")
    x["time"] = pd.to_datetime(x["time"], utc=True)
    ms = x["time"].map(lambda value: int(value.timestamp() * 1000))
    x["bucket_ms"] = (ms // bucket) * bucket
    g = x.groupby("bucket_ms", sort=True)
    out = g.agg(
        xau=("close", "last"),
        spread=("spread", "mean"),
        ticks=("ticks", "sum"),
    ).reset_index()
    out["time"] = pd.to_datetime(out["bucket_ms"], unit="ms", utc=True)
    return out.set_index("time").sort_index()


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
        "dd": float((peak - eq).max()) if len(pnl) else 0.0,
    }


def session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    if session == "london":
        return np.array([(7 <= t.hour < 16) for t in index])
    if session == "ny":
        return np.array([(13 <= t.hour < 21) for t in index])
    return np.ones(len(index), dtype=bool)


def trade_stats(df: pd.DataFrame, leader: str, feature: str, window: int, threshold: float, mode: str, session: str, horizon: int, cost: float, vol_window: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    d = df.dropna(subset=[leader, "xau", "spread"]).copy()
    vals = fh.feature_values(d, leader, window, feature, vol_window)
    sig = np.asarray(np.where(vals > threshold, 1.0, np.where(vals < -threshold, -1.0, 0.0)), dtype=float)
    if mode == "fade":
        sig = -sig
    future = (d["xau"].shift(-horizon) - d["xau"]).astype(float).values
    smask = np.asarray(session_mask(pd.DatetimeIndex(d.index), session), dtype=bool)
    mask = np.isfinite(sig) & np.isfinite(future) & (sig != 0.0) & smask
    pnl = sig[mask] * future[mask] - d["spread"].fillna(d["spread"].median()).astype(float).values[mask] * cost
    return pnl, d.index[mask]


def main() -> int:
    args = parse_args()
    leaders, errors = cm.load_leaders(args.symbols, args.interval, args.range, Path(args.cache_dir))
    xau_old = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), args.interval)
    old = leaders.join(xau_old[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    old = old.dropna(subset=["xau", "spread"])
    if args.exact_only:
        old = old[old["exact_all"] == 1]
    if args.no_gap_fill:
        old = old[old["gap_fill"] == 0]

    forward_xau = aggregate_forward_xau(args.forward_xau, args.interval)
    forward = leaders.join(forward_xau[["xau", "spread", "ticks"]], how="inner").sort_index().dropna(subset=["xau", "spread"])

    rows = []
    for leader in args.symbols:
        if leader not in old.columns or leader not in forward.columns:
            continue
        old_leader = old.dropna(subset=[leader])
        if len(old_leader) < 500:
            continue
        for feature in args.features:
            for window in args.windows:
                train_values = fh.feature_values(old_leader, leader, window, feature, args.vol_window)
                sample = np.abs(train_values[: len(train_values) // 2])
                sample = sample[np.isfinite(sample)]
                if len(sample) < 250:
                    continue
                for quantile in args.quantiles:
                    threshold = float(np.quantile(sample, quantile))
                    for mode in ["follow", "fade"]:
                        for session in args.sessions:
                            for horizon in args.horizons:
                                for cost in args.costs:
                                    p, t = trade_stats(old_leader, leader, feature, window, threshold, mode, session, horizon, cost, args.vol_window)
                                    if len(p) < 120:
                                        continue
                                    a = len(p) // 2
                                    b = 3 * len(p) // 4
                                    train = stat(p[:a], t[:a])
                                    val = stat(p[a:b], t[a:b])
                                    old_test = stat(p[b:], t[b:])
                                    if not (
                                        train["pnl"] > 0 and val["pnl"] > 0 and old_test["pnl"] > 0
                                        and args.min_tpd <= train["tpd"] <= args.max_tpd
                                        and args.min_tpd <= val["tpd"] <= args.max_tpd
                                        and args.min_tpd <= old_test["tpd"] <= args.max_tpd
                                    ):
                                        continue
                                    pf, tf = trade_stats(forward, leader, feature, window, threshold, mode, session, horizon, cost, args.vol_window)
                                    fwd = stat(pf, tf)
                                    rows.append(
                                        {
                                            "name": f"{leader}_{args.interval}_{feature}_{mode}_w{window}_q{quantile}_{session}_h{horizon}_cost{cost}",
                                            "leader": leader,
                                            "interval": args.interval,
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
                                        }
                                    )
    rows.sort(key=lambda r: (r["forward"]["pnl"], r["old_test"]["pnl"]), reverse=True)
    forward_pass = [r for r in rows if r["forward"]["pnl"] > 0 and args.min_tpd <= r["forward"]["tpd"] <= args.max_tpd]
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "interval": args.interval,
        "range": args.range,
        "old_rows": int(len(old)),
        "forward_rows": int(len(forward)),
        "old_start": str(old.index.min()),
        "old_end": str(old.index.max()),
        "forward_start": str(forward.index.min()),
        "forward_end": str(forward.index.max()),
        "leader_errors": errors,
        "candidate_count": len(rows),
        "forward_positive_count": len(forward_pass),
        "top": rows[:1000],
        "top_forward": forward_pass[:250],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["interval", "old_rows", "forward_rows", "candidate_count", "forward_positive_count"]}, indent=2))
    for r in forward_pass[:20]:
        print(r["name"], "old", round(r["train"]["pnl"], 1), round(r["val"]["pnl"], 1), round(r["old_test"]["pnl"], 1), "FWD", round(r["forward"]["pnl"], 1), round(r["forward"]["tpd"], 2), round(r["forward"]["pf"], 2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
