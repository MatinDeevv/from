#!/usr/bin/env python3
"""Search one local leader CSV on old XAU and score untouched forward XAU."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import validate_locked_local_leader as vl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--old-leader-csv", required=True)
    p.add_argument("--forward-leader-csv", required=True)
    p.add_argument("--forward-xau-csv", required=True)
    p.add_argument("--leader-name", default="EURUSD")
    p.add_argument("--out", required=True)
    p.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    p.add_argument("--features", nargs="*", default=["raw", "volz", "divergence"], choices=["raw", "volz", "divergence"])
    p.add_argument("--windows", nargs="*", type=int, default=[12, 18, 24, 30, 36, 42, 48, 60])
    p.add_argument("--quantiles", nargs="*", type=float, default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--horizons", nargs="*", type=int, default=[6, 9, 12, 18, 24])
    p.add_argument("--costs", nargs="*", type=float, default=[2.0])
    p.add_argument("--sessions", nargs="*", default=["london", "ny", "all"])
    p.add_argument("--filters", nargs="*", default=["none"], choices=["none", "xau_vol_high", "xau_vol_low", "leader_vol_high", "leader_vol_low", "spread_low"])
    p.add_argument("--vol-window", type=int, default=48)
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--no-gap-fill", action="store_true", default=True)
    p.add_argument("--allow-gap-fill", action="store_false", dest="no_gap_fill")
    return p.parse_args()


def load_old(args: argparse.Namespace) -> pd.DataFrame:
    ns = argparse.Namespace(
        master=args.master,
        leader_csv=args.old_leader_csv,
        xau_csv=None,
        leader_name=args.leader_name,
        exact_only=args.exact_only,
        no_gap_fill=args.no_gap_fill,
    )
    return vl.load_joined(ns)


def load_forward(args: argparse.Namespace) -> pd.DataFrame:
    leader = pd.read_csv(args.forward_leader_csv, parse_dates=["time"]).set_index("time").sort_index()
    xau = pd.read_csv(args.forward_xau_csv, parse_dates=["time"]).rename(columns={"close": "xau"}).set_index("time").sort_index()
    leader = leader.rename(columns={"close": args.leader_name})
    return leader[[args.leader_name]].join(xau[["xau", "spread", "ticks"]], how="inner").dropna().sort_index()


def stat(pnl: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float | int]:
    return vl.summarize(pnl, times)


def filter_values(px: pd.DataFrame, args: argparse.Namespace, filter_name: str) -> pd.Series:
    if filter_name in {"xau_vol_high", "xau_vol_low"}:
        return np.log(px["xau"]).diff().rolling(args.vol_window).std().shift(1)
    if filter_name in {"leader_vol_high", "leader_vol_low"}:
        return np.log(px[args.leader_name]).diff().rolling(args.vol_window).std().shift(1)
    if filter_name == "spread_low":
        return px["spread"].shift(1)
    return pd.Series(1.0, index=px.index)


def filter_mask(px: pd.DataFrame, args: argparse.Namespace, filter_name: str, threshold: float | None) -> pd.Series:
    if filter_name == "none":
        return pd.Series(True, index=px.index)
    values = filter_values(px, args, filter_name)
    if filter_name.endswith("_high"):
        return values >= float(threshold)
    return values <= float(threshold)


def trade_stats(px: pd.DataFrame, args: argparse.Namespace, feature: str, mode: str, window: int, threshold: float, horizon: int, cost: float, session: str, filter_name: str, filter_threshold: float | None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    ns = argparse.Namespace(
        leader_name=args.leader_name,
        window=window,
        threshold=threshold,
        horizon_bars=horizon,
        cost_mult=cost,
        mode=mode,
        feature=feature,
        vol_window=args.vol_window,
        session=session,
        execution="overlap",
    )
    trades = vl.build_trades(px, ns)
    mask = filter_mask(px, args, filter_name, filter_threshold).reindex(trades.index).fillna(False)
    trades = trades[mask.values]
    return trades["pnl"].values, trades.index


def threshold_sample(px: pd.DataFrame, args: argparse.Namespace, feature: str, window: int) -> np.ndarray:
    ns = argparse.Namespace(leader_name=args.leader_name, window=window, feature=feature, vol_window=args.vol_window)
    vals = vl.feature_values(px, ns)
    vals = np.abs(vals[: len(vals) // 2])
    return vals[np.isfinite(vals)]


def fit_filter_threshold(px: pd.DataFrame, args: argparse.Namespace, filter_name: str) -> float | None:
    if filter_name == "none":
        return None
    values = filter_values(px, args, filter_name).iloc[: len(px) // 2].dropna().values
    if len(values) < 500:
        return None
    if filter_name == "spread_low":
        return float(np.quantile(values, 0.60))
    return float(np.quantile(values, 0.50))


def main() -> int:
    args = parse_args()
    old = load_old(args)
    forward = load_forward(args)
    rows = []
    for feature in args.features:
        for window in args.windows:
            sample = threshold_sample(old, args, feature, window)
            if len(sample) < 500:
                continue
            for quantile in args.quantiles:
                threshold = float(np.quantile(sample, quantile))
                for mode in ["follow", "fade"]:
                    for session in args.sessions:
                        for filter_name in args.filters:
                            filter_threshold = fit_filter_threshold(old, args, filter_name)
                            if filter_name != "none" and filter_threshold is None:
                                continue
                            for horizon in args.horizons:
                                for cost in args.costs:
                                    p, t = trade_stats(old, args, feature, mode, window, threshold, horizon, cost, session, filter_name, filter_threshold)
                                    if len(p) < 300:
                                        continue
                                    a = len(p) // 2
                                    b = 3 * len(p) // 4
                                    train = stat(p[:a], t[:a])
                                    val = stat(p[a:b], t[a:b])
                                    old_test = stat(p[b:], t[b:])
                                    if not (
                                        train["pnl"] > 0
                                        and val["pnl"] > 0
                                        and old_test["pnl"] > 0
                                        and 15 <= train["trades_per_day"] <= 35
                                        and 15 <= val["trades_per_day"] <= 35
                                        and 15 <= old_test["trades_per_day"] <= 35
                                    ):
                                        continue
                                    pf, tf = trade_stats(forward, args, feature, mode, window, threshold, horizon, cost, session, filter_name, filter_threshold)
                                    fwd = stat(pf, tf)
                                    rows.append(
                                        {
                                            "name": f"{args.leader_name}_{feature}_{mode}_w{window}_q{quantile}_{session}_{filter_name}_h{horizon}_cost{cost}",
                                            "feature": feature,
                                            "mode": mode,
                                            "window": window,
                                            "quantile": quantile,
                                            "threshold": threshold,
                                            "session": session,
                                            "filter": filter_name,
                                            "filter_threshold": filter_threshold,
                                            "horizon": horizon,
                                            "cost": cost,
                                            "train": train,
                                            "val": val,
                                            "old_test": old_test,
                                            "forward": fwd,
                                        }
                                    )
    rows.sort(key=lambda r: (r["forward"]["pnl"], r["old_test"]["pnl"]), reverse=True)
    forward_band = [r for r in rows if r["forward"]["pnl"] > 0 and 15 <= r["forward"]["trades_per_day"] <= 35]
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "old_rows": int(len(old)),
        "forward_rows": int(len(forward)),
        "old_start": str(old.index.min()),
        "old_end": str(old.index.max()),
        "forward_start": str(forward.index.min()),
        "forward_end": str(forward.index.max()),
        "candidate_count": len(rows),
        "forward_positive_15_35_count": len(forward_band),
        "top": rows[:500],
        "top_forward_15_35": forward_band[:100],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["old_rows", "forward_rows", "candidate_count", "forward_positive_15_35_count"]}, indent=2))
    for row in forward_band[:20]:
        print(row["name"], "old", round(row["train"]["pnl"], 1), round(row["val"]["pnl"], 1), round(row["old_test"]["pnl"], 1), "FWD", round(row["forward"]["pnl"], 1), round(row["forward"]["trades_per_day"], 1), round(row["forward"]["profit_factor"], 2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
