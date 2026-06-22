#!/usr/bin/env python3
"""Validate one locked cross-market XAUUSD rule without re-searching."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    parser.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trades-out", required=True)
    parser.add_argument("--leader", default="SPY")
    parser.add_argument("--interval", choices=["5m"], default="5m")
    parser.add_argument("--range", default="60d")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--quantile", type=float, default=0.40)
    parser.add_argument("--horizon-bars", type=int, default=9)
    parser.add_argument("--cost-mult", type=float, default=2.0)
    parser.add_argument("--mode", choices=["fade", "follow"], default="fade")
    parser.add_argument("--session", choices=["all", "london", "ny", "overlap", "asia"], default="all")
    parser.add_argument("--exact-only", action="store_true", default=True)
    parser.add_argument("--allow-non-exact", action="store_false", dest="exact_only")
    parser.add_argument("--no-gap-fill", action="store_true", default=True)
    parser.add_argument("--allow-gap-fill", action="store_false", dest="no_gap_fill")
    parser.add_argument("--bootstrap-trials", type=int, default=5000)
    parser.add_argument("--control-trials", type=int, default=1000)
    parser.add_argument("--execution", choices=["overlap", "cooldown", "netted"], default="overlap")
    return parser.parse_args()


def summarize(pnl: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float | int]:
    if len(pnl) == 0:
        return {
            "trades": 0,
            "days": 0.0,
            "trades_per_day": 0.0,
            "pnl": 0.0,
            "avg": 0.0,
            "median": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_trade": 0.0,
            "max_dd": 0.0,
        }
    days = max(1e-9, (times[-1] - times[0]).total_seconds() / 86400.0)
    avg = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    gross_win = float(np.sum(pnl[pnl > 0.0]))
    gross_loss = float(-np.sum(pnl[pnl < 0.0]))
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(pnl)),
        "days": float(days),
        "trades_per_day": float(len(pnl) / days),
        "pnl": float(np.sum(pnl)),
        "avg": avg,
        "median": float(np.median(pnl)),
        "win_rate": float(np.mean(pnl > 0.0)),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0.0 else 0.0,
        "sharpe_trade": float(avg / sd * math.sqrt(len(pnl))) if sd > 0.0 else 0.0,
        "max_dd": float(np.max(peak - eq)) if len(eq) else 0.0,
    }


def split_stats(trades: pd.DataFrame) -> dict[str, object]:
    n = len(trades)
    a = n // 2
    b = (3 * n) // 4
    return {
        "train": summarize(trades["pnl"].values[:a], trades.index[:a]),
        "val": summarize(trades["pnl"].values[a:b], trades.index[a:b]),
        "test": summarize(trades["pnl"].values[b:], trades.index[b:]),
        "all": summarize(trades["pnl"].values, trades.index),
    }


def build_trades(px: pd.DataFrame, args: argparse.Namespace, *, mode: str | None = None, shift: int = 0, execution: str | None = None) -> tuple[pd.DataFrame, float]:
    mode = mode or args.mode
    execution = execution or args.execution
    leader_ret = np.log(px[args.leader]).diff()
    momentum = leader_ret.rolling(args.window).sum().shift(1)
    threshold_sample = momentum.iloc[: max(1, len(momentum) // 2)].dropna().abs().values
    threshold = float(np.quantile(threshold_sample, args.quantile))
    base = np.where(momentum.values > threshold, 1.0, np.where(momentum.values < -threshold, -1.0, 0.0))
    signal = -base if mode == "fade" else base
    if shift:
        signal = np.roll(signal, shift)
    masks = cm.session_masks(px.index)
    future = px["xau"].shift(-args.horizon_bars) - px["xau"]
    spread = px["spread"].fillna(px["spread"].median()).clip(lower=0.0)
    valid = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0) & masks[args.session]
    if execution == "cooldown":
        selected = np.zeros(len(px), dtype=bool)
        next_allowed = 0
        for i, ok in enumerate(valid):
            if ok and i >= next_allowed:
                selected[i] = True
                next_allowed = i + args.horizon_bars
        valid = selected
    trades = pd.DataFrame(
        {
            "entry_time": px.index[valid],
            "exit_time": px.index.to_series().shift(-args.horizon_bars).values[valid],
            "direction": signal[valid].astype(int),
            "leader_close": px[args.leader].values[valid],
            "xau_entry": px["xau"].values[valid],
            "xau_exit": px["xau"].shift(-args.horizon_bars).values[valid],
            "gross": signal[valid] * future.values[valid],
            "spread_cost": spread.values[valid] * args.cost_mult,
            "pnl": signal[valid] * future.values[valid] - spread.values[valid] * args.cost_mult,
        },
        index=px.index[valid],
    )
    trades["equity"] = trades["pnl"].cumsum()
    if execution == "netted":
        # Convert overlapping entry signals into a one-bar marked-to-market
        # strategy. Position is the average active direction across the
        # rule's holding window, clipped to one unit of exposure.
        raw = pd.Series(signal, index=px.index).where(valid, 0.0)
        pos = pd.Series(0.0, index=px.index)
        for lag in range(args.horizon_bars):
            pos = pos.add(raw.shift(lag).fillna(0.0), fill_value=0.0)
        pos = pos.clip(-1.0, 1.0)
        one_bar = px["xau"].diff().shift(-1)
        pos_prev = pos.shift(1).fillna(0.0)
        turnover = (pos - pos_prev).abs()
        cost = px["spread"].fillna(px["spread"].median()).clip(lower=0.0) * args.cost_mult * turnover
        pnl = (pos * one_bar).fillna(0.0) - cost
        net = pd.DataFrame(
            {
                "entry_time": px.index,
                "exit_time": px.index.to_series().shift(-1).values,
                "direction": np.sign(pos).astype(int).values,
                "leader_close": px[args.leader].values,
                "xau_entry": px["xau"].values,
                "xau_exit": px["xau"].shift(-1).values,
                "gross": (pos * one_bar).fillna(0.0).values,
                "spread_cost": cost.values,
                "pnl": pnl.values,
            },
            index=px.index,
        )
        net = net[(pos != 0.0) & np.isfinite(net["pnl"].values)].copy()
        net["equity"] = net["pnl"].cumsum()
        trades = net
    return trades, threshold


def bootstrap_summary(values: np.ndarray, trials: int) -> dict[str, float | int]:
    if len(values) == 0 or trials <= 0:
        return {"trials": 0}
    rng = np.random.default_rng(20260618)
    sums = np.empty(trials)
    means = np.empty(trials)
    for i in range(trials):
        sample = rng.choice(values, size=len(values), replace=True)
        sums[i] = np.sum(sample)
        means[i] = np.mean(sample)
    return {
        "trials": int(trials),
        "pnl_p01": float(np.quantile(sums, 0.01)),
        "pnl_p05": float(np.quantile(sums, 0.05)),
        "pnl_p50": float(np.quantile(sums, 0.50)),
        "pnl_p95": float(np.quantile(sums, 0.95)),
        "avg_p05": float(np.quantile(means, 0.05)),
        "prob_sum_positive": float(np.mean(sums > 0.0)),
        "prob_avg_positive": float(np.mean(means > 0.0)),
    }


def timing_controls(px: pd.DataFrame, args: argparse.Namespace, real_test_pnl: float, real_all_pnl: float) -> dict[str, float | int]:
    if args.control_trials <= 0:
        return {"trials": 0}
    rng = np.random.default_rng(42)
    test_pnls = []
    all_pnls = []
    for i in range(args.control_trials):
        shift = int(rng.integers(100, max(101, len(px) - 100)))
        trades, _ = build_trades(px, args, shift=shift)
        if len(trades) < 100:
            continue
        n = len(trades)
        b = (3 * n) // 4
        test_pnls.append(float(np.sum(trades["pnl"].values[b:])))
        all_pnls.append(float(np.sum(trades["pnl"].values)))
    test = np.array(test_pnls)
    all_ = np.array(all_pnls)
    return {
        "trials": int(len(test)),
        "test_mean": float(np.mean(test)),
        "test_p95": float(np.quantile(test, 0.95)),
        "test_p99": float(np.quantile(test, 0.99)),
        "test_rank_pct": float(np.mean(test < real_test_pnl)),
        "all_mean": float(np.mean(all_)),
        "all_p95": float(np.quantile(all_, 0.95)),
        "all_rank_pct": float(np.mean(all_ < real_all_pnl)),
    }


def main() -> int:
    args = parse_args()
    leaders, errors = cm.load_leaders([args.leader], args.interval, args.range, Path(args.cache_dir))
    xau = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), args.interval)
    px = leaders.join(xau[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    px = px.dropna(subset=["xau", "spread", args.leader])
    if args.exact_only:
        px = px[px["exact_all"] == 1]
    if args.no_gap_fill:
        px = px[px["gap_fill"] == 0]
    if len(px) < 500:
        raise RuntimeError(f"too few aligned rows: {len(px)}")

    trades, threshold = build_trades(px, args)
    follow_trades, _ = build_trades(px, args, mode="follow" if args.mode == "fade" else "fade")
    splits = split_stats(trades)
    reverse_splits = split_stats(follow_trades)

    weekly = []
    for _, group in trades.groupby(pd.Grouper(freq="7D")):
        if len(group) >= 20:
            weekly.append({"start": str(group.index.min()), "end": str(group.index.max()), **summarize(group["pnl"].values, group.index)})
    folds = []
    n = len(trades)
    cuts = [0, n // 4, n // 2, (3 * n) // 4, n]
    for i in range(4):
        part = trades.iloc[cuts[i] : cuts[i + 1]]
        folds.append({"fold": i + 1, "start": str(part.index.min()), "end": str(part.index.max()), **summarize(part["pnl"].values, part.index)})

    report = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "rule": {
            "leader": args.leader,
            "mode": args.mode,
            "window": args.window,
            "quantile": args.quantile,
            "threshold_train_only": threshold,
            "horizon_bars": args.horizon_bars,
            "cost_mult": args.cost_mult,
            "session": args.session,
            "execution": args.execution,
            "exact_only": args.exact_only,
            "no_gap_fill": args.no_gap_fill,
        },
        "data": {
            "rows_aligned": int(len(px)),
            "time_start": str(px.index.min()),
            "time_end": str(px.index.max()),
            "leader_errors": errors,
        },
        "splits": splits,
        "reverse_direction_splits": reverse_splits,
        "folds": folds,
        "weekly_blocks": weekly,
        "bootstrap_all": bootstrap_summary(trades["pnl"].values, args.bootstrap_trials),
        "bootstrap_test": bootstrap_summary(trades["pnl"].values[(3 * len(trades)) // 4 :], args.bootstrap_trials),
        "timing_controls": timing_controls(px, args, float(splits["test"]["pnl"]), float(splits["all"]["pnl"])),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_out, index=False)

    print(json.dumps({"rows_aligned": report["data"]["rows_aligned"], "trades": len(trades), "threshold": threshold}, indent=2))
    print("splits", {k: (round(v["pnl"], 2), round(v["trades_per_day"], 1), round(v["win_rate"], 3)) for k, v in splits.items()})
    print("reverse", {k: (round(v["pnl"], 2), round(v["trades_per_day"], 1), round(v["win_rate"], 3)) for k, v in reverse_splits.items()})
    print("wrote", out)
    print("trades", trades_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
