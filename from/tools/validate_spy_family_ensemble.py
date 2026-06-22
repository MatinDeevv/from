#!/usr/bin/env python3
"""Validate a locked SPY-fade rule family on exact/no-gap XAUUSD data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cross_market_edge_search as cm


LOCKED_RULES = [
    {"window": 6, "quantile": 0.40, "horizon": 9},
    {"window": 6, "quantile": 0.45, "horizon": 9},
    {"window": 4, "quantile": 0.40, "horizon": 9},
    {"window": 2, "quantile": 0.40, "horizon": 12},
    {"window": 3, "quantile": 0.40, "horizon": 12},
    {"window": 6, "quantile": 0.45, "horizon": 12},
]

TRAIN_POSITIVE_RULES = [
    {"window": 6, "quantile": 0.40, "horizon": 9},
    {"window": 6, "quantile": 0.45, "horizon": 9},
    {"window": 4, "quantile": 0.40, "horizon": 9},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    p.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    p.add_argument("--out", required=True)
    p.add_argument("--trades-out", required=True)
    p.add_argument("--cost-mult", type=float, default=2.0)
    p.add_argument("--control-trials", type=int, default=1000)
    p.add_argument("--preset", choices=["all6", "train_positive3"], default="all6")
    return p.parse_args()


def stat(pnl: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float | int]:
    if len(pnl) == 0:
        return {"trades": 0, "days": 0.0, "trades_per_day": 0.0, "pnl": 0.0, "avg": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "sharpe_trade": 0.0, "max_dd": 0.0}
    days = max(1e-9, (times[-1] - times[0]).total_seconds() / 86400.0)
    avg = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    win = float(np.sum(pnl[pnl > 0.0]))
    loss = float(-np.sum(pnl[pnl < 0.0]))
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(pnl)),
        "days": float(days),
        "trades_per_day": float(len(pnl) / days),
        "pnl": float(np.sum(pnl)),
        "avg": avg,
        "win_rate": float(np.mean(pnl > 0.0)),
        "profit_factor": float(win / loss) if loss > 0.0 else 0.0,
        "sharpe_trade": float(avg / sd * math.sqrt(len(pnl))) if sd > 0.0 else 0.0,
        "max_dd": float(np.max(peak - eq)) if len(eq) else 0.0,
    }


def split_stats(trades: pd.DataFrame) -> dict[str, object]:
    n = len(trades)
    a = n // 2
    b = 3 * n // 4
    return {
        "train": stat(trades.pnl.values[:a], trades.index[:a]),
        "val": stat(trades.pnl.values[a:b], trades.index[a:b]),
        "test": stat(trades.pnl.values[b:], trades.index[b:]),
        "all": stat(trades.pnl.values, trades.index),
    }


def rule_trades(px: pd.DataFrame, rule: dict[str, float | int], cost_mult: float, shift: int = 0) -> tuple[pd.DataFrame, float]:
    momentum = np.log(px.SPY).diff().rolling(int(rule["window"])).sum().shift(1)
    threshold = float(np.quantile(np.abs(momentum.iloc[: len(px) // 2].dropna().values), float(rule["quantile"])))
    signal = -np.where(momentum.values > threshold, 1.0, np.where(momentum.values < -threshold, -1.0, 0.0))
    if shift:
        signal = np.roll(signal, shift)
    horizon = int(rule["horizon"])
    future = px.xau.shift(-horizon) - px.xau
    valid = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0)
    gross = signal[valid] * future.values[valid]
    cost = px.spread.fillna(px.spread.median()).clip(lower=0.0).values[valid] * cost_mult
    out = pd.DataFrame(
        {
            "rule": f"w{rule['window']}_q{rule['quantile']}_h{horizon}",
            "direction": signal[valid].astype(int),
            "xau_entry": px.xau.values[valid],
            "xau_exit": px.xau.shift(-horizon).values[valid],
            "gross": gross,
            "spread_cost": cost,
            "pnl": gross - cost,
        },
        index=px.index[valid],
    )
    return out.sort_index(), threshold


def selected_rules(preset: str) -> list[dict[str, float | int]]:
    return TRAIN_POSITIVE_RULES if preset == "train_positive3" else LOCKED_RULES


def ensemble_trades(px: pd.DataFrame, cost_mult: float, rules: list[dict[str, float | int]], shift: int = 0) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    parts = []
    meta = []
    weight = 1.0 / len(rules)
    for rule in rules:
        trades, threshold = rule_trades(px, rule, cost_mult, shift)
        trades = trades.copy()
        trades["raw_pnl"] = trades["pnl"]
        trades["pnl"] = trades["pnl"] * weight
        trades["weight"] = weight
        parts.append(trades)
        meta.append({**rule, "threshold_train_only": threshold, "component_trades": int(len(trades))})
    all_trades = pd.concat(parts).sort_index()
    all_trades["equity"] = all_trades.pnl.cumsum()
    return all_trades, meta


def circular_controls(px: pd.DataFrame, cost_mult: float, rules: list[dict[str, float | int]], real_all: float, real_test: float, trials: int) -> dict[str, float | int]:
    if trials <= 0:
        return {"trials": 0}
    rng = np.random.default_rng(20260618)
    all_vals = []
    test_vals = []
    for _ in range(trials):
        shift = int(rng.integers(100, max(101, len(px) - 100)))
        trades, _ = ensemble_trades(px, cost_mult, rules, shift)
        n = len(trades)
        b = 3 * n // 4
        all_vals.append(float(trades.pnl.sum()))
        test_vals.append(float(trades.pnl.values[b:].sum()))
    all_arr = np.array(all_vals)
    test_arr = np.array(test_vals)
    return {
        "trials": int(trials),
        "all_mean": float(all_arr.mean()),
        "all_p95": float(np.quantile(all_arr, 0.95)),
        "all_p99": float(np.quantile(all_arr, 0.99)),
        "all_rank_pct": float(np.mean(all_arr < real_all)),
        "test_mean": float(test_arr.mean()),
        "test_p95": float(np.quantile(test_arr, 0.95)),
        "test_p99": float(np.quantile(test_arr, 0.99)),
        "test_rank_pct": float(np.mean(test_arr < real_test)),
    }


def main() -> int:
    args = parse_args()
    leaders, errors = cm.load_leaders(["SPY"], "5m", "60d", Path(args.cache_dir))
    xau = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), "5m")
    px = leaders.join(xau[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    px = px.dropna(subset=["SPY", "xau", "spread"])
    px = px[(px.exact_all == 1) & (px.gap_fill == 0)].copy()
    if len(px) < 500:
        raise RuntimeError(f"too few rows: {len(px)}")

    rules = selected_rules(args.preset)
    trades, meta = ensemble_trades(px, args.cost_mult, rules)
    splits = split_stats(trades)
    components = []
    for rule in rules:
        component, threshold = rule_trades(px, rule, args.cost_mult)
        components.append({"rule": rule, "threshold_train_only": threshold, **split_stats(component)})

    folds = []
    n = len(trades)
    cuts = [0, n // 4, n // 2, 3 * n // 4, n]
    for i in range(4):
        part = trades.iloc[cuts[i] : cuts[i + 1]]
        folds.append({"fold": i + 1, "start": str(part.index.min()), "end": str(part.index.max()), **stat(part.pnl.values, part.index)})

    report = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "rule_family": f"equal-weight locked SPY fade nearby-parameter family: {args.preset}",
        "rules": meta,
        "data": {"rows_aligned": int(len(px)), "time_start": str(px.index.min()), "time_end": str(px.index.max()), "leader_errors": errors},
        "cost_mult": args.cost_mult,
        "splits": splits,
        "folds": folds,
        "components": components,
        "circular_controls": circular_controls(px, args.cost_mult, rules, float(splits["all"]["pnl"]), float(splits["test"]["pnl"]), args.control_trials),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_out)

    print(json.dumps({"rows_aligned": len(px), "weighted_trades": len(trades), "rules": len(rules), "preset": args.preset}, indent=2))
    print("splits", {k: (round(v["pnl"], 2), round(v["trades_per_day"], 1), round(v["win_rate"], 3)) for k, v in splits.items()})
    print("folds", [(f["fold"], round(f["pnl"], 2), round(f["trades_per_day"], 1), round(f["win_rate"], 3)) for f in folds])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
