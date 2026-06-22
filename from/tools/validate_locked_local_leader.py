#!/usr/bin/env python3
"""Validate a locked local leader-bar rule against restored XAUUSD master bars."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


TF_MS = 5 * 60_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    p.add_argument("--leader-csv", required=True)
    p.add_argument("--xau-csv", default=None)
    p.add_argument("--leader-name", default="EURUSD")
    p.add_argument("--out", required=True)
    p.add_argument("--trades-out", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--horizon-bars", type=int, required=True)
    p.add_argument("--cost-mult", type=float, default=2.0)
    p.add_argument("--mode", choices=["fade", "follow"], default="follow")
    p.add_argument("--feature", choices=["raw", "volz", "divergence"], default="divergence")
    p.add_argument("--vol-window", type=int, default=48)
    p.add_argument("--session", choices=["all", "london", "ny", "overlap", "asia"], default="london")
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--no-gap-fill", action="store_true", default=True)
    p.add_argument("--allow-gap-fill", action="store_false", dest="no_gap_fill")
    p.add_argument("--execution", choices=["overlap", "cooldown", "netted"], default="overlap")
    p.add_argument("--control-trials", type=int, default=1000)
    return p.parse_args()


def load_xau(master: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
    WITH src AS (
      SELECT
        floor(epoch_ms(timestamp_utc) / {TF_MS}) * {TF_MS} AS bucket_ms,
        epoch_ms(timestamp_utc) AS t_ms,
        mid_close,
        spread_mean,
        tick_count,
        timestamp_dukascopy_exact_verified_all AS exact_all,
        external_gap_fill AS gap_fill
      FROM read_parquet(?)
      WHERE timestamp_utc >= ?
        AND timestamp_utc <= ?
        AND NOT timestamp_mapping_low_confidence
        AND mid_close IS NOT NULL
    ), agg AS (
      SELECT
        bucket_ms,
        arg_max(mid_close, t_ms) AS xau,
        avg(spread_mean) AS spread,
        sum(tick_count) AS ticks,
        min(CASE WHEN exact_all THEN 1 ELSE 0 END) AS exact_all,
        max(CASE WHEN gap_fill THEN 1 ELSE 0 END) AS gap_fill
      FROM src
      GROUP BY bucket_ms
    )
    SELECT * FROM agg ORDER BY bucket_ms
    """
    df = con.execute(query, [master, start.to_pydatetime(), end.to_pydatetime()]).fetchdf()
    df["time"] = pd.to_datetime(df["bucket_ms"], unit="ms", utc=True)
    return df.set_index("time").sort_index()


def load_joined(args: argparse.Namespace) -> pd.DataFrame:
    leader = pd.read_csv(args.leader_csv, parse_dates=["time"]).set_index("time").sort_index()
    if leader.index.tz is None:
        leader.index = leader.index.tz_localize("UTC")
    else:
        leader.index = leader.index.tz_convert("UTC")
    leader = leader.rename(columns={"close": args.leader_name})
    if args.xau_csv:
        xau = pd.read_csv(args.xau_csv, parse_dates=["time"]).rename(columns={"close": "xau"}).set_index("time").sort_index()
        px = leader[[args.leader_name]].join(xau[["xau", "spread", "ticks"]], how="inner").sort_index()
        px["exact_all"] = 1
        px["gap_fill"] = 0
    else:
        xau = load_xau(args.master, leader.index.min(), leader.index.max())
        px = leader[[args.leader_name]].join(xau[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    px = px.dropna(subset=[args.leader_name, "xau", "spread"])
    if args.exact_only:
        px = px[px["exact_all"] == 1]
    if args.no_gap_fill:
        px = px[px["gap_fill"] == 0]
    return px


def feature_values(px: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    leader_ret = np.log(px[args.leader_name]).diff()
    momentum = leader_ret.rolling(args.window).sum().shift(1)
    if args.feature == "raw":
        return momentum.values
    if args.feature == "volz":
        vol = leader_ret.rolling(args.vol_window).std().shift(1) * math.sqrt(args.window)
        return (momentum / vol.replace(0.0, np.nan)).values
    if args.feature == "divergence":
        xau_ret = np.log(px["xau"]).diff()
        xau_momentum = xau_ret.rolling(args.window).sum().shift(1)
        leader_vol = leader_ret.rolling(args.vol_window).std().shift(1) * math.sqrt(args.window)
        xau_vol = xau_ret.rolling(args.vol_window).std().shift(1) * math.sqrt(args.window)
        leader_z = momentum / leader_vol.replace(0.0, np.nan)
        xau_z = xau_momentum / xau_vol.replace(0.0, np.nan)
        return (leader_z - xau_z).values
    raise ValueError(args.feature)


def session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    if session == "london":
        return np.array([(7 <= t.hour < 16) for t in index])
    if session == "ny":
        return np.array([(13 <= t.hour < 21) for t in index])
    if session == "overlap":
        return np.array([(13 <= t.hour < 16) for t in index])
    if session == "asia":
        return np.array([(0 <= t.hour < 7) for t in index])
    return np.ones(len(index), dtype=bool)


def summarize(pnl: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float | int]:
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


def build_trades(px: pd.DataFrame, args: argparse.Namespace, *, shift: int = 0, reverse: bool = False) -> pd.DataFrame:
    values = feature_values(px, args)
    signal = np.where(values > args.threshold, 1.0, np.where(values < -args.threshold, -1.0, 0.0))
    if args.mode == "fade":
        signal = -signal
    if reverse:
        signal = -signal
    if shift:
        signal = np.roll(signal, shift)
    future = px["xau"].shift(-args.horizon_bars) - px["xau"]
    valid = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0) & session_mask(px.index, args.session)
    if args.execution == "cooldown":
        selected = np.zeros(len(px), dtype=bool)
        next_allowed = 0
        for i, ok in enumerate(valid):
            if ok and i >= next_allowed:
                selected[i] = True
                next_allowed = i + args.horizon_bars
        valid = selected
    if args.execution == "netted":
        raw = pd.Series(signal, index=px.index).where(valid, 0.0)
        pos = pd.Series(0.0, index=px.index)
        for lag in range(args.horizon_bars):
            pos = pos.add(raw.shift(lag).fillna(0.0), fill_value=0.0)
        pos = pos.clip(-1.0, 1.0)
        one_bar = px["xau"].diff().shift(-1)
        turnover = (pos - pos.shift(1).fillna(0.0)).abs()
        cost = px["spread"].fillna(px["spread"].median()).clip(lower=0.0) * args.cost_mult * turnover
        gross = (pos * one_bar).fillna(0.0)
        out = pd.DataFrame({"direction": np.sign(pos).astype(int), "xau_entry": px["xau"], "xau_exit": px["xau"].shift(-1), "gross": gross, "spread_cost": cost, "pnl": gross - cost}, index=px.index)
        out = out[(pos != 0.0) & np.isfinite(out["pnl"])].copy()
    else:
        spread = px["spread"].fillna(px["spread"].median()).clip(lower=0.0).values[valid]
        gross = signal[valid] * future.values[valid]
        out = pd.DataFrame(
            {
                "direction": signal[valid].astype(int),
                "leader_close": px[args.leader_name].values[valid],
                "xau_entry": px["xau"].values[valid],
                "xau_exit": px["xau"].shift(-args.horizon_bars).values[valid],
                "gross": gross,
                "spread_cost": spread * args.cost_mult,
                "pnl": gross - spread * args.cost_mult,
            },
            index=px.index[valid],
        )
    out["equity"] = out["pnl"].cumsum()
    return out


def split_stats(trades: pd.DataFrame) -> dict[str, object]:
    n = len(trades)
    cuts = [0, n // 2, (3 * n) // 4, n]
    return {
        "train": summarize(trades["pnl"].values[cuts[0] : cuts[1]], trades.index[cuts[0] : cuts[1]]),
        "val": summarize(trades["pnl"].values[cuts[1] : cuts[2]], trades.index[cuts[1] : cuts[2]]),
        "test": summarize(trades["pnl"].values[cuts[2] : cuts[3]], trades.index[cuts[2] : cuts[3]]),
        "all": summarize(trades["pnl"].values, trades.index),
    }


def fold_stats(trades: pd.DataFrame, folds: int = 8) -> list[dict[str, object]]:
    out = []
    n = len(trades)
    for i in range(folds):
        a = i * n // folds
        b = (i + 1) * n // folds
        part = trades.iloc[a:b]
        out.append({"fold": i + 1, "start": str(part.index.min()), "end": str(part.index.max()), **summarize(part["pnl"].values, part.index)})
    return out


def controls(px: pd.DataFrame, args: argparse.Namespace, real_all: float, real_test: float) -> dict[str, float | int]:
    if args.control_trials <= 0:
        return {"trials": 0}
    rng = np.random.default_rng(20260618)
    all_vals = []
    test_vals = []
    for _ in range(args.control_trials):
        shift = int(rng.integers(100, max(101, len(px) - 100)))
        trades = build_trades(px, args, shift=shift)
        if len(trades) < 100:
            continue
        n = len(trades)
        all_vals.append(float(trades["pnl"].sum()))
        test_vals.append(float(trades["pnl"].values[(3 * n) // 4 :].sum()))
    all_arr = np.array(all_vals)
    test_arr = np.array(test_vals)
    return {
        "trials": int(len(all_arr)),
        "all_mean": float(all_arr.mean()) if len(all_arr) else 0.0,
        "all_p95": float(np.quantile(all_arr, 0.95)) if len(all_arr) else 0.0,
        "all_p99": float(np.quantile(all_arr, 0.99)) if len(all_arr) else 0.0,
        "all_rank_pct": float(np.mean(all_arr < real_all)) if len(all_arr) else 0.0,
        "test_mean": float(test_arr.mean()) if len(test_arr) else 0.0,
        "test_p95": float(np.quantile(test_arr, 0.95)) if len(test_arr) else 0.0,
        "test_p99": float(np.quantile(test_arr, 0.99)) if len(test_arr) else 0.0,
        "test_rank_pct": float(np.mean(test_arr < real_test)) if len(test_arr) else 0.0,
    }


def main() -> int:
    args = parse_args()
    px = load_joined(args)
    if len(px) < 500:
        raise RuntimeError(f"too few aligned rows: {len(px)}")
    trades = build_trades(px, args)
    reverse = build_trades(px, args, reverse=True)
    splits = split_stats(trades)
    reverse_splits = split_stats(reverse)
    folds = fold_stats(trades)
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rule": {
            "leader_csv": args.leader_csv,
            "xau_csv": args.xau_csv,
            "leader_name": args.leader_name,
            "window": args.window,
            "threshold_locked": args.threshold,
            "horizon_bars": args.horizon_bars,
            "cost_mult": args.cost_mult,
            "mode": args.mode,
            "feature": args.feature,
            "vol_window": args.vol_window,
            "session": args.session,
            "execution": args.execution,
            "exact_only": args.exact_only,
            "no_gap_fill": args.no_gap_fill,
        },
        "data": {
            "rows_aligned": int(len(px)),
            "time_start": str(px.index.min()),
            "time_end": str(px.index.max()),
        },
        "splits": splits,
        "reverse_splits": reverse_splits,
        "folds": folds,
        "circular_controls": controls(px, args, float(splits["all"]["pnl"]), float(splits["test"]["pnl"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_out)
    print(json.dumps({"rows_aligned": len(px), "trades": len(trades), "all": splits["all"], "test": splits["test"], "reverse_all": reverse_splits["all"]}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
