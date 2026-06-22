#!/usr/bin/env python3
"""Validate the locked SPY->XAU rule on a forward XAU CSV."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cross_market_edge_search as cm


LOCKED_THRESHOLD = 0.0005740430387565


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xau-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trades-out", required=True)
    p.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    p.add_argument("--leader", default="SPY")
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--threshold", type=float, default=LOCKED_THRESHOLD)
    p.add_argument("--horizon-bars", type=int, default=9)
    p.add_argument("--cost-mult", type=float, default=2.0)
    p.add_argument("--mode", choices=["fade", "follow"], default="fade")
    p.add_argument("--feature", choices=["raw", "volz", "divergence"], default="raw")
    p.add_argument("--vol-window", type=int, default=48)
    p.add_argument("--session", choices=["all", "london", "ny", "overlap", "asia"], default="all")
    p.add_argument("--execution", choices=["overlap", "cooldown", "netted"], default="overlap")
    p.add_argument("--control-trials", type=int, default=1000)
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


def load_joined(args: argparse.Namespace) -> pd.DataFrame:
    xau = pd.read_csv(args.xau_csv)
    xau["time"] = pd.to_datetime(xau["time"], utc=True)
    xau = xau.rename(columns={"close": "xau"}).set_index("time").sort_index()
    leaders, errors = cm.load_leaders([args.leader], "5m", "60d", Path(args.cache_dir))
    px = leaders.join(xau[["xau", "spread", "ticks"]], how="inner").dropna().sort_index()
    px.attrs["leader_errors"] = errors
    return px


def feature_values(px: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    leader_ret = np.log(px[args.leader]).diff()
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
        return ((momentum / leader_vol.replace(0.0, np.nan)) - (xau_momentum / xau_vol.replace(0.0, np.nan))).values
    raise ValueError(args.feature)


def trades_for(px: pd.DataFrame, args: argparse.Namespace, shift: int = 0, reverse: bool = False) -> pd.DataFrame:
    values = feature_values(px, args)
    signal = np.where(values > args.threshold, 1.0, np.where(values < -args.threshold, -1.0, 0.0))
    if args.mode == "fade":
        signal = -signal
    if reverse:
        signal = -signal
    if shift:
        signal = np.roll(signal, shift)
    future = px["xau"].shift(-args.horizon_bars) - px["xau"]
    if args.session == "london":
        session_mask = np.array([(7 <= t.hour < 16) for t in px.index])
    elif args.session == "ny":
        session_mask = np.array([(13 <= t.hour < 21) for t in px.index])
    elif args.session == "overlap":
        session_mask = np.array([(13 <= t.hour < 16) for t in px.index])
    elif args.session == "asia":
        session_mask = np.array([(0 <= t.hour < 7) for t in px.index])
    else:
        session_mask = np.ones(len(px), dtype=bool)
    valid = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0) & session_mask
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
        gross = (pos * one_bar).fillna(0.0)
        cost = px["spread"].fillna(px["spread"].median()).clip(lower=0.0) * args.cost_mult * turnover
        pnl = gross - cost
        out = pd.DataFrame({"direction": np.sign(pos).astype(int), "xau_entry": px["xau"], "xau_exit": px["xau"].shift(-1), "gross": gross, "spread_cost": cost, "pnl": pnl}, index=px.index)
        out = out[(pos != 0.0) & np.isfinite(out["pnl"])].copy()
    else:
        gross = signal[valid] * future.values[valid]
        cost = px["spread"].fillna(px["spread"].median()).clip(lower=0.0).values[valid] * args.cost_mult
        out = pd.DataFrame({"direction": signal[valid].astype(int), "xau_entry": px["xau"].values[valid], "xau_exit": px["xau"].shift(-args.horizon_bars).values[valid], "gross": gross, "spread_cost": cost, "pnl": gross - cost}, index=px.index[valid])
    out["equity"] = out["pnl"].cumsum()
    return out


def controls(px: pd.DataFrame, args: argparse.Namespace, real: float) -> dict[str, float | int]:
    if args.control_trials <= 0:
        return {"trials": 0}
    rng = np.random.default_rng(20260618)
    vals = []
    for _ in range(args.control_trials):
        shift = int(rng.integers(100, max(101, len(px) - 100)))
        vals.append(float(trades_for(px, args, shift=shift)["pnl"].sum()))
    arr = np.array(vals)
    return {
        "trials": int(len(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "rank_pct": float(np.mean(arr < real)),
    }


def main() -> int:
    args = parse_args()
    px = load_joined(args)
    trades = trades_for(px, args)
    reverse = trades_for(px, args, reverse=True)
    summary = stat(trades["pnl"].values, trades.index)
    reverse_summary = stat(reverse["pnl"].values, reverse.index)
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rule": {
            "leader": args.leader,
            "window": args.window,
            "threshold_locked": args.threshold,
            "horizon_bars": args.horizon_bars,
            "cost_mult": args.cost_mult,
            "mode": args.mode,
            "feature": args.feature,
            "vol_window": args.vol_window,
            "session": args.session,
            "execution": args.execution,
        },
        "data": {
            "xau_csv": args.xau_csv,
            "rows_aligned": int(len(px)),
            "time_start": str(px.index.min()),
            "time_end": str(px.index.max()),
            "leader_errors": px.attrs.get("leader_errors", {}),
        },
        "summary": summary,
        "reverse_summary": reverse_summary,
        "circular_controls": controls(px, args, float(summary["pnl"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_out)
    print(json.dumps({"rows_aligned": len(px), "trades": len(trades), "pnl": summary["pnl"], "tpd": summary["trades_per_day"], "reverse_pnl": reverse_summary["pnl"]}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
