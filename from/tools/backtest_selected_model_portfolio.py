#!/usr/bin/env python3
"""Backtest selected cross-market XAU models with dollar notional sizing."""

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
import long_history_yahoo_forward_search as lh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--selected", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trades-out", required=True)
    p.add_argument("--capital", type=float, default=5_000_000.0)
    p.add_argument("--cap-gross-exposure", action="store_true")
    p.add_argument("--period", choices=["forward", "old"], default="forward")
    p.add_argument("--forward-xau", default=r"C:\Users\marti\from\data\derived\xauusd_dukascopy_5m_forward_20260522_20260618.csv")
    p.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    p.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    p.add_argument("--interval", default="5m")
    p.add_argument("--range", default="60d")
    return p.parse_args()


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


def load_px(args: argparse.Namespace, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    leaders, errors = cm.load_leaders(symbols, args.interval, args.range, Path(args.cache_dir))
    if args.period == "forward":
        if args.interval == "5m":
            xau = pd.read_csv(args.forward_xau, parse_dates=["time"]).rename(columns={"close": "xau"}).set_index("time").sort_index()
        else:
            xau = lh.aggregate_forward_xau(args.forward_xau, args.interval)
        px = leaders.join(xau[["xau", "spread", "ticks"]], how="inner").sort_index()
    else:
        xau = cm.load_xau(args.master, leaders.index.min().to_pydatetime(), leaders.index.max().to_pydatetime(), args.interval)
        px = leaders.join(xau[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
        px = px[(px["gap_fill"] == 0)]
    return px.dropna(subset=["xau", "spread"]), errors


def trades_for_model(px: pd.DataFrame, model: dict[str, object], allocation: float) -> pd.DataFrame:
    leader = str(model["leader"])
    d = px.dropna(subset=[leader]).copy()
    values = fh.feature_values(d, leader, int(model["window"]), str(model["feature"]), 48)
    signal = np.where(values > float(model["threshold"]), 1.0, np.where(values < -float(model["threshold"]), -1.0, 0.0))
    if str(model["mode"]) == "fade":
        signal = -signal
    horizon = int(model["horizon"])
    future = d["xau"].shift(-horizon) - d["xau"]
    mask = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0) & session_mask(d.index, str(model["session"]))
    spread = d["spread"].fillna(d["spread"].median()).clip(lower=0.0)
    gross_per_oz = signal[mask] * future.values[mask]
    cost_per_oz = spread.values[mask] * float(model["cost"])
    entry = d["xau"].values[mask]
    ounces = allocation / entry
    pnl = (gross_per_oz - cost_per_oz) * ounces
    out = pd.DataFrame(
        {
            "model": str(model["name"]),
            "leader": leader,
            "entry_time": d.index[mask],
            "exit_time": d.index.to_series().shift(-horizon).values[mask],
            "direction": signal[mask].astype(int),
            "xau_entry": entry,
            "xau_exit": d["xau"].shift(-horizon).values[mask],
            "gross_per_oz": gross_per_oz,
            "cost_per_oz": cost_per_oz,
            "ounces": ounces,
            "notional": allocation,
            "pnl": pnl,
        }
    )
    out = out.dropna(subset=["exit_time", "pnl"]).sort_values("exit_time")
    out["pnl_per_dollar"] = (out["gross_per_oz"] - out["cost_per_oz"]) / out["xau_entry"]
    return out


def stats(pnl: np.ndarray, times: pd.Series | pd.DatetimeIndex, capital: float) -> dict[str, float | int]:
    if len(pnl) == 0:
        return {"trades": 0, "pnl": 0.0, "return_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}
    t = pd.DatetimeIndex(times)
    days = max(1e-9, (t.max() - t.min()).total_seconds() / 86400.0)
    win = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    eq = pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return {
        "trades": int(len(pnl)),
        "days": float(days),
        "trades_per_day": float(len(pnl) / days),
        "pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / capital * 100.0),
        "avg_trade": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(win / loss) if loss > 0 else 0.0,
        "max_dd": float((peak - eq).max()) if len(eq) else 0.0,
        "max_dd_pct": float(((peak - eq).max() if len(eq) else 0.0) / capital * 100.0),
    }


def exposure_stats(trades: pd.DataFrame, capital: float) -> dict[str, float]:
    events = []
    for row in trades.itertuples(index=False):
        events.append((pd.Timestamp(row.entry_time).tz_convert("UTC") if pd.Timestamp(row.entry_time).tzinfo else pd.Timestamp(row.entry_time).tz_localize("UTC"), float(row.notional)))
        events.append((pd.Timestamp(row.exit_time).tz_convert("UTC") if pd.Timestamp(row.exit_time).tzinfo else pd.Timestamp(row.exit_time).tz_localize("UTC"), -float(row.notional)))
    events.sort(key=lambda x: (x[0], 0 if x[1] < 0 else 1))
    active = 0.0
    peak = 0.0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return {"peak_gross_notional": peak, "peak_gross_exposure_pct_of_capital": peak / capital * 100.0}


def apply_gross_cap(trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    trades = trades.sort_values(["entry_time", "exit_time"]).reset_index(drop=True).copy()
    events = []
    for i, row in trades.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_ = pd.Timestamp(row["exit_time"])
        entry = entry.tz_convert("UTC") if entry.tzinfo else entry.tz_localize("UTC")
        exit_ = exit_.tz_convert("UTC") if exit_.tzinfo else exit_.tz_localize("UTC")
        events.append((entry, 1, i))
        events.append((exit_, 0, i))
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0.0
    assigned = np.zeros(len(trades), dtype=float)
    for _, event_type, i in events:
        if event_type == 0:
            active -= assigned[i]
            if active < 1e-6:
                active = 0.0
        else:
            desired = float(trades.at[i, "notional"])
            available = max(0.0, capital - active)
            fill = min(desired, available)
            assigned[i] = fill
            active += fill
    trades["desired_notional"] = trades["notional"]
    trades["notional"] = assigned
    trades = trades[trades["notional"] > 0.0].copy()
    trades["ounces"] = trades["notional"] / trades["xau_entry"]
    trades["pnl"] = trades["pnl_per_dollar"] * trades["notional"]
    trades = trades.sort_values("exit_time")
    trades["equity"] = trades["pnl"].cumsum()
    return trades


def main() -> int:
    args = parse_args()
    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    models = selected["models"]
    symbols = sorted({str(m["leader"]) for m in models})
    px, errors = load_px(args, symbols)
    allocation = args.capital / len(models)
    frames = [trades_for_model(px, m, allocation) for m in models]
    trades = pd.concat(frames, ignore_index=True).sort_values("exit_time") if frames else pd.DataFrame()
    if args.cap_gross_exposure and len(trades):
        trades = apply_gross_cap(trades, args.capital)
    trades["equity"] = trades["pnl"].cumsum()
    model_stats = {}
    for name, group in trades.groupby("model"):
        model_stats[name] = stats(group["pnl"].values, group["exit_time"], args.capital)
    pnl = trades["pnl"].values
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "period": args.period,
        "capital": args.capital,
        "model_count": len(models),
        "allocation_per_model": allocation,
        "cap_gross_exposure": bool(args.cap_gross_exposure),
        "symbols": symbols,
        "leader_errors": errors,
        "data": {
            "rows": int(len(px)),
            "start": str(px.index.min()),
            "end": str(px.index.max()),
        },
        "portfolio": stats(pnl, trades["exit_time"] if len(trades) else pd.DatetimeIndex([]), args.capital),
        "exposure": exposure_stats(trades, args.capital) if len(trades) else {},
        "model_stats": model_stats,
        "models": models,
        "sizing_note": "Each model requests equal notional allocation of capital/model_count per signal. If cap_gross_exposure is true, new trades are skipped or down-sized when gross active notional would exceed account capital.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_out, index=False)
    print(json.dumps({"period": args.period, "portfolio": report["portfolio"], "exposure": report["exposure"]}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
