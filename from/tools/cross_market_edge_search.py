#!/usr/bin/env python3
"""Cross-market lead/lag search for timestamp-restored XAUUSD bars.

The script intentionally treats the XAUUSD master parquet as the execution
market and public Yahoo bars as delayed leader-market proxies. Every signal is
shifted by one completed leader bar before it can trade XAUUSD.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = [
    "SPY",
    "TLT",
    "GC=F",
    "GLD",
    "IAU",
    "SI=F",
    "DX-Y.NYB",
    "EURUSD=X",
    "ZN=F",
    "ZB=F",
    "CL=F",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=r"C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=r"C:\Users\marti\from\data\cache\yahoo")
    parser.add_argument("--interval", choices=["5m", "15m", "60m"], default="5m")
    parser.add_argument("--range", default=None, help="Yahoo chart range. Defaults: 60d for 5m, 60d for 15m, 730d for 60m.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--no-gap-fill", action="store_true")
    parser.add_argument("--leader-ffill-bars", type=int, default=0)
    parser.add_argument("--min-trades-day", type=float, default=15.0)
    parser.add_argument("--max-trades-day", type=float, default=35.0)
    parser.add_argument("--control-trials", type=int, default=300)
    parser.add_argument("--top-controls", type=int, default=8)
    return parser.parse_args()


def interval_ms(interval: str) -> int:
    if interval == "5m":
        return 5 * 60_000
    if interval == "15m":
        return 15 * 60_000
    if interval == "60m":
        return 60 * 60_000
    raise ValueError(interval)


def default_range(interval: str) -> str:
    return "730d" if interval == "60m" else "60d"


def yahoo_cache_path(cache_dir: Path, symbol: str, interval: str, range_: str) -> Path:
    safe = symbol.replace("=", "_").replace("^", "_").replace(".", "_").replace("-", "_")
    return cache_dir / f"{safe}_{interval}_{range_}.csv"


def fetch_yahoo(symbol: str, interval: str, range_: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = yahoo_cache_path(cache_dir, symbol, interval, range_)
    if cache.exists():
        df = pd.read_csv(cache)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.dropna().drop_duplicates("time").set_index("time").sort_index()

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.parse.quote(symbol, safe="")
        + f"?range={range_}&interval={interval}&includePrePost=true&events=history"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = data["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(result.get("timestamp", []), unit="s", utc=True),
            symbol: quote.get("close") or [],
        }
    ).dropna()
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df.to_csv(cache)
    return df


def load_leaders(symbols: list[str], interval: str, range_: str, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            frames.append(fetch_yahoo(symbol, interval, range_, cache_dir))
        except Exception as exc:  # network data is opportunistic; record failures.
            errors[symbol] = repr(exc)
        time.sleep(0.1)
    if not frames:
        raise RuntimeError("no leader data loaded")
    return pd.concat(frames, axis=1, sort=True), errors


def load_xau(master: str, start: dt.datetime, end: dt.datetime, interval: str) -> pd.DataFrame:
    bucket = interval_ms(interval)
    con = duckdb.connect()
    query = f"""
    WITH src AS (
      SELECT
        floor(epoch_ms(timestamp_utc) / {bucket}) * {bucket} AS bucket_ms,
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
    df = con.execute(query, [master, start, end]).fetchdf()
    df["time"] = pd.to_datetime(df["bucket_ms"], unit="ms", utc=True)
    return df.set_index("time").sort_index()


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    eq = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return float(np.max(peak - eq))


def stats(pnl: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float | int]:
    if len(pnl) == 0:
        return {"trades": 0, "days": 0.0, "trades_per_day": 0.0, "pnl": 0.0, "avg": 0.0, "win_rate": 0.0, "sharpe_trade": 0.0, "max_dd": 0.0}
    days = max(1e-9, (times[-1] - times[0]).total_seconds() / 86400.0)
    avg = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    return {
        "trades": int(len(pnl)),
        "days": float(days),
        "trades_per_day": float(len(pnl) / days),
        "pnl": float(np.sum(pnl)),
        "avg": avg,
        "win_rate": float(np.mean(pnl > 0.0)),
        "sharpe_trade": float(avg / sd * math.sqrt(len(pnl))) if sd > 0.0 else 0.0,
        "max_dd": max_drawdown(pnl),
    }


def evaluate(px: pd.DataFrame, signal: np.ndarray, horizon: int, cost_mult: float, mask: np.ndarray) -> dict[str, object] | None:
    future = px["xau"].shift(-horizon) - px["xau"]
    spread = px["spread"].fillna(px["spread"].median()).clip(lower=0.0)
    valid = np.isfinite(signal) & np.isfinite(future.values) & (signal != 0.0) & mask
    if int(np.sum(valid)) < 150:
        return None
    pnl = signal[valid] * future.values[valid] - spread.values[valid] * cost_mult
    times = px.index[valid]
    n = len(pnl)
    train_end = int(n * 0.50)
    val_end = int(n * 0.75)
    if train_end < 60 or val_end - train_end < 30 or n - val_end < 30:
        return None
    return {
        "train": stats(pnl[:train_end], times[:train_end]),
        "val": stats(pnl[train_end:val_end], times[train_end:val_end]),
        "test": stats(pnl[val_end:], times[val_end:]),
        "all": stats(pnl, times),
    }


def session_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(index), dtype=bool),
        "london": np.array([(7 <= t.hour < 16) for t in index]),
        "ny": np.array([(13 <= t.hour < 21) for t in index]),
        "overlap": np.array([(13 <= t.hour < 16) for t in index]),
        "asia": np.array([(0 <= t.hour < 7) for t in index]),
    }


def candidate_search(px: pd.DataFrame, leaders: list[str], min_tpd: float, max_tpd: float, interval: str) -> list[dict[str, object]]:
    if interval == "60m":
        horizons = [1, 2, 3, 4, 6]
        min_tpd = min(min_tpd, 2.0)
        max_tpd = max(max_tpd, 12.0)
    elif interval == "15m":
        horizons = [1, 2, 3, 4, 6, 8]
    else:
        horizons = [3, 6, 9, 12]
    windows = [1, 2, 3, 4, 6, 9, 12]
    quantiles = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    costs = [1.0, 1.5, 2.0]
    rows: list[dict[str, object]] = []

    for leader in leaders:
        if leader not in px.columns:
            continue
        leader_px = px.dropna(subset=[leader]).copy()
        if len(leader_px) < 500:
            continue
        leader_masks = session_masks(leader_px.index)
        leader_days = max(1e-9, (leader_px.index[-1] - leader_px.index[0]).total_seconds() / 86400.0)
        leader_train_row_end = max(1, len(leader_px) // 2)
        series = leader_px[leader]
        leader_ret = np.log(series).diff()
        for window in windows:
            momentum = leader_ret.rolling(window).sum().shift(1).values
            train_momentum = momentum[:leader_train_row_end]
            finite = np.abs(train_momentum[np.isfinite(train_momentum)])
            if len(finite) < 100:
                continue
            for q in quantiles:
                threshold = float(np.quantile(finite, q))
                base = np.where(momentum > threshold, 1.0, np.where(momentum < -threshold, -1.0, 0.0))
                for mode in ["follow", "fade"]:
                    signal = base if mode == "follow" else -base
                    for session, session_mask in leader_masks.items():
                        mask = (signal != 0.0) & session_mask
                        rough_tpd = float(np.sum(mask) / leader_days)
                        if rough_tpd < min_tpd or rough_tpd > max_tpd:
                            continue
                        for horizon in horizons:
                            for cost in costs:
                                result = evaluate(leader_px, signal, horizon, cost, mask)
                                if result is None:
                                    continue
                                rows.append(
                                    {
                                        "name": f"{leader}_{mode}_lag1_w{window}_q{q}_{session}",
                                        "leader": leader,
                                        "mode": mode,
                                        "window": window,
                                        "quantile": q,
                                        "threshold": threshold,
                                        "session": session,
                                        "horizon_bars": horizon,
                                        "cost_mult": cost,
                                        **result,
                                    }
                                )
    return rows


def qualifies(row: dict[str, object], min_tpd: float, max_tpd: float, require_test: bool = True) -> bool:
    for split in ["train", "val", "test"] if require_test else ["train", "val"]:
        stat = row[split]
        assert isinstance(stat, dict)
        if float(stat["pnl"]) <= 0.0:
            return False
        if not (min_tpd <= float(stat["trades_per_day"]) <= max_tpd):
            return False
    return True


def control_test(px: pd.DataFrame, row: dict[str, object], trials: int) -> dict[str, float | int]:
    leader = str(row["leader"])
    window = int(row["window"])
    q = float(row["quantile"])
    horizon = int(row["horizon_bars"])
    cost = float(row["cost_mult"])
    mode = str(row["mode"])
    session = str(row["session"])

    leader_px = px.dropna(subset=[leader]).copy()
    leader_ret = np.log(leader_px[leader]).diff()
    momentum = leader_ret.rolling(window).sum().shift(1)
    train_row_end = max(1, len(leader_px) // 2)
    finite = np.abs(momentum.iloc[:train_row_end].dropna().values)
    threshold = float(np.quantile(finite, q))
    base = np.where(momentum.values > threshold, 1.0, np.where(momentum.values < -threshold, -1.0, 0.0))
    signal = base if mode == "follow" else -base
    mask = (signal != 0.0) & session_masks(leader_px.index)[session]
    real = evaluate(leader_px, signal, horizon, cost, mask)
    if real is None:
        return {"trials": 0}

    future = leader_px["xau"].shift(-horizon) - leader_px["xau"]
    spread = leader_px["spread"].fillna(leader_px["spread"].median()).clip(lower=0.0)
    rng = np.random.default_rng(123)
    test_values = []
    all_values = []
    for i in range(trials):
        shuffled = signal.copy()
        if i < trials // 2:
            shift = int(rng.integers(100, max(101, len(shuffled) - 100)))
            shuffled = np.roll(shuffled, shift)
        else:
            rng.shuffle(shuffled)
        valid = np.isfinite(shuffled) & np.isfinite(future.values) & (shuffled != 0.0) & session_masks(leader_px.index)[session]
        pnl = shuffled[valid] * future.values[valid] - spread.values[valid] * cost
        if len(pnl) < 100:
            continue
        val_end = int(len(pnl) * 0.75)
        test_values.append(float(np.sum(pnl[val_end:])))
        all_values.append(float(np.sum(pnl)))

    test_arr = np.array(test_values)
    all_arr = np.array(all_values)
    real_test = float(real["test"]["pnl"])
    real_all = float(real["all"]["pnl"])
    return {
        "trials": int(len(test_arr)),
        "real_test_pnl": real_test,
        "control_test_mean": float(np.mean(test_arr)) if len(test_arr) else 0.0,
        "control_test_p95": float(np.quantile(test_arr, 0.95)) if len(test_arr) else 0.0,
        "control_test_p99": float(np.quantile(test_arr, 0.99)) if len(test_arr) else 0.0,
        "control_test_rank_pct": float(np.mean(test_arr < real_test)) if len(test_arr) else 0.0,
        "real_all_pnl": real_all,
        "control_all_mean": float(np.mean(all_arr)) if len(all_arr) else 0.0,
        "control_all_p95": float(np.quantile(all_arr, 0.95)) if len(all_arr) else 0.0,
        "control_all_rank_pct": float(np.mean(all_arr < real_all)) if len(all_arr) else 0.0,
    }


def main() -> int:
    args = parse_args()
    range_ = args.range or default_range(args.interval)
    leaders_raw, errors = load_leaders(args.symbols, args.interval, range_, Path(args.cache_dir))
    xau = load_xau(args.master, leaders_raw.index.min().to_pydatetime(), leaders_raw.index.max().to_pydatetime(), args.interval)
    px = leaders_raw.join(xau[["xau", "spread", "ticks", "exact_all", "gap_fill"]], how="inner").sort_index()
    if args.leader_ffill_bars > 0:
        px[leaders_raw.columns] = px[leaders_raw.columns].ffill(limit=args.leader_ffill_bars)
    px = px.dropna(subset=["xau", "spread"])
    if args.exact_only:
        px = px[px["exact_all"] == 1]
    if args.no_gap_fill:
        px = px[px["gap_fill"] == 0]
    leaders = [c for c in leaders_raw.columns if c in px.columns]
    px = px.dropna(subset=leaders, how="all")
    if len(px) < 500:
        raise RuntimeError(f"too few aligned rows: {len(px)}")

    rows = candidate_search(px, leaders, args.min_trades_day, args.max_trades_day, args.interval)
    if args.interval == "60m":
        q_min, q_max = 2.0, 12.0
    else:
        q_min, q_max = args.min_trades_day, args.max_trades_day
    qualified = [r for r in rows if qualifies(r, q_min, q_max, require_test=True)]
    qualified.sort(key=lambda r: (float(r["test"]["pnl"]), float(r["val"]["pnl"])), reverse=True)

    controls = []
    for row in qualified[: args.top_controls]:
        c = control_test(px, row, args.control_trials)
        controls.append({"name": row["name"], "horizon_bars": row["horizon_bars"], "cost_mult": row["cost_mult"], **c})

    report = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "master": args.master,
        "interval": args.interval,
        "range": range_,
        "exact_only": bool(args.exact_only),
        "no_gap_fill": bool(args.no_gap_fill),
        "leader_ffill_bars": int(args.leader_ffill_bars),
        "symbols_loaded": leaders,
        "leader_errors": errors,
        "rows_aligned": int(len(px)),
        "time_start": str(px.index.min()),
        "time_end": str(px.index.max()),
        "candidate_count": len(rows),
        "qualified_count": len(qualified),
        "promotion_rule": "per-leader aligned rows; threshold quantiles fit on first-half train rows only; leader signal lagged one completed bar; train/val/test pnl positive; frequency inside requested band; actual spread cost included",
        "top": qualified[:500],
        "controls": controls,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["interval", "rows_aligned", "time_start", "time_end", "candidate_count", "qualified_count"]}, indent=2))
    for row in qualified[:15]:
        print(
            row["name"],
            "h",
            row["horizon_bars"],
            "cost",
            row["cost_mult"],
            "tr",
            round(float(row["train"]["pnl"]), 2),
            round(float(row["train"]["trades_per_day"]), 1),
            "val",
            round(float(row["val"]["pnl"]), 2),
            round(float(row["val"]["trades_per_day"]), 1),
            "test",
            round(float(row["test"]["pnl"]), 2),
            round(float(row["test"]["trades_per_day"]), 1),
        )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
