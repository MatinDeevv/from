#!/usr/bin/env python3
"""Walk-forward test of XAUUSD strategies derived from local ICT/SMC indicators."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import strict_screening_pipeline as ss


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DERIVED = ROOT / "data" / "derived"
INDICATORS = ROOT / "indicators"
XAU_PARQUET = DERIVED / "duka_XAUUSD_5m_20230101_20260522.parquet"
XAU_HOLDOUT_CSV = DERIVED / "xauusd_dukascopy_5m_forward_20260522_20260618.csv"

FOLDS = [
    (1, "2023-07-01", "2023-12-31"),
    (2, "2024-01-01", "2024-06-30"),
    (3, "2024-07-01", "2024-12-31"),
    (4, "2025-01-01", "2025-06-30"),
    (5, "2025-07-01", "2025-12-31"),
    (6, "2026-01-01", "2026-05-21"),
]

HOLDS = [6, 12, 18, 24, 36, 48]
MODES = ["follow", "fade"]
SESSION_FILTERS = ["all", "london", "nyam", "nypm", "killzones"]


def load_xau() -> pd.DataFrame:
    con = duckdb.connect()
    path = str(XAU_PARQUET).replace("'", "''")
    df = con.execute(
        f"""
        select time, open, high, low, close, spread, ticks
        from read_parquet('{path}')
        order by time
        """
    ).fetchdf()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=["close"]).drop_duplicates("time").set_index("time").sort_index()


def load_holdout() -> pd.DataFrame:
    df = pd.read_csv(XAU_HOLDOUT_CSV, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=["close"]).drop_duplicates("time").set_index("time").sort_index()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def confirmed_pivots(high: pd.Series, low: pd.Series, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    ph = np.full(len(h), np.nan)
    pl = np.full(len(l), np.nan)
    for i in range(left, len(h) - right):
        if h[i] >= np.nanmax(h[i - left : i + right + 1]):
            ph[i + right] = h[i]
        if l[i] <= np.nanmin(l[i - left : i + right + 1]):
            pl[i + right] = l[i]
    return pd.Series(ph, index=high.index), pd.Series(pl, index=low.index)


def cross_above(s: pd.Series, level: pd.Series) -> pd.Series:
    return (s > level) & (s.shift(1) <= level.shift(1))


def cross_below(s: pd.Series, level: pd.Series) -> pd.Series:
    return (s < level) & (s.shift(1) >= level.shift(1))


def ny_minutes(index: pd.DatetimeIndex) -> pd.Series:
    ny = index.tz_convert("America/New_York")
    return pd.Series(ny.hour * 60 + ny.minute, index=index)


def session_mask(index: pd.DatetimeIndex, session: str) -> pd.Series:
    m = ny_minutes(index)
    if session == "all":
        return pd.Series(True, index=index)
    if session == "london":
        return (m >= 2 * 60) & (m < 5 * 60)
    if session == "nyam":
        return (m >= 9 * 60 + 30) & (m < 11 * 60)
    if session == "nypm":
        return (m >= 13 * 60 + 30) & (m < 16 * 60)
    if session == "killzones":
        asia = (m >= 20 * 60) | (m < 0)
        london = (m >= 2 * 60) & (m < 5 * 60)
        nyam = (m >= 9 * 60 + 30) & (m < 11 * 60)
        nypm = (m >= 13 * 60 + 30) & (m < 16 * 60)
        return asia | london | nyam | nypm
    raise ValueError(session)


def session_levels(df: pd.DataFrame, start_minute: int, end_minute: int, prefix: str) -> pd.DataFrame:
    m = ny_minutes(df.index)
    ny_date = pd.Series(df.index.tz_convert("America/New_York").date, index=df.index)
    if start_minute < end_minute:
        active = (m >= start_minute) & (m < end_minute)
        sess_date = ny_date
    else:
        active = (m >= start_minute) | (m < end_minute)
        sess_date = ny_date.where(m >= start_minute, ny_date - pd.Timedelta(days=1))
    tmp = df.loc[active, ["high", "low"]].copy()
    tmp["session_date"] = sess_date.loc[active].to_numpy()
    levels = tmp.groupby("session_date").agg(session_high=("high", "max"), session_low=("low", "min"))
    prev = levels.shift(1).rename(columns={"session_high": f"{prefix}_prev_high", "session_low": f"{prefix}_prev_low"})
    out = pd.DataFrame(index=df.index)
    key = sess_date.astype(str)
    prev.index = prev.index.astype(str)
    out[f"{prefix}_prev_high"] = key.map(prev[f"{prefix}_prev_high"])
    out[f"{prefix}_prev_low"] = key.map(prev[f"{prefix}_prev_low"])
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    a10 = atr(d, 10)
    a14 = atr(d, 14)
    body = (d["close"] - d["open"]).abs()
    mean_body5 = body.rolling(5, min_periods=3).mean()
    vol_osc = 100.0 * (ema(d["ticks"].fillna(0.0), 5) - ema(d["ticks"].fillna(0.0), 10)) / ema(d["ticks"].fillna(0.0), 10).replace(0, np.nan)
    signals: dict[str, pd.Series] = {}

    # ict.txt: support/resistance pivot breaks with volume and wick variants.
    ph15, pl15 = confirmed_pivots(d["high"], d["low"], 15, 15)
    res15 = ph15.ffill()
    sup15 = pl15.ffill()
    bull_wick = (d["open"] - d["low"]) > (d["close"] - d["open"])
    bear_wick = (d["open"] - d["close"]) < (d["high"] - d["open"])
    signals["ict_sr_break_volume"] = np.where(cross_above(d["close"], res15) & (vol_osc > 20) & ~bull_wick, 1, np.where(cross_below(d["close"], sup15) & (vol_osc > 20) & ~bear_wick, -1, 0))
    signals["ict_sr_wick_break"] = np.where(cross_above(d["close"], res15) & bull_wick, 1, np.where(cross_below(d["close"], sup15) & bear_wick, -1, 0))

    # trend.txt: ATR-slope pivot trendline breaks, with non-backpainted confirmation.
    ph14, pl14 = confirmed_pivots(d["high"], d["low"], 14, 14)
    slope = (a14 / 14.0).fillna(0.0)
    upper = pd.Series(np.nan, index=d.index)
    lower = pd.Series(np.nan, index=d.index)
    last_upper = np.nan
    last_lower = np.nan
    slope_ph = 0.0
    slope_pl = 0.0
    for i, idx in enumerate(d.index):
        if np.isfinite(ph14.iat[i]):
            last_upper = ph14.iat[i]
            slope_ph = slope.iat[i]
        elif np.isfinite(last_upper):
            last_upper -= slope_ph
        if np.isfinite(pl14.iat[i]):
            last_lower = pl14.iat[i]
            slope_pl = slope.iat[i]
        elif np.isfinite(last_lower):
            last_lower += slope_pl
        upper.iat[i] = last_upper
        lower.iat[i] = last_lower
    signals["trendline_break"] = np.where(cross_above(d["close"], upper), 1, np.where(cross_below(d["close"], lower), -1, 0))

    # smc.txt / ict2.txt: structure breaks, CHoCH, FVGs, equal highs/lows, order-block proxy.
    for length, name in [(5, "internal"), (10, "ict_mss"), (50, "swing")]:
        ph, pl = confirmed_pivots(d["high"], d["low"], length, 1 if name == "ict_mss" else length)
        last_high = ph.ffill()
        last_low = pl.ffill()
        bull_break = cross_above(d["close"], last_high)
        bear_break = cross_below(d["close"], last_low)
        raw = pd.Series(np.where(bull_break, 1, np.where(bear_break, -1, 0)), index=d.index)
        trend = raw.replace(0, np.nan).ffill().fillna(0)
        choch = pd.Series(np.where((raw == 1) & (trend.shift(1) < 0), 1, np.where((raw == -1) & (trend.shift(1) > 0), -1, 0)), index=d.index)
        bos = pd.Series(np.where((raw == 1) & (trend.shift(1) >= 0), 1, np.where((raw == -1) & (trend.shift(1) <= 0), -1, 0)), index=d.index)
        signals[f"smc_{name}_bos"] = bos
        signals[f"smc_{name}_choch"] = choch

        eqh = (ph.notna() & (ph - ph.ffill().shift(1)).abs().le(a14 * 0.1)).astype(int)
        eql = (pl.notna() & (pl - pl.ffill().shift(1)).abs().le(a14 * 0.1)).astype(int)
        signals[f"smc_{name}_equal_liquidity"] = eqh - eql

    bullish_fvg = (d["low"] > d["high"].shift(2)) & (d["close"].shift(1) > d["high"].shift(2)) & (body.shift(1) > mean_body5.shift(1) * 0.36)
    bearish_fvg = (d["high"] < d["low"].shift(2)) & (d["close"].shift(1) < d["low"].shift(2)) & (body.shift(1) > mean_body5.shift(1) * 0.36)
    signals["smc_fvg_create"] = np.where(bullish_fvg, 1, np.where(bearish_fvg, -1, 0))
    fvg_mid = pd.Series(np.nan, index=d.index)
    fvg_bias = pd.Series(0, index=d.index, dtype=float)
    last_mid = np.nan
    last_bias = 0
    for i in range(len(d)):
        if bullish_fvg.iat[i]:
            last_mid = (d["low"].iat[i] + d["high"].shift(2).iat[i]) / 2.0
            last_bias = 1
        elif bearish_fvg.iat[i]:
            last_mid = (d["high"].iat[i] + d["low"].shift(2).iat[i]) / 2.0
            last_bias = -1
        fvg_mid.iat[i] = last_mid
        fvg_bias.iat[i] = last_bias
    touch_fvg = ((d["low"] <= fvg_mid) & (d["high"] >= fvg_mid) & fvg_bias.ne(0)).fillna(False)
    signals["smc_fvg_mid_touch"] = np.where(touch_fvg, fvg_bias, 0)

    # Order-block proxy: last opposite candle before a structure break, then later mitigation/touch.
    ob_mid = pd.Series(np.nan, index=d.index)
    ob_bias = pd.Series(0, index=d.index, dtype=float)
    last_ob_mid = np.nan
    last_ob_bias = 0
    structure = pd.Series(signals["smc_internal_bos"], index=d.index)
    for i in range(3, len(d)):
        if structure.iat[i] == 1:
            prev = d.iloc[max(0, i - 10) : i]
            opp = prev[prev["close"] < prev["open"]]
            if not opp.empty:
                row = opp.iloc[-1]
                last_ob_mid = (row["high"] + row["low"]) / 2.0
                last_ob_bias = 1
        elif structure.iat[i] == -1:
            prev = d.iloc[max(0, i - 10) : i]
            opp = prev[prev["close"] > prev["open"]]
            if not opp.empty:
                row = opp.iloc[-1]
                last_ob_mid = (row["high"] + row["low"]) / 2.0
                last_ob_bias = -1
        ob_mid.iat[i] = last_ob_mid
        ob_bias.iat[i] = last_ob_bias
    touch_ob = ((d["low"] <= ob_mid) & (d["high"] >= ob_mid) & ob_bias.ne(0)).fillna(False)
    signals["smc_order_block_touch"] = np.where(touch_ob, ob_bias, 0)

    # kill.txt / ict2.txt: killzone pivots and session high/low sweeps.
    sessions = {
        "asia": (20 * 60, 24 * 60),
        "london": (2 * 60, 5 * 60),
        "nyam": (9 * 60 + 30, 11 * 60),
        "nypm": (13 * 60 + 30, 16 * 60),
    }
    for name, (start, end) in sessions.items():
        lv = session_levels(d, start, end, name)
        hi = lv[f"{name}_prev_high"]
        lo = lv[f"{name}_prev_low"]
        breakout = np.where(cross_above(d["close"], hi), 1, np.where(cross_below(d["close"], lo), -1, 0))
        sweep = np.where((d["high"] > hi) & (d["close"] < hi), -1, np.where((d["low"] < lo) & (d["close"] > lo), 1, 0))
        signals[f"kill_{name}_prev_range_break"] = breakout
        signals[f"kill_{name}_liquidity_sweep"] = sweep

    ny_m = ny_minutes(d.index)
    true_day_open = d["open"].where(ny_m == 0).ffill()
    six_open = d["open"].where(ny_m == 6 * 60).ffill()
    ten_open = d["open"].where(ny_m == 10 * 60).ffill()
    fourteen_open = d["open"].where(ny_m == 14 * 60).ffill()
    for name, level in [("true_day_open", true_day_open), ("six_open", six_open), ("ten_open", ten_open), ("fourteen_open", fourteen_open)]:
        signals[f"kill_open_cross_{name}"] = np.where(cross_above(d["close"], level), 1, np.where(cross_below(d["close"], level), -1, 0))

    sig_df = pd.DataFrame(signals, index=d.index).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    provenance = {
        "ict.txt": ["pivot support/resistance breaks", "volume oscillator", "wick break variants"],
        "trend.txt": ["ATR-slope pivot trendline breaks"],
        "smc.txt": ["internal/swing BOS", "CHoCH", "equal highs/lows", "FVG create/touch", "order-block touch proxy"],
        "ict2.txt": ["MSS/BOS", "liquidity", "FVG", "killzones"],
        "kill.txt": ["Asia/London/NY killzone previous range breaks", "session sweeps", "opening-price crosses"],
    }
    return sig_df, provenance


def trade_pnl(df: pd.DataFrame, signal: pd.Series, hold: int, mode: str, session: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    sig = signal.reindex(df.index).fillna(0).to_numpy(float)
    if mode == "fade":
        sig = -sig
    mask_sess = session_mask(pd.DatetimeIndex(df.index), session).to_numpy(bool)
    future = (df["close"].shift(-hold) - df["close"]).to_numpy(float)
    spread = df["spread"].fillna(df["spread"].median()).clip(lower=0.0).to_numpy(float)
    mask = np.isfinite(sig) & np.isfinite(future) & np.isfinite(spread) & (sig != 0) & mask_sess
    pnl = sig[mask] * future[mask] - spread[mask] * 2.0
    return pnl, pd.DatetimeIndex(df.index[mask])


def bootstrap_low(pnl: np.ndarray, trials: int = 1000) -> float:
    if len(pnl) < 2:
        return float("-inf")
    rng = np.random.default_rng(20260618)
    n = len(pnl)
    totals = np.empty(trials)
    for i in range(trials):
        totals[i] = rng.choice(pnl, size=n, replace=True).sum()
    return float(np.quantile(totals, 0.025))


def subperiod_result(pnl: np.ndarray) -> dict:
    if len(pnl) < 4 or pnl.sum() <= 0:
        return {"pass": False, "periods": [], "max_share": None}
    periods = [float(pnl[i * len(pnl) // 4 : (i + 1) * len(pnl) // 4].sum()) for i in range(4)]
    total = float(sum(periods))
    return {"pass": bool(max(periods) / total <= 0.75), "periods": periods, "max_share": float(max(periods) / total)}


def run_candidate(df: pd.DataFrame, signals: pd.DataFrame, signal_name: str, hold: int, mode: str, session: str) -> dict:
    folds = []
    passes = 0
    for fold, start, end in FOLDS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        test = df[(df.index >= start_ts) & (df.index <= end_ts)]
        sig = signals.loc[test.index, signal_name]
        pnl, times = trade_pnl(test, sig, hold, mode, session)
        st = ss.stat(pnl, times)
        sub = subperiod_result(pnl)
        boot = bootstrap_low(pnl) if st["pnl"] > 0 and st["trades"] >= 50 and sub["pass"] else float("-inf")
        ok = bool(st["pnl"] > 0 and st["trades"] >= 50 and sub["pass"] and boot > 0)
        passes += int(ok)
        folds.append({"fold": fold, "pnl": st["pnl"], "trades": st["trades"], "pf": st["pf"], "bootstrap_low": boot, "subperiod": sub, "pass": ok})
        if passes + (len(FOLDS) - fold) < 4:
            break
    return {"signal": signal_name, "hold": hold, "mode": mode, "session": session, "fold_passes": passes, "validated": passes >= 4, "folds": folds}


def holdout_result(candidate: dict, df: pd.DataFrame, signals: pd.DataFrame) -> dict:
    pnl, times = trade_pnl(df, signals[candidate["signal"]], candidate["hold"], candidate["mode"], candidate["session"])
    st = ss.stat(pnl, times)
    weekly = []
    if len(pnl):
        eq = pd.DataFrame({"time": times, "pnl": pnl}).set_index("time")["pnl"].resample("W").sum().cumsum().reset_index()
        weekly = [{"week": str(r["time"]), "cum_pnl": float(r["pnl"])} for _, r in eq.iterrows()]
    return {"stats": st, "weekly_equity": weekly}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS / "indicator_strategy_search_results.json"))
    args = parser.parse_args()

    xau = load_xau()
    signals, provenance = build_features(xau)
    results = []
    validated = []
    for signal_name in signals.columns:
        if int((signals[signal_name] != 0).sum()) < 100:
            continue
        for hold in HOLDS:
            for mode in MODES:
                for session in SESSION_FILTERS:
                    cand = run_candidate(xau, signals, signal_name, hold, mode, session)
                    results.append(cand)
                    if cand["validated"]:
                        validated.append(cand)

    final_holdout_opened = bool(validated)
    holdouts = []
    if final_holdout_opened:
        holdout = load_holdout()
        hsig, _ = build_features(pd.concat([xau.tail(500), holdout]).sort_index())
        hsig = hsig.reindex(holdout.index).fillna(0)
        for cand in sorted(validated, key=lambda c: c["fold_passes"], reverse=True):
            holdouts.append({"candidate": {k: cand[k] for k in ["signal", "hold", "mode", "session", "fold_passes"]}, "holdout": holdout_result(cand, holdout, hsig)})

    out = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "indicator_dir": str(INDICATORS),
        "indicator_files": sorted([p.name for p in INDICATORS.glob("*.txt")]),
        "feature_provenance": provenance,
        "xau_file": str(XAU_PARQUET),
        "xau_rows": int(len(xau)),
        "xau_start": str(xau.index.min()),
        "xau_end": str(xau.index.max()),
        "signals_tested": int(signals.shape[1]),
        "candidates_evaluated": len(results),
        "validated_count": len(validated),
        "validated": validated,
        "all_candidates": results,
        "final_holdout_opened": final_holdout_opened,
        "final_holdout_result": holdouts if final_holdout_opened else "not opened",
    }
    Path(args.out).write_text(json.dumps(ss.json_sanitize(out), indent=2), encoding="utf-8")
    print(json.dumps({"signals_tested": out["signals_tested"], "candidates_evaluated": len(results), "validated_count": len(validated), "final_holdout_opened": final_holdout_opened, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
