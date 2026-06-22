#!/usr/bin/env python3
"""XAUUSD-only walk-forward using self-derived features."""

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
XAU_PARQUET = DERIVED / "duka_XAUUSD_5m_20230101_20260522.parquet"
XAU_HOLDOUT_CSV = DERIVED / "xauusd_dukascopy_5m_forward_20260522_20260618.csv"

FEATURES = ["xau_volz", "xau_session_volz", "xau_momentum_ratio", "xau_rsi", "xau_ew_zspread"]
WINDOWS = [4, 6, 9, 12, 18, 24]
QUANTILES = [0.55, 0.65, 0.75]
HOLDS = [18, 24, 30, 36, 42, 48]
MODES = ["follow", "fade"]
SESSIONS = ["london", "ny", "all"]
FOLDS = [
    (1, "2023-07-01", "2023-12-31"),
    (2, "2024-01-01", "2024-06-30"),
    (3, "2024-07-01", "2024-12-31"),
    (4, "2025-01-01", "2025-06-30"),
    (5, "2025-07-01", "2025-12-31"),
    (6, "2026-01-01", "2026-05-21"),
]


def load_xau() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"select time, close as xau, spread, ticks from read_parquet('{str(XAU_PARQUET).replace(chr(39), chr(39) * 2)}') order by time").fetchdf()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=["xau"]).drop_duplicates("time").set_index("time").sort_index()


def load_holdout() -> pd.DataFrame:
    df = pd.read_csv(XAU_HOLDOUT_CSV, parse_dates=["time"]).rename(columns={"close": "xau"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=["xau"]).drop_duplicates("time").set_index("time").sort_index()


def session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    if session == "london":
        return np.array([(8 <= t.hour < 12) for t in index], dtype=bool)
    if session == "ny":
        return np.array([(13 <= t.hour < 21) for t in index], dtype=bool)
    return np.ones(len(index), dtype=bool)


def rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def feature_values(df: pd.DataFrame, window: int, feature: str, session: str) -> np.ndarray:
    px = df["xau"].astype(float)
    ret = np.log(px).diff()
    mom = ret.rolling(window).sum().shift(1)
    if feature in {"xau_volz", "xau_session_volz"}:
        vol = ret.rolling(48).std().shift(1) * math.sqrt(window)
        out = mom / vol.replace(0.0, np.nan)
        if feature == "xau_session_volz":
            out = out.mask(~session_mask(pd.DatetimeIndex(df.index), session), 0.0)
        return out.to_numpy(float)
    if feature == "xau_momentum_ratio":
        denom = ret.rolling(window * 2).sum().shift(1).abs()
        return (mom / denom.replace(0.0, np.nan)).to_numpy(float)
    if feature == "xau_rsi":
        return ((rsi(px, window).shift(1) - 50.0) / 50.0).to_numpy(float)
    if feature == "xau_ew_zspread":
        ew_mean = ret.ewm(span=window * 4, adjust=False, min_periods=window).mean().shift(1)
        ew_std = ret.ewm(span=window * 4, adjust=False, min_periods=window).std().shift(1)
        return ((mom - ew_mean * window) / (ew_std.replace(0.0, np.nan) * math.sqrt(window))).to_numpy(float)
    raise ValueError(feature)


def trade_pnl(df: pd.DataFrame, feature: str, window: int, threshold: float, mode: str, session: str, hold: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    d = df.dropna(subset=["xau", "spread"]).copy()
    vals = feature_values(d, window, feature, session)
    sig = np.where(vals > threshold, 1.0, np.where(vals < -threshold, -1.0, 0.0))
    if mode == "fade":
        sig = -sig
    future = (d["xau"].shift(-hold) - d["xau"]).to_numpy(float)
    spread = d["spread"].fillna(d["spread"].median()).clip(lower=0.0).to_numpy(float)
    mask = np.isfinite(sig) & np.isfinite(future) & np.isfinite(spread) & (sig != 0) & session_mask(pd.DatetimeIndex(d.index), session)
    pnl = sig[mask] * future[mask] - spread[mask] * 2.0
    return pnl, pd.DatetimeIndex(d.index[mask])


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
    max_share = max(periods) / float(sum(periods))
    return {"pass": bool(max_share <= 0.75), "periods": periods, "max_share": float(max_share)}


def run_candidate(df: pd.DataFrame, feature: str, window: int, quantile: float, hold: int, mode: str, session: str) -> dict:
    folds = []
    passes = 0
    for fold, start, end in FOLDS:
        fit = df[df.index < pd.Timestamp(start, tz="UTC")]
        test = df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=5))]
        vals = feature_values(fit, window, feature, session)
        sample = np.abs(vals[np.isfinite(vals)])
        if len(sample) < 100:
            folds.append({"fold": fold, "skipped": True, "reason": "insufficient fit sample"})
            continue
        threshold = float(np.quantile(sample, quantile))
        pnl, times = trade_pnl(test, feature, window, threshold, mode, session, hold)
        st = ss.stat(pnl, times)
        sub = subperiod_result(pnl)
        normal_low = ss.normal_lower_bound(pnl) if st["pnl"] > 0 and st["trades"] >= 50 and sub["pass"] else float("-inf")
        # The bootstrap gate is exact but expensive. Run it only after the cheap exact gates pass.
        boot = bootstrap_low(pnl) if st["pnl"] > 0 and st["trades"] >= 50 and sub["pass"] else float("-inf")
        ok = bool(st["pnl"] > 0 and st["trades"] >= 50 and boot > 0 and sub["pass"])
        passes += int(ok)
        folds.append({"fold": fold, "skipped": False, "threshold": threshold, "pnl": st["pnl"], "trades": st["trades"], "pf": st["pf"], "normal_lower_bound": normal_low, "bootstrap_low": boot, "subperiod": sub, "pass": ok})
        if passes + (len(FOLDS) - fold) < 4:
            break
    return {"feature": feature, "window": window, "quantile": quantile, "hold": hold, "mode": mode, "session": session, "fold_passes": passes, "validated": passes >= 4, "folds": folds}


def holdout_result(candidate: dict, train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    vals = feature_values(train_df, candidate["window"], candidate["feature"], candidate["session"])
    sample = np.abs(vals[np.isfinite(vals)])
    threshold = float(np.quantile(sample, candidate["quantile"]))
    pnl, times = trade_pnl(holdout_df, candidate["feature"], candidate["window"], threshold, candidate["mode"], candidate["session"], candidate["hold"])
    st = ss.stat(pnl, times)
    if len(pnl):
        eq = np.cumsum(pnl)
        weekly = pd.DataFrame({"time": times, "pnl": pnl}).set_index("time")["pnl"].resample("W").sum().cumsum().reset_index()
        weekly_rows = [{"week": str(r["time"]), "cum_pnl": float(r["pnl"])} for _, r in weekly.iterrows()]
    else:
        weekly_rows = []
    return {"threshold": threshold, "stats": st, "weekly_equity": weekly_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS / "xau_only_walkforward_results.json"))
    args = parser.parse_args()
    xau = load_xau()
    results = []
    validated = []
    for feature in FEATURES:
        for window in WINDOWS:
            for quantile in QUANTILES:
                for hold in HOLDS:
                    for mode in MODES:
                        for session in SESSIONS:
                            cand = run_candidate(xau, feature, window, quantile, hold, mode, session)
                            results.append(cand)
                            if cand["validated"]:
                                validated.append(cand)
    final_holdout_opened = bool(validated)
    holdouts = []
    if final_holdout_opened:
        holdout_df = load_holdout()
        train_df = xau[xau.index <= pd.Timestamp("2026-05-21 23:59:59", tz="UTC")]
        for cand in sorted(validated, key=lambda c: c["fold_passes"], reverse=True):
            holdouts.append({"candidate": {k: cand[k] for k in ["feature", "window", "quantile", "hold", "mode", "session", "fold_passes"]}, "holdout": holdout_result(cand, train_df, holdout_df)})
    out = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "xau_file": str(XAU_PARQUET),
        "xau_rows": int(len(xau)),
        "xau_start": str(xau.index.min()),
        "xau_end": str(xau.index.max()),
        "candidates_evaluated": len(results),
        "validated_count": len(validated),
        "validated": validated,
        "all_candidates": results,
        "final_holdout_opened": final_holdout_opened,
        "final_holdout_result": holdouts if final_holdout_opened else "not opened",
    }
    Path(args.out).write_text(json.dumps(ss.json_sanitize(out), indent=2), encoding="utf-8")
    print(json.dumps({"candidates_evaluated": len(results), "validated_count": len(validated), "final_holdout_opened": final_holdout_opened, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
