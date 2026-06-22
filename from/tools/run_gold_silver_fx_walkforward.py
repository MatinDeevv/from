#!/usr/bin/env python3
"""Walk-forward for native Dukascopy gold/silver/FX leaders."""

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
XAU = DERIVED / "duka_XAUUSD_5m_20230101_20260522.parquet"
SYMBOLS = ["XAGUSD", "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY"]
FAMILY = {"XAGUSD": "metals", **{s: "fx" for s in SYMBOLS if s != "XAGUSD"}}
FEATURES = ["volz", "session_volz", "momentum_ratio", "divergence", "ew_zspread"]
SESSIONS = ["london", "ny", "all"]
MODES = ["follow", "fade"]
WINDOWS = [4, 6, 9, 12, 18, 24]
QUANTILES = [0.55, 0.65, 0.75]
HOLDS = [18, 24, 30, 36, 42, 48]
FOLDS = [
    (1, "2023-07-01", "2023-12-31"),
    (2, "2024-01-01", "2024-06-30"),
    (3, "2024-07-01", "2024-12-31"),
    (4, "2025-01-01", "2025-06-30"),
    (5, "2025-07-01", "2025-12-31"),
    (6, "2026-01-01", "2026-05-21"),
]


def load_parquet(path: Path, name: str) -> pd.DataFrame:
    con = duckdb.connect()
    col = "xau" if name == "XAUUSD" else name
    price_col = "close"
    df = con.execute(f"select time, {price_col} as {col}, spread from read_parquet('{str(path).replace(chr(39), chr(39)*2)}') order by time").fetchdf()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=[col]).drop_duplicates("time").set_index("time").sort_index()


def session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    if session == "london":
        return np.array([(8 <= t.hour < 12) for t in index], dtype=bool)
    if session == "ny":
        return np.array([(13 <= t.hour < 21) for t in index], dtype=bool)
    return np.ones(len(index), dtype=bool)


def feature_values(df: pd.DataFrame, leader: str, window: int, feature: str, session: str) -> np.ndarray:
    lr = np.log(df[leader]).diff()
    xr = np.log(df["xau"]).diff()
    lmom = lr.rolling(window).sum().shift(1)
    xmom = xr.rolling(window).sum().shift(1)
    if feature in {"volz", "session_volz"}:
        vol = lr.rolling(48).std().shift(1) * math.sqrt(window)
        out = lmom / vol.replace(0.0, np.nan)
        if feature == "session_volz":
            out = out.mask(~session_mask(pd.DatetimeIndex(df.index), session), 0.0)
        return out.to_numpy(float)
    if feature == "momentum_ratio":
        return (lmom / xmom.abs().replace(0.0, np.nan)).to_numpy(float)
    if feature == "divergence":
        lv = lr.rolling(48).std().shift(1) * math.sqrt(window)
        xv = xr.rolling(48).std().shift(1) * math.sqrt(window)
        return ((lmom / lv.replace(0.0, np.nan)) - (xmom / xv.replace(0.0, np.nan))).to_numpy(float)
    if feature == "ew_zspread":
        lv = lr.ewm(span=48, adjust=False, min_periods=12).std().shift(1) * math.sqrt(window)
        xv = xr.ewm(span=48, adjust=False, min_periods=12).std().shift(1) * math.sqrt(window)
        return ((lmom / lv.replace(0.0, np.nan)) - (xmom / xv.replace(0.0, np.nan))).to_numpy(float)
    raise ValueError(feature)


def trade_pnl(df: pd.DataFrame, leader: str, feature: str, window: int, threshold: float, mode: str, session: str, hold: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    d = df.dropna(subset=[leader, "xau", "spread"]).copy()
    vals = feature_values(d, leader, window, feature, session)
    sig = np.where(vals > threshold, 1.0, np.where(vals < -threshold, -1.0, 0.0))
    if mode == "fade":
        sig = -sig
    future = (d["xau"].shift(-hold) - d["xau"]).to_numpy(float)
    mask = np.isfinite(sig) & np.isfinite(future) & (sig != 0) & session_mask(pd.DatetimeIndex(d.index), session)
    pnl = sig[mask] * future[mask] - d["spread"].fillna(d["spread"].median()).to_numpy(float)[mask] * 2.0
    return pnl, pd.DatetimeIndex(d.index[mask])


def coverage_ratio(leader: pd.DataFrame, start: str, end: str) -> float:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    subset = leader[(leader.index >= s) & (leader.index <= e)]
    expected = len(pd.date_range(s, e, freq="5min"))
    return min(1.0, len(subset) / max(1, expected))


def bootstrap_low(pnl: np.ndarray, trials: int = 1000) -> float:
    if len(pnl) < 2:
        return float("-inf")
    rng = np.random.default_rng(20260618)
    n = len(pnl)
    totals = np.empty(trials)
    for i in range(trials):
        totals[i] = rng.choice(pnl, size=n, replace=True).sum()
    return float(np.quantile(totals, 0.025))


def subperiod_pass(pnl: np.ndarray) -> bool:
    if len(pnl) < 4 or pnl.sum() <= 0:
        return False
    parts = [pnl[i * len(pnl) // 4 : (i + 1) * len(pnl) // 4].sum() for i in range(4)]
    return max(parts) / pnl.sum() <= 0.75


def quality(symbol: str, df: pd.DataFrame) -> dict:
    deltas = df.index.to_series().diff().dropna().dt.total_seconds() / 60
    gaps = deltas[deltas > 10]
    return {
        "symbol": symbol,
        "bars": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "gap_events_gt_10m": int(len(gaps)),
        "max_gap_5m_bars": float((gaps.max() / 5 - 1) if len(gaps) else 0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPORTS / "gold_silver_fx_walkforward_results.json"))
    args = p.parse_args()

    xau = load_parquet(XAU, "XAUUSD")[["xau", "spread"]]
    leaders = {}
    download_log = json.loads((REPORTS / "gold_silver_fx_probe_log.json").read_text(encoding="utf-8"))
    coverage = [{"symbol": "XAUUSD", "source": str(XAU), "start": str(xau.index.min()), "end": str(xau.index.max()), "bar_count": int(len(xau))}]
    quality_rows = []
    for sym in SYMBOLS:
        path = DERIVED / f"leader_{sym}_5m_20230101_20260521.parquet"
        if not path.exists():
            coverage.append({"symbol": sym, "source": "not downloaded", "start": None, "end": None, "bar_count": 0})
            continue
        df = load_parquet(path, sym)[[sym]]
        leaders[sym] = df
        coverage.append({"symbol": sym, "source": str(path), "start": str(df.index.min()), "end": str(df.index.max()), "bar_count": int(len(df))})
        quality_rows.append(quality(sym, df))

    all_candidates = []
    validated = []
    skip_counts = {}
    for sym, ldf in leaders.items():
        merged = xau.join(ldf, how="inner").sort_index()
        for mode in MODES:
            for feature in FEATURES:
                for session in SESSIONS:
                    for window in WINDOWS:
                        for q in QUANTILES:
                            for hold in HOLDS:
                                fold_rows = []
                                passes = 0
                                for fold, start, end in FOLDS:
                                    cov = coverage_ratio(ldf, start, end)
                                    if cov < 0.80:
                                        fold_rows.append({"fold": fold, "skipped": True, "coverage_ratio": cov, "reason": "coverage below 80%"})
                                        skip_counts[f"{sym}|{fold}|coverage below 80%"] = skip_counts.get(f"{sym}|{fold}|coverage below 80%", 0) + 1
                                        continue
                                    fit = merged[merged.index < pd.Timestamp(start, tz="UTC")]
                                    test = merged[(merged.index >= pd.Timestamp(start, tz="UTC")) & (merged.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
                                    vals = feature_values(fit.dropna(subset=[sym, "xau"]), sym, window, feature, session)
                                    sample = np.abs(vals[np.isfinite(vals)])
                                    if len(sample) < 100:
                                        fold_rows.append({"fold": fold, "skipped": True, "coverage_ratio": cov, "reason": "insufficient fit sample"})
                                        continue
                                    threshold = float(np.quantile(sample, q))
                                    pnl, times = trade_pnl(test, sym, feature, window, threshold, mode, session, hold)
                                    st = ss.stat(pnl, times)
                                    boot = bootstrap_low(pnl) if st["pnl"] > 0 and st["trades"] >= 50 else float("-inf")
                                    sub_ok = subperiod_pass(pnl)
                                    ok = bool(st["pnl"] > 0 and st["trades"] >= 50 and boot > 0 and sub_ok)
                                    passes += int(ok)
                                    fold_rows.append({"fold": fold, "skipped": False, "coverage_ratio": cov, "pnl": st["pnl"], "trades": st["trades"], "pf": st["pf"], "bootstrap_low": boot, "subperiod_pass": sub_ok, "pass": ok})
                                cand = {"symbol": sym, "family": FAMILY[sym], "mode": mode, "feature": feature, "session": session, "window": window, "quantile": q, "hold": hold, "fold_passes": passes, "validated": passes >= 4, "folds": fold_rows}
                                all_candidates.append(cand)
                                if cand["validated"]:
                                    validated.append(cand)
    families = sorted(set(c["family"] for c in validated))
    portfolio_pass = bool("metals" in families and "fx" in families)
    out = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "download_log": download_log,
        "coverage": coverage,
        "quality": quality_rows,
        "candidates_evaluated": len(all_candidates),
        "validated_count": len(validated),
        "validated": validated,
        "skip_counts": skip_counts,
        "portfolio": {"portfolio_pass": portfolio_pass, "families": families, "allocator_swing": None, "effective_bets": 0.0, "reason": "No validated metals+FX portfolio" if not portfolio_pass else "validated families exist"},
        "final_holdout_opened": False,
        "final_holdout_result": "not opened",
    }
    Path(args.out).write_text(json.dumps(ss.json_sanitize(out), indent=2), encoding="utf-8")
    print(json.dumps({"coverage": coverage, "candidates_evaluated": len(all_candidates), "validated_count": len(validated), "portfolio_pass": portfolio_pass, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
