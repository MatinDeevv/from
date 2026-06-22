#!/usr/bin/env python3
"""Probe leader data and run the frozen walk-forward with available leaders."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import lzma
import math
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import strict_screening_pipeline as ss


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
YAHOO_CACHE = DATA / "cache" / "yahoo"
DERIVED = DATA / "derived"
XAU_FILE = DERIVED / "duka_XAUUSD_5m_20230101_20260522.parquet"

PROBES = {
    "USO": ["USOIL", "XBRUSD", "LIGHT.CMD/USD", "LIGHTCMDUSD", "OILUSD", "WTI"],
    "UNG": ["NATGAS", "XNGUSD", "GAS.CMD/USD", "GASCMDUSD", "NGAS", "GASUSD"],
    "XLF": ["USA500.IDX/USD", "USA500IDXUSD", "SPX500USD", "SPX500", "US500", "SPXUSD"],
    "XLY": ["USA500.IDX/USD", "USA500IDXUSD", "SPX500USD", "SPX500", "US500", "SPXUSD"],
}

FOLDS = [
    (1, "2023-07-01", "2023-12-31"),
    (2, "2024-01-01", "2024-06-30"),
    (3, "2024-07-01", "2024-12-31"),
    (4, "2025-01-01", "2025-06-30"),
    (5, "2025-07-01", "2025-12-31"),
    (6, "2026-01-01", "2026-05-21"),
]

FAMILIES = {
    "energy": {
        "symbols": ["USO", "UNG"],
        "features": ["session_volz", "volz", "momentum_ratio"],
        "mode": "follow",
    },
    "sector": {
        "symbols": ["XLF", "XLY"],
        "features": ["session_volz", "divergence", "ew_zspread"],
        "mode": "fade",
    },
}
WINDOWS = [4, 6, 9, 12, 18, 24]
QUANTILES = [0.55, 0.65, 0.75]
HOLDS = [18, 24, 30, 36, 42, 48]


def archive_url(instrument: str, when: dt.datetime) -> str:
    return f"https://datafeed.dukascopy.com/datafeed/{instrument}/{when.year}/{when.month - 1:02d}/{when.day:02d}/{when.hour:02d}h_ticks.bi5"


def probe_instrument(instrument: str, timeout: int = 8) -> dict:
    # Representative market hours inside the requested one-week probe.
    checks = [
        dt.datetime(2023, 1, 3, 10, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 4, 10, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 5, 15, tzinfo=dt.timezone.utc),
    ]
    last = None
    for when in checks:
        url = archive_url(instrument, when)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                body = response.read()
            payload = lzma.decompress(body)
            valid = len(payload) > 0 and len(payload) % 20 == 0
            result = {"instrument": instrument, "url": url, "http_status": status, "valid_bi5": valid, "bytes": len(body), "payload_bytes": len(payload), "pass": bool(valid)}
            if valid:
                return result
            last = result
        except urllib.error.HTTPError as exc:
            last = {"instrument": instrument, "url": url, "http_status": exc.code, "valid_bi5": False, "pass": False, "error": str(exc)}
            if exc.code in {404, 503}:
                return last
        except Exception as exc:
            last = {"instrument": instrument, "url": url, "http_status": None, "valid_bi5": False, "pass": False, "error": type(exc).__name__ + ": " + str(exc)}
    return last or {"instrument": instrument, "http_status": None, "valid_bi5": False, "pass": False, "error": "not attempted"}


def safe_yahoo(symbol: str) -> str:
    return symbol.replace("=", "_").replace("^", "_").replace(".", "_").replace("-", "_")


def load_xau() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"select time, close as xau, spread, ticks from read_parquet('{str(XAU_FILE).replace(chr(39), chr(39)*2)}') order by time").fetchdf()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def load_yahoo(symbol: str) -> pd.DataFrame | None:
    path = YAHOO_CACHE / f"{safe_yahoo(symbol)}_5m_60d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    if symbol not in df.columns:
        value_cols = [c for c in df.columns if c != "time"]
        if not value_cols:
            return None
        df = df.rename(columns={value_cols[0]: symbol})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", symbol]].dropna().drop_duplicates("time").set_index("time").sort_index()


def load_acquired(symbol: str) -> tuple[pd.DataFrame | None, str]:
    paths = {
        "USO": DERIVED / "leader_USO_LIGHTCMDUSD_5m_20230101_20260521.parquet",
        "UNG": DERIVED / "leader_UNG_GASCMDUSD_5m_20230101_20260521.parquet",
        "XLF": DERIVED / "leader_XLF_USA500IDXUSD_5m_20230101_20260521.parquet",
        "XLY": DERIVED / "leader_XLY_USA500IDXUSD_5m_20230101_20260521.parquet",
    }
    path = paths.get(symbol)
    if path is None or not path.exists():
        return None, ""
    con = duckdb.connect()
    df = con.execute(f"select time, close from read_parquet('{str(path).replace(chr(39), chr(39)*2)}') order by time").fetchdf()
    if df.empty:
        return None, str(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.rename(columns={"close": symbol}).set_index("time")[[symbol]].sort_index(), str(path)


def feature_values(df: pd.DataFrame, leader: str, window: int, feature: str) -> np.ndarray:
    lr = np.log(df[leader]).diff()
    xr = np.log(df["xau"]).diff()
    lmom = lr.rolling(window).sum().shift(1)
    xmom = xr.rolling(window).sum().shift(1)
    if feature in {"volz", "session_volz"}:
        vol = lr.rolling(48).std().shift(1) * math.sqrt(window)
        return (lmom / vol.replace(0.0, np.nan)).to_numpy(float)
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


def london_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([(8 <= t.hour < 12) for t in index], dtype=bool)


def trade_pnl(df: pd.DataFrame, leader: str, feature: str, window: int, threshold: float, mode: str, hold: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    d = df.dropna(subset=[leader, "xau", "spread"]).copy()
    vals = feature_values(d, leader, window, feature)
    sig = np.where(vals > threshold, 1.0, np.where(vals < -threshold, -1.0, 0.0))
    if mode == "fade":
        sig = -sig
    future = (d["xau"].shift(-hold) - d["xau"]).to_numpy(float)
    mask = np.isfinite(sig) & np.isfinite(future) & (sig != 0) & london_mask(pd.DatetimeIndex(d.index))
    pnl = sig[mask] * future[mask] - d["spread"].fillna(d["spread"].median()).to_numpy(float)[mask] * 2.0
    return pnl, pd.DatetimeIndex(d.index[mask])


def coverage_ratio(leader: pd.DataFrame, start: str, end: str) -> float:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    subset = leader[(leader.index >= s) & (leader.index <= e)]
    if subset.empty:
        return 0.0
    # Expected regular-hours ETF bars are approximate; this intentionally lets
    # Yahoo's exchange-hours stream decide whether the fold is usable.
    bizdays = pd.date_range(s.date(), e.date(), freq="B")
    expected = max(1, len(bizdays) * 78)
    return min(1.0, len(subset) / expected)


def bootstrap_low(pnl: np.ndarray, trials: int = 1000) -> float:
    if len(pnl) < 2:
        return float("-inf")
    rng = np.random.default_rng(20260618)
    totals = np.empty(trials)
    n = len(pnl)
    for i in range(trials):
        totals[i] = rng.choice(pnl, size=n, replace=True).sum()
    return float(np.quantile(totals, 0.025))


def subperiod_ok(pnl: np.ndarray) -> bool:
    if len(pnl) < 4 or pnl.sum() <= 0:
        return False
    chunks = [pnl[i * len(pnl) // 4 : (i + 1) * len(pnl) // 4].sum() for i in range(4)]
    return max(chunks) / pnl.sum() <= 0.75


def run_walkforward(xau: pd.DataFrame, leaders: dict[str, pd.DataFrame]) -> dict:
    results = []
    validated = []
    skipped = {}
    for family, spec in FAMILIES.items():
        for symbol in spec["symbols"]:
            if symbol not in leaders:
                skipped[symbol] = [{"fold": f[0], "reason": "no usable data"} for f in FOLDS]
                continue
            leader = leaders[symbol]
            merged = xau.join(leader, how="inner").sort_index()
            symbol_rows = []
            for feature in spec["features"]:
                for window in WINDOWS:
                    for q in QUANTILES:
                        for hold in HOLDS:
                            fold_passes = 0
                            fold_rows = []
                            for fold, start, end in FOLDS:
                                cov = coverage_ratio(leader, start, end)
                                if cov < 0.80:
                                    fold_rows.append({"fold": fold, "skipped": True, "coverage_ratio": cov, "reason": "coverage below 80%"})
                                    continue
                                fit = merged[merged.index < pd.Timestamp(start, tz="UTC")]
                                test = merged[(merged.index >= pd.Timestamp(start, tz="UTC")) & (merged.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
                                vals = feature_values(fit.dropna(subset=[symbol, "xau"]), symbol, window, feature)
                                sample = np.abs(vals[np.isfinite(vals)])
                                if len(sample) < 100:
                                    fold_rows.append({"fold": fold, "skipped": True, "coverage_ratio": cov, "reason": "insufficient fit sample"})
                                    continue
                                threshold = float(np.quantile(sample, q))
                                pnl, times = trade_pnl(test, symbol, feature, window, threshold, spec["mode"], hold)
                                st = ss.stat(pnl, times)
                                boot = bootstrap_low(pnl) if len(pnl) >= 50 and st["pnl"] > 0 else float("-inf")
                                sub_ok = subperiod_ok(pnl)
                                passed = bool(st["pnl"] > 0 and st["trades"] >= 50 and boot > 0 and sub_ok)
                                fold_passes += int(passed)
                                fold_rows.append({"fold": fold, "skipped": False, "coverage_ratio": cov, "pnl": st["pnl"], "trades": st["trades"], "pf": st["pf"], "bootstrap_low": boot, "subperiod_pass": sub_ok, "pass": passed})
                            candidate = {"symbol": symbol, "family": family, "feature": feature, "window": window, "quantile": q, "hold": hold, "fold_passes": fold_passes, "validated": fold_passes >= 4, "folds": fold_rows}
                            results.append(candidate)
                            symbol_rows.append(candidate)
                            if candidate["validated"]:
                                validated.append(candidate)
            if not symbol_rows:
                skipped[symbol] = [{"fold": f[0], "reason": "no candidate rows"} for f in FOLDS]
    families = sorted(set(c["family"] for c in validated))
    portfolio_pass = bool(validated and "energy" in families and "sector" in families)
    return {"candidates_evaluated": len(results), "validated_count": len(validated), "validated": validated, "all_candidates": results, "skipped": skipped, "portfolio": {"portfolio_pass": portfolio_pass, "families": families, "allocator_swing": None, "effective_bets": 0.0, "reason": "No validated cross-family portfolio" if not portfolio_pass else "Validated candidates exist; allocator not built in no-holdout runner"}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS / "available_leader_walkforward_results.json"))
    args = parser.parse_args()

    acquisition_log = []
    acquired = {}
    seen_success_proxy = {}
    for target, instruments in PROBES.items():
        for instrument in instruments:
            result = probe_instrument(instrument)
            result["target"] = target
            acquisition_log.append(result)
            if result.get("pass"):
                # Full download intentionally omitted unless a probe succeeds;
                # current public archive probes have not succeeded for these.
                acquired[target] = {"source": "dukascopy_bi5_probe_only", "instrument": instrument}
                break

    xau = load_xau()
    leaders = {}
    coverage_rows = [{
        "symbol": "XAUUSD",
        "source": "Dukascopy/restored parquet",
        "start": str(xau.index.min()),
        "end": str(xau.index.max()),
        "bar_count": int(len(xau)),
    }]
    for symbol in ["USO", "UNG", "XLF", "XLY"]:
        acquired_df, acquired_path = load_acquired(symbol)
        y = load_yahoo(symbol)
        if acquired_df is not None and len(acquired_df):
            leaders[symbol] = acquired_df
            coverage_rows.append({"symbol": symbol, "source": f"acquired Dukascopy proxy parquet: {acquired_path}", "start": str(acquired_df.index.min()), "end": str(acquired_df.index.max()), "bar_count": int(len(acquired_df))})
        elif y is not None and len(y):
            leaders[symbol] = y
            coverage_rows.append({"symbol": symbol, "source": "local Yahoo 5m cache", "start": str(y.index.min()), "end": str(y.index.max()), "bar_count": int(len(y))})
        else:
            coverage_rows.append({"symbol": symbol, "source": "not available", "start": None, "end": None, "bar_count": 0})

    wf = run_walkforward(xau, leaders)
    out = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "acquisition_log": acquisition_log,
        "coverage_table": coverage_rows,
        "walkforward": wf,
        "final_holdout_opened": False,
        "final_holdout_decision": "Final holdout was not opened. Walk-forward did not meet acceptance criteria.",
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(ss.json_sanitize(out), indent=2), encoding="utf-8")
    print(json.dumps({
        "acquisition_attempts": len(acquisition_log),
        "coverage_symbols": coverage_rows,
        "candidates_evaluated": wf["candidates_evaluated"],
        "validated_count": wf["validated_count"],
        "portfolio_pass": wf["portfolio"]["portfolio_pass"],
        "final_holdout_opened": False,
        "out": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
