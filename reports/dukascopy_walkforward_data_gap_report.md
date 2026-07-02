# Dukascopy Walk-Forward Acquisition Report

## Status

**Insufficient data acquired to run study.**

Specific gap: XAUUSD has usable long-history coverage from the local restored master, but no required 2023-present 5m leader data was acquired for either the energy family or the sector/proxy family. The minimum condition in `walkforward_study_design.md` was therefore not met.

Final holdout was not opened. Walk-forward did not meet acceptance criteria.

## Frozen Design

The study design was written before any walk-forward backtest results:

- `C:\Users\marti\from\reports\walkforward_study_design.md`

No walk-forward backtest was run and no final holdout was opened.

## XAUUSD Coverage

Local source:

- `C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet`

Exported 5m file:

- `C:\Users\marti\from\data\derived\duka_XAUUSD_5m_20230101_20260522.parquet`

Quality summary:

| Field | Value |
|---|---:|
| Rows | 164,744 |
| Start | 2023-01-03 19:00:00-05:00 |
| End | 2026-05-22 09:50:00-04:00 |
| Null close bars | 0 |
| Gap-fill bars | 2,752 |
| Gap events >10 minutes | 3,334 |
| Max consecutive missing 5m bars | 1,224 |
| Spread p50 | 0.4152 |
| Spread p95 | 0.8285 |
| Spread p99 | 1.2385 |

The XAU export is enough to cover the walk-forward fit/test schedule through May 21, 2026. It still has calendar gaps/gap-fill bars that would need closure classification before a production-grade data acceptance decision, but XAU was not the binding blocker in this run.

## Required Leader Acquisition Status

| Target | Requested / probed instruments | Status | Reason |
|---|---|---|---|
| USO proxy | `USOIL`, `LIGHTCMDUSD`, `LIGHT.CMDUSD` | Not acquired | Dukascopy archive probes returned timeouts or HTTP 503. No 2023-present 5m file exists locally. |
| UNG proxy | `NATGAS`, `GASCMDUSD`, `GAS.CMDUSD` | Not acquired | Dukascopy archive probes returned timeouts or HTTP 503. No 2023-present 5m file exists locally. |
| XLF proxy | `SPX500`, `US30`, `USA500IDXUSD`, `USA30IDXUSD` | Not acquired | Dukascopy archive probes returned timeouts or HTTP 503. Yahoo XLF 5m cache is only 60 days and fails the January 1, 2023 minimum. |
| XLY proxy | Same sector proxies as XLF | Not acquired | Same sector proxy gap. Yahoo XLY 5m cache is only 60 days and fails the January 1, 2023 minimum. |

## Evidence Commands

```powershell
& 'C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools/fetch_dukascopy_5m.py --instrument <instrument> --start <date> --end <date> ...
```

The existing generic downloader can fetch and aggregate Dukascopy `.bi5` ticks for instruments that the public archive exposes. Archive probes were successful for some known FX/metals examples (`USDJPY`, `XAUUSD`) but not for the required energy and sector/proxy symbols.

## Why The Walk-Forward Was Not Run

The prompt requires at least:

- XAUUSD
- one energy leader/proxy
- one sector leader/proxy
- matching 5m coverage from at least January 1, 2023

Only XAUUSD coverage was available locally. The available Yahoo 5m files for `XLF`, `XLY`, `UNG`, and `USO` are limited to roughly 60 days, which is the exact underpowered data problem this prompt was written to avoid.

Running the walk-forward on that data would violate the study design and would reopen the old methodology failure: testing a short window that cannot support meaningful fold/subperiod statistics.

## Required Next Fix

Use a proper bulk data acquisition path for the missing leaders before any backtest:

1. Confirm exact Dukascopy archive symbols for WTI oil, natural gas, USA500, and USA30 using JForex instrument metadata or a known-good downloader library.
2. Download raw `.bi5` tick files for `2023-01-01` through at least `2026-05-21`.
3. Save raw tick files before resampling.
4. Resample to 5m OHLCV/parquet with mid-price close and average spread.
5. Re-run the data quality checks.
6. Only if at least one energy and one sector/proxy leader pass the data checks, run the frozen walk-forward study.

## Required Conclusion

Insufficient data acquired to run study. Missing 2023-present 5m Dukascopy/proxy coverage for at least one energy leader and at least one sector leader. Next step: acquire verified Dukascopy/JForex tick history for `LIGHT.CMD/USD` or equivalent, `GAS.CMD/USD` or equivalent, and `USA500.IDX/USD`/`USA30.IDX/USD` or a better XLF/XLY proxy, then rerun the frozen data quality gate before any walk-forward backtest.
