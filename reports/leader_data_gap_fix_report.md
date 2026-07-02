# Leader Data Gap Fix Report

## Status

**Insufficient leader data acquired. `POLYGON_API_KEY` is not set. Manual/API acquisition is required before the walk-forward can run.**

No walk-forward backtest was run. The final holdout was not opened. The existing XAU file was not modified.

## Scope

This run only addressed the leader-data acquisition blocker from:

- `C:\Users\marti\Downloads\codex_fix_leader_data_gap_prompt.md`

Frozen study design remains:

- `C:\Users\marti\from\reports\walkforward_study_design.md`

Existing XAU file left untouched:

- `C:\Users\marti\from\data\derived\duka_XAUUSD_5m_20230101_20260522.parquet`

## Tier 1 - Dukascopy bi5 Probe

The prompt explicitly said not to retry the previous failed CFD symbols and to probe only the correct bi5-format energy names first.

| Instrument | Intended proxy | Probe result | Decision |
|---|---|---|---|
| `XBRUSD` | Brent crude substitute for USO/WTI | Timed out on three 2023 market-hour probes | Not acquired |
| `XNGUSD` | Natural gas substitute for UNG | HTTP 404 on three 2023 market-hour probes | Not available in tested bi5 archive path |

Probe windows:

- `2023/00/03/10h_ticks.bi5`
- `2023/05/15/10h_ticks.bi5`
- `2023/10/15/10h_ticks.bi5`

No raw tick files or 5m parquet files were produced from Tier 1 because no valid `.bi5` payload was acquired.

## Tier 2 - Polygon.io

Required environment variable:

- `POLYGON_API_KEY`

Result:

- `POLYGON_API_KEY=not set`

Per prompt, the acquisition stopped here. No API key was hardcoded and no Polygon requests were attempted.

Required Polygon tickers once the key is available:

| Target | Preferred Polygon ticker | Fallback |
|---|---|---|
| UNG | `UNG` ETF 5m adjusted aggregates | `C:NGc1` continuous natural-gas futures |
| USO | `USO` ETF 5m adjusted aggregates | `C:CLc1` continuous WTI futures |
| XLF | `XLF` ETF 5m adjusted aggregates | Manual vendor / IBKR / Alpaca if Polygon unavailable |
| XLY | `XLY` ETF 5m adjusted aggregates | Manual vendor / IBKR / Alpaca if Polygon unavailable |

## Acquisition Log

| Leader | Source used | Raw file saved | Date range acquired | Quality checks | PASS/FAIL |
|---|---|---|---|---|---|
| `UNG` | Unavailable | None | None | Not run | FAIL - no Polygon key and `XNGUSD` bi5 unavailable |
| `USO` | Unavailable | None | None | Not run | FAIL - no Polygon key and `XBRUSD` bi5 not acquired |
| `XLF` | Unavailable | None | None | Not run | FAIL - no Polygon key; yfinance 5m remains capped at ~60d |
| `XLY` | Unavailable | None | None | Not run | FAIL - no Polygon key; yfinance 5m remains capped at ~60d |

## Alignment Report

No leader parquet files were produced, so no merged XAU/leader alignment grid was created.

| Symbol | Bars on merged grid | Forward-filled bars | Forward-filled % | Earliest common timestamp |
|---|---:|---:|---:|---|
| `UNG` | n/a | n/a | n/a | n/a |
| `USO` | n/a | n/a | n/a | n/a |
| `XLF` | n/a | n/a | n/a | n/a |
| `XLY` | n/a | n/a | n/a | n/a |

Effective study start date could not be computed. The requirement that common aligned coverage start on or before `2023-07-01` is not met.

## Expected Output Files

These files were **not** created because acquisition failed:

- `leader_UNG_5m_20230101_20260521.parquet`
- `leader_USO_5m_20230101_20260521.parquet`
- `leader_XLF_5m_20230101_20260521.parquet`
- `leader_XLY_5m_20230101_20260521.parquet`

## Go / No-Go Decision

**Insufficient leader data acquired. `POLYGON_API_KEY` is not set, and the Tier 1 Dukascopy bi5 probes did not produce usable energy leader data. Manual acquisition required before walk-forward can run. Recommended next source: set `POLYGON_API_KEY` and rerun acquisition through Polygon.io aggregate bars for `UNG`, `USO`, `XLF`, and `XLY`; if Polygon is unavailable, use IBKR TWS API, Alpaca Markets API, or a paid ETF intraday vendor.**
