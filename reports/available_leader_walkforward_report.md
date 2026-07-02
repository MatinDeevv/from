# Available-Leader Dukascopy/Yahoo Walk-Forward Run

Prompt followed: `C:\Users\marti\Downloads\codex_download_and_run_prompt.md`. This run tried all listed Dukascopy bi5 names, downloaded full-range data for the names that probed successfully, then ran the frozen walk-forward with whatever usable leader data existed. The final holdout was not opened because the walk-forward portfolio did not pass.

## Acquisition Log

| Target | Instrument | HTTP status | Valid bi5 | Pass | Error | URL |
| --- | --- | --- | --- | --- | --- | --- |
| USO | USOIL | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/USOIL/2023/00/03/10h_ticks.bi5 |
| USO | XBRUSD | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/XBRUSD/2023/00/03/10h_ticks.bi5 |
| USO | LIGHT.CMD/USD | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/LIGHT.CMD/USD/2023/00/03/10h_ticks.bi5 |
| USO | LIGHTCMDUSD | 200 | True | True |  | https://datafeed.dukascopy.com/datafeed/LIGHTCMDUSD/2023/00/03/10h_ticks.bi5 |
| UNG | NATGAS | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/NATGAS/2023/00/03/10h_ticks.bi5 |
| UNG | XNGUSD | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/XNGUSD/2023/00/03/10h_ticks.bi5 |
| UNG | GAS.CMD/USD | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/GAS.CMD/USD/2023/00/03/10h_ticks.bi5 |
| UNG | GASCMDUSD | 200 | True | True |  | https://datafeed.dukascopy.com/datafeed/GASCMDUSD/2023/00/03/10h_ticks.bi5 |
| XLF | USA500.IDX/USD | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/USA500.IDX/USD/2023/00/03/10h_ticks.bi5 |
| XLF | USA500IDXUSD | 200 | True | True |  | https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2023/00/03/10h_ticks.bi5 |
| XLY | USA500.IDX/USD | 404 | False | False | HTTP Error 404: Not Found | https://datafeed.dukascopy.com/datafeed/USA500.IDX/USD/2023/00/03/10h_ticks.bi5 |
| XLY | USA500IDXUSD | 200 | True | True |  | https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2023/00/03/10h_ticks.bi5 |

Successful probes were fully downloaded with the fast Dukascopy downloader: `LIGHTCMDUSD` for USO/WTI proxy, `GASCMDUSD` for UNG/natural-gas proxy, and `USA500IDXUSD` as the sector index proxy for both XLF and XLY.

## Coverage Table

| Symbol | Source | Start date | End date | Bar count |
| --- | --- | --- | --- | --- |
| XAUUSD | Dukascopy/restored parquet | 2023-01-04 00:00:00+00:00 | 2026-05-22 13:50:00+00:00 | 164744 |
| USO | acquired Dukascopy proxy parquet: C:\Users\marti\from\data\derived\leader_USO_LIGHTCMDUSD_5m_20230101_20260521.parquet | 2023-02-16 23:00:00+00:00 | 2026-05-21 18:55:00+00:00 | 8038 |
| UNG | acquired Dukascopy proxy parquet: C:\Users\marti\from\data\derived\leader_UNG_GASCMDUSD_5m_20230101_20260521.parquet | 2023-01-03 02:00:00+00:00 | 2026-05-20 14:55:00+00:00 | 3792 |
| XLF | acquired Dukascopy proxy parquet: C:\Users\marti\from\data\derived\leader_XLF_USA500IDXUSD_5m_20230101_20260521.parquet | 2023-01-03 03:00:00+00:00 | 2026-05-20 03:55:00+00:00 | 3301 |
| XLY | acquired Dukascopy proxy parquet: C:\Users\marti\from\data\derived\leader_XLY_USA500IDXUSD_5m_20230101_20260521.parquet | 2023-01-03 03:00:00+00:00 | 2026-05-20 03:55:00+00:00 | 3301 |

## Data Quality

| Symbol | File | Bars | Approx expected % | Null close | Gap events >10m | Max gap bars | Spread p50 | Spread p95 | Spread p99 | Quality pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| USO | C:\Users\marti\from\data\derived\leader_USO_LIGHTCMDUSD_5m_20230101_20260521.parquet | 8038 | 12.1236802413273 | 0.0 | 655 | 1464 | 0.05000000000000048 | 0.05550119374960942 | 0.07854202305697099 | False |
| UNG | C:\Users\marti\from\data\derived\leader_UNG_GASCMDUSD_5m_20230101_20260521.parquet | 3792 | 5.719457013574661 | 0.0 | 312 | 15912 | 0.010199732084417363 | 0.010836568346938875 | 0.010974999209211718 | False |
| XLF | C:\Users\marti\from\data\derived\leader_XLF_USA500IDXUSD_5m_20230101_20260521.parquet | 3301 | 4.9788838612368025 | 0.0 | 281 | 13788 | 0.5057319224151299 | 0.7018581613811979 | 0.7049881389349908 | False |
| XLY | C:\Users\marti\from\data\derived\leader_XLY_USA500IDXUSD_5m_20230101_20260521.parquet | 3301 | 4.9788838612368025 | 0.0 | 281 | 13788 | 0.5057319224151299 | 0.7018581613811979 | 0.7049881389349908 | False |

All acquired leader files fail the quality gate due to sparse public archive coverage. The files cover the calendar range in first/last timestamp terms, but not in usable 5m bar density.

## Alignment Report

| Symbol | XAU grid bars | Original leader bars | Bars after 1-bar ffill | Forward-fill bars | Forward-fill % | Usable % on XAU grid | Earliest aligned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USO | 164744 | 8038 | 5839 | 441 | 0.268% | 3.544% | 2023-02-16 23:00:00+00:00 |
| UNG | 164744 | 3792 | 2914 | 230 | 0.140% | 1.769% | 2023-01-10 10:00:00+00:00 |
| XLF | 164744 | 3301 | 2319 | 184 | 0.112% | 1.408% | 2023-01-13 19:00:00+00:00 |
| XLY | 164744 | 3301 | 2319 | 184 | 0.112% | 1.408% | 2023-01-13 19:00:00+00:00 |

Forward-fill percentage is low only because most missing regions remain missing after the one-bar limit. Usable aligned coverage is only 1.4% to 3.5% of the XAU grid, so the walk-forward fold coverage rule rejects every fold.

## Walk-Forward Results

Candidates evaluated: `1296`
Walk-forward validated candidates: `0`

| Symbol | Fold | Reason | Candidate-fold skips |
| --- | --- | --- | --- |
| UNG | 1 | coverage below 80% | 324 |
| UNG | 2 | coverage below 80% | 324 |
| UNG | 3 | coverage below 80% | 324 |
| UNG | 4 | coverage below 80% | 324 |
| UNG | 5 | coverage below 80% | 324 |
| UNG | 6 | coverage below 80% | 324 |
| USO | 1 | coverage below 80% | 324 |
| USO | 2 | coverage below 80% | 324 |
| USO | 3 | coverage below 80% | 324 |
| USO | 4 | coverage below 80% | 324 |
| USO | 5 | coverage below 80% | 324 |
| USO | 6 | coverage below 80% | 324 |
| XLF | 1 | coverage below 80% | 324 |
| XLF | 2 | coverage below 80% | 324 |
| XLF | 3 | coverage below 80% | 324 |
| XLF | 4 | coverage below 80% | 324 |
| XLF | 5 | coverage below 80% | 324 |
| XLF | 6 | coverage below 80% | 324 |
| XLY | 1 | coverage below 80% | 324 |
| XLY | 2 | coverage below 80% | 324 |
| XLY | 3 | coverage below 80% | 324 |
| XLY | 4 | coverage below 80% | 324 |
| XLY | 5 | coverage below 80% | 324 |
| XLY | 6 | coverage below 80% | 324 |

### Validated Candidates

None. No candidate passed at least 4 of 6 folds; in fact every candidate-fold was skipped for coverage below 80%.

## Portfolio Verdict

| Portfolio pass | Families represented | Allocator swing | Effective bets | Reason |
| --- | --- | --- | --- | --- |
| False | none | None | 0.0 | No validated cross-family portfolio |

## Final Holdout

Final holdout was not opened. Walk-forward did not meet acceptance criteria.

## Conclusion

Ran with what exists. Corrected Dukascopy names did produce files, but the public archive coverage is too sparse for the frozen walk-forward: USO proxy has 8,038 bars, UNG proxy has 3,792 bars, and the USA500 sector proxy has 3,301 bars over a period that should contain roughly 66k regular-session 5m bars. All walk-forward folds fail the 80% coverage requirement, so there are zero validated candidates, no valid portfolio, and the final holdout remains sealed.

Machine-readable outputs: `C:\Users\marti\from\reports\available_leader_walkforward_results.json` and `C:\Users\marti\from\reports\available_leader_quality_alignment.json`.