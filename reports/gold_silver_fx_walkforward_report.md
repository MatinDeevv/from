# Gold / Silver / FX Dukascopy Walk-Forward Report

Prompt followed: `C:\Users\marti\Downloads\codex_gold_silver_fx_walkforward_prompt.md`. Only native Dukascopy instruments were used. No ETFs, no proxies, no external APIs. Final holdout was not opened because the walk-forward portfolio did not pass.

## Download Log

| Symbol | Probe HTTP | Valid bi5 | Probe pass | Downloaded bars | Start | End | Probe error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XAGUSD | 200 | True | True | 2147 | 2023-01-09 16:00:00+00:00 | 2026-05-21 22:55:00+00:00 |  |
| EURUSD | 200 | True | True | 1805 | 2023-01-13 08:00:00+00:00 | 2026-05-20 03:55:00+00:00 |  |
| USDJPY | 200 | True | True | 2001 | 2023-01-03 17:00:00+00:00 | 2026-05-18 01:55:00+00:00 |  |
| GBPUSD | 200 | True | True | 2136 | 2023-01-02 00:00:00+00:00 | 2026-05-19 18:55:00+00:00 |  |
| USDCHF | 200 | True | True | 2208 | 2023-01-05 18:00:00+00:00 | 2026-05-21 07:55:00+00:00 |  |
| AUDUSD | 200 | True | True | 2073 | 2023-01-02 01:00:00+00:00 | 2026-05-18 01:55:00+00:00 |  |
| NZDUSD | 200 | True | True | 1951 | 2023-01-02 21:25:00+00:00 | 2026-05-19 08:55:00+00:00 |  |
| USDCAD | 200 | True | True | 1934 | 2023-01-03 18:00:00+00:00 | 2026-05-21 06:55:00+00:00 |  |
| EURGBP | 200 | True | True | 2697 | 2023-01-01 23:15:00+00:00 | 2026-05-19 11:55:00+00:00 |  |
| EURJPY | 200 | True | True | 2508 | 2023-01-03 08:00:00+00:00 | 2026-05-19 10:55:00+00:00 |  |
| GBPJPY | 200 | True | True | 2435 | 2023-01-10 10:00:00+00:00 | 2026-05-21 10:55:00+00:00 |  |

All requested symbols probed successfully, and parquet files were created for every requested leader. However, the public archive returned mostly `HTTP_503` during bulk download, so each file contains only a sparse subset of the expected multi-year 5m bars.

## Coverage Table

| Symbol | Source | Start date | End date | Bar count |
| --- | --- | --- | --- | --- |
| XAUUSD | C:\Users\marti\from\data\derived\duka_XAUUSD_5m_20230101_20260522.parquet | 2023-01-04 00:00:00+00:00 | 2026-05-22 13:50:00+00:00 | 164744 |
| XAGUSD | C:\Users\marti\from\data\derived\leader_XAGUSD_5m_20230101_20260521.parquet | 2023-01-09 16:00:00+00:00 | 2026-05-21 22:55:00+00:00 | 2147 |
| EURUSD | C:\Users\marti\from\data\derived\leader_EURUSD_5m_20230101_20260521.parquet | 2023-01-13 08:00:00+00:00 | 2026-05-20 03:55:00+00:00 | 1805 |
| USDJPY | C:\Users\marti\from\data\derived\leader_USDJPY_5m_20230101_20260521.parquet | 2023-01-03 17:00:00+00:00 | 2026-05-18 01:55:00+00:00 | 2001 |
| GBPUSD | C:\Users\marti\from\data\derived\leader_GBPUSD_5m_20230101_20260521.parquet | 2023-01-02 00:00:00+00:00 | 2026-05-19 18:55:00+00:00 | 2136 |
| USDCHF | C:\Users\marti\from\data\derived\leader_USDCHF_5m_20230101_20260521.parquet | 2023-01-05 18:00:00+00:00 | 2026-05-21 07:55:00+00:00 | 2208 |
| AUDUSD | C:\Users\marti\from\data\derived\leader_AUDUSD_5m_20230101_20260521.parquet | 2023-01-02 01:00:00+00:00 | 2026-05-18 01:55:00+00:00 | 2073 |
| NZDUSD | C:\Users\marti\from\data\derived\leader_NZDUSD_5m_20230101_20260521.parquet | 2023-01-02 21:25:00+00:00 | 2026-05-19 08:55:00+00:00 | 1951 |
| USDCAD | C:\Users\marti\from\data\derived\leader_USDCAD_5m_20230101_20260521.parquet | 2023-01-03 18:00:00+00:00 | 2026-05-21 06:55:00+00:00 | 1934 |
| EURGBP | C:\Users\marti\from\data\derived\leader_EURGBP_5m_20230101_20260521.parquet | 2023-01-01 23:15:00+00:00 | 2026-05-19 11:55:00+00:00 | 2697 |
| EURJPY | C:\Users\marti\from\data\derived\leader_EURJPY_5m_20230101_20260521.parquet | 2023-01-03 08:00:00+00:00 | 2026-05-19 10:55:00+00:00 | 2508 |
| GBPJPY | C:\Users\marti\from\data\derived\leader_GBPJPY_5m_20230101_20260521.parquet | 2023-01-10 10:00:00+00:00 | 2026-05-21 10:55:00+00:00 | 2435 |

## Quality Snapshot

| Symbol | Bars | Start | End | Gap events >10m | Max gap in 5m bars |
| --- | --- | --- | --- | --- | --- |
| XAGUSD | 2147 | 2023-01-09 16:00:00+00:00 | 2026-05-21 22:55:00+00:00 | 178 | 8412.0 |
| EURUSD | 1805 | 2023-01-13 08:00:00+00:00 | 2026-05-20 03:55:00+00:00 | 151 | 13920.0 |
| USDJPY | 2001 | 2023-01-03 17:00:00+00:00 | 2026-05-18 01:55:00+00:00 | 167 | 10692.0 |
| GBPUSD | 2136 | 2023-01-02 00:00:00+00:00 | 2026-05-19 18:55:00+00:00 | 176 | 7500.0 |
| USDCHF | 2208 | 2023-01-05 18:00:00+00:00 | 2026-05-21 07:55:00+00:00 | 183 | 7320.0 |
| AUDUSD | 2073 | 2023-01-02 01:00:00+00:00 | 2026-05-18 01:55:00+00:00 | 169 | 11952.0 |
| NZDUSD | 1951 | 2023-01-02 21:25:00+00:00 | 2026-05-19 08:55:00+00:00 | 162 | 8868.0 |
| USDCAD | 1934 | 2023-01-03 18:00:00+00:00 | 2026-05-21 06:55:00+00:00 | 161 | 12156.0 |
| EURGBP | 2697 | 2023-01-01 23:15:00+00:00 | 2026-05-19 11:55:00+00:00 | 223 | 6240.0 |
| EURJPY | 2508 | 2023-01-03 08:00:00+00:00 | 2026-05-19 10:55:00+00:00 | 208 | 10536.0 |
| GBPJPY | 2435 | 2023-01-10 10:00:00+00:00 | 2026-05-21 10:55:00+00:00 | 201 | 15912.0 |

## Walk-Forward Results

Candidates evaluated: `35640`
Walk-forward validated candidates: `0`

No candidates validated. No candidate passed at least 4 of 6 folds.

### Fold Skip Summary

| Symbol | Fold | Reason | Candidate-fold skips |
| --- | --- | --- | --- |
| AUDUSD | 1 | coverage below 80% | 3240 |
| AUDUSD | 2 | coverage below 80% | 3240 |
| AUDUSD | 3 | coverage below 80% | 3240 |
| AUDUSD | 4 | coverage below 80% | 3240 |
| AUDUSD | 5 | coverage below 80% | 3240 |
| AUDUSD | 6 | coverage below 80% | 3240 |
| EURGBP | 1 | coverage below 80% | 3240 |
| EURGBP | 2 | coverage below 80% | 3240 |
| EURGBP | 3 | coverage below 80% | 3240 |
| EURGBP | 4 | coverage below 80% | 3240 |
| EURGBP | 5 | coverage below 80% | 3240 |
| EURGBP | 6 | coverage below 80% | 3240 |
| EURJPY | 1 | coverage below 80% | 3240 |
| EURJPY | 2 | coverage below 80% | 3240 |
| EURJPY | 3 | coverage below 80% | 3240 |
| EURJPY | 4 | coverage below 80% | 3240 |
| EURJPY | 5 | coverage below 80% | 3240 |
| EURJPY | 6 | coverage below 80% | 3240 |
| EURUSD | 1 | coverage below 80% | 3240 |
| EURUSD | 2 | coverage below 80% | 3240 |
| EURUSD | 3 | coverage below 80% | 3240 |
| EURUSD | 4 | coverage below 80% | 3240 |
| EURUSD | 5 | coverage below 80% | 3240 |
| EURUSD | 6 | coverage below 80% | 3240 |
| GBPJPY | 1 | coverage below 80% | 3240 |
| GBPJPY | 2 | coverage below 80% | 3240 |
| GBPJPY | 3 | coverage below 80% | 3240 |
| GBPJPY | 4 | coverage below 80% | 3240 |
| GBPJPY | 5 | coverage below 80% | 3240 |
| GBPJPY | 6 | coverage below 80% | 3240 |
| GBPUSD | 1 | coverage below 80% | 3240 |
| GBPUSD | 2 | coverage below 80% | 3240 |
| GBPUSD | 3 | coverage below 80% | 3240 |
| GBPUSD | 4 | coverage below 80% | 3240 |
| GBPUSD | 5 | coverage below 80% | 3240 |
| GBPUSD | 6 | coverage below 80% | 3240 |
| NZDUSD | 1 | coverage below 80% | 3240 |
| NZDUSD | 2 | coverage below 80% | 3240 |
| NZDUSD | 3 | coverage below 80% | 3240 |
| NZDUSD | 4 | coverage below 80% | 3240 |
| NZDUSD | 5 | coverage below 80% | 3240 |
| NZDUSD | 6 | coverage below 80% | 3240 |
| USDCAD | 1 | coverage below 80% | 3240 |
| USDCAD | 2 | coverage below 80% | 3240 |
| USDCAD | 3 | coverage below 80% | 3240 |
| USDCAD | 4 | coverage below 80% | 3240 |
| USDCAD | 5 | coverage below 80% | 3240 |
| USDCAD | 6 | coverage below 80% | 3240 |
| USDCHF | 1 | coverage below 80% | 3240 |
| USDCHF | 2 | coverage below 80% | 3240 |
| USDCHF | 3 | coverage below 80% | 3240 |
| USDCHF | 4 | coverage below 80% | 3240 |
| USDCHF | 5 | coverage below 80% | 3240 |
| USDCHF | 6 | coverage below 80% | 3240 |
| USDJPY | 1 | coverage below 80% | 3240 |
| USDJPY | 2 | coverage below 80% | 3240 |
| USDJPY | 3 | coverage below 80% | 3240 |
| USDJPY | 4 | coverage below 80% | 3240 |
| USDJPY | 5 | coverage below 80% | 3240 |
| USDJPY | 6 | coverage below 80% | 3240 |
| XAGUSD | 1 | coverage below 80% | 3240 |
| XAGUSD | 2 | coverage below 80% | 3240 |
| XAGUSD | 3 | coverage below 80% | 3240 |
| XAGUSD | 4 | coverage below 80% | 3240 |
| XAGUSD | 5 | coverage below 80% | 3240 |
| XAGUSD | 6 | coverage below 80% | 3240 |

Every candidate-fold was skipped for coverage below 80%. This is a data-density failure, not a profitable/unprofitable strategy result.

## Portfolio Verdict

| Portfolio pass | Families represented | Allocator swing | Effective bets | Reason |
| --- | --- | --- | --- | --- |
| False | none | None | 0.0 | No validated metals+FX portfolio |

## Final Holdout

Final holdout was not opened. Walk-forward did not meet acceptance criteria.

## Conclusion

Downloaded and ran with what Dukascopy returned. The native symbols are valid, but bulk archive coverage is too sparse in this environment: each leader has roughly 1.8k to 2.7k bars over a period that should contain hundreds of thousands of continuous 5m FX bars. The frozen walk-forward therefore has zero validated candidates, no valid metals+FX portfolio, and the final holdout remains sealed.

Machine-readable output: `C:\Users\marti\from\reports\gold_silver_fx_walkforward_results.json`.