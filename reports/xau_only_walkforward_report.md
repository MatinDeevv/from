# XAUUSD-Only Walk-Forward Validation

Created UTC: `2026-06-18T23:09:42.890487+00:00`

## Verdict

No XAUUSD-only candidate validated. The best candidates passed only 2 of 6 frozen walk-forward folds, below the required 4 of 6. The sealed May 22-Jun 17 2026 holdout was not opened.

## Data

- XAU file: `C:\Users\marti\from\data\derived\duka_XAUUSD_5m_20230101_20260522.parquet`
- Rows: `164,744`
- Start: `2023-01-04 00:00:00+00:00`
- End: `2026-05-22 13:50:00+00:00`

## Search Space

- Features: `xau_volz`, `xau_session_volz`, `xau_momentum_ratio`, `xau_rsi`, `xau_ew_zspread`
- Windows: `4, 6, 9, 12, 18, 24`
- Quantiles: `0.55, 0.65, 0.75`
- Holds: `18, 24, 30, 36, 42, 48` 5-minute bars
- Modes: `follow`, `fade`
- Sessions: `london`, `ny`, `all`
- Candidates evaluated: `3,240`

## Frozen Fold Gate

A fold passed only if PnL > 0, trades >= 50, exact bootstrap 95% lower bound > 0 using 1000 resamples, and no quarter-subperiod contributed more than 75% of fold PnL. A candidate needed at least 4 passing folds out of 6.

## Candidate Pass Distribution

| Passing folds | Candidate count |
|---:|---:|
| 0 | 3,217 |
| 1 | 17 |
| 2 | 6 |

## Fold Gate Outcomes

The runner short-circuited candidates once they could no longer reach 4 of 6 folds, so fold-gate counts are for evaluated folds, not all 19,440 possible candidate-fold pairs.

| Outcome | Evaluated folds |
|---|---:|
| pnl<=0 | 8,986 |
| subperiod_concentration | 715 |
| pass | 29 |
| bootstrap_low<=0 | 19 |

## Top Near-Misses

![Top candidate fold PnL](C:/Users/marti/from/reports/xau_only_top_candidates_fold_pnl.png)

### 1. `xau_ew_zspread` w24 q0.75 h48 `fade` `london`

- Passing folds: `2/6`
- Evaluated-fold total PnL: `2,846.56`
- Evaluated-fold trades: `5,702`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 750.56 | 729 | 281.31 | 0.58 | True |
| 2 | 221.18 | 1,171 | n/a | 2.37 | False |
| 3 | 294.58 | 1,421 | n/a | 2.14 | False |
| 4 | 3,318.80 | 1,238 | 2,060.98 | 0.53 | True |
| 5 | -1,738.56 | 1,143 | n/a | n/a | False |

### 2. `xau_ew_zspread` w24 q0.75 h42 `fade` `london`

- Passing folds: `2/6`
- Evaluated-fold total PnL: `725.17`
- Evaluated-fold trades: `5,702`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 720.06 | 729 | 292.57 | 0.66 | True |
| 2 | 295.07 | 1,171 | n/a | 2.53 | False |
| 3 | -194.69 | 1,421 | n/a | n/a | False |
| 4 | 2,628.07 | 1,238 | 1,524.28 | 0.38 | True |
| 5 | -2,723.34 | 1,143 | n/a | n/a | False |

### 3. `xau_rsi` w12 q0.75 h42 `fade` `london`

- Passing folds: `2/6`
- Evaluated-fold total PnL: `94.49`
- Evaluated-fold trades: `5,385`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 650.24 | 838 | 198.56 | 0.63 | True |
| 2 | -348.19 | 1,142 | n/a | n/a | False |
| 3 | 433.33 | 1,200 | n/a | 1.86 | False |
| 4 | 1,609.71 | 1,177 | 58.81 | 0.44 | True |
| 5 | -2,250.61 | 1,028 | n/a | n/a | False |

### 4. `xau_rsi` w9 q0.75 h42 `fade` `london`

- Passing folds: `2/6`
- Evaluated-fold total PnL: `-388.03`
- Evaluated-fold trades: `5,438`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 759.72 | 900 | 283.63 | 0.57 | True |
| 2 | -199.40 | 1,116 | n/a | n/a | False |
| 3 | 354.05 | 1,211 | n/a | 1.52 | False |
| 4 | 1,706.00 | 1,157 | 187.24 | 0.72 | True |
| 5 | -3,008.39 | 1,054 | n/a | n/a | False |

### 5. `xau_rsi` w12 q0.65 h42 `fade` `london`

- Passing folds: `2/6`
- Evaluated-fold total PnL: `-635.71`
- Evaluated-fold trades: `7,564`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 594.10 | 1,218 | 49.63 | 0.68 | True |
| 2 | -40.49 | 1,576 | n/a | n/a | False |
| 3 | 142.16 | 1,674 | n/a | 5.94 | False |
| 4 | 2,503.26 | 1,628 | 710.33 | 0.48 | True |
| 5 | -3,834.73 | 1,468 | n/a | n/a | False |

### 6. `xau_ew_zspread` w24 q0.65 h42 `fade` `london`

- Passing folds: `2/6`
- Evaluated-fold total PnL: `-1,530.96`
- Evaluated-fold trades: `7,937`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 833.04 | 1,131 | 262.03 | 0.72 | True |
| 2 | -545.24 | 1,577 | n/a | n/a | False |
| 3 | -397.01 | 1,914 | n/a | n/a | False |
| 4 | 2,796.05 | 1,681 | 1,435.30 | 0.58 | True |
| 5 | -4,217.80 | 1,634 | n/a | n/a | False |

## Final Holdout

- Opened: `False`
- Result: `not opened`

## Background Silver/FX Download Status

The first background run stalled on XAGUSD after repeated Dukascopy 503 responses. It was restarted with 3 workers instead of 12 to reduce throttling. Bar counts should be treated as incomplete until the new background run finishes and writes fresh parquet outputs.

## Files

- Raw JSON: `C:\Users\marti\from\reports\xau_only_walkforward_results.json`
- Diagnostic chart: `C:\Users\marti\from\reports\xau_only_top_candidates_fold_pnl.png`
- Runner: `C:\Users\marti\from\from\tools\run_xau_only_walkforward.py`
- Downloader wrapper: `C:\Users\marti\from\from\tools\download_fx_silver_background.ps1`
