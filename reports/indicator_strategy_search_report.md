# Indicator-Based XAUUSD Strategy Search

Created from result JSON: `C:\Users\marti\from\reports\indicator_strategy_search_results.json`

## Verdict

The indicator-derived search found `28` walk-forward survivors out of `1560` candidates. The sealed May 22-Jun 17 2026 holdout was opened. Holdout was mixed: `8/28` survivors were positive, and `3/28` were both positive and had at least 50 holdout trades.

Best current candidate: `kill_asia_liquidity_sweep`, `fade`, `killzones`, hold `48` bars. It made `291.98` XAU price units on `69` holdout trades, PF `1.84`, about `3.02` trades/day. This is promising but does not satisfy the earlier 20-trades/day target by itself.

## Indicator Sources Used

- `ict.txt`: pivot support/resistance breaks, volume oscillator, wick break variants
- `trend.txt`: ATR-slope pivot trendline breaks
- `smc.txt`: internal/swing BOS, CHoCH, equal highs/lows, FVG create/touch, order-block touch proxy
- `ict2.txt`: MSS/BOS, liquidity, FVG, killzones
- `kill.txt`: Asia/London/NY killzone previous range breaks, session sweeps, opening-price crosses

## Data And Gates

- XAU file: `C:\Users\marti\from\data\derived\duka_XAUUSD_5m_20230101_20260522.parquet`
- Rows: `164,744` from `2023-01-04 00:00:00+00:00` to `2026-05-22 13:50:00+00:00`
- Signals tested: `27`
- Folds: frozen 2023H2, 2024H1, 2024H2, 2025H1, 2025H2, 2026-01-01 to 2026-05-21
- Fold pass gate: PnL > 0, trades >= 50, bootstrap 95% lower > 0 with 1000 resamples, no quarter-subperiod > 75% of fold PnL
- Candidate validation gate: at least 4 of 6 folds pass

## Walk-Forward Survivors By Signal

| Signal | Survivor count |
|---|---:|
| `kill_nyam_liquidity_sweep` | 10 |
| `kill_london_liquidity_sweep` | 5 |
| `kill_asia_liquidity_sweep` | 4 |
| `kill_open_cross_six_open` | 4 |
| `kill_nypm_liquidity_sweep` | 3 |
| `trendline_break` | 1 |
| `kill_asia_prev_range_break` | 1 |

## Holdout Results

![Holdout PnL](C:/Users/marti/from/reports/indicator_strategy_holdout_pnl.png)

| Rank | Signal | Hold | Mode | Session | Trades | TPD | PnL | PF | Sharpe | Max DD |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `kill_asia_liquidity_sweep` | 48 | `fade` | `killzones` | 69 | 3.02 | 291.98 | 1.84 | 1.58 | 182.54 |
| 2 | `kill_asia_liquidity_sweep` | 36 | `fade` | `killzones` | 69 | 3.02 | 290.80 | 1.86 | 1.53 | 123.73 |
| 3 | `kill_asia_liquidity_sweep` | 18 | `fade` | `killzones` | 69 | 3.02 | 234.74 | 2.03 | 1.89 | 78.32 |
| 4 | `kill_nyam_liquidity_sweep` | 6 | `fade` | `all` | 46 | 1.76 | 119.90 | 1.73 | 1.22 | 52.74 |
| 5 | `kill_nyam_liquidity_sweep` | 12 | `fade` | `all` | 46 | 1.76 | 118.63 | 1.57 | 0.92 | 88.90 |
| 6 | `kill_nyam_liquidity_sweep` | 12 | `fade` | `killzones` | 30 | 1.15 | 49.69 | 1.29 | 0.41 | 93.18 |
| 7 | `kill_asia_liquidity_sweep` | 12 | `fade` | `nypm` | 3 | 86.40 | 3.58 | 2.02 | 0.50 | 3.52 |
| 8 | `kill_nyam_liquidity_sweep` | 24 | `fade` | `all` | 46 | 1.76 | 2.08 | 1.00 | 0.01 | 184.04 |
| 9 | `kill_london_liquidity_sweep` | 24 | `fade` | `london` | 4 | 0.99 | -3.66 | 0.82 | -0.15 | 20.64 |
| 10 | `kill_nyam_liquidity_sweep` | 18 | `fade` | `all` | 46 | 1.76 | -16.95 | 0.95 | -0.10 | 147.22 |
| 11 | `kill_open_cross_six_open` | 48 | `fade` | `killzones` | 41 | 1.77 | -17.44 | 0.97 | -0.07 | 115.52 |
| 12 | `kill_nyam_liquidity_sweep` | 24 | `fade` | `killzones` | 30 | 1.15 | -29.02 | 0.91 | -0.18 | 175.50 |
| 13 | `kill_london_liquidity_sweep` | 6 | `fade` | `nypm` | 10 | 0.38 | -34.77 | 0.69 | -0.38 | 106.99 |
| 14 | `kill_nyam_liquidity_sweep` | 36 | `fade` | `killzones` | 30 | 1.15 | -39.01 | 0.89 | -0.22 | 237.52 |
| 15 | `kill_nyam_liquidity_sweep` | 36 | `fade` | `all` | 46 | 1.76 | -39.46 | 0.92 | -0.20 | 240.29 |
| 16 | `kill_nypm_liquidity_sweep` | 12 | `fade` | `nypm` | 6 | 0.40 | -51.30 | 0.20 | -1.06 | 56.22 |
| 17 | `kill_nypm_liquidity_sweep` | 6 | `fade` | `nypm` | 6 | 0.40 | -57.56 | 0.11 | -1.21 | 57.56 |
| 18 | `kill_nyam_liquidity_sweep` | 18 | `fade` | `killzones` | 30 | 1.15 | -57.69 | 0.80 | -0.40 | 143.51 |
| 19 | `kill_open_cross_six_open` | 6 | `fade` | `nypm` | 8 | 0.35 | -63.07 | 0.08 | -1.35 | 64.41 |
| 20 | `kill_london_liquidity_sweep` | 12 | `fade` | `nypm` | 10 | 0.38 | -63.92 | 0.60 | -0.50 | 157.85 |
| 21 | `kill_nyam_liquidity_sweep` | 48 | `fade` | `all` | 46 | 1.76 | -70.61 | 0.87 | -0.33 | 263.16 |
| 22 | `kill_open_cross_six_open` | 6 | `fade` | `killzones` | 41 | 1.77 | -84.71 | 0.68 | -0.77 | 108.88 |
| 23 | `kill_open_cross_six_open` | 18 | `fade` | `killzones` | 41 | 1.77 | -93.64 | 0.79 | -0.51 | 130.62 |
| 24 | `kill_asia_prev_range_break` | 12 | `fade` | `killzones` | 48 | 2.10 | -116.72 | 0.62 | -0.97 | 181.55 |
| 25 | `kill_nypm_liquidity_sweep` | 24 | `fade` | `nypm` | 6 | 0.40 | -125.85 | 0.02 | -1.78 | 125.85 |
| 26 | `kill_london_liquidity_sweep` | 12 | `fade` | `killzones` | 40 | 1.54 | -200.31 | 0.55 | -1.24 | 348.07 |
| 27 | `kill_london_liquidity_sweep` | 18 | `fade` | `killzones` | 40 | 1.54 | -286.37 | 0.47 | -1.66 | 404.01 |
| 28 | `trendline_break` | 48 | `fade` | `all` | 260 | 9.93 | -322.78 | 0.90 | -0.57 | 885.48 |

## Best Holdout Equity

![Best holdout equity](C:/Users/marti/from/reports/indicator_strategy_best_holdout_equity.png)

## Top Walk-Forward Fold Tables

### 1. `kill_nypm_liquidity_sweep` h12 `fade` `nypm`

- Passing folds: `5/6`; evaluated-fold PnL `14,028.39`; trades `352`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 353.89 | 73 | 176.78 | 0.39 | True |
| 2 | 592.55 | 62 | 150.73 | 0.74 | True |
| 3 | 1,679.41 | 59 | 730.01 | 0.66 | True |
| 4 | 3,617.84 | 57 | 2,164.62 | 0.42 | True |
| 5 | 80.86 | 46 | n/a | 0.42 | False |
| 6 | 7,703.83 | 55 | 4,020.30 | 0.74 | True |

### 2. `kill_nypm_liquidity_sweep` h6 `fade` `nypm`

- Passing folds: `5/6`; evaluated-fold PnL `10,626.38`; trades `352`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 363.87 | 73 | 192.69 | 0.38 | True |
| 2 | 211.16 | 62 | 106.38 | 0.41 | True |
| 3 | 1,602.80 | 59 | 674.17 | 0.68 | True |
| 4 | 2,796.78 | 57 | 1,604.10 | 0.47 | True |
| 5 | -21.24 | 46 | n/a | n/a | False |
| 6 | 5,673.01 | 55 | 2,089.60 | 0.64 | True |

### 3. `kill_nyam_liquidity_sweep` h12 `fade` `all`

- Passing folds: `4/6`; evaluated-fold PnL `47,823.59`; trades `3,627`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 1,588.64 | 633 | 1,162.39 | 0.59 | True |
| 2 | 3,700.48 | 828 | 2,722.77 | 0.52 | True |
| 3 | 7,451.75 | 609 | 5,546.23 | 0.57 | True |
| 4 | 27,655.54 | 644 | 23,616.06 | 0.60 | True |
| 5 | 37.83 | 477 | n/a | 6.20 | False |
| 6 | 7,389.36 | 436 | n/a | 1.01 | False |

### 4. `kill_nyam_liquidity_sweep` h24 `fade` `all`

- Passing folds: `4/6`; evaluated-fold PnL `46,359.06`; trades `3,627`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 1,645.27 | 633 | 1,203.59 | 0.63 | True |
| 2 | 3,610.61 | 828 | 2,571.66 | 0.51 | True |
| 3 | 6,889.13 | 609 | 4,727.42 | 0.52 | True |
| 4 | 24,374.16 | 644 | 20,361.56 | 0.68 | True |
| 5 | 399.72 | 477 | n/a | 1.45 | False |
| 6 | 9,440.15 | 436 | n/a | 0.95 | False |

### 5. `kill_nyam_liquidity_sweep` h36 `fade` `all`

- Passing folds: `4/6`; evaluated-fold PnL `46,163.37`; trades `3,627`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 1,827.86 | 633 | 1,355.62 | 0.63 | True |
| 2 | 3,076.76 | 828 | 2,063.96 | 0.51 | True |
| 3 | 6,239.85 | 609 | 4,295.17 | 0.60 | True |
| 4 | 25,491.40 | 644 | 21,155.99 | 0.68 | True |
| 5 | -39.31 | 477 | n/a | n/a | False |
| 6 | 9,566.82 | 436 | n/a | 0.96 | False |

### 6. `kill_nyam_liquidity_sweep` h18 `fade` `all`

- Passing folds: `4/6`; evaluated-fold PnL `45,855.20`; trades `3,627`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 1,769.63 | 633 | 1,320.04 | 0.59 | True |
| 2 | 3,357.09 | 828 | 2,198.22 | 0.48 | True |
| 3 | 6,563.30 | 609 | 4,570.33 | 0.58 | True |
| 4 | 24,804.68 | 644 | 20,650.94 | 0.59 | True |
| 5 | 246.15 | 477 | n/a | 1.30 | False |
| 6 | 9,114.35 | 436 | n/a | 0.96 | False |

### 7. `kill_nyam_liquidity_sweep` h48 `fade` `all`

- Passing folds: `4/6`; evaluated-fold PnL `45,674.41`; trades `3,626`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 1,694.90 | 633 | 1,194.95 | 0.64 | True |
| 2 | 3,460.91 | 828 | 2,308.43 | 0.46 | True |
| 3 | 5,653.27 | 609 | 3,811.59 | 0.69 | True |
| 4 | 25,685.26 | 644 | 21,190.51 | 0.62 | True |
| 5 | -125.56 | 477 | n/a | n/a | False |
| 6 | 9,305.63 | 435 | n/a | 0.96 | False |

### 8. `kill_nyam_liquidity_sweep` h6 `fade` `all`

- Passing folds: `4/6`; evaluated-fold PnL `40,821.97`; trades `3,627`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 1,715.32 | 633 | 1,325.60 | 0.63 | True |
| 2 | 3,140.44 | 828 | 2,360.72 | 0.56 | True |
| 3 | 6,581.69 | 609 | 4,633.36 | 0.61 | True |
| 4 | 24,568.68 | 644 | 20,415.63 | 0.64 | True |
| 5 | -313.74 | 477 | n/a | n/a | False |
| 6 | 5,129.58 | 436 | n/a | 1.08 | False |

### 9. `kill_nyam_liquidity_sweep` h36 `fade` `killzones`

- Passing folds: `4/6`; evaluated-fold PnL `29,792.47`; trades `1,777`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 501.29 | 284 | 239.70 | 0.58 | True |
| 2 | 1,468.14 | 386 | 805.02 | 0.45 | True |
| 3 | 2,874.81 | 309 | 1,334.53 | 0.64 | True |
| 4 | 15,351.92 | 331 | 12,267.56 | 0.58 | True |
| 5 | 269.54 | 240 | -289.55 | 0.68 | False |
| 6 | 9,326.77 | 227 | n/a | 1.00 | False |

### 10. `kill_nyam_liquidity_sweep` h12 `fade` `killzones`

- Passing folds: `4/6`; evaluated-fold PnL `29,294.48`; trades `1,777`

| Fold | PnL | Trades | Bootstrap low | Subperiod max share | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 493.90 | 284 | 250.10 | 0.46 | True |
| 2 | 1,756.24 | 386 | 1,114.46 | 0.41 | True |
| 3 | 4,286.06 | 309 | 2,736.54 | 0.65 | True |
| 4 | 14,894.03 | 331 | 11,832.22 | 0.58 | True |
| 5 | 117.02 | 240 | -280.06 | 0.74 | False |
| 6 | 7,747.23 | 227 | n/a | 1.01 | False |

## Mechanism Read

The strongest group is not a generic retail trend signal. It is a session microstructure pattern: fade prior Asia-session liquidity sweeps during ICT killzones. In plain terms, when price raids the prior Asia high/low and then rejects, the tested edge is mean reversion rather than continuation. London/NY sweep variants validated in-sample too, but many failed or had too few trades in the sealed holdout.

## Caveats

- Current best candidate is below the 20-trades/day target.
- Several validated survivors are duplicate parameterizations of the same killzone sweep idea, so treat them as one family, not 28 independent edges.
- The indicator files are TradingView drawing scripts; this runner implements measurable non-repainting approximations of their concepts, not a byte-for-byte Pine clone.
- PnL is in XAU price units before position sizing. No live trading sizing model is implied by this report.
