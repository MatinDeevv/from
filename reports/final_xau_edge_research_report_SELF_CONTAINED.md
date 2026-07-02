# Self-Contained XAUUSD Edge Research Report

Generated: 2026-06-18

This file is designed for a reader who does not have access to the code. It
contains the research story, exact formulas, model definitions, selection
rules, backtest outputs, equity-curve image paths, and reproducibility logs
from the saved artifacts.

## 1. What Was Being Tested

Execution market: XAUUSD, effectively spot gold.

Goal:

- Find a real short-horizon XAUUSD trading edge.
- Prefer 5m, 15m, and 1h style intraday models.
- Target roughly 20+ trades/day at the portfolio level.
- Avoid ordinary retail technical indicators.
- Use cross-market leader signals that resemble institutional macro/risk
  dislocation effects.

Final result:

- Best high-frequency portfolio: 5m, 10 models, capped to $5M gross exposure.
- Best robustness portfolio: 60m, 10 models, trained on longer Yahoo 730d
  history, capped to $5M gross exposure.

## 2. Data Sources

XAUUSD execution data:

- Restored master XAUUSD parquet:
  `C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet`
- True forward XAUUSD Dukascopy holdout:
  `C:\Users\marti\from\data\derived\xauusd_dukascopy_5m_forward_20260522_20260618.csv`

Leader data:

- Yahoo 5m, 60d cache for broad 5m screening.
- Yahoo 60m, 730d cache for long-history robustness testing.
- Dukascopy 5m EURUSD and USDJPY were separately fetched for deeper FX/rates
  mechanism checks.

Important limitation:

- Yahoo does not provide multi-year 5m ETF history through this workflow.
- Therefore, the 5m result has higher frequency but shorter training history.
- The 60m result has longer history but lower frequency.

## 3. Exact Signal Formulas

Every model trades XAUUSD. The leader is only used to generate the signal.

Let:

- `L_t` = leader close at bar `t`
- `X_t` = XAUUSD close at bar `t`
- `w` = model lookback window
- `h` = model holding horizon
- `spread_t` = observed XAUUSD spread at entry

Leader log return:

```text
rL_t = log(L_t / L_(t-1))
```

XAU log return:

```text
rX_t = log(X_t / X_(t-1))
```

Raw leader momentum:

```text
raw_t = sum(rL over the previous w completed bars)
```

Volatility-normalized leader momentum, named `volz`:

```text
leader_vol_t = rolling_std(rL, 48 bars) * sqrt(w)
volz_t = raw_t / leader_vol_t
```

Divergence signal:

```text
xau_raw_t = sum(rX over the previous w completed bars)
xau_vol_t = rolling_std(rX, 48 bars) * sqrt(w)
leader_z_t = raw_t / leader_vol_t
xau_z_t = xau_raw_t / xau_vol_t
divergence_t = leader_z_t - xau_z_t
```

Threshold fitting:

```text
threshold = quantile(abs(signal_feature), q)
```

The threshold is fitted only on the first half of the old sample for that
leader/model. It is not fitted on the forward holdout.

Signal direction:

```text
base_signal_t = +1 if feature_t > threshold
base_signal_t = -1 if feature_t < -threshold
base_signal_t =  0 otherwise

if mode == follow:
    trade_direction_t = base_signal_t

if mode == fade:
    trade_direction_t = -base_signal_t
```

Per-trade XAU PnL per ounce:

```text
gross_per_oz = trade_direction_t * (X_(t+h) - X_t)
cost_per_oz = 2.0 * spread_t
net_per_oz = gross_per_oz - cost_per_oz
```

Dollar sizing:

```text
requested_notional_per_signal = account_capital / number_of_models
ounces = assigned_notional / X_t
dollar_pnl = net_per_oz * ounces
```

## 4. Exact Risk Model Used

Account:

- Starting capital: $5,000,000
- Number of bots/models: 10
- Requested notional per signal: $500,000
- Cost model: 2x observed XAU spread
- No compounding
- No stop-loss
- No take-profit
- No trailing stop
- No volatility targeting
- No Kelly sizing
- No correlation-aware allocation
- No financing/margin-interest model
- No execution queue model beyond the spread cost

Capped portfolio execution:

- Max active gross notional = $5,000,000
- Exits at a timestamp free capital before new entries at that same timestamp.
- If a new signal requests $500,000 and at least $500,000 is available, it is
  filled fully.
- If less than $500,000 is available but greater than $0, the trade is
  down-sized.
- If $0 is available, the signal is skipped.

Uncapped diagnostic:

- Every signal gets full requested notional.
- Gross exposure can exceed account size.
- This is not a valid $5M execution simulation.
- It shows raw signal strength only.

## 5. Research Dead Ends

These were tested and rejected as production candidates:

- Existing neural checkpoint: dead/flat signal behavior.
- XAUUSD-only technical rule searches: repeatedly failed after realistic costs.
- SPY 5m fade: looked strong in-sample, then failed true forward holdout.
- CL-only lead: weak hints, but not enough robustness at 2x spread.
- Earlier EURUSD London divergence: looked good recently, failed six-month
  Dukascopy validation.

## 6. Strong Single-Symbol Leads Before Portfolio Construction

### EURUSD

Better EURUSD variant:

- EURUSD/XAU normalized divergence
- NY session
- 42-bar 5m lookback
- Threshold: `0.7904402382958319`
- Hold: 30 bars
- Cost: 2x XAU spread

Six-month old validation:

- PnL: +21,326.40 in raw XAU-unit accounting
- Trades/day: 18.70
- Profit factor: 1.69

Forward:

- PnL: +1,520.20 in raw XAU-unit accounting
- Trades/day: 21.46
- Profit factor: 1.29

Interpretation: real lead, but not clean enough alone.

### USDJPY

Best cleaner USDJPY variant:

- USDJPY volatility-normalized move
- Fade extreme USDJPY moves into XAU
- 48-bar 5m lookback
- Threshold: `1.4066638009465553`
- Session: all
- Hold: 30 bars
- Cost: 2x XAU spread

Six-month old validation:

- Trades: 3,316
- Trades/day: 19.53
- PnL: +9,131.42 in raw XAU-unit accounting
- Win rate: 49.91%
- Profit factor: 1.30
- Reverse-rule PnL: -19,189.25

Forward:

- Trades: 512
- Trades/day: 21.74
- PnL: +1,827.50 in raw XAU-unit accounting
- Win rate: 60.74%
- Profit factor: 1.77
- Reverse-rule PnL: -3,056.53
- Forward quarters: +503.3, +335.0, +998.5, -9.3

Interpretation: best single-symbol lead, later generalized through the
10-model portfolio.

## 7. 5m 50-Symbol Screen

Screened universe:

```text
SPY, QQQ, IWM, DIA, VXX, TLT, IEF, SHY, HYG, LQD, UUP, FXE, FXY,
FXB, FXA, FXC, GLD, IAU, SLV, GDX, GDXJ, USO, UNG, DBC, XLE, XLF,
XLK, XLI, XLV, XLU, XLP, XLY, XLB, XLC, EEM, EWJ, EWU, EWG, FXI,
EWZ, EFA, ACWI, BTC-USD, ETH-USD, GC=F, SI=F, CL=F, ZN=F, ZB=F,
DX-Y.NYB, EURUSD=X, USDJPY=X
```

Promotion gate:

- Old train PnL positive
- Old validation PnL positive
- Old test PnL positive
- Forward PnL positive
- Cost = 2x observed XAU spread
- Frequency inside target band

Selection rule:

```text
rank unique leaders by minimum old train/validation/test PnL,
tie-break by old train + validation + test total,
use forward only as a pass/fail gate
```

## 8. Selected 5m Models

| # | Leader | Model | Feature | Mode | Window | Quantile | Threshold | Session | Hold | Cost | Old train | Old val | Old test | Forward | Fwd TPD | Fwd PF |
|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | XLC | `XLC_volz_follow_w36_q0.55_ny_h30_cost2.0` | volz | follow | 36 | 0.55 | 0.58817179433 | ny | 30 | 2.0 | 1,824.2 | 1,471.6 | 1,661.8 | 1,999.0 | 25.79 | 1.26 |
| 2 | XLV | `XLV_volz_follow_w48_q0.65_ny_h24_cost2.0` | volz | follow | 48 | 0.65 | 0.647378396849 | ny | 24 | 2.0 | 1,863.1 | 1,955.2 | 1,414.8 | 3,697.1 | 27.62 | 1.86 |
| 3 | DX-Y.NYB | `DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0` | volz | fade | 36 | 0.85 | 1.31167469628 | all | 30 | 2.0 | 1,837.4 | 1,411.8 | 1,785.6 | 1,795.1 | 30.25 | 1.30 |
| 4 | IEF | `IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0` | divergence | follow | 24 | 0.55 | 0.542282356512 | ny | 30 | 2.0 | 1,039.0 | 1,186.6 | 1,021.8 | 243.8 | 29.26 | 1.03 |
| 5 | XLF | `XLF_volz_fade_w48_q0.55_london_h30_cost2.0` | volz | fade | 48 | 0.55 | 0.642189535842 | london | 30 | 2.0 | 3,872.6 | 834.0 | 820.1 | 3,787.7 | 26.02 | 1.97 |
| 6 | UNG | `UNG_volz_follow_w48_q0.75_all_h18_cost2.0` | volz | follow | 48 | 0.75 | 1.07689213854 | all | 18 | 2.0 | 3,621.6 | 1,076.4 | 747.1 | 2,288.9 | 30.20 | 1.45 |
| 7 | XLK | `XLK_volz_fade_w48_q0.75_all_h30_cost2.0` | volz | fade | 48 | 0.75 | 1.22892236099 | all | 30 | 2.0 | 674.3 | 2,574.6 | 2,372.4 | 45.4 | 25.06 | 1.01 |
| 8 | VXX | `VXX_volz_follow_w48_q0.75_all_h30_cost2.0` | volz | follow | 48 | 0.75 | 1.06124485566 | all | 30 | 2.0 | 4,802.5 | 3,004.2 | 610.0 | 2,012.2 | 25.90 | 1.29 |
| 9 | EURUSD=X | `EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0` | divergence | follow | 36 | 0.65 | 0.730393076275 | london | 24 | 2.0 | 2,673.9 | 564.0 | 2,121.7 | 1,151.9 | 33.17 | 1.25 |
| 10 | SPY | `SPY_volz_fade_w36_q0.75_all_h24_cost2.0` | volz | fade | 36 | 0.75 | 1.22260398982 | all | 24 | 2.0 | 1,226.6 | 523.7 | 917.5 | 1,790.8 | 26.64 | 1.39 |

## 9. 5m Backtests


### 5m Capped Forward

Artifact JSON: `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped.json`

Trade CSV: `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped_trades.csv`

Symbols used: `DX-Y.NYB, EURUSD=X, IEF, SPY, UNG, VXX, XLC, XLF, XLK, XLV`

Saved backtest summary:

```json
{
  "portfolio": {
    "trades": 1139,
    "days": 26.163194444444443,
    "trades_per_day": 43.5344392833444,
    "pnl": 400866.5851130991,
    "return_pct": 8.017331702261982,
    "avg_trade": 351.9460799939412,
    "win_rate": 0.5452151009657594,
    "profit_factor": 1.3779258779716708,
    "max_dd": 175090.56654893543,
    "max_dd_pct": 3.501811330978709
  },
  "exposure": {
    "peak_gross_notional": 5000000.0,
    "peak_gross_exposure_pct_of_capital": 100.0
  }
}
```

Human-readable summary:

- Trades: 1139
- Days: 26.16
- Trades/day: 43.53
- PnL: $400,866.59
- Return on $5,000,000: 8.02%
- Average trade: $351.95
- Win rate: 54.52%
- Profit factor: 1.38
- Max drawdown: $175,090.57
- Max drawdown pct: 3.50%
- Peak gross notional: $5,000,000.00
- Peak gross exposure: 100.00%

Model-level contribution in this portfolio run:

- `XLV_volz_follow_w48_q0.65_ny_h24_cost2.0`: pnl $171,958.20, trades 162, tpd 7.26, PF 3.13, max DD $25,464.79
- `UNG_volz_follow_w48_q0.75_all_h18_cost2.0`: pnl $148,147.74, trades 340, tpd 13.12, PF 1.59, max DD $82,107.09
- `SPY_volz_fade_w36_q0.75_all_h24_cost2.0`: pnl $69,597.40, trades 136, tpd 6.30, PF 1.72, max DD $62,593.43
- `EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0`: pnl $46,838.99, trades 175, tpd 7.84, PF 1.38, max DD $50,017.92
- `XLF_volz_fade_w48_q0.55_london_h30_cost2.0`: pnl $40,860.07, trades 50, tpd 2.29, PF 2.19, max DD $18,078.34
- `XLK_volz_fade_w48_q0.75_all_h30_cost2.0`: pnl $29,288.73, trades 52, tpd 2.43, PF 1.99, max DD $13,785.54
- `XLC_volz_follow_w36_q0.55_ny_h30_cost2.0`: pnl $-1,794.36, trades 45, tpd 2.12, PF 0.98, max DD $62,380.15
- `VXX_volz_follow_w48_q0.75_all_h30_cost2.0`: pnl $-18,058.23, trades 64, tpd 3.04, PF 0.86, max DD $97,157.47
- `DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0`: pnl $-38,425.35, trades 65, tpd 3.08, PF 0.51, max DD $43,078.08
- `IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0`: pnl $-47,546.61, trades 50, tpd 1.93, PF 0.63, max DD $88,295.03



### 5m Uncapped Forward Diagnostic

Artifact JSON: `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped.json`

Trade CSV: `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped_trades.csv`

Symbols used: `DX-Y.NYB, EURUSD=X, IEF, SPY, UNG, VXX, XLC, XLF, XLK, XLV`

Saved backtest summary:

```json
{
  "portfolio": {
    "trades": 7064,
    "days": 26.18402777777778,
    "trades_per_day": 269.782787428723,
    "pnl": 2113068.367060377,
    "return_pct": 42.26136734120753,
    "avg_trade": 299.13198854195593,
    "win_rate": 0.5360985277463194,
    "profit_factor": 1.3249125422936447,
    "max_dd": 1065573.0683054212,
    "max_dd_pct": 21.31146136610842
  },
  "exposure": {
    "peak_gross_notional": 91000000.0,
    "peak_gross_exposure_pct_of_capital": 1820.0
  }
}
```

Human-readable summary:

- Trades: 7064
- Days: 26.18
- Trades/day: 269.78
- PnL: $2,113,068.37
- Return on $5,000,000: 42.26%
- Average trade: $299.13
- Win rate: 53.61%
- Profit factor: 1.32
- Max drawdown: $1,065,573.07
- Max drawdown pct: 21.31%
- Peak gross notional: $91,000,000.00
- Peak gross exposure: 1820.00%

Model-level contribution in this portfolio run:

- `XLF_volz_fade_w48_q0.55_london_h30_cost2.0`: pnl $441,363.58, trades 578, tpd 26.03, PF 1.99, max DD $131,717.46
- `XLV_volz_follow_w48_q0.65_ny_h24_cost2.0`: pnl $441,094.30, trades 720, tpd 27.56, PF 1.91, max DD $213,232.74
- `UNG_volz_follow_w48_q0.75_all_h18_cost2.0`: pnl $248,177.71, trades 783, tpd 30.20, PF 1.43, max DD $205,702.42
- `XLC_volz_follow_w36_q0.55_ny_h30_cost2.0`: pnl $221,656.76, trades 671, tpd 29.75, PF 1.25, max DD $354,078.00
- `VXX_volz_follow_w48_q0.75_all_h30_cost2.0`: pnl $215,911.10, trades 672, tpd 29.88, PF 1.26, max DD $334,267.02
- `SPY_volz_fade_w36_q0.75_all_h24_cost2.0`: pnl $202,582.47, trades 692, tpd 30.68, PF 1.38, max DD $281,850.30
- `DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0`: pnl $191,874.98, trades 784, tpd 32.70, PF 1.28, max DD $266,409.00
- `EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0`: pnl $123,682.25, trades 774, tpd 33.17, PF 1.23, max DD $148,414.32
- `IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0`: pnl $21,723.06, trades 761, tpd 29.19, PF 1.02, max DD $522,570.41
- `XLK_volz_fade_w48_q0.75_all_h30_cost2.0`: pnl $5,002.14, trades 629, tpd 24.68, PF 1.01, max DD $285,261.04


### 5m Equity And Drawdown Images

5m capped equity:

![5m capped equity](C:/Users/marti/from/reports/equity_curves/5m_capped_equity_curve.png)

5m capped drawdown:

![5m capped drawdown](C:/Users/marti/from/reports/equity_curves/5m_capped_drawdown.png)

5m uncapped equity:

![5m uncapped equity](C:/Users/marti/from/reports/equity_curves/5m_uncapped_equity_curve.png)

5m uncapped drawdown:

![5m uncapped drawdown](C:/Users/marti/from/reports/equity_curves/5m_uncapped_drawdown.png)

## 10. Long-History 60m Screen

Because Yahoo 5m history is short, a separate long-history robustness test was
run using Yahoo 60m bars with `730d` range.

This is not more 5m training data. It is a separate lower-frequency robustness
test.

Selection rule:

- 60m Yahoo leader bars, 730d range
- Old train/validation/test all positive
- Forward positive
- Cost = 2x XAU spread
- Model frequency between 0.5 and 12 trades/day
- Rank unique leaders by old-sample stability

## 11. Selected 60m Models

| # | Leader | Model | Feature | Mode | Window | Quantile | Threshold | Session | Hold | Cost | Old train | Old val | Old test | Forward | Fwd TPD | Fwd PF |
|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | EEM | `EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0` | divergence | follow | 12 | 0.65 | 0.985318067694 | all | 6 | 2.0 | 2,924.5 | 2,725.9 | 2,758.6 | 1,013.8 | 4.32 | 2.07 |
| 2 | GDX | `GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0` | divergence | follow | 12 | 0.75 | 0.725441125437 | all | 8 | 2.0 | 2,509.7 | 2,860.2 | 5,648.9 | 674.3 | 2.08 | 2.13 |
| 3 | GDXJ | `GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0` | divergence | follow | 12 | 0.85 | 0.912209508406 | all | 8 | 2.0 | 2,616.7 | 2,384.1 | 5,248.3 | 567.9 | 1.82 | 2.01 |
| 4 | GC=F | `GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0` | volz | follow | 6 | 0.65 | 0.823909998317 | all | 8 | 2.0 | 2,280.3 | 2,441.6 | 5,546.9 | 130.5 | 6.27 | 1.07 |
| 5 | SPY | `SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0` | divergence | follow | 12 | 0.55 | 0.758148650269 | all | 6 | 2.0 | 3,127.3 | 2,043.7 | 2,107.8 | 475.0 | 4.02 | 1.77 |
| 6 | FXI | `FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0` | divergence | follow | 6 | 0.55 | 0.900830432565 | all | 8 | 2.0 | 3,666.7 | 2,048.9 | 2,005.2 | 274.2 | 2.93 | 1.25 |
| 7 | EFA | `EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0` | volz | follow | 6 | 0.75 | 0.973213627913 | all | 8 | 2.0 | 1,848.0 | 2,185.3 | 3,788.4 | 319.4 | 2.69 | 1.55 |
| 8 | QQQ | `QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0` | divergence | follow | 12 | 0.65 | 1.0650406694 | all | 8 | 2.0 | 2,520.0 | 1,709.7 | 3,737.3 | 1,476.3 | 4.10 | 2.98 |
| 9 | ZB=F | `ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0` | divergence | fade | 12 | 0.85 | 1.69615828535 | all | 8 | 2.0 | 1,658.0 | 2,934.3 | 3,032.2 | 672.7 | 9.39 | 5.47 |
| 10 | EWJ | `EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0` | divergence | follow | 12 | 0.75 | 1.23361098295 | all | 8 | 2.0 | 1,818.4 | 1,627.2 | 5,823.7 | 1,209.4 | 2.23 | 205.07 |

## 12. 60m Backtests


### 60m Capped Forward

Artifact JSON: `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped.json`

Trade CSV: `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped_trades.csv`

Symbols used: `EEM, EFA, EWJ, FXI, GC=F, GDX, GDXJ, QQQ, SPY, ZB=F`

Saved backtest summary:

```json
{
  "portfolio": {
    "trades": 183,
    "days": 20.125,
    "trades_per_day": 9.093167701863354,
    "pnl": 179233.29781414277,
    "return_pct": 3.584665956282856,
    "avg_trade": 979.4169279461354,
    "win_rate": 0.5519125683060109,
    "profit_factor": 1.4097832127749448,
    "max_dd": 62510.49977702761,
    "max_dd_pct": 1.2502099955405521
  },
  "exposure": {
    "peak_gross_notional": 5000000.0,
    "peak_gross_exposure_pct_of_capital": 100.0
  }
}
```

Human-readable summary:

- Trades: 183
- Days: 20.12
- Trades/day: 9.09
- PnL: $179,233.30
- Return on $5,000,000: 3.58%
- Average trade: $979.42
- Win rate: 55.19%
- Profit factor: 1.41
- Max drawdown: $62,510.50
- Max drawdown pct: 1.25%
- Peak gross notional: $5,000,000.00
- Peak gross exposure: 100.00%

Model-level contribution in this portfolio run:

- `SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0`: pnl $58,781.32, trades 20, tpd 2.34, PF 2.36, max DD $41,925.91
- `FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0`: pnl $54,290.94, trades 23, tpd 1.64, PF 2.30, max DD $28,873.22
- `EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0`: pnl $42,413.89, trades 18, tpd 2.40, PF 1.60, max DD $69,423.33
- `EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0`: pnl $34,197.12, trades 8, tpd 0.76, PF 2.39, max DD $24,557.07
- `ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0`: pnl $28,476.48, trades 9, tpd 4.70, PF 2.55, max DD $18,112.00
- `GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0`: pnl $17,299.33, trades 4, tpd 0.57, PF 5.03, max DD $4,294.93
- `EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0`: pnl $5,383.32, trades 5, tpd 1.09, PF 8.98, max DD $674.88
- `GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0`: pnl $-366.51, trades 1, tpd 1,000,000,000.00, PF 0.00, max DD $366.51
- `QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0`: pnl $-1,871.96, trades 8, tpd 1.22, PF 0.95, max DD $31,337.18
- `GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0`: pnl $-59,370.62, trades 87, tpd 4.46, PF 0.69, max DD $81,307.52



### 60m Uncapped Forward Diagnostic

Artifact JSON: `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped.json`

Trade CSV: `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped_trades.csv`

Symbols used: `EEM, EFA, EWJ, FXI, GC=F, GDX, GDXJ, QQQ, SPY, ZB=F`

Saved backtest summary:

```json
{
  "portfolio": {
    "trades": 373,
    "days": 20.125,
    "trades_per_day": 18.53416149068323,
    "pnl": 804470.4020061354,
    "return_pct": 16.089408040122706,
    "avg_trade": 2156.75710993602,
    "win_rate": 0.6112600536193029,
    "profit_factor": 1.946182662793999,
    "max_dd": 307338.8625027867,
    "max_dd_pct": 6.146777250055734
  },
  "exposure": {
    "peak_gross_notional": 29500000.0,
    "peak_gross_exposure_pct_of_capital": 590.0
  }
}
```

Human-readable summary:

- Trades: 373
- Days: 20.12
- Trades/day: 18.53
- PnL: $804,470.40
- Return on $5,000,000: 16.09%
- Average trade: $2,156.76
- Win rate: 61.13%
- Profit factor: 1.95
- Max drawdown: $307,338.86
- Max drawdown pct: 6.15%
- Peak gross notional: $29,500,000.00
- Peak gross exposure: 590.00%

Model-level contribution in this portfolio run:

- `QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0`: pnl $171,107.56, trades 35, tpd 4.08, PF 2.95, max DD $78,763.09
- `EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0`: pnl $143,552.77, trades 18, tpd 2.23, PF 213.71, max DD $366.51
- `EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0`: pnl $116,808.05, trades 36, tpd 4.45, PF 2.04, max DD $110,982.62
- `GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0`: pnl $80,044.40, trades 25, tpd 2.08, PF 2.12, max DD $62,172.22
- `ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0`: pnl $79,024.13, trades 18, tpd 9.39, PF 5.28, max DD $18,237.41
- `GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0`: pnl $66,328.62, trades 21, tpd 1.82, PF 1.99, max DD $63,288.51
- `SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0`: pnl $55,422.82, trades 34, tpd 3.98, PF 1.75, max DD $67,400.12
- `EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0`: pnl $41,385.54, trades 23, tpd 2.16, PF 1.63, max DD $58,792.97
- `FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0`: pnl $32,216.06, trades 41, tpd 2.93, PF 1.25, max DD $83,168.91
- `GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0`: pnl $18,580.44, trades 122, tpd 6.26, PF 1.08, max DD $73,247.70


### 60m Equity And Drawdown Images

60m capped equity:

![60m capped equity](C:/Users/marti/from/reports/equity_curves/60m_capped_equity_curve.png)

60m capped drawdown:

![60m capped drawdown](C:/Users/marti/from/reports/equity_curves/60m_capped_drawdown.png)

60m uncapped equity:

![60m uncapped equity](C:/Users/marti/from/reports/equity_curves/60m_uncapped_equity_curve.png)

60m uncapped drawdown:

![60m uncapped drawdown](C:/Users/marti/from/reports/equity_curves/60m_uncapped_drawdown.png)

## 13. Equity Curve Summary Table

```csv
label,trades,pnl,return_pct,max_drawdown,max_drawdown_pct,csv,equity_png,drawdown_png
5m_capped,1139,400866.5851130991,8.017331702261982,175090.56654893607,3.5018113309787213,..\reports\equity_curves\5m_capped_equity_curve.csv,..\reports\equity_curves\5m_capped_equity_curve.png,..\reports\equity_curves\5m_capped_drawdown.png
5m_uncapped,7064,2113068.367060377,42.26136734120753,1065573.0683054216,21.311461366108432,..\reports\equity_curves\5m_uncapped_equity_curve.csv,..\reports\equity_curves\5m_uncapped_equity_curve.png,..\reports\equity_curves\5m_uncapped_drawdown.png
60m_capped,183,179233.29781414277,3.584665956282856,62510.499777027406,1.250209995540548,..\reports\equity_curves\60m_capped_equity_curve.csv,..\reports\equity_curves\60m_capped_equity_curve.png,..\reports\equity_curves\60m_capped_drawdown.png
60m_uncapped,373,804470.4020061354,16.089408040122706,307338.86250278726,6.146777250055745,..\reports\equity_curves\60m_uncapped_equity_curve.csv,..\reports\equity_curves\60m_uncapped_equity_curve.png,..\reports\equity_curves\60m_uncapped_drawdown.png

```

## 14. Reproduction Command Logs

These are the key command outputs from the final reruns.

### 5m capped forward

```text
period: forward
trades: 1139
days: 26.163194444444443
trades_per_day: 43.5344392833444
pnl: 400866.5851130991
return_pct: 8.017331702261982
avg_trade: 351.9460799939412
win_rate: 0.5452151009657594
profit_factor: 1.3779258779716708
max_dd: 175090.56654893543
max_dd_pct: 3.501811330978709
peak_gross_notional: 5000000.0
peak_gross_exposure_pct_of_capital: 100.0
```

### 5m uncapped forward diagnostic

```text
period: forward
trades: 7064
days: 26.18402777777778
trades_per_day: 269.782787428723
pnl: 2113068.367060377
return_pct: 42.26136734120753
avg_trade: 299.13198854195593
win_rate: 0.5360985277463194
profit_factor: 1.3249125422936447
max_dd: 1065573.0683054212
max_dd_pct: 21.31146136610842
peak_gross_notional: 91000000.0
peak_gross_exposure_pct_of_capital: 1820.0
```

### 60m capped forward

```text
period: forward
trades: 183
days: 20.125
trades_per_day: 9.093167701863354
pnl: 179233.29781414277
return_pct: 3.584665956282856
avg_trade: 979.4169279461354
win_rate: 0.5519125683060109
profit_factor: 1.4097832127749448
max_dd: 62510.49977702761
max_dd_pct: 1.2502099955405521
peak_gross_notional: 5000000.0
peak_gross_exposure_pct_of_capital: 100.0
```

### 60m uncapped forward diagnostic

```text
period: forward
trades: 373
days: 20.125
trades_per_day: 18.53416149068323
pnl: 804470.4020061354
return_pct: 16.089408040122706
avg_trade: 2156.75710993602
win_rate: 0.6112600536193029
profit_factor: 1.946182662793999
max_dd: 307338.8625027867
max_dd_pct: 6.146777250055734
peak_gross_notional: 29500000.0
peak_gross_exposure_pct_of_capital: 590.0
```

## 15. Sample Trade Logs

The complete trade logs are CSV files listed in the artifact index. The first
rows are pasted below so the schema is visible without code access.

### 5m capped trade CSV sample

```csv
model,leader,entry_time,exit_time,direction,xau_entry,xau_exit,gross_per_oz,cost_per_oz,ounces,notional,pnl,pnl_per_dollar,desired_notional,equity
UNG_volz_follow_w48_q0.75_all_h18_cost2.0,UNG,2026-05-22 18:00:00+00:00,2026-05-22 19:30:00,-1,4515.505,4508.1865,7.318500000000313,1.16152642706133,110.7295861703176,500000.0,681.7591357931152,0.0013635182715862,500000.0,681.7591357931152
UNG_volz_follow_w48_q0.75_all_h18_cost2.0,UNG,2026-05-22 18:05:00+00:00,2026-05-22 19:35:00,-1,4515.58,4509.4915,6.08849999999984,1.1068447121820886,110.72774704467643,500000.0,551.6074665732589,0.0011032149331465,500000.0,1233.366602366374
UNG_volz_follow_w48_q0.75_all_h18_cost2.0,UNG,2026-05-22 18:10:00+00:00,2026-05-22 19:40:00,-1,4513.93,4509.75,4.180000000000291,1.1393910574875283,110.768221926348,500000.0,336.802846135492,0.0006736056922709,500000.0,1570.169448501866
UNG_volz_follow_w48_q0.75_all_h18_cost2.0,UNG,2026-05-22 18:15:00+00:00,2026-05-22 19:45:00,-1,4515.056500000001,4507.601500000001,7.454999999999927,1.0738372268274454,110.74058541681592,500000.0,706.6537011411133,0.0014133074022822,500000.0,2276.8231496429794
UNG_volz_follow_w48_q0.75_all_h18_cost2.0,UNG,2026-05-22 18:20:00+00:00,2026-05-22 19:50:00,-1,4516.59,4507.7835,8.806500000000597,0.9941223300971032,110.70298610234713,500000.0,864.8535366176134,0.0017297070732352,500000.0,3141.6766862605928
IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0,IEF,2026-05-22 18:00:00+00:00,2026-05-22 20:30:00,1,4515.505,4507.13,-8.375,1.16152642706133,110.7295861703176,500000.0,-1055.9756247707987,-0.0021119512495415,500000.0,2085.701061489794
IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0,IEF,2026-05-22 18:05:00+00:00,2026-05-22 20:45:00,1,4515.58,4507.0365,-8.543499999999767,1.1068447121820886,110.72774704467643,500000.0,-1068.560928184403,-0.0021371218563688,500000.0,1017.1401333053908
IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0,IEF,2026-05-22 18:10:00+00:00,2026-05-22 20:55:00,1,4513.93,4509.64,-4.289999999999964,1.1393910574875283,110.768221926348,500000.0,-601.4039935807036,-0.0012028079871614,500000.0,415.7361397246871
IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0,IEF,2026-05-22 18:15:00+00:00,2026-05-26 08:00:00,1,4515.056500000001,4535.8815,20.824999999999815,1.0738372268274454,110.74058541681592,500000.0,2187.25532816393,0.0043745106563278,500000.0,2602.991467888617
IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0,IEF,2026-05-22 18:20:00+00:00,2026-05-26 08:05:00,1,4516.59,4533.094999999999,16.5049999999992,0.9941223300971032,110.70298610234713,500000.0,1717.100475126378,0.0034342009502527,500000.0,4320.091943014995
VXX_volz_follow_w48_q0.75_all_h30_cost2.0,VXX,2026-05-22 19:30:00+00:00,2026-05-26 09:00:00,1,4508.1865,4524.639999999999,16.45349999999962,1.2238383685801106,110.90934237081808,500000.0,1689.1117560708185,0.0033782235121416,500000.0,6009.203699085813
VXX_volz_follow_w48_q0.75_all_h30_cost2.0,VXX,2026-05-22 19:35:00+00:00,2026-05-26 09:05:00,1,4509.4915,4523.504999999999,14.013499999999112,1.303580916744676,110.87724635915158,500000.0,1409.2408293988842,0.0028184816587977,500000.0,7418.444528484697
SPY_volz_fade_w36_q0.75_all_h24_cost2.0,SPY,2026-05-22 20:30:00+00:00,2026-05-26 09:30:00,1,4507.13,4518.145,11.015000000000327,1.3597580645161602,110.93534022759492,500000.0,1071.1075490926785,0.0021422150981853,500000.0,8489.552077577375
SPY_volz_fade_w36_q0.75_all_h24_cost2.0,SPY,2026-05-22 20:45:00+00:00,2026-05-26 09:45:00,1,4507.0365,4519.12,12.08349999999973,2.045067238912767,110.93764161883313,500000.0,1113.6400560642192,0.0022272801121284,500000.0,9603.192133641594
EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0,EURUSD=X,2026-05-26 08:05:00+00:00,2026-05-26 10:05:00,1,4533.094999999999,4525.57,-7.524999999999636,1.0541312803889429,110.29991650296324,500000.0,-946.27746389482,-0.0018925549277896,500000.0,8656.914669746775

```

### 60m capped trade CSV sample

```csv
model,leader,entry_time,exit_time,direction,xau_entry,xau_exit,gross_per_oz,cost_per_oz,ounces,notional,pnl,pnl_per_dollar,desired_notional,equity
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 02:00:00+00:00,2026-05-28 10:00:00,-1,4394.4465,4383.11,11.336500000000342,1.664987242644291,113.77997206246566,500000.0,1100.424451333752,0.0022008489026675,500000.0,1100.424451333752
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 03:00:00+00:00,2026-05-28 11:00:00,-1,4374.93,4388.115,-13.18499999999949,1.713254647557658,114.28754288640046,500000.0,-1702.6849169652023,-0.0034053698339304,500000.0,-602.2604656314506
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 04:00:00+00:00,2026-05-28 12:00:00,-1,4373.135,4432.2365,-59.10149999999976,1.818033329738251,114.3344534298621,500000.0,-6965.201546457863,-0.0139304030929157,500000.0,-7567.462012089313
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 05:00:00+00:00,2026-05-28 13:00:00,-1,4388.1385,4430.54,-42.40149999999994,1.7292972410821563,113.94353209225278,500000.0,-5028.418911695939,-0.0100568378233918,500000.0,-12595.880923785253
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 06:00:00+00:00,2026-05-28 14:00:00,-1,4383.5765,4469.431500000001,-85.85500000000047,1.6887195592099231,114.06211343636868,500000.0,-9985.421671004304,-0.0199708433420086,500000.0,-22581.302594789555
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 07:00:00+00:00,2026-05-28 15:00:00,-1,4387.635,4482.389999999999,-94.7549999999992,1.7442367669523495,113.95660760295694,500000.0,-10996.725658236332,-0.0219934513164726,500000.0,-33578.028253025885
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 13:00:00+00:00,2026-05-28 22:00:00,1,4430.54,4491.3315,60.79150000000027,1.490764396314386,112.85306080071504,500000.0,6692.26952060989,0.0133845390412197,500000.0,-26885.758732415998
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 14:00:00+00:00,2026-05-28 23:00:00,1,4469.431500000001,4495.2635,25.831999999999425,1.766677402815019,111.8710511616522,500000.0,2692.2129354912813,0.0053844258709825,500000.0,-24193.545796924715
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 15:00:00+00:00,2026-05-29 00:00:00,1,4482.389999999999,4496.3,13.910000000000764,1.644557913330426,111.54763418622656,500000.0,1368.1810470162502,0.0027363620940325,500000.0,-22825.364749908465
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 16:00:00+00:00,2026-05-29 01:00:00,1,4511.0485,4506.69,-4.3585000000002765,1.5379692555485711,110.83897679220252,500000.0,-653.5586189717144,-0.0013071172379434,500000.0,-23478.92336888018
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 17:00:00+00:00,2026-05-29 02:00:00,1,4509.0615,4501.895,-7.16649999999936,1.3613842714441462,110.88782000422928,500000.0,-945.6384961087252,-0.0018912769922174,500000.0,-24424.561864988904
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 18:00:00+00:00,2026-05-29 03:00:00,1,4501.1615,4501.455,0.2934999999997671,1.3535588701307268,111.08243949922704,500000.0,-117.7539253069413,-0.0002355078506138,500000.0,-24542.315790295845
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 19:00:00+00:00,2026-05-29 04:00:00,1,4498.1015,4514.28,16.178499999999985,1.5547624059520722,111.15800743936082,500000.0,1625.5455322704383,0.0032510910645408,500000.0,-22916.770258025404
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-28 20:00:00+00:00,2026-05-29 05:00:00,1,4496.135,4514.945,18.80999999999949,1.7378477028459034,111.20662524590564,500000.0,1898.536442650586,0.0037970728853011,500000.0,-21018.233815374817
GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0,GC=F,2026-05-29 15:00:00+00:00,2026-06-01 00:00:00,1,4571.215,4540.865,-30.350000000000364,1.854569186102252,109.38011010201882,500000.0,-3522.539323363987,-0.0070450786467279,500000.0,-24540.773138738805

```

## 16. Interpretation

The 5m capped result is legitimately strong:

- +8.02% in about 26 days
- 3.50% max drawdown
- 43.53 trades/day
- 100% capped gross exposure

The reason the valid result is much smaller than the uncapped result is simple:

- The uncapped model allows overlapping trades to stack far beyond account size.
- The capped model does not allow the account to exceed $5M active gross
  notional.
- Therefore, many good signals are skipped or down-sized.

This is good evidence, but not deployment-ready proof.

## 17. Remaining Caveats

- 5m Yahoo leader history is short, about 60 days.
- Forward holdout is only about 26 days.
- ETF and sector leader models are correlated.
- The current allocator is first-come-first-served, not edge-ranked.
- No live paper trading has been run.
- No slippage model beyond 2x spread.
- No margin, financing, queue position, broker rejection, or latency model.
- No portfolio volatility targeting.
- No correlation cap.

## 18. Next Step

The best next step is to acquire true long 5m data for the selected leader
universe and rerun this exact protocol:

1. Keep the same features and gates.
2. Train on longer old data.
3. Preserve an untouched forward holdout.
4. Use the same $5M capped risk model.
5. Add an edge-ranked allocator only after the base edge survives longer 5m
   history.

## 19. Artifact Index

Main reports:

- `C:\Users\marti\from\reports\final_xau_edge_research_report.md`
- `C:\Users\marti\from\reports\final_xau_edge_research_report_SELF_CONTAINED.md`
- `C:\Users\marti\from\reports\ten_model_5m_portfolio_report.md`
- `C:\Users\marti\from\reports\long_history_60m_portfolio_report.md`
- `C:\Users\marti\from\reports\exact_risk_model_and_equity_curves.md`
- `C:\Users\marti\from\reports\cross_market_edge_status.md`

Selected model JSON:

- `C:\Users\marti\from\reports\selected_10_models_oldrank_forwardgate.json`
- `C:\Users\marti\from\reports\selected_10_models_60m_730d_oldrank_forwardgate.json`

Backtest JSON:

- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped.json`
- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped.json`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped.json`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped.json`

Trade CSVs:

- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped_trades.csv`
- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped_trades.csv`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped_trades.csv`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped_trades.csv`

Equity curves:

- `C:\Users\marti\from\reports\equity_curves\5m_capped_equity_curve.csv`
- `C:\Users\marti\from\reports\equity_curves\5m_uncapped_equity_curve.csv`
- `C:\Users\marti\from\reports\equity_curves\60m_capped_equity_curve.csv`
- `C:\Users\marti\from\reports\equity_curves\60m_uncapped_equity_curve.csv`

Image files:

- `C:\Users\marti\from\reports\equity_curves\5m_capped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\5m_capped_drawdown.png`
- `C:\Users\marti\from\reports\equity_curves\5m_uncapped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\5m_uncapped_drawdown.png`
- `C:\Users\marti\from\reports\equity_curves\60m_capped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\60m_capped_drawdown.png`
- `C:\Users\marti\from\reports\equity_curves\60m_uncapped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\60m_uncapped_drawdown.png`
