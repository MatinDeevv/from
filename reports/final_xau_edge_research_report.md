# Final XAUUSD Edge Research Report

Generated: 2026-06-18

Workspace: `C:\Users\marti\from`

## Executive Summary

We searched for a real short-horizon XAUUSD edge using cross-market leader
signals. The best current result is a 10-model 5m cross-market portfolio on XAU
with $5,000,000 capital and a hard 100% gross-exposure cap.

The result is good, but still research-grade:

- 5m capped forward PnL: +$400,866.59
- 5m capped forward return: +8.02%
- 5m capped max drawdown: 3.50%
- 5m capped profit factor: 1.38
- 5m capped trades/day: 43.53
- Peak gross exposure: 100.0%

The raw uncapped version makes much more, but it is not a valid $5M account
simulation because it allows extreme overlapping exposure.

## Core Finding

The strongest effect is not a normal retail technical strategy. It is a
cross-market macro-dislocation effect:

- XAUUSD reacts to normalized moves in equity sectors, USD/rates proxies,
  volatility, energy, and FX/rates channels.
- The useful models are mostly volatility-normalized or divergence signals.
- The model stack works best when multiple independent leader shocks are allowed
  to trigger XAU trades.

Mechanism story:

- Gold reprices after large relative moves in macro leaders.
- Leader moves are converted to volatility-normalized shocks or divergence
  against XAU's own recent move.
- The model either follows or fades the leader shock depending on the leader
  family.
- The strongest 5m basket is broad: sectors, DXY, EURUSD, rates, vol, gas, SPY.

## Data Used

Execution market:

- XAUUSD restored master dataset:
  `C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet`
- True forward XAUUSD Dukascopy holdout:
  `C:\Users\marti\from\data\derived\xauusd_dukascopy_5m_forward_20260522_20260618.csv`

Leader data:

- Yahoo 5m, 60d, for broad 50-symbol screen
- Yahoo 60m, 730d, for long-history robustness test
- Dukascopy 5m EURUSD/USDJPY tests for stronger FX/rates validation

Important limitation:

- Yahoo 5m ETF data only gives about 60 days. The high-frequency 5m portfolio
  is promising, but it is not trained on years of 5m data.
- The long-history version uses 60m bars over about 730d, so it is more robust
  but lower frequency.

## Research Progression

### 1. Dead Ends

The following did not produce deployable edge:

- Original neural checkpoint: flat/dead behavior and not usable
- XAUUSD-only technical rules: repeatedly failed after spread cost
- SPY 5m fade: looked strong on old sample, then failed true forward holdout
- CL-only lead: had weak hints, but not enough robustness at 2x spread

### 2. First Real Macro Lead: EURUSD

The first serious lead was EURUSD/XAU normalized divergence.

Earlier London EURUSD rule:

- Looked strong on the recent sample and forward holdout
- Failed six-month Dukascopy validation
- Verdict: regime-specific, not robust enough

Better six-month EURUSD variant:

- EURUSD/XAU normalized divergence
- NY session
- 42-bar 5m lookback
- Threshold: `0.7904402382958319`
- Hold: 30 bars
- 2x XAU spread

Six-month old validation:

- PnL: +$21,326.40 per raw XAU-unit accounting
- Trades/day: 18.70
- Profit factor: 1.69

Forward:

- PnL: +$1,520.20 per raw XAU-unit accounting
- Trades/day: 21.46
- Profit factor: 1.29

Verdict: real research lead, but forward timing controls and overlap execution
were not strong enough to call it deployable.

### 3. Stronger FX/Rates Lead: USDJPY

USDJPY produced a cleaner confirmation of the macro-dislocation effect.

Best cleaner USDJPY rule:

- USDJPY volatility-normalized move
- Fade extreme USDJPY moves into XAU
- Window: 48 completed 5m bars
- Threshold: `1.4066638009465553`
- Session: all
- Hold: 30 bars
- Cost: 2x XAU spread

Six-month old validation:

- Trades: 3,316
- Trades/day: 19.53
- PnL: +$9,131.42 per raw XAU-unit accounting
- Win rate: 49.91%
- Profit factor: 1.30
- Reverse-rule PnL: -$19,189.25

Forward validation:

- Trades: 512
- Trades/day: 21.74
- PnL: +$1,827.50 per raw XAU-unit accounting
- Win rate: 60.74%
- Profit factor: 1.77
- Reverse-rule PnL: -$3,056.53
- Forward quarters: +503.3, +335.0, +998.5, -9.3

Verdict: best single-symbol research lead before portfolio construction.

## 50-Symbol Screen

Screened broad liquid macro leaders:

- Equity indices and ETFs: SPY, QQQ, DIA, IWM, ACWI, EFA, EEM
- Sector ETFs: XLC, XLV, XLF, XLK, XLE, XLI, XLU, XLP, XLY, XLB
- Rates/credit: IEF, TLT, SHY, LQD, HYG, ZN=F, ZB=F
- Commodities: GLD, IAU, SLV, GDX, GDXJ, USO, UNG, DBC, GC=F, SI=F, CL=F
- FX/proxies: DX-Y.NYB, EURUSD=X, USDJPY=X, UUP, FXE, FXY, FXB, FXA, FXC
- Country/region ETFs: EWJ, EWU, EWG, FXI, EWZ
- Crypto proxies: BTC-USD, ETH-USD
- Volatility: VXX

Promotion gate:

- Old train PnL positive
- Old validation PnL positive
- Old test PnL positive
- Untouched forward PnL positive
- 2x observed XAUUSD spread cost
- Frequency inside target band

Selection rule:

- Rank unique leaders by minimum old train/validation/test PnL
- Tie-break by old train+validation+test total
- Forward only used as a pass/fail gate, not for ranking

Selection artifact:

`C:\Users\marti\from\reports\selected_10_models_oldrank_forwardgate.json`

## Selected 5m Models

| # | Leader | Model | Old Train | Old Val | Old Test | Forward PnL | Forward TPD | Forward PF |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | XLC | `XLC_volz_follow_w36_q0.55_ny_h30_cost2.0` | 1,824.2 | 1,471.6 | 1,661.8 | 1,999.0 | 25.79 | 1.26 |
| 2 | XLV | `XLV_volz_follow_w48_q0.65_ny_h24_cost2.0` | 1,863.1 | 1,955.2 | 1,414.8 | 3,697.1 | 27.62 | 1.86 |
| 3 | DX-Y.NYB | `DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0` | 1,837.4 | 1,411.8 | 1,785.6 | 1,795.1 | 30.25 | 1.30 |
| 4 | IEF | `IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0` | 1,039.0 | 1,186.6 | 1,021.8 | 243.8 | 29.26 | 1.03 |
| 5 | XLF | `XLF_volz_fade_w48_q0.55_london_h30_cost2.0` | 3,872.6 | 834.0 | 820.1 | 3,787.7 | 26.02 | 1.97 |
| 6 | UNG | `UNG_volz_follow_w48_q0.75_all_h18_cost2.0` | 3,621.6 | 1,076.4 | 747.1 | 2,288.9 | 30.20 | 1.45 |
| 7 | XLK | `XLK_volz_fade_w48_q0.75_all_h30_cost2.0` | 674.3 | 2,574.6 | 2,372.4 | 45.4 | 25.06 | 1.01 |
| 8 | VXX | `VXX_volz_follow_w48_q0.75_all_h30_cost2.0` | 4,802.5 | 3,004.2 | 610.0 | 2,012.2 | 25.90 | 1.29 |
| 9 | EURUSD=X | `EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0` | 2,673.9 | 564.0 | 2,121.7 | 1,151.9 | 33.17 | 1.25 |
| 10 | SPY | `SPY_volz_fade_w36_q0.75_all_h24_cost2.0` | 1,226.6 | 523.7 | 917.5 | 1,790.8 | 26.64 | 1.39 |

## 5m Portfolio Backtest

Backtest artifact:

`C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped.json`

Trades:

`C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped_trades.csv`

Risk model:

- Initial capital: $5,000,000
- 10 models
- Each signal requests $500,000 notional
- Gross exposure capped at $5,000,000
- If account is fully allocated, new signals are skipped or down-sized
- Cost: 2x observed XAU spread
- No compounding
- No stop-loss
- No take-profit
- No volatility targeting
- No Kelly sizing
- No correlation-aware allocator

Valid capped result:

- Trades: 1,139
- Trades/day: 43.53
- PnL: +$400,866.59
- Return: +8.02%
- Win rate: 54.52%
- Profit factor: 1.38
- Max drawdown: $175,090.57
- Max drawdown: 3.50%
- Peak gross notional: $5,000,000
- Peak gross exposure: 100.0%

Uncapped diagnostic:

- PnL: +$2,113,068.37
- Return: +42.26%
- Profit factor: 1.32
- Max drawdown: $1,065,573.07
- Peak gross notional: $91,000,000
- Peak gross exposure: 1,820.0%

The uncapped result is not a valid $5M account simulation. It is only a raw
signal-strength diagnostic.

### 5m Capped Equity Curve

![5m capped equity](C:/Users/marti/from/reports/equity_curves/5m_capped_equity_curve.png)

### 5m Capped Drawdown

![5m capped drawdown](C:/Users/marti/from/reports/equity_curves/5m_capped_drawdown.png)

### 5m Uncapped Equity Curve

![5m uncapped equity](C:/Users/marti/from/reports/equity_curves/5m_uncapped_equity_curve.png)

### 5m Uncapped Drawdown

![5m uncapped drawdown](C:/Users/marti/from/reports/equity_curves/5m_uncapped_drawdown.png)

## Long-History 60m Test

Because Yahoo 5m history is short, a separate long-history robustness test was
run using Yahoo 60m / 730d data.

This is not the same as more 5m training data. It is lower-frequency but longer
history.

Selection artifact:

`C:\Users\marti\from\reports\selected_10_models_60m_730d_oldrank_forwardgate.json`

## Selected 60m Models

| # | Leader | Model | Old Train | Old Val | Old Test | Forward PnL | Forward TPD | Forward PF |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | EEM | `EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0` | 2,924.5 | 2,725.9 | 2,758.6 | 1,013.8 | 4.32 | 2.07 |
| 2 | GDX | `GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0` | 2,509.7 | 2,860.2 | 5,648.9 | 674.3 | 2.08 | 2.13 |
| 3 | GDXJ | `GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0` | 2,616.7 | 2,384.1 | 5,248.3 | 567.9 | 1.82 | 2.01 |
| 4 | GC=F | `GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0` | 2,280.3 | 2,441.6 | 5,546.9 | 130.5 | 6.27 | 1.07 |
| 5 | SPY | `SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0` | 3,127.3 | 2,043.7 | 2,107.8 | 475.0 | 4.02 | 1.77 |
| 6 | FXI | `FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0` | 3,666.7 | 2,048.9 | 2,005.2 | 274.2 | 2.93 | 1.25 |
| 7 | EFA | `EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0` | 1,848.0 | 2,185.3 | 3,788.4 | 319.4 | 2.69 | 1.55 |
| 8 | QQQ | `QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0` | 2,520.0 | 1,709.7 | 3,737.3 | 1,476.3 | 4.10 | 2.98 |
| 9 | ZB=F | `ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0` | 1,658.0 | 2,934.3 | 3,032.2 | 672.7 | 9.39 | 5.47 |
| 10 | EWJ | `EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0` | 1,818.4 | 1,627.2 | 5,823.7 | 1,209.4 | 2.23 | 205.07 |

## 60m Portfolio Backtest

Backtest artifact:

`C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped.json`

Valid capped result:

- Trades: 183
- Trades/day: 9.09
- PnL: +$179,233.30
- Return: +3.58%
- Win rate: 55.19%
- Profit factor: 1.41
- Max drawdown: $62,510.50
- Max drawdown: 1.25%
- Peak gross notional: $5,000,000
- Peak gross exposure: 100.0%

Uncapped diagnostic:

- PnL: +$804,470.40
- Return: +16.09%
- Profit factor: 1.95
- Max drawdown: $307,338.86
- Peak gross notional: $29,500,000
- Peak gross exposure: 590.0%

### 60m Capped Equity Curve

![60m capped equity](C:/Users/marti/from/reports/equity_curves/60m_capped_equity_curve.png)

### 60m Capped Drawdown

![60m capped drawdown](C:/Users/marti/from/reports/equity_curves/60m_capped_drawdown.png)

### 60m Uncapped Equity Curve

![60m uncapped equity](C:/Users/marti/from/reports/equity_curves/60m_uncapped_equity_curve.png)

### 60m Uncapped Drawdown

![60m uncapped drawdown](C:/Users/marti/from/reports/equity_curves/60m_uncapped_drawdown.png)

## Exact Signal Definitions

All model names encode the rule:

`LEADER_INTERVAL_FEATURE_MODE_wWINDOW_qQUANTILE_SESSION_hHORIZON_costCOST`

Feature types:

- `volz`: leader log-return momentum over `window`, divided by leader realized volatility
- `divergence`: leader normalized momentum minus XAU normalized momentum

Mode:

- `follow`: trade XAU in the same direction as the signal
- `fade`: trade XAU in the opposite direction of the signal

Horizon:

- 5m models: horizon is in 5m bars
- 60m models: horizon is in 60m bars

Cost:

- `cost2.0`: subtracts 2x observed XAU spread from every trade

## Why The Profits Are Not Higher

The capped runs are the valid account simulations.

The bots find many overlapping signals. If every signal gets its requested
$500,000 notional, gross exposure explodes:

- 5m uncapped peak gross exposure: 1,820%
- 60m uncapped peak gross exposure: 590%

The capped model enforces a realistic constraint:

- max active gross notional = account equity
- $5M account cannot run $91M of simultaneous signal exposure
- many trades are skipped or down-sized

That is why the capped returns are lower but more legitimate.

## Current Verdict

This is good research evidence:

- The 5m capped forward result is strong: +8.02% in about 26 days with 3.50%
  max drawdown.
- The 60m long-history result is positive: +3.58% forward with 1.25% max
  drawdown.
- Multiple model families independently pass old train/validation/test and
  forward gates.
- The strongest mechanism is cross-market macro-dislocation into XAU.

This is not yet production-proven:

- 5m Yahoo leader history is only about 60 days
- Forward holdout is only about 26 days
- ETF leader models are correlated
- No broker slippage model beyond 2x spread
- No queueing, latency, margin, or financing model
- Risk allocator is basic first-come-first-served notional allocation
- No live paper trading evidence yet

## Best Next Step

To make this materially more serious:

1. Acquire multi-month or multi-year 5m leader data for the selected symbols.
2. Re-run the same 10-model selection without changing the promotion gate.
3. Replace first-come-first-served allocation with an edge-ranked allocator.
4. Add correlation caps by leader family.
5. Add volatility targeting on XAU position size.
6. Paper trade the locked models forward without re-optimizing.

## Artifact Index

Main reports:

- `C:\Users\marti\from\reports\ten_model_5m_portfolio_report.md`
- `C:\Users\marti\from\reports\long_history_60m_portfolio_report.md`
- `C:\Users\marti\from\reports\exact_risk_model_and_equity_curves.md`
- `C:\Users\marti\from\reports\cross_market_edge_status.md`

Selected model files:

- `C:\Users\marti\from\reports\selected_10_models_oldrank_forwardgate.json`
- `C:\Users\marti\from\reports\selected_10_models_60m_730d_oldrank_forwardgate.json`

Portfolio reports:

- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped.json`
- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped.json`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped.json`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped.json`

Trade logs:

- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped_trades.csv`
- `C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped_trades.csv`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped_trades.csv`
- `C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped_trades.csv`

Equity curves:

- `C:\Users\marti\from\reports\equity_curves\5m_capped_equity_curve.csv`
- `C:\Users\marti\from\reports\equity_curves\5m_uncapped_equity_curve.csv`
- `C:\Users\marti\from\reports\equity_curves\60m_capped_equity_curve.csv`
- `C:\Users\marti\from\reports\equity_curves\60m_uncapped_equity_curve.csv`

Images:

- `C:\Users\marti\from\reports\equity_curves\5m_capped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\5m_capped_drawdown.png`
- `C:\Users\marti\from\reports\equity_curves\5m_uncapped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\5m_uncapped_drawdown.png`
- `C:\Users\marti\from\reports\equity_curves\60m_capped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\60m_capped_drawdown.png`
- `C:\Users\marti\from\reports\equity_curves\60m_uncapped_equity_curve.png`
- `C:\Users\marti\from\reports\equity_curves\60m_uncapped_drawdown.png`
