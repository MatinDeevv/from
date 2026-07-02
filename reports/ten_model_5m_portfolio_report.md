# Ten-Model XAUUSD Cross-Market Portfolio

Generated: 2026-06-18

## Selection Rule

Screened 52 liquid macro leaders on Yahoo 5m data:

- Equity/index/sector ETFs
- Rates/credit ETFs
- Commodity ETFs/futures
- FX proxies and Yahoo FX symbols
- Country ETFs
- Crypto proxies

Promotion gate:

- 5m XAUUSD execution market
- no external XAU gap-fill rows in old sample
- 2x observed XAUUSD spread cost
- old train PnL positive
- old validation PnL positive
- old test PnL positive
- forward PnL positive
- old and forward frequency inside 15-35 trades/day

The final 10 were ranked by old-sample stability, not by forward PnL:

- rank unique leaders by minimum old train/validation/test PnL
- tie-break by old train+validation+test total
- require untouched forward pass

Selection artifact:

`C:\Users\marti\from\reports\selected_10_models_oldrank_forwardgate.json`

## Selected 10 Models

| # | Leader | Model | Old train | Old val | Old test | Forward PnL | Forward TPD | Forward PF |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | XLC | `XLC_volz_follow_w36_q0.55_ny_h30_cost2.0` | 1,824.2 | 1,471.6 | 1,661.8 | 1,999.0 | 25.8 | 1.26 |
| 2 | XLV | `XLV_volz_follow_w48_q0.65_ny_h24_cost2.0` | 1,863.1 | 1,955.2 | 1,414.8 | 3,697.1 | 27.6 | 1.86 |
| 3 | DX-Y.NYB | `DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0` | 1,837.4 | 1,411.8 | 1,785.6 | 1,795.1 | 30.3 | 1.30 |
| 4 | IEF | `IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0` | 1,039.0 | 1,186.6 | 1,021.8 | 243.8 | 29.3 | 1.03 |
| 5 | XLF | `XLF_volz_fade_w48_q0.55_london_h30_cost2.0` | 3,872.6 | 834.0 | 820.1 | 3,787.7 | 26.0 | 1.97 |
| 6 | UNG | `UNG_volz_follow_w48_q0.75_all_h18_cost2.0` | 3,621.6 | 1,076.4 | 747.1 | 2,288.9 | 30.2 | 1.45 |
| 7 | XLK | `XLK_volz_fade_w48_q0.75_all_h30_cost2.0` | 674.3 | 2,574.6 | 2,372.4 | 45.4 | 25.1 | 1.01 |
| 8 | VXX | `VXX_volz_follow_w48_q0.75_all_h30_cost2.0` | 4,802.5 | 3,004.2 | 610.0 | 2,012.2 | 25.9 | 1.29 |
| 9 | EURUSD=X | `EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0` | 2,673.9 | 564.0 | 2,121.7 | 1,151.9 | 33.2 | 1.25 |
| 10 | SPY | `SPY_volz_fade_w36_q0.75_all_h24_cost2.0` | 1,226.6 | 523.7 | 917.5 | 1,790.8 | 26.6 | 1.39 |

## $5M Portfolio Backtest

Sizing:

- $5,000,000 starting capital
- 10 models
- each signal requests $500,000 notional
- capped version never allows active gross notional above $5,000,000
- if capital is fully allocated, new signals are skipped or down-sized

Forward capped report:

`C:\Users\marti\from\reports\selected_10_models_5m_forward_portfolio_capped.json`

Forward capped trades:

`C:\Users\marti\from\reports\selected_10_models_5m_forward_portfolio_capped_trades.csv`

Forward capped result:

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

Old capped report:

`C:\Users\marti\from\reports\selected_10_models_5m_old_portfolio_capped.json`

Old capped result:

- Trades: 1,916
- Trades/day: 32.51
- PnL: +$925,881.80
- Return: +18.52%
- Win rate: 55.64%
- Profit factor: 1.57
- Max drawdown: $132,116.32
- Max drawdown: 2.64%
- Peak gross notional: $5,000,000
- Peak gross exposure: 100.0%

## Uncapped Diagnostic

Uncapped forward report:

`C:\Users\marti\from\reports\selected_10_models_5m_forward_portfolio.json`

Uncapped forward result:

- PnL: +$2,113,068.37
- Return: +42.26%
- Profit factor: 1.32
- Max drawdown: $1,065,573.07
- Peak gross notional: $94,000,000
- Peak gross exposure: 1,880%

Verdict: the uncapped result is not a real $5M execution backtest. It is only a
signal-strength diagnostic. The capped result is the valid $5M result.

## Current Verdict

This is the strongest 10-model basket found so far under the current framework.
It passes old train/validation/test and true forward gates, and it remains
profitable when capped to $5M gross exposure.

It is still not production-proven:

- selection still uses a short Yahoo 5m history window
- the selected ETF leaders are correlated
- some capped portfolio contributions turn negative because capital contention
  changes which trades are actually taken
- the forward window is only 2026-05-22 to 2026-06-17
- no broker slippage or queueing model beyond 2x spread is included

Honest status: real research edge candidate, not deployment-ready.
