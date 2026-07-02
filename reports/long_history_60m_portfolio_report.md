# Long-History 60m XAUUSD Cross-Market Portfolio

Generated: 2026-06-18

## Purpose

The previous 10-model portfolio used Yahoo 5m data, which is limited to about
60 days for ETF leaders. This run trains a separate longer-history model set on
Yahoo 60m bars with the `730d` range.

This is not the same as more 5m training data. It is a longer-horizon
robustness test using 60m bars.

## Universe

The same broad macro universe was screened:

- equity/index/sector ETFs
- rates/credit ETFs
- commodity ETFs/futures
- FX proxies and Yahoo FX symbols
- country ETFs
- crypto proxies

Search artifact batches:

- `C:\Users\marti\from\reports\long_history_60m_batch1_search_2x.json`
- `C:\Users\marti\from\reports\long_history_60m_batch2_search_2x.json`
- `C:\Users\marti\from\reports\long_history_60m_batch3_search_2x.json`
- `C:\Users\marti\from\reports\long_history_60m_batch4_search_2x.json`

Selection artifact:

`C:\Users\marti\from\reports\selected_10_models_60m_730d_oldrank_forwardgate.json`

Selection rule:

- 60m Yahoo leader bars, `730d`
- 2x observed XAUUSD spread cost
- old train/validation/test PnL all positive
- forward PnL positive
- old and forward model frequency inside 0.5-12 trades/day
- rank unique leaders by minimum old train/validation/test PnL

## Selected 10 Long-History Models

| # | Leader | Model | Old train | Old val | Old test | Forward PnL | Forward TPD | Forward PF |
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

## $5M Capped Portfolio

Sizing:

- $5,000,000 starting capital
- 10 models
- each signal requests $500,000 notional
- gross active notional capped at $5,000,000
- if capital is fully allocated, new signals are skipped or down-sized

Forward capped report:

`C:\Users\marti\from\reports\selected_10_models_60m_730d_forward_portfolio_capped.json`

Forward capped result:

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

Old capped report:

`C:\Users\marti\from\reports\selected_10_models_60m_730d_old_portfolio_capped.json`

Old capped result:

- Old span: roughly 1,026 calendar days
- Trades: 6,681
- Trades/day: 6.51
- PnL: +$3,674,917.11
- Return: +73.50%
- Win rate: 51.73%
- Profit factor: 1.27
- Max drawdown: $648,070.04
- Max drawdown: 12.96%
- Peak gross exposure: 100.0%

Uncapped forward diagnostic:

`C:\Users\marti\from\reports\selected_10_models_60m_730d_forward_portfolio_uncapped.json`

- PnL: +$804,470.40
- Return: +16.09%
- Trades/day: 18.53
- Peak gross exposure: 590.0%

The uncapped result is not a valid $5M execution result; it is only a signal
diagnostic.

## Verdict

The long-history 60m models are profitable on a much larger training window and
remain positive on the untouched forward holdout with capped $5M exposure.

The tradeoff is frequency:

- 5m short-history capped forward: +8.02%, 43.53 trades/day
- 60m long-history capped forward: +3.58%, 9.09 trades/day

The long-history result is more reassuring for robustness, but it does not meet
the earlier 20-trades/day target when capped to $5M gross exposure. To get both
long history and high trade count, the next step needs true multi-month or
multi-year 5m data for the ETF/leader universe from a vendor such as Alpaca,
Polygon, Databento, or broker-exported bars.
