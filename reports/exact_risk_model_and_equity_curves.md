# Exact Risk Model And Equity Curves

Generated: 2026-06-18

## Why The Profits Look Lower

The profitable but lower-looking numbers are from the valid capped account
simulation.

The raw signals overlap heavily. If every signal gets its requested notional,
the 5m model stack reaches about $91M gross exposure on a $5M account. That is
18.2x gross exposure. The earlier uncapped 5m forward diagnostic made
+$2.11M, but it used that extreme overlapping notional.

The valid capped run enforces max gross active exposure of $5M. When the account
is already allocated, new trades are skipped or down-sized. That cuts a lot of
gross opportunity but keeps the test executable.

## Exact Bot Risk Model Used

The current bots/backtests use a very simple notional model.

Account:

- Initial capital: $5,000,000
- No compounding
- No portfolio-level volatility targeting
- No Kelly sizing
- No per-model drawdown throttle
- No stop-loss
- No take-profit
- No trailing stop
- No margin interest
- No borrow/financing model

Per signal:

- 10 selected models
- Each model requests `capital / model_count`
- With $5M and 10 models, each new signal requests $500,000 notional
- Trade direction is the model signal: `+1` long XAU, `-1` short XAU
- Ounces traded: `assigned_notional / XAU_entry_price`
- Gross PnL: `direction * (XAU_exit - XAU_entry) * ounces`
- Transaction cost: `2.0 * observed_XAU_spread * ounces`
- Net PnL: `gross_pnl - transaction_cost`

Capped execution:

- Max active gross notional: $5,000,000
- Trades are processed by entry time
- Exits at the same timestamp are freed before new entries
- If available notional is >= requested notional, full signal is taken
- If available notional is between 0 and requested notional, signal is down-sized
- If available notional is 0, signal is skipped
- No ranking is applied among simultaneous new signals beyond current CSV order

Uncapped diagnostic:

- Every signal gets full requested notional
- Gross exposure can exceed account size
- This is not a real $5M execution test
- Use this only to inspect raw signal strength

## Re-Run Results

### 5m Capped Forward

Report:

`C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped.json`

Trades:

`C:\Users\marti\from\reports\rerun_5m_forward_portfolio_capped_trades.csv`

Result:

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

Equity curve:

`C:\Users\marti\from\reports\equity_curves\5m_capped_equity_curve.csv`

`C:\Users\marti\from\reports\equity_curves\5m_capped_equity_curve.png`

Drawdown:

`C:\Users\marti\from\reports\equity_curves\5m_capped_drawdown.png`

### 5m Uncapped Forward

Report:

`C:\Users\marti\from\reports\rerun_5m_forward_portfolio_uncapped.json`

Result:

- Trades: 7,064
- Trades/day: 269.78
- PnL: +$2,113,068.37
- Return: +42.26%
- Profit factor: 1.32
- Max drawdown: $1,065,573.07
- Max drawdown: 21.31%
- Peak gross notional: $91,000,000
- Peak gross exposure: 1,820.0%

Equity curve:

`C:\Users\marti\from\reports\equity_curves\5m_uncapped_equity_curve.csv`

`C:\Users\marti\from\reports\equity_curves\5m_uncapped_equity_curve.png`

Drawdown:

`C:\Users\marti\from\reports\equity_curves\5m_uncapped_drawdown.png`

### 60m Capped Forward

Report:

`C:\Users\marti\from\reports\rerun_60m_forward_portfolio_capped.json`

Result:

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

Equity curve:

`C:\Users\marti\from\reports\equity_curves\60m_capped_equity_curve.csv`

`C:\Users\marti\from\reports\equity_curves\60m_capped_equity_curve.png`

Drawdown:

`C:\Users\marti\from\reports\equity_curves\60m_capped_drawdown.png`

### 60m Uncapped Forward

Report:

`C:\Users\marti\from\reports\rerun_60m_forward_portfolio_uncapped.json`

Result:

- Trades: 373
- Trades/day: 18.53
- PnL: +$804,470.40
- Return: +16.09%
- Profit factor: 1.95
- Max drawdown: $307,338.86
- Max drawdown: 6.15%
- Peak gross notional: $29,500,000
- Peak gross exposure: 590.0%

Equity curve:

`C:\Users\marti\from\reports\equity_curves\60m_uncapped_equity_curve.csv`

`C:\Users\marti\from\reports\equity_curves\60m_uncapped_equity_curve.png`

Drawdown:

`C:\Users\marti\from\reports\equity_curves\60m_uncapped_drawdown.png`

## Bottom Line

The current capped risk model is conservative in one specific way: it limits
gross active notional to account equity. That is why the profits are much lower
than the raw signal diagnostics.

The model is not using sophisticated risk. It is not volatility-scaled,
correlation-aware, Kelly-sized, regime-gated, or dynamic. It is mostly an equal
notional signal stack with a hard gross-exposure cap.

The next real improvement is not just "more leverage." The better next step is
a portfolio allocator that ranks simultaneous signals by expected edge, caps
correlated model groups, and sizes by realized XAU volatility.
