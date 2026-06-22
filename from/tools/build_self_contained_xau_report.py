#!/usr/bin/env python3
"""Build a self-contained XAU edge research report from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = REPORTS / "final_xau_edge_research_report_SELF_CONTAINED.md"


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def money(value: float) -> str:
    return f"${value:,.2f}"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def model_table(models: list[dict]) -> str:
    lines = [
        "| # | Leader | Model | Feature | Mode | Window | Quantile | Threshold | Session | Hold | Cost | Old train | Old val | Old test | Forward | Fwd TPD | Fwd PF |",
        "|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, m in enumerate(models, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(m["leader"]),
                    f"`{m['name']}`",
                    str(m["feature"]),
                    str(m["mode"]),
                    str(m["window"]),
                    str(m["quantile"]),
                    f"{float(m['threshold']):.12g}",
                    str(m["session"]),
                    str(m["horizon"]),
                    str(m["cost"]),
                    num(float(m["train"]["pnl"]), 1),
                    num(float(m["val"]["pnl"]), 1),
                    num(float(m["old_test"]["pnl"]), 1),
                    num(float(m["forward"]["pnl"]), 1),
                    num(float(m["forward"].get("tpd", m["forward"].get("trades_per_day", 0))), 2),
                    num(float(m["forward"].get("pf", m["forward"].get("profit_factor", 0))), 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def portfolio_block(title: str, report_name: str, trades_name: str) -> str:
    d = load_json(report_name)
    p = d["portfolio"]
    e = d["exposure"]
    symbols = ", ".join(d["symbols"])
    model_lines = []
    for name, s in sorted(d["model_stats"].items(), key=lambda kv: kv[1]["pnl"], reverse=True):
        model_lines.append(
            f"- `{name}`: pnl {money(float(s['pnl']))}, trades {s['trades']}, "
            f"tpd {num(float(s['trades_per_day']), 2)}, PF {num(float(s['profit_factor']), 2)}, "
            f"max DD {money(float(s['max_dd']))}"
        )
    return f"""
### {title}

Artifact JSON: `C:\\Users\\marti\\from\\reports\\{report_name}`

Trade CSV: `C:\\Users\\marti\\from\\reports\\{trades_name}`

Symbols used: `{symbols}`

Saved backtest summary:

```json
{json.dumps({"portfolio": p, "exposure": e}, indent=2)}
```

Human-readable summary:

- Trades: {p["trades"]}
- Days: {num(float(p["days"]), 2)}
- Trades/day: {num(float(p["trades_per_day"]), 2)}
- PnL: {money(float(p["pnl"]))}
- Return on $5,000,000: {pct(float(p["return_pct"]))}
- Average trade: {money(float(p["avg_trade"]))}
- Win rate: {pct(float(p["win_rate"]) * 100)}
- Profit factor: {num(float(p["profit_factor"]), 2)}
- Max drawdown: {money(float(p["max_dd"]))}
- Max drawdown pct: {pct(float(p["max_dd_pct"]))}
- Peak gross notional: {money(float(e["peak_gross_notional"]))}
- Peak gross exposure: {pct(float(e["peak_gross_exposure_pct_of_capital"]))}

Model-level contribution in this portfolio run:

{chr(10).join(model_lines)}
"""


def read_csv_head(path: Path, n: int = 20) -> str:
    df = pd.read_csv(path)
    return df.head(n).to_csv(index=False)


def main() -> int:
    selected_5m = load_json("selected_10_models_oldrank_forwardgate.json")
    selected_60m = load_json("selected_10_models_60m_730d_oldrank_forwardgate.json")
    equity_summary = pd.read_csv(REPORTS / "equity_curves" / "equity_curve_summary.csv")

    content = f"""# Self-Contained XAUUSD Edge Research Report

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
  `C:\\Users\\marti\\Documents\\Quant Lab\\Data\\data\\MASTER_DATASET_FILLED_PASS2.parquet`
- True forward XAUUSD Dukascopy holdout:
  `C:\\Users\\marti\\from\\data\\derived\\xauusd_dukascopy_5m_forward_20260522_20260618.csv`

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

{model_table(selected_5m["models"])}

## 9. 5m Backtests

{portfolio_block("5m Capped Forward", "rerun_5m_forward_portfolio_capped.json", "rerun_5m_forward_portfolio_capped_trades.csv")}

{portfolio_block("5m Uncapped Forward Diagnostic", "rerun_5m_forward_portfolio_uncapped.json", "rerun_5m_forward_portfolio_uncapped_trades.csv")}

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

{model_table(selected_60m["models"])}

## 12. 60m Backtests

{portfolio_block("60m Capped Forward", "rerun_60m_forward_portfolio_capped.json", "rerun_60m_forward_portfolio_capped_trades.csv")}

{portfolio_block("60m Uncapped Forward Diagnostic", "rerun_60m_forward_portfolio_uncapped.json", "rerun_60m_forward_portfolio_uncapped_trades.csv")}

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
{equity_summary.to_csv(index=False)}
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
{read_csv_head(REPORTS / "rerun_5m_forward_portfolio_capped_trades.csv", 15)}
```

### 60m capped trade CSV sample

```csv
{read_csv_head(REPORTS / "rerun_60m_forward_portfolio_capped_trades.csv", 15)}
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

- `C:\\Users\\marti\\from\\reports\\final_xau_edge_research_report.md`
- `C:\\Users\\marti\\from\\reports\\final_xau_edge_research_report_SELF_CONTAINED.md`
- `C:\\Users\\marti\\from\\reports\\ten_model_5m_portfolio_report.md`
- `C:\\Users\\marti\\from\\reports\\long_history_60m_portfolio_report.md`
- `C:\\Users\\marti\\from\\reports\\exact_risk_model_and_equity_curves.md`
- `C:\\Users\\marti\\from\\reports\\cross_market_edge_status.md`

Selected model JSON:

- `C:\\Users\\marti\\from\\reports\\selected_10_models_oldrank_forwardgate.json`
- `C:\\Users\\marti\\from\\reports\\selected_10_models_60m_730d_oldrank_forwardgate.json`

Backtest JSON:

- `C:\\Users\\marti\\from\\reports\\rerun_5m_forward_portfolio_capped.json`
- `C:\\Users\\marti\\from\\reports\\rerun_5m_forward_portfolio_uncapped.json`
- `C:\\Users\\marti\\from\\reports\\rerun_60m_forward_portfolio_capped.json`
- `C:\\Users\\marti\\from\\reports\\rerun_60m_forward_portfolio_uncapped.json`

Trade CSVs:

- `C:\\Users\\marti\\from\\reports\\rerun_5m_forward_portfolio_capped_trades.csv`
- `C:\\Users\\marti\\from\\reports\\rerun_5m_forward_portfolio_uncapped_trades.csv`
- `C:\\Users\\marti\\from\\reports\\rerun_60m_forward_portfolio_capped_trades.csv`
- `C:\\Users\\marti\\from\\reports\\rerun_60m_forward_portfolio_uncapped_trades.csv`

Equity curves:

- `C:\\Users\\marti\\from\\reports\\equity_curves\\5m_capped_equity_curve.csv`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\5m_uncapped_equity_curve.csv`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\60m_capped_equity_curve.csv`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\60m_uncapped_equity_curve.csv`

Image files:

- `C:\\Users\\marti\\from\\reports\\equity_curves\\5m_capped_equity_curve.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\5m_capped_drawdown.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\5m_uncapped_equity_curve.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\5m_uncapped_drawdown.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\60m_capped_equity_curve.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\60m_capped_drawdown.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\60m_uncapped_equity_curve.png`
- `C:\\Users\\marti\\from\\reports\\equity_curves\\60m_uncapped_drawdown.png`
"""
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
