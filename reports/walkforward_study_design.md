# Frozen Walk-Forward Study Design

Created before any walk-forward backtest results were run.

## Objective

Validate only the two 5m signal families that survived the prior strict/fresh screens:

- Family A: energy London-follow
- Family B: US sector London-fade

This study is not allowed to expand the universe, add features, or tune thresholds after seeing results.

## Required Data

Target date range:

- Preferred: `2022-01-01` to present
- Absolute minimum: `2023-01-01` to present

Required instruments:

- XAU/USD execution market: `XAUUSD`
- Energy family: `USOIL` or Dukascopy WTI proxy for USO, and `NATGAS` or Dukascopy natural-gas proxy for UNG
- Sector family: XLF/XLY direct data if available, otherwise documented Dukascopy proxies such as `SPX500`/`US30` or `USA500.IDX/USD`/`USA30.IDX/USD`

Minimum condition to proceed:

- XAUUSD plus at least one energy leader and at least one sector/proxy leader must have matching 5m coverage from at least `2023-01-01` through `2026-05-21`.
- Any file with more than 1% missing bars or a max undocumented consecutive gap greater than 60 bars is rejected.

## Data Construction

Raw Dukascopy `.bi5` tick files must be saved before resampling.

Tick resampling:

- Bar timeframe: 5 minutes
- Price: mid-price `(bid + ask) / 2`
- OHLC: first, max, min, last mid within each 5m bucket
- Spread: average `(ask - bid)` within each 5m bucket
- Volume fields: average ask/bid volume if available
- Timestamps: UTC, aligned to exact 5m bucket starts

## Signal Families

### Family A - Energy London-Follow

- Leaders: `USOIL` and `NATGAS` or documented equivalents
- Features: `session_volz`, `volz`, `momentum_ratio`
- Mode: follow
- Session: London, `08:00` to `12:00` UTC
- Windows: `{4, 6, 9, 12, 18, 24}`
- Quantiles: `{0.55, 0.65, 0.75}`
- Holds: `{18, 24, 30, 36, 42, 48}` 5m bars

### Family B - US Sector London-Fade

- Leaders: XLF proxy and XLY proxy
- Features: `session_volz`, `divergence`, `ew_zspread`
- Mode: fade
- Session: London, `08:00` to `12:00` UTC
- Windows: `{4, 6, 9, 12, 18, 24}`
- Quantiles: `{0.55, 0.65, 0.75}`
- Holds: `{18, 24, 30, 36, 42, 48}` 5m bars

No other families, leaders, modes, sessions, features, windows, quantiles, or holds are allowed in this study.

## Walk-Forward Schedule

Anchored expanding-window walk-forward. Thresholds are fitted using the full fit period up to the fold fit end date. Test windows are non-overlapping.

| Fold | Fit end date | Test start | Test end |
|---|---|---|---|
| 1 | `2023-06-30` | `2023-07-01` | `2023-12-31` |
| 2 | `2023-12-31` | `2024-01-01` | `2024-06-30` |
| 3 | `2024-06-30` | `2024-07-01` | `2024-12-31` |
| 4 | `2024-12-31` | `2025-01-01` | `2025-06-30` |
| 5 | `2025-06-30` | `2025-07-01` | `2025-12-31` |
| 6 | `2025-12-31` | `2026-01-01` | `2026-05-21` |

The May 22 to June 17, 2026 window is sealed final holdout and is not opened unless the walk-forward portfolio passes.

## Per-Fold Candidate Criteria

A candidate passes a fold only if all are true in the fold test period:

- PnL > 0
- Trade count >= 50
- Stationary/block bootstrap 95% CI lower bound > 0 using 1000 resamples
- Four calendar sub-periods: no single sub-period contributes more than 75% of total fold PnL

A candidate is walk-forward validated only if it passes at least 4 of 6 folds.

## Aggregate Candidate Metrics

For each walk-forward validated candidate:

- Per-fold PnL and trade count for all six folds
- Cross-fold Sharpe: mean fold return divided by standard deviation of fold returns
- Fraction of folds positive
- Bootstrap CI on the six fold-return observations, with small-sample caveat

## Portfolio Criteria

A valid walk-forward portfolio requires:

- At least one survivor from Family A
- At least one survivor from Family B
- Allocator policies tested: `edge_rank`, `inverse_vol`, `name_order`
- Allocator PnL swing <= 30%
- Effective bets >= 2
- Cumulative stitched equity curve across all six fold test periods reported

## Final Holdout Rule

Open the sealed May 22 to June 17, 2026 final holdout only if the walk-forward portfolio passes every Phase 1 criterion.

If the walk-forward portfolio fails or data is insufficient, write:

`Final holdout was not opened. Walk-forward did not meet acceptance criteria.`
