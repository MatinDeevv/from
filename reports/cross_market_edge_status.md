# Cross-Market XAUUSD Edge Status

Generated: 2026-06-17

## Current Verdict

We have a plausible edge candidate with materially stronger evidence after the
latest leakage/alignment fixes, but it is still not a production-ready strategy.

The best surviving family is:

- Execution market: timestamp-restored XAUUSD from `MASTER_DATASET_FILLED_PASS2.parquet`
- Bar size: 5 minutes
- Filter: `external_gap_fill == false`; strongest corrected run also requires `timestamp_dukascopy_exact_verified_all == true`
- Leader: mostly `SPY`; some secondary futures/ETF clusters exist
- Direction: fade short-term SPY moves into XAUUSD
- Session: all exact leader-aligned SPY rows, with London/NY variants also present
- Hold: 45-60 minutes depending on variant
- Trade rate: roughly 23-34 trades/day depending on threshold and cost stress

## Strongest Corrected Run

Report:

`C:\Users\marti\from\reports\cross_market_train_threshold_5m_exact_nogap_leaderaligned.json`

Strict mechanics:

- Per-leader aligned rows only; missing leader rows are not part of that leader's threshold or split
- Threshold quantiles fitted on first-half train rows only
- Leader signals lagged by one completed 5m bar
- No same-bar XAUUSD lookahead
- No external gap-fill rows
- XAUUSD rows require exact Dukascopy verification
- Actual XAUUSD spread charged
- Chronological train/validation/test split

Summary:

- aligned rows: 3,950
- time range: 2026-03-24 14:20 UTC to 2026-05-22 13:50 UTC
- candidate count: 18,168
- qualified train+validation+test count: 345
- strongest displayed cluster: `SPY` fade

Top examples:

- `SPY_fade_lag1_w6_q0.4_all`, hold 9 bars, cost 2x
  - train: +322.75, 33.9 trades/day
  - validation: +1,581.78, 16.8 trades/day
  - test: +1,455.64, 25.7 trades/day
  - test win rate: 51.7%

- `SPY_fade_lag1_w6_q0.45_all`, hold 9 bars, cost 2x
  - train: +453.08, 31.0 trades/day
  - validation: +1,422.52, 15.3 trades/day
  - test: +1,226.70, 23.4 trades/day
  - test win rate: 52.5%

## Timing Controls

Report:

`C:\Users\marti\from\reports\cross_market_train_threshold_5m_exact_nogap_leaderaligned.json`

Top controls versus circular/shuffled signal timing:

- `SPY_fade_lag1_w6_q0.4_all`, hold 12, cost 1x
  - real test: +2,190.43
  - control test p95: +1,786.47
  - control test p99: +2,290.33
  - rank: 98.0 percentile

- `SPY_fade_lag1_w6_q0.4_all`, hold 9, cost 2x
  - real test: +1,455.64
  - control test p95: +1,342.24
  - control test p99: +1,564.41
  - rank: 96.3 percentile

- `SPY_fade_lag1_w6_q0.4_all`, hold 9, cost 1.5x
  - real test: +1,565.81
  - control test p95: +1,451.61
  - control test p99: +1,673.97
  - rank: 96.3 percentile

## Locked Rule Validation

Report:

`C:\Users\marti\from\reports\locked_spy_fade_w6_q40_h9_cost2_exact.json`

Trades:

`C:\Users\marti\from\reports\locked_spy_fade_w6_q40_h9_cost2_exact_trades.csv`

Fixed rule:

- Leader: `SPY`
- Direction: fade SPY momentum
- Momentum window: 6 completed 5m bars
- Threshold: first-half train-only 40% absolute-momentum quantile, `0.0005740430`
- Hold: 9 bars
- Cost: 2x XAUUSD spread
- Rows: exact Dukascopy verified, no external gap fills

Locked-rule result:

- aligned rows: 2,603
- trades: 1,497
- all PnL: +3,360.16
- all trades/day: 25.4
- all win rate: 48.8%
- all profit factor: 1.38
- test PnL: +1,455.64
- test trades/day: 25.7
- test win rate: 51.7%

Reverse-direction ablation:

- reverse all PnL: -7,566.15
- reverse train PnL: -2,736.59
- reverse validation PnL: -2,492.53
- reverse test PnL: -2,337.03

Bootstrap:

- all-period 5th percentile PnL: +1,404.75
- all-period probability of positive sum: 99.72%
- test 5th percentile PnL: +643.24
- test probability of positive sum: 99.98%

Circular timing controls:

- all-period rank versus 2,000 circular signal shifts: 99.2 percentile
- test rank versus 2,000 circular signal shifts: 92.4 percentile
- test control p95: +1,635.56, above the real test PnL, so the final split alone is not a 95th-percentile timing outlier

## Weaknesses

The sample is still short: about two months of 5m leader data. Earlier broad
SPY runs showed an unstable April fold, and the corrected exact run still needs
true longer-history validation before live trading.

The locked SPY rule also has one negative chronological fold:

- fold 1: +638.89
- fold 2: -316.14
- fold 3: +1,581.78
- fold 4: +1,455.64

That makes the evidence real but not final.

## Locked Family Validation

Report:

`C:\Users\marti\from\reports\locked_spy_family_trainpositive3_cost2_exact.json`

Trades:

`C:\Users\marti\from\reports\locked_spy_family_trainpositive3_cost2_exact_trades.csv`

This validates a fixed equal-weight basket of the three SPY fade components
that each individually passed train/validation/test at 2x spread:

- `SPY_fade`, window 6, q 0.40, hold 9
- `SPY_fade`, window 6, q 0.45, hold 9
- `SPY_fade`, window 4, q 0.40, hold 9

Family result:

- component-weighted trades: 4,326
- all PnL: +2,901.70
- all component trades/day: 73.4
- all win rate: 48.7%
- all profit factor: 1.34
- test PnL: +1,174.01
- test component trades/day: 74.1
- test win rate: 51.8%

Family fold stability:

- fold 1: +522.51
- fold 2: -63.49
- fold 3: +1,268.67
- fold 4: +1,174.01

Circular timing controls:

- all-period rank versus 1,000 circular signal shifts: 98.7 percentile
- test rank versus 1,000 circular signal shifts: 89.8 percentile

Interpretation: the 3-rule family reduces the bad fold materially versus the
single-rule validation, but it represents multiple component signals. For the
20-trades/day target, the single locked rule is cleaner; the family is evidence
that the effect is not just one exact threshold.

## Execution Realism

The original locked-rule validation counts overlapping 45-minute holds as
separate trades. That matches a signal-count interpretation of 20+ trades/day,
but it is not the same as a one-position execution model.

Additional reports:

- `C:\Users\marti\from\reports\locked_spy_fade_w6_q40_h9_cost2_exact_cooldown.json`
- `C:\Users\marti\from\reports\locked_spy_fade_w6_q40_h9_cost2_exact_netted.json`

Cooldown mode, no new trade until the prior 45-minute hold exits:

- trades: 246
- all trades/day: 4.2
- all PnL: +68.57
- train PnL: -153.48
- bootstrap probability of positive sum: 52.1%

Verdict: cooldown execution does not satisfy the 20 trades/day requirement and
does not preserve the edge robustly.

Netted mode, one unit max exposure marked to market each active bar:

- active rows: 2,339
- active rows/day: 39.7
- all PnL: +628.48
- train PnL: +100.24
- validation PnL: +124.74
- test PnL: +403.51
- reverse all PnL: -1,255.85
- circular timing all-period rank: 96.3 percentile
- circular timing test rank: 99.6 percentile

Verdict: the timing relation survives one-unit netted exposure, but this is an
active-bar strategy, not 20 independent completed trades/day.

## Forward Holdout

Forward XAUUSD data was fetched directly from Dukascopy after the restored
master dataset ended.

XAU forward data:

- `C:\Users\marti\from\data\derived\xauusd_dukascopy_5m_forward_20260522_20260618.csv`
- `C:\Users\marti\from\reports\xauusd_dukascopy_5m_forward_20260522_20260618.json`

Forward window:

- start: 2026-05-22 13:55 UTC
- end: 2026-06-17 23:55 UTC
- bars: 5,047
- fetch errors: 0

The XAU price seam is continuous: the restored master ends at 4515.725 on
2026-05-22 13:50 UTC, and the forward Dukascopy series starts at 4513.830 on
2026-05-22 13:55 UTC with similar spread.

Locked SPY rule forward reports:

- `C:\Users\marti\from\reports\forward_locked_spy_overlap_20260522_20260618.json`
- `C:\Users\marti\from\reports\forward_locked_spy_cooldown_20260522_20260618.json`
- `C:\Users\marti\from\reports\forward_locked_spy_netted_20260522_20260618.json`

Forward locked SPY results:

- overlap: -2,688.97 PnL, 63.2 trades/day
- cooldown: -304.81 PnL, 10.8 trades/day
- netted: -786.70 PnL, 101.8 active rows/day

Verdict: the locked SPY candidate fails the true forward holdout.

Pre-forward-selected holdout scan:

- `C:\Users\marti\from\reports\forward_holdout_selected_rules_full.json`
- old exact/no-gap rows: 3,595
- forward rows: 4,801
- pre-forward selected candidates: 736
- forward-positive candidates inside 15-35 trades/day: 0
- forward-positive 2x-cost candidates inside 15-35 trades/day: 0

Some crude-oil (`CL=F`) follow variants survived forward, including 8 positive
2x-cost candidates, but their forward frequency was roughly 38-49 trades/day,
outside the 20/day target band used here.

Current forward verdict at this stage: the original SPY result was a serious
candidate, but true forward data killed it. Later feature-normalized macro
tests found a stronger EURUSD/DXY lead family; see the addendum below.

Broader old-sample selection:

- `C:\Users\marti\from\reports\forward_holdout_selected_rules_nogap_old_full.json`
- old sample: no-gap restored rows, not exact-only
- old rows: 7,726
- forward rows: 4,801
- pre-forward selected candidates: 459
- forward-positive candidates inside 15-35 trades/day: 43
- forward-positive 2x-cost candidates inside 15-35 trades/day: 1

The only 2x target-band survivor was:

- `CL=F_follow_w3_q0.5_ny_h12_cost2.0`
- old train PnL: +36.99
- old validation PnL: +183.67
- old test PnL: -315.89
- forward PnL: +6.41
- forward trades/day: 29.8
- forward profit factor: 1.00
- forward drawdown: 853.69

Verdict: this crude-oil rule is not a tradable edge. A later
volatility-normalized CL variant did survive overlap accounting at 2x cost, but
failed the cooldown model and had an unstable forward second half.

## Macro Feature-Normalized Addendum

Latest reports:

- `C:\Users\marti\from\reports\forward_holdout_macro_features_rates_fx_energy_2x.json`
- `C:\Users\marti\from\reports\forward_locked_eurusd_div_w36_q70_london_h18_cost2_overlap.json`
- `C:\Users\marti\from\reports\forward_locked_eurusd_div_w36_q70_london_h18_cost2_cooldown.json`
- `C:\Users\marti\from\reports\forward_locked_eurusd_div_w36_q70_london_h18_cost2_netted.json`
- `C:\Users\marti\from\reports\forward_locked_dxy_volz_w36_q85_all_h18_cost2_overlap.json`

Stronger current lead:

- Signal: `EURUSD=X` divergence versus XAU's own prior move
- Direction: follow divergence
- Window: 36 completed 5m bars
- Threshold: train-fitted 70% absolute divergence quantile, `0.8169827795693141`
- Session: London, 07:00-16:00 UTC
- Hold: 18 bars, 90 minutes
- Cost: 2x observed XAUUSD spread

Pre-forward selected result from the scanner:

- old train PnL: +2,659.6
- old validation PnL: +423.8
- old test PnL: +1,363.6
- forward PnL: +1,594.0
- forward trades/day: 28.5
- forward profit factor: 1.57

Locked forward overlap validation:

- trades: 662
- trades/day: 28.47
- PnL: +1,594.04
- win rate: 58.46%
- profit factor: 1.57
- reverse-rule PnL: -3,091.46
- max drawdown: 557.01
- circular-control rank: 89.7 percentile

Forward chronological split:

- first half: +970.5 PnL, 32.9 trades/day, profit factor 1.65
- second half: +623.5 PnL, 25.1 trades/day, profit factor 1.48
- quarter 1: +115.6
- quarter 2: +854.9
- quarter 3: +42.0
- quarter 4: +581.6

Execution-model stress:

- cooldown: +191.62 PnL, but only 3.44 trades/day
- one-unit netted: +245.80 PnL, but 73.09 active rows/day
- overlap accounting is the only version matching the 20/day target

Independent macro confirmation:

- `DX-Y.NYB` volatility-normalized fade, window 36, q 0.85, all sessions, hold 18
- old train/validation/test: +1,174.8 / +574.1 / +528.0
- forward: +1,426.8, 30.6 trades/day, profit factor 1.33
- reverse-rule PnL: -3,461.3

Important caveats:

- Exact-only old-row filtering did not reproduce the EURUSD lead; the result
  uses no-gap restored rows but not the strict exact-only subset.
- The forward window is only 2026-05-22 to 2026-06-17.
- The overlap circular-control rank is below a clean 95% threshold, though the
  cooldown and netted variants ranked 96.0% and 94.6% respectively.

Current honest verdict after this first forward check: this was the best
actual edge candidate found at the time, but the longer Dukascopy validation
below supersedes it. The mechanism is plausible for gold: EURUSD/DXY
dislocations lead short-horizon XAU continuation/repricing.

## Six-Month Dukascopy EURUSD Validation

New data pulled from Dukascopy:

- `C:\Users\marti\from\data\derived\eurusd_dukascopy_5m_20251201_20260522.csv`
- `C:\Users\marti\from\data\derived\eurusd_dukascopy_5m_forward_20260522_20260618.csv`
- `C:\Users\marti\from\reports\eurusd_dukascopy_5m_20251201_20260522_combined.json`
- `C:\Users\marti\from\reports\eurusd_dukascopy_5m_forward_20260522_20260618.json`

The earlier London rule was locked and tested over the six-month old sample:

- Report: `C:\Users\marti\from\reports\locked_eurusd_div_w36_thr0817_london_h18_cost2_202512_202605_overlap.json`
- Old rows aligned: 24,046
- Trades: 3,848
- Trades/day: 22.50
- All-period PnL: -5,680.04
- Test PnL: +3,457.08

Verdict: the earlier London rule was regime-specific. It worked in the recent
period and forward holdout, but failed the six-month pre-forward sample.

Six-month old-to-forward search then found a stronger NY-session variant:

- Search report: `C:\Users\marti\from\reports\local_eurusd_6mo_to_forward_search_2x_filters.json`
- Rule: EURUSD/XAU normalized divergence, follow
- Window: 42 completed 5m bars
- Threshold: `0.7904402382958319`
- Session: NY, 13:00-21:00 UTC
- Hold: 30 bars, 150 minutes
- Cost: 2x observed XAUUSD spread
- Filter: none; old-fitted XAU volatility, EURUSD volatility, and spread
  filters did not improve the survivor set

Six-month old locked validation:

- Report: `C:\Users\marti\from\reports\locked_eurusd_div_w42_thr0790_ny_h30_cost2_202512_202605_overlap.json`
- Trades: 3,180
- Trades/day: 18.70
- PnL: +21,326.40
- Win rate: 58.27%
- Profit factor: 1.69
- Reverse-rule PnL: -30,567.03
- Old train/validation/test PnL: +7,336.1 / +12,525.8 / +1,464.5
- Circular-control all-period rank: 98.2 percentile
- Eight fold PnLs: +3,427.0, -378.1, +907.8, +3,379.3, +10,030.2,
  +2,495.6, -306.0, +1,770.5

Forward locked validation:

- Report: `C:\Users\marti\from\reports\forward_locked_eurusd_div_w42_thr0790_ny_h30_cost2_overlap.json`
- Trades: 557
- Trades/day: 21.46
- PnL: +1,520.20
- Win rate: 55.66%
- Profit factor: 1.29
- Reverse-rule PnL: -2,831.05
- First half / second half: +1,374.2 / +146.0
- Quarter PnLs: +1,161.9, +212.3, -1,488.9, +1,634.9
- Circular-control all-period rank: 73.2 percentile

Execution stress on the forward window:

- Cooldown report: `C:\Users\marti\from\reports\forward_locked_eurusd_div_w42_thr0790_ny_h30_cost2_cooldown.json`
- Cooldown: +80.62 PnL, but only 1.43 trades/day
- Netted report: `C:\Users\marti\from\reports\forward_locked_eurusd_div_w42_thr0790_ny_h30_cost2_netted.json`
- Netted: +22.16 PnL, effectively flat, and not a 20/day trade model

Current honest verdict at this point: this was the strongest EURUSD candidate
found, but the USDJPY validation below is cleaner and now supersedes it as the
best single lead.

## Six-Month Dukascopy USDJPY Validation

New USDJPY data pulled from Dukascopy:

- `C:\Users\marti\from\data\derived\usdjpy_dukascopy_5m_20251201_20260522.csv`
- `C:\Users\marti\from\data\derived\usdjpy_dukascopy_5m_forward_20260522_20260618.csv`
- `C:\Users\marti\from\reports\usdjpy_dukascopy_5m_20251201_20260522_combined.json`
- `C:\Users\marti\from\reports\usdjpy_dukascopy_5m_forward_20260522_20260618.json`

Search report:

- `C:\Users\marti\from\reports\local_usdjpy_6mo_to_forward_search_2x.json`
- Old rows: 24,070
- Forward rows: 5,047
- Six-month old train/validation/test positive candidates: 39
- Forward-positive candidates inside 15-35 trades/day: 23

Dominant family:

- Leader: USDJPY
- Feature: volatility-normalized leader move
- Direction: fade USDJPY extreme moves into XAU
- Session: all sessions
- Hold: 150 minutes
- Cost: 2x observed XAUUSD spread

Top aggregate variant:

- `USDJPY_volz_fade_w48_q0.8_all_h30_cost2.0`
- Old train/validation/test: +2,867.8 / +4,973.2 / +2,355.1
- Forward: +2,255.7, 29.9 trades/day, profit factor 1.66
- Forward reverse-rule PnL: -3,959.0
- Caveat: forward final quarter was -359.9

Cleaner locked variant:

- Rule: USDJPY volatility-normalized fade
- Window: 48 completed 5m bars
- Threshold: `1.4066638009465553`
- Session: all
- Hold: 30 bars, 150 minutes
- Cost: 2x observed XAUUSD spread
- Reports:
  - `C:\Users\marti\from\reports\locked_usdjpy_volz_w48_thr1407_all_h30_cost2_202512_202605_overlap.json`
  - `C:\Users\marti\from\reports\forward_locked_usdjpy_volz_w48_thr1407_all_h30_cost2_overlap.json`
  - `C:\Users\marti\from\reports\forward_locked_usdjpy_volz_w48_thr1407_all_h30_cost2_cooldown.json`
  - `C:\Users\marti\from\reports\forward_locked_usdjpy_volz_w48_thr1407_all_h30_cost2_netted.json`

Six-month old locked validation:

- Trades: 3,316
- Trades/day: 19.53
- PnL: +9,131.42
- Win rate: 49.91%
- Profit factor: 1.30
- Reverse-rule PnL: -19,189.25
- Old train/validation/test PnL: +3,395.0 / +4,070.9 / +1,665.5
- Circular-control all-period rank: 93.1 percentile
- Eight fold PnLs: +997.5, -2,954.2, +2,460.8, +2,891.0,
  +524.4, +3,546.5, +330.1, +1,335.5

Forward locked validation:

- Trades: 512
- Trades/day: 21.74
- PnL: +1,827.50
- Win rate: 60.74%
- Profit factor: 1.77
- Max drawdown: 636.38
- Reverse-rule PnL: -3,056.53
- First half / second half: +838.3 / +989.2
- Quarter PnLs: +503.3, +335.0, +998.5, -9.3
- Circular-control all-period rank: 85.7 percentile

Execution stress:

- Cooldown: -19.17 PnL, only 2.13 trades/day
- Netted: +128.79 PnL, 78.76 active rows/day, final split negative

Current honest verdict: USDJPY is now the best evidence of a real mechanism.
It gives cross-pair confirmation that gold reacts to normalized FX/rates shock
states, and the forward shape is cleaner than EURUSD. It is still not
production-proven because the circular-control rank is below 95%, one old fold
is deeply negative, and the edge still depends on overlapping-signal accounting.
It is the best current research lead, not a deployable system yet.

Report:

`C:\Users\marti\from\reports\cross_market_spy_family_stability_nogap.json`

Older no-gap broader example at 2x spread:

- `SPY_fade`, window 6, q 0.35, London, hold 9 bars, cost 2x
  - all: +574.47, 32.8 trades/day
  - fold 1: +926.4
  - fold 2: -748.4
  - fold 3: +308.2
  - fold 4: +88.2

Market-quality filters were tested in:

`C:\Users\marti\from\reports\cross_market_spy_filter_search.json`

They did not cleanly remove the bad fold in the broader no-gap sample. The
corrected exact run is stronger, but still lacks a long enough 5m out-of-sample
history.

## Not Supported

- XAUUSD-only technical rules: repeatedly failed after costs
- Existing neural checkpoint: dead/flat behavior in adversarial validation
- 15m exact/no-gap cross-market target-20 search: no candidates
- 60m exact/no-gap target-frequency search: no candidates for 20/day

## Next Best Test

Acquire or import longer intraday leader data for:

- SPY or ES futures
- TLT or US Treasury futures
- GC futures
- DXY or EURUSD

Then rerun fixed, predeclared families over at least 1-2 years of 5m data.
After the forward holdout, SPY should be treated as killed unless longer data
shows this was a temporary regime failure. The best fresh lead worth further
research is now the USDJPY/XAU normalized macro-dislocation family, with EURUSD
as secondary confirmation and CL much weaker.
