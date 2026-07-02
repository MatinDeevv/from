# Fresh XAU/USD Edge Discovery Report

This is a clean-slate mechanism-first screen using the strict gates from the rebuilt pipeline. The prior roster is treated as dead; prior survivors are used only as sanity-check priors, not as models to preserve.

# Fresh XAU Edge Discovery - Phase 0 and Phase 1

## Phase 0 - Data Documentation

This fresh run uses only deterministic mechanism-family symbols, not the old full 50-symbol fishing universe.

Source files:
- Yahoo leader bars from `C:\Users\marti\from\data\cache\yahoo`
- XAU execution bars from `C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet`
- Forward XAU holdout from `C:\Users\marti\from\data\derived\xauusd_dukascopy_5m_forward_20260522_20260618.csv`

Data profile files:
- `C:\Users\marti\from\reports\fresh_edge_5m_mechanism_data_profile.csv`
- `C:\Users\marti\from\reports\fresh_edge_60m_mechanism_data_profile.csv`

Gap rule: missing bars are counted as null close values or duplicate timestamps inside the vendor-provided bar stream. Normal exchange closures are not counted as missing bars. Under that rule, all 16 planned symbols have 0.0% missing/duplicate bars in both 5m and 60m cache files. This does not prove a true pre-forward availability snapshot; it only makes the local data assumption explicit.

Universe construction rule: include only symbols named by the six predeclared mechanism families below. This rule is deterministic and could be written before the forward window: energy commodities, US risk sectors, silver, long-duration rates, safe-haven FX, and crypto liquidity/risk impulse.

Symbols:
- Energy / commodity complex: `UNG`, `USO`, `CL=F`
- US sector risk appetite: `XLF`, `XLV`, `XLY`, `XLK`
- Silver leads gold: `SI=F`
- Long-duration rates: `TLT`, `IEF`, `ZB=F`
- FX safe-haven divergence: `USDJPY=X`, `DX-Y.NYB`, `EURUSD=X`
- New mechanism, crypto liquidity/risk impulse: `BTC-USD`, `ETH-USD`

Hard-coded split dates:
- 5m train: `2026-03-24 08:00:00+00:00` to `2026-04-23 23:59:59+00:00`
- 5m validation: `2026-04-24 00:00:00+00:00` to `2026-05-08 23:59:59+00:00`
- 5m old test: `2026-05-09 00:00:00+00:00` to `2026-05-22 13:50:00+00:00`
- 5m forward: `2026-05-22 13:55:00+00:00` to `2026-06-17 23:55:00+00:00`
- 60m train: `2023-07-21 08:00:00+00:00` to `2025-07-01 00:00:00+00:00`
- 60m validation: `2025-07-01 00:00:00+00:00` to `2026-01-01 00:00:00+00:00`
- 60m old test: `2026-01-01 00:00:00+00:00` to `2026-05-22 12:59:59+00:00`
- 60m forward: `2026-05-22 13:00:00+00:00` to `2026-06-17 23:00:00+00:00`

Known limitation: the 5m forward window remains short and was already used to design stricter gates. Any survivor is in-sample under the new gate until a future untouched window or longer reconstructed leader history is available.

## Phase 1 - Mechanism-First Hypotheses

### 1. Energy / Commodity Complex Leads Gold

Lead-lag story: energy ETFs and crude futures can reprice global inflation/liquidity shocks before XAU fully reacts, especially around London hours when global commodity desks are active. If commodity vol-adjusted momentum is strong, gold may follow the broader commodity impulse.

Expected regime: inflation scare, commodity squeeze, or liquidity impulse. Expected strongest in London. It should be weaker in quiet risk-neutral sessions.

Expected sign: follow. A positive vol-adjusted energy move should imply positive XAU drift over the next several 5m bars; a negative energy move should imply negative XAU drift.

Falsification: forward PnL positive only in one late subperiod, bootstrap lower bound <= 0, or reverse-rule PnL not materially negative.

### 2. US Sector Risk-Appetite Fades Gold

Lead-lag story: US cyclicals/defensives can reflect equity-risk appetite that competes with defensive gold demand. When sectors surge on risk appetite, XAU can lag then fade; when sectors dump, XAU can catch a safe-haven bid.

Expected regime: equity risk-on/risk-off transitions, strongest in London/NY overlap or US hours. Sector-specific versions may differ: financials and consumer discretionary should be cleaner risk appetite proxies than health care.

Expected sign: fade. Positive sector vol-adjusted impulse should predict lower/lagging XAU; negative sector impulse should predict higher XAU.

Falsification: follow beats fade across the forward window, or fade only works because of one subperiod/concentrated event.

### 3. Silver Leads Gold

Lead-lag story: silver and gold share precious-metal drivers, but silver can move faster during commodity/risk liquidity impulses. CME silver momentum can lead short-horizon XAU repricing.

Expected regime: broad metals movement, inflation repricing, commodity liquidity shocks. All sessions are plausible because futures trade nearly around the clock.

Expected sign: follow. Strong SI=F volz should predict same-direction XAU movement.

Falsification: SI=F follow has fewer than 100 forward trades, bootstrap lower bound <= 0, or fade performs symmetrically well.

### 4. Long-Duration Rates Fade Gold

Lead-lag story: duration instruments react to real-rate and growth expectations. A bond rally can initially coexist with risk-off flows, but if the prior strict screen finds fade, the possible mechanism is short-term overreaction in duration relative to gold or rate-driven USD/liquidity repricing that gold absorbs inversely after a lag.

Expected regime: macro/rates repricing, not single-stock risk. All sessions are plausible for futures; ETF versions should be more US-session sensitive.

Expected sign: fade for TLT/IEF/ZB=F, matching the strict-gate survivor prior.

Falsification: effective signal requires fewer than 100 forward trades, only works in ETF hours with no futures confirmation, or follow and fade are both profitable.

### 5. FX Safe-Haven Divergence

Lead-lag story: USD and JPY moves can precede gold when safe-haven or dollar-liquidity flows hit FX first. DXY strength often pressures gold, while USDJPY can capture dollar/risk and yen safe-haven behavior. EURUSD is an inverse dollar proxy.

Expected regime: dollar liquidity, safe-haven FX shocks, macro data windows. London and NY should matter most.

Expected sign: DXY follow for USD strength should often imply XAU fade/down; EURUSD strength should often imply XAU follow/up if dollar weakness drives gold. USDJPY is ambiguous, so both follow and fade are tested.

Falsification: no direction has stable subperiods, or latency/reverse behavior suggests the signal is just timestamp alignment noise.

### 6. Crypto Liquidity / Risk Impulse

Lead-lag story: BTC/ETH trade continuously and can react quickly to global liquidity/risk impulses outside ETF market hours. If gold is repricing the same liquidity shock more slowly, crypto vol-adjusted momentum or cross-asset z-spread could lead XAU.

Expected regime: broad liquidity/risk-on shocks, weekend-to-Asia handoff, and dollar-liquidity moves. This is the new family because the prior strict survivors were concentrated in commodities/sectors/rates, leaving portfolio independence weak.

Expected sign: test both follow and fade, but the primary story is follow for liquidity risk-on and possible fade during stress deleveraging.

Falsification: high forward PnL but failed bootstrap/subperiod stability, or no improvement from the new momentum-ratio / EW z-spread features.


## Phase 2 - Full Grid Stats

Fresh discovery script: `C:\Users\marti\from\from\tools\fresh_edge_discovery.py`

Features tested: `volz`, `divergence`, `momentum_ratio`, `ew_zspread`, `session_volz`. 5m holds: h6 through h48 step 6. 60m holds: h1, h2, h3, h4, h6, h8. Minimum forward trade count: 100.

| TF | Full grid JSONL | Rows | Summary JSON | Individual survivors |
| --- | --- | --- | --- | --- |
| 5m | C:\Users\marti\from\reports\fresh_edge_5m_full_grid.jsonl | 57600 | C:\Users\marti\from\reports\fresh_edge_5m_summary.json | 42 |
| 60m | C:\Users\marti\from\reports\fresh_edge_60m_full_grid.jsonl | 43200 | C:\Users\marti\from\reports\fresh_edge_60m_summary.json | 0 |

| TF | Evaluated | Insufficient trades | Failed old gate | Failed bootstrap | Failed subperiod | Failed parameter | Failed reverse | Survived all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | 57600 | 0 | 57084 | 57555 | 51466 | 57558 | 57556 | 42 |
| 60m | 43200 | 36282 | 42751 | 43200 | 37215 | 43200 | 43200 | 0 |

Counts overlap. A row can fail multiple criteria. The `insufficient_forward_trades` rejection is applied before bootstrap/subperiod interpretation because fewer than 100 forward trades makes the statistical tests meaningless.

### Rejection Breakdown By Mechanism

#### 5m
| Mechanism | Evaluated | Insufficient | Old | Bootstrap | Subperiod | Parameter | Reverse | Survivors | Survival rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Energy / commodity complex leads gold | 3840 | 0 | 3698 | 3808 | 2159 | 3808 | 3808 | 32 | 0.833% |
| US sector risk-appetite fades gold | 10240 | 0 | 10092 | 10228 | 9281 | 10230 | 10229 | 10 | 0.098% |
| Silver leads gold | 1280 | 0 | 1280 | 1280 | 1280 | 1280 | 1280 | 0 | 0.000% |
| Long-duration rates fade gold | 3840 | 0 | 3840 | 3840 | 3646 | 3840 | 3840 | 0 | 0.000% |
| FX safe-haven divergence | 23040 | 0 | 22824 | 23039 | 21474 | 23040 | 23039 | 0 | 0.000% |
| Crypto liquidity / risk impulse | 15360 | 0 | 15350 | 15360 | 13626 | 15360 | 15360 | 0 | 0.000% |

#### 60m
| Mechanism | Evaluated | Insufficient | Old | Bootstrap | Subperiod | Parameter | Reverse | Survivors | Survival rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Energy / commodity complex leads gold | 2880 | 2778 | 2880 | 2880 | 2524 | 2880 | 2880 | 0 | 0.000% |
| US sector risk-appetite fades gold | 7680 | 7680 | 7670 | 7680 | 6530 | 7680 | 7680 | 0 | 0.000% |
| Silver leads gold | 960 | 642 | 893 | 960 | 857 | 960 | 960 | 0 | 0.000% |
| Long-duration rates fade gold | 2880 | 2622 | 2745 | 2880 | 2386 | 2880 | 2880 | 0 | 0.000% |
| FX safe-haven divergence | 17280 | 13840 | 17160 | 17280 | 14694 | 17280 | 17280 | 0 | 0.000% |
| Crypto liquidity / risk impulse | 11520 | 8720 | 11403 | 11520 | 10224 | 11520 | 11520 | 0 | 0.000% |

## Phase 2 - Survivor Table

### 5m: 42 individual survivors
| Name | Mechanism | Leader | Feature | Mode | W | Q | Session | Hold | Train | Val | Old test | Forward | Trades | PF | Boot low | Boot high | Neg periods | Max share | Reverse | Param fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNG_5m_session_volz_follow_w18_q0.65_london_h30_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 18 | 0.65 | london | 30 | $3,179.01 | $779.23 | $443.43 | $7,708.72 | 964 | 2.27 | $2,355.71 | $13,981.00 | 1 | 0.47 | $-9,909.78 | 0 |
| UNG_5m_session_volz_follow_w18_q0.65_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 18 | 0.65 | london | 36 | $5,818.04 | $1,478.39 | $58.65 | $7,109.54 | 964 | 1.89 | $754.04 | $14,838.16 | 1 | 0.50 | $-9,310.60 | 0 |
| UNG_5m_momentum_ratio_follow_w18_q0.55_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | momentum_ratio | follow | 18 | 0.55 | london | 36 | $6,424.56 | $176.55 | $567.04 | $6,933.25 | 889 | 1.90 | $376.49 | $14,027.92 | 1 | 0.52 | $-8,929.47 | 0 |
| UNG_5m_volz_follow_w18_q0.55_london_h30_cost2.0 | Energy / commodity complex leads gold | UNG | volz | follow | 18 | 0.55 | london | 30 | $2,320.08 | $374.78 | $473.20 | $6,730.28 | 813 | 2.35 | $2,304.88 | $12,216.27 | 1 | 0.53 | $-8,587.21 | 0 |
| XLF_5m_divergence_fade_w24_q0.65_london_h48_cost2.0 | US sector risk-appetite fades gold | XLF | divergence | fade | 24 | 0.65 | london | 48 | $6,186.29 | $544.87 | $607.37 | $6,667.69 | 657 | 2.07 | $1,020.95 | $12,251.58 | 1 | 0.46 | $-8,188.43 | 0 |
| UNG_5m_session_volz_follow_w12_q0.65_london_h30_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 12 | 0.65 | london | 30 | $998.74 | $310.49 | $193.19 | $6,476.05 | 896 | 2.04 | $2,125.12 | $11,506.88 | 0 | 0.49 | $-8,520.30 | 0 |
| UNG_5m_volz_follow_w18_q0.55_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | volz | follow | 18 | 0.55 | london | 36 | $5,011.66 | $1,050.67 | $161.28 | $6,438.46 | 813 | 1.99 | $554.51 | $12,954.31 | 1 | 0.53 | $-8,295.39 | 0 |
| UNG_5m_session_volz_follow_w9_q0.65_london_h30_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 9 | 0.65 | london | 30 | $81.31 | $47.57 | $139.90 | $5,833.25 | 875 | 1.90 | $1,876.08 | $10,335.69 | 0 | 0.49 | $-7,827.70 | 0 |
| UNG_5m_session_volz_follow_w12_q0.65_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 12 | 0.65 | london | 36 | $3,888.19 | $1,195.57 | $38.72 | $5,746.79 | 896 | 1.72 | $143.28 | $11,928.04 | 1 | 0.52 | $-7,791.04 | 0 |
| XLF_5m_divergence_fade_w24_q0.65_london_h42_cost2.0 | US sector risk-appetite fades gold | XLF | divergence | fade | 24 | 0.65 | london | 42 | $5,580.75 | $659.69 | $293.73 | $5,292.37 | 657 | 1.91 | $100.50 | $10,205.78 | 1 | 0.47 | $-6,813.11 | 0 |
| USO_5m_volz_follow_w12_q0.55_london_h36_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 12 | 0.55 | london | 36 | $578.83 | $17.94 | $66.54 | $5,276.23 | 739 | 1.98 | $2,719.09 | $7,889.51 | 0 | 0.39 | $-6,993.61 | 0 |
| UNG_5m_session_volz_follow_w24_q0.75_london_h24_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 24 | 0.75 | london | 24 | $1,608.02 | $28.10 | $72.49 | $4,814.33 | 668 | 2.30 | $865.41 | $9,308.33 | 0 | 0.64 | $-6,331.32 | 0 |
| USO_5m_momentum_ratio_follow_w18_q0.55_london_h30_cost2.0 | Energy / commodity complex leads gold | USO | momentum_ratio | follow | 18 | 0.55 | london | 30 | $1,555.30 | $750.92 | $22.13 | $4,652.84 | 654 | 2.05 | $1,184.52 | $8,406.42 | 0 | 0.51 | $-6,094.56 | 0 |
| USO_5m_volz_follow_w12_q0.55_london_h42_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 12 | 0.55 | london | 42 | $2,711.43 | $108.40 | $317.28 | $4,592.57 | 739 | 1.68 | $2,111.53 | $7,187.41 | 0 | 0.36 | $-6,309.94 | 0 |
| USO_5m_volz_follow_w9_q0.55_london_h42_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 9 | 0.55 | london | 42 | $3,749.54 | $19.92 | $138.87 | $4,472.78 | 730 | 1.64 | $1,644.69 | $7,106.90 | 0 | 0.35 | $-6,172.34 | 0 |
| XLY_5m_divergence_fade_w24_q0.55_london_h30_cost2.0 | US sector risk-appetite fades gold | XLY | divergence | fade | 24 | 0.55 | london | 30 | $5,695.80 | $1,057.21 | $336.13 | $4,444.25 | 633 | 1.79 | $392.99 | $8,336.32 | 1 | 0.63 | $-5,913.96 | 0 |
| USO_5m_session_volz_follow_w12_q0.75_london_h42_cost2.0 | Energy / commodity complex leads gold | USO | session_volz | follow | 12 | 0.75 | london | 42 | $1,989.05 | $40.87 | $140.44 | $4,438.09 | 638 | 1.82 | $2,153.34 | $7,078.75 | 0 | 0.47 | $-5,928.43 | 0 |
| USO_5m_session_volz_follow_w9_q0.65_london_h42_cost2.0 | Energy / commodity complex leads gold | USO | session_volz | follow | 9 | 0.65 | london | 42 | $4,902.62 | $256.11 | $62.09 | $4,337.34 | 906 | 1.46 | $1,093.57 | $7,507.50 | 0 | 0.37 | $-6,434.01 | 0 |
| XLF_5m_volz_fade_w24_q0.65_london_h36_cost2.0 | US sector risk-appetite fades gold | XLF | volz | fade | 24 | 0.65 | london | 36 | $3,868.40 | $677.17 | $273.86 | $4,337.19 | 481 | 2.55 | $1,025.59 | $7,940.09 | 1 | 0.67 | $-5,440.29 | 0 |
| USO_5m_session_volz_follow_w9_q0.75_london_h42_cost2.0 | Energy / commodity complex leads gold | USO | session_volz | follow | 9 | 0.75 | london | 42 | $2,888.91 | $335.93 | $192.37 | $4,310.77 | 630 | 1.76 | $1,896.47 | $6,730.41 | 0 | 0.34 | $-5,779.78 | 0 |
| USO_5m_volz_follow_w9_q0.65_london_h42_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 9 | 0.65 | london | 42 | $2,686.91 | $68.89 | $143.24 | $4,300.98 | 575 | 1.85 | $1,999.40 | $6,743.82 | 0 | 0.45 | $-5,646.21 | 0 |
| XLY_5m_ew_zspread_fade_w18_q0.55_london_h36_cost2.0 | US sector risk-appetite fades gold | XLY | ew_zspread | fade | 18 | 0.55 | london | 36 | $3,745.89 | $464.99 | $306.96 | $4,110.63 | 653 | 1.60 | $18.27 | $8,017.73 | 1 | 0.59 | $-5,624.67 | 0 |
| USO_5m_volz_follow_w4_q0.55_london_h48_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 4 | 0.55 | london | 48 | $2,447.20 | $146.97 | $127.22 | $4,076.44 | 766 | 1.44 | $1,296.50 | $7,068.36 | 1 | 0.45 | $-5,854.07 | 0 |
| XLY_5m_divergence_fade_w24_q0.55_london_h24_cost2.0 | US sector risk-appetite fades gold | XLY | divergence | fade | 24 | 0.55 | london | 24 | $3,818.01 | $321.00 | $559.49 | $4,029.57 | 633 | 1.86 | $683.57 | $7,675.07 | 0 | 0.60 | $-5,499.29 | 0 |
| XLF_5m_session_volz_fade_w18_q0.75_london_h42_cost2.0 | US sector risk-appetite fades gold | XLF | session_volz | fade | 18 | 0.75 | london | 42 | $2,905.89 | $182.80 | $486.34 | $3,989.25 | 545 | 1.89 | $70.03 | $8,065.76 | 1 | 0.57 | $-5,245.31 | 0 |
| USO_5m_momentum_ratio_follow_w24_q0.55_london_h24_cost2.0 | Energy / commodity complex leads gold | USO | momentum_ratio | follow | 24 | 0.55 | london | 24 | $2,211.98 | $605.24 | $1.56 | $3,933.69 | 617 | 2.12 | $186.22 | $7,955.41 | 0 | 0.53 | $-5,279.37 | 0 |
| UNG_5m_session_volz_follow_w6_q0.65_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | session_volz | follow | 6 | 0.65 | london | 36 | $1,176.65 | $825.68 | $139.43 | $3,893.33 | 894 | 1.44 | $145.71 | $7,973.85 | 0 | 0.49 | $-5,926.96 | 0 |
| XLY_5m_divergence_fade_w18_q0.55_london_h30_cost2.0 | US sector risk-appetite fades gold | XLY | divergence | fade | 18 | 0.55 | london | 30 | $2,570.18 | $628.97 | $723.55 | $3,762.36 | 610 | 1.65 | $977.15 | $6,815.67 | 1 | 0.54 | $-5,181.34 | 0 |
| USO_5m_session_volz_follow_w4_q0.75_london_h48_cost2.0 | Energy / commodity complex leads gold | USO | session_volz | follow | 4 | 0.75 | london | 48 | $2,859.68 | $65.21 | $161.75 | $3,695.46 | 688 | 1.44 | $908.10 | $6,322.77 | 1 | 0.47 | $-5,294.68 | 0 |
| USO_5m_momentum_ratio_follow_w18_q0.65_london_h30_cost2.0 | Energy / commodity complex leads gold | USO | momentum_ratio | follow | 18 | 0.65 | london | 30 | $106.84 | $828.96 | $53.77 | $3,668.12 | 524 | 2.01 | $569.82 | $6,591.30 | 0 | 0.50 | $-4,819.86 | 0 |
| XLY_5m_divergence_fade_w12_q0.55_london_h36_cost2.0 | US sector risk-appetite fades gold | XLY | divergence | fade | 12 | 0.55 | london | 36 | $2,538.67 | $274.44 | $609.06 | $3,587.36 | 593 | 1.59 | $369.95 | $6,642.73 | 0 | 0.41 | $-4,958.37 | 0 |
| UNG_5m_volz_follow_w6_q0.55_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | volz | follow | 6 | 0.55 | london | 36 | $1,354.46 | $503.54 | $86.05 | $3,519.83 | 776 | 1.46 | $204.94 | $7,113.64 | 0 | 0.59 | $-5,293.09 | 0 |
| XLY_5m_session_volz_fade_w24_q0.75_london_h36_cost2.0 | US sector risk-appetite fades gold | XLY | session_volz | fade | 24 | 0.75 | london | 36 | $3,281.60 | $233.15 | $1,115.00 | $3,462.99 | 527 | 1.66 | $57.92 | $7,153.43 | 1 | 0.57 | $-4,676.39 | 0 |
| USO_5m_session_volz_follow_w18_q0.75_london_h18_cost2.0 | Energy / commodity complex leads gold | USO | session_volz | follow | 18 | 0.75 | london | 18 | $519.98 | $96.02 | $32.11 | $3,171.27 | 683 | 2.11 | $981.13 | $5,488.78 | 0 | 0.55 | $-4,728.25 | 0 |
| USO_5m_volz_follow_w12_q0.55_london_h24_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 12 | 0.55 | london | 24 | $68.03 | $366.75 | $21.23 | $3,159.40 | 739 | 1.70 | $764.01 | $5,648.78 | 1 | 0.54 | $-4,876.78 | 0 |
| UNG_5m_volz_follow_w4_q0.65_london_h36_cost2.0 | Energy / commodity complex leads gold | UNG | volz | follow | 4 | 0.65 | london | 36 | $263.52 | $297.67 | $7.97 | $3,135.62 | 607 | 1.50 | $599.25 | $5,927.00 | 0 | 0.51 | $-4,523.92 | 0 |
| USO_5m_momentum_ratio_follow_w18_q0.65_london_h24_cost2.0 | Energy / commodity complex leads gold | USO | momentum_ratio | follow | 18 | 0.65 | london | 24 | $342.30 | $668.42 | $116.53 | $2,788.24 | 524 | 1.90 | $349.27 | $5,352.87 | 0 | 0.46 | $-3,939.98 | 0 |
| USO_5m_volz_follow_w2_q0.55_london_h48_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 2 | 0.55 | london | 48 | $592.30 | $123.76 | $278.14 | $2,779.76 | 771 | 1.27 | $459.42 | $5,148.89 | 0 | 0.60 | $-4,563.01 | 0 |
| USO_5m_session_volz_follow_w9_q0.75_london_h24_cost2.0 | Energy / commodity complex leads gold | USO | session_volz | follow | 9 | 0.75 | london | 24 | $460.04 | $423.34 | $77.44 | $2,765.20 | 630 | 1.72 | $748.43 | $5,092.48 | 0 | 0.51 | $-4,234.21 | 0 |
| USO_5m_momentum_ratio_follow_w18_q0.55_london_h18_cost2.0 | Energy / commodity complex leads gold | USO | momentum_ratio | follow | 18 | 0.55 | london | 18 | $871.96 | $345.74 | $9.87 | $2,411.81 | 654 | 1.75 | $497.79 | $4,288.34 | 1 | 0.47 | $-3,853.52 | 0 |
| USO_5m_volz_follow_w9_q0.65_london_h24_cost2.0 | Energy / commodity complex leads gold | USO | volz | follow | 9 | 0.65 | london | 24 | $327.54 | $268.52 | $8.35 | $2,398.99 | 575 | 1.67 | $375.80 | $4,650.17 | 1 | 0.61 | $-3,744.22 | 0 |
| USO_5m_momentum_ratio_follow_w18_q0.65_london_h18_cost2.0 | Energy / commodity complex leads gold | USO | momentum_ratio | follow | 18 | 0.65 | london | 18 | $404.42 | $412.47 | $163.46 | $1,907.92 | 524 | 1.73 | $175.12 | $3,803.83 | 1 | 0.44 | $-3,059.67 | 0 |

### 60m: 0 individual survivors
No individual candidates survived all strict gates.

## Phase 3 - Portfolio Verdict

| TF | Portfolio PASS | Selected models | Mechanism families | Allocator swing | Effective bets | Kaiser >1 | Equity curve CSV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | False | 4 | 2 | 0.414 | 2.093 | 1 | C:\Users\marti\from\reports\fresh_edge_5m_equity_curve.csv |
| 60m | False | 0 | 0 | n/a | 0.000 | n/a | C:\Users\marti\from\reports\fresh_edge_60m_equity_curve.csv |

### 5m allocator policies
| Policy | PnL | Return | PF | Max DD | Trades |
| --- | --- | --- | --- | --- | --- |
| edge_rank | $459,288.04 | 9.19% | 2.09 | $107,445.46 | 195 |
| inverse_vol | $268,990.75 | 5.38% | 1.58 | $123,904.17 | 200 |
| name_order | $268,990.75 | 5.38% | 1.58 | $123,904.17 | 200 |
Selected models: UNG_5m_session_volz_follow_w18_q0.65_london_h30_cost2.0, XLF_5m_divergence_fade_w24_q0.65_london_h48_cost2.0, USO_5m_volz_follow_w12_q0.55_london_h36_cost2.0, XLY_5m_divergence_fade_w24_q0.55_london_h30_cost2.0

### 60m allocator policies
No portfolio was built because no individual survivors existed.

Portfolio verdict: **FAIL**. 5m fails because survivors represent only 2 mechanism families, effective bets are 2.093 < 5, and allocator swing is 0.414 > 0.30. 60m fails because there are zero individual survivors.

## Phase 4 - Mechanism Autopsy

### Energy / commodity complex leads gold
5m: top20 positive 20/20, survived 1/20, bootstrap-low near zero (> -$500) 1/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:19, bootstrap:19, reverse:19 60m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, reverse:20
Autopsy: alive as an individual 5m research lead. It produced most 5m survivors, especially UNG session-local/volz/momentum-ratio follow in London. It is not a portfolio by itself because it is one crowded mechanism family.

### US sector risk-appetite fades gold
5m: top20 positive 20/20, survived 1/20, bootstrap-low near zero (> -$500) 1/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:19, bootstrap:19, reverse:19 60m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers insufficient_forward_trades:20, old_gate:20, bootstrap:20
Autopsy: alive but narrower. XLF/XLY-style sector fade generated 5m survivors and gives some diversity versus energy, but not enough to pass portfolio independence or allocator stability.

### Silver leads gold
5m: top20 positive 8/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, subperiod:20 60m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers bootstrap:20, reverse:20, parameter:20
Autopsy: dead in this forward window under the stricter 100-trade/bootstrap/parameter criteria. 60m also fails the trade-count and bootstrap gates.

### Long-duration rates fade gold
5m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, reverse:20 60m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers insufficient_forward_trades:20, bootstrap:20, reverse:20
Autopsy: dead in this fresh screen despite prior strict 60m hints. The new 100-forward-trade rule and strict bootstrap gate eliminate it.

### FX safe-haven divergence
5m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, reverse:20 60m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, reverse:20
Autopsy: possibly alive only with longer/better data, not with current evidence. Many rows have positive forward PnL, but the top candidates fail bootstrap/parameter/reverse gates, so this is not tradable evidence.

### Crypto liquidity / risk impulse
5m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, reverse:20 60m: top20 positive 20/20, survived 0/20, bootstrap-low near zero (> -$500) 0/20, median bootstrap low $-1,000,000,000.00, main killers old_gate:20, bootstrap:20, reverse:20
Autopsy: dead as tested. The new family did not supply independent survivors or a portfolio-diversifying leg.

## Single Honest Conclusion

**Found 42 individual 5m survivors but no valid portfolio due to mechanism concentration, low effective independence, and allocator instability. The best live research lead is the 5m energy / commodity complex London follow family, with US sector risk-appetite fade as the only useful secondary family.**

The result is not deployable. It is an in-sample strict-gate survivor set on a short forward window that was already used to design the methodology. The next real step is longer 5m leader history and a future untouched forward window, not another threshold tweak.