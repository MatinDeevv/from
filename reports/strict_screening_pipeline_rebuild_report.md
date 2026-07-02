# Strict XAU Screening Pipeline Rebuild

Generated on 2026-06-18 from a fresh rerun of the rebuilt screening pipeline. This report follows `C:\Users\marti\Downloads\codex_rebuild_screening_pipeline_prompt.md` and treats all survivors as in-sample survivors of a stricter gate, not validated live edge.

## What Changed

- Added `C:\Users\marti\from\from\tools\strict_screening_pipeline.py`.
- Persisted every evaluated grid row to JSONL, including rejected rows.
- Added a timestamped universe snapshot file per timeframe.
- Added fixed hard gates: old train/val/test/forward positive gate, block-bootstrap lower bound, subperiod consistency, parameter perturbation robustness, and reverse-rule check.
- Added portfolio-level allocator checks across edge-ranked, inverse-volatility, and name-order policies.
- Added effective-bet participation-ratio check for final portfolio concentration.
- Preserved the required framing: any survivor is not out-of-sample validated because this stricter gate was designed after seeing the killed roster.

## Command Log

```powershell
& 'C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile tools/strict_screening_pipeline.py
& 'C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools/strict_screening_pipeline.py --timeframe 60m --out-prefix C:\Users\marti\from\reports\strict_screen_60m_rebuild --bootstrap-trials 2000
# 60m exit code: 0; grid rows: 117,504; survivors: 3
& 'C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools/strict_screening_pipeline.py --timeframe 5m --out-prefix C:\Users\marti\from\reports\strict_screen_5m_rebuild --bootstrap-trials 2000
# 5m exit code: 0; grid rows: 112,896; survivors: 22
```

## Output Files

| TF | Full persisted grid | Row count | Expected full-grid rows | Summary JSON | Universe snapshot |
| --- | --- | --- | --- | --- | --- |
| 5m | C:\Users\marti\from\reports\strict_screen_5m_rebuild_full_grid.jsonl | 112896 | 112896 | C:\Users\marti\from\reports\strict_screen_5m_rebuild_summary.json | C:\Users\marti\from\reports\strict_screen_5m_rebuild_universe_snapshot.json |
| 60m | C:\Users\marti\from\reports\strict_screen_60m_rebuild_full_grid.jsonl | 117504 | 117504 | C:\Users\marti\from\reports\strict_screen_60m_rebuild_summary.json | C:\Users\marti\from\reports\strict_screen_60m_rebuild_universe_snapshot.json |

The row counts match the expected grid sizes for all included symbols. The 5m grid is 49 symbols x 2 features x 8 windows x 4 quantiles x 2 modes x 3 sessions x 6 holds x 1 cost = 112,896 rows. The 60m grid is 51 symbols with the same dimensionality = 117,504 rows.

## Universe Snapshot Caveat

5m: included 49/49 requested symbols. Snapshot policy: Uses local Yahoo cache files already present before this rebuild run; not a true pre-forward vendor snapshot.
60m: included 51/51 requested symbols. Snapshot policy: Uses local Yahoo cache files already present before this rebuild run; not a true pre-forward vendor snapshot.

This is improved because the inclusion/exclusion list is explicit and saved, but it is still **not** a true pre-forward vendor availability snapshot. The report therefore keeps the survivorship-bias caveat open.

## Rejection Breakdown

| TF | Evaluated | Failed old gate | Failed bootstrap | Failed subperiod | Failed parameter | Failed reverse | Survived all individual gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | 112896 | 112643 | 112873 | 105337 | 112874 | 112643 | 22 |
| 60m | 117504 | 116415 | 117496 | 105886 | 117501 | 116415 | 3 |

Counts are overlapping, not mutually exclusive. A candidate can fail multiple gates. For runtime practicality, the expensive 2000-resample bootstrap and parameter perturbation are only fully run for rows that first pass the old economic gate and cheaper preconditions; rows short-circuited before those tests are still logged as rejected in the full grid.

## Full Grid Sample Rows

### 5m sample
| Name | Old | Bootstrap | Subperiod | Parameter | Reverse | Forward PnL | Forward trades | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACWI_5m_volz_follow_w2_q0.55_all_h6_cost2.0 | False | False | False | False | False | $-1,372.86 | 827 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_5m_volz_follow_w2_q0.55_all_h12_cost2.0 | False | False | False | False | False | $-1,335.60 | 821 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_5m_volz_follow_w2_q0.55_all_h18_cost2.0 | False | False | False | False | False | $-2,480.86 | 815 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_5m_volz_follow_w2_q0.55_all_h24_cost2.0 | False | False | False | False | False | $-2,789.53 | 810 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_5m_volz_follow_w2_q0.55_all_h30_cost2.0 | False | False | False | False | False | $-2,928.28 | 806 | old_gate,bootstrap,subperiod,parameter,reverse |

### 60m sample
| Name | Old | Bootstrap | Subperiod | Parameter | Reverse | Forward PnL | Forward trades | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACWI_60m_volz_follow_w2_q0.55_all_h1_cost2.0 | False | False | False | False | False | $-82.24 | 18 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_60m_volz_follow_w2_q0.55_all_h2_cost2.0 | False | False | False | False | False | $-310.49 | 18 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_60m_volz_follow_w2_q0.55_all_h3_cost2.0 | False | False | False | False | False | $-516.22 | 18 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_60m_volz_follow_w2_q0.55_all_h4_cost2.0 | False | False | False | False | False | $-626.08 | 18 | old_gate,bootstrap,subperiod,parameter,reverse |
| ACWI_60m_volz_follow_w2_q0.55_all_h6_cost2.0 | False | False | False | False | False | $-383.50 | 18 | old_gate,bootstrap,subperiod,parameter,reverse |

## Individual Survivors

### 5m: 22 individual survivors
| Name | Leader | Feature | Mode | W | Q | Session | Hold | Train PnL | Val PnL | Old-test PnL | Forward PnL | Fwd trades | Fwd PF | Bootstrap CI low | Bootstrap CI high | Neg subperiods | Max subperiod share | Reverse PnL | Param failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNG_5m_volz_follow_w18_q0.55_london_h30_cost2.0 | UNG | volz | follow | 18 | 0.55 | london | 30 | $2,271.44 | $394.35 | $886.92 | $6,527.10 | 798 | 2.34 | $1,801.98 | $12,145.56 | 1 | 0.54 | $-8,351.57 | 0 |
| USO_5m_volz_follow_w24_q0.55_london_h24_cost2.0 | USO | volz | follow | 24 | 0.55 | london | 24 | $1,472.99 | $5.36 | $75.46 | $6,285.39 | 779 | 3.09 | $3,565.22 | $8,820.52 | 0 | 0.36 | $-8,054.26 | 0 |
| UNG_5m_volz_follow_w18_q0.55_london_h36_cost2.0 | UNG | volz | follow | 18 | 0.55 | london | 36 | $4,842.68 | $830.81 | $661.05 | $6,178.17 | 798 | 1.96 | $300.90 | $12,831.57 | 1 | 0.55 | $-8,002.63 | 0 |
| USO_5m_volz_follow_w18_q0.55_london_h30_cost2.0 | USO | volz | follow | 18 | 0.55 | london | 30 | $841.83 | $207.51 | $263.36 | $6,158.91 | 735 | 2.59 | $3,392.92 | $9,105.75 | 0 | 0.38 | $-7,844.29 | 0 |
| UNG_5m_volz_follow_w18_q0.55_london_h24_cost2.0 | UNG | volz | follow | 18 | 0.55 | london | 24 | $142.37 | $0.94 | $671.30 | $5,954.98 | 798 | 2.39 | $1,847.48 | $10,595.32 | 0 | 0.57 | $-7,779.45 | 0 |
| UNG_5m_volz_follow_w12_q0.55_london_h36_cost2.0 | UNG | volz | follow | 12 | 0.55 | london | 36 | $1,864.50 | $789.77 | $400.35 | $5,481.85 | 768 | 1.83 | $125.05 | $11,234.58 | 1 | 0.54 | $-7,236.77 | 0 |
| UNG_5m_volz_follow_w9_q0.55_london_h36_cost2.0 | UNG | volz | follow | 9 | 0.55 | london | 36 | $2,749.02 | $487.28 | $8.28 | $5,254.50 | 775 | 1.75 | $903.96 | $9,933.59 | 0 | 0.50 | $-7,030.26 | 0 |
| USO_5m_volz_follow_w18_q0.55_london_h24_cost2.0 | USO | volz | follow | 18 | 0.55 | london | 24 | $652.29 | $306.10 | $31.46 | $5,139.04 | 735 | 2.46 | $2,514.73 | $8,022.38 | 0 | 0.43 | $-6,824.43 | 0 |
| UNG_5m_volz_follow_w24_q0.65_london_h30_cost2.0 | UNG | volz | follow | 24 | 0.65 | london | 30 | $3,509.28 | $112.78 | $541.12 | $4,629.06 | 635 | 2.05 | $84.27 | $9,487.51 | 0 | 0.65 | $-6,069.24 | 0 |
| UNG_5m_volz_follow_w24_q0.65_london_h24_cost2.0 | UNG | volz | follow | 24 | 0.65 | london | 24 | $1,273.56 | $272.35 | $475.87 | $4,445.08 | 635 | 2.26 | $813.23 | $8,676.11 | 0 | 0.65 | $-5,885.26 | 0 |
| XLF_5m_volz_fade_w24_q0.65_london_h36_cost2.0 | XLF | volz | fade | 24 | 0.65 | london | 36 | $3,839.79 | $1,121.76 | $136.70 | $4,420.28 | 490 | 2.55 | $1,000.03 | $8,159.30 | 0 | 0.67 | $-5,544.63 | 0 |
| UNG_5m_volz_follow_w9_q0.65_london_h36_cost2.0 | UNG | volz | follow | 9 | 0.65 | london | 36 | $1,651.20 | $543.77 | $67.85 | $4,309.39 | 624 | 1.76 | $422.62 | $8,872.23 | 0 | 0.59 | $-5,730.00 | 0 |
| UNG_5m_volz_follow_w6_q0.75_all_h36_cost2.0 | UNG | volz | follow | 6 | 0.75 | all | 36 | $772.41 | $177.89 | $453.51 | $4,243.80 | 602 | 1.76 | $874.76 | $7,582.30 | 0 | 0.41 | $-5,650.15 | 0 |
| XLV_5m_volz_follow_w24_q0.75_all_h36_cost2.0 | XLV | volz | follow | 24 | 0.75 | all | 36 | $2,490.96 | $1,289.47 | $638.47 | $4,169.00 | 645 | 1.73 | $106.75 | $8,378.10 | 1 | 0.59 | $-5,754.12 | 0 |
| UNG_5m_volz_follow_w9_q0.55_london_h24_cost2.0 | UNG | volz | follow | 9 | 0.55 | london | 24 | $222.78 | $91.43 | $7.57 | $3,986.82 | 775 | 1.75 | $768.57 | $7,554.68 | 0 | 0.59 | $-5,762.58 | 0 |
| XLV_5m_volz_follow_w24_q0.75_all_h30_cost2.0 | XLV | volz | follow | 24 | 0.75 | all | 30 | $1,747.93 | $312.77 | $476.13 | $3,922.28 | 651 | 1.75 | $534.70 | $7,622.36 | 1 | 0.52 | $-5,525.41 | 0 |
| XLY_5m_volz_fade_w18_q0.55_london_h36_cost2.0 | XLY | volz | fade | 18 | 0.55 | london | 36 | $1,765.43 | $782.25 | $1,176.89 | $3,743.56 | 516 | 1.75 | $659.59 | $7,298.72 | 0 | 0.50 | $-4,930.50 | 0 |
| XLY_5m_volz_fade_w18_q0.55_london_h30_cost2.0 | XLY | volz | fade | 18 | 0.55 | london | 30 | $2,438.45 | $247.11 | $1,452.05 | $3,648.52 | 516 | 1.89 | $1,212.52 | $6,425.64 | 1 | 0.43 | $-4,835.46 | 0 |
| UNG_5m_volz_follow_w6_q0.55_london_h36_cost2.0 | UNG | volz | follow | 6 | 0.55 | london | 36 | $1,438.29 | $420.83 | $146.52 | $3,489.51 | 760 | 1.46 | $363.43 | $7,216.40 | 0 | 0.57 | $-5,226.85 | 0 |
| UNG_5m_volz_follow_w6_q0.65_london_h36_cost2.0 | UNG | volz | follow | 6 | 0.65 | london | 36 | $1,606.51 | $390.95 | $110.58 | $3,169.31 | 608 | 1.53 | $78.24 | $6,508.95 | 0 | 0.56 | $-4,556.28 | 0 |
| XLY_5m_volz_fade_w18_q0.55_london_h24_cost2.0 | XLY | volz | fade | 18 | 0.55 | london | 24 | $2,517.40 | $182.98 | $1,023.14 | $3,111.28 | 516 | 1.93 | $1,068.83 | $5,139.38 | 0 | 0.34 | $-4,298.22 | 0 |
| XLY_5m_volz_fade_w18_q0.55_london_h18_cost2.0 | XLY | volz | fade | 18 | 0.55 | london | 18 | $1,870.41 | $307.06 | $673.14 | $2,010.89 | 516 | 1.67 | $117.60 | $3,859.62 | 0 | 0.31 | $-3,197.83 | 0 |

### 60m: 3 individual survivors
| Name | Leader | Feature | Mode | W | Q | Session | Hold | Train PnL | Val PnL | Old-test PnL | Forward PnL | Fwd trades | Fwd PF | Bootstrap CI low | Bootstrap CI high | Neg subperiods | Max subperiod share | Reverse PnL | Param failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SI=F_60m_volz_follow_w12_q0.75_all_h6_cost2.0 | SI=F | volz | follow | 12 | 0.75 | all | 6 | $382.11 | $1,796.94 | $5,915.79 | $907.64 | 83 | 2.56 | $9.53 | $1,866.40 | 1 | 0.56 | $-1,130.77 | 0 |
| TLT_60m_volz_fade_w3_q0.65_all_h2_cost2.0 | TLT | volz | fade | 3 | 0.65 | all | 2 | $269.75 | $742.61 | $678.34 | $452.59 | 36 | 3.74 | $93.47 | $922.45 | 0 | 0.75 | $-541.52 | 0 |
| IEF_60m_volz_fade_w4_q0.65_all_h2_cost2.0 | IEF | volz | fade | 4 | 0.65 | all | 2 | $280.14 | $893.10 | $1,177.03 | $224.06 | 26 | 2.87 | $67.96 | $402.97 | 1 | 0.47 | $-286.62 | 0 |

## Portfolio Checks

| TF | Selected models | Allocator pass | Allocator swing | Effective bets | Kaiser components >1 | Portfolio verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 5m | 5 | False | 0.326 | 3.303 | 2 | FAIL |
| 60m | 3 | True | 0.106 | 2.611 | 2 | FAIL |

### 5m allocator policies
| Policy | PnL | Return | PF | Max DD | Trades |
| --- | --- | --- | --- | --- | --- |
| edge_rank | $395,261.70 | 7.91% | 1.54 | $111,136.99 | 353 |
| inverse_vol | $289,244.93 | 5.78% | 1.36 | $112,144.55 | 380 |
| name_order | $266,401.43 | 5.33% | 1.33 | $112,660.86 | 363 |

Selected model names: UNG_5m_volz_follow_w18_q0.55_london_h30_cost2.0, USO_5m_volz_follow_w24_q0.55_london_h24_cost2.0, XLF_5m_volz_fade_w24_q0.65_london_h36_cost2.0, XLV_5m_volz_follow_w24_q0.75_all_h36_cost2.0, XLY_5m_volz_fade_w18_q0.55_london_h36_cost2.0

### 60m allocator policies
| Policy | PnL | Return | PF | Max DD | Trades |
| --- | --- | --- | --- | --- | --- |
| edge_rank | $370,589.92 | 7.41% | 2.79 | $69,651.55 | 89 |
| inverse_vol | $400,414.53 | 8.01% | 3.11 | $69,651.55 | 91 |
| name_order | $361,147.13 | 7.22% | 2.76 | $70,681.48 | 88 |

Selected model names: SI=F_60m_volz_follow_w12_q0.75_all_h6_cost2.0, TLT_60m_volz_fade_w3_q0.65_all_h2_cost2.0, IEF_60m_volz_fade_w4_q0.65_all_h2_cost2.0

## Interpretation

The rebuilt gate is doing what the prompt asked: most old-style positives do not survive stability, bootstrap, and perturbation checks. The dominant failure modes are old-gate failure, parameter-gate failure, and bootstrap lower-bound failure. Subperiod failure is also widespread.

The 5m screen produced 22 individual survivors, but the final portfolio fails because the allocator swing is 0.326, above the 0.30 limit, and effective bets are only 3.303, below the required 5. The script selected only 5 models under the de-correlation preference, so there is no valid 10-model 5m portfolio.

The 60m screen produced 3 individual survivors. Its allocator swing passes at 0.106, but effective bets are only 2.611 and there are only 3 models, so there is no valid 10-model 60m portfolio.

## Required Framing

**The surviving candidates are in-sample survivors of a stricter gate, not yet out-of-sample validated.**

This framing is required even where the individual metrics look good, because the stricter gate was designed after seeing the prior roster fail on the same short forward window. A clean validation requires a future window that had no influence on the gate design, or a much longer reconstructed 5m history with a truly frozen procedure.

## Final Answer

No valid 10-model portfolio was found under the rebuilt screening pipeline. The pipeline now persists full evaluated grids and blocks the exact garbage-passing behavior that killed the prior roster, but the current data does not support a portfolio that passes both individual strict gates and portfolio-level allocator/effective-independence requirements.