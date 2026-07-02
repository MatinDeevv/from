# Adversarial XAU Edge Audit Report

Generated from the live artifact run on 2026-06-18. This report is intentionally adversarial: a single failed kill-test is enough to reject the current edge claim as production-ready.

## Source Artifacts

- Working directory: `C:\Users\marti\from\from`
- Prompt file: `C:\Users\marti\Downloads\codex_destroy_xau_edge_prompt.md`
- Audit script: `C:\Users\marti\from\from\tools\adversarial_xau_edge_audit.py`
- Machine-readable output: `C:\Users\marti\from\reports\adversarial_xau_edge_audit_results.json`
- Main prior self-contained report: `C:\Users\marti\from\reports\final_xau_edge_research_report_SELF_CONTAINED.md`
- 5m selected model file: `C:\Users\marti\from\reports\selected_10_models_oldrank_forwardgate.json`
- 60m selected model file: `C:\Users\marti\from\reports\selected_10_models_60m_730d_oldrank_forwardgate.json`

## Command Log

```powershell
& 'C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile tools/adversarial_xau_edge_audit.py
# Exit code: 0
& 'C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools/adversarial_xau_edge_audit.py
# Exit code: 0
# Output: wrote C:\Users\marti\from\reports\adversarial_xau_edge_audit_results.json
```

## Executive Result

| Attack | Status | Threshold / Note |
| --- | --- | --- |
| multiple_comparisons | 5m: INCONCLUSIVE, 60m: INCONCLUSIVE | see section |
| leakage_timestamp_audit | FAIL | PASS requires no material leakage/alignment/availability caveats |
| reverse_rule | 5m: PASS, 60m: PASS | see section |
| block_bootstrap | 5m: FAIL, 60m: FAIL | see section |
| parameter_sensitivity | 5m: FAIL, 60m: FAIL | see section |
| independence | 5m: PASS, 60m: FAIL | see section |
| allocator_order | 5m: FAIL, 60m: PASS | see section |
| cost_latency | 5m: PASS, 60m: PASS | see section |
| subperiod | 5m: FAIL, 60m: FAIL | see section |
| leave_one_out | 5m: FAIL, 60m: FAIL | see section |

The edge did **not** survive adversarial validation. The strongest kill evidence is not one isolated problem: both 5m and 60m portfolio block-bootstrap 95% return intervals cross below zero; both timeframes fail subperiod consistency; both have fragile parameter sensitivity; both have leave-one-out failures; and the static leakage/availability audit contains unresolved bias risks.

## Attack 1 - Multiple Comparisons / Data Snooping

Status: **INCONCLUSIVE** for both timeframes. The saved artifacts do not contain the full evaluated hyperparameter grid, so a true White Reality Check / SPA-style correction cannot be reconstructed. The run used the saved promoted-candidate rows only, which is a weaker null and cannot validate the edge.

| TF | Status | Old-gated candidates | Positive fwd candidates | Actual capped return | Actual capped PnL | Promoted-null p50 raw PnL | Promoted-null p95 | Promoted-null p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | INCONCLUSIVE | 496 | 198 | 8.02% | $400,866.59 | $16,239.06 | $24,154.67 | $27,671.85 |
| 60m | INCONCLUSIVE | 1948 | 940 | 3.58% | $179,233.30 | $3,851.42 | $5,850.60 | $6,800.14 |

Interpretation: this does not kill the edge by itself, but it prevents any strong claim that the model selection process beat data snooping. The actual capped portfolio is not directly comparable to the promoted-candidate raw-PnL null.

## Attack 2 - Leakage / Timestamp / Availability Audit

Status: **FAIL**. PASS requires no material leakage/alignment/availability caveats

| Item | Verdict | Detail |
| --- | --- | --- |
| spread_t | INCONCLUSIVE/WEAK | Cost uses observed average spread of the same execution bar at signal entry. This is conservative as a cost subtraction but not a tradable known-at-signal quote. It is not alpha leakage, but it is not a broker-real quote model. |
| threshold | PASS for saved models | Thresholds in search scripts are fitted from the first half of old data only; selected JSON stores fixed thresholds used in forward reruns. |
| timestamp alignment | INCONCLUSIVE | 5m Yahoo leader and restored XAU are inner-joined on timestamps. A prior bug in 60m forward aggregation collapsed timestamps to 1970 and was fixed before final 60m reruns. No independent timezone audit against exchange calendars was done. |
| rolling windows | PASS | Leader momentum and vol use rolling values shifted by one bar before signal generation in the search functions. |
| universe availability | FAIL/BIAS RISK | The 50-symbol universe was manually selected for liquid available Yahoo symbols with usable data. This can bias toward instruments that happen to have clean data and macro relevance in the tested window. |

Interpretation: the alpha formula does not show obvious lookahead in rolling features or saved thresholds, but the universe/data-availability risk and incomplete independent timezone audit are enough to mark this failed for production evidence.

## Attack 3 - Reverse-Rule Sanity Check

5m: **PASS**. PASS requires reverse PnL <= -25% of forward PnL for every model. Weak reverse models: 0.
| Model | Forward PnL | Forward PF | Reverse PnL | Reverse PF | Reverse/abs(forward) |
| --- | --- | --- | --- | --- | --- |
| XLC_volz_follow_w36_q0.55_ny_h30_cost2.0 | $221,656.76 | 1.25 | $-406,755.57 | 0.66 | -1.84 |
| XLV_volz_follow_w48_q0.65_ny_h24_cost2.0 | $441,094.30 | 1.91 | $-646,939.76 | 0.39 | -1.47 |
| DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0 | $191,874.98 | 1.28 | $-419,628.55 | 0.58 | -2.19 |
| IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0 | $21,723.06 | 1.02 | $-235,045.27 | 0.79 | -10.82 |
| XLF_volz_fade_w48_q0.55_london_h30_cost2.0 | $441,363.58 | 1.99 | $-591,799.07 | 0.39 | -1.34 |
| UNG_volz_follow_w48_q0.75_all_h18_cost2.0 | $248,177.71 | 1.43 | $-472,308.38 | 0.50 | -1.90 |
| XLK_volz_fade_w48_q0.75_all_h30_cost2.0 | $5,002.14 | 1.01 | $-181,720.28 | 0.72 | -36.33 |
| VXX_volz_follow_w48_q0.75_all_h30_cost2.0 | $215,911.10 | 1.26 | $-419,142.76 | 0.64 | -1.94 |
| EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0 | $123,682.25 | 1.23 | $-322,846.97 | 0.59 | -2.61 |
| SPY_volz_fade_w36_q0.75_all_h24_cost2.0 | $202,582.47 | 1.38 | $-398,499.66 | 0.53 | -1.97 |

60m: **PASS**. PASS requires reverse PnL <= -25% of forward PnL for every model. Weak reverse models: 0.
| Model | Forward PnL | Forward PF | Reverse PnL | Reverse PF | Reverse/abs(forward) |
| --- | --- | --- | --- | --- | --- |
| EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0 | $116,808.05 | 2.04 | $-127,020.69 | 0.46 | -1.09 |
| GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $80,044.40 | 2.12 | $-87,123.69 | 0.44 | -1.09 |
| GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0 | $66,328.62 | 1.99 | $-72,374.08 | 0.47 | -1.09 |
| GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0 | $18,580.44 | 1.08 | $-55,924.26 | 0.79 | -3.01 |
| SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0 | $55,422.82 | 1.75 | $-64,908.95 | 0.52 | -1.17 |
| FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0 | $32,216.06 | 1.25 | $-43,534.36 | 0.74 | -1.35 |
| EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0 | $41,385.54 | 1.63 | $-47,989.84 | 0.56 | -1.16 |
| QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0 | $171,107.56 | 2.95 | $-180,861.39 | 0.32 | -1.06 |
| ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0 | $79,024.13 | 5.28 | $-84,703.38 | 0.17 | -1.07 |
| EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $143,552.77 | 213.71 | $-148,846.22 | 0.00 | -1.04 |

Interpretation: this attack passed. Reversing the rules generally loses money, so the saved direction is not random sign-flipping noise.

## Attack 4 - Block Bootstrap Robustness

| TF | Status | Trials | Block | Return 95% CI | PF 95% CI | Max DD 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 5m | FAIL | 5000 | 30 | [-1.36%, 18.05%] | [0.95, 2.09] | [1.25%, 6.60%] |
| 60m | FAIL | 5000 | 8 | [-0.92%, 8.60%] | [0.91, 2.25] | [0.80%, 3.07%] |

5m individual models with return CI lower bound <= 0: 10/10
60m individual models with return CI lower bound <= 0: 8/10

Interpretation: this is a direct kill-test failure. PASS required the portfolio lower bound to stay above zero; both 5m and 60m lower bounds are negative.

## Attack 5 - Parameter Sensitivity / Cliff Test

5m: **FAIL**. Fragile models: 6/10. PASS requires all +/- perturbation variants to remain positive and above 50% of base PnL
| Model | Base PnL | Base PF | Cliff variants | Worst variant PnL |
| --- | --- | --- | --- | --- |
| XLC_volz_follow_w36_q0.55_ny_h30_cost2.0 | $221,656.76 | 1.25 | 1 | $-35,803.58 |
| XLV_volz_follow_w48_q0.65_ny_h24_cost2.0 | $441,094.30 | 1.91 | 0 | $286,302.49 |
| DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0 | $191,874.98 | 1.28 | 1 | $-90,858.62 |
| IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0 | $21,723.06 | 1.02 | 3 | $-103,335.62 |
| XLF_volz_fade_w48_q0.55_london_h30_cost2.0 | $441,363.58 | 1.99 | 0 | $304,006.13 |
| UNG_volz_follow_w48_q0.75_all_h18_cost2.0 | $248,177.71 | 1.43 | 0 | $205,488.97 |
| XLK_volz_fade_w48_q0.75_all_h30_cost2.0 | $5,002.14 | 1.01 | 1 | $-28,771.56 |
| VXX_volz_follow_w48_q0.75_all_h30_cost2.0 | $215,911.10 | 1.26 | 1 | $36,129.61 |
| EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0 | $123,682.25 | 1.23 | 1 | $52,029.94 |
| SPY_volz_fade_w36_q0.75_all_h24_cost2.0 | $202,582.47 | 1.38 | 0 | $106,575.99 |

60m: **FAIL**. Fragile models: 6/10. PASS requires all +/- perturbation variants to remain positive and above 50% of base PnL
| Model | Base PnL | Base PF | Cliff variants | Worst variant PnL |
| --- | --- | --- | --- | --- |
| EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0 | $116,808.05 | 2.04 | 0 | $78,062.60 |
| GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $80,044.40 | 2.12 | 1 | $22,261.22 |
| GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0 | $66,328.62 | 1.99 | 1 | $6,029.24 |
| GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0 | $18,580.44 | 1.08 | 3 | $-36,734.09 |
| SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0 | $55,422.82 | 1.75 | 1 | $10,990.45 |
| FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0 | $32,216.06 | 1.25 | 3 | $-14,169.80 |
| EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0 | $41,385.54 | 1.63 | 1 | $8,657.83 |
| QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0 | $171,107.56 | 2.95 | 0 | $88,818.10 |
| ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0 | $79,024.13 | 5.28 | 0 | $61,508.12 |
| EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $143,552.77 | 213.71 | 0 | $90,661.59 |

Note: threshold/quantile perturbation is approximate because the saved artifacts do not preserve the original training feature vectors. Window and horizon perturbations were rerun directly.

## Attack 6 - Effective Independence / Correlation

| TF | Status | Participation-ratio effective bets | Eigenvalues > 1 | Eigenvalues |
| --- | --- | --- | --- | --- |
| 5m | PASS | 5.42 | 5 | 0.05, 0.09, 0.23, 0.43, 0.54, 1.04, 1.21, 1.32, 1.96, 3.12 |
| 60m | FAIL | 2.81 | 3 | 0.00, 0.00, 0.03, 0.08, 0.14, 0.41, 0.56, 1.28, 2.13, 5.37 |

Interpretation: 5m barely passes the >=5 effective-bets threshold. 60m fails: the ten models act more like roughly 2.8 independent bets, which weakens the ensemble diversification story.

## Attack 7 - Capped Allocator Order Dependence

5m: **FAIL**. PnL swing ratio: 0.60. PASS requires allocator policy PnL swing <= 30% of existing PnL
| Policy | PnL | Return | PF | Max DD |
| --- | --- | --- | --- | --- |
| existing | $348,581.01 | 6.97% | 1.34 | $291,860.20 |
| reverse_model | $558,890.88 | 11.18% | 1.61 | $171,005.46 |
| edge_rank | $477,508.29 | 9.55% | 1.49 | $173,237.99 |

60m: **PASS**. PnL swing ratio: 0.19. PASS requires allocator policy PnL swing <= 30% of existing PnL
| Policy | PnL | Return | PF | Max DD |
| --- | --- | --- | --- | --- |
| existing | $282,810.25 | 5.66% | 1.78 | $75,004.71 |
| reverse_model | $281,088.31 | 5.62% | 1.78 | $64,917.99 |
| edge_rank | $334,247.72 | 6.68% | 2.02 | $66,231.25 |

Interpretation: 5m fails because the capped portfolio result changes materially based on which simultaneous signals get filled first. That means a meaningful part of the reported result is allocator-policy dependent, not just model edge.

## Attack 8 - Cost and Latency Stress

5m: **PASS**. PASS requires positive uncapped PnL at 5x cost and 1-bar latency
| Stress | PnL | Return | PF | Trades |
| --- | --- | --- | --- | --- |
| cost_2.0x_uncapped | $2,113,068.37 | 42.26% | 1.32 | 7064 |
| cost_3.0x_uncapped | $1,617,663.89 | 32.35% | 1.24 | 7064 |
| cost_5.0x_uncapped | $626,854.94 | 12.54% | 1.09 | 7064 |
| latency_1_bar_uncapped | $2,091,715.73 | 41.83% | 1.32 | 7062 |
| latency_2_bar_uncapped | $2,121,185.16 | 42.42% | 1.33 | 7060 |

60m: **PASS**. PASS requires positive uncapped PnL at 5x cost and 1-bar latency
| Stress | PnL | Return | PF | Trades |
| --- | --- | --- | --- | --- |
| cost_2.0x_uncapped | $804,470.40 | 16.09% | 1.95 | 373 |
| cost_3.0x_uncapped | $777,266.29 | 15.55% | 1.90 | 373 |
| cost_5.0x_uncapped | $722,858.07 | 14.46% | 1.82 | 373 |
| latency_1_bar_uncapped | $1,077,358.50 | 21.55% | 2.41 | 373 |
| latency_2_bar_uncapped | $1,236,714.65 | 24.73% | 2.73 | 373 |

Interpretation: this attack passed on the uncapped signal stack. It does not rescue the edge because other capped and robustness tests fail, and uncapped exposure is not an executable $5M portfolio by itself.

## Attack 9 - Subperiod Consistency

5m: **FAIL**. Bad individual models: 10/10. PASS requires all four portfolio quarters positive and no individual model with negative quarter or >75% PnL from one quarter
| Q1 PnL | Q2 PnL | Q3 PnL | Q4 PnL |
| --- | --- | --- | --- |
| $1,076,546.03 | $987,916.22 | $-533,331.21 | $581,937.33 |
| Model | Q1 | Q2 | Q3 | Q4 | Max quarter share |
| --- | --- | --- | --- | --- | --- |
| XLC_volz_follow_w36_q0.55_ny_h30_cost2.0 | $24,772.93 | $129,619.99 | $-77,118.65 | $144,382.49 | 0.65 |
| XLV_volz_follow_w48_q0.65_ny_h24_cost2.0 | $144,038.08 | $-47,983.31 | $-52,118.17 | $397,157.70 | 0.90 |
| DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0 | $138,088.17 | $65,169.88 | $237,610.30 | $-248,993.37 | 1.24 |
| IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0 | $111,990.05 | $-13,926.61 | $-322,550.35 | $246,209.97 | 11.33 |
| XLF_volz_fade_w48_q0.55_london_h30_cost2.0 | $201,992.11 | $-55,838.49 | $350,066.87 | $-54,856.92 | 0.79 |
| UNG_volz_follow_w48_q0.75_all_h18_cost2.0 | $166,007.12 | $150,502.97 | $106,635.25 | $-174,967.62 | 0.67 |
| XLK_volz_fade_w48_q0.75_all_h30_cost2.0 | $127,848.27 | $19,881.27 | $-250,166.63 | $107,439.23 | 25.56 |
| VXX_volz_follow_w48_q0.75_all_h30_cost2.0 | $271,296.02 | $-6,533.99 | $162,675.50 | $-211,526.44 | 1.26 |
| EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0 | $-3,178.84 | $185,352.22 | $-86,201.26 | $27,710.13 | 1.50 |
| SPY_volz_fade_w36_q0.75_all_h24_cost2.0 | $152,636.15 | $119,803.16 | $-215,643.65 | $145,786.80 | 0.75 |

60m: **FAIL**. Bad individual models: 8/10. PASS requires all four portfolio quarters positive and no individual model with negative quarter or >75% PnL from one quarter
| Q1 PnL | Q2 PnL | Q3 PnL | Q4 PnL |
| --- | --- | --- | --- |
| $54,406.48 | $271,300.29 | $-183,918.53 | $662,682.17 |
| Model | Q1 | Q2 | Q3 | Q4 | Max quarter share |
| --- | --- | --- | --- | --- | --- |
| EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0 | $41,223.77 | $51,474.29 | $-82,651.47 | $106,761.46 | 0.91 |
| GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $26,865.83 | $-12,235.89 | $18,410.19 | $47,004.28 | 0.59 |
| GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0 | $31,252.62 | $2,888.92 | $-47,425.91 | $79,612.99 | 1.20 |
| GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0 | $-36,501.80 | $-27,450.24 | $103,640.39 | $-21,107.90 | 5.58 |
| SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0 | $13,701.89 | $31,782.02 | $-54,287.74 | $64,226.65 | 1.16 |
| FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0 | $16,303.50 | $-65,239.86 | $78,230.94 | $2,921.48 | 2.43 |
| EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0 | $-13,907.12 | $-43,995.36 | $30,131.87 | $69,156.15 | 1.67 |
| QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0 | $8,885.74 | $91,461.62 | $-18,556.99 | $89,317.20 | 0.53 |
| ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0 | $37,041.79 | $22,081.17 | $10,107.67 | $9,793.50 | 0.47 |
| EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $10,002.76 | $16,625.07 | $25,199.84 | $91,725.11 | 0.64 |

Interpretation: both portfolios have a negative third quarter of the forward trade sequence. This is a serious stability failure, especially because all 5m individual models and 8/10 60m models fail the subperiod rule.

## Attack 10 - Leave-One-Out Concentration

5m: **FAIL**. Base capped-policy PnL in this attack: $348,581.01, return 6.97%, max DD $291,860.20. Bad removals: 2/10. PASS requires no leave-one-out portfolio to flip negative and no max-DD change >30%
| Removed model | PnL | Return | Max DD | PnL change vs base | DD change vs base |
| --- | --- | --- | --- | --- | --- |
| DX-Y.NYB_volz_fade_w36_q0.85_all_h30_cost2.0 | $325,632.98 | 6.51% | $234,085.34 | -6.58% | -19.80% |
| EURUSD=X_divergence_follow_w36_q0.65_london_h24_cost2.0 | $392,233.68 | 7.84% | $223,725.16 | 12.52% | -23.35% |
| IEF_divergence_follow_w24_q0.55_ny_h30_cost2.0 | $301,389.88 | 6.03% | $330,194.37 | -13.54% | 13.13% |
| SPY_volz_fade_w36_q0.75_all_h24_cost2.0 | $272,309.88 | 5.45% | $297,832.69 | -21.88% | 2.05% |
| UNG_volz_follow_w48_q0.75_all_h18_cost2.0 | $276,658.98 | 5.53% | $204,813.62 | -20.63% | -29.82% |
| VXX_volz_follow_w48_q0.75_all_h30_cost2.0 | $366,254.61 | 7.33% | $289,738.61 | 5.07% | -0.73% |
| XLC_volz_follow_w36_q0.55_ny_h30_cost2.0 | $441,339.33 | 8.83% | $212,506.25 | 26.61% | -27.19% |
| XLF_volz_fade_w48_q0.55_london_h30_cost2.0 | $298,576.13 | 5.97% | $381,510.34 | -14.35% | 30.72% |
| XLK_volz_fade_w48_q0.75_all_h30_cost2.0 | $384,434.82 | 7.69% | $299,646.98 | 10.29% | 2.67% |
| XLV_volz_follow_w48_q0.65_ny_h24_cost2.0 | $438,265.04 | 8.77% | $203,672.84 | 25.73% | -30.22% |

60m: **FAIL**. Base capped-policy PnL in this attack: $282,810.25, return 5.66%, max DD $75,004.71. Bad removals: 2/10. PASS requires no leave-one-out portfolio to flip negative and no max-DD change >30%
| Removed model | PnL | Return | Max DD | PnL change vs base | DD change vs base |
| --- | --- | --- | --- | --- | --- |
| EEM_60m_divergence_follow_w12_q0.65_all_h6_cost2.0 | $193,177.42 | 3.86% | $94,990.13 | -31.69% | 26.65% |
| EFA_60m_volz_follow_w6_q0.75_all_h8_cost2.0 | $204,320.40 | 4.09% | $115,078.57 | -27.75% | 53.43% |
| EWJ_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $269,068.78 | 5.38% | $75,004.71 | -4.86% | 0.00% |
| FXI_60m_divergence_follow_w6_q0.55_all_h8_cost2.0 | $282,107.13 | 5.64% | $62,883.03 | -0.25% | -16.16% |
| GC=F_60m_volz_follow_w6_q0.65_all_h8_cost2.0 | $310,384.58 | 6.21% | $97,435.15 | 9.75% | 29.91% |
| GDXJ_60m_divergence_follow_w12_q0.85_all_h8_cost2.0 | $282,810.25 | 5.66% | $75,004.71 | 0.00% | 0.00% |
| GDX_60m_divergence_follow_w12_q0.75_all_h8_cost2.0 | $280,093.02 | 5.60% | $90,356.48 | -0.96% | 20.47% |
| QQQ_60m_divergence_follow_w12_q0.65_all_h8_cost2.0 | $264,920.70 | 5.30% | $73,995.21 | -6.33% | -1.35% |
| SPY_60m_divergence_follow_w12_q0.55_all_h6_cost2.0 | $315,154.80 | 6.30% | $75,004.71 | 11.44% | 0.00% |
| ZB=F_60m_divergence_fade_w12_q0.85_all_h8_cost2.0 | $251,058.71 | 5.02% | $101,658.55 | -11.23% | 35.54% |

Interpretation: both timeframes fail because removing some models changes drawdown by more than the allowed 30%. This says the ensemble is not stable enough to treat as ten equally robust independent edges.

## Methodology Notes

- Reverse-rule, parameter, cost/latency, independence, subperiod, and leave-one-out tests reconstruct trades from the saved model JSON and the local forward price data.
- Block bootstrap uses saved capped portfolio trade CSVs and resamples contiguous PnL blocks: 30 trades for 5m and 8 trades for 60m.
- Cost/latency stress is uncapped. This is useful as a signal test but not sufficient for live portfolio validation because the prior uncapped runs exceeded 100% gross exposure.
- The allocator-order test uses a reconstructed cap engine and policy permutations. Its base PnL is close but not identical to the original published capped rerun because it is built from reconstructed model-level trade streams.
- Multiple-comparisons testing is incomplete because the full rejected grid was not preserved. The correct fix is to rerun the full search while saving every candidate and then apply a block/permutation reality-check across all tried rules.

## Required Follow-Up Before Any Live Use

1. Rerun the full grid and persist every tried rule, not only promoted rows.
2. Lock a broker-realistic execution model: tradable bid/ask at signal time, latency, size limits, and session constraints.
3. Rebuild the universe from a timestamped availability list that existed before the test window.
4. Require positive lower-bound bootstrap evidence after all selection and allocator choices are frozen.
5. Require subperiod stability after model selection, not just aggregate positive PnL.

## Verdict

**Killed as a production-ready edge claim.**

The current evidence still contains something interesting: reverse rules lose, cost/latency stress remains positive uncapped, and the raw forward portfolios made money. But the prompt asked whether the edge survives adversarial validation. It does not. The decisive failures are the negative lower bounds in the block-bootstrap portfolio CIs, broad parameter fragility, negative subperiods, allocator dependence in 5m, 60m dependence concentration, and unresolved universe/timestamp/data-availability bias risks.

The honest status is: **research lead worth re-testing from scratch, not a validated deployable trading edge.**

## Full Machine-Readable Output

The complete JSON output is saved at `C:\Users\marti\from\reports\adversarial_xau_edge_audit_results.json`.