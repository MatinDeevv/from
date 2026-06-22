# XAUUSD model-fleet executive summary

- Completed dense registry: 24 models.
- Decisions: 0 GO, 0 WATCH, 24 NO.
- Best completed model: `mlp_h1024_cm1.00_bk1.00_cg0.60_s42` (NO), holdout
  PF=0.21, t=-1.29, Kelly=-1.012, N_eff=5.7.
- Dense null distribution: N=16, PF p95=1.09 / p99=1.27 and t p95=0.13 /
  p99=0.40.

The null set is underpowered for certification, but it does not change the
observed result: all completed models were losers under an optimistic cost
model. Do not promote any completed configuration; the valid follow-up is the
long-horizon-first rerun documented in `FINAL_VERDICT.md`.
