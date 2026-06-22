# Medallion-Lite — GPU-native stacked ensemble for XAUUSD

Goal: a backtested, after-cost-profitable trading signal for gold, validated on an
honest holdout, that fully uses an 8×L4 / 96-vCPU / 384 GB VM. Deployment target:
**backtest PF / Sharpe / Kelly on an untouched holdout** (research artifact, no live
wiring required).

Guiding truth: model size does not create edge. Profit comes from (1) economically
meaningful labels, (2) heavy GPU feature-extractors that reach order-flow dynamics the
243-dim summary discards, (3) ensemble diversity, and (4) validation honest enough that
the reported edge is the live edge. Compute saturation = depth (big GPU sequence models)
× breadth (hundreds of purged walk-forward fits).

## Tier 0 — Labels & data (highest profit per effort)
- **Triple-barrier labeling** (vol-scaled TP/SL + time barrier; first-touch wins).
  Replaces `delta > threshold*spread`. New `include/data/labeler.hpp`.
- **Sample-uniqueness weights** (overlapping windows are not independent).
- **Purged + embargoed walk-forward CV** (López de Prado) — removes the leakage the
  current overlapping-window split has.
- **Meta-labeling** target stored now; meta-model trained in Tier 3.
- Cache **FTC3**: per sample carry summary, the [512×27] window ref, triple-barrier
  outcome (label, first-touch return, cost, sample_weight, time index for purge).

## Tier 1 — Base learners (GPU-native, BF16 tensor cores, full [512×27] window)
- **A. Temporal Transformer** — conv stem → 6–8 attention layers, d_model 384–512.
- **B. TCN** — dilated causal convolutions.
- **C. GRU/LSTM** — port the existing CPU recurrent layer to GPU.
- **D. MoE head** — 8–16 experts, top-2 routing, regime specialization.
On-the-fly window construction: keep ~100M ticks resident on GPU (~11 GB), gather
windows in a kernel (avoids the 440 GB materialized-window blowup).

## Tier 2 — Multi-GPU (saturate 8×L4)
- **NCCL data-parallel** for the flagship transformer (1 model, all 8 L4, effective
  batch ~32 K, mixed precision). New `include/cuda/nccl_allreduce.hpp`.
- **Fleet mode**: each L4 trains a different (architecture × WF fold × regime × seed);
  ~180 fits over 6 h.

## Tier 3 — Stacking
- Out-of-fold base predictions → meta-model (small NN / gradient boost) → final signal.
- Regime-gated routing over the 7 existing session/vol environments.

## Tier 4 — Sizing & honest validation
- Position size = meta-prob × fractional-Kelly, capped.
- One untouched final holdout (most-recent slice); report after-cost PnL / Sharpe /
  maxDD / PF / Kelly there ONCE.
- **Deflated Sharpe** accounting for the number of configurations tried (kills the
  select-best-of-N-on-val overfitting the audit flagged as the #1 risk).

## Model count
4 base archs × ~15 WF folds × 3 seeds ≈ **180 base fits** + 1 NCCL flagship + 1 stacker
(+ 7 optional regime specialists) → one deployable ensemble.

## Build phases (each independently testable)
- **P1** triple-barrier labels + sample weights + purged/embargoed walk-forward +
  honest holdout + deflated metrics. *(C++, no CUDA. Foundation; may reveal whether edge
  exists before heavy GPU work.)*
- **P2** GPU sequence model: transformer in cuBLAS, BF16, on-the-fly windows.
- **P3** NCCL 8-GPU data-parallel.
- **P4** stacking + meta-model + regime routing.
- **P5** flagship scale-up + final deflated-Sharpe holdout report; launcher rewrite to
  orchestrate fleet + flagship across 8×L4×6 h.

## Status
- Done: stability fixes (grad clip, GPU weight reupload, FTC2 real-PnL metric, purge
  gap), sweep/ensemble launcher. See git diff.
- Deep audit (7-lens adversarial) findings fold into P1/P2 as a hardening pass.
- Current: building P1.
