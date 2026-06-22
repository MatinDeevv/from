# EDGE_UPGRADE — max-backtest-edge attack log

Goal: squeeze real, after-cost edge on an honest holdout. NOT "add 200 indicators."
Edge comes from honest labels + leak-free validation + real cost/sizing, not feature count.

Status date: 2026-06-21. Branch: detached HEAD off `main`.

---

## What was already in the repo (baseline — do not redo)
- 27 microstructure features x 9 scales -> 243-dim summary.
- Triple-barrier labels in `include/data/windower.hpp` (vol-scaled TP/SL + cost floor, no lookahead).
- Purged + embargoed walk-forward with N_eff overlap correction (`cli/walkforward.cpp`, `cli/wf_metrics.hpp`).
- Cost-aware trade stats: PF, Kelly, Sharpe, CI, maxDD (`wf_metrics.hpp::compute_stats`).
- Temperature calibrator (`include/analysis/calibrator.hpp`, was unused at inference).
- Active model = `SequenceModel` (summary MLP, ~97K params), saved FSQ3 via `SequenceModelIO`.

---

## Stage 1 — sample-uniqueness + time-decay weights  [DONE, builds]
**Why:** overlapping triple-barrier windows are not independent -> overcounting biases the fit / inflates confidence.

- NEW `include/data/sample_weights.hpp`:
  - `compute_uniqueness(entries, horizon)` — avg(1/concurrency) over each label span, O(n log n) sweep.
  - `compute_time_decay(entries, tail)` — linear recency weighting.
  - `compute_sample_weights(entries, horizon, tail)` — combined, normalized to mean 1.
- WIRED into `cli/walkforward.cpp`:
  - new flags `--no-uniqueness` (default ON), `--weight-tail` (default 1.0 = uniqueness only).
  - real weights computed from `all_t` after load (overwrites the 1.0 placeholder in `all_w`).
  - `train_on_indices` importance-resamples each fold's train slice ∝ uniqueness (deterministic in seed).
    Works for BOTH the GPU kernel (uniform internal sampling) and the CPU path — zero kernel change.

**Not done:** train.cpp still uses FTC2 cache (no `t_index`) + `y_dir` labels -> wiring weights there needs a cache-format bump. Skipped (walkforward is the honest harness).

---

## Stage 2 — backtest truth rewrite  [DONE, builds]
**Why:** old `cli/backtest.cpp` loaded the WRONG model (giant `FromModel`, not the trained `SequenceModel`) and used naive delta labels, flat spread cost, bogus sqrt(252) Sharpe, no calibration, no sizing. It was orphaned.

Full rewrite of `cli/backtest.cpp`:
- Loads the real `SequenceModel` artifact (FSQ3 carries 2nd-pass feat_norm).
- First-pass Welford: loads `norm1.bin` (auto-detect sibling / `<model>.norm1`) else fits on warmup prefix + freezes. (infer.cpp uses a fresh normalizer — latent skew; backtest does it right.)
- Trades the SAME triple-barrier labels as walk-forward (`tb_ret`, `tb_cost`, real per-trade spread + slippage + commission, in return units).
- Temperature calibration: fit T on first `--calib-frac` (default 0.2), apply to the rest, report ECE before/after.
- Position sizing: confidence-proportional, capped at `--kelly-frac` x Kelly stake estimated from the calib slice (no lookahead). Cost scales with size.
- Metrics via `wf_metrics::compute_stats` (unit + sized rows) + per-regime (env_id) breakdown + PSR/DSR.
- Reports tb-accuracy vs naive-y-accuracy gap.
- New flags: `--conf-gate --calib-frac --kelly-frac --size-cap --commission --slippage-mult --trials --no-calib --warmup-frac --norm1 --barrier-k --cost-mult --window/--stride/--horizon`.

---

## Stage 3 — meta-labeling false-signal filter  [IN PROGRESS]
**Why:** a 2nd model deciding WHETHER to act on each primary signal filters false signals a raw confidence gate lets through. Biggest precision/Sharpe lever.

- NEW `include/model/meta_labeler.hpp` [DONE]:
  - `MetaLabeler` — L2 logistic regression, internal standardization, full-batch GD.
  - `meta_features(probs, env_id, out)` -> 12-dim row [p_up,p_neutral,p_down,conf,dir_edge, 7x regime onehot].
- WIRED into `cli/backtest.cpp` [PARTIAL]:
  - DONE: `--meta`, `--meta-gate` flags; gather gated calib signals + target(net>0); fit MetaLabeler.
  - **TODO: apply meta in the TEST loop** — compute meta_p per gated test signal, gate by `--meta-gate`,
    collect `meta_nets` (+ meta-prob-sized variant), print a "meta" stats row. (Was mid-edit when paused.)
  - **TODO: rebuild + verify compiles after the test-loop wiring.**

---

## Stage 4 — deflated Sharpe + robustness  [helper DONE, integration partial]
- DONE in `cli/wf_metrics.hpp`:
  - `normal_cdf`, `normal_ppf` (Acklam probit).
  - `SharpeRobustness` + `sharpe_robustness(nets, n_eff, n_trials)` -> PSR (P(SR>0)) and DSR (P(SR>expected-max-under-n-trials)), skew/kurt-aware.
- DONE: backtest prints PSR/DSR.
- **TODO: surface DSR in `cli/walkforward.cpp` pooled-OOF + holdout summary** with `n_trials = folds x hyperparam-grid` so the reported edge is multiple-testing honest.

---

## Build / verify
- CPU build (clean as of Stage 2): `cmake -S . -B build-check -DFROM_CUDA=OFF && cmake --build build-check --target from --config Release -j 4`
- After Stage 3 test-loop wiring: rebuild + run `from backtest --model weights_best.from --data <ticks>.parquet --meta` and sanity-check the unit/sized/meta rows.

---

## Backlog (not started — future edge)
- Stacking ensemble: combine base learners' OOF preds -> meta-model (Tier 3).
- Train.cpp: bump cache to FTC3 (carry t_index) + train on tb_label, apply sample weights in the loss.
- Fix infer.cpp first-pass normalization skew (load norm1 like backtest now does).
- Vol-targeted sizing using an in-window (non-lookahead) vol estimate carried on the Sample.
- Combinatorial purged CV (CPCV) + PBO for a real overfit probability.
