# `from` / Medallion-Lite — Technical Documentation

**A GPU-native, microstructure-driven direction-prediction engine for XAUUSD (spot gold), with an emphasis on statistically honest validation.**

> Scope of this document. This is a research-engineering reference, written at the level
> of a graduate course in quantitative finance + a systems-programming text. It documents
> what the codebase *actually does* (verified against source), the financial-econometric
> theory each component implements, the known failure modes, and the validation philosophy
> that separates a backtest artifact from a tradeable edge. It does **not** claim a live,
> deployed, money-making strategy — it claims a disciplined apparatus for discovering
> whether such an edge exists, and for refusing to fool itself when it does not.

---

## Table of contents

1. [Thesis and design philosophy](#1-thesis-and-design-philosophy)
2. [System architecture](#2-system-architecture)
3. [The data substrate: ticks to features](#3-the-data-substrate-ticks-to-features)
4. [Microstructure feature theory (the 27)](#4-microstructure-feature-theory-the-27)
5. [Multi-scale summarization (the 243-dim representation)](#5-multi-scale-summarization-the-243-dim-representation)
6. [Labeling: from threshold heuristics to triple-barrier](#6-labeling-from-threshold-heuristics-to-triple-barrier)
7. [Sample dependence and uniqueness weighting](#7-sample-dependence-and-uniqueness-weighting)
8. [The learning machinery](#8-the-learning-machinery)
9. [GPU systems engineering](#9-gpu-systems-engineering)
10. [Validation: purged walk-forward, deflation, and the holdout discipline](#10-validation-purged-walk-forward-deflation-and-the-holdout-discipline)
11. [Position sizing and the economics of the signal](#11-position-sizing-and-the-economics-of-the-signal)
12. [The Medallion-Lite roadmap (tiers P1–P5)](#12-the-medallion-lite-roadmap-tiers-p1p5)
13. [Audit log: bugs found, weaknesses, and the epistemics of self-distrust](#13-audit-log-bugs-found-weaknesses-and-the-epistemics-of-self-distrust)
14. [Reproducibility, build, and CLI](#14-reproducibility-build-and-cli)
15. [Glossary](#15-glossary)
16. [References](#16-references)

---

## 1. Thesis and design philosophy

The governing belief of this project is contrarian to most retail ML-trading efforts:

> **Model size does not create edge.** A 97K-parameter MLP and a 100M-parameter
> transformer trained on the same leaky labels and validated on the same overlapping
> split will both report a fictional Sharpe. Profit, if it exists at all, comes from
> four places, in descending order of return-on-effort:
>
> 1. **Economically meaningful labels** — what you ask the model to predict matters more
>    than the model.
> 2. **Features that reach genuine order-flow dynamics** rather than restating price.
> 3. **Ensemble diversity** — uncorrelated errors, not a bigger single learner.
> 4. **Validation honest enough that the reported edge equals the live edge.**

This is the López de Prado school of quantitative finance translated into a from-scratch
C++/CUDA codebase. The single most important sentence in the entire repository is the
project's own stated guiding truth:

> *"Compute saturation = depth (big GPU sequence models) × breadth (hundreds of purged
> walk-forward fits)."*

Depth without breadth overfits one split. Breadth without honest deflation overfits the
*search* — you try 200 configurations and report the luckiest. The architecture below is
designed so that **both** axes scale while the validation machinery actively works to
discount the luck out of the result.

### 1.1 Why XAUUSD, why ticks

Spot gold is a deep, nearly-24/5 market with rich microstructure and no single dominant
exchange order book — quotes are dealer-driven, spreads are state-dependent, and order-flow
imbalance is informative at short horizons. The engine consumes **tick-level** data
(approximately 539M rows in the reference dataset) rather than bars, because the signal
this project hunts — short-horizon directional drift conditioned on order-flow state —
lives at the resolution that bar aggregation destroys. A 1-minute OHLC bar throws away the
*sequence* of arrivals that carries the information.

---

## 2. System architecture

The end-to-end pipeline, as implemented:

```
Raw XAUUSD ticks (~539M rows, parquet)
   │
   ▼  ParquetReader  ────────────────  chunked, multi-threaded columnar read
   │
   ▼  TickProcessor  ────────────────  27 microstructure features per tick
   │                                    + Welford online normalization (frozen @100K)
   │
   ▼  Windower       ────────────────  512-tick windows, stride 64, horizon 128–256
   │                                    + triple-barrier labels {UP, NEUTRAL, DOWN}
   │
   ▼  MultiScaleSummarizer  ─────────  9 statistical scales × 27 features = 243-dim
   │                                    + second-pass z-score (persisted in model)
   │
   ▼  Sample-uniqueness weights  ────  López de Prado concurrency weighting
   │
   ▼  GpuTrainer (cuBLAS)  ──────────  MLP 243 → 256 → 128 → 3, Adam, class-weighted CE
   │                                    ~2M samples/sec on a single RTX 3050 4GB
   │
   ▼  Walk-forward CV  ──────────────  anchored, PURGED + EMBARGOED folds
   │                                    + deflated Sharpe over #configs tried
   │
   ▼  Untouched holdout  ────────────  reported ONCE: PF / Sharpe / Kelly / maxDD
```

The two trainers correspond to the two scales the project bets on:

- **`GpuTrainer`** — the production 3-layer MLP over the 243-dim *summary* vector.
  ~97K parameters, microsecond inference. This is the "breadth" workhorse: cheap enough
  to fit hundreds of times across folds, seeds, and regimes.
- **`DeepMlpTrainer`** (`include/cuda/deep_mlp_trainer.hpp`) — an N-layer MLP that
  consumes the **raw** `[window × 27]` tensor (e.g. `256 × 27 = 6912` inputs) rather than
  the summary, with hidden stack `{2048, 1024, 512}`. This is the "depth" path: it reaches
  dynamics the 243-dim summary necessarily discards. Windows are *gathered on the fly* from
  a GPU-resident per-tick feature stream, avoiding the ~440 GB blowup of materializing
  every window.

Everything is a single self-contained C++17/CUDA binary, `from`, dispatching subcommands
(`train`, `train-fast`, `walkforward`, `wfdeep`, `validate-adversarial`, `infer`,
`inspect`, `bench`, `test`).

---

## 3. The data substrate: ticks to features

### 3.1 Columnar ingest

`ParquetReader` reads the tick file in large chunks (default `chunk_size = 4,000,000`
rows) across a worker pool (default 16 threads). Tick fields are the dealer quote stream:
ask, bid, ask-volume, bid-volume, timestamp. Throughput is the binding constraint for the
first epoch; subsequent runs hit a binary cache.

### 3.2 Online normalization (Welford)

`TickProcessor` computes each of the 27 features and feeds them through a **Welford
online mean/variance estimator**. Welford's algorithm maintains a numerically stable
running mean and `M2` (sum of squared deviations) in a single pass:

```
count += 1
delta  = x - mean
mean  += delta / count
M2    += delta * (x - mean)
var    = M2 / count
```

The normalizer is **frozen after the first 100K samples**, so the statistics are fixed
before they can absorb information from the validation period — a small but real
leakage-prevention measure. The frozen Welford stats are persisted in the model file so
inference applies identical scaling.

### 3.3 The cache

The first run pays the parquet → features → windows cost (minutes). The result is
serialized to a binary cache; subsequent runs load instantly. The cache stores a
dimension signature and **auto-invalidates on mismatch** — change the feature count or
scale count and it regenerates with a clear log line rather than silently feeding the
trainer wrong-shaped data. (This invalidation discipline exists because a stale cache is
one of the most insidious sources of phantom backtest results.)

---

## 4. Microstructure feature theory (the 27)

Each tick is mapped to a 27-dimensional vector. The features are deliberately chosen from
the market-microstructure literature so that each one has an economic interpretation —
not "indicator soup," but estimators of latent quantities (information asymmetry, price
impact, liquidity, regime).

| # | Feature | What it estimates |
|---|---------|-------------------|
| 1–6 | Ask, Bid, Mid, AskVol, BidVol, Spread | Raw quote state |
| 7 | Normalized spread | Spread / mid — scale-free transaction cost |
| 8 | **OFI** (Order Flow Imbalance) | Net pressure from quote/size revisions |
| 9 | ΔOFI | Acceleration of order-flow pressure |
| 10–11 | Mid velocity, Mid acceleration | First/second derivative of mid |
| 12 | Microprice deviation | (Size-weighted fair price) − mid; lead indicator |
| 13–14 | Tick rate, log tick rate | Arrival intensity (a volatility proxy) |
| 15–17 | Realized vol @ 16 / 64 / 256 ticks | Multi-horizon volatility |
| 18 | Roll spread | Effective spread from serial covariance of price changes |
| 19 | Lee–Ready sum (32-tick) | Net signed trade direction via the tick test |
| 20 | Amihud illiquidity | \|return\| per unit volume — price-impact proxy |
| 21 | Bid–ask bounce | Mean-reverting microstructure noise component |
| 22 | **Kyle–Hasbrouck λ** (64-tick) | Price impact per unit signed volume (information asymmetry) |
| 23 | Trade imbalance ratio (32-tick) | Asymmetry of buyer vs seller aggression |
| 24 | Volatility regime indicator | RV16 / RV256 — trending vs ranging |
| 25 | Spread compression signal | z-score of norm spread vs 64-tick baseline |
| 26 | Autocorrelation lag-1 (32-tick) | Mean-reversion vs momentum sign |
| 27 | Volume-clock deviation | Tick rate vs EMA baseline |

### 4.1 The estimators that matter

**OFI (Order Flow Imbalance).** Following Cont, Kukanov, and Stoikov (2014), OFI
aggregates the signed changes in bid/ask sizes and prices. It is among the strongest
known short-horizon predictors of mid-price moves because it measures *demand for
immediacy* directly from the book, rather than inferring it from past returns.

**Kyle's λ / Hasbrouck.** Kyle's (1985) lambda is the slope of price against signed
order flow — the market's price-impact coefficient and a direct measure of adverse
selection / information asymmetry. A high λ means informed traders are moving the price.
Estimated here as a 64-tick rolling regression of price change on signed volume.

> **Honesty note (see §13):** in an earlier audited state this feature was hardcoded to
> `0.0f` ("simplified — expensive, skip for speed"), which meant the model wasted capacity
> learning that dimension 21 was constant. The audit flagged it; the roadmap restores the
> real rolling estimator. This is exactly the kind of silent dead-feature that a casual
> ML pipeline never catches.

**Amihud illiquidity** (Amihud 2002): `|return| / volume`, the canonical low-frequency
price-impact measure, here computed at tick scale. **Roll's (1984) implied spread**
recovers the effective spread from the negative serial covariance of price changes — a
classic identification trick. **Lee–Ready** (1991) classifies each trade as buyer- or
seller-initiated via the tick/quote test, summed over 32 ticks to give net aggression.

The point of this feature set is that it spans *four* distinct economic channels —
**information asymmetry** (λ, OFI), **liquidity/cost** (Amihud, Roll, spread compression),
**momentum/mean-reversion** (autocorr, velocity), and **regime** (vol ratio, volume
clock). A model fed only price-derived features cannot separate these; this one can.

---

## 5. Multi-scale summarization (the 243-dim representation)

A 512-tick window of 27 features is a `512 × 27` matrix. The summary path reduces it to a
fixed 243-dim vector by computing **9 statistical scales** for each of the 27 features
(9 × 27 = 243):

| Scale | Statistic | Why |
|-------|-----------|-----|
| 0 | Full-window mean | Level |
| 1 | Full-window std | Dispersion |
| 2 | Last-64 mean | Recent level |
| 3 | Last-64 std | Recent dispersion |
| 4 | Last-16 mean | Very recent level |
| 5 | Last-16 std | Very recent dispersion |
| 6 | Last tick value | Current state |
| 7 | **Linear-regression slope** | Trend direction & magnitude |
| 8 | **Short/long ratio** (mean₁₆ / mean_all, clamped [−3, 3]) | Regime: is the recent regime above/below window baseline |

Scales 7 and 8 were deliberate replacements for a weaker original design. The original
scale 7 was a window **max**, which is dominated by outliers, has no recency bias, and
conflates timing — it tells you *that* an extreme happened, not *when* or *in which
direction*. Replacing it with an OLS slope gives the model the one statistic most directly
tied to directional drift. Scale 8 (short/long ratio) is a cheap, powerful regime
indicator: a single number capturing "is recent behavior above or below the window's own
baseline."

### 5.1 Two-pass normalization

Summary vectors live on a wildly different scale per dimension (a slope and a std are not
comparable). A **second-pass z-score** is fit on the first 200K training summary vectors
and the resulting `feat_mean` / `feat_std` are **saved into the model file** (format v3,
magic `FSQ3`) so inference applies the identical transform.

> This too is an audit artifact. The single worst bug ever found in this codebase (BUG 1)
> was that this second-pass normalization was computed at train time, used in-memory, and
> **never persisted** — so inference fed the model raw-scale summaries it had never seen.
> The model's predictions were, in the audit's words, "meaningless." The fix — persisting
> the stats in the model binary — is why the format carries a version bump. See §13.

---

## 6. Labeling: from threshold heuristics to triple-barrier

### 6.1 The naive label and why it is wrong

The original label was `sign(Δ) where |Δ| > k · spread` — i.e., "did the mid move more
than 2× the spread over a fixed horizon?" This has two fatal defects:

1. **Fixed horizon ignores path.** A trade that hits +3 pips at tick 30 and then bleeds to
   −1 pip at the horizon is labeled by the *endpoint*, even though a real position with a
   stop would have exited at the high. The label does not match how money is actually made
   or lost.
2. **Fixed threshold ignores volatility regime.** 2× spread is a different event in a calm
   session than in a news spike. Constant thresholds mislabel both regimes.

### 6.2 Triple-barrier labeling

The roadmap (and the walk-forward path) move to **triple-barrier labeling** (López de
Prado, *Advances in Financial Machine Learning*, ch. 3). For each sample:

- An **upper barrier** (take-profit) and **lower barrier** (stop-loss), each set at a
  **volatility-scaled** multiple of recent realized vol — so the barriers breathe with the
  regime.
- A **vertical barrier** (time limit) at the horizon.
- The label is the **first barrier touched**: UP if TP first, DOWN if SL first, NEUTRAL if
  time expires untouched.

This produces labels that correspond to the outcome of an actual bracketed trade, scaled
to the prevailing volatility. The walk-forward metrics are computed **after costs** from
these labels (first-touch return minus spread cost), so a profit factor reported by the
system is a profit factor on simulated executions, not on a dimensionless proxy.

> Backtest-units bug (BUG 2): the legacy `backtest` command computed PnL in a
> dimensionless `|Δ| / spread` unit and subtracted a `0.30` "cost" of unspecified
> denomination. Its Sharpe and drawdown were therefore meaningless in absolute terms. The
> walk-forward path (`wf_metrics.hpp`) replaces this with after-cost net-return units and
> honest standard errors.

---

## 7. Sample dependence and uniqueness weighting

This is the most subtle — and most commonly ignored — issue in financial ML, and the
codebase handles it explicitly in `include/data/sample_weights.hpp`.

### 7.1 The problem

Windows are emitted every `stride = 64` ticks but each label spans a `horizon` of 128–256
ticks. **Consecutive samples therefore share future ticks.** Their labels are not
independent draws. Counting both at full weight:

- **inflates the effective sample size** (you think you have N independent observations;
  you have far fewer), which inflates every t-statistic and Sharpe, and
- **biases the fit** toward whichever regime happened to generate the most overlap.

### 7.2 The fix: concurrency-based uniqueness

From López de Prado ch. 4, implemented exactly:

1. Each label `i` occupies a tick span `[entry_i, entry_i + horizon)`.
2. **Concurrency** `c(t)` = number of label spans covering tick `t`.
3. **Uniqueness** `u_i = mean over the span of 1/c(t)` ∈ (0, 1].
4. Optional **linear time-decay** so recent samples count more than stale ones.

The implementation is an `O(n log n)` sweep: coordinate-compress all span endpoints into
elementary intervals, build a difference array for concurrency, take a prefix sum of
`length / concurrency`, and answer each sample's average-`1/c` as a single range query.
Weights are normalized to **mean 1.0** so they slot into the existing class-weighted loss
without rescaling it.

### 7.3 Why this propagates into the statistics, not just the loss

`wf_metrics.hpp` uses the overlap factor `block = reach / stride` to compute an
**effective sample count** `N_eff = trades / block`, and derives the standard error,
t-statistic, and 95% confidence interval from `N_eff` — **not** the raw trade count. This
is the difference between an honest and a dishonest Sharpe: the raw count would understate
the standard error by a factor of `√block`, manufacturing significance out of overlap.

```cpp
// wf_metrics.hpp — the honest standard error
st.n_eff = n / block;            // discount overlapping samples
st.se    = sd / std::sqrt(st.n_eff);
st.t_stat = mean / st.se;        // t on N_eff, not n
st.ci_lo = mean - 1.96 * st.se;
st.ci_hi = mean + 1.96 * st.se;
```

---

## 8. The learning machinery

### 8.1 The summary model (`GpuTrainer`)

A 3-layer fully-connected network: `243 → 256 → 128 → 3`, ReLU hidden activations,
softmax output over `{UP, NEUTRAL, DOWN}` (class convention `0=UP, 1=NEUTRAL, 2=DOWN`
everywhere). ~97K parameters. Trained with:

- **Adam** (β₁ = 0.9, β₂ = 0.999, ε = 1e-8) with **global-L2 gradient clipping**.
- **Class-weighted cross-entropy.** Gold ticks are overwhelmingly NEUTRAL; unweighted CE
  collapses to predicting NEUTRAL always. Class weights restore the minority directional
  classes. (A regression here — class-weight collapse — was a named fix in the git log.)
- **Label smoothing** proportional to distance from the decision threshold, so a sample
  that barely crossed the barrier does not receive the same hard target as an unambiguous
  one. This directly attacks overconfidence on borderline samples.
- **Cosine LR annealing with warm restarts** (Loshchilov & Hutter, SGDR).

### 8.2 The deep model (`DeepMlpTrainer`)

The N-layer generalization, consuming the raw `[window × 27]` window. The forward and
backward passes are textbook batched GEMM, reused verbatim across both trainers:

```
forward  layer l:  act[l+1] = act[l] @ W[l]^T            (then ReLU / softmax)
wgrad    layer l:  gW[l]     = grad[l+1]^T @ act[l]
igrad    layer l:  grad[l]   = grad[l+1] @ W[l]
```

ReLU on every hidden layer, softmax on output, Adam + global-L2 clip — identical optimizer
kernels to the summary trainer, so the two paths differ only in *what they see*, not *how
they learn*. This is deliberate: it isolates the value of the richer representation.

### 8.3 The broader layer library

The repository contains a substantial from-scratch layer/zoo beyond the production MLP —
attention, conv1d, GRU, LSTM, temporal-fusion, mixture-of-experts, FiLM conditioning,
gated residual networks, hyperbolic layers — plus training machinery (SAM sharpness-aware
minimization, IRM invariant-risk minimization, gradient surgery / PCGrad, experience
replay, curriculum). These are the building blocks for the Tier-1 base learners (temporal
transformer, TCN, GRU, MoE head) in the roadmap. They are infrastructure for *ensemble
diversity*, the third profit lever.

---

## 9. GPU systems engineering

Performance is a first-class concern because **breadth** (hundreds of fits) is only
affordable if a single fit is fast.

### 9.1 The fast pipeline (`train-fast`)

The throughput design:

- **Large batches** (8192, scaling to 16384+) to saturate the GEMM units.
- **Triple-buffered GPU pipeline** — prefetch queue (target depth 20–24) feeding a GPU
  pipeline (depth 2–3) so the device never stalls on host data.
- **16+ worker threads** for parallel feature extraction / windowing.
- **cuBLAS** for all dense linear algebra; custom CUDA kernels for the fused forward
  bias+ReLU, softmax+cross-entropy, and the Adam update.

Reported throughput on a single **RTX 3050 4GB**: ~2M samples/sec, ~88% GPU utilization,
~2.8 GB VRAM. The text-mode telemetry surfaces the exact bottleneck per step
(`data=` host time, `gpu=` device time, `queue=` prefetch depth), so a starved pipeline is
diagnosable at a glance.

### 9.2 On-the-fly window gather

The depth path would need ~440 GB if every `256 × 27` window were materialized. Instead the
**per-tick feature stream lives resident on the GPU once** (`[n_ticks × 27]`, ~11 GB for
100M ticks) and each sample's window is **gathered in a kernel** from a per-sample entry
index. This trades a tiny amount of gather bandwidth for a ~40× memory reduction and is
what makes raw-window training feasible at all on commodity GPUs.

### 9.3 Multi-GPU roadmap

Tier 2 targets an 8×L4 / 96-vCPU / 384 GB VM via two modes:

- **NCCL data-parallel** for one flagship transformer across all 8 GPUs (effective batch
  ~32K, mixed precision / BF16 tensor cores).
- **Fleet mode** — each GPU trains a different `(architecture × fold × regime × seed)`,
  yielding ~180 fits in a single ~6-hour window.

### 9.4 A note on the fused-kernel dimension hazard

The fused forward kernel originally hardcoded `IN = 176` and stepped `for (k = 0; k < IN;
k += 4)`, assuming divisibility by 4. The dimension grew to 243 (`243 % 4 = 3`), which
requires explicit tail handling. This is documented in the audit precisely because a
hardcoded, silently-wrong loop bound is the kind of bug that does not crash — it just
quietly corrupts the last few features. Dimension constants now flow from `common.h`.

---

## 10. Validation: purged walk-forward, deflation, and the holdout discipline

This section is the heart of the project. **The entire apparatus exists to produce a
number you are allowed to believe.**

### 10.1 Why a naive split lies

A random or even simple-temporal train/val split leaks in financial data because of label
overlap (§7): a training sample whose horizon extends into the validation period shares
future ticks with validation samples. The model sees the answer.

### 10.2 Anchored, purged, embargoed walk-forward

`walkforward.cpp` implements anchored walk-forward CV with two leakage guards from López
de Prado ch. 7:

- **Purge.** Remove training samples whose label span overlaps the validation window's
  ticks. No shared future between train and test.
- **Embargo.** Additionally drop a buffer of training samples immediately *after* the
  validation window, because serial correlation can leak backwards through features.

Each fold trains the **same** `SequenceModel` via `GpuTrainer` and reports honest,
after-cost metrics (winrate, edge, profit factor, Kelly, max drawdown, Sharpe, t-stat,
N_eff-based CI). The deep variant `wfdeep.cpp` does the same over raw windows and — by
design — `#include`s the *same* `wf_metrics.hpp`, so the summary and deep paths produce
**byte-identical statistics logic**. You cannot accidentally grade the two models on
different rulers.

### 10.3 Deflated Sharpe — discounting the search

This is the lever that kills the #1 overfitting risk the audit identified:
**select-best-of-N-on-validation.** If you try 200 configurations and report the best
in-sample Sharpe, you have not found an edge; you have found the maximum of 200 noise
draws. The **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) corrects the observed
Sharpe for (a) the number of independent trials, (b) the variance of the trial Sharpes,
and (c) the non-normality (skew/kurtosis) of returns, yielding the probability that the
true Sharpe exceeds zero given the size of the search. The roadmap reports DSR over the
count of configurations tried, so the final number is honest about how hard you looked.

### 10.4 The one-shot holdout

The final, most-recent slice of data is an **untouched holdout**. It is evaluated **once**,
at the very end, and the after-cost PF / Sharpe / Kelly / maxDD on it is the reported
result. Touching it more than once — tuning anything after seeing it — converts it into a
validation set and forfeits its meaning. The discipline of *one look* is the only thing
that makes the holdout number trustworthy.

### 10.5 Adversarial validation

`validate-adversarial.cpp` evaluates the signal against **naive baselines** — pure
momentum and pure contrarian strategies — across regime/session buckets, with per-trade
PnL, 2× spread sensitivity, and per-environment breakdowns. A signal that cannot beat
"buy because it went up" after costs is not a signal. This is the sniff test before any
holdout is spent.

---

## 11. Position sizing and the economics of the signal

A directional probability is not a strategy until it is sized. The roadmap's sizing rule:

> **position size = meta-probability × fractional-Kelly, capped.**

**Kelly fraction.** For a bet with win probability `p` and win/loss ratio `b`,
`f* = p − (1−p)/b` is the growth-optimal fraction (Kelly 1956). The codebase computes this
directly from the trade distribution:

```cpp
// wf_metrics.hpp
double b = avg_win / avg_loss;
st.kelly = winrate - (1.0 - winrate) / (b + 1e-12);
```

**Why fractional, why capped.** Full Kelly is growth-optimal but assumes the edge is known
exactly. It never is — `p` and `b` are estimated with error, and full Kelly on an
overestimated edge is ruinous. Fractional Kelly (a constant fraction of `f*`) trades a
little growth for a large reduction in drawdown variance and robustness to estimation
error. The cap bounds tail risk from a single mis-sized bet.

**Meta-labeling.** The roadmap stores a meta-label target now and trains a meta-model in
Tier 3: the primary model decides *direction*, the meta-model decides *whether to take the
trade at all* and with what confidence. This López de Prado construction (ch. 3) is what
turns a noisy directional classifier into a sized, selective strategy — it concentrates
capital where the base model is reliable and stands aside where it is not.

The validation metric table the project holds itself to:

| Metric | Meaning | "Real edge" threshold |
|--------|---------|----------------------|
| `dir_acc` | Accuracy on UP/DOWN (ignoring NEUTRAL) | > 0.52 |
| `winrate` | Win rate on confidence-gated trades | > 0.55 |
| `edge` | Expected value per trade | > 0.02 |
| `pf` | Profit factor (gross win / gross loss) | > 1.3 |
| `kelly` | Optimal bet fraction | > 0.05 |

`kelly > 0` after a few thousand steps is the project's tripwire for "real signal";
`kelly < 0` is treated as a *bug signal* (usually a normalization-pipeline fault), not a
modeling result — a healthy instinct.

---

## 12. The Medallion-Lite roadmap (tiers P1–P5)

The project is staged so each phase is independently testable and each can *kill the thesis
early* before heavy compute is spent.

- **P1 — Honest foundation (C++, no CUDA).** Triple-barrier labels, sample-uniqueness
  weights, purged/embargoed walk-forward, untouched holdout, deflated metrics. *This phase
  may reveal whether an edge exists at all before any GPU work* — the most important
  return-on-effort decision in the project. **(Current focus.)**
- **P2 — GPU sequence model.** Transformer in cuBLAS, BF16, on-the-fly windows
  (`wfdeep` / `DeepMlpTrainer`).
- **P3 — NCCL 8-GPU data-parallel.** Scale the flagship.
- **P4 — Stacking.** Out-of-fold base predictions → meta-model → regime-gated routing over
  the 7 session/volatility environments.
- **P5 — Flagship scale-up + final deflated-Sharpe holdout report.** Fleet + flagship
  orchestrated across 8×L4 for ~6 hours; the single honest number at the end.

**Model census at full scale:** 4 base architectures × ~15 walk-forward folds × 3 seeds ≈
**180 base fits** + 1 NCCL flagship + 1 stacker (+ 7 optional regime specialists) → one
deployable ensemble. Depth × breadth, as promised.

---

## 13. Audit log: bugs found, weaknesses, and the epistemics of self-distrust

The repository carries a full adversarial audit (`AUDIT_LOG.md`, 7-lens). It is reproduced
here in summary because **the audit is a feature, not an embarrassment** — a quant system
that has never been audited is a system whose backtest you should not trust.

**Critical bugs found:**

1. **Second-pass normalization never persisted (deployment blocker).** Stats computed at
   train time, applied in-memory, never written to the model file; inference fed raw-scale
   inputs → predictions meaningless. *Fixed by persisting `feat_mean`/`feat_std` in model
   format v3.*
2. **Backtest PnL in synthetic units.** Dimensionless `|Δ|/spread` with an
   unspecified-unit `0.30` cost; absolute Sharpe/drawdown meaningless. *Fixed by after-cost
   net-return units in the walk-forward path.*
3. **Kyle–Hasbrouck λ hardcoded to 0.** A dead feature consuming model capacity. *Roadmap
   restores the rolling estimator.*

**Architectural weaknesses found:** weak summary statistic (max → slope), missing
short/long ratio (added), hardcoded `IN = 176` in two places + a `% 4` kernel assumption,
no profit-factor/Kelly in validation (added), silent low-confidence skips at inference
(ambiguous "flat vs missed"), hard one-hot labels (label smoothing added).

The meta-lesson is the point: **every one of these bugs would have produced a
plausible-looking backtest.** The normalization bug in particular would have shown a model
that "trained fine" and then silently failed live. The project's defense is not cleverness
— it is a standing assumption that the backtest is lying until proven otherwise, encoded as
purging, embargo, N_eff standard errors, deflation, a one-shot holdout, and adversarial
baselines.

---

## 14. Reproducibility, build, and CLI

### 14.1 Build

```bash
# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFROM_NATIVE_ARCH=ON
cmake --build build --config Release

# GPU (CUDA 12.x + cuBLAS)
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 \
      -DFROM_CUDA=ON -DFROM_NATIVE_ARCH=ON
cmake --build build-cuda --config Release
```

### 14.2 Core commands

```bash
# Fast training (production path)
from train-fast --data XAUUSD_ticks_all.parquet --max-steps 100000

# Purged/embargoed walk-forward over the summary model
from walkforward --data XAUUSD_ticks_all.parquet

# Walk-forward over raw windows (deep model)
from wfdeep --data XAUUSD_ticks_all.parquet

# Adversarial baselines (momentum / contrarian / regime buckets)
from validate-adversarial --data XAUUSD_ticks_all.parquet

# Inference, inspection, benchmarking
from infer --model weights.from --data ticks.parquet --output signals.bin
from inspect --file weights.from
from bench --gpu
```

### 14.3 Determinism caveats

Fixed seeds are used per fit, but multi-threaded float reduction and cuBLAS algorithm
selection make bit-exact reproducibility across machines aspirational, not guaranteed.
The **statistics** are reproducible (the walk-forward path is deterministic given the cache
and seeds); the last ULP of a weight is not. For a research artifact whose output is a
deflated Sharpe and a holdout PF, this is the correct trade-off.

---

## 15. Glossary

- **OFI** — Order Flow Imbalance; signed net change in book pressure.
- **Kyle's λ** — price-impact coefficient; slope of price on signed order flow.
- **Amihud illiquidity** — |return| per unit volume.
- **Roll spread** — effective spread implied by serial covariance of price changes.
- **Lee–Ready** — trade-sign classification via tick/quote test.
- **Triple-barrier** — labeling by first touch among TP / SL / time barriers.
- **Purge / Embargo** — removal of train samples overlapping (purge) or immediately
  following (embargo) the validation window.
- **Uniqueness weight** — `mean(1/concurrency)` over a label's span; down-weights
  overlapping samples.
- **N_eff** — effective sample count = trades / overlap factor; the denominator for honest
  standard errors.
- **Deflated Sharpe Ratio (DSR)** — Sharpe corrected for the number of trials, trial
  variance, and return non-normality.
- **Kelly fraction** — growth-optimal bet size `p − (1−p)/b`.
- **Meta-labeling** — a second model deciding whether to act on the primary model's signal.

---

## 16. References

- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of
  Portfolio Management.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The price impact of order book events.*
  Journal of Financial Econometrics.
- Hasbrouck, J. (2007). *Empirical Market Microstructure.* Oxford University Press.
- Kelly, J. (1956). *A New Interpretation of Information Rate.* Bell System Technical
  Journal.
- Kyle, A. (1985). *Continuous Auctions and Insider Trading.* Econometrica.
- Lee, C. & Ready, M. (1991). *Inferring Trade Direction from Intraday Data.* Journal of
  Finance.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (Chs. 3, 4,
  7 — labeling, sample weights, cross-validation — are the project's backbone.)
- Amihud, Y. (2002). *Illiquidity and stock returns.* Journal of Financial Markets.
- Roll, R. (1984). *A Simple Implicit Measure of the Effective Bid-Ask Spread.* Journal of
  Finance.
- Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm
  Restarts.* ICLR.

---

*This document describes a research apparatus. It does not constitute financial advice,
nor a claim of realized trading profit. Its single most important contribution is the
machinery that lets it tell the difference between an edge and a daydream.*
