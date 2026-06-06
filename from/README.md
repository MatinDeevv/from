# from — XAUUSD Direction Prediction Engine

Custom C++ ML framework for tick-level gold price direction prediction. Trains a 3-layer MLP on microstructure features using cuBLAS-accelerated GPU training.

## Architecture

```
Raw XAUUSD Ticks (539M rows)
    -> TickProcessor (27 microstructure features per tick)
    -> Windower (512-tick windows, 256-tick prediction horizon)
    -> MultiScaleSummarizer (9 scales x 27 features = 243-dim input)
    -> MLP: 243 -> 256 -> 128 -> 3 (UP/NEUTRAL/DOWN)
    -> Confidence-gated trading decisions
```

**~97K parameters** — small enough to run inference in microseconds.

## Features

### 27 Microstructure Features

1-6: Ask, Bid, Mid, Ask Volume, Bid Volume, Spread
7-8: Normalized Spread, OFI (Order Flow Imbalance)
9-11: Delta OFI, Mid Velocity, Mid Acceleration
12: Microprice Deviation
13-14: Tick Rate, Log Tick Rate
15-17: Realized Volatility (16/64/256 tick windows)
18: Roll Spread
19: Lee-Ready Sum (32-tick)
20: Amihud Illiquidity
21: Bid-Ask Bounce
22: **Kyle-Hasbrouck Lambda** — price impact per unit signed volume (64-tick rolling)
23: **Trade Imbalance Ratio** — asymmetry of buyer/seller aggression (32-tick)
24: **Volatility Regime Indicator** — RV16/RV256 ratio (trending vs ranging)
25: **Spread Compression Signal** — z-score of norm spread vs 64-tick baseline
26: **Autocorrelation Lag-1** — mean reversion vs momentum indicator (32-tick)
27: **Volume Clock Deviation** — tick rate vs EMA baseline

### 9 Summary Scales

For each of the 27 features, the MultiScaleSummarizer computes:
0. Full-window mean
1. Full-window std
2. Last-64-tick mean
3. Last-64-tick std
4. Last-16-tick mean
5. Last-16-tick std
6. Last tick value
7. **Linear regression slope** (trend direction/magnitude over window)
8. **Short/long ratio** (mean_16 / mean_all, clamped [-3, 3])

### Normalization Pipeline

Two-pass normalization ensures correct scale at both train and inference time:

1. **Welford normalizer** — applied to raw 27 features during tick processing (frozen after 100K samples)
2. **Second-pass z-score** — computed on the 243-dim summary vectors from the first 200K training samples, then saved into the `.from` model file. Applied identically at inference.

Both passes are persisted: the Welford stats in the normalizer, the summary z-score stats (`feat_mean`/`feat_std`) in the model binary (format v3).

### Training Features

- **Label smoothing** — confidence proportional to distance from threshold
- **Class-weighted cross-entropy** (prevents majority-class collapse)
- **Cosine LR annealing** with warm restarts
- **Trading-relevant validation**: dir_acc, win rate, edge, Kelly fraction, profit factor
- **2.0x spread threshold** labels only real moves
- **Temporal train/val split** (no future leakage)
- **cuBLAS GPU training**: ~2M samples/sec on RTX 3050 4GB
- **Binary cache**: first run processes parquet (~5min), subsequent runs instant. Auto-invalidates on dimension mismatch.

## Build

```bash
# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFROM_NATIVE_ARCH=ON
cmake --build build --config Release

# GPU (CUDA + cuBLAS) — requires CUDA Toolkit 12.x
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DFROM_CUDA=ON -DFROM_NATIVE_ARCH=ON
cmake --build build-cuda --config Release
```

## Train

```bash
./from train --data XAUUSD_ticks_all.parquet --lr 0.0003 --batch-size 1024 \
  --max-steps 999999999 --validate-every 5000 --save-every 50000 --max-samples 5000000
```

Note: delete any existing `.cache` file when changing features/scales — the cache auto-detects dimension mismatches and regenerates, but prints a clear message when this happens.

## Validation Metrics Guide

| Metric | What it means | Good threshold |
|--------|--------------|----------------|
| `dir_acc` | Accuracy on UP/DOWN predictions (ignoring NEUTRAL) | > 0.52 |
| `winrate` | Win rate on confidence-gated trades (>50% conf) | > 0.55 |
| `edge` | Expected value per trade | > 0.02 |
| `pf` | Profit factor (gross profit / gross loss) | > 1.3 |
| `kelly` | Kelly fraction (optimal bet sizing) | > 0.05 = real edge |
| `prec_UP/DN` | Precision per direction | > 0.50 |

Kelly > 0 after 2000 steps indicates the model is finding real signal. Kelly < 0 means expected loss — check normalization pipeline.

## File Structure

```
cli/            Command implementations (train, infer, test, backtest, bench)
include/
  common.h     Constants (27 features, feature enum)
  cuda/        GPU trainer (cuBLAS forward/backward/Adam)
  data/        Parquet reader, tick processor, windower, normalizer
  layers/      Neural network layer implementations
  model/       SequenceModel (main MLP), serializer, regime gate
  training/    Optimizer, loss, scheduler, curriculum
  utils/       Config parser, timer, thread pool
src/            Implementations (.cpp, .cu)
config.toml    Hyperparameters
CMakeLists.txt Build system
AUDIT_LOG.md   Bug audit and architecture notes
```

## Model File Format

Version 3 (magic: `FSQ3`). Backward-compatible with v2 files (loaded without feat_norm — inference will require retraining).

Contents: weights (w1/b1/w2/b2/w3/b3) + regime gate (centroids, covariance, calibration) + feat_norm (mean/std vectors for second-pass normalization).
