# from — XAUUSD Direction Prediction Engine

Custom C++ ML framework for tick-level gold price direction prediction. Trains a 3-layer MLP on microstructure features using cuBLAS-accelerated GPU training.

## Architecture

```
Raw XAUUSD Ticks (539M rows)
    -> TickProcessor (22 microstructure features per tick)
    -> Windower (512-tick windows, 256-tick prediction horizon)
    -> MultiScaleSummarizer (8 scales x 22 features = 176-dim input)
    -> MLP: 176 -> 256 -> 128 -> 3 (UP/NEUTRAL/DOWN)
    -> Confidence-gated trading decisions
```

**78,595 parameters** — small enough to run inference in microseconds.

## Features

- **22 microstructure features**: OFI, velocity, acceleration, multi-scale volatility (RV16/64/256), Lee-Ready, Amihud, bid-ask bounce, microprice deviation
- **Z-score normalized inputs** with clipping
- **Class-weighted cross-entropy** (prevents majority-class collapse)
- **Cosine LR annealing** with warm restarts
- **Trading-relevant validation**: directional accuracy, confidence-gated win rate, expected edge
- **2.0x spread threshold** labels only real moves
- **Temporal train/val split** (no future leakage)
- **cuBLAS GPU training**: ~2M samples/sec on RTX 3050 4GB
- **Binary cache**: first run processes parquet (~5min), subsequent runs instant

## Build

```bash
# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFROM_NATIVE_ARCH=ON
cmake --build build --config Release

# GPU (CUDA + cuBLAS)
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DFROM_CUDA=ON -DFROM_NATIVE_ARCH=ON
cmake --build build-cuda --config Release
```

## Train

```bash
./from train --data XAUUSD_ticks_all.parquet --lr 0.0003 --batch-size 1024 \
  --max-steps 999999999 --validate-every 5000 --save-every 50000 --max-samples 5000000
```

On Windows: `RUN.cmd`

## Validation Metrics

- `dir_acc` — accuracy on directional (UP/DOWN) predictions only
- `winrate` — win rate on confidence-gated trades (>50% confidence)
- `edge` — expected value per trade (positive = profitable)
- `prec_UP/DN` — precision per direction

## File Structure

```
cli/            Command implementations (train, infer, test, bench)
include/
  common.h     Constants (22 features, epsilon)
  cuda/        GPU trainer (cuBLAS forward/backward/Adam)
  data/        Parquet reader, tick processor, windower, normalizer
  layers/      Neural network layer implementations
  model/       SequenceModel (main MLP), serializer, regime gate
  training/    Optimizer, loss, scheduler, curriculum
  utils/       Config parser, timer, thread pool
src/            Implementations (.cpp, .cu)
config.toml    Hyperparameters
CMakeLists.txt Build system
RUN.cmd        Windows training launcher
```
