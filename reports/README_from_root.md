# from — XAUUSD Direction Prediction Engine

Custom C++ ML framework for tick-level gold price direction prediction. Trains a 3-layer MLP on microstructure features using cuBLAS-accelerated GPU training.

See [`from/`](from/) for the full source code and documentation.

## Quick Start

```bash
cd from
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DFROM_CUDA=ON -DFROM_NATIVE_ARCH=ON
cmake --build build-cuda --config Release
# Place XAUUSD_ticks_all.parquet in from/, then:
build-cuda/Release/from.exe train --data XAUUSD_ticks_all.parquet --lr 0.0003 --batch-size 1024 --max-steps 999999999 --validate-every 5000 --save-every 50000 --max-samples 5000000
```

## Architecture

```
Raw XAUUSD Ticks (539M rows, 6.4GB parquet)
    -> 22 microstructure features per tick
    -> 512-tick windows, 256-tick prediction horizon
    -> 8 statistical scales x 22 features = 176-dim input
    -> MLP: 176 -> 256 -> 128 -> 3 (UP/NEUTRAL/DOWN)
    -> Confidence-gated trading (only trade when >50% confident)
```

78,595 parameters | ~2M samples/sec on RTX 3050 | Z-score normalized | Class-weighted CE loss
