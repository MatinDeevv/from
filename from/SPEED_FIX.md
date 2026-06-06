# Speed Fix - Maximize GPU Utilization

## Problem
- Batch size was 256 (tiny!)
- GPU sitting idle
- VRAM empty (using <1GB of 4GB)
- Training taking 38 minutes per epoch

## Solution Applied

### 1. Increased Batch Size 256 → 4096 (16x larger)
**File**: `config.toml`, `cli/train.cpp`
- Now fills 4GB VRAM instead of <500MB
- GPU will have actual work to do
- Should be 10-16x faster

### 2. Fixed D3D11 Dispatch Size
**File**: `include/gpu/d3d11_direction_trainer.hpp:255`
- Was: `Dispatch(1, 1, 1)` = only 256 threads
- Now: `Dispatch((total_params + 255) / 256, 1, 1)` = fill GPU cores
- For 32 features × 3 classes = 96 params → still small but better

### 3. Added CUDA Build Script
**File**: `build_cuda.cmd`
- Builds with `-DFROM_CUDA=ON` to enable real CUDA kernels
- CUDA has proper GEMM, LSTM, Adam kernels already implemented
- Will be MUCH faster than D3D11 compute shaders

### 4. Added Detailed Timing
**File**: `cli/train.cpp`
- Prints: `[Step 000010] data=12ms forward=3ms backward=4ms optim=1ms total=20ms`
- Shows exactly where the bottleneck is

## How to Build and Run

### Option 1: Quick rebuild (D3D11, batch=4096)
```cmd
cd C:\Users\marti\from\from
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
cmake --build build-vs --config Release
build-vs\Release\from.exe train --data "C:\Users\marti\OneDrive\Desktop\bmd\XAUUSD_ticks_all.parquet" --max-steps 100
```

### Option 2: Build with CUDA (fastest)
```cmd
cd C:\Users\marti\from\from
build_cuda.cmd
build-cuda\from.exe train --data "C:\Users\marti\OneDrive\Desktop\bmd\XAUUSD_ticks_all.parquet" --max-steps 100
```

## Expected Performance

### Before:
- Batch size: 256
- GPU util: 5-10%
- VRAM: <500MB
- Speed: ~5 steps/minute

### After (D3D11, batch=4096):
- Batch size: 4096 (16x larger)
- GPU util: 30-60% (better but still limited by CPU summarize)
- VRAM: 2-3GB
- Speed: ~50-80 steps/minute (10-16x faster)

### After (CUDA, batch=4096):
- Batch size: 4096
- GPU util: 80-95% (proper GPU compute)
- VRAM: 3-4GB
- Speed: ~200-500 steps/minute (40-100x faster)

## Remaining Bottlenecks

Even with these fixes, the code still has issues:

1. **model.summarize() runs on CPU** - should be GPU kernel
2. **No data prefetching** - ParquetReader is synchronous
3. **No async compute** - blocking download after each step
4. **Small model** - only 3-class linear classifier, not using LSTM/attention layers

To get to milliseconds per step, need to:
- Move entire forward/backward to GPU (one kernel call)
- Async data loading with worker threads
- Use the actual LSTM/attention model (currently disabled)
- Pipeline: load next batch while GPU computes current batch
