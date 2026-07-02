# Performance Diagnosis - Training Too Slow

## Current Status: 38 minutes per epoch (540M ticks)
**Expected**: Milliseconds per step
**Actual**: Seconds per step

## Root Causes Identified

### 1. FAKE GPU ACCELERATION (CRITICAL)
File: `include/gpu/d3d11_direction_trainer.hpp:226-263`

The "GPU training" is actually CPU training with GPU overhead:
- Calls `model.evaluate()` on **CPU** for entire batch
- Calls `model.summarize()` on **CPU** for every sample (256 × 512 × 16 = 2.1M operations)
- Dispatches trivial compute shader with `Dispatch(1, 1, 1)` - only 256 threads total!
- **Immediately downloads results back** with blocking CPU Map/Unmap
- No overlap, no pipelining, no async compute

**This is CPU training disguised as GPU, not real GPU acceleration.**

### 2. Synchronous Parquet Reading
File: `include/data/parquet_reader.hpp`, `src/data/parquet_reader.cpp`

The ParquetReader is completely synchronous:
- No prefetching
- No async I/O
- No worker threads
- GPU sits idle while waiting for data

Config says `num_workers = 16` but they're not being used for data loading.

### 3. CPU Model Bottleneck
File: `include/model/direction_model.hpp:35-45`

The `summarize()` function:
- Runs on CPU for every sample
- Loops over 512 timesteps × 16 features per sample
- No vectorization (no AVX2/NEON)
- No parallelization (no OpenMP)

For batch_size=256: **2.1 million float operations on single CPU core**

## Timing Breakdown Added

Modified `cli/train.cpp` to print every 10 steps:
```
[Step 000010] data=12ms forward=3ms backward=4ms optim=1ms total=20ms rows/sec=14000000
```

This will show whether the bottleneck is:
- Data loading (should be <10ms)
- Forward pass (should be <5ms for batch=256)
- Backward pass (should be <5ms)
- Optimizer (should be <2ms)

## What Needs to be Fixed

### Priority 1: Move summarize() to GPU
The entire forward/backward pass should be one GPU kernel call:
- Upload raw 512×16 sequences to GPU once
- Compute summarization on GPU
- Compute forward pass on GPU
- Compute gradients on GPU
- Download only final loss scalar + updated weights

### Priority 2: Async Data Pipeline
- Spawn worker threads to read/decode/window data
- Use triple-buffering: read next chunk while GPU processes current batch
- ParquetReader should decode in parallel (chunk_size=2M is huge)

### Priority 3: Real GPU Dispatch
Current: `Dispatch(1, 1, 1)` = 256 threads total
Should be: `Dispatch((batch_size + 255) / 256, 1, 1)` for batch-level parallelism

### Priority 4: Eliminate CPU Fallback
The D3D11 trainer falls back to CPU `model.evaluate()` and `model.summarize()`.
Should dispatch compute shaders for everything.

## Next Steps

1. Run with new timing diagnostics to confirm bottleneck
2. Disable fake GPU training (use CPU path explicitly) 
3. Profile actual wall time: data loading vs compute
4. Implement proper GPU kernels or use CUDA if available
