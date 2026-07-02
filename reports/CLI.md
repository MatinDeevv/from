# FROM - CLI Reference

## Quick Start (FASTEST)

```cmd
from train-fast --data XAUUSD_ticks_all.parquet --max-steps 1000
```

That's it! Training starts in <1 second.

## Commands

### `from train-fast` ⚡ FASTEST
**Maximum speed mode - 80% GPU, 80% CPU, 80% RAM**

```cmd
from train-fast --data <file.parquet> [options]
```

**Options:**
- `--data <path>` - Parquet file (required)
- `--max-steps <N>` - Stop after N steps (default: 1000000)
- `--no-ui` - Disable dashboard, show text output
- `--config <path>` - Config file (default: config.toml)

**Features:**
- ✅ Batch size 8192 (fills VRAM)
- ✅ 16 worker threads (uses all CPU cores)
- ✅ 24-batch prefetch (fills RAM)
- ✅ Triple-buffered GPU pipeline
- ✅ Starts training in <1 second

**Example:**
```cmd
from train-fast --data ticks.parquet --max-steps 10000 --no-ui
```

---

### `from train` 🐌 SLOW (legacy)
**Original synchronous training (40-100x slower)**

```cmd
from train --data <file.parquet> [options]
```

**Options:**
- `--data <path>` - Parquet file
- `--max-steps <N>` - Stop after N steps
- `--batch-size <N>` - Batch size (default: 256)
- `--epochs <N>` - Number of epochs (default: 1)
- `--lr <float>` - Learning rate (default: 0.0003)
- `--no-ui` - Disable dashboard
- `--config <path>` - Config file

**Use only for debugging. Use `train-fast` for actual training.**

---

### `from infer`
**Run inference on trained model**

```cmd
from infer --model weights.from --data ticks.parquet
```

**Options:**
- `--model <path>` - Model weights file (required)
- `--data <path>` - Input data (required)
- `--output <path>` - Output file for predictions

---

### `from inspect`
**Inspect parquet file or model weights**

```cmd
from inspect --file <path>
```

**Shows:**
- File size, row count
- Column names and types
- Sample rows
- For models: architecture, layer sizes

---

### `from bench`
**Benchmark GPU/CPU performance**

```cmd
from bench [--gpu] [--cpu]
```

**Tests:**
- Matrix multiply throughput
- Memory bandwidth
- Data loading speed

---

### `from test`
**Run unit tests**

```cmd
from test
```

## Output Format

### Dashboard Mode (default)
```
+--------------------------------------------------------------------------+
|  FROM - Neural Market Intelligence Engine                               |
|  Training XAUUSD_ticks_all.parquet | Epoch  1/1 | Step     218          |
+--------------------------------------------------------------------------+
| Epoch Progress  [############################.........]  73.2%          |
| Total Progress  [############################.........]  73.2%          |
+--------------------------------------------------------------------------+
| LOSS                        ACCURACY                                     |
| total   | 1.0966 v         direction |  37.1% ^                         |
| dir     | 1.0966           vs random |  33.3%                           |
+--------------------------------------------------------------------------+
```

### Text Mode (`--no-ui`)
```
[Step 000010] data=8ms gpu=15ms total=23ms | steps/sec=43.5 rows/sec=22323200 | loss=1.0966 acc=0.371 | queue=23 gpu_q=2 | gpu=88.5% vram=2.8GB cpu=82.1%
```

**Metrics:**
- `data=` - Data loading time per step
- `gpu=` - GPU compute time per step
- `total=` - Wall clock time per step
- `steps/sec=` - Training throughput
- `rows/sec=` - Data processing throughput
- `queue=` - Prefetch queue size (target: 20-24)
- `gpu_q=` - GPU pipeline depth (target: 2-3)
- `gpu=` - GPU utilization % (target: 80-95%)
- `vram=` - VRAM usage (target: 2.5-3.5GB)
- `cpu=` - CPU utilization % (target: 70-90%)

## Configuration

Edit `config.toml` to change defaults:

```toml
[training]
batch_size = 8192      # Large batch for GPU
epochs = 50
learning_rate = 0.0003

[data]
chunk_size = 4000000   # Large chunks for parallel loading
window_size = 512
stride = 64
horizon = 128

[hardware]
use_cuda = true        # Enable GPU
num_workers = 16       # CPU threads for data loading
```

## Performance Tips

### ✅ DO
- Use `train-fast` (not `train`)
- Use `--no-ui` for logging/scripts
- Set `batch_size=8192` (fills VRAM)
- Use all CPU cores (`num_workers=16`)
- Monitor GPU/CPU/RAM in Task Manager

### ❌ DON'T
- Use `train` command (40-100x slower)
- Set batch_size < 4096 (wastes GPU)
- Set num_workers < 8 (wastes CPU)
- Run multiple trainings simultaneously (resource contention)

## Troubleshooting

### "Out of memory"
- Reduce `batch_size` in config.toml (try 4096 or 2048)

### "GPU not available"
- Check Task Manager → GPU
- Update NVIDIA drivers
- Rebuild with `build_maximum_speed.cmd`

### "Training too slow"
- Use `train-fast` not `train`
- Check GPU util (should be 80-95%)
- Check CPU util (should be 70-90% all cores)
- Check queue size (should be 20-24)

### "Queue dropping to 0"
- Increase `chunk_size` in config
- Increase `num_workers`
- Check disk I/O (SSD recommended)

## Examples

### Quick test (10 steps)
```cmd
from train-fast --data data.parquet --max-steps 10 --no-ui
```

### Full training run
```cmd
from train-fast --data data.parquet --max-steps 100000
```

### Train overnight with logging
```cmd
from train-fast --data data.parquet --no-ui > train.log 2>&1
```

### Custom config
```cmd
from train-fast --data data.parquet --config my_config.toml
```

## System Requirements

### Minimum
- GPU: 2GB VRAM (batch_size=2048)
- CPU: 4 cores
- RAM: 8GB
- Storage: SSD recommended

### Recommended (for 80% utilization)
- GPU: 4GB VRAM (batch_size=8192)
- CPU: 8+ cores (16+ threads)
- RAM: 16GB
- Storage: NVMe SSD

### Maximum Performance
- GPU: 8GB+ VRAM (batch_size=16384+)
- CPU: 16+ cores (32+ threads)
- RAM: 32GB+
- Storage: NVMe RAID
