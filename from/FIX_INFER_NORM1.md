# Fix: Inference first-pass normalization (norm1) — make validated edge tradeable

Date: 2026-06-21

## Goal

"Make the edge tradeable." The validated models (walk-forward / wfdeep artifacts)
could not be deployed correctly: inference fed wrong-scale features into a model
trained on z-scored inputs, producing meaningless signals.

## Background — the two-pass normalization contract

A deployable `from` sequence model needs BOTH normalization passes reproduced at
inference time, in order:

1. **norm1** — first-pass Welford per-feature z-score, applied to RAW features
   BEFORE the `MultiScaleSummarizer`. Persisted as `norm1.bin` (magic `NRM1`).
2. **feat_norm** — second-pass z-score on the 243-dim summary vector. Persisted
   inside `model.from` (FSQ3 v3, `feat_norm_ready` flag).

`cli/walkforward.cpp` and `cli/wfdeep.cpp` write the full deployable artifact dir:

```
<model_dir>/<id>/
  model.from      weights + feat_norm (second pass)
  norm1.bin       first-pass Welford Normalizer state (NRM1)
  meta.json
  report.txt
```

## Bug

`cli/infer.cpp` constructed a fresh default `Normalizer` and applied it with
`update=false`, but **never loaded `norm1.bin`**. A default Normalizer has
`count=0` → variance defaults to `1.0`, mean `0.0` → features pass through
essentially RAW. The model was trained on properly z-scored features → scale
mismatch → degraded / meaningless predictions.

This is the same class as audit BUG1 (second-pass stats unsaved), one layer
lower — the first-pass normalizer was saved by the trainer but never reloaded by
inference.

Note: audit BUG1 (feat_norm save/load) and BUG2 (backtest PnL in pips, real
entry/exit mids, first-touch barrier exit) were already fixed in the codebase
before this change. This fix closes the remaining first-pass gap.

## Fix

Two files, three hunks. No behavior change to training; inference now reproduces
the exact raw→normalized mapping the model trained on.

### `include/io/artifact.hpp`

Added `load_norm1()` — byte-for-byte inverse of the existing `save_norm1()`,
mirroring the inline `NRM1` reader in `cli/walkforward.cpp`. Reads
`mean/m2/count/frozen` and applies via `Normalizer::set_state`. Returns false on
missing file / bad magic / truncation; leaves the Normalizer untouched on
failure.

### `cli/infer.cpp`

- `#include "io/artifact.hpp"`.
- Track the first model file actually loaded (`loaded_model_path`) in both the
  single and ensemble branches.
- After loading the model(s), resolve the first-pass normalizer:
  - `--norm1 <path>` if provided, else `norm1.bin` beside the model file.
  - **Hard-fail** (`require`) if the file is missing or fails to load. Silent
    wrong-scale inference is worse than a loud stop.
  - Load it into the `Normalizer` used by `normalize_chunk`.

## Usage

```
from infer --model run_dir/<id>/model.from --input ticks.parquet --output signals.csv
# auto-loads run_dir/<id>/norm1.bin

# override location:
from infer --model weights_best.from --norm1 path/to/norm1.bin ...
```

## Scope / not done

- `cli/train.cpp` (legacy quick-trainer) does NOT emit `norm1.bin`. Its
  `Normalizer` is scoped inside the `!cache_loaded` block and absent on
  cache-load; emitting a correct sidecar there is a cache-format refactor with a
  cache-miss trap. After this change, `infer` on a `train`-produced model
  hard-fails unless `--norm1` is passed. Deployable models come from
  `walkforward` / `wfdeep`.
- `wfdeep` `deep.bin` (raw-window deep MLP) has no CPU `infer` path; separate
  deployment story. Its per-tick mean/std are embedded in `deep.bin`, not in
  `norm1.bin`.

## Verification

Not compiled here — no MSVC vcvars CRT environment in the shell. Changes use
existing APIs and mirror proven walk-forward code. Build with the CUDA target to
confirm:

```
cmake --build build-cuda --target from
```

## Files changed

- `include/io/artifact.hpp` — added `load_norm1()`.
- `cli/infer.cpp` — include, track model path, load + require `norm1.bin`.
