# AUDIT LOG — Full System Audit 2026-06-05

## 2026-06-22 Production-hardening session

[2026-06-22 | SECTION 0/2 | include/model/sequence_model.hpp:SequenceModelIO]
ACTION: Audited all 191 tracked text files (35,990 lines) and traced the active train/infer checkpoint path.
FOUND: The active model is 9 x 127 = 1,143 summary inputs, then 256 -> 128 -> 3 (326,147 parameters); it is not the legacy 176-input / 78,595-parameter model described below. FSQ2/FSQ3 accepted arbitrary vector lengths and could silently load an incompatible model.
CHANGED: Replaced the active SequenceModel checkpoint envelope with a 32-byte FROM v1 header containing magic, version, architecture hash, parameter count, normalization presence and reserved bytes. Load now checks exact vector sizes. Added `from verify-checkpoint --checkpoint <path>`.
VERIFIED: Directly compiled cli/verify_checkpoint.cpp and cli/main.cpp with MSVC x64 /std:c++20; both succeeded. A full build exceeded the available 120-second command window without returning diagnostics.
NEXT: Validate the data loader timestamp contract and replace the remaining hardcoded cost model.

[2026-06-22 | SECTION 2 | src/data/parquet_reader.cpp:ParquetReader]
ACTION: Traced the tick-data boundary used by train, infer, walk-forward, and backtest.
FOUND: The reader previously trusted positional columns and accepted any INT64 timestamp, including seconds, wrapped clocks, and out-of-order records.
CHANGED: Enforced the six expected physical column types and added load-time UTC-millisecond range, row-count, and monotonicity checks across chunk boundaries.
VERIFIED: Direct MSVC x64 compilation of src/data/parquet_reader.cpp succeeded.
NEXT: Replace return-unit hardcoded commission/slippage defaults with named three-scenario execution costs.

[2026-06-22 | SECTION 2 | cli/backtest.cpp:run_backtest; cli/cost_analysis.cpp:run_cost_analysis]
ACTION: Replaced the backtest's hardcoded return-unit commission/slippage model.
FOUND: The prior model mixed an observed spread return with hardcoded execution constants, so it could not express broker USD costs or the requested scenarios.
CHANGED: Added costs configuration, converted commission and slippage from USD to return units using each trade's entry price, added fixed-spread override support, and added `cost-analysis` for optimistic/realistic/conservative runs. Realistic negative net edge emits a prominent non-profitable warning.
VERIFIED: Direct MSVC x64 compilation of cli/backtest.cpp, cli/cost_analysis.cpp, and cli/main.cpp succeeded.
NEXT: Reconfigure and complete a full executable build, then run CLI help and checkpoint verification against a newly generated fixture.

[2026-06-22 | SECTION 2 | tools/checkpoint_smoke.cpp]
ACTION: Added a focused checkpoint serialization smoke test.
FOUND: The architecture resolves to 326,147 parameters from the current 1,143-input source definition.
CHANGED: Added a standalone save-and-inspect test that removes its temporary checkpoint after validation.
VERIFIED: Built and ran the test with the x64 Visual Studio toolchain; output: `checkpoint smoke passed: 326147 parameters`.
NEXT: Continue the executable-path audit; full CMake builds currently exceed the command execution window and leave child compiler processes, so use targeted checks while investigating the build stall.

[2026-06-22 | SECTION 2 | cli/infer.cpp:run_infer; cli/train.cpp:run_train]
ACTION: Compared inference defaults with the active training configuration.
FOUND: Inference hardcoded horizon=128 and confidence=0.65, while training defaulted to horizon=256 and used a separate hardcoded 0.45 validation gate.
CHANGED: Made inference and training validation read window, stride, horizon, direction threshold, and confidence gate from config.toml with explicit CLI overrides.
VERIFIED: Direct MSVC x64 compilation of cli/infer.cpp and cli/train.cpp succeeded.
NEXT: Add an executable feature-parity test that compares train and inference preprocessing on the same synthetic tick stream.

[2026-06-22 | SECTION 5 | cli/wf_metrics.hpp:daily_sharpe; cli/backtest.cpp:run_backtest]
ACTION: Audited the reported Sharpe metric against the required UTC-calendar-day definition.
FOUND: Backtest and walk-forward output only per-trade Sharpe, which is not annualized and excludes no-trade days.
CHANGED: Added daily_sharpe that includes all UTC calendar days between first and last trade, treats no-trade days as zero P&L, returns zero for zero variance, and prints it in backtest output. Added a 252-day synthetic test with expected Sharpe 0.1*sqrt(252).
VERIFIED: Direct MSVC x64 compilation of cli/backtest.cpp and cli/test.cpp succeeded.
VERIFIED: tools/daily_sharpe_smoke.cpp built and ran; output `daily Sharpe smoke passed: 1.58745`.
NEXT: Continue feature-parity and adversarial-validation work; full executable integration remains pending.

[2026-06-22 | SECTION 2 | include/model/sequence_model.hpp:SequenceModelIO::CheckpointHeader]
ACTION: Checked the binary checkpoint header layout against the required wire format.
CHANGED: Added a compile-time 32-byte layout assertion for the FROM v1 header.
VERIFIED: Rebuilt and ran the checkpoint smoke test; it passed with 326147 parameters.
NEXT: Continue requirement-by-requirement validation.

## CRITICAL BUGS

### BUG 1: Second-pass normalization stats never saved (DEPLOYMENT BLOCKER)
- Location: `cli/train.cpp:224-258` computes `feat_mean`/`feat_std` over 200K samples
- Applied to all data in-memory, but vectors are local to `run_train` scope
- `SequenceModelIO::save` at `include/model/sequence_model.hpp:583-612` does NOT write them
- `cli/infer.cpp` never computes or applies second-pass normalization
- Result: model trained on z-scored 176-dim input; inference feeds raw summarizer output
- Impact: model receives wrong-scale inputs at inference → predictions meaningless

### BUG 2: Backtest PnL in synthetic units, not pips
- Location: `cli/backtest.cpp:159`
- `y_mag = abs(delta) / (mean_spread + eps)` — dimensionless spread-multiple
- `spread_cost = 0.30` subtracted, but 0.30 of what? Not pips.
- `Sample` has no `entry_mid`/`exit_mid` — can't compute real price delta
- Sharpe, drawdown, total PnL numbers are meaningless in absolute terms

### BUG 3: Kyle-Hasbrouck feature hardcoded to 0.0f
- Location: `src/data/tick_processor.cpp:169`
- `row[FROM_FEAT_KYLE_HASBROUCK] = 0.0f;` — dead feature
- Model wastes capacity learning that position 21 is always zero
- Comment says "simplified - expensive, skip for speed" but never implemented

## ARCHITECTURAL WEAKNESSES

### WEAK 1: MultiScaleSummarizer scale 7 = max (weak statistic)
- Location: `include/model/sequence_model.hpp:118`
- `out[7 * D + d] = mx[d]` — max over entire window
- Dominated by outliers, no recency bias, conflates timing
- Should be: slope (linear regression) for directional trend

### WEAK 2: Only 8 scales, missing short-term/long-term ratio
- No feature captures "is recent behavior above/below window baseline"
- Simple mean_16/mean_all ratio would be a strong regime indicator

### WEAK 3: GPU trainer hardcodes IN=176
- Location: `include/cuda/gpu_trainer.hpp:57` — `static constexpr int IN = 176`
- Location: `src/cuda/gpu_trainer_kernels.cu:245` — `#define FUSED_IN 176`
- Fused kernel loop `for (int k = 0; k < FUSED_IN; k += 4)` assumes divisibility by 4
- Any dimension change requires manual update in multiple places

### WEAK 4: No profit factor or Kelly in validation
- Only reports: dir_acc, conf_winrate, edge, prec_UP, prec_DN
- Missing profit factor (gross_profit/gross_loss) and Kelly fraction
- Can't assess live viability from current metrics

### WEAK 5: Inference skips low-confidence windows silently
- `cli/infer.cpp:119` — only writes signal when confidence >= threshold
- Downstream consumer can't distinguish "flat signal" from "missed write"

### WEAK 6: No label smoothing
- `include/data/windower.hpp:84-88` — hard one-hot labels
- Borderline samples (delta barely exceeds threshold) get same confidence as clear signals
- Model becomes overconfident on uncertain samples

## DIMENSION CHAIN (current → proposed)
- FROM_MAX_FEATURES: 22 → 27 (5 new features)
- SEQ_NUM_SCALES: 8 → 9 (add slope, replace max)
- SEQ_SUMMARY_DIM: 176 → 243 (9 × 27)
- W1 matrix: [256, 176] → [256, 243]
- Note: 243 % 4 = 3, fused kernel needs tail handling
