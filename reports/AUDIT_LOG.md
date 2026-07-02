# AUDIT LOG — Full System Audit 2026-06-05

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
