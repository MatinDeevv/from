# 100-Feature Expansion — Work Log & TODO

Status: **code written, NOT yet compile-verified**. Date started: 2026-06-21.

Goal: add 100 small, causal, per-tick features to the XAUUSD direction engine.
27 → 127 features. MLP summary auto-scales 9×27=243 → 9×127=**1143** dims.

---

## What changed (files)

1. **`include/common.h`**
   - `FROM_MAX_FEATURES` 27 → **127**.
   - Added enum entries `FROM_FEAT_*` indices **27..126** (100 new), grouped by family.

2. **`include/model/sequence_model.hpp`** (AVX summarizer)
   - Fixed landmine: accumulators were fixed `[32]` → overflow when D>32.
   - Now `constexpr size_t ACC = ((SEQ_IN_FEATURES+7)/8)*8 + 8;` (covers D=127 + AVX 8-wide tail slack). 7 arrays resized: `sum_all/sum_sq_all/sum_64/sum_sq_64/sum_16/sum_sq_16/sum_tx`.

3. **`include/data/tick_processor.hpp`**
   - Added `struct HistRow` + `std::deque<HistRow> hist_` ring (`HIST_CAP=1024`).
   - Persistent state across chunks: `ema_mid_[4]`, `rsi_ag_[3]/rsi_al_[3]`, `prev_accel_`, `dir_run_`, init flags.

4. **`src/data/tick_processor.cpp`**
   - New scoped block before the FINITE GUARD, in the sequential loop.
   - Computes all 100 features: one backward scan over `hist_` (≤1024) snapshotting at windows {8,16,32,64,128,256,512,1024}; plus short 64-tick lag loop (autocorr/bipower/corr/semivar) and 128-tick drawdown loop. EMA/RSI/run-length use persistent state.
   - All causal (ticks ≤ i). Existing FINITE GUARD clamps any NaN/Inf to 0.

Everything downstream auto-scales off `FROM_MAX_FEATURES` / `SEQ_IN_FEATURES`
(windower array, normalizer, summarizer, `SEQ_SUMMARY_DIM`, GPU trainer `IN`).
No other hardcoded `27`/`243` found in functional code.

---

## Feature catalog (indices 27..126)

- **27-34** MOM_{8,16,32,64,128,256,512,1024} — sum log-returns
- **35-39** RV_{8,32,128,512,1024} — sqrt sum r²
- **40-44** ABSRET_{8,16,32,64,128} — mean |return|
- **45-46** RSTD_{64,256} — return std
- **47-50** SKEW/KURT_{64,256}
- **51-53** SHARPE_{16,64,256} — mean/std
- **54-57** EMADEV_{16,64,256,1024} — (mid-EMA)/mid
- **58-60** EMAX_{16_64,64_256,256_1024} — EMA crossovers
- **61-63** RSI_{14,32,64} (Wilder, [0,1])
- **64-66** DONCH_{32,128,512} — channel position
- **67-69** MIDZ_{64,256,1024} — mid z-score
- **70-74** OFIM_{8,32,128}, OFISD_{32,128}
- **75-76** SVOLSUM_{32,128} — VPIN-style signed-vol cumsum
- **77-80** VOLM/VOLSD_{32,128}
- **81-83** SPRM_64, SPRSD_64, SPRZ_64
- **84-87** VELM/VELSD_{32,128}
- **88** ACCM_32
- **89-90** DIRRUN (signed run len), FRACUP_64
- **91-92** AC2_64, AC5_64 — autocorr lag 2/5
- **93** VARRATIO — (RV²_8/8)/(RV²_64/64)
- **94-95** AMIHUDM_64, KYLEM_64 — smoothed liquidity
- **96-97** MDEVM_32, MDEVSD_32 — microprice dev
- **98** JUMP_64 — RV vs bipower
- **99-100** SEMIUP_64, SEMIDN_64 — semivariance
- **101-107** TOD_SIN_H/COS_H/SIN_M/COS_M, SESS_ASIA/LONDON/NY
- **108-109** DSPR_{8,32} — spread momentum
- **110** SIGNENT_64 — sign entropy
- **111** MDD_128 — max drawdown
- **112-113** VWAPD_{64,256} — dist from rolling VWAP
- **114-115** CORR_RO_64 (ret·ofi), CORR_RV_64 (ret·vol)
- **116-117** ACCSIGN, JERK
- **118-119** RVR_8_64, RVR_64_512 — RV ratios
- **120-121** LR_{64,128} — Lee-Ready signed-tick sum
- **122-123** TIR_{64,128} — trade imbalance
- **124** TRM_64 — mean tick rate
- **125** BOUNCEF_64 — bounce frequency
- **126** NRANGE_64 — normalized high-low range

---

## TODO

- [ ] **Compile-verify.** Blocked: bash shell can't get MSVC CRT headers
      (`vcvars64.bat` sets INCLUDE but `cl` still hits `C1083: stddef.h`
      — vswhere.exe missing → vsdevcmd partial init). Build via the repo's
      own scripts instead: `BUILD_FULL_MODEL.cmd` / `build_cuda.cmd` from a
      proper Developer Command Prompt, OR fix vswhere on PATH.
- [ ] Fix any compile errors surfaced (review done, but unverified).
- [ ] **Retrain required** — old checkpoints are 27-feat, incompatible with 127.
      Arch/dims changed; serializer arch_hash will differ.
- [ ] Sanity: dump a feature row, confirm all 127 finite, no constant/degenerate
      columns (e.g. early-history windows before warmup).
- [ ] Perf check: backward scan is ~1024 iters/tick. Preprocessing is one-time
      (cached downstream) but confirm acceptable on full dataset; if slow,
      convert big-window sums (512/1024) to incremental rollers.
- [ ] Walk-forward eval: compare 127-feat vs 27-feat baseline (Sharpe/winrate)
      to confirm the new features actually add edge, prune dead ones.
- [ ] Optional: feature-importance pass; drop any of the 100 that don't help.

---

## Notes / risks

- `prev_spread_` member declared but unused (DSPR uses rolling mean). Harmless; remove if lint complains.
- Session flags overlap by design (London/NY overlap window) — not one-hot.
- MLP first layer grows 256×243 → 256×1143 params; verify GPU mem OK at batch size.
