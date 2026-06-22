#ifndef FROM_COMMON_H
#define FROM_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FROM_EPS_F 1.0e-8f
#define FROM_EPS_D 1.0e-12
#define FROM_MAX_FEATURES 127
#define FROM_DEFAULT_WINDOW 512
#define FROM_DEFAULT_HORIZON 128

typedef enum FromFeatureIndex {
    FROM_FEAT_ASK = 0,
    FROM_FEAT_BID = 1,
    FROM_FEAT_MID = 2,
    FROM_FEAT_ASK_VOL = 3,
    FROM_FEAT_BID_VOL = 4,
    FROM_FEAT_SPREAD = 5,
    FROM_FEAT_NORM_SPREAD = 6,
    FROM_FEAT_OFI = 7,
    FROM_FEAT_D_OFI = 8,
    FROM_FEAT_MID_VELOCITY = 9,
    FROM_FEAT_MID_ACCEL = 10,
    FROM_FEAT_MICRO_DEV = 11,
    FROM_FEAT_TICK_RATE = 12,
    FROM_FEAT_LOG_TICK_RATE = 13,
    FROM_FEAT_RV_16 = 14,
    FROM_FEAT_RV_64 = 15,
    FROM_FEAT_RV_256 = 16,
    FROM_FEAT_ROLL_SPREAD = 17,
    FROM_FEAT_LEE_READY_SUM_32 = 18,
    FROM_FEAT_AMIHUD = 19,
    FROM_FEAT_BID_ASK_BOUNCE = 20,
    FROM_FEAT_KYLE_HASBROUCK = 21,
    FROM_FEAT_TRADE_IMBALANCE = 22,
    FROM_FEAT_VOL_REGIME = 23,
    FROM_FEAT_SPREAD_COMPRESSION = 24,
    FROM_FEAT_AUTOCORR_LAG1 = 25,
    FROM_FEAT_VOL_CLOCK_DEV = 26,

    /* ---- 100-feature expansion (all causal: computed from ticks <= i) ----
       Multi-window microstructure + price-action features. Every column is
       derived from a backward scan over rolling history; no lookahead. */

    /* Return momentum: sum of log-returns over N ticks */
    FROM_FEAT_MOM_8    = 27,
    FROM_FEAT_MOM_16   = 28,
    FROM_FEAT_MOM_32   = 29,
    FROM_FEAT_MOM_64   = 30,
    FROM_FEAT_MOM_128  = 31,
    FROM_FEAT_MOM_256  = 32,
    FROM_FEAT_MOM_512  = 33,
    FROM_FEAT_MOM_1024 = 34,

    /* Realized vol (sqrt sum r^2) at scales not already covered by 16/64/256 */
    FROM_FEAT_RV_8     = 35,
    FROM_FEAT_RV_32    = 36,
    FROM_FEAT_RV_128   = 37,
    FROM_FEAT_RV_512   = 38,
    FROM_FEAT_RV_1024  = 39,

    /* Mean absolute return (mean-abs vol proxy) */
    FROM_FEAT_ABSRET_8   = 40,
    FROM_FEAT_ABSRET_16  = 41,
    FROM_FEAT_ABSRET_32  = 42,
    FROM_FEAT_ABSRET_64  = 43,
    FROM_FEAT_ABSRET_128 = 44,

    /* Return std */
    FROM_FEAT_RSTD_64  = 45,
    FROM_FEAT_RSTD_256 = 46,

    /* Higher moments of returns */
    FROM_FEAT_SKEW_64  = 47,
    FROM_FEAT_SKEW_256 = 48,
    FROM_FEAT_KURT_64  = 49,
    FROM_FEAT_KURT_256 = 50,

    /* Mean-return / std (information-ratio style) */
    FROM_FEAT_SHARPE_16  = 51,
    FROM_FEAT_SHARPE_64  = 52,
    FROM_FEAT_SHARPE_256 = 53,

    /* EMA-of-mid relative deviation (spans 16/64/256/1024) */
    FROM_FEAT_EMADEV_16   = 54,
    FROM_FEAT_EMADEV_64   = 55,
    FROM_FEAT_EMADEV_256  = 56,
    FROM_FEAT_EMADEV_1024 = 57,

    /* EMA crossovers (fast-slow)/mid */
    FROM_FEAT_EMAX_16_64    = 58,
    FROM_FEAT_EMAX_64_256   = 59,
    FROM_FEAT_EMAX_256_1024 = 60,

    /* Wilder RSI (periods 14/32/64), scaled to [0,1] */
    FROM_FEAT_RSI_14 = 61,
    FROM_FEAT_RSI_32 = 62,
    FROM_FEAT_RSI_64 = 63,

    /* Donchian channel position (mid-min)/(max-min) */
    FROM_FEAT_DONCH_32  = 64,
    FROM_FEAT_DONCH_128 = 65,
    FROM_FEAT_DONCH_512 = 66,

    /* Mid z-score over window */
    FROM_FEAT_MIDZ_64   = 67,
    FROM_FEAT_MIDZ_256  = 68,
    FROM_FEAT_MIDZ_1024 = 69,

    /* OFI rolling mean / std */
    FROM_FEAT_OFIM_8    = 70,
    FROM_FEAT_OFIM_32   = 71,
    FROM_FEAT_OFIM_128  = 72,
    FROM_FEAT_OFISD_32  = 73,
    FROM_FEAT_OFISD_128 = 74,

    /* Signed-volume cumulative imbalance (VPIN-style) */
    FROM_FEAT_SVOLSUM_32  = 75,
    FROM_FEAT_SVOLSUM_128 = 76,

    /* Volume rolling mean / std */
    FROM_FEAT_VOLM_32   = 77,
    FROM_FEAT_VOLM_128  = 78,
    FROM_FEAT_VOLSD_32  = 79,
    FROM_FEAT_VOLSD_128 = 80,

    /* Spread rolling mean / std / z-score (64) */
    FROM_FEAT_SPRM_64  = 81,
    FROM_FEAT_SPRSD_64 = 82,
    FROM_FEAT_SPRZ_64  = 83,

    /* Velocity rolling mean / std */
    FROM_FEAT_VELM_32  = 84,
    FROM_FEAT_VELM_128 = 85,
    FROM_FEAT_VELSD_32 = 86,
    FROM_FEAT_VELSD_128 = 87,

    /* Acceleration rolling mean (32) */
    FROM_FEAT_ACCM_32 = 88,

    /* Tick-direction persistence */
    FROM_FEAT_DIRRUN    = 89,  /* signed current run length */
    FROM_FEAT_FRACUP_64 = 90,  /* fraction of up-ticks over 64 */

    /* Return autocorrelation at higher lags (64-tick) */
    FROM_FEAT_AC2_64 = 91,
    FROM_FEAT_AC5_64 = 92,

    /* Variance ratio (RV_8^2/8) / (RV_64^2/64) — mean-reversion vs trend */
    FROM_FEAT_VARRATIO = 93,

    /* Smoothed liquidity */
    FROM_FEAT_AMIHUDM_64 = 94,
    FROM_FEAT_KYLEM_64   = 95,

    /* Microprice deviation rolling mean / std (32) */
    FROM_FEAT_MDEVM_32  = 96,
    FROM_FEAT_MDEVSD_32 = 97,

    /* Jump fraction (RV vs bipower variation, 64) */
    FROM_FEAT_JUMP_64 = 98,

    /* Semivariance up / down (64) */
    FROM_FEAT_SEMIUP_64 = 99,
    FROM_FEAT_SEMIDN_64 = 100,

    /* Time-of-day cyclical + session flags */
    FROM_FEAT_TOD_SIN_H = 101,
    FROM_FEAT_TOD_COS_H = 102,
    FROM_FEAT_TOD_SIN_M = 103,
    FROM_FEAT_TOD_COS_M = 104,
    FROM_FEAT_SESS_ASIA   = 105,
    FROM_FEAT_SESS_LONDON = 106,
    FROM_FEAT_SESS_NY     = 107,

    /* Spread momentum (spread - rolling mean) */
    FROM_FEAT_DSPR_8  = 108,
    FROM_FEAT_DSPR_32 = 109,

    /* Sign entropy of returns (64) */
    FROM_FEAT_SIGNENT_64 = 110,

    /* Max drawdown over 128 */
    FROM_FEAT_MDD_128 = 111,

    /* Distance from rolling VWAP */
    FROM_FEAT_VWAPD_64  = 112,
    FROM_FEAT_VWAPD_256 = 113,

    /* Rolling correlations (64) */
    FROM_FEAT_CORR_RO_64 = 114,  /* return vs OFI */
    FROM_FEAT_CORR_RV_64 = 115,  /* return vs volume */

    /* Acceleration sign + jerk */
    FROM_FEAT_ACCSIGN = 116,
    FROM_FEAT_JERK    = 117,

    /* RV ratios across scales */
    FROM_FEAT_RVR_8_64   = 118,
    FROM_FEAT_RVR_64_512 = 119,

    /* Lee-Ready signed-tick sums at extra windows */
    FROM_FEAT_LR_64  = 120,
    FROM_FEAT_LR_128 = 121,

    /* Trade imbalance at extra windows */
    FROM_FEAT_TIR_64  = 122,
    FROM_FEAT_TIR_128 = 123,

    /* Tick-rate rolling mean (64) */
    FROM_FEAT_TRM_64 = 124,

    /* Bounce frequency (64) */
    FROM_FEAT_BOUNCEF_64 = 125,

    /* Normalized mid range (high-low)/mid over 64 */
    FROM_FEAT_NRANGE_64 = 126
} FromFeatureIndex;

typedef enum FromDType {
    FROM_DTYPE_F32 = 0,
    FROM_DTYPE_F64 = 1,
    FROM_DTYPE_I32 = 2
} FromDType;

typedef struct FromHeader {
    char     magic[4];
    uint32_t version;
    uint64_t num_layers;
    uint64_t total_params;
    uint64_t creation_time;
    uint32_t arch_hash;
    char     description[196];
} FromHeader;

typedef struct LayerBlock {
    char     layer_name[64];
    uint64_t num_tensors;
} LayerBlock;

typedef struct TensorBlock {
    char     tensor_name[32];
    uint32_t ndim;
    uint64_t shape[8];
    uint64_t num_elements;
    uint32_t dtype;
} TensorBlock;

float from_kahan_sum_f32(const float* arr, size_t n);
double from_kahan_sum_f64(const double* arr, size_t n);

void from_gemm_tile_f32(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N,
    size_t lda, size_t ldb, size_t ldc);

uint64_t from_crc64(const uint8_t* data, size_t n);

int from_snappy_uncompress(
    const uint8_t* input, size_t input_len,
    uint8_t* output, size_t* output_len);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <stdexcept>
#include <string>

namespace from {

inline void require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

template <class T>
inline T clamp_value(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

}  // namespace from
#endif

#endif

