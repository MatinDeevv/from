#ifndef FROM_COMMON_H
#define FROM_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FROM_EPS_F 1.0e-8f
#define FROM_EPS_D 1.0e-12
#define FROM_MAX_FEATURES 22
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
    FROM_FEAT_KYLE_HASBROUCK = 21
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

