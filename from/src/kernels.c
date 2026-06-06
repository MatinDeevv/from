#include "common.h"

#include <math.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX__)))
#include <immintrin.h>
#endif

float from_kahan_sum_f32(const float* arr, size_t n) {
    float sum = 0.0f;
    float comp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float y = arr[i] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

double from_kahan_sum_f64(const double* arr, size_t n) {
    double sum = 0.0;
    double comp = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double y = arr[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

void from_gemm_tile_f32(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N,
    size_t lda, size_t ldb, size_t ldc) {
    const int threads = 20;
    int ii = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
#endif
    for (ii = 0; ii < (int)M; ii++) {
        size_t i = (size_t)ii;
        float* c_row = C + i * ldc;
        const float* a_row = A + i * lda;
        for (size_t k = 0; k < K; ++k) {
            float aik = a_row[k];
            const float* b_row = B + k * ldb;
            size_t j = 0;
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX__))
            __m256 vaik = _mm256_set1_ps(aik);
            for (; j + 8 <= N; j += 8) {
                __m256 vb = _mm256_loadu_ps(b_row + j);
                __m256 vc = _mm256_loadu_ps(c_row + j);
#if defined(__FMA__) || defined(_MSC_VER)
                vc = _mm256_fmadd_ps(vaik, vb, vc);
#else
                vc = _mm256_add_ps(vc, _mm256_mul_ps(vaik, vb));
#endif
                _mm256_storeu_ps(c_row + j, vc);
            }
#endif
            for (; j < N; ++j) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

uint64_t from_crc64(const uint8_t* data, size_t n) {
    uint64_t crc = UINT64_C(0xffffffffffffffff);
    const uint64_t poly = UINT64_C(0xC96C5795D7870F42);
    for (size_t i = 0; i < n; ++i) {
        crc ^= (uint64_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint64_t mask = (uint64_t)0 - (crc & 1U);
            crc = (crc >> 1) ^ (poly & mask);
        }
    }
    return ~crc;
}

static int read_varint32(const uint8_t* input, size_t input_len, size_t* pos, uint32_t* out) {
    uint32_t result = 0;
    uint32_t shift = 0;
    while (*pos < input_len && shift <= 28) {
        uint8_t byte = input[(*pos)++];
        result |= (uint32_t)(byte & 0x7fU) << shift;
        if ((byte & 0x80U) == 0) {
            *out = result;
            return 1;
        }
        shift += 7;
    }
    return 0;
}

static int copy_from_output(uint8_t* output, size_t out_pos, size_t offset, size_t len, size_t out_cap) {
    if (offset == 0 || offset > out_pos || out_pos + len > out_cap) {
        return 0;
    }
    for (size_t i = 0; i < len; ++i) {
        output[out_pos + i] = output[out_pos - offset + i];
    }
    return 1;
}

int from_snappy_uncompress(
    const uint8_t* input, size_t input_len,
    uint8_t* output, size_t* output_len) {
    if (!input || !output || !output_len) {
        return 0;
    }
    size_t pos = 0;
    uint32_t expected = 0;
    if (!read_varint32(input, input_len, &pos, &expected)) {
        return 0;
    }
    if (*output_len < expected) {
        *output_len = expected;
        return 0;
    }
    size_t out_pos = 0;
    while (pos < input_len && out_pos < expected) {
        uint8_t tag = input[pos++];
        uint8_t type = tag & 0x03U;
        if (type == 0) {
            size_t len = (size_t)(tag >> 2) + 1U;
            if (len >= 61U) {
                size_t extra = len - 60U;
                if (pos + extra > input_len) {
                    return 0;
                }
                len = 1U;
                for (size_t i = 0; i < extra; ++i) {
                    len += (size_t)input[pos++] << (8U * i);
                }
            }
            if (pos + len > input_len || out_pos + len > expected) {
                return 0;
            }
            memcpy(output + out_pos, input + pos, len);
            pos += len;
            out_pos += len;
        } else if (type == 1) {
            if (pos >= input_len) {
                return 0;
            }
            size_t len = 4U + ((size_t)(tag >> 2) & 0x7U);
            size_t offset = ((size_t)(tag & 0xE0U) << 3U) | input[pos++];
            if (!copy_from_output(output, out_pos, offset, len, expected)) {
                return 0;
            }
            out_pos += len;
        } else if (type == 2) {
            if (pos + 2U > input_len) {
                return 0;
            }
            size_t len = 1U + (size_t)(tag >> 2);
            size_t offset = (size_t)input[pos] | ((size_t)input[pos + 1U] << 8U);
            pos += 2U;
            if (!copy_from_output(output, out_pos, offset, len, expected)) {
                return 0;
            }
            out_pos += len;
        } else {
            if (pos + 4U > input_len) {
                return 0;
            }
            size_t len = 1U + (size_t)(tag >> 2);
            size_t offset = (size_t)input[pos] |
                            ((size_t)input[pos + 1U] << 8U) |
                            ((size_t)input[pos + 2U] << 16U) |
                            ((size_t)input[pos + 3U] << 24U);
            pos += 4U;
            if (!copy_from_output(output, out_pos, offset, len, expected)) {
                return 0;
            }
            out_pos += len;
        }
    }
    *output_len = out_pos;
    return out_pos == expected;
}

