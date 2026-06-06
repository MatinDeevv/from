#pragma once

#include <cstddef>
#include <cstring>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace from {

// AVX2 dot product of two float vectors
inline float dot_avx2(const float* a, const float* b, size_t n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    // Tail
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

// Matrix-vector: out[j] = bias[j] + dot(W[j*K..], x, K), for j in [0, M)
inline void gemv_avx2(const float* W, const float* x, const float* bias,
                      float* out, size_t M, size_t K) {
    for (size_t j = 0; j < M; ++j) {
        out[j] = bias[j] + dot_avx2(W + j * K, x, K);
    }
}

// Batch GEMM: C[n,j] = bias[j] + sum_k A[n,k] * B[j,k]  (B is row-major [M x K])
// A is [N x K], B is [M x K], C is [N x M]
inline void gemm_avx2(const float* A, const float* B, const float* bias,
                      float* C, size_t N, size_t M, size_t K) {
    #ifdef _OPENMP
    #pragma omp parallel for if(N > 64)
    #endif
    for (int n = 0; n < static_cast<int>(N); ++n) {
        const float* a_row = A + n * K;
        float* c_row = C + n * M;
        for (size_t j = 0; j < M; ++j) {
            c_row[j] = bias[j] + dot_avx2(B + j * K, a_row, K);
        }
    }
}

// GEMM with ReLU fused: C[n,j] = max(0, bias[j] + dot(B[j], A[n]))
// Also writes relu mask (mask can be nullptr for inference)
inline void gemm_relu_avx2(const float* A, const float* B, const float* bias,
                           float* C, char* mask, size_t N, size_t M, size_t K) {
    #ifdef _OPENMP
    #pragma omp parallel for if(N > 64)
    #endif
    for (int n = 0; n < static_cast<int>(N); ++n) {
        const float* a_row = A + n * K;
        float* c_row = C + n * M;
        for (size_t j = 0; j < M; ++j) {
            float val = bias[j] + dot_avx2(B + j * K, a_row, K);
            char active = val > 0.0f ? 1 : 0;
            c_row[j] = active ? val : 0.0f;
            if (mask) mask[n * M + j] = active;
        }
    }
}

// Backward GEMM: grad_input[n,k] = sum_j grad_out[n,j] * W[j,k]
// W is [M x K], grad_out is [N x M], grad_input is [N x K]
inline void gemm_backward_input_avx2(const float* grad_out, const float* W,
                                     float* grad_input, size_t N, size_t M, size_t K) {
    #ifdef _OPENMP
    #pragma omp parallel for if(N > 64)
    #endif
    for (int n = 0; n < static_cast<int>(N); ++n) {
        const float* go = grad_out + n * M;
        float* gi = grad_input + n * K;
        std::memset(gi, 0, K * sizeof(float));
        for (size_t j = 0; j < M; ++j) {
            if (go[j] == 0.0f) continue;
            float g = go[j];
            __m256 vg = _mm256_set1_ps(g);
            const float* wrow = W + j * K;
            size_t k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 vw = _mm256_loadu_ps(wrow + k);
                __m256 vgi = _mm256_loadu_ps(gi + k);
                vgi = _mm256_fmadd_ps(vg, vw, vgi);
                _mm256_storeu_ps(gi + k, vgi);
            }
            for (; k < K; ++k) gi[k] += g * wrow[k];
        }
    }
}

// Backward weight accumulation: grad_W[j,k] += sum_n grad_out[n,j] * input[n,k]
inline void gemm_backward_weight_avx2(const float* grad_out, const float* input,
                                      float* grad_W, float* grad_bias,
                                      size_t N, size_t M, size_t K) {
    for (size_t n = 0; n < N; ++n) {
        const float* go = grad_out + n * M;
        const float* x = input + n * K;
        for (size_t j = 0; j < M; ++j) {
            float g = go[j];
            if (g == 0.0f) continue;
            grad_bias[j] += g;
            __m256 vg = _mm256_set1_ps(g);
            float* gw = grad_W + j * K;
            size_t k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 vx = _mm256_loadu_ps(x + k);
                __m256 vgw = _mm256_loadu_ps(gw + k);
                vgw = _mm256_fmadd_ps(vg, vx, vgw);
                _mm256_storeu_ps(gw + k, vgw);
            }
            for (; k < K; ++k) gw[k] += g * x[k];
        }
    }
}

// AVX2 vector scale: out[i] *= scale
inline void scale_avx2(float* data, size_t n, float scale) {
    __m256 vs = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_mul_ps(v, vs);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) data[i] *= scale;
}

// AVX2 SGD update: w[i] -= lr * grad[i]
inline void sgd_update_avx2(float* w, const float* grad, size_t n, float lr) {
    __m256 vlr = _mm256_set1_ps(lr);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vg = _mm256_loadu_ps(grad + i);
        vw = _mm256_fnmadd_ps(vlr, vg, vw);
        _mm256_storeu_ps(w + i, vw);
    }
    for (; i < n; ++i) w[i] -= lr * grad[i];
}

// Gradient norm squared (AVX2)
inline float grad_norm2_avx2(const float* data, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        sum = _mm256_fmadd_ps(v, v, sum);
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < n; ++i) result += data[i] * data[i];
    return result;
}

}  // namespace from
