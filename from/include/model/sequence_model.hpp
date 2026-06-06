#pragma once

#include "common.h"
#include "data/windower.hpp"
#include "model/fast_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace from {

// ============================================================
// Multi-Scale MLP: Fast CPU model that captures temporal patterns
// Summary: 8 statistics × 22 features = 176 inputs
// Network: 176 → 256 → 128 → 3 (~79K params)
// Speed target: 500+ steps/sec on CPU with batch=32
// ============================================================

static constexpr size_t SEQ_IN_FEATURES = FROM_MAX_FEATURES;  // 22
static constexpr size_t SEQ_WINDOW = 512;
static constexpr size_t SEQ_NUM_CLASSES = 3;
static constexpr size_t SEQ_NUM_SCALES = 9;
static constexpr size_t SEQ_SUMMARY_DIM = SEQ_NUM_SCALES * SEQ_IN_FEATURES;  // 243
static constexpr size_t SEQ_HIDDEN1 = 256;
static constexpr size_t SEQ_HIDDEN2 = 128;

// Single-pass cache-friendly summarizer with AVX2
struct MultiScaleSummarizer {
    static void summarize(const Sample& s, float* out) {
        const size_t seq = s.X.shape()[0];
        const size_t D = SEQ_IN_FEATURES;
        const float* data = s.X.data_ptr();

        // Accumulators (pad to 32 for AVX alignment)
        alignas(32) float sum_all[32] = {};
        alignas(32) float sum_sq_all[32] = {};
        alignas(32) float sum_64[32] = {};
        alignas(32) float sum_sq_64[32] = {};
        alignas(32) float sum_16[32] = {};
        alignas(32) float sum_sq_16[32] = {};
        alignas(32) float sum_tx[32] = {};  // sum of t*x[d] for slope computation

        const size_t start_64 = seq > 64 ? seq - 64 : 0;
        const size_t start_16 = seq > 16 ? seq - 16 : 0;

        // Single pass over all timesteps (row-major friendly)
        for (size_t t = 0; t < seq; ++t) {
            const float* row = data + t * D;
            float tf = static_cast<float>(t);
            size_t d = 0;
            // AVX2: process 8 features at a time
            __m256 vt = _mm256_set1_ps(tf);
            for (; d + 7 < D; d += 8) {
                __m256 v = _mm256_loadu_ps(row + d);
                __m256 sa = _mm256_loadu_ps(sum_all + d);
                __m256 sqa = _mm256_loadu_ps(sum_sq_all + d);
                __m256 stx = _mm256_loadu_ps(sum_tx + d);
                _mm256_storeu_ps(sum_all + d, _mm256_add_ps(sa, v));
                _mm256_storeu_ps(sum_sq_all + d, _mm256_fmadd_ps(v, v, sqa));
                _mm256_storeu_ps(sum_tx + d, _mm256_fmadd_ps(vt, v, stx));
                if (t >= start_64) {
                    __m256 s64 = _mm256_loadu_ps(sum_64 + d);
                    __m256 sq64 = _mm256_loadu_ps(sum_sq_64 + d);
                    _mm256_storeu_ps(sum_64 + d, _mm256_add_ps(s64, v));
                    _mm256_storeu_ps(sum_sq_64 + d, _mm256_fmadd_ps(v, v, sq64));
                }
                if (t >= start_16) {
                    __m256 s16 = _mm256_loadu_ps(sum_16 + d);
                    __m256 sq16 = _mm256_loadu_ps(sum_sq_16 + d);
                    _mm256_storeu_ps(sum_16 + d, _mm256_add_ps(s16, v));
                    _mm256_storeu_ps(sum_sq_16 + d, _mm256_fmadd_ps(v, v, sq16));
                }
            }
            for (; d < D; ++d) {
                float v = row[d];
                sum_all[d] += v;
                sum_sq_all[d] += v * v;
                sum_tx[d] += tf * v;
                if (t >= start_64) { sum_64[d] += v; sum_sq_64[d] += v * v; }
                if (t >= start_16) { sum_16[d] += v; sum_sq_16[d] += v * v; }
            }
        }

        // Precompute slope denominator: n*sum_t2 - sum_t^2
        float fseq = static_cast<float>(seq);
        float sum_t = fseq * (fseq - 1.0f) * 0.5f;
        float sum_t2 = fseq * (fseq - 1.0f) * (2.0f * fseq - 1.0f) / 6.0f;
        float slope_denom = fseq * sum_t2 - sum_t * sum_t;

        // Compute means and stds from sums
        float inv_all = 1.0f / fseq;
        float inv_64 = 1.0f / static_cast<float>(seq - start_64);
        float inv_16 = 1.0f / static_cast<float>(seq - start_16);
        const float* last_row = data + (seq - 1) * D;

        for (size_t d = 0; d < D; ++d) {
            float mean_a = sum_all[d] * inv_all;
            float var_a = sum_sq_all[d] * inv_all - mean_a * mean_a;
            float mean_6 = sum_64[d] * inv_64;
            float var_6 = sum_sq_64[d] * inv_64 - mean_6 * mean_6;
            float mean_1 = sum_16[d] * inv_16;
            float var_1 = sum_sq_16[d] * inv_16 - mean_1 * mean_1;

            out[0 * D + d] = mean_a;
            out[1 * D + d] = std::sqrt(std::max(var_a, 0.0f) + 1e-8f);
            out[2 * D + d] = mean_6;
            out[3 * D + d] = std::sqrt(std::max(var_6, 0.0f) + 1e-8f);
            out[4 * D + d] = mean_1;
            out[5 * D + d] = std::sqrt(std::max(var_1, 0.0f) + 1e-8f);
            out[6 * D + d] = last_row[d];

            // Scale 7: linear regression slope (replaces max)
            float slope = (slope_denom > 1e-8f)
                ? (fseq * sum_tx[d] - sum_t * sum_all[d]) / slope_denom
                : 0.0f;
            out[7 * D + d] = slope;

            // Scale 8: short-term/long-term ratio (mean_16 / mean_all)
            float ratio = (std::abs(mean_a) > 1e-6f) ? mean_1 / mean_a : 0.0f;
            ratio = std::max(-3.0f, std::min(3.0f, ratio));
            out[8 * D + d] = ratio;
        }
    }
};

// Simple matrix helper
struct Mat {
    std::vector<float> data;
    size_t rows = 0, cols = 0;

    Mat() = default;
    Mat(size_t r, size_t c) : data(r * c, 0.0f), rows(r), cols(c) {}

    float& at(size_t i, size_t j) { return data[i * cols + j]; }
    const float& at(size_t i, size_t j) const { return data[i * cols + j]; }
    float* row(size_t i) { return data.data() + i * cols; }
    const float* row(size_t i) const { return data.data() + i * cols; }
    size_t size() const { return data.size(); }
    void zero() { std::fill(data.begin(), data.end(), 0.0f); }
};

// ============================================================
// Regime Gate — Mahalanobis OOD detection on hidden layer
// ============================================================
struct RegimeGate {
    static constexpr size_t NUM_REGIMES = 3;
    static constexpr size_t HIDDEN_DIM = SEQ_HIDDEN2;  // 128
    Mat centroids;           // [3, 128]
    Mat covariance_diag;     // [3, 128]
    std::vector<size_t> regime_counts;
    float distance_95pct = 1e10f;
    std::vector<float> training_distances;
    bool calibrated = false;

    void init() {
        centroids = Mat(NUM_REGIMES, HIDDEN_DIM);
        covariance_diag = Mat(NUM_REGIMES, HIDDEN_DIM);
        for (float& v : covariance_diag.data) v = 1.0f;
        regime_counts.resize(NUM_REGIMES, 0);
    }

    void update(const float* hidden, size_t regime) {
        if (regime >= NUM_REGIMES) return;
        regime_counts[regime]++;
        float n = static_cast<float>(regime_counts[regime]);
        for (size_t d = 0; d < HIDDEN_DIM; ++d) {
            float delta = hidden[d] - centroids.at(regime, d);
            centroids.at(regime, d) += delta / n;
            float delta2 = hidden[d] - centroids.at(regime, d);
            covariance_diag.at(regime, d) += delta * delta2;
        }
    }

    void finalize() {
        for (size_t r = 0; r < NUM_REGIMES; ++r) {
            if (regime_counts[r] > 1) {
                for (size_t d = 0; d < HIDDEN_DIM; ++d) {
                    covariance_diag.at(r, d) /= static_cast<float>(regime_counts[r] - 1);
                    if (covariance_diag.at(r, d) < 1e-6f) covariance_diag.at(r, d) = 1e-6f;
                }
            }
        }
        if (!training_distances.empty()) {
            std::sort(training_distances.begin(), training_distances.end());
            size_t idx = static_cast<size_t>(0.95 * static_cast<double>(training_distances.size()));
            distance_95pct = training_distances[std::min(idx, training_distances.size() - 1)];
        }
        calibrated = true;
    }

    float mahalanobis(const float* hidden, size_t regime) const {
        float dist = 0.0f;
        for (size_t d = 0; d < HIDDEN_DIM; ++d) {
            float diff = hidden[d] - centroids.at(regime, d);
            dist += diff * diff / covariance_diag.at(regime, d);
        }
        return std::sqrt(dist);
    }

    std::tuple<size_t, float, bool> classify(const float* hidden) const {
        float min_dist = 1e30f;
        size_t best_regime = 0;
        for (size_t r = 0; r < NUM_REGIMES; ++r) {
            float d = mahalanobis(hidden, r);
            if (d < min_dist) { min_dist = d; best_regime = r; }
        }
        bool ood = calibrated && (min_dist > distance_95pct);
        return {best_regime, min_dist, ood};
    }
};

// ============================================================
// SequenceModel: Multi-Scale Summary → 3-Layer MLP
// ============================================================
class SequenceModel {
public:
    // Layer 1: Linear(176, 256) + ReLU
    std::vector<float> w1;  // [256 × 176]
    std::vector<float> b1;  // [256]
    // Layer 2: Linear(256, 128) + ReLU
    std::vector<float> w2;  // [128 × 256]
    std::vector<float> b2;  // [128]
    // Layer 3: Linear(128, 3)
    std::vector<float> w3;  // [3 × 128]
    std::vector<float> b3;  // [3]

    // Gradients
    std::vector<float> gw1, gb1, gw2, gb2, gw3, gb3;

    // Adam state (m = first moment, v = second moment)
    std::vector<float> mw1, mb1, mw2, mb2, mw3, mb3;
    std::vector<float> vw1, vb1, vw2, vb2, vw3, vb3;
    size_t adam_t = 0;

    // Cache for backward
    std::vector<float> cache_input;   // [batch × 176]
    std::vector<float> cache_h1;      // [batch × 256] after ReLU
    std::vector<float> cache_h2;      // [batch × 128] after ReLU
    std::vector<char> relu1_mask;
    std::vector<char> relu2_mask;
    std::vector<float> grad_h2_buf;   // [batch × 128] reused
    std::vector<float> grad_h1_buf;   // [batch × 256] reused

    RegimeGate regime;

    // Second-pass normalization (z-score on summary features)
    std::vector<float> feat_mean;   // [SEQ_SUMMARY_DIM]
    std::vector<float> feat_std;    // [SEQ_SUMMARY_DIM]
    bool feat_norm_ready = false;

    void apply_feat_norm(float* summary, size_t n) const {
        if (!feat_norm_ready) return;
        for (size_t i = 0; i < n; ++i) {
            float* row = summary + i * SEQ_SUMMARY_DIM;
            for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
                row[d] = (row[d] - feat_mean[d]) / feat_std[d];
                if (row[d] >  5.0f) row[d] =  5.0f;
                if (row[d] < -5.0f) row[d] = -5.0f;
            }
        }
    }

    float lr = 0.00005f;
    float last_grad_norm = 0.0f;
    size_t total_params = 0;
    uint32_t seed;

    explicit SequenceModel(float learning_rate = 0.00005f, uint32_t seed_val = 42)
        : lr(learning_rate), seed(seed_val) {
        std::mt19937 rng(seed_val);

        auto init_vec = [&](std::vector<float>& v, size_t n, float std_dev) {
            v.resize(n);
            std::normal_distribution<float> dist(0.0f, std_dev);
            for (float& x : v) x = dist(rng);
        };

        init_vec(w1, SEQ_HIDDEN1 * SEQ_SUMMARY_DIM, std::sqrt(2.0f / static_cast<float>(SEQ_SUMMARY_DIM)));
        b1.resize(SEQ_HIDDEN1, 0.0f);
        init_vec(w2, SEQ_HIDDEN2 * SEQ_HIDDEN1, std::sqrt(2.0f / static_cast<float>(SEQ_HIDDEN1)));
        b2.resize(SEQ_HIDDEN2, 0.0f);
        init_vec(w3, SEQ_NUM_CLASSES * SEQ_HIDDEN2, std::sqrt(2.0f / static_cast<float>(SEQ_HIDDEN2)));
        b3.resize(SEQ_NUM_CLASSES, 0.0f);

        gw1.resize(SEQ_HIDDEN1 * SEQ_SUMMARY_DIM, 0.0f);
        gb1.resize(SEQ_HIDDEN1, 0.0f);
        gw2.resize(SEQ_HIDDEN2 * SEQ_HIDDEN1, 0.0f);
        gb2.resize(SEQ_HIDDEN2, 0.0f);

        // Adam moments
        mw1.resize(w1.size(), 0.0f); vw1.resize(w1.size(), 0.0f);
        mb1.resize(b1.size(), 0.0f); vb1.resize(b1.size(), 0.0f);
        mw2.resize(w2.size(), 0.0f); vw2.resize(w2.size(), 0.0f);
        mb2.resize(b2.size(), 0.0f); vb2.resize(b2.size(), 0.0f);
        mw3.resize(w3.size(), 0.0f); vw3.resize(w3.size(), 0.0f);
        mb3.resize(b3.size(), 0.0f); vb3.resize(b3.size(), 0.0f);
        gw3.resize(SEQ_NUM_CLASSES * SEQ_HIDDEN2, 0.0f);
        gb3.resize(SEQ_NUM_CLASSES, 0.0f);

        total_params = SEQ_HIDDEN1 * SEQ_SUMMARY_DIM + SEQ_HIDDEN1 +
                       SEQ_HIDDEN2 * SEQ_HIDDEN1 + SEQ_HIDDEN2 +
                       SEQ_NUM_CLASSES * SEQ_HIDDEN2 + SEQ_NUM_CLASSES;

        regime.init();
    }

    // Summarize a batch — uses pre-computed summary if available
    void summarize_batch(const std::vector<Sample>& samples, size_t start, size_t n, float* out) const {
        #ifdef _OPENMP
        #pragma omp parallel for if(n > 8)
        #endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const Sample& s = samples[start + i];
            if (s.summary.size() == SEQ_SUMMARY_DIM) {
                std::memcpy(out + i * SEQ_SUMMARY_DIM, s.summary.data(), SEQ_SUMMARY_DIM * sizeof(float));
            } else {
                MultiScaleSummarizer::summarize(s, out + i * SEQ_SUMMARY_DIM);
            }
        }
    }

    // Pre-compute summary for a sample (call from data pipeline)
    static void precompute_summary(Sample& s) {
        s.summary.resize(SEQ_SUMMARY_DIM);
        MultiScaleSummarizer::summarize(s, s.summary.data());
    }

    // Forward pass: [n, 176] → [n, 3] logits (AVX2 + OpenMP)
    // Uses internal buffers (cache_h1, cache_h2) — zero allocations in steady state
    void forward(const float* input, size_t n, float* logits, bool training) {
        cache_h1.resize(n * SEQ_HIDDEN1);
        cache_h2.resize(n * SEQ_HIDDEN2);

        if (training) {
            cache_input.resize(n * SEQ_SUMMARY_DIM);
            std::memcpy(cache_input.data(), input, n * SEQ_SUMMARY_DIM * sizeof(float));
            relu1_mask.resize(n * SEQ_HIDDEN1);
            relu2_mask.resize(n * SEQ_HIDDEN2);
        }

        // Layer 1: h1 = ReLU(input @ W1^T + b1) → writes directly to cache_h1
        gemm_relu_avx2(input, w1.data(), b1.data(), cache_h1.data(),
                       training ? relu1_mask.data() : nullptr, n, SEQ_HIDDEN1, SEQ_SUMMARY_DIM);

        // Layer 2: h2 = ReLU(h1 @ W2^T + b2) → writes directly to cache_h2
        gemm_relu_avx2(cache_h1.data(), w2.data(), b2.data(), cache_h2.data(),
                       training ? relu2_mask.data() : nullptr, n, SEQ_HIDDEN2, SEQ_HIDDEN1);

        // Layer 3: logits = h2 @ W3^T + b3
        gemm_avx2(cache_h2.data(), w3.data(), b3.data(), logits, n, SEQ_NUM_CLASSES, SEQ_HIDDEN2);
    }

    static void softmax(const float* logits, size_t n, float* probs) {
        for (size_t i = 0; i < n; ++i) {
            const float* l = logits + i * SEQ_NUM_CLASSES;
            float* p = probs + i * SEQ_NUM_CLASSES;
            float m = std::max({l[0], l[1], l[2]});
            float s = 0.0f;
            for (size_t c = 0; c < SEQ_NUM_CLASSES; ++c) {
                p[c] = std::exp(l[c] - m);
                s += p[c];
            }
            for (size_t c = 0; c < SEQ_NUM_CLASSES; ++c) p[c] /= (s + FROM_EPS_F);
        }
    }

    // Backward pass: takes grad_logits [n × 3], does backprop + Adam update
    void backward(const float* grad_logits, size_t n) {
        // Backward through Layer 3 (AVX2)
        std::memset(gw3.data(), 0, gw3.size() * sizeof(float));
        std::memset(gb3.data(), 0, gb3.size() * sizeof(float));
        grad_h2_buf.resize(n * SEQ_HIDDEN2);
        std::memset(grad_h2_buf.data(), 0, n * SEQ_HIDDEN2 * sizeof(float));

        gemm_backward_weight_avx2(grad_logits, cache_h2.data(), gw3.data(), gb3.data(),
                                  n, SEQ_NUM_CLASSES, SEQ_HIDDEN2);
        gemm_backward_input_avx2(grad_logits, w3.data(), grad_h2_buf.data(),
                                 n, SEQ_NUM_CLASSES, SEQ_HIDDEN2);

        // ReLU backward layer 2
        for (size_t i = 0; i < n * SEQ_HIDDEN2; ++i) {
            if (!relu2_mask[i]) grad_h2_buf[i] = 0.0f;
        }

        // Backward through Layer 2 (AVX2)
        std::memset(gw2.data(), 0, gw2.size() * sizeof(float));
        std::memset(gb2.data(), 0, gb2.size() * sizeof(float));
        grad_h1_buf.resize(n * SEQ_HIDDEN1);
        std::memset(grad_h1_buf.data(), 0, n * SEQ_HIDDEN1 * sizeof(float));

        gemm_backward_weight_avx2(grad_h2_buf.data(), cache_h1.data(), gw2.data(), gb2.data(),
                                  n, SEQ_HIDDEN2, SEQ_HIDDEN1);
        gemm_backward_input_avx2(grad_h2_buf.data(), w2.data(), grad_h1_buf.data(),
                                 n, SEQ_HIDDEN2, SEQ_HIDDEN1);

        // ReLU backward layer 1
        for (size_t i = 0; i < n * SEQ_HIDDEN1; ++i) {
            if (!relu1_mask[i]) grad_h1_buf[i] = 0.0f;
        }

        // Backward through Layer 1 (AVX2)
        std::memset(gw1.data(), 0, gw1.size() * sizeof(float));
        std::memset(gb1.data(), 0, gb1.size() * sizeof(float));

        gemm_backward_weight_avx2(grad_h1_buf.data(), cache_input.data(), gw1.data(), gb1.data(),
                                  n, SEQ_HIDDEN1, SEQ_SUMMARY_DIM);

        // Gradient norm + clipping (AVX2)
        float norm2 = grad_norm2_avx2(gw1.data(), gw1.size())
                    + grad_norm2_avx2(gb1.data(), gb1.size())
                    + grad_norm2_avx2(gw2.data(), gw2.size())
                    + grad_norm2_avx2(gb2.data(), gb2.size())
                    + grad_norm2_avx2(gw3.data(), gw3.size())
                    + grad_norm2_avx2(gb3.data(), gb3.size());
        float grad_norm = std::sqrt(norm2);
        last_grad_norm = grad_norm;

        if (grad_norm > 1.0f) {
            float clip = 1.0f / (grad_norm + FROM_EPS_F);
            scale_avx2(gw1.data(), gw1.size(), clip);
            scale_avx2(gb1.data(), gb1.size(), clip);
            scale_avx2(gw2.data(), gw2.size(), clip);
            scale_avx2(gb2.data(), gb2.size(), clip);
            scale_avx2(gw3.data(), gw3.size(), clip);
            scale_avx2(gb3.data(), gb3.size(), clip);
        }

        // Adam update
        ++adam_t;
        adam_update(w1.data(), gw1.data(), mw1.data(), vw1.data(), w1.size());
        adam_update(b1.data(), gb1.data(), mb1.data(), vb1.data(), b1.size());
        adam_update(w2.data(), gw2.data(), mw2.data(), vw2.data(), w2.size());
        adam_update(b2.data(), gb2.data(), mb2.data(), vb2.data(), b2.size());
        adam_update(w3.data(), gw3.data(), mw3.data(), vw3.data(), w3.size());
        adam_update(b3.data(), gb3.data(), mb3.data(), vb3.data(), b3.size());
    }

    float train_batch(const std::vector<Sample>& samples, size_t start, size_t n, float* accuracy) {
        // Summarize
        std::vector<float> input(n * SEQ_SUMMARY_DIM);
        summarize_batch(samples, start, n, input.data());

        // Forward
        std::vector<float> logits(n * SEQ_NUM_CLASSES);
        forward(input.data(), n, logits.data(), true);

        // Softmax + loss
        std::vector<float> probs(n * SEQ_NUM_CLASSES);
        softmax(logits.data(), n, probs.data());

        float loss = 0.0f;
        size_t correct = 0;
        std::vector<float> grad_logits(n * SEQ_NUM_CLASSES);

        for (size_t i = 0; i < n; ++i) {
            const Sample& s = samples[start + i];
            size_t truth = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                if (s.y_dir[c] > s.y_dir[truth]) truth = c;
            }
            size_t pred = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                if (probs[i * SEQ_NUM_CLASSES + c] > probs[i * SEQ_NUM_CLASSES + pred]) pred = c;
            }
            if (pred == truth) ++correct;
            loss -= std::log(probs[i * SEQ_NUM_CLASSES + truth] + FROM_EPS_F);

            for (size_t c = 0; c < SEQ_NUM_CLASSES; ++c) {
                grad_logits[i * SEQ_NUM_CLASSES + c] = (probs[i * SEQ_NUM_CLASSES + c] - s.y_dir[c]) / static_cast<float>(n);
            }
        }

        backward(grad_logits.data(), n);

        *accuracy = static_cast<float>(correct) / static_cast<float>(n);
        return loss / static_cast<float>(n);
    }

    void adam_update(float* w, const float* g, float* m, float* v, size_t n) {
        constexpr float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(adam_t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(adam_t));
        float step_size = lr / bc1;
        float inv_bc2 = 1.0f / bc2;

        __m256 vbeta1 = _mm256_set1_ps(beta1);
        __m256 vbeta2 = _mm256_set1_ps(beta2);
        __m256 v1mb1 = _mm256_set1_ps(1.0f - beta1);
        __m256 v1mb2 = _mm256_set1_ps(1.0f - beta2);
        __m256 vstep = _mm256_set1_ps(step_size);
        __m256 vinv_bc2 = _mm256_set1_ps(inv_bc2);
        __m256 veps = _mm256_set1_ps(eps);

        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 vg = _mm256_loadu_ps(g + i);
            __m256 vm = _mm256_loadu_ps(m + i);
            __m256 vv = _mm256_loadu_ps(v + i);
            __m256 vw = _mm256_loadu_ps(w + i);

            // m = beta1*m + (1-beta1)*g
            vm = _mm256_fmadd_ps(vbeta1, vm, _mm256_mul_ps(v1mb1, vg));
            // v = beta2*v + (1-beta2)*g*g
            vv = _mm256_fmadd_ps(vbeta2, vv, _mm256_mul_ps(v1mb2, _mm256_mul_ps(vg, vg)));
            // vh = v / bc2
            __m256 vh = _mm256_mul_ps(vv, vinv_bc2);
            // w -= step_size * m / (sqrt(vh) + eps)
            __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vh), veps);
            vw = _mm256_fnmadd_ps(vstep, _mm256_div_ps(vm, denom), vw);

            _mm256_storeu_ps(m + i, vm);
            _mm256_storeu_ps(v + i, vv);
            _mm256_storeu_ps(w + i, vw);
        }
        for (; i < n; ++i) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
            float vh = v[i] * inv_bc2;
            w[i] -= step_size * m[i] / (std::sqrt(vh) + eps);
        }
    }

    float evaluate(const std::vector<Sample>& samples, float* accuracy) const {
        if (samples.empty()) { *accuracy = 0.0f; return 0.0f; }
        size_t n = std::min(samples.size(), static_cast<size_t>(1024));

        std::vector<float> input(n * SEQ_SUMMARY_DIM);
        const_cast<SequenceModel*>(this)->summarize_batch(samples, 0, n, input.data());

        std::vector<float> logits(n * SEQ_NUM_CLASSES);
        const_cast<SequenceModel*>(this)->forward(input.data(), n, logits.data(), false);

        std::vector<float> probs(n * SEQ_NUM_CLASSES);
        softmax(logits.data(), n, probs.data());

        float loss = 0.0f;
        size_t correct = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t truth = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                if (samples[i].y_dir[c] > samples[i].y_dir[truth]) truth = c;
            }
            size_t pred = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                if (probs[i * SEQ_NUM_CLASSES + c] > probs[i * SEQ_NUM_CLASSES + pred]) pred = c;
            }
            if (pred == truth) ++correct;
            loss -= std::log(probs[i * SEQ_NUM_CLASSES + truth] + FROM_EPS_F);
        }
        *accuracy = static_cast<float>(correct) / static_cast<float>(n);
        return loss / static_cast<float>(n);
    }

    // Get hidden representation (layer 2 output) for regime detection
    void get_hidden(const std::vector<Sample>& samples, size_t start, size_t n, float* out) {
        std::vector<float> input(n * SEQ_SUMMARY_DIM);
        summarize_batch(samples, start, n, input.data());

        // Layer 1
        std::vector<float> h1(n * SEQ_HIDDEN1);
        for (size_t i = 0; i < n; ++i) {
            const float* x = input.data() + i * SEQ_SUMMARY_DIM;
            for (size_t j = 0; j < SEQ_HIDDEN1; ++j) {
                float val = b1[j];
                const float* wrow = w1.data() + j * SEQ_SUMMARY_DIM;
                for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) val += wrow[d] * x[d];
                h1[i * SEQ_HIDDEN1 + j] = val > 0.0f ? val : 0.0f;
            }
        }

        // Layer 2
        for (size_t i = 0; i < n; ++i) {
            const float* h = h1.data() + i * SEQ_HIDDEN1;
            for (size_t j = 0; j < SEQ_HIDDEN2; ++j) {
                float val = b2[j];
                const float* wrow = w2.data() + j * SEQ_HIDDEN1;
                for (size_t d = 0; d < SEQ_HIDDEN1; ++d) val += wrow[d] * h[d];
                out[i * SEQ_HIDDEN2 + j] = val > 0.0f ? val : 0.0f;
            }
        }
    }
};

// ============================================================
// Ensemble: 3 models with majority voting
// ============================================================
struct Ensemble {
    static constexpr uint32_t SEEDS[3] = {42, 137, 2718};

    struct Prediction {
        size_t direction;
        float confidence;
        size_t agree_count;
        bool ood;
        size_t regime;
        float size_multiplier;
    };
};

// ============================================================
// Serialization
// ============================================================
class SequenceModelIO {
public:
    static void save(const SequenceModel& model, const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) return;

        char magic[4] = {'F', 'S', 'Q', '3'};
        out.write(magic, 4);
        uint32_t version = 3;
        out.write(reinterpret_cast<const char*>(&version), 4);
        out.write(reinterpret_cast<const char*>(&model.seed), 4);
        out.write(reinterpret_cast<const char*>(&model.lr), 4);

        auto write_vec = [&](const std::vector<float>& v) {
            uint64_t sz = v.size();
            out.write(reinterpret_cast<const char*>(&sz), 8);
            out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(sz * 4));
        };

        write_vec(model.w1); write_vec(model.b1);
        write_vec(model.w2); write_vec(model.b2);
        write_vec(model.w3); write_vec(model.b3);

        // Regime gate
        write_vec(model.regime.centroids.data);
        write_vec(model.regime.covariance_diag.data);
        uint64_t rc[3] = {model.regime.regime_counts[0], model.regime.regime_counts[1], model.regime.regime_counts[2]};
        out.write(reinterpret_cast<const char*>(rc), 24);
        out.write(reinterpret_cast<const char*>(&model.regime.distance_95pct), 4);
        uint8_t cal = model.regime.calibrated ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&cal), 1);

        // Second-pass normalization (v3+)
        uint8_t has_fn = model.feat_norm_ready ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_fn), 1);
        if (model.feat_norm_ready) {
            write_vec(model.feat_mean);
            write_vec(model.feat_std);
        }
    }

    static bool load(const std::string& path, SequenceModel& model) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        char magic[4];
        in.read(magic, 4);
        // Accept both v2 (FSQ2) and v3 (FSQ3)
        uint32_t version = 0;
        if (std::memcmp(magic, "FSQ3", 4) == 0) {
            in.read(reinterpret_cast<char*>(&version), 4);
        } else if (std::memcmp(magic, "FSQ2", 4) == 0) {
            in.read(reinterpret_cast<char*>(&version), 4);
            version = 2;
        } else {
            return false;
        }
        if (version < 2 || version > 3) return false;

        in.read(reinterpret_cast<char*>(&model.seed), 4);
        in.read(reinterpret_cast<char*>(&model.lr), 4);

        auto read_vec = [&](std::vector<float>& v) {
            uint64_t sz;
            in.read(reinterpret_cast<char*>(&sz), 8);
            v.resize(static_cast<size_t>(sz));
            in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sz * 4));
        };

        read_vec(model.w1); read_vec(model.b1);
        read_vec(model.w2); read_vec(model.b2);
        read_vec(model.w3); read_vec(model.b3);

        // Regime gate
        read_vec(model.regime.centroids.data);
        model.regime.centroids.rows = RegimeGate::NUM_REGIMES;
        model.regime.centroids.cols = RegimeGate::HIDDEN_DIM;
        read_vec(model.regime.covariance_diag.data);
        model.regime.covariance_diag.rows = RegimeGate::NUM_REGIMES;
        model.regime.covariance_diag.cols = RegimeGate::HIDDEN_DIM;
        uint64_t rc[3];
        in.read(reinterpret_cast<char*>(rc), 24);
        model.regime.regime_counts = {static_cast<size_t>(rc[0]), static_cast<size_t>(rc[1]), static_cast<size_t>(rc[2])};
        in.read(reinterpret_cast<char*>(&model.regime.distance_95pct), 4);
        uint8_t cal;
        in.read(reinterpret_cast<char*>(&cal), 1);
        model.regime.calibrated = cal != 0;

        // Second-pass normalization (v3+)
        if (version >= 3) {
            uint8_t has_fn = 0;
            if (in.read(reinterpret_cast<char*>(&has_fn), 1) && has_fn) {
                read_vec(model.feat_mean);
                read_vec(model.feat_std);
                model.feat_norm_ready = true;
            }
        }

        return true;
    }
};

}  // namespace from
