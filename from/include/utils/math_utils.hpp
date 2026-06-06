#pragma once

#include "common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace from {

inline float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

inline float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

inline float log_sum_exp(const std::vector<float>& xs) {
    if (xs.empty()) {
        return -std::numeric_limits<float>::infinity();
    }
    float mx = *std::max_element(xs.begin(), xs.end());
    float sum = 0.0f;
    for (float x : xs) {
        sum += std::exp(x - mx);
    }
    return mx + std::log(sum + FROM_EPS_F);
}

inline std::vector<float> stable_softmax(const std::vector<float>& logits) {
    std::vector<float> out(logits.size(), 0.0f);
    if (logits.empty()) {
        return out;
    }
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(logits[i] - mx);
        sum += out[i];
    }
    for (float& x : out) {
        x /= (sum + FROM_EPS_F);
    }
    return out;
}

inline bool cholesky_decompose(const std::vector<float>& a, std::vector<float>& l, size_t n) {
    l.assign(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < j; ++k) {
                sum += l[i * n + k] * l[j * n + k];
            }
            if (i == j) {
                float v = a[i * n + i] - sum;
                if (v <= 0.0f) {
                    return false;
                }
                l[i * n + j] = std::sqrt(v);
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / (l[j * n + j] + FROM_EPS_F);
            }
        }
    }
    return true;
}

inline std::vector<float> cholesky_inverse(const std::vector<float>& a, size_t n) {
    std::vector<float> l;
    require(cholesky_decompose(a, l, n), "Cholesky decomposition failed");
    std::vector<float> inv_l(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        inv_l[i * n + i] = 1.0f / (l[i * n + i] + FROM_EPS_F);
        for (size_t j = 0; j < i; ++j) {
            float sum = 0.0f;
            for (size_t k = j; k < i; ++k) {
                sum += l[i * n + k] * inv_l[k * n + j];
            }
            inv_l[i * n + j] = -sum / (l[i * n + i] + FROM_EPS_F);
        }
    }
    std::vector<float> inv(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = std::max(i, j); k < n; ++k) {
                sum += inv_l[k * n + i] * inv_l[k * n + j];
            }
            inv[i * n + j] = sum;
        }
    }
    return inv;
}

}  // namespace from

