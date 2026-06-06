#pragma once

#include "tensor.hpp"

#include <cmath>
#include <random>

namespace from::init {

inline Tensor<float> kaiming_uniform(const std::vector<size_t>& shape, size_t fan_in, uint64_t seed = 1) {
    float bound = std::sqrt(6.0f / static_cast<float>(std::max<size_t>(1, fan_in)));
    return Tensor<float>::rand_uniform(shape, -bound, bound, seed);
}

inline Tensor<float> xavier_uniform(const std::vector<size_t>& shape, size_t fan_in, size_t fan_out, uint64_t seed = 1) {
    float bound = std::sqrt(6.0f / static_cast<float>(std::max<size_t>(1, fan_in + fan_out)));
    return Tensor<float>::rand_uniform(shape, -bound, bound, seed);
}

inline Tensor<float> orthogonal(const std::vector<size_t>& shape, uint64_t seed = 1) {
    require(shape.size() == 2, "orthogonal initializer expects 2D shape");
    Tensor<float> q = Tensor<float>::randn(shape, 0.0f, 1.0f, seed);
    size_t rows = shape[0];
    size_t cols = shape[1];
    for (size_t c = 0; c < cols; ++c) {
        for (size_t prev = 0; prev < c; ++prev) {
            float dot = 0.0f;
            for (size_t r = 0; r < rows; ++r) {
                dot += q.at(r, c) * q.at(r, prev);
            }
            for (size_t r = 0; r < rows; ++r) {
                q.at(r, c) -= dot * q.at(r, prev);
            }
        }
        float norm = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            norm += q.at(r, c) * q.at(r, c);
        }
        norm = std::sqrt(norm + FROM_EPS_F);
        for (size_t r = 0; r < rows; ++r) {
            q.at(r, c) /= norm;
        }
    }
    return q;
}

inline Tensor<float> sparse(const std::vector<size_t>& shape, float density = 0.1f, uint64_t seed = 1) {
    Tensor<float> t(shape);
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution keep(density);
    std::normal_distribution<float> vals(0.0f, 0.01f);
    for (size_t i = 0; i < t.numel(); ++i) {
        t[i] = keep(rng) ? vals(rng) : 0.0f;
    }
    return t;
}

}  // namespace from::init

