#pragma once

#include "data/windower.hpp"

#include <random>

namespace from {

struct AugmentConfig {
    bool enabled = true;
    float magnitude_prob = 0.3f;
    float time_warp_prob = 0.2f;
    float noise_prob = 0.4f;
    float spread_shock_prob = 0.15f;
    float tick_dropout_prob = 0.2f;
    float regime_splice_prob = 0.1f;
};

class Augmenter {
    AugmentConfig cfg_;
    std::mt19937 rng_;

    bool hit(float p) {
        std::bernoulli_distribution d(p);
        return d(rng_);
    }

public:
    explicit Augmenter(AugmentConfig cfg = {}, uint32_t seed = 1) : cfg_(cfg), rng_(seed) {}

    void apply(Sample& sample) {
        if (!cfg_.enabled) {
            return;
        }
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        if (hit(cfg_.magnitude_prob)) {
            float scale = 0.98f + 0.04f * uni(rng_);
            for (size_t i = 0; i < sample.X.numel(); ++i) {
                sample.X[i] *= scale;
            }
        }
        if (hit(cfg_.noise_prob)) {
            std::normal_distribution<float> noise(0.0f, 0.001f);
            for (size_t i = 0; i < sample.X.numel(); ++i) {
                sample.X[i] += noise(rng_);
            }
        }
        if (hit(cfg_.spread_shock_prob) && sample.X.shape()[0] > 20) {
            std::uniform_int_distribution<size_t> pos(0, sample.X.shape()[0] - 20);
            std::uniform_int_distribution<size_t> len(5, 20);
            std::uniform_real_distribution<float> mult(2.0f, 5.0f);
            size_t start = pos(rng_);
            size_t end = std::min(sample.X.shape()[0], start + len(rng_));
            float m = mult(rng_);
            for (size_t t = start; t < end; ++t) {
                sample.X.at(t, 5) *= m;
                sample.X.at(t, 6) *= m;
            }
        }
        if (hit(cfg_.tick_dropout_prob)) {
            std::uniform_real_distribution<float> frac(0.01f, 0.05f);
            float p = frac(rng_);
            for (size_t t = 0; t < sample.X.shape()[0]; ++t) {
                if (uni(rng_) < p) {
                    for (size_t d = 0; d < sample.X.shape()[1]; ++d) {
                        sample.X.at(t, d) = 0.0f;
                    }
                }
            }
        }
        if (hit(cfg_.time_warp_prob) && sample.X.shape()[0] > 2) {
            Tensor<float> original = sample.X;
            float speed = 0.9f + 0.2f * uni(rng_);
            size_t seq = sample.X.shape()[0];
            size_t dim = sample.X.shape()[1];
            for (size_t t = 0; t < seq; ++t) {
                float src = std::min<float>(static_cast<float>(seq - 1), static_cast<float>(t) * speed);
                size_t lo = static_cast<size_t>(src);
                size_t hi = std::min(seq - 1, lo + 1);
                float a = src - static_cast<float>(lo);
                for (size_t d = 0; d < dim; ++d) {
                    sample.X.at(t, d) = original.at(lo, d) * (1.0f - a) + original.at(hi, d) * a;
                }
            }
        }
    }
};

}  // namespace from

