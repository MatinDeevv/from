#pragma once

#include "layers/layer_base.hpp"

#include <random>

namespace from {

class Dropout : public ILayer {
    float p_;
    bool variational_;
    std::mt19937 rng_;

public:
    explicit Dropout(float p = 0.1f, bool variational = false, uint32_t seed = 1)
        : p_(p), variational_(variational), rng_(seed) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        if (!training || p_ <= 0.0f) {
            return x;
        }
        Tensor<float> y = x.contiguous();
        std::bernoulli_distribution keep(1.0 - p_);
        float scale = 1.0f / (1.0f - p_ + FROM_EPS_F);
        if (variational_ && x.shape().size() == 3) {
            size_t b = x.shape()[0], s = x.shape()[1], d = x.shape()[2];
            for (size_t n = 0; n < b; ++n) {
                for (size_t c = 0; c < d; ++c) {
                    float m = keep(rng_) ? scale : 0.0f;
                    for (size_t t = 0; t < s; ++t) {
                        y.at(n, t, c) *= m;
                    }
                }
            }
            return y;
        }
        for (size_t i = 0; i < y.numel(); ++i) {
            y[i] *= keep(rng_) ? scale : 0.0f;
        }
        return y;
    }

    std::string name() const override { return "dropout"; }
};

}  // namespace from

