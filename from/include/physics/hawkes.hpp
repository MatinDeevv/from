#pragma once

#include "layers/lstm.hpp"
#include "layers/linear.hpp"
#include "utils/math_utils.hpp"

namespace from {

class NeuralHawkesProcess : public ILayer {
    LSTM lstm_;
    Linear intensity_;

public:
    explicit NeuralHawkesProcess(size_t feature_dim = 3, size_t hidden = 32, uint64_t seed = 1)
        : lstm_(feature_dim, hidden, 1, 0.0f, seed), intensity_(hidden, 1, seed + 10) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        Tensor<float> h = lstm_.forward(x, training);
        Tensor<float> lam = intensity_.forward(h, training).unary(softplus);
        return lam;
    }

    float negative_log_likelihood(const Tensor<float>& intensity, const Tensor<float>& dt) const {
        float ll = 0.0f;
        for (size_t i = 0; i < intensity.numel(); ++i) {
            float lam = std::max(intensity[i], 1e-6f);
            float interval = dt.numel() == intensity.numel() ? dt[i] : 0.1f;
            ll += -std::log(lam) + lam * interval;
        }
        return ll / static_cast<float>(std::max<size_t>(1, intensity.numel()));
    }

    std::vector<ParameterRef> parameters() override {
        auto p = lstm_.parameters();
        auto q = intensity_.parameters();
        p.insert(p.end(), q.begin(), q.end());
        return p;
    }

    std::string name() const override { return "hawkes"; }
};

}  // namespace from

