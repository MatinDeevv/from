#pragma once

#include "activations.hpp"
#include "layers/layer_norm.hpp"
#include "layers/linear.hpp"

namespace from {

class GatedResidualNetwork : public ILayer {
    Linear fc1_;
    Linear fc2_;
    Linear gate_;
    LayerNorm norm_;

public:
    GatedResidualNetwork(size_t dim = 64, size_t hidden = 128, uint64_t seed = 1)
        : fc1_(dim, hidden, seed), fc2_(hidden, dim, seed + 1), gate_(dim, dim, seed + 2), norm_(dim) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        Tensor<float> h = fc1_.forward(x, training).unary([](float v) { return act::elu(v); });
        h = fc2_.forward(h, training);
        Tensor<float> g = gate_.forward(x, training).unary(sigmoid);
        Tensor<float> y = h * g + x;
        return norm_.forward(y, training);
    }

    std::vector<ParameterRef> parameters() override {
        auto p = fc1_.parameters();
        auto q = fc2_.parameters();
        auto r = gate_.parameters();
        auto n = norm_.parameters();
        p.insert(p.end(), q.begin(), q.end());
        p.insert(p.end(), r.begin(), r.end());
        p.insert(p.end(), n.begin(), n.end());
        return p;
    }

    std::string name() const override { return "gated_residual"; }
};

}  // namespace from
