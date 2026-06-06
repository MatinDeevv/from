#pragma once

#include "activations.hpp"
#include "layers/linear.hpp"

namespace from {

class NeuralODELayer : public ILayer {
    size_t dim_;
    size_t steps_;
    Linear f1_;
    Linear f2_;

    Tensor<float> deriv(const Tensor<float>& h, bool training) {
        return f2_.forward(f1_.forward(h, training).unary([](float v) { return std::tanh(v); }), training);
    }

public:
    NeuralODELayer(size_t dim = 128, size_t hidden = 64, size_t steps = 10, uint64_t seed = 1)
        : dim_(dim), steps_(steps), f1_(dim, hidden, seed), f2_(hidden, dim, seed + 1) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        require(x.shape().size() == 3, "NeuralODE expects [batch, seq, dim]");
        Tensor<float> y = x.contiguous();
        float dt = 1.0f / static_cast<float>(std::max<size_t>(1, steps_));
        for (size_t step = 0; step < steps_; ++step) {
            Tensor<float> flat = y.reshape({y.shape()[0] * y.shape()[1], dim_});
            Tensor<float> k1 = deriv(flat, training);
            Tensor<float> k2 = deriv(flat + k1 * (dt * 0.5f), training);
            Tensor<float> k3 = deriv(flat + k2 * (dt * 0.5f), training);
            Tensor<float> k4 = deriv(flat + k3 * dt, training);
            Tensor<float> update = (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (dt / 6.0f);
            y = (flat + update).reshape(y.shape());
        }
        return y;
    }

    std::vector<ParameterRef> parameters() override {
        auto p = f1_.parameters();
        auto q = f2_.parameters();
        p.insert(p.end(), q.begin(), q.end());
        return p;
    }

    std::string name() const override { return "neural_ode"; }
};

}  // namespace from
