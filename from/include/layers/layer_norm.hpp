#pragma once

#include "layers/layer_base.hpp"

namespace from {

class LayerNorm : public ILayer {
    size_t dim_;
    float eps_;

public:
    Tensor<float> gamma;
    Tensor<float> beta;
    Tensor<float> grad_gamma;
    Tensor<float> grad_beta;

    explicit LayerNorm(size_t dim = 1, float eps = 1e-5f)
        : dim_(dim),
          eps_(eps),
          gamma(Tensor<float>::ones({dim})),
          beta(Tensor<float>::zeros({dim})),
          grad_gamma(Tensor<float>::zeros({dim})),
          grad_beta(Tensor<float>::zeros({dim})) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        (void)training;
        Tensor<float> y = x.layer_norm(x.shape().size() - 1, eps_);
        size_t d = x.shape().back();
        for (size_t i = 0; i < y.numel(); ++i) {
            size_t c = i % d;
            y[i] = y[i] * gamma.at(c) + beta.at(c);
        }
        return y;
    }

    std::vector<ParameterRef> parameters() override {
        return {{"gamma", &gamma, &grad_gamma, false}, {"beta", &beta, &grad_beta, false}};
    }

    std::string name() const override { return "layer_norm"; }
};

}  // namespace from

