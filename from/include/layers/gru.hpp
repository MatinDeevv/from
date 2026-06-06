#pragma once

#include "initializers.hpp"
#include "layers/layer_base.hpp"
#include "utils/math_utils.hpp"

namespace from {

class GRU : public ILayer {
    size_t input_dim_;
    size_t hidden_dim_;

public:
    Tensor<float> wx;
    Tensor<float> wh;
    Tensor<float> bias;
    Tensor<float> grad_wx;
    Tensor<float> grad_wh;
    Tensor<float> grad_bias;

    GRU(size_t input_dim = 64, size_t hidden_dim = 128, uint64_t seed = 1)
        : input_dim_(input_dim),
          hidden_dim_(hidden_dim),
          wx(init::xavier_uniform({3 * hidden_dim, input_dim}, input_dim, 3 * hidden_dim, seed)),
          wh(init::orthogonal({3 * hidden_dim, hidden_dim}, seed + 1)),
          bias(Tensor<float>::zeros({3 * hidden_dim})),
          grad_wx(Tensor<float>::zeros({3 * hidden_dim, input_dim})),
          grad_wh(Tensor<float>::zeros({3 * hidden_dim, hidden_dim})),
          grad_bias(Tensor<float>::zeros({3 * hidden_dim})) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        (void)training;
        require(x.shape().size() == 3, "GRU expects [batch, seq, dim]");
        size_t batch = x.shape()[0], seq = x.shape()[1];
        Tensor<float> h({batch, hidden_dim_});
        Tensor<float> out({batch, seq, hidden_dim_});
        for (size_t t = 0; t < seq; ++t) {
            for (size_t n = 0; n < batch; ++n) {
                for (size_t u = 0; u < hidden_dim_; ++u) {
                    float r = bias.at(u), z = bias.at(hidden_dim_ + u), nn = bias.at(2 * hidden_dim_ + u);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        r += wx.at(u, i) * x.at(n, t, i);
                        z += wx.at(hidden_dim_ + u, i) * x.at(n, t, i);
                        nn += wx.at(2 * hidden_dim_ + u, i) * x.at(n, t, i);
                    }
                    float hr = 0.0f, hz = 0.0f, hn = 0.0f;
                    for (size_t i = 0; i < hidden_dim_; ++i) {
                        hr += wh.at(u, i) * h.at(n, i);
                        hz += wh.at(hidden_dim_ + u, i) * h.at(n, i);
                        hn += wh.at(2 * hidden_dim_ + u, i) * h.at(n, i);
                    }
                    r = sigmoid(r + hr);
                    z = sigmoid(z + hz);
                    float cand = std::tanh(nn + r * hn);
                    h.at(n, u) = (1.0f - z) * h.at(n, u) + z * cand;
                    out.at(n, t, u) = h.at(n, u);
                }
            }
        }
        return out;
    }

    std::vector<ParameterRef> parameters() override {
        return {{"wx", &wx, &grad_wx, false}, {"wh", &wh, &grad_wh, false}, {"bias", &bias, &grad_bias, false}};
    }
    std::string name() const override { return "gru"; }
};

}  // namespace from

