#pragma once

#include "activations.hpp"
#include "initializers.hpp"
#include "layers/layer_base.hpp"

namespace from {

class LSTM : public ILayer {
    size_t input_dim_;
    size_t hidden_dim_;
    size_t layers_;
    float dropout_;

public:
    std::vector<Tensor<float>> wx;
    std::vector<Tensor<float>> wh;
    std::vector<Tensor<float>> b;
    std::vector<Tensor<float>> grad_wx;
    std::vector<Tensor<float>> grad_wh;
    std::vector<Tensor<float>> grad_b;

    LSTM(size_t input_dim = 64, size_t hidden_dim = 128, size_t layers = 1, float dropout = 0.0f, uint64_t seed = 1)
        : input_dim_(input_dim), hidden_dim_(hidden_dim), layers_(layers), dropout_(dropout) {
        for (size_t l = 0; l < layers_; ++l) {
            size_t in = l == 0 ? input_dim_ : hidden_dim_;
            wx.push_back(init::xavier_uniform({4 * hidden_dim_, in}, in, 4 * hidden_dim_, seed + l));
            wh.push_back(init::orthogonal({4 * hidden_dim_, hidden_dim_}, seed + 100 + l));
            b.push_back(Tensor<float>::zeros({4 * hidden_dim_}));
            grad_wx.push_back(Tensor<float>::zeros({4 * hidden_dim_, in}));
            grad_wh.push_back(Tensor<float>::zeros({4 * hidden_dim_, hidden_dim_}));
            grad_b.push_back(Tensor<float>::zeros({4 * hidden_dim_}));
        }
    }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        (void)training;
        require(x.shape().size() == 3, "LSTM expects [batch, seq, dim]");
        Tensor<float> layer_in = x.contiguous();
        size_t batch = x.shape()[0], seq = x.shape()[1];
        for (size_t l = 0; l < layers_; ++l) {
            Tensor<float> h({batch, hidden_dim_});
            Tensor<float> c({batch, hidden_dim_});
            Tensor<float> out({batch, seq, hidden_dim_});
            for (size_t t = 0; t < seq; ++t) {
                for (size_t n = 0; n < batch; ++n) {
                    for (size_t u = 0; u < hidden_dim_; ++u) {
                        float gates[4] = {b[l].at(u), b[l].at(hidden_dim_ + u), b[l].at(2 * hidden_dim_ + u), b[l].at(3 * hidden_dim_ + u)};
                        for (size_t i = 0; i < layer_in.shape()[2]; ++i) {
                            float xv = layer_in.at(n, t, i);
                            for (size_t g = 0; g < 4; ++g) {
                                gates[g] += wx[l].at(g * hidden_dim_ + u, i) * xv;
                            }
                        }
                        for (size_t i = 0; i < hidden_dim_; ++i) {
                            float hv = h.at(n, i);
                            for (size_t g = 0; g < 4; ++g) {
                                gates[g] += wh[l].at(g * hidden_dim_ + u, i) * hv;
                            }
                        }
                        float ig = sigmoid(gates[0]);
                        float fg = sigmoid(gates[1]);
                        float gg = std::tanh(gates[2]);
                        float og = sigmoid(gates[3]);
                        c.at(n, u) = fg * c.at(n, u) + ig * gg;
                        h.at(n, u) = og * std::tanh(c.at(n, u));
                        out.at(n, t, u) = h.at(n, u);
                    }
                }
            }
            layer_in = out;
        }
        return layer_in;
    }

    std::vector<ParameterRef> parameters() override {
        std::vector<ParameterRef> p;
        for (size_t l = 0; l < layers_; ++l) {
            p.push_back({"wx_" + std::to_string(l), &wx[l], &grad_wx[l], false});
            p.push_back({"wh_" + std::to_string(l), &wh[l], &grad_wh[l], false});
            p.push_back({"b_" + std::to_string(l), &b[l], &grad_b[l], false});
        }
        return p;
    }

    std::string name() const override { return "lstm"; }
};

}  // namespace from

