#pragma once

#include "initializers.hpp"
#include "layers/layer_base.hpp"

namespace from {

class Linear : public ILayer {
    size_t in_features_;
    size_t out_features_;
    Tensor<float> last_input_;

public:
    Tensor<float> weight;
    Tensor<float> bias;
    Tensor<float> grad_weight;
    Tensor<float> grad_bias;

    Linear(size_t in_features = 1, size_t out_features = 1, uint64_t seed = 1)
        : in_features_(in_features),
          out_features_(out_features),
          weight(init::kaiming_uniform({out_features, in_features}, in_features, seed)),
          bias(Tensor<float>::zeros({out_features})),
          grad_weight(Tensor<float>::zeros({out_features, in_features})),
          grad_bias(Tensor<float>::zeros({out_features})) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        (void)training;
        last_input_ = x.contiguous();
        if (x.shape().size() == 2) {
            Tensor<float> y = x.matmul(weight.T2());
            for (size_t i = 0; i < y.shape()[0]; ++i) {
                for (size_t j = 0; j < out_features_; ++j) {
                    y.at(i, j) += bias.at(j);
                }
            }
            return y;
        }
        require(x.shape().size() == 3, "Linear expects 2D or 3D input");
        size_t b = x.shape()[0], s = x.shape()[1], in = x.shape()[2];
        require(in == in_features_, "Linear input feature mismatch");
        Tensor<float> flat = x.contiguous().reshape({b * s, in});
        Tensor<float> y = flat.matmul(weight.T2());
        for (size_t i = 0; i < y.shape()[0]; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                y.at(i, j) += bias.at(j);
            }
        }
        return y.reshape({b, s, out_features_});
    }

    Tensor<float> backward(const Tensor<float>& grad_out) override {
        std::fill(grad_weight.data_ptr(), grad_weight.data_ptr() + grad_weight.numel(), 0.0f);
        std::fill(grad_bias.data_ptr(), grad_bias.data_ptr() + grad_bias.numel(), 0.0f);
        Tensor<float> go = grad_out.contiguous();
        Tensor<float> x = last_input_.contiguous();
        if (go.shape().size() == 3) {
            go = go.reshape({go.shape()[0] * go.shape()[1], go.shape()[2]});
            x = x.reshape({x.shape()[0] * x.shape()[1], x.shape()[2]});
        }
        for (size_t n = 0; n < go.shape()[0]; ++n) {
            for (size_t o = 0; o < out_features_; ++o) {
                grad_bias.at(o) += go.at(n, o);
                for (size_t i = 0; i < in_features_; ++i) {
                    grad_weight.at(o, i) += go.at(n, o) * x.at(n, i);
                }
            }
        }
        Tensor<float> grad_in = go.matmul(weight);
        if (last_input_.shape().size() == 3) {
            return grad_in.reshape(last_input_.shape());
        }
        return grad_in;
    }

    std::vector<ParameterRef> parameters() override {
        return {{"weight", &weight, &grad_weight, false}, {"bias", &bias, &grad_bias, false}};
    }

    std::string name() const override { return "linear"; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
};

}  // namespace from

