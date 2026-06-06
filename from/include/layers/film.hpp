#pragma once

#include "layers/linear.hpp"

namespace from {

class FiLM : public ILayer {
    Linear gamma_;
    Linear beta_;
    Tensor<float> context_;

public:
    FiLM(size_t context_dim = 64, size_t feature_dim = 64, uint64_t seed = 1)
        : gamma_(context_dim, feature_dim, seed), beta_(context_dim, feature_dim, seed + 1) {}

    void set_context(const Tensor<float>& context) { context_ = context; }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        Tensor<float> context = context_.numel() == 0 ? Tensor<float>::zeros({x.shape()[0], gamma_.in_features()}) : context_;
        Tensor<float> g = gamma_.forward(context, training);
        Tensor<float> b = beta_.forward(context, training);
        Tensor<float> y = x.contiguous();
        size_t batch = x.shape()[0];
        size_t dim = x.shape().back();
        for (size_t n = 0; n < batch; ++n) {
            for (size_t d = 0; d < dim; ++d) {
                if (x.shape().size() == 2) {
                    y.at(n, d) = (1.0f + g.at(n, d)) * x.at(n, d) + b.at(n, d);
                } else {
                    for (size_t t = 0; t < x.shape()[1]; ++t) {
                        y.at(n, t, d) = (1.0f + g.at(n, d)) * x.at(n, t, d) + b.at(n, d);
                    }
                }
            }
        }
        return y;
    }

    std::vector<ParameterRef> parameters() override {
        auto p = gamma_.parameters();
        auto q = beta_.parameters();
        p.insert(p.end(), q.begin(), q.end());
        return p;
    }

    std::string name() const override { return "film"; }
};

}  // namespace from

