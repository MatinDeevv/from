#pragma once

#include "layers/linear.hpp"
#include "utils/math_utils.hpp"

namespace from {

class KalmanFilterLayer : public ILayer {
    size_t input_dim_;
    size_t state_dim_;
    Linear q_head_;
    Linear r_head_;
    Tensor<float> state_;
    Tensor<float> cov_;

public:
    KalmanFilterLayer(size_t input_dim = 23, size_t state_dim = 16, uint64_t seed = 1)
        : input_dim_(input_dim), state_dim_(state_dim), q_head_(input_dim, state_dim, seed),
          r_head_(input_dim, state_dim, seed + 1) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        require(x.shape().size() == 3, "Kalman expects [batch, seq, dim]");
        size_t b = x.shape()[0], s = x.shape()[1];
        Tensor<float> out({b, s, state_dim_ * 2});
        state_ = Tensor<float>::zeros({b, state_dim_});
        cov_ = Tensor<float>::ones({b, state_dim_});
        Tensor<float> flat = x.contiguous().reshape({b * s, input_dim_});
        Tensor<float> q_all = q_head_.forward(flat, training).unary(softplus).reshape({b, s, state_dim_});
        Tensor<float> r_all = r_head_.forward(flat, training).unary(softplus).reshape({b, s, state_dim_});
        for (size_t t = 0; t < s; ++t) {
            for (size_t n = 0; n < b; ++n) {
                for (size_t d = 0; d < state_dim_; ++d) {
                    float z = x.at(n, t, d % input_dim_);
                    float p_pred = cov_.at(n, d) + q_all.at(n, t, d) + 1e-5f;
                    float gain = p_pred / (p_pred + r_all.at(n, t, d) + 1e-5f);
                    float pred = state_.at(n, d);
                    state_.at(n, d) = pred + gain * (z - pred);
                    cov_.at(n, d) = (1.0f - gain) * p_pred;
                    out.at(n, t, d) = state_.at(n, d);
                    out.at(n, t, state_dim_ + d) = cov_.at(n, d);
                }
            }
        }
        return out;
    }

    std::vector<ParameterRef> parameters() override {
        auto p = q_head_.parameters();
        auto r = r_head_.parameters();
        p.insert(p.end(), r.begin(), r.end());
        return p;
    }

    std::string name() const override { return "kalman"; }
};

}  // namespace from

