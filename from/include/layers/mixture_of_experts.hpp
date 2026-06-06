#pragma once

#include "activations.hpp"
#include "layers/linear.hpp"

#include <algorithm>
#include <numeric>
namespace from {

class MixtureOfExperts : public ILayer {
    size_t dim_;
    size_t num_experts_;
    size_t top_k_;
    size_t expert_dim_;
    Linear router_;
    std::vector<Linear> up_;
    std::vector<Linear> down_;
    std::vector<float> utilization_;

public:
    float load_balance_loss = 0.0f;

    MixtureOfExperts(size_t dim = 128, size_t num_experts = 16, size_t top_k = 2, size_t expert_dim = 512, uint64_t seed = 1)
        : dim_(dim), num_experts_(num_experts), top_k_(top_k), expert_dim_(expert_dim), router_(dim, num_experts, seed),
          utilization_(num_experts, 0.0f) {
        for (size_t i = 0; i < num_experts_; ++i) {
            up_.emplace_back(dim_, expert_dim_, seed + 10 + i);
            down_.emplace_back(expert_dim_, dim_, seed + 100 + i);
        }
    }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        require(x.shape().size() == 3, "MoE expects [batch, seq, dim]");
        size_t b = x.shape()[0], s = x.shape()[1];
        Tensor<float> flat = x.contiguous().reshape({b * s, dim_});
        Tensor<float> logits = router_.forward(flat, training);
        Tensor<float> probs = logits.softmax(1);
        Tensor<float> out({b * s, dim_});
        std::fill(utilization_.begin(), utilization_.end(), 0.0f);
        for (size_t n = 0; n < flat.shape()[0]; ++n) {
            std::vector<size_t> idx(num_experts_);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + std::min(top_k_, num_experts_), idx.end(),
                              [&](size_t a, size_t c) { return probs.at(n, a) > probs.at(n, c); });
            float norm = 0.0f;
            for (size_t k = 0; k < top_k_; ++k) norm += probs.at(n, idx[k]);
            for (size_t k = 0; k < top_k_; ++k) {
                size_t e = idx[k];
                utilization_[e] += 1.0f;
                Tensor<float> token({1, dim_});
                for (size_t d = 0; d < dim_; ++d) token.at(0, d) = flat.at(n, d);
                Tensor<float> y = down_[e].forward(up_[e].forward(token, training).unary(act::gelu), training);
                float w = probs.at(n, e) / (norm + FROM_EPS_F);
                for (size_t d = 0; d < dim_; ++d) out.at(n, d) += w * y.at(0, d);
            }
        }
        float tokens = static_cast<float>(flat.shape()[0]);
        load_balance_loss = 0.0f;
        for (size_t e = 0; e < num_experts_; ++e) {
            float f = utilization_[e] / (tokens + FROM_EPS_F);
            float p = 0.0f;
            for (size_t n = 0; n < flat.shape()[0]; ++n) p += probs.at(n, e);
            p /= tokens + FROM_EPS_F;
            load_balance_loss += f * p;
        }
        load_balance_loss *= 0.01f * static_cast<float>(num_experts_);
        return out.reshape({b, s, dim_});
    }

    const std::vector<float>& utilization() const { return utilization_; }

    std::vector<ParameterRef> parameters() override {
        auto p = router_.parameters();
        for (size_t e = 0; e < num_experts_; ++e) {
            auto a = up_[e].parameters();
            auto b = down_[e].parameters();
            p.insert(p.end(), a.begin(), a.end());
            p.insert(p.end(), b.begin(), b.end());
        }
        return p;
    }

    std::string name() const override { return "mixture_of_experts"; }
};

}  // namespace from
