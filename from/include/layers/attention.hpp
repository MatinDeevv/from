#pragma once

#include "initializers.hpp"
#include "layers/layer_base.hpp"
#include "utils/math_utils.hpp"

namespace from {

class MultiHeadAttention : public ILayer {
    size_t d_model_;
    size_t heads_;
    size_t d_head_;

public:
    Tensor<float> wq, wk, wv, wo, rel_bias;
    Tensor<float> grad_wq, grad_wk, grad_wv, grad_wo, grad_rel_bias;
    Tensor<float> last_attention;

    MultiHeadAttention(size_t d_model = 128, size_t heads = 4, uint64_t seed = 1)
        : d_model_(d_model),
          heads_(heads),
          d_head_(std::max<size_t>(1, d_model / heads)),
          wq(init::xavier_uniform({d_model, d_model}, d_model, d_model, seed)),
          wk(init::xavier_uniform({d_model, d_model}, d_model, d_model, seed + 1)),
          wv(init::xavier_uniform({d_model, d_model}, d_model, d_model, seed + 2)),
          wo(init::xavier_uniform({d_model, d_model}, d_model, d_model, seed + 3)),
          rel_bias(Tensor<float>::zeros({heads, 1024})),
          grad_wq(Tensor<float>::zeros({d_model, d_model})),
          grad_wk(Tensor<float>::zeros({d_model, d_model})),
          grad_wv(Tensor<float>::zeros({d_model, d_model})),
          grad_wo(Tensor<float>::zeros({d_model, d_model})),
          grad_rel_bias(Tensor<float>::zeros({heads, 1024})) {}

    Tensor<float> project(const Tensor<float>& x, const Tensor<float>& w) const {
        size_t b = x.shape()[0], s = x.shape()[1];
        return x.reshape({b * s, d_model_}).matmul(w.T2()).reshape({b, s, d_model_});
    }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        (void)training;
        require(x.shape().size() == 3 && x.shape()[2] == d_model_, "Attention expects [batch, seq, d_model]");
        size_t batch = x.shape()[0], seq = x.shape()[1];
        Tensor<float> q = project(x.contiguous(), wq);
        Tensor<float> k = project(x.contiguous(), wk);
        Tensor<float> v = project(x.contiguous(), wv);
        Tensor<float> concat({batch, seq, d_model_});
        last_attention = Tensor<float>({batch, heads_, seq, seq});
        float scale = 1.0f / std::sqrt(static_cast<float>(d_head_));
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads_; ++h) {
                for (size_t i = 0; i < seq; ++i) {
                    std::vector<float> scores(i + 1, 0.0f);
                    for (size_t j = 0; j <= i; ++j) {
                        float dot = 0.0f;
                        for (size_t d = 0; d < d_head_; ++d) {
                            size_t c = h * d_head_ + d;
                            if (c < d_model_) {
                                dot += q.at(b, i, c) * k.at(b, j, c);
                            }
                        }
                        size_t lag = std::min<size_t>(1023, i - j);
                        scores[j] = dot * scale + rel_bias.at(h, lag);
                    }
                    std::vector<float> probs = stable_softmax(scores);
                    for (size_t j = 0; j <= i; ++j) {
                        last_attention.at(b, h, i, j) = probs[j];
                    }
                    for (size_t d = 0; d < d_head_; ++d) {
                        size_t c = h * d_head_ + d;
                        if (c >= d_model_) continue;
                        float acc = 0.0f;
                        for (size_t j = 0; j <= i; ++j) {
                            acc += probs[j] * v.at(b, j, c);
                        }
                        concat.at(b, i, c) = acc;
                    }
                }
            }
        }
        return project(concat, wo);
    }

    std::vector<ParameterRef> parameters() override {
        return {{"wq", &wq, &grad_wq, false}, {"wk", &wk, &grad_wk, false}, {"wv", &wv, &grad_wv, false},
                {"wo", &wo, &grad_wo, false}, {"rel_bias", &rel_bias, &grad_rel_bias, false}};
    }

    std::string name() const override { return "attention"; }
};

}  // namespace from
