#pragma once

#include "layers/attention.hpp"
#include "layers/gated_residual.hpp"
#include "layers/layer_norm.hpp"

namespace from {

class TemporalFusionBlock : public ILayer {
    MultiHeadAttention attn_;
    GatedResidualNetwork grn_;
    Linear ff1_;
    Linear ff2_;
    LayerNorm norm_;

public:
    TemporalFusionBlock(size_t d_model = 128, size_t heads = 4, size_t d_ff = 256, uint64_t seed = 1)
        : attn_(d_model, heads, seed), grn_(d_model, d_ff, seed + 10), ff1_(d_model, d_ff, seed + 20),
          ff2_(d_ff, d_model, seed + 21), norm_(d_model) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        Tensor<float> a = attn_.forward(x, training);
        Tensor<float> r = norm_.forward(a + x, training);
        Tensor<float> h = ff1_.forward(r, training).unary([](float v) { return act::elu(v); });
        h = ff2_.forward(h, training);
        return grn_.forward(h + r, training);
    }

    std::vector<ParameterRef> parameters() override {
        auto p = attn_.parameters();
        auto q = grn_.parameters();
        auto r = ff1_.parameters();
        auto s = ff2_.parameters();
        auto n = norm_.parameters();
        p.insert(p.end(), q.begin(), q.end());
        p.insert(p.end(), r.begin(), r.end());
        p.insert(p.end(), s.begin(), s.end());
        p.insert(p.end(), n.begin(), n.end());
        return p;
    }

    const Tensor<float>& last_attention() const { return attn_.last_attention; }
    std::string name() const override { return "temporal_fusion"; }
};

}  // namespace from
