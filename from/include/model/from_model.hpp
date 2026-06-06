#pragma once

#include "data/normalizer.hpp"
#include "layers/conv1d.hpp"
#include "layers/dropout.hpp"
#include "layers/film.hpp"
#include "layers/hyperbolic.hpp"
#include "layers/layer_norm.hpp"
#include "layers/lstm.hpp"
#include "layers/mixture_of_experts.hpp"
#include "layers/temporal_fusion.hpp"
#include "physics/hawkes.hpp"
#include "physics/kalman.hpp"
#include "physics/neural_ode.hpp"

namespace from {

struct FromConfig {
    size_t feature_dim = FROM_MAX_FEATURES;
    size_t conv_out = 64;
    size_t lstm_hidden = 64;
    size_t lstm_layers = 1;
    size_t attention_heads = 4;
    size_t attention_d_ff = 128;
    size_t tft_blocks = 1;
    size_t moe_experts = 4;
    size_t moe_top_k = 2;
    size_t moe_expert_dim = 128;
    size_t hyperbolic_dim = 16;
    size_t kalman_state_dim = 16;
    size_t ode_hidden = 32;
    size_t ode_steps = 2;
    float dropout = 0.1f;
};

struct FromOutput {
    Tensor<float> logits_dir;
    Tensor<float> logits_q;
    Tensor<float> pred_vol;
    Tensor<float> pred_spread;
    Tensor<float> pred_noise;
    Tensor<float> probs_dir;
    Tensor<float> embedding;
    Tensor<float> ood_score;
    float load_balance_loss = 0.0f;
};

class FromModel {
    FromConfig cfg_;
    NeuralHawkesProcess hawkes_;
    KalmanFilterLayer kalman_;
    Conv1D conv_;
    LayerNorm conv_norm_;
    Dropout dropout_;
    MixtureOfExperts moe_;
    LSTM lstm_;
    NeuralODELayer ode_;
    std::vector<TemporalFusionBlock> tft_;
    PoincareBall hyp_;
    FiLM film_;
    Linear dir_head_;
    Linear quant_head_;
    Linear vol_head_;
    Linear spread_head_;
    Linear noise_head_;
    float temperature_ = 1.0f;

    Tensor<float> concat_last_dim(const Tensor<float>& a, const Tensor<float>& b) const {
        require(a.shape().size() == b.shape().size() && a.shape()[0] == b.shape()[0] && a.shape()[1] == b.shape()[1],
                "concat shape mismatch");
        Tensor<float> out({a.shape()[0], a.shape()[1], a.shape()[2] + b.shape()[2]});
        for (size_t n = 0; n < a.shape()[0]; ++n) {
            for (size_t t = 0; t < a.shape()[1]; ++t) {
                for (size_t d = 0; d < a.shape()[2]; ++d) out.at(n, t, d) = a.at(n, t, d);
                for (size_t d = 0; d < b.shape()[2]; ++d) out.at(n, t, a.shape()[2] + d) = b.at(n, t, d);
            }
        }
        return out;
    }

public:
    explicit FromModel(FromConfig cfg = {})
        : cfg_(cfg),
          hawkes_(3, 16),
          kalman_(cfg.feature_dim + 1, cfg.kalman_state_dim),
          conv_(cfg.kalman_state_dim * 2, cfg.conv_out),
          conv_norm_(cfg.conv_out),
          dropout_(cfg.dropout),
          moe_(cfg.conv_out, cfg.moe_experts, cfg.moe_top_k, cfg.moe_expert_dim),
          lstm_(cfg.conv_out, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout),
          ode_(cfg.lstm_hidden, cfg.ode_hidden, cfg.ode_steps),
          hyp_(cfg.lstm_hidden, cfg.hyperbolic_dim),
          film_(cfg.hyperbolic_dim, cfg.lstm_hidden),
          dir_head_(cfg.lstm_hidden, 3),
          quant_head_(cfg.lstm_hidden, 7),
          vol_head_(cfg.lstm_hidden, 1),
          spread_head_(cfg.lstm_hidden, 1),
          noise_head_(cfg.lstm_hidden, 1) {
        for (size_t i = 0; i < cfg.tft_blocks; ++i) {
            tft_.emplace_back(cfg.lstm_hidden, cfg.attention_heads, cfg.attention_d_ff, 100 + i);
        }
    }

    FromOutput forward(const Tensor<float>& x, bool training = true) {
        require(x.shape().size() == 3 && x.shape()[2] == cfg_.feature_dim, "FromModel expects [batch, seq, 22]");
        Tensor<float> hfeat({x.shape()[0], x.shape()[1], 3});
        for (size_t n = 0; n < x.shape()[0]; ++n) {
            for (size_t t = 0; t < x.shape()[1]; ++t) {
                hfeat.at(n, t, 0) = x.at(n, t, 9);
                hfeat.at(n, t, 1) = x.at(n, t, 5);
                hfeat.at(n, t, 2) = x.at(n, t, 13);
            }
        }
        Tensor<float> intensity = hawkes_.forward(hfeat, training);
        Tensor<float> augmented = concat_last_dim(x, intensity);
        Tensor<float> filtered = kalman_.forward(augmented, training);
        Tensor<float> h = conv_norm_.forward(conv_.forward(filtered, training), training);
        h = dropout_.forward(h, training);
        Tensor<float> moe_out = moe_.forward(h, training);
        h = conv_norm_.forward(h + moe_out, training);
        h = lstm_.forward(h, training);
        h = ode_.forward(h, training);
        for (auto& block : tft_) {
            h = block.forward(h, training);
        }
        Tensor<float> last({x.shape()[0], cfg_.lstm_hidden});
        for (size_t n = 0; n < x.shape()[0]; ++n) {
            for (size_t d = 0; d < cfg_.lstm_hidden; ++d) {
                last.at(n, d) = h.at(n, x.shape()[1] - 1, d);
            }
        }
        Tensor<float> emb = hyp_.forward(last, training);
        film_.set_context(emb);
        Tensor<float> conditioned = film_.forward(last, training);
        FromOutput out;
        out.logits_dir = dir_head_.forward(conditioned, training);
        out.logits_q = quant_head_.forward(conditioned, training);
        out.pred_vol = vol_head_.forward(conditioned, training).unary(softplus);
        out.pred_spread = spread_head_.forward(conditioned, training).unary(softplus);
        out.pred_noise = noise_head_.forward(conditioned, training).unary(softplus);
        out.probs_dir = (out.logits_dir / std::max(temperature_, FROM_EPS_F)).softmax(1);
        out.embedding = emb;
        out.ood_score = hyp_.fermi_dirac(emb).mean(1).neg() + 1.0f;
        out.load_balance_loss = moe_.load_balance_loss;
        return out;
    }

    std::vector<ParameterRef> parameters() {
        std::vector<ParameterRef> p;
        auto append = [&](std::vector<ParameterRef> q) { p.insert(p.end(), q.begin(), q.end()); };
        append(hawkes_.parameters());
        append(kalman_.parameters());
        append(conv_.parameters());
        append(conv_norm_.parameters());
        append(moe_.parameters());
        append(lstm_.parameters());
        append(ode_.parameters());
        for (auto& block : tft_) append(block.parameters());
        append(hyp_.parameters());
        append(film_.parameters());
        append(dir_head_.parameters());
        append(quant_head_.parameters());
        append(vol_head_.parameters());
        append(spread_head_.parameters());
        append(noise_head_.parameters());
        return p;
    }

    size_t parameter_count() {
        size_t n = 0;
        for (auto& p : parameters()) {
            n += p.value->numel();
        }
        return n;
    }

    const Conv1D& conv() const { return conv_; }
    const std::vector<TemporalFusionBlock>& tft_blocks() const { return tft_; }
    void set_temperature(float t) { temperature_ = std::max(0.05f, t); }
    float temperature() const { return temperature_; }
};

}  // namespace from

