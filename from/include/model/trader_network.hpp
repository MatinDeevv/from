#pragma once

/*
 * TRADER NETWORK - The actual neural net that thinks like a trader
 *
 * Inputs (per sample):
 *   - Raw ticks:          [512, 16]     (last 30 seconds microstructure)
 *   - 1s candles:         [64, 10]      (last minute of price action)
 *   - 1m candles:         [64, 10]      (last hour structure)
 *   - 5m candles:         [64, 10]      (last 5 hours swing)
 *   - 1h candles:         [24, 10]      (last day context)
 *   - Level memory:       [16, 4]       (nearest S/R levels)
 *   - Market context:     [16]          (regime, session, vol, etc)
 *
 * Outputs:
 *   - direction:          [-1, +1]      (continuous, not categorical!)
 *   - magnitude:          [0, ∞)        (expected profit in pips)
 *   - confidence:         [0, 1]        (should we trade or sit out?)
 *   - stop_distance:      [0, ∞)        (where to put stop loss)
 *   - take_profit:        [0, ∞)        (where to take profit)
 *
 * Loss function:
 *   - PROFIT loss: penalize bad trades by actual money lost
 *   - Asymmetric: losing 20 pips is 5x worse than gaining 20 pips is good
 *   - Abstain reward: model gets rewarded for saying "no trade" on noise
 *   - Sharpe loss: maximize risk-adjusted returns, not just returns
 */

#include "model/market_model.hpp"
#include "common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace from {

// Feature dimensions for each input stream
constexpr size_t TICK_SEQ = 512;
constexpr size_t TICK_DIM = 16;
constexpr size_t CANDLE_DIM = 10;  // OHLC + volume + spread + ofi + ticks + vwap + delta
constexpr size_t CANDLE_1S_SEQ = 64;
constexpr size_t CANDLE_1M_SEQ = 64;
constexpr size_t CANDLE_5M_SEQ = 64;
constexpr size_t CANDLE_1H_SEQ = 24;
constexpr size_t LEVEL_COUNT = 16;
constexpr size_t LEVEL_DIM = 4;
constexpr size_t CONTEXT_DIM = 16;

// Model hidden dimensions
constexpr size_t EMBED_DIM = 128;  // Per-stream embedding

struct TraderInput {
    std::vector<float> ticks;       // [512, 16]
    std::vector<float> candles_1s;  // [64, 10]
    std::vector<float> candles_1m;  // [64, 10]
    std::vector<float> candles_5m;  // [64, 10]
    std::vector<float> candles_1h;  // [24, 10]
    std::vector<float> levels;      // [16, 4]
    std::vector<float> context;     // [16]
};

struct TraderOutput {
    float direction;     // -1 to +1 (short to long)
    float magnitude;     // Expected move size (pips)
    float confidence;    // 0 to 1 (abstain to full position)
    float stop_loss;     // Distance to stop (pips)
    float take_profit;   // Distance to TP (pips)
};

// --- PROFIT-BASED LOSS FUNCTION ---
struct TraderLoss {
    // Actual P&L if we traded the model's signal
    static float compute(const TraderOutput& pred, const TradeTarget& target, float spread) {
        float total_loss = 0.0f;

        // 1. PROFIT LOSS — did we make money?
        float position_size = pred.confidence * pred.direction;  // Signed position
        float pnl = position_size * target.profit_after_spread;

        // Asymmetric: losses hurt 3x more than gains feel good
        if (pnl < 0) {
            total_loss += -pnl * 3.0f;  // Penalize losses heavily!
        } else {
            total_loss += -pnl * 1.0f;  // Reward gains (negative loss)
        }

        // 2. CONFIDENCE CALIBRATION — reward abstaining on noise
        if (target.direction == 0) {
            // No clear signal — model should have low confidence
            total_loss += pred.confidence * 2.0f;  // Penalize trading noise
        } else {
            // Clear signal — model should have high confidence
            if (pred.confidence < 0.5f) {
                total_loss += (1.0f - pred.confidence) * 0.5f;  // Mild penalty for missing
            }
        }

        // 3. DIRECTION ACCURACY — basic cross-entropy on direction
        if (target.direction != 0) {
            float target_dir = (target.direction > 0) ? 1.0f : -1.0f;
            float dir_error = (pred.direction - target_dir);
            total_loss += dir_error * dir_error * 0.5f;
        }

        // 4. RISK MANAGEMENT — reward good stop placement
        if (target.direction != 0 && pred.confidence > 0.3f) {
            // Stop should be beyond max adverse excursion
            float stop_error = std::max(0.0f, target.max_adverse - pred.stop_loss);
            total_loss += stop_error * 0.3f;

            // TP should be close to max favorable
            float tp_error = std::abs(pred.take_profit - target.max_favorable);
            total_loss += tp_error * 0.1f;
        }

        return total_loss;
    }

    // SHARPE LOSS — maximize risk-adjusted returns over a batch
    static float sharpe_loss(const std::vector<float>& pnls) {
        if (pnls.size() < 2) return 0.0f;

        float mean = std::accumulate(pnls.begin(), pnls.end(), 0.0f) / pnls.size();
        float variance = 0.0f;
        for (float p : pnls) {
            float d = p - mean;
            variance += d * d;
        }
        variance /= (pnls.size() - 1);
        float std_dev = std::sqrt(variance + 1e-8f);

        // Negative Sharpe = loss (we want to maximize Sharpe)
        return -(mean / std_dev);
    }
};

// --- CANDLE ENCODER ---
// Encodes a sequence of candles into a fixed-size embedding
// Using simple temporal attention (learnable query attends to candle sequence)
struct CandleEncoder {
    // Weights: project candle → hidden, then temporal pool
    std::vector<float> W_proj;   // [CANDLE_DIM, EMBED_DIM]
    std::vector<float> b_proj;   // [EMBED_DIM]
    std::vector<float> W_attn;   // [EMBED_DIM, 1] — attention scores
    size_t seq_len;

    CandleEncoder(size_t seq = 64) : seq_len(seq) {
        W_proj.resize(CANDLE_DIM * EMBED_DIM);
        b_proj.resize(EMBED_DIM, 0.0f);
        W_attn.resize(EMBED_DIM);

        // Xavier init
        float scale = 1.0f / std::sqrt(static_cast<float>(CANDLE_DIM));
        for (auto& w : W_proj) w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
        scale = 1.0f / std::sqrt(static_cast<float>(EMBED_DIM));
        for (auto& w : W_attn) w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    // Forward: [seq_len, CANDLE_DIM] → [EMBED_DIM]
    std::vector<float> forward(const float* candles) const {
        // Project each candle to hidden
        std::vector<float> hidden(seq_len * EMBED_DIM);
        std::vector<float> scores(seq_len);

        for (size_t t = 0; t < seq_len; t++) {
            const float* inp = candles + t * CANDLE_DIM;
            float* h = hidden.data() + t * EMBED_DIM;

            for (size_t d = 0; d < EMBED_DIM; d++) {
                float sum = b_proj[d];
                for (size_t f = 0; f < CANDLE_DIM; f++) {
                    sum += inp[f] * W_proj[f * EMBED_DIM + d];
                }
                h[d] = std::tanh(sum);  // Activation
            }

            // Attention score
            float score = 0.0f;
            for (size_t d = 0; d < EMBED_DIM; d++) {
                score += h[d] * W_attn[d];
            }
            scores[t] = score;
        }

        // Softmax over time
        float max_s = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (auto& s : scores) { s = std::exp(s - max_s); sum_exp += s; }
        for (auto& s : scores) s /= (sum_exp + 1e-8f);

        // Weighted sum of hidden states
        std::vector<float> output(EMBED_DIM, 0.0f);
        for (size_t t = 0; t < seq_len; t++) {
            const float* h = hidden.data() + t * EMBED_DIM;
            for (size_t d = 0; d < EMBED_DIM; d++) {
                output[d] += scores[t] * h[d];
            }
        }

        return output;
    }
};

// --- LEVEL ENCODER ---
// Encodes support/resistance levels relative to current price
struct LevelEncoder {
    std::vector<float> W;  // [LEVEL_DIM, EMBED_DIM / 2]
    size_t out_dim;

    LevelEncoder() : out_dim(EMBED_DIM / 2) {
        W.resize(LEVEL_DIM * out_dim);
        float scale = 1.0f / std::sqrt(static_cast<float>(LEVEL_DIM));
        for (auto& w : W) w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    std::vector<float> forward(const float* levels) const {
        // Max-pool over 16 levels after projection
        std::vector<float> output(out_dim, -1e9f);

        for (size_t l = 0; l < LEVEL_COUNT; l++) {
            const float* inp = levels + l * LEVEL_DIM;
            for (size_t d = 0; d < out_dim; d++) {
                float sum = 0.0f;
                for (size_t f = 0; f < LEVEL_DIM; f++) {
                    sum += inp[f] * W[f * out_dim + d];
                }
                output[d] = std::max(output[d], std::tanh(sum));
            }
        }
        return output;
    }
};

// --- FULL TRADER NETWORK ---
class TraderNetwork {
    CandleEncoder enc_1s_{CANDLE_1S_SEQ};
    CandleEncoder enc_1m_{CANDLE_1M_SEQ};
    CandleEncoder enc_5m_{CANDLE_5M_SEQ};
    CandleEncoder enc_1h_{CANDLE_1H_SEQ};
    LevelEncoder enc_levels_;

    // Fusion layer: concat all embeddings → decision
    // Total input: 4 × EMBED_DIM + EMBED_DIM/2 + CONTEXT_DIM = 592
    static constexpr size_t FUSION_INPUT = 4 * EMBED_DIM + EMBED_DIM / 2 + CONTEXT_DIM;
    static constexpr size_t FUSION_HIDDEN = 256;

    std::vector<float> W1_;  // [FUSION_INPUT, FUSION_HIDDEN]
    std::vector<float> b1_;  // [FUSION_HIDDEN]
    std::vector<float> W2_;  // [FUSION_HIDDEN, 5] (direction, magnitude, confidence, SL, TP)
    std::vector<float> b2_;  // [5]

public:
    TraderNetwork() {
        W1_.resize(FUSION_INPUT * FUSION_HIDDEN);
        b1_.resize(FUSION_HIDDEN, 0.0f);
        W2_.resize(FUSION_HIDDEN * 5);
        b2_.resize(5, 0.0f);

        float scale1 = 1.0f / std::sqrt(static_cast<float>(FUSION_INPUT));
        for (auto& w : W1_) w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale1;
        float scale2 = 1.0f / std::sqrt(static_cast<float>(FUSION_HIDDEN));
        for (auto& w : W2_) w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale2;
    }

    TraderOutput forward(const TraderInput& input) {
        // Encode each timeframe
        auto emb_1s = enc_1s_.forward(input.candles_1s.data());
        auto emb_1m = enc_1m_.forward(input.candles_1m.data());
        auto emb_5m = enc_5m_.forward(input.candles_5m.data());
        auto emb_1h = enc_1h_.forward(input.candles_1h.data());
        auto emb_levels = enc_levels_.forward(input.levels.data());

        // Concatenate all embeddings + context
        std::vector<float> fused(FUSION_INPUT);
        size_t offset = 0;
        auto copy_in = [&](const std::vector<float>& v) {
            std::memcpy(fused.data() + offset, v.data(), v.size() * sizeof(float));
            offset += v.size();
        };
        copy_in(emb_1s);
        copy_in(emb_1m);
        copy_in(emb_5m);
        copy_in(emb_1h);
        copy_in(emb_levels);
        std::memcpy(fused.data() + offset, input.context.data(),
                    std::min(CONTEXT_DIM, input.context.size()) * sizeof(float));

        // Hidden layer with GELU activation
        std::vector<float> hidden(FUSION_HIDDEN);
        for (size_t h = 0; h < FUSION_HIDDEN; h++) {
            float sum = b1_[h];
            for (size_t i = 0; i < FUSION_INPUT; i++) {
                sum += fused[i] * W1_[i * FUSION_HIDDEN + h];
            }
            // GELU approximation
            hidden[h] = 0.5f * sum * (1.0f + std::tanh(0.7978845f * (sum + 0.044715f * sum * sum * sum)));
        }

        // Output layer
        float out[5];
        for (size_t o = 0; o < 5; o++) {
            out[o] = b2_[o];
            for (size_t h = 0; h < FUSION_HIDDEN; h++) {
                out[o] += hidden[h] * W2_[h * 5 + o];
            }
        }

        // Apply output activations
        TraderOutput result;
        result.direction = std::tanh(out[0]);           // -1 to +1
        result.magnitude = std::abs(out[1]) * 10.0f;   // Scale to pips
        result.confidence = 1.0f / (1.0f + std::exp(-out[2])); // Sigmoid 0-1
        result.stop_loss = std::abs(out[3]) * 5.0f + 0.5f;    // Min 0.5 pip SL
        result.take_profit = std::abs(out[4]) * 10.0f + 0.5f;  // Min 0.5 pip TP

        return result;
    }

    size_t param_count() const {
        return W1_.size() + b1_.size() + W2_.size() + b2_.size();
        // + encoder params (not counted here for brevity)
    }
};

}  // namespace from
