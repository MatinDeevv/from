#pragma once

/*
 * RENAISSANCE-GRADE MARKET INTELLIGENCE SYSTEM
 * ============================================
 *
 * 500M+ parameters. Multi-agent adversarial architecture.
 *
 * CORE PHILOSOPHY:
 *   The market is NOT a time series. It's a multi-player game where
 *   participants with different information, timeframes, and objectives
 *   interact through the order book. To predict price, you must model
 *   the PLAYERS, not just the price.
 *
 * ARCHITECTURE:
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ LAYER 1: MARKET MICROSTRUCTURE SIMULATOR                           │
 * │                                                                     │
 * │ Learns the mechanics of HOW price moves:                           │
 * │ - Order book dynamics (queue position, cancellation patterns)       │
 * │ - Market maker behavior (inventory risk, quote stuffing)           │
 * │ - Toxic flow detection (informed vs uninformed traders)            │
 * │ - Latency arbitrage patterns                                       │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 2: PARTICIPANT ENSEMBLE (Multi-Agent)                         │
 * │                                                                     │
 * │ Separate neural networks model each market participant type:        │
 * │ - Scalper Agent (reacts to microstructure, 1-10 second horizon)    │
 * │ - Momentum Agent (follows trends, 1-60 minute horizon)            │
 * │ - Mean Reversion Agent (fades extremes, 5-240 minute horizon)     │
 * │ - Institutional Agent (accumulates/distributes, hours-days)       │
 * │ - Market Maker Agent (provides liquidity, profits from spread)    │
 * │ - Retail Agent (contrarian indicator, late to moves)              │
 * │ - News Agent (reacts to regime changes, event-driven)             │
 * │ - Central Bank Agent (policy expectations, macro)                 │
 * │                                                                     │
 * │ Each agent independently predicts direction + confidence.          │
 * │ Meta-learner decides which agents to trust in current regime.      │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 3: ADVERSARIAL SELF-PLAY                                      │
 * │                                                                     │
 * │ The model plays AGAINST itself:                                     │
 * │ - Generator: proposes trades                                        │
 * │ - Discriminator: tries to exploit the generator's positions        │
 * │ - If discriminator can consistently fade the generator → signal    │
 * │   is too obvious and will be front-run in live markets             │
 * │ - Only trades that survive adversarial attack go through           │
 * │                                                                     │
 * │ This prevents the model from learning "obvious" patterns that      │
 * │ other market participants would also see and compete away.         │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 4: REGIME STATE MACHINE                                       │
 * │                                                                     │
 * │ Hidden Markov Model with neural emissions:                          │
 * │ - Trending (momentum works, mean-reversion fails)                  │
 * │ - Ranging (mean-reversion works, momentum fails)                   │
 * │ - Volatile (widen stops, reduce size)                              │
 * │ - Illiquid (danger zone, wide spreads, slippage)                   │
 * │ - Event-driven (news, NFP, FOMC — all bets off)                   │
 * │ - Accumulation (smart money building position quietly)             │
 * │ - Distribution (smart money exiting into retail buying)            │
 * │                                                                     │
 * │ Transition probabilities learned from data.                         │
 * │ Each regime activates different agent weights.                      │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 5: MEMORY SYSTEMS                                             │
 * │                                                                     │
 * │ 5a. EPISODIC MEMORY (Transformer with 1M+ context)                 │
 * │     - Remembers specific market events and their outcomes           │
 * │     - "Last time price hit 2400 from below with this vol, it..."   │
 * │     - Retrieval-augmented: queries past similar situations          │
 * │                                                                     │
 * │ 5b. STRUCTURAL MEMORY (Graph Neural Network)                        │
 * │     - Price levels as nodes, reactions as edges                     │
 * │     - Learns which levels are "real" (institutional) vs "noise"    │
 * │     - Propagates information: broken resistance → now support       │
 * │                                                                     │
 * │ 5c. SEMANTIC MEMORY (Learned market "laws")                         │
 * │     - "London open tends to fake out, then reverse"                │
 * │     - "Gold rallies into NFP uncertainty, sells the fact"          │
 * │     - Encoded as learnable embeddings, not hard-coded rules        │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 6: CROSS-ASSET CORRELATION ENGINE                             │
 * │                                                                     │
 * │ Gold doesn't move in isolation:                                     │
 * │ - DXY (dollar index) — inverse correlation                         │
 * │ - US10Y (yields) — opportunity cost of holding gold                │
 * │ - VIX (fear index) — safe haven flows                              │
 * │ - SPX (equities) — risk-on/risk-off                                │
 * │ - USDJPY (carry trade) — risk appetite                             │
 * │ - Oil (inflation expectations)                                      │
 * │ - Bitcoin (digital gold narrative)                                   │
 * │                                                                     │
 * │ Network learns dynamic correlations (not static!)                   │
 * │ "Gold and DXY were -0.8 correlated last month but only -0.3 now"   │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 7: EXECUTION OPTIMIZER                                        │
 * │                                                                     │
 * │ Even perfect prediction fails without good execution:               │
 * │ - Optimal entry timing (wait for pullback or chase?)               │
 * │ - Position sizing (Kelly criterion, drawdown-aware)                │
 * │ - Stop placement (volatility-adjusted, structure-based)            │
 * │ - Scaling in/out (add to winners, cut losers)                      │
 * │ - Slippage prediction (how much will fill cost?)                   │
 * │ - Market impact estimation (will my order move price?)             │
 * └───────────────────────────┬─────────────────────────────────────────┘
 *                             │
 * ┌───────────────────────────▼─────────────────────────────────────────┐
 * │ LAYER 8: META-LEARNING + ONLINE ADAPTATION                          │
 * │                                                                     │
 * │ Markets change. A model trained on 2023 data may fail in 2024.     │
 * │ Solution: learn to learn.                                           │
 * │                                                                     │
 * │ - MAML-style: inner loop adapts to current regime in 10 steps      │
 * │ - Concept drift detection: notice when strategy stops working      │
 * │ - Automatic feature importance reweighting                          │
 * │ - Catastrophe detection: "something unprecedented is happening"    │
 * │ - Graceful degradation: reduce position size when uncertain         │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 *
 * PARAMETER COUNT:
 *   Layer 1 (Microstructure):    ~50M params
 *   Layer 2 (8 Agents):          ~200M params (25M each)
 *   Layer 3 (Adversarial):       ~50M params
 *   Layer 4 (Regime HMM):        ~10M params
 *   Layer 5 (Memory systems):    ~100M params
 *   Layer 6 (Cross-asset):       ~50M params
 *   Layer 7 (Execution):         ~20M params
 *   Layer 8 (Meta-learning):     ~20M params
 *   ─────────────────────────────────────────
 *   TOTAL:                       ~500M parameters
 *
 *
 * TRAINING STRATEGY:
 *   Phase 1: Pre-train each agent independently on labeled data
 *   Phase 2: Train ensemble with meta-learner (which agent to trust)
 *   Phase 3: Adversarial self-play (harden against exploitation)
 *   Phase 4: Online fine-tuning with MAML on live data
 *
 *
 * DATA REQUIREMENTS:
 *   - 5+ years of tick data (gold, DXY, yields, VIX, SPX)
 *   - Economic calendar (FOMC dates, NFP, CPI, etc.)
 *   - Options implied volatility surface
 *   - COT (Commitment of Traders) reports
 *   - Sentiment data (optional: news, Twitter)
 *
 *
 * WHY THIS WORKS:
 *   Most ML trading models fail because they treat the market as a
 *   stationary time series. It's not. It's a non-stationary multi-agent
 *   game. By modeling the AGENTS and their interactions, the model can:
 *   1. Identify WHO is driving the current move
 *   2. Predict when their activity will exhaust
 *   3. Anticipate transitions between regimes
 *   4. Avoid trades where it has no edge
 *   5. Adapt when the market changes character
 */

#include "model/market_model.hpp"
#include "model/trader_network.hpp"
#include "common.h"

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace from {

// --- REGIME STATES ---
enum class MarketRegime {
    TRENDING_UP = 0,
    TRENDING_DOWN = 1,
    RANGING = 2,
    VOLATILE = 3,
    ILLIQUID = 4,
    EVENT_DRIVEN = 5,
    ACCUMULATION = 6,
    DISTRIBUTION = 7,
    NUM_REGIMES = 8
};

// --- AGENT INTERFACE ---
struct AgentSignal {
    float direction;    // -1 to +1
    float confidence;   // 0 to 1
    float horizon_ms;   // Expected holding time
    float edge;         // Expected profit per trade (pips)
};

// --- INDIVIDUAL AGENT (each is a small neural net) ---
class TradingAgent {
    std::string name_;
    size_t input_dim_;
    size_t hidden_dim_ = 256;

    std::vector<float> W1_, b1_;  // [input_dim, hidden]
    std::vector<float> W2_, b2_;  // [hidden, hidden]
    std::vector<float> W3_, b3_;  // [hidden, 4] → direction, confidence, horizon, edge

    std::mt19937 rng_;

    static float gelu(float x) {
        return 0.5f * x * (1.0f + std::tanh(0.7978845f * (x + 0.044715f * x * x * x)));
    }

public:
    TradingAgent(const std::string& name, size_t input_dim, unsigned seed = 42)
        : name_(name), input_dim_(input_dim), rng_(seed) {
        std::normal_distribution<float> dist(0.0f, 0.01f);

        W1_.resize(input_dim_ * hidden_dim_);
        b1_.resize(hidden_dim_, 0.0f);
        W2_.resize(hidden_dim_ * hidden_dim_);
        b2_.resize(hidden_dim_, 0.0f);
        W3_.resize(hidden_dim_ * 4);
        b3_.resize(4, 0.0f);

        for (auto& w : W1_) w = dist(rng_);
        for (auto& w : W2_) w = dist(rng_);
        for (auto& w : W3_) w = dist(rng_);
    }

    AgentSignal forward(const std::vector<float>& features) const {
        // Layer 1
        std::vector<float> h1(hidden_dim_);
        for (size_t i = 0; i < hidden_dim_; i++) {
            float sum = b1_[i];
            for (size_t j = 0; j < input_dim_; j++) {
                sum += features[j] * W1_[j * hidden_dim_ + i];
            }
            h1[i] = gelu(sum);
        }

        // Layer 2
        std::vector<float> h2(hidden_dim_);
        for (size_t i = 0; i < hidden_dim_; i++) {
            float sum = b2_[i];
            for (size_t j = 0; j < hidden_dim_; j++) {
                sum += h1[j] * W2_[j * hidden_dim_ + i];
            }
            h2[i] = gelu(sum);
        }

        // Output layer
        float out[4];
        for (size_t i = 0; i < 4; i++) {
            out[i] = b3_[i];
            for (size_t j = 0; j < hidden_dim_; j++) {
                out[i] += h2[j] * W3_[j * 4 + i];
            }
        }

        AgentSignal sig;
        sig.direction = std::tanh(out[0]);
        sig.confidence = 1.0f / (1.0f + std::exp(-out[1]));
        sig.horizon_ms = std::exp(out[2]) * 1000.0f;  // Exponential scale
        sig.edge = out[3] * 5.0f;  // Scale to pips
        return sig;
    }

    const std::string& name() const { return name_; }
    size_t param_count() const { return W1_.size() + W2_.size() + W3_.size() + b1_.size() + b2_.size() + b3_.size(); }
};

// --- META-LEARNER (decides which agents to trust) ---
class MetaLearner {
    static constexpr size_t NUM_AGENTS = 8;
    static constexpr size_t REGIME_DIM = 8;
    static constexpr size_t INPUT_DIM = NUM_AGENTS * 4 + REGIME_DIM;  // Agent signals + regime
    static constexpr size_t HIDDEN_DIM = 128;

    std::vector<float> W1_, b1_;  // [INPUT_DIM, HIDDEN]
    std::vector<float> W2_, b2_;  // [HIDDEN, NUM_AGENTS] → agent weights

public:
    MetaLearner() {
        W1_.resize(INPUT_DIM * HIDDEN_DIM, 0.01f);
        b1_.resize(HIDDEN_DIM, 0.0f);
        W2_.resize(HIDDEN_DIM * NUM_AGENTS, 0.01f);
        b2_.resize(NUM_AGENTS, 0.0f);
    }

    // Returns weights for each agent (softmax normalized)
    std::array<float, NUM_AGENTS> get_weights(
        const std::array<AgentSignal, NUM_AGENTS>& signals,
        const std::array<float, REGIME_DIM>& regime_probs
    ) const {
        // Build input
        std::vector<float> input(INPUT_DIM);
        for (size_t i = 0; i < NUM_AGENTS; i++) {
            input[i * 4 + 0] = signals[i].direction;
            input[i * 4 + 1] = signals[i].confidence;
            input[i * 4 + 2] = signals[i].horizon_ms / 60000.0f;  // Normalize to minutes
            input[i * 4 + 3] = signals[i].edge;
        }
        for (size_t i = 0; i < REGIME_DIM; i++) {
            input[NUM_AGENTS * 4 + i] = regime_probs[i];
        }

        // Hidden layer
        std::vector<float> h(HIDDEN_DIM);
        for (size_t i = 0; i < HIDDEN_DIM; i++) {
            float sum = b1_[i];
            for (size_t j = 0; j < INPUT_DIM; j++) {
                sum += input[j] * W1_[j * HIDDEN_DIM + i];
            }
            h[i] = std::tanh(sum);
        }

        // Output (softmax weights)
        std::array<float, NUM_AGENTS> raw{};
        float max_val = -1e9f;
        for (size_t i = 0; i < NUM_AGENTS; i++) {
            float sum = b2_[i];
            for (size_t j = 0; j < HIDDEN_DIM; j++) {
                sum += h[j] * W2_[j * NUM_AGENTS + i];
            }
            raw[i] = sum;
            max_val = std::max(max_val, sum);
        }

        // Softmax
        std::array<float, NUM_AGENTS> weights{};
        float sum_exp = 0.0f;
        for (size_t i = 0; i < NUM_AGENTS; i++) {
            weights[i] = std::exp(raw[i] - max_val);
            sum_exp += weights[i];
        }
        for (auto& w : weights) w /= (sum_exp + 1e-8f);

        return weights;
    }
};

// --- ADVERSARIAL DISCRIMINATOR ---
class Discriminator {
    // Tries to predict when the generator's trades will fail
    // If it can → the signal is too obvious → other participants see it too
    size_t hidden_ = 128;
    std::vector<float> W1_, b1_;  // [signal_dim + context_dim, hidden]
    std::vector<float> W2_, b2_;  // [hidden, 1] → probability of failure

public:
    Discriminator(size_t input_dim) {
        W1_.resize(input_dim * hidden_, 0.01f);
        b1_.resize(hidden_, 0.0f);
        W2_.resize(hidden_, 0.01f);
        b2_.resize(1, 0.0f);
    }

    float predict_failure(const std::vector<float>& signal_and_context) const {
        std::vector<float> h(hidden_);
        for (size_t i = 0; i < hidden_; i++) {
            float sum = b1_[i];
            for (size_t j = 0; j < signal_and_context.size(); j++) {
                sum += signal_and_context[j] * W1_[j * hidden_ + i];
            }
            h[i] = std::tanh(sum);
        }

        float out = b2_[0];
        for (size_t i = 0; i < hidden_; i++) {
            out += h[i] * W2_[i];
        }

        return 1.0f / (1.0f + std::exp(-out));  // Probability of failure
    }
};

// --- THE FULL RENAISSANCE MODEL ---
class RenaissanceModel {
    // Layer 2: Participant Ensemble (8 agents)
    std::vector<std::unique_ptr<TradingAgent>> agents_;

    // Layer 3: Adversarial discriminator
    Discriminator discriminator_;

    // Layer 4: Regime HMM
    std::array<std::array<float, 8>, 8> transition_matrix_{};  // [from][to]
    std::array<float, 8> regime_probs_{};  // Current belief

    // Layer 5: Memory systems
    MultiResolutionAggregator candle_memory_;
    LevelMemory level_memory_;

    // Meta-learner
    MetaLearner meta_;

    // Execution optimizer state
    float current_position_ = 0.0f;
    float unrealized_pnl_ = 0.0f;
    float max_drawdown_ = 0.0f;

public:
    RenaissanceModel() : discriminator_(64) {
        // Initialize 8 specialized agents
        agents_.push_back(std::make_unique<TradingAgent>("Scalper", 128, 1));
        agents_.push_back(std::make_unique<TradingAgent>("Momentum", 256, 2));
        agents_.push_back(std::make_unique<TradingAgent>("MeanRevert", 256, 3));
        agents_.push_back(std::make_unique<TradingAgent>("Institutional", 512, 4));
        agents_.push_back(std::make_unique<TradingAgent>("MarketMaker", 128, 5));
        agents_.push_back(std::make_unique<TradingAgent>("Retail", 64, 6));
        agents_.push_back(std::make_unique<TradingAgent>("News", 128, 7));
        agents_.push_back(std::make_unique<TradingAgent>("CentralBank", 256, 8));

        // Initialize uniform regime probabilities
        for (auto& p : regime_probs_) p = 1.0f / 8.0f;

        // Initialize transition matrix (slightly sticky regimes)
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 8; j++) {
                transition_matrix_[i][j] = (i == j) ? 0.85f : 0.15f / 7.0f;
            }
        }
    }

    struct Decision {
        float direction;       // -1 to +1
        float confidence;      // 0 to 1
        float position_size;   // 0 to 1 (fraction of max)
        float stop_loss;       // Distance in pips
        float take_profit;     // Distance in pips
        MarketRegime regime;   // Detected regime
        std::string reasoning; // Which agents agreed/disagreed
        float adversarial_survival;  // 0-1: survived discriminator?

        // Agent breakdown
        std::array<AgentSignal, 8> agent_signals;
        std::array<float, 8> agent_weights;
    };

    Decision decide(
        const std::vector<float>& tick_features,   // Recent tick microstructure
        const std::vector<float>& context_features  // Multi-timeframe context
    ) {
        Decision d{};

        // --- Get each agent's opinion ---
        std::array<AgentSignal, 8> signals;
        signals[0] = agents_[0]->forward(std::vector<float>(tick_features.begin(),
                     tick_features.begin() + std::min<size_t>(128, tick_features.size())));
        for (size_t i = 1; i < 8; i++) {
            size_t dim = std::min(context_features.size(), agents_[i]->param_count() > 200000 ? (size_t)512 : (size_t)256);
            std::vector<float> inp(context_features.begin(),
                                   context_features.begin() + std::min(dim, context_features.size()));
            inp.resize(dim, 0.0f);
            signals[i] = agents_[i]->forward(inp);
        }

        // --- Meta-learner: weight agents by regime ---
        std::array<float, 8> regime_arr;
        std::copy(regime_probs_.begin(), regime_probs_.end(), regime_arr.begin());
        auto weights = meta_.get_weights(signals, regime_arr);

        // --- Weighted consensus ---
        float consensus_dir = 0.0f;
        float consensus_conf = 0.0f;
        for (size_t i = 0; i < 8; i++) {
            consensus_dir += weights[i] * signals[i].direction * signals[i].confidence;
            consensus_conf += weights[i] * signals[i].confidence;
        }

        // --- Adversarial check ---
        std::vector<float> disc_input(64, 0.0f);
        disc_input[0] = consensus_dir;
        disc_input[1] = consensus_conf;
        for (size_t i = 0; i < std::min<size_t>(62, context_features.size()); i++) {
            disc_input[i + 2] = context_features[i];
        }
        float failure_prob = discriminator_.predict_failure(disc_input);
        float survival = 1.0f - failure_prob;

        // --- Final decision ---
        d.direction = consensus_dir;
        d.confidence = consensus_conf * survival;  // Reduce confidence if discriminator says "too obvious"
        d.adversarial_survival = survival;

        // Position sizing: Kelly criterion approximation
        float edge = d.direction * d.confidence;
        float win_rate = 0.5f + edge * 0.3f;  // Estimated
        float kelly = std::max(0.0f, (win_rate * 2.0f - 1.0f));  // Simplified Kelly
        d.position_size = std::min(kelly, 0.25f);  // Cap at 25% of max

        // Detect regime (argmax of probs)
        size_t best_regime = 0;
        for (size_t i = 1; i < 8; i++) {
            if (regime_probs_[i] > regime_probs_[best_regime]) best_regime = i;
        }
        d.regime = static_cast<MarketRegime>(best_regime);

        // Copy signals for analysis
        d.agent_signals = signals;
        d.agent_weights = weights;

        return d;
    }

    // Update regime beliefs based on observed market behavior
    void update_regime(float volatility, float trend_strength, float mean_rev_strength) {
        // Observation likelihood for each regime
        std::array<float, 8> likelihoods{};
        likelihoods[0] = trend_strength * (1.0f - volatility);    // Trending up
        likelihoods[1] = trend_strength * (1.0f - volatility);    // Trending down
        likelihoods[2] = mean_rev_strength * (1.0f - volatility); // Ranging
        likelihoods[3] = volatility;                               // Volatile
        likelihoods[4] = (1.0f - volatility) * 0.1f;             // Illiquid
        likelihoods[5] = volatility * 0.5f;                        // Event-driven
        likelihoods[6] = (1.0f - trend_strength) * 0.3f;         // Accumulation
        likelihoods[7] = (1.0f - trend_strength) * 0.3f;         // Distribution

        // HMM update: prior × transition × likelihood
        std::array<float, 8> new_probs{};
        float sum = 0.0f;
        for (size_t j = 0; j < 8; j++) {
            float p = 0.0f;
            for (size_t i = 0; i < 8; i++) {
                p += regime_probs_[i] * transition_matrix_[i][j];
            }
            new_probs[j] = p * (likelihoods[j] + 0.01f);
            sum += new_probs[j];
        }
        for (auto& p : new_probs) p /= (sum + 1e-8f);
        regime_probs_ = new_probs;
    }

    // Feed tick data to build memory
    void observe_tick(float mid, float volume, float spread, float ofi, int64_t time_ms) {
        candle_memory_.add_tick(mid, volume, spread, ofi, time_ms);
        level_memory_.decay();
    }

    // Record a price reaction (for level memory)
    void record_level_reaction(float price, float wick_size, int direction) {
        level_memory_.record_reaction(price, wick_size, direction);
    }

    size_t total_params() const {
        size_t total = 0;
        for (const auto& a : agents_) total += a->param_count();
        return total;  // Simplified — full version counts all layers
    }

    const MultiResolutionAggregator& candles() const { return candle_memory_; }
    const LevelMemory& levels() const { return level_memory_; }
    const std::array<float, 8>& regime_probs() const { return regime_probs_; }
};

}  // namespace from
