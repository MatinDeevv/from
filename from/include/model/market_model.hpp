#pragma once

/*
 * MARKET MODEL - Understands gold like a real trader
 *
 * Architecture:
 *   1. Multi-Resolution Encoder (tick → 1s → 1m → 5m → 1h → 4h → 1d)
 *   2. Level Memory (remembers where price reacted — support/resistance)
 *   3. Order Flow Analyzer (who is aggressive, where is liquidity)
 *   4. Regime Detector (trending/ranging/volatile/session)
 *   5. Confidence Gate (abstains when uncertain)
 *   6. Profit Predictor (spread-adjusted, risk/reward aware)
 *
 * Lookback: 1 month compressed into hierarchical memory
 *   - Raw ticks: 512 most recent (~30 seconds)
 *   - 1-second bars: 3600 (last hour)
 *   - 1-minute bars: 1440 (last day)
 *   - 5-minute bars: 2016 (last week)
 *   - 1-hour bars: 720 (last month)
 *   - Price levels: 128 levels where price reacted strongly
 */

#include "common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

namespace from {

// --- CANDLE STRUCTURE ---
struct Candle {
    float open = 0, high = 0, low = 0, close = 0;
    float volume = 0;
    float spread_avg = 0;
    float ofi_sum = 0;       // Net order flow imbalance
    float tick_count = 0;    // How many ticks in this candle
    float vwap = 0;          // Volume-weighted avg price
    float delta = 0;         // close - open
    int64_t time_ms = 0;
};

// --- PRICE LEVEL (support/resistance memory) ---
struct PriceLevel {
    float price = 0;
    float strength = 0;      // How many times price reacted here
    float last_touch = 0;    // How many bars since last touch
    float rejection_size = 0; // Average rejection wick size
    int type = 0;            // 0=support, 1=resistance, 2=both
};

// --- MULTI-RESOLUTION AGGREGATOR ---
// Compresses raw ticks into candles at multiple timeframes
class MultiResolutionAggregator {
    // Ring buffers for each timeframe
    std::deque<Candle> candles_1s_;    // 3600 = 1 hour
    std::deque<Candle> candles_1m_;    // 1440 = 1 day
    std::deque<Candle> candles_5m_;    // 2016 = 1 week
    std::deque<Candle> candles_1h_;    // 720  = 1 month

    // Current building candle for each timeframe
    Candle building_1s_, building_1m_, building_5m_, building_1h_;
    int64_t start_1s_ = 0, start_1m_ = 0, start_5m_ = 0, start_1h_ = 0;
    bool started_ = false;

    static const size_t MAX_1S = 3600;
    static const size_t MAX_1M = 1440;
    static const size_t MAX_5M = 2016;
    static const size_t MAX_1H = 720;

    void push_candle(std::deque<Candle>& buf, const Candle& c, size_t max_size) {
        buf.push_back(c);
        while (buf.size() > max_size) buf.pop_front();
    }

    void close_1s(int64_t time) {
        building_1s_.time_ms = time;
        push_candle(candles_1s_, building_1s_, MAX_1S);

        // Aggregate into 1m
        if (start_1m_ == 0) start_1m_ = time;
        update_candle(building_1m_, building_1s_.close, building_1s_.volume,
                      building_1s_.spread_avg, building_1s_.ofi_sum);
        building_1m_.high = std::max(building_1m_.high, building_1s_.high);
        building_1m_.low = std::min(building_1m_.low, building_1s_.low);

        if (time - start_1m_ >= 60000) {
            close_1m(time);
            start_1m_ = time;
            building_1m_ = {};
        }

        // Reset
        building_1s_ = {};
        start_1s_ = time;
    }

    void close_1m(int64_t time) {
        building_1m_.time_ms = time;
        building_1m_.delta = building_1m_.close - building_1m_.open;
        push_candle(candles_1m_, building_1m_, MAX_1M);

        // Aggregate into 5m
        if (start_5m_ == 0) start_5m_ = time;
        update_candle(building_5m_, building_1m_.close, building_1m_.volume,
                      building_1m_.spread_avg, building_1m_.ofi_sum);
        building_5m_.high = std::max(building_5m_.high, building_1m_.high);
        building_5m_.low = std::min(building_5m_.low, building_1m_.low);

        if (time - start_5m_ >= 300000) {
            close_5m(time);
            start_5m_ = time;
            building_5m_ = {};
        }
    }

    void close_5m(int64_t time) {
        building_5m_.time_ms = time;
        building_5m_.delta = building_5m_.close - building_5m_.open;
        push_candle(candles_5m_, building_5m_, MAX_5M);

        // Aggregate into 1h
        if (start_1h_ == 0) start_1h_ = time;
        update_candle(building_1h_, building_5m_.close, building_5m_.volume,
                      building_5m_.spread_avg, building_5m_.ofi_sum);
        building_1h_.high = std::max(building_1h_.high, building_5m_.high);
        building_1h_.low = std::min(building_1h_.low, building_5m_.low);

        if (time - start_1h_ >= 3600000) {
            building_1h_.time_ms = time;
            building_1h_.delta = building_1h_.close - building_1h_.open;
            push_candle(candles_1h_, building_1h_, MAX_1H);
            start_1h_ = time;
            building_1h_ = {};
        }
    }

    void update_candle(Candle& c, float price, float vol, float spread, float ofi) {
        if (c.open == 0) { c.open = price; c.high = price; c.low = price; }
        c.close = price;
        c.high = std::max(c.high, price);
        c.low = std::min(c.low, price);
        c.volume += vol;
        c.spread_avg += spread;
        c.ofi_sum += ofi;
        c.tick_count += 1.0f;
    }

public:
    void add_tick(float mid, float volume, float spread, float ofi, int64_t time_ms) {
        if (!started_) {
            start_1s_ = time_ms;
            start_1m_ = time_ms;
            start_5m_ = time_ms;
            start_1h_ = time_ms;
            started_ = true;
        }

        // Build 1-second candle
        update_candle(building_1s_, mid, volume, spread, ofi);

        // Close when second boundary crossed
        if (time_ms - start_1s_ >= 1000) {
            close_1s(time_ms);
        }
    }

    const std::deque<Candle>& candles_1s() const { return candles_1s_; }
    const std::deque<Candle>& candles_1m() const { return candles_1m_; }
    const std::deque<Candle>& candles_5m() const { return candles_5m_; }
    const std::deque<Candle>& candles_1h() const { return candles_1h_; }
};

// --- LEVEL MEMORY ---
// Remembers where price reacted strongly (support/resistance)
class LevelMemory {
    std::vector<PriceLevel> levels_;
    static const size_t MAX_LEVELS = 128;
    float price_tolerance_ = 0.5f;  // Pips tolerance for "same level"

public:
    void record_reaction(float price, float rejection_size, int direction) {
        // Check if level already exists
        for (auto& lvl : levels_) {
            if (std::abs(lvl.price - price) < price_tolerance_) {
                lvl.strength += 1.0f;
                lvl.last_touch = 0.0f;
                lvl.rejection_size = 0.9f * lvl.rejection_size + 0.1f * rejection_size;
                if (direction > 0 && lvl.type == 1) lvl.type = 2;  // Both
                if (direction < 0 && lvl.type == 0) lvl.type = 2;
                return;
            }
        }

        // New level
        PriceLevel lvl;
        lvl.price = price;
        lvl.strength = 1.0f;
        lvl.last_touch = 0.0f;
        lvl.rejection_size = rejection_size;
        lvl.type = (direction > 0) ? 0 : 1;  // Support or resistance

        if (levels_.size() < MAX_LEVELS) {
            levels_.push_back(lvl);
        } else {
            // Replace weakest level
            size_t weakest = 0;
            for (size_t i = 1; i < levels_.size(); i++) {
                if (levels_[i].strength < levels_[weakest].strength) weakest = i;
            }
            levels_[weakest] = lvl;
        }
    }

    void decay() {
        // Age all levels (strength decays, recency increases)
        for (auto& lvl : levels_) {
            lvl.last_touch += 1.0f;
            lvl.strength *= 0.999f;  // Slow decay
        }
    }

    // Find nearest support/resistance relative to current price
    void get_context(float current_price, float* nearest_support, float* nearest_resistance,
                     float* support_strength, float* resistance_strength) const {
        *nearest_support = current_price - 100.0f;  // Default far away
        *nearest_resistance = current_price + 100.0f;
        *support_strength = 0.0f;
        *resistance_strength = 0.0f;

        for (const auto& lvl : levels_) {
            if (lvl.price < current_price && (lvl.type == 0 || lvl.type == 2)) {
                if (lvl.price > *nearest_support) {
                    *nearest_support = lvl.price;
                    *support_strength = lvl.strength;
                }
            }
            if (lvl.price > current_price && (lvl.type == 1 || lvl.type == 2)) {
                if (lvl.price < *nearest_resistance) {
                    *nearest_resistance = lvl.price;
                    *resistance_strength = lvl.strength;
                }
            }
        }
    }

    // Encode all levels as features relative to current price
    void encode(float current_price, float* output, size_t max_levels = 16) const {
        // Sort by distance to current price
        std::vector<std::pair<float, const PriceLevel*>> sorted;
        for (const auto& lvl : levels_) {
            sorted.push_back({std::abs(lvl.price - current_price), &lvl});
        }
        std::sort(sorted.begin(), sorted.end());

        // Output: [distance, strength, type, recency] × max_levels
        for (size_t i = 0; i < max_levels; i++) {
            float* out = output + i * 4;
            if (i < sorted.size()) {
                out[0] = (sorted[i].second->price - current_price);  // Signed distance
                out[1] = sorted[i].second->strength;
                out[2] = static_cast<float>(sorted[i].second->type);
                out[3] = 1.0f / (sorted[i].second->last_touch + 1.0f);  // Recency
            } else {
                out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
            }
        }
    }

    const std::vector<PriceLevel>& levels() const { return levels_; }
};

// --- PROFIT TARGET (spread-adjusted) ---
struct TradeTarget {
    float profit_after_spread;  // (move - spread) × direction. Negative = loss.
    float risk_reward;          // |profit| / |max_adverse_excursion|
    float max_favorable;        // Best price seen in horizon
    float max_adverse;          // Worst price seen in horizon
    float confidence;           // 1.0 = clear signal, 0.0 = noise
    int direction;              // +1 = long, -1 = short, 0 = no trade
};

TradeTarget compute_trade_target(
    const double* mid, const double* spread,
    size_t start, size_t horizon, float min_profit_pips
) {
    TradeTarget t{};
    double entry = mid[start];
    double entry_spread = spread[start];
    double exit_price = mid[start + horizon - 1];

    double move = exit_price - entry;
    double half_spread = entry_spread * 0.5;

    // Max favorable/adverse excursion
    double max_fav_long = 0, max_adv_long = 0;
    double max_fav_short = 0, max_adv_short = 0;
    for (size_t i = 1; i < horizon; i++) {
        double diff = mid[start + i] - entry;
        max_fav_long = std::max(max_fav_long, diff);
        max_adv_long = std::max(max_adv_long, -diff);
        max_fav_short = std::max(max_fav_short, -diff);
        max_adv_short = std::max(max_adv_short, diff);
    }

    // Long trade profit
    double long_profit = move - half_spread;
    double short_profit = -move - half_spread;

    if (long_profit > short_profit && long_profit > min_profit_pips) {
        t.direction = 1;
        t.profit_after_spread = static_cast<float>(long_profit);
        t.max_favorable = static_cast<float>(max_fav_long);
        t.max_adverse = static_cast<float>(max_adv_long);
    } else if (short_profit > long_profit && short_profit > min_profit_pips) {
        t.direction = -1;
        t.profit_after_spread = static_cast<float>(short_profit);
        t.max_favorable = static_cast<float>(max_fav_short);
        t.max_adverse = static_cast<float>(max_adv_short);
    } else {
        t.direction = 0;
        t.profit_after_spread = 0.0f;
        t.confidence = 0.0f;
        return t;
    }

    // Risk/reward
    t.risk_reward = (t.max_adverse > 0.0001f)
        ? t.max_favorable / t.max_adverse
        : 10.0f;

    // Confidence: high when move is large relative to noise
    float noise = static_cast<float>(max_fav_long + max_adv_long) * 0.5f;
    t.confidence = std::min(1.0f, std::abs(t.profit_after_spread) / (noise + 0.01f));

    return t;
}

// --- MARKET CONTEXT FEATURES ---
// What a real trader looks at before entering
struct MarketContext {
    // Multi-timeframe trend
    float trend_1m = 0;     // -1 to +1
    float trend_5m = 0;
    float trend_1h = 0;

    // Volatility regime
    float volatility = 0;   // Current realized vol
    float vol_percentile = 0; // Where vol is relative to recent history

    // Session
    float session_progress = 0; // 0-1 within current session
    int session_id = 0;         // 0=asia, 1=london, 2=ny

    // Support/resistance
    float dist_to_support = 0;
    float dist_to_resistance = 0;
    float support_strength = 0;
    float resistance_strength = 0;

    // Order flow
    float cum_delta = 0;        // Cumulative buy-sell imbalance
    float absorption_ratio = 0; // Passive vs aggressive flow

    // Spread regime
    float spread_percentile = 0; // Is spread wide or narrow?
};

}  // namespace from
