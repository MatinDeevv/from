#pragma once

#include "data/tick_processor.hpp"

#include <array>
#include <cmath>
#include <deque>
#include <vector>

namespace from {

struct Sample {
    Tensor<float> X;
    std::vector<float> summary;  // pre-computed [SEQ_SUMMARY_DIM] if available
    std::array<float, 3> y_dir{0.0f, 1.0f, 0.0f};
    float y_mag = 0.0f;
    float y_vol = 0.0f;
    float y_spread = 0.0f;
    double entry_mid = 0.0;   // mid price at window end (trade entry)
    double exit_mid  = 0.0;   // mid price at horizon end (trade exit)
    double entry_spread = 0.0; // spread at trade entry
    long long entry_time_ms = 0; // wall-clock ms at trade entry (for session/day strategy knobs)
    int env_id = 0;
    float difficulty = 0.0f;

    // --- Medallion-Lite Phase 1: triple-barrier labels (additive; do not affect y_dir) ---
    int       tb_label     = 1;     // triple-barrier class 0=UP / 1=NEUTRAL / 2=DOWN
    float     tb_ret       = 0.0f;  // signed realized return entry->first-touch = (touch_mid-entry_mid)/entry_mid
    float     tb_cost      = 0.0f;  // cost in return units = entry_spread/entry_mid (one full spread ~ round trip)
    float     uniqueness_w = 1.0f;  // sample weight (P1: leave 1.0; true uniqueness is a later phase)
    long long t_index      = 0;     // GLOBAL entry tick index (monotonic across chunks) for purge/embargo
};

class Windower {
    size_t window_;
    size_t stride_;
    size_t horizon_;
    float threshold_mult_;
    std::deque<std::array<float, FROM_MAX_FEATURES>> features_;
    std::deque<double> mid_;
    std::deque<double> spread_;
    std::deque<int64_t> time_;
    size_t since_last_ = 0;

    // Medallion-Lite Phase 1: global tick index = absolute index of the tick currently at
    // the FRONT of the deque (incremented on every pop_front). When a sample is emitted, the
    // window starts at absolute index consumed_.
    size_t consumed_ = 0;
    // Triple-barrier multipliers (per-tick sigma scaled to a price offset). Default 1.0;
    // configurable later without breaking existing callers.
    double k_tp_ = 1.0;
    double k_sl_ = 1.0;
    // Minimum barrier in round-trip-spread units: a labeled move must clear ~this many
    // spreads of cost. Keeps labels economically meaningful (see triple-barrier block).
    double cost_mult_ = 1.5;

    int environment_for(size_t idx, double rv, float difficulty) const {
        int64_t ms_day = 24LL * 60LL * 60LL * 1000LL;
        int64_t t = ((time_[idx] % ms_day) + ms_day) % ms_day;
        int hour = static_cast<int>(t / (60LL * 60LL * 1000LL));
        if (hour >= 7 && hour < 12) return 0;
        if (hour >= 13 && hour < 17) return 1;
        if (hour >= 23 || hour < 4) return 2;
        if (rv > 0.001) return 3;
        if (rv < 0.0001) return 4;
        if (difficulty < 2.0f) return 5;
        return 6;
    }

public:
    Windower(size_t window_size = 512, size_t stride = 64, size_t horizon = 128, float threshold_mult = 0.5f)
        : window_(window_size), stride_(stride), horizon_(horizon), threshold_mult_(threshold_mult) {}

    // Tune the triple-barrier: k = barrier in horizon-vol units, cost_mult = minimum
    // barrier in round-trip-spread units. Lets the walk-forward sweep explore cost-aware
    // label definitions without rebuilding.
    void set_barriers(double k_tp, double k_sl, double cost_mult) {
        k_tp_ = k_tp; k_sl_ = k_sl; cost_mult_ = cost_mult;
    }

    std::vector<Sample> add(const FeatureChunk& chunk) {
        for (size_t i = 0; i < chunk.size; ++i) {
            std::array<float, FROM_MAX_FEATURES> row{};
            for (size_t d = 0; d < FROM_MAX_FEATURES; ++d) {
                row[d] = chunk.features.at(i, d);
            }
            features_.push_back(row);
            mid_.push_back(chunk.mid[i]);
            spread_.push_back(chunk.spread[i]);
            time_.push_back(chunk.time_ms[i]);
        }

        std::vector<Sample> samples;
        while (features_.size() >= window_ + horizon_) {
            if (since_last_ == 0) {
                Sample s;
                s.X = Tensor<float>({window_, FROM_MAX_FEATURES});
                for (size_t t = 0; t < window_; ++t) {
                    for (size_t d = 0; d < FROM_MAX_FEATURES; ++d) {
                        s.X.at(t, d) = features_[t][d];
                    }
                }
                double mean_spread = 0.0;
                double future_rv = 0.0;
                for (size_t t = window_; t < window_ + horizon_; ++t) {
                    mean_spread += spread_[t];
                    double r = std::log((mid_[t] + FROM_EPS_D) / (mid_[t - 1] + FROM_EPS_D));
                    future_rv += r * r;
                }
                mean_spread /= static_cast<double>(horizon_);
                future_rv = std::sqrt(future_rv);
                double delta = mid_[window_ + horizon_ - 1] - mid_[window_];
                double threshold = threshold_mult_ * mean_spread;

                // Store entry/exit mid for real PnL calculation
                s.entry_mid = mid_[window_];
                s.exit_mid  = mid_[window_ + horizon_ - 1];
                s.entry_spread = spread_[window_];
                s.entry_time_ms = static_cast<long long>(time_[window_]);

                // Label smoothing: confidence scales with distance from threshold
                float delta_f = static_cast<float>(delta);
                float thresh_f = static_cast<float>(threshold);
                float margin = std::abs(delta_f) - thresh_f;
                float confidence = std::min(1.0f, 0.5f + margin / (thresh_f + 1e-6f) * 0.5f);
                float residual = (1.0f - confidence) / 2.0f;

                if (delta > threshold) {
                    s.y_dir = {confidence, residual, residual};
                } else if (delta < -threshold) {
                    s.y_dir = {residual, residual, confidence};
                } else {
                    float neutral_conf = 1.0f - std::abs(delta_f) / (thresh_f + 1e-6f) * 0.3f;
                    neutral_conf = std::max(0.4f, neutral_conf);
                    s.y_dir = {(1.0f - neutral_conf) / 2.0f, neutral_conf, (1.0f - neutral_conf) / 2.0f};
                }
                s.y_mag = static_cast<float>(std::abs(delta) / (mean_spread + FROM_EPS_D));
                s.y_vol = static_cast<float>(future_rv);
                s.y_spread = static_cast<float>(mean_spread);
                double mean_ret = std::abs(delta) / static_cast<double>(horizon_);
                s.difficulty = static_cast<float>(future_rv / (mean_ret + FROM_EPS_D));
                s.env_id = environment_for(window_, future_rv, s.difficulty);

                // --- Triple-barrier labeling (no lookahead) ----------------------------
                // Entry tick absolute index = window-start absolute index.
                s.t_index = static_cast<long long>(consumed_);

                // entry_sigma: per-tick realized vol from IN-WINDOW returns ONLY (1..window_-1).
                // Uses only past+window data => no lookahead.
                double var = 0.0;
                for (size_t t = 1; t < window_; ++t) {
                    double r = std::log((mid_[t] + FROM_EPS_D) / (mid_[t - 1] + FROM_EPS_D));
                    var += r * r;
                }
                double sigma = (window_ > 1) ? std::sqrt(var / static_cast<double>(window_ - 1)) : 0.0;

                // Barriers must be (a) scaled to the prediction HORIZON — a tradeable move is
                // ~sigma*sqrt(horizon), not one tick — and (b) at least cost_mult_ round-trip
                // spreads, so an UP/DOWN label always marks a move that CLEARS the cost to
                // capture it. Without the horizon scale + cost floor, labels mark sub-cost
                // noise and every trade loses ~the spread (winrate -> 0).
                double sigma_h   = sigma * std::sqrt(static_cast<double>(horizon_));
                double cost_floor = cost_mult_ * mean_spread;  // price units, ~round-trip cost
                double tp = std::max(s.entry_mid * sigma_h * k_tp_, cost_floor);
                double sl = std::max(s.entry_mid * sigma_h * k_sl_, cost_floor);
                if (!(tp > 0.0)) tp = (mean_spread > 0.0 ? mean_spread : 1e-6);
                if (!(sl > 0.0)) sl = (mean_spread > 0.0 ? mean_spread : 1e-6);

                // Scan FUTURE path [window_, window_+horizon_) for first barrier touch.
                double touch_mid = mid_[window_ + horizon_ - 1];  // default: vertical (time) barrier
                int tb_label = 1;  // NEUTRAL until a horizontal barrier is touched
                for (size_t t = window_; t < window_ + horizon_; ++t) {
                    double p = mid_[t];
                    if (p >= s.entry_mid + tp) { tb_label = 0; touch_mid = p; break; }  // UP
                    if (p <= s.entry_mid - sl) { tb_label = 2; touch_mid = p; break; }  // DOWN
                }

                s.tb_label     = tb_label;
                s.exit_mid     = touch_mid;  // override horizon-end exit so PnL uses true exit
                s.tb_ret       = static_cast<float>((touch_mid - s.entry_mid) / (s.entry_mid + FROM_EPS_D));
                s.tb_cost      = static_cast<float>(s.entry_spread / (s.entry_mid + FROM_EPS_D));
                s.uniqueness_w = 1.0f;
                // -----------------------------------------------------------------------

                samples.push_back(std::move(s));
            }
            features_.pop_front();
            mid_.pop_front();
            spread_.pop_front();
            time_.pop_front();
            ++consumed_;  // front tick advanced by one absolute index
            since_last_ = (since_last_ + 1) % stride_;
        }
        return samples;
    }
};

}  // namespace from
