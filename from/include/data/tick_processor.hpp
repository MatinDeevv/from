#pragma once

#include "data/parquet_reader.hpp"
#include "tensor.hpp"

#include <array>
#include <deque>

namespace from {

struct FeatureChunk {
    Tensor<float> features;
    std::vector<double> mid;
    std::vector<double> spread;
    std::vector<int64_t> time_ms;
    size_t size = 0;
};

class TickProcessor {
    double prev_mid_ = 0.0;
    double prev_ofi_ = 0.0;
    double prev_velocity_ = 0.0;
    double prev_dmid_ = 0.0;
    int prev_dir_ = 0;
    std::deque<double> returns_;
    std::deque<double> ret_sq_window_16_;
    std::deque<double> ret_sq_window_64_;
    std::deque<double> ret_sq_window_256_;
    std::deque<double> dmid_window_;
    std::deque<int> dir_window_;
    std::deque<double> ofi_window_;
    std::deque<double> micro_delta_window_;
    std::deque<double> mid_delta_window_;
    double ret_sq_sum_16_ = 0.0;
    double ret_sq_sum_64_ = 0.0;
    double ret_sq_sum_256_ = 0.0;
    int dir_sum_32_ = 0;

    // Kyle-Hasbrouck rolling buffers
    std::deque<double> kh_dprice_;
    std::deque<double> kh_svol_;

    // Trade imbalance (feature 22) — 32-tick window
    std::deque<float> buy_vol_32_;
    std::deque<float> sell_vol_32_;
    float buy_sum_32_f_ = 0.0f;
    float sell_sum_32_f_ = 0.0f;

    // Spread compression (feature 24) — 64-tick window
    std::deque<double> ns_window_64_;
    double ns_sum_64_ = 0.0;
    double ns_sq_sum_64_ = 0.0;

    // Autocorrelation (feature 25) — 33-tick return buffer
    std::deque<double> ac_returns_;

    // Volume clock deviation (feature 26) — EMA baseline
    double ema_tick_rate_ = 0.0;
    double ema_tick_sq_ = 0.0;
    static constexpr double EMA_ALPHA = 0.001;

    static double rolling_rv(const std::deque<double>& xs, size_t n);

    // ---- 100-feature expansion state (features 27..126) -------------------
    // Compact per-tick history; all expansion features are computed by a
    // bounded backward scan over this ring (longest window = 1024).
    struct HistRow {
        double ret, absret, mid, ofi, svol, vol, vel, accel;
        double spread, microdev, amihud, kyle, rate;
        float bounce;
        int dir;
    };
    std::deque<HistRow> hist_;
    static constexpr size_t HIST_CAP = 1024;

    // Persistent EMA-of-mid state (spans 16/64/256/1024) — survive chunks.
    double ema_mid_[4] = {0.0, 0.0, 0.0, 0.0};
    bool   ema_init_ = false;

    // Wilder RSI running averages (periods 14/32/64).
    double rsi_ag_[3] = {0.0, 0.0, 0.0};  // avg gain
    double rsi_al_[3] = {0.0, 0.0, 0.0};  // avg loss
    bool   rsi_init_ = false;

    double prev_spread_ = 0.0;
    double prev_accel_  = 0.0;
    int    dir_run_     = 0;   // signed consecutive same-direction tick count

public:
    FeatureChunk process(const TickChunk& chunk);
};

}  // namespace from

