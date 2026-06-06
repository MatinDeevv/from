#pragma once

#include "data/tick_processor.hpp"

#include <array>
#include <cmath>
#include <deque>
#include <vector>

namespace from {

struct Sample {
    Tensor<float> X;
    std::vector<float> summary;  // pre-computed [176] if available
    std::array<float, 3> y_dir{0.0f, 1.0f, 0.0f};
    float y_mag = 0.0f;
    float y_vol = 0.0f;
    float y_spread = 0.0f;
    int env_id = 0;
    float difficulty = 0.0f;
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
                if (delta > threshold) {
                    s.y_dir = {1.0f, 0.0f, 0.0f};
                } else if (delta < -threshold) {
                    s.y_dir = {0.0f, 0.0f, 1.0f};
                }
                s.y_mag = static_cast<float>(std::abs(delta) / (mean_spread + FROM_EPS_D));
                s.y_vol = static_cast<float>(future_rv);
                s.y_spread = static_cast<float>(mean_spread);
                double mean_ret = std::abs(delta) / static_cast<double>(horizon_);
                s.difficulty = static_cast<float>(future_rv / (mean_ret + FROM_EPS_D));
                s.env_id = environment_for(window_, future_rv, s.difficulty);
                samples.push_back(std::move(s));
            }
            features_.pop_front();
            mid_.pop_front();
            spread_.pop_front();
            time_.pop_front();
            since_last_ = (since_last_ + 1) % stride_;
        }
        return samples;
    }
};

}  // namespace from
