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

    static double rolling_rv(const std::deque<double>& xs, size_t n);

public:
    FeatureChunk process(const TickChunk& chunk);
};

}  // namespace from

