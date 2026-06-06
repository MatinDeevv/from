#pragma once

#include "tensor.hpp"

#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>

namespace from {

class Normalizer {
    std::vector<double> mean_;
    std::vector<double> m2_;
    std::vector<uint64_t> count_;
    bool frozen_ = false;
    double eps_ = 1e-8;

public:
    explicit Normalizer(size_t dims = FROM_MAX_FEATURES)
        : mean_(dims, 0.0), m2_(dims, 0.0), count_(dims, 0) {}

    void update_one(const float* x) {
        if (frozen_) {
            return;
        }
        for (size_t d = 0; d < mean_.size(); ++d) {
            count_[d] += 1;
            double delta = static_cast<double>(x[d]) - mean_[d];
            mean_[d] += delta / static_cast<double>(count_[d]);
            double delta2 = static_cast<double>(x[d]) - mean_[d];
            m2_[d] += delta * delta2;
        }
    }

    void normalize_one(float* x, bool update = true) {
        if (update) {
            update_one(x);
        }
        for (size_t d = 0; d < mean_.size(); ++d) {
            double var = count_[d] > 1 ? m2_[d] / static_cast<double>(count_[d] - 1) : 1.0;
            x[d] = static_cast<float>((static_cast<double>(x[d]) - mean_[d]) / std::sqrt(var + eps_));
        }
    }

    Tensor<float> normalize_chunk(const Tensor<float>& x, bool update = true) {
        Tensor<float> out = x.contiguous();
        size_t dims = mean_.size();
        for (size_t i = 0; i < out.numel(); i += dims) {
            normalize_one(out.data_ptr() + i, update);
        }
        return out;
    }

    void freeze() { frozen_ = true; }
    void unfreeze() { frozen_ = false; }
    bool frozen() const { return frozen_; }
    const std::vector<double>& mean() const { return mean_; }
    const std::vector<double>& m2() const { return m2_; }
    const std::vector<uint64_t>& count() const { return count_; }

    void set_state(std::vector<double> mean, std::vector<double> m2, std::vector<uint64_t> count, bool frozen) {
        mean_ = std::move(mean);
        m2_ = std::move(m2);
        count_ = std::move(count);
        frozen_ = frozen;
    }
};

}  // namespace from
