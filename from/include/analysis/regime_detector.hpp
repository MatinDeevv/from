#pragma once

#include "tensor.hpp"
#include "utils/math_utils.hpp"

#include <algorithm>
#include <unordered_map>

namespace from {

class RegimeDetector {
    std::unordered_map<int, std::vector<float>> means_;
    std::unordered_map<int, std::vector<float>> inv_diag_cov_;
    float threshold_ = 10.0f;

public:
    void fit(const Tensor<float>& hidden, const std::vector<int>& labels) {
        size_t dim = hidden.shape()[1];
        std::unordered_map<int, int> counts;
        for (size_t n = 0; n < hidden.shape()[0]; ++n) {
            int c = labels[n];
            means_[c].resize(dim, 0.0f);
            inv_diag_cov_[c].resize(dim, 0.0f);
            counts[c] += 1;
            for (size_t d = 0; d < dim; ++d) means_[c][d] += hidden.at(n, d);
        }
        for (auto& kv : means_) {
            for (float& v : kv.second) v /= static_cast<float>(counts[kv.first]);
        }
        for (size_t n = 0; n < hidden.shape()[0]; ++n) {
            int c = labels[n];
            for (size_t d = 0; d < dim; ++d) {
                float e = hidden.at(n, d) - means_[c][d];
                inv_diag_cov_[c][d] += e * e;
            }
        }
        std::vector<float> distances;
        for (auto& kv : inv_diag_cov_) {
            for (float& v : kv.second) v = 1.0f / (v / std::max(1, counts[kv.first] - 1) + 1e-4f);
        }
        for (size_t n = 0; n < hidden.shape()[0]; ++n) distances.push_back(score_one(hidden, n));
        std::sort(distances.begin(), distances.end());
        if (!distances.empty()) threshold_ = distances[static_cast<size_t>(0.99f * static_cast<float>(distances.size() - 1))];
    }

    float score_one(const Tensor<float>& hidden, size_t n) const {
        float best = 1e30f;
        for (const auto& kv : means_) {
            int c = kv.first;
            float m = 0.0f;
            for (size_t d = 0; d < hidden.shape()[1]; ++d) {
                float e = hidden.at(n, d) - kv.second[d];
                m += e * e * inv_diag_cov_.at(c)[d];
            }
            best = std::min(best, std::sqrt(m));
        }
        return best;
    }

    Tensor<float> score(const Tensor<float>& hidden) const {
        Tensor<float> out({hidden.shape()[0], 1});
        for (size_t n = 0; n < hidden.shape()[0]; ++n) out.at(n, 0) = score_one(hidden, n);
        return out;
    }

    bool is_ood(float score) const { return score > threshold_; }
    float threshold() const { return threshold_; }
};

}  // namespace from
