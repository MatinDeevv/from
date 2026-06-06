#pragma once

#include "tensor.hpp"

#include <unordered_map>

namespace from {

inline float irm_penalty(const Tensor<float>& logits, const Tensor<float>& targets, const std::vector<int>& env_ids) {
    std::unordered_map<int, float> sums;
    std::unordered_map<int, int> counts;
    for (size_t n = 0; n < logits.shape()[0]; ++n) {
        float grad_w = 0.0f;
        for (size_t c = 0; c < logits.shape()[1]; ++c) {
            grad_w += (logits.at(n, c) - targets.at(n, c)) * logits.at(n, c);
        }
        sums[env_ids[n]] += grad_w * grad_w;
        counts[env_ids[n]] += 1;
    }
    float penalty = 0.0f;
    for (auto& kv : sums) {
        penalty += kv.second / static_cast<float>(std::max(1, counts[kv.first]));
    }
    return penalty;
}

inline float irm_weight(size_t step, size_t warmup_steps, float target = 0.1f) {
    return target * std::min(1.0f, static_cast<float>(step) / static_cast<float>(std::max<size_t>(1, warmup_steps)));
}

}  // namespace from

