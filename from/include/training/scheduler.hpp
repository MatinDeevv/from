#pragma once

#include <cmath>
#include <algorithm>

namespace from {

class LRScheduler {
    float max_lr_;
    float min_lr_;
    size_t warmup_;
    size_t total_;

public:
    LRScheduler(float max_lr = 1e-3f, float min_lr = 1e-5f, size_t warmup = 1000, size_t total = 100000)
        : max_lr_(max_lr), min_lr_(min_lr), warmup_(warmup), total_(total) {}

    float cosine_warmup(size_t step) const {
        if (step < warmup_) {
            return max_lr_ * static_cast<float>(step + 1) / static_cast<float>(std::max<size_t>(1, warmup_));
        }
        float t = static_cast<float>(step - warmup_) / static_cast<float>(std::max<size_t>(1, total_ - warmup_));
        t = std::clamp(t, 0.0f, 1.0f);
        return min_lr_ + 0.5f * (max_lr_ - min_lr_) * (1.0f + std::cos(3.14159265358979323846f * t));
    }

    float one_cycle(size_t step) const {
        float t = static_cast<float>(step) / static_cast<float>(std::max<size_t>(1, total_));
        if (t < 0.3f) return min_lr_ + (max_lr_ - min_lr_) * (t / 0.3f);
        float u = (t - 0.3f) / 0.7f;
        return min_lr_ + 0.5f * (max_lr_ - min_lr_) * (1.0f + std::cos(3.14159265358979323846f * u));
    }
};

class ReduceOnPlateau {
    float lr_;
    float best_ = 1e30f;
    size_t bad_ = 0;
    size_t patience_;

public:
    ReduceOnPlateau(float lr, size_t patience = 100) : lr_(lr), patience_(patience) {}
    float update(float loss) {
        if (loss < best_) {
            best_ = loss;
            bad_ = 0;
        } else if (++bad_ >= patience_) {
            lr_ *= 0.5f;
            bad_ = 0;
        }
        return lr_;
    }
};

}  // namespace from

