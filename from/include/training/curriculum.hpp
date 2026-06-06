#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace from {

class CurriculumManager {
    std::vector<float> difficulties_;
    std::vector<size_t> sorted_;

public:
    void set_difficulties(std::vector<float> d) {
        difficulties_ = std::move(d);
        sorted_.resize(difficulties_.size());
        std::iota(sorted_.begin(), sorted_.end(), 0);
        std::sort(sorted_.begin(), sorted_.end(), [&](size_t a, size_t b) { return difficulties_[a] < difficulties_[b]; });
    }

    size_t allowed_count(size_t step) const {
        if (sorted_.empty()) return 0;
        float frac = step < 1000 ? 0.25f : (step < 3000 ? 0.50f : (step < 6000 ? 0.75f : 1.0f));
        return std::max<size_t>(1, static_cast<size_t>(frac * static_cast<float>(sorted_.size())));
    }

    const std::vector<size_t>& sorted_indices() const { return sorted_; }
};

}  // namespace from
