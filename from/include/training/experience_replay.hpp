#pragma once

#include "common.h"
#include "data/windower.hpp"

#include <deque>
#include <vector>

namespace from {

class ExperienceReplay {
    size_t capacity_;
    size_t anchor_capacity_;
    std::deque<Sample> buffer_;
    std::vector<Sample> anchors_;

public:
    ExperienceReplay(size_t capacity = 2000, size_t anchor_capacity = 200)
        : capacity_(capacity), anchor_capacity_(anchor_capacity) {}

    void add(const Sample& sample) {
        if (buffer_.size() >= capacity_) buffer_.pop_front();
        buffer_.push_back(sample);
        if (anchors_.size() < anchor_capacity_ || sample.y_mag > anchors_[anchors_.size() % anchor_capacity_].y_mag) {
            if (anchors_.size() < anchor_capacity_) anchors_.push_back(sample);
            else anchors_[anchors_.size() % anchor_capacity_] = sample;
        }
    }

    const std::deque<Sample>& buffer() const { return buffer_; }
    const std::vector<Sample>& anchors() const { return anchors_; }
};

inline std::vector<float> gem_project(std::vector<float> grad_new, const std::vector<float>& grad_anchor) {
    float dot = 0.0f, norm = 0.0f;
    for (size_t i = 0; i < grad_new.size(); ++i) {
        dot += grad_new[i] * grad_anchor[i];
        norm += grad_anchor[i] * grad_anchor[i];
    }
    if (dot < 0.0f) {
        float scale = dot / (norm + FROM_EPS_F);
        for (size_t i = 0; i < grad_new.size(); ++i) grad_new[i] -= scale * grad_anchor[i];
    }
    return grad_new;
}

}  // namespace from
