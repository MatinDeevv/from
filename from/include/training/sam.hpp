#pragma once

#include "common.h"
#include "layers/layer_base.hpp"

#include <cmath>

namespace from {

class SAM {
    std::vector<ParameterRef> params_;
    std::vector<Tensor<float>> perturb_;
    float rho_;
    bool adaptive_;

public:
    SAM(std::vector<ParameterRef> params, float rho = 0.05f, bool adaptive = false)
        : params_(std::move(params)), rho_(rho), adaptive_(adaptive) {
        for (auto& p : params_) perturb_.push_back(Tensor<float>::zeros(p.value->shape()));
    }

    void first_step() {
        float norm2 = 0.0f;
        for (auto& p : params_) {
            for (size_t i = 0; i < p.value->numel(); ++i) {
                float g = (*p.grad)[i] * (adaptive_ ? std::abs((*p.value)[i]) : 1.0f);
                norm2 += g * g;
            }
        }
        float scale = rho_ / (std::sqrt(norm2) + FROM_EPS_F);
        for (size_t pi = 0; pi < params_.size(); ++pi) {
            auto& p = params_[pi];
            for (size_t i = 0; i < p.value->numel(); ++i) {
                float e = (*p.grad)[i] * (adaptive_ ? std::abs((*p.value)[i]) : 1.0f) * scale;
                perturb_[pi][i] = e;
                (*p.value)[i] += e;
            }
        }
    }

    void second_step() {
        for (size_t pi = 0; pi < params_.size(); ++pi) {
            auto& p = params_[pi];
            for (size_t i = 0; i < p.value->numel(); ++i) {
                (*p.value)[i] -= perturb_[pi][i];
            }
        }
    }
};

}  // namespace from
