#pragma once

#include "common.h"

#include <cmath>
#include <vector>

namespace from {

inline std::vector<std::vector<float>> pcgrad(std::vector<std::vector<float>> grads) {
    for (size_t i = 0; i < grads.size(); ++i) {
        for (size_t j = 0; j < grads.size(); ++j) {
            if (i == j) continue;
            float dot = 0.0f, norm = 0.0f;
            for (size_t k = 0; k < grads[i].size(); ++k) {
                dot += grads[i][k] * grads[j][k];
                norm += grads[j][k] * grads[j][k];
            }
            if (dot < 0.0f) {
                float scale = dot / (norm + FROM_EPS_F);
                for (size_t k = 0; k < grads[i].size(); ++k) {
                    grads[i][k] -= scale * grads[j][k];
                }
            }
        }
    }
    return grads;
}

}  // namespace from
