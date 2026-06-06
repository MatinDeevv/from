#pragma once

#include "tensor.hpp"
#include "utils/math_utils.hpp"

#include <array>

namespace from {

struct LossBreakdown {
    float total = 0.0f;
    float direction = 0.0f;
    float quantile = 0.0f;
    float volatility = 0.0f;
    float spread = 0.0f;
    float hawkes = 0.0f;
    float contrastive = 0.0f;
    float irm = 0.0f;
    float load = 0.0f;
};

inline float directional_cross_entropy(const Tensor<float>& logits, const Tensor<float>& target, float catastrophe_mult = 3.0f) {
    Tensor<float> probs = logits.softmax(1);
    float loss = 0.0f;
    const float weights[3] = {2.0f, 0.5f, 2.0f};
    for (size_t n = 0; n < logits.shape()[0]; ++n) {
        size_t pred = 0, truth = 0;
        for (size_t c = 1; c < 3; ++c) {
            if (probs.at(n, c) > probs.at(n, pred)) pred = c;
            if (target.at(n, c) > target.at(n, truth)) truth = c;
        }
        float mult = ((pred == 0 && truth == 2) || (pred == 2 && truth == 0)) ? catastrophe_mult : 1.0f;
        for (size_t c = 0; c < 3; ++c) {
            loss -= mult * weights[c] * target.at(n, c) * std::log(probs.at(n, c) + FROM_EPS_F);
        }
    }
    return loss / static_cast<float>(logits.shape()[0]);
}

inline float quantile_loss(const Tensor<float>& pred, const Tensor<float>& y, const std::vector<float>& qs) {
    float loss = 0.0f;
    for (size_t n = 0; n < pred.shape()[0]; ++n) {
        for (size_t q = 0; q < pred.shape()[1]; ++q) {
            float e = y.at(n, 0) - pred.at(n, q);
            loss += std::max(qs[q] * e, (qs[q] - 1.0f) * e);
        }
    }
    return loss / static_cast<float>(pred.shape()[0] * pred.shape()[1]);
}

inline float mse_loss(const Tensor<float>& pred, const Tensor<float>& y) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.numel(); ++i) {
        float d = pred[i] - y[i % y.numel()];
        loss += d * d;
    }
    return loss / static_cast<float>(std::max<size_t>(1, pred.numel()));
}

inline float huber_loss(const Tensor<float>& pred, const Tensor<float>& y, float delta = 0.5f) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.numel(); ++i) {
        float d = std::abs(pred[i] - y[i % y.numel()]);
        loss += d <= delta ? 0.5f * d * d : delta * (d - 0.5f * delta);
    }
    return loss / static_cast<float>(std::max<size_t>(1, pred.numel()));
}

inline float triplet_margin_loss(const Tensor<float>& anchor, const Tensor<float>& positive, const Tensor<float>& negative, float margin = 1.0f) {
    float loss = 0.0f;
    for (size_t n = 0; n < anchor.shape()[0]; ++n) {
        float dp = 0.0f, dn = 0.0f;
        for (size_t d = 0; d < anchor.shape()[1]; ++d) {
            float ap = anchor.at(n, d) - positive.at(n, d);
            float an = anchor.at(n, d) - negative.at(n, d);
            dp += ap * ap;
            dn += an * an;
        }
        loss += std::max(0.0f, std::sqrt(dp) - std::sqrt(dn) + margin);
    }
    return loss / static_cast<float>(std::max<size_t>(1, anchor.shape()[0]));
}

}  // namespace from

