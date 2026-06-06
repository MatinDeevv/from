#pragma once

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
namespace from {

class TemperatureCalibrator {
    float temperature_ = 1.0f;

public:
    void fit(const Tensor<float>& logits, const Tensor<float>& target, size_t steps = 200, float lr = 0.01f) {
        float t = temperature_;
        for (size_t step = 0; step < steps; ++step) {
            float grad = 0.0f;
            Tensor<float> probs = (logits / t).softmax(1);
            for (size_t n = 0; n < logits.shape()[0]; ++n) {
                for (size_t c = 0; c < logits.shape()[1]; ++c) {
                    grad += (target.at(n, c) - probs.at(n, c)) * logits.at(n, c) / (t * t + FROM_EPS_F);
                }
            }
            t = std::max(0.05f, t - lr * grad / static_cast<float>(logits.shape()[0]));
        }
        temperature_ = t;
    }

    Tensor<float> apply(const Tensor<float>& logits) const {
        return (logits / temperature_).softmax(1);
    }

    float expected_calibration_error(const Tensor<float>& probs, const Tensor<float>& target, size_t bins = 10) const {
        std::vector<float> conf_sum(bins, 0.0f), acc_sum(bins, 0.0f), count(bins, 0.0f);
        for (size_t n = 0; n < probs.shape()[0]; ++n) {
            size_t pred = 0, truth = 0;
            for (size_t c = 1; c < probs.shape()[1]; ++c) {
                if (probs.at(n, c) > probs.at(n, pred)) pred = c;
                if (target.at(n, c) > target.at(n, truth)) truth = c;
            }
            float conf = probs.at(n, pred);
            size_t bin = std::min(bins - 1, static_cast<size_t>(conf * static_cast<float>(bins)));
            conf_sum[bin] += conf;
            acc_sum[bin] += pred == truth ? 1.0f : 0.0f;
            count[bin] += 1.0f;
        }
        float ece = 0.0f;
        float total = static_cast<float>(probs.shape()[0]);
        for (size_t b = 0; b < bins; ++b) {
            if (count[b] > 0.0f) {
                ece += (count[b] / total) * std::abs(conf_sum[b] / count[b] - acc_sum[b] / count[b]);
            }
        }
        return ece;
    }

    float temperature() const { return temperature_; }
};

}  // namespace from
