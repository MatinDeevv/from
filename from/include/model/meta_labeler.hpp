#pragma once
// ============================================================================
// meta_labeler.hpp — Lopez de Prado meta-labeling (AFML ch. 3).
//
// The primary model decides DIRECTION (up / neutral / down). A meta-model then
// decides WHETHER TO ACT on each primary signal: it predicts P(this trade is
// profitable after cost). Filtering by that probability cuts the false signals
// a raw confidence gate lets through, raising precision and Sharpe while leaving
// the primary model untouched. It is trained ONLY on the samples the primary
// model wanted to trade, with the binary target {1 = net>0, 0 = net<=0}.
//
// Implementation: L2-regularized logistic regression with internal feature
// standardization (full-batch gradient descent). Tiny, deterministic, no GPU.
// Header-only.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <vector>

namespace from {

class MetaLabeler {
    size_t dim_ = 0;
    std::vector<double> w_;       // [dim]
    double b_ = 0.0;
    std::vector<double> mean_;    // [dim] feature standardization
    std::vector<double> std_;     // [dim]
    bool trained_ = false;

    static double sigmoid(double z) {
        if (z >= 0.0) { double e = std::exp(-z); return 1.0 / (1.0 + e); }
        double e = std::exp(z); return e / (1.0 + e);
    }

public:
    bool trained() const { return trained_; }
    size_t dim() const { return dim_; }

    // X: row-major [n x dim], y: [n] in {0,1}. l2 = ridge penalty, lr = step.
    void fit(const std::vector<float>& X, const std::vector<uint8_t>& y, size_t dim,
             size_t steps = 600, double lr = 0.1, double l2 = 1e-3) {
        dim_ = dim;
        size_t n = y.size();
        w_.assign(dim_, 0.0);
        b_ = 0.0;
        mean_.assign(dim_, 0.0);
        std_.assign(dim_, 1.0);
        if (n == 0 || dim_ == 0) { trained_ = false; return; }

        // Standardize features (store stats for predict()).
        for (size_t i = 0; i < n; ++i)
            for (size_t d = 0; d < dim_; ++d) mean_[d] += X[i * dim_ + d];
        for (size_t d = 0; d < dim_; ++d) mean_[d] /= static_cast<double>(n);
        for (size_t i = 0; i < n; ++i)
            for (size_t d = 0; d < dim_; ++d) {
                double v = X[i * dim_ + d] - mean_[d];
                std_[d] += v * v;
            }
        for (size_t d = 0; d < dim_; ++d) {
            std_[d] = std::sqrt(std_[d] / static_cast<double>(n) + 1e-8);
            if (std_[d] < 1e-6) std_[d] = 1.0;
        }

        std::vector<double> xs(dim_);
        double inv_n = 1.0 / static_cast<double>(n);
        for (size_t step = 0; step < steps; ++step) {
            std::vector<double> gw(dim_, 0.0);
            double gb = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double z = b_;
                for (size_t d = 0; d < dim_; ++d) {
                    xs[d] = (X[i * dim_ + d] - mean_[d]) / std_[d];
                    z += w_[d] * xs[d];
                }
                double pr = sigmoid(z);
                double err = pr - static_cast<double>(y[i]);
                for (size_t d = 0; d < dim_; ++d) gw[d] += err * xs[d];
                gb += err;
            }
            for (size_t d = 0; d < dim_; ++d)
                w_[d] -= lr * (gw[d] * inv_n + l2 * w_[d]);
            b_ -= lr * gb * inv_n;
        }
        trained_ = true;
    }

    // P(profitable) for a single feature row [dim].
    double predict(const float* x) const {
        if (!trained_) return 0.5;
        double z = b_;
        for (size_t d = 0; d < dim_; ++d)
            z += w_[d] * ((x[d] - mean_[d]) / std_[d]);
        return sigmoid(z);
    }
};

// Build the standard meta feature row from a calibrated primary prediction.
// [p_up, p_neutral, p_down, max_conf, directional_edge, 7x regime one-hot] = 12.
inline constexpr size_t META_FEATURE_DIM = 12;
inline void meta_features(const float probs[3], int env_id, float* out) {
    out[0] = probs[0];
    out[1] = probs[1];
    out[2] = probs[2];
    float mx = probs[0];
    if (probs[1] > mx) mx = probs[1];
    if (probs[2] > mx) mx = probs[2];
    out[3] = mx;                                   // confidence
    out[4] = std::fabs(probs[0] - probs[2]);       // directional edge up vs down
    for (int r = 0; r < 7; ++r) out[5 + r] = 0.0f;
    int e = (env_id >= 0 && env_id < 7) ? env_id : 6;
    out[5 + e] = 1.0f;
}

}  // namespace from
