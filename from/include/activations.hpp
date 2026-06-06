#pragma once

#include "tensor.hpp"
#include "utils/math_utils.hpp"

#include <array>
#include <cmath>

namespace from::act {

inline float relu(float x) { return std::max(0.0f, x); }
inline float relu_grad(float x) { return x > 0.0f ? 1.0f : 0.0f; }

inline float leaky_relu(float x, float alpha = 0.01f) { return x > 0.0f ? x : alpha * x; }
inline float leaky_relu_grad(float x, float alpha = 0.01f) { return x > 0.0f ? 1.0f : alpha; }

inline float prelu(float x, float alpha) { return leaky_relu(x, alpha); }
inline float prelu_grad_x(float x, float alpha) { return leaky_relu_grad(x, alpha); }
inline float prelu_grad_alpha(float x) { return x > 0.0f ? 0.0f : x; }

inline float elu(float x, float alpha = 1.0f) { return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f); }
inline float elu_grad(float x, float alpha = 1.0f) { return x > 0.0f ? 1.0f : elu(x, alpha) + alpha; }

inline float selu(float x) {
    constexpr float lambda = 1.0507009873554804934193349852946f;
    constexpr float alpha = 1.6732632423543772848170429916717f;
    return lambda * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
}
inline float selu_grad(float x) {
    constexpr float lambda = 1.0507009873554804934193349852946f;
    constexpr float alpha = 1.6732632423543772848170429916717f;
    return x > 0.0f ? lambda : lambda * alpha * std::exp(x);
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}
inline float gelu_grad(float x) {
    constexpr float inv_sqrt_2pi = 0.39894228040143267794f;
    return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f))) + x * std::exp(-0.5f * x * x) * inv_sqrt_2pi;
}

inline float silu(float x) {
    float s = sigmoid(x);
    return x * s;
}
inline float silu_grad(float x) {
    float s = sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

inline float mish(float x) {
    return x * std::tanh(softplus(x));
}
inline float mish_grad(float x) {
    float sp = softplus(x);
    float tsp = std::tanh(sp);
    float sig = sigmoid(x);
    return tsp + x * sig * (1.0f - tsp * tsp);
}

inline float tanh_grad_from_x(float x) {
    float y = std::tanh(x);
    return 1.0f - y * y;
}

inline float sigmoid_grad_from_x(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

inline float alrelu(float x, float alpha = 0.1f) { return x > 0.0f ? x : alpha * std::log1p(-x); }
inline float alrelu_grad(float x, float alpha = 0.1f) { return x > 0.0f ? 1.0f : -alpha / (1.0f - x); }

inline float brelu(float x) { return std::min(6.0f, std::max(0.0f, x)); }
inline float brelu_grad(float x) { return (x > 0.0f && x < 6.0f) ? 1.0f : 0.0f; }

inline float soft_clipping(float x, float alpha = 10.0f) {
    return (softplus(alpha * x) - softplus(alpha * (x - 1.0f))) / alpha;
}
inline float soft_clipping_grad(float x, float alpha = 10.0f) {
    return sigmoid(alpha * x) - sigmoid(alpha * (x - 1.0f));
}

struct RationalActivation {
    std::array<float, 4> p{0.0f, 1.0f, 0.0f, 0.0f};
    std::array<float, 4> q{1.0f, 0.0f, 0.0f, 0.0f};

    float forward(float x) const {
        float x2 = x * x;
        float x3 = x2 * x;
        float num = p[0] + p[1] * x + p[2] * x2 + p[3] * x3;
        float den = q[0] + q[1] * x + q[2] * x2 + q[3] * x3;
        return num / (den + FROM_EPS_F);
    }

    float grad_x(float x) const {
        float x2 = x * x;
        float x3 = x2 * x;
        float num = p[0] + p[1] * x + p[2] * x2 + p[3] * x3;
        float den = q[0] + q[1] * x + q[2] * x2 + q[3] * x3;
        float dnum = p[1] + 2.0f * p[2] * x + 3.0f * p[3] * x2;
        float dden = q[1] + 2.0f * q[2] * x + 3.0f * q[3] * x2;
        return (dnum * den - num * dden) / ((den * den) + FROM_EPS_F);
    }
};

inline Tensor<float> apply(const Tensor<float>& x, float (*fn)(float)) {
    return x.unary(fn);
}

}  // namespace from::act

