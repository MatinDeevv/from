#pragma once

#include "layers/layer_base.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

namespace from {

class Optimizer {
protected:
    std::vector<ParameterRef> params_;
    float lr_;
    float weight_decay_;
    uint64_t step_ = 0;

public:
    Optimizer(std::vector<ParameterRef> params, float lr = 1e-3f, float weight_decay = 0.0f)
        : params_(std::move(params)), lr_(lr), weight_decay_(weight_decay) {}
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void set_lr(float lr) { lr_ = lr; }
    float lr() const { return lr_; }

    void zero_grad() {
        for (auto& p : params_) {
            if (p.grad) {
                std::fill(p.grad->data_ptr(), p.grad->data_ptr() + p.grad->numel(), 0.0f);
            }
        }
    }

    void clip_grad_norm(float max_norm) {
        float norm2 = 0.0f;
        for (auto& p : params_) {
            if (!p.grad) continue;
            for (size_t i = 0; i < p.grad->numel(); ++i) {
                norm2 += (*p.grad)[i] * (*p.grad)[i];
            }
        }
        float norm = std::sqrt(norm2 + FROM_EPS_F);
        if (norm > max_norm) {
            float scale = max_norm / (norm + FROM_EPS_F);
            for (auto& p : params_) {
                if (!p.grad) continue;
                for (size_t i = 0; i < p.grad->numel(); ++i) (*p.grad)[i] *= scale;
            }
        }
    }
};

class SGD : public Optimizer {
    float momentum_;
    bool nesterov_;
    std::vector<Tensor<float>> velocity_;

public:
    SGD(std::vector<ParameterRef> params, float lr = 1e-3f, float momentum = 0.9f, bool nesterov = false)
        : Optimizer(std::move(params), lr), momentum_(momentum), nesterov_(nesterov) {
        for (auto& p : params_) velocity_.push_back(Tensor<float>::zeros(p.value->shape()));
    }
    void step() override {
        ++step_;
        for (size_t pi = 0; pi < params_.size(); ++pi) {
            auto& p = params_[pi];
            for (size_t i = 0; i < p.value->numel(); ++i) {
                velocity_[pi][i] = momentum_ * velocity_[pi][i] + (*p.grad)[i];
                float g = nesterov_ ? momentum_ * velocity_[pi][i] + (*p.grad)[i] : velocity_[pi][i];
                (*p.value)[i] -= lr_ * g;
            }
        }
    }
};

class AdamW : public Optimizer {
    float beta1_;
    float beta2_;
    float eps_;
    bool decoupled_;
    bool lamb_;
    bool noise_aware_;
    float uncertainty_ = 0.0f;
    std::mt19937 rng_{1};
    std::vector<Tensor<float>> m_;
    std::vector<Tensor<float>> v_;

public:
    AdamW(std::vector<ParameterRef> params, float lr = 1e-3f, float weight_decay = 0.01f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, bool decoupled = true, bool lamb = false)
        : Optimizer(std::move(params), lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps),
          decoupled_(decoupled), lamb_(lamb), noise_aware_(false) {
        for (auto& p : params_) {
            m_.push_back(Tensor<float>::zeros(p.value->shape()));
            v_.push_back(Tensor<float>::zeros(p.value->shape()));
        }
    }

    void set_uncertainty(float u) {
        uncertainty_ = u;
        noise_aware_ = true;
    }

    void step() override {
        ++step_;
        std::normal_distribution<float> noise(0.0f, uncertainty_ * 0.01f);
        for (size_t pi = 0; pi < params_.size(); ++pi) {
            auto& p = params_[pi];
            float theta_norm = 0.0f;
            float update_norm = 0.0f;
            std::vector<float> updates(p.value->numel(), 0.0f);
            for (size_t i = 0; i < p.value->numel(); ++i) {
                float g = (*p.grad)[i];
                if (!decoupled_) g += weight_decay_ * (*p.value)[i];
                if (p.hyperbolic) {
                    float x = (*p.value)[i];
                    g *= ((1.0f - x * x) * 0.5f) * ((1.0f - x * x) * 0.5f);
                }
                m_[pi][i] = beta1_ * m_[pi][i] + (1.0f - beta1_) * g;
                v_[pi][i] = beta2_ * v_[pi][i] + (1.0f - beta2_) * g * g;
                float mh = m_[pi][i] / (1.0f - std::pow(beta1_, static_cast<float>(step_)));
                float vh = v_[pi][i] / (1.0f - std::pow(beta2_, static_cast<float>(step_)));
                float u = mh / (std::sqrt(vh) + eps_);
                if (decoupled_) u += weight_decay_ * (*p.value)[i];
                updates[i] = u;
                theta_norm += (*p.value)[i] * (*p.value)[i];
                update_norm += u * u;
            }
            float trust = 1.0f;
            if (lamb_) {
                trust = std::min(10.0f, std::sqrt(theta_norm)) / (std::sqrt(update_norm) + eps_);
            }
            for (size_t i = 0; i < p.value->numel(); ++i) {
                float n = noise_aware_ ? noise(rng_) : 0.0f;
                (*p.value)[i] -= lr_ * trust * updates[i] + lr_ * n;
                if (p.hyperbolic) {
                    (*p.value)[i] = clamp_value((*p.value)[i], -0.9999f, 0.9999f);
                }
            }
        }
    }
};

using Adam = AdamW;
using LAMB = AdamW;
using RiemannianAdam = AdamW;

}  // namespace from
