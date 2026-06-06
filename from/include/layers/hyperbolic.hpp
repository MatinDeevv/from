#pragma once

#include "layers/linear.hpp"

#include <cmath>
#include <vector>
namespace from {

class PoincareBall : public ILayer {
    size_t in_dim_;
    size_t out_dim_;
    float c_;
    Linear tangent_;

    static float dot(const std::vector<float>& a, const std::vector<float>& b) {
        float s = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }

    static float norm(const std::vector<float>& a) {
        return std::sqrt(dot(a, a) + FROM_EPS_F);
    }

public:
    Tensor<float> centroids;
    Tensor<float> grad_centroids;

    PoincareBall(size_t in_dim = 128, size_t out_dim = 64, float curvature = 1.0f, uint64_t seed = 1)
        : in_dim_(in_dim), out_dim_(out_dim), c_(curvature), tangent_(in_dim, out_dim, seed),
          centroids(Tensor<float>::rand_uniform({7, out_dim}, -0.01f, 0.01f, seed + 10)),
          grad_centroids(Tensor<float>::zeros({7, out_dim})) {}

    std::vector<float> project(std::vector<float> x) const {
        float n = norm(x);
        float max_n = 0.9999f / std::sqrt(c_);
        if (n > max_n) {
            float scale = max_n / (n + FROM_EPS_F);
            for (float& v : x) v *= scale;
        }
        return x;
    }

    std::vector<float> mobius_add(const std::vector<float>& x, const std::vector<float>& y) const {
        float xy = dot(x, y);
        float x2 = dot(x, x);
        float y2 = dot(y, y);
        float den = 1.0f + 2.0f * c_ * xy + c_ * c_ * x2 * y2 + FROM_EPS_F;
        std::vector<float> out(x.size(), 0.0f);
        for (size_t i = 0; i < x.size(); ++i) {
            out[i] = ((1.0f + 2.0f * c_ * xy + c_ * y2) * x[i] + (1.0f - c_ * x2) * y[i]) / den;
        }
        return project(out);
    }

    std::vector<float> exp0(const std::vector<float>& v) const {
        float nv = norm(v);
        float scale = std::tanh(std::sqrt(c_) * nv) / (std::sqrt(c_) * nv + FROM_EPS_F);
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = scale * v[i];
        return project(out);
    }

    std::vector<float> log0(const std::vector<float>& y) const {
        float ny = std::min(0.9999f / std::sqrt(c_), norm(y));
        float scale = std::atanh(std::sqrt(c_) * ny) / (std::sqrt(c_) * ny + FROM_EPS_F);
        std::vector<float> out(y.size());
        for (size_t i = 0; i < y.size(); ++i) out[i] = scale * y[i];
        return out;
    }

    float distance(const std::vector<float>& x, const std::vector<float>& y) const {
        std::vector<float> negx(x.size());
        for (size_t i = 0; i < x.size(); ++i) negx[i] = -x[i];
        std::vector<float> z = mobius_add(negx, y);
        float nz = std::min(0.9999f / std::sqrt(c_), norm(z));
        return 2.0f / std::sqrt(c_) * std::atanh(std::sqrt(c_) * nz);
    }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        Tensor<float> y = tangent_.forward(x, training);
        Tensor<float> out(y.shape());
        for (size_t n = 0; n < y.shape()[0]; ++n) {
            std::vector<float> v(out_dim_);
            for (size_t d = 0; d < out_dim_; ++d) v[d] = y.at(n, d);
            v = exp0(v);
            for (size_t d = 0; d < out_dim_; ++d) out.at(n, d) = v[d];
        }
        return out;
    }

    Tensor<float> fermi_dirac(const Tensor<float>& x, float r = 1.0f, float t = 0.5f) const {
        Tensor<float> out({x.shape()[0], centroids.shape()[0]});
        for (size_t n = 0; n < x.shape()[0]; ++n) {
            std::vector<float> a(out_dim_);
            for (size_t d = 0; d < out_dim_; ++d) a[d] = x.at(n, d);
            for (size_t cidx = 0; cidx < centroids.shape()[0]; ++cidx) {
                std::vector<float> b(out_dim_);
                for (size_t d = 0; d < out_dim_; ++d) b[d] = centroids.at(cidx, d);
                float dist = distance(a, b);
                out.at(n, cidx) = 1.0f / (std::exp((dist - r) / (t + FROM_EPS_F)) + 1.0f);
            }
        }
        return out;
    }

    std::vector<ParameterRef> parameters() override {
        auto p = tangent_.parameters();
        p.push_back({"centroids", &centroids, &grad_centroids, true});
        return p;
    }

    std::string name() const override { return "hyperbolic"; }
};

}  // namespace from
