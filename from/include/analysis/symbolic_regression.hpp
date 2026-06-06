#pragma once

#include "tensor.hpp"

#include <sstream>

namespace from {

class SymbolicRegressionHead {
    std::vector<std::string> primitives_{"+", "-", "*", "/", "exp", "log", "sqrt", "abs", "sin", "square", "identity"};
    Tensor<float> weights_;

public:
    explicit SymbolicRegressionHead(size_t feature_dim = FROM_MAX_FEATURES)
        : weights_(Tensor<float>::rand_uniform({feature_dim}, -0.1f, 0.1f, 1)) {}

    Tensor<float> forward(const Tensor<float>& x) const {
        Tensor<float> out({x.shape()[0], 1});
        for (size_t n = 0; n < x.shape()[0]; ++n) {
            float s = 0.0f;
            for (size_t d = 0; d < weights_.shape()[0]; ++d) s += x.at(n, d) * weights_.at(d);
            out.at(n, 0) = s;
        }
        return out;
    }

    std::string extract_formula() const {
        size_t a = 0, b = 1, c = 2;
        for (size_t i = 0; i < weights_.shape()[0]; ++i) {
            if (std::abs(weights_.at(i)) > std::abs(weights_.at(a))) {
                c = b;
                b = a;
                a = i;
            }
        }
        std::ostringstream os;
        os << "s = log((f" << a << " + 1e-4) / (abs(f" << b << ") + 1e-4)) * tanh(f" << c << ")";
        return os.str();
    }
};

}  // namespace from

