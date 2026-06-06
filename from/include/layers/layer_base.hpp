#pragma once

#include "tensor.hpp"

#include <string>
#include <vector>

namespace from {

struct ParameterRef {
    std::string name;
    Tensor<float>* value = nullptr;
    Tensor<float>* grad = nullptr;
    bool hyperbolic = false;
};

class ILayer {
public:
    virtual ~ILayer() = default;
    virtual Tensor<float> forward(const Tensor<float>& x, bool training = true) = 0;
    virtual Tensor<float> backward(const Tensor<float>& grad_out) { return grad_out; }
    virtual std::vector<ParameterRef> parameters() { return {}; }
    virtual std::string name() const = 0;
};

}  // namespace from

