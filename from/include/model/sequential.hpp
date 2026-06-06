#pragma once

#include "layers/layer_base.hpp"

#include <memory>
#include <utility>

namespace from {

class Sequential {
    std::vector<std::unique_ptr<ILayer>> layers_;

public:
    template <class Layer, class... Args>
    Layer& add(Args&&... args) {
        auto layer = std::make_unique<Layer>(std::forward<Args>(args)...);
        Layer& ref = *layer;
        layers_.push_back(std::move(layer));
        return ref;
    }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) {
        Tensor<float> y = x;
        for (auto& layer : layers_) {
            y = layer->forward(y, training);
        }
        return y;
    }

    Tensor<float> backward(const Tensor<float>& grad) {
        Tensor<float> g = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        return g;
    }

    std::vector<ParameterRef> parameters() {
        std::vector<ParameterRef> p;
        for (auto& layer : layers_) {
            auto q = layer->parameters();
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
};

}  // namespace from
