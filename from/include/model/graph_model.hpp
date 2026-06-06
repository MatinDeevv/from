#pragma once

#include "layers/layer_base.hpp"

#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>

namespace from {

struct GraphNode {
    std::string name;
    std::vector<std::string> inputs;
    std::unique_ptr<ILayer> layer;
};

class GraphModel {
    std::vector<GraphNode> nodes_;

public:
    template <class Layer, class... Args>
    void add_node(const std::string& name, std::vector<std::string> inputs, Args&&... args) {
        nodes_.push_back(GraphNode{name, std::move(inputs), std::make_unique<Layer>(std::forward<Args>(args)...)});
    }

    Tensor<float> forward(const std::string& input_name, const Tensor<float>& input, bool training = true) {
        std::unordered_map<std::string, Tensor<float>> values;
        values[input_name] = input;
        for (auto& node : nodes_) {
            require(!node.inputs.empty(), "Graph node requires at least one input");
            Tensor<float> x = values.at(node.inputs[0]);
            for (size_t i = 1; i < node.inputs.size(); ++i) {
                x = x + values.at(node.inputs[i]);
            }
            values[node.name] = node.layer->forward(x, training);
        }
        return nodes_.empty() ? input : values.at(nodes_.back().name);
    }

    std::vector<ParameterRef> parameters() {
        std::vector<ParameterRef> p;
        for (auto& n : nodes_) {
            auto q = n.layer->parameters();
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
};

}  // namespace from
