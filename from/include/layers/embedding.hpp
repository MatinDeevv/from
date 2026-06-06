#pragma once

#include "initializers.hpp"
#include "layers/layer_base.hpp"

namespace from {

class Embedding : public ILayer {
    size_t num_embeddings_;
    size_t dim_;

public:
    Tensor<float> table;
    Tensor<float> grad_table;

    Embedding(size_t num_embeddings = 8, size_t dim = 16, uint64_t seed = 1)
        : num_embeddings_(num_embeddings),
          dim_(dim),
          table(init::xavier_uniform({num_embeddings, dim}, num_embeddings, dim, seed)),
          grad_table(Tensor<float>::zeros({num_embeddings, dim})) {}

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        Tensor<float> out({x.numel(), dim_});
        for (size_t i = 0; i < x.numel(); ++i) {
            size_t idx = static_cast<size_t>(std::abs(x[i])) % num_embeddings_;
            for (size_t d = 0; d < dim_; ++d) {
                out.at(i, d) = table.at(idx, d);
            }
        }
        return out;
    }

    std::vector<ParameterRef> parameters() override { return {{"table", &table, &grad_table, false}}; }
    std::string name() const override { return "embedding"; }
};

}  // namespace from

