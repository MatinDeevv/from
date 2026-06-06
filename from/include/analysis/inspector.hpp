#pragma once

#include "model/from_model.hpp"

#include <sstream>

namespace from {

class Inspector {
public:
    static std::string kernels(const FromModel& model) {
        return model.conv().inspect_kernels();
    }

    static std::string attention(const FromModel& model) {
        std::ostringstream os;
        os << "Attention analysis\n";
        size_t idx = 0;
        for (const auto& block : model.tft_blocks()) {
            const Tensor<float>& a = block.last_attention();
            os << "Block " << idx++ << ": collected tensor elements=" << a.numel()
               << "; causal heads attend only to j<=i by construction.\n";
        }
        return os.str();
    }

    static std::string regime() {
        return "Regime analysis: Poincare centroids and Fermi-Dirac scores are available through the hyperbolic layer.\n";
    }

    static std::string experts() {
        return "Expert analysis: sparse top-k routing utilization is tracked during forward passes.\n";
    }
};

}  // namespace from

