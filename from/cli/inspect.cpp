#include "commands.hpp"

#include "analysis/inspector.hpp"
#include "analysis/symbolic_regression.hpp"
#include "data/normalizer.hpp"
#include "model/serializer.hpp"

#include <iostream>

namespace from {

int run_inspect(const CliArgs& args) {
    std::string model_path = args.get("--model", "weights.from");
    std::string mode = args.get("--mode", "kernels");
    FromModel model(FromConfig{FROM_MAX_FEATURES, 16, 16, 1, 2, 32, 1, 2, 1, 32, 8, 16, 16, 1, 0.1f});
    Normalizer normalizer(FROM_MAX_FEATURES);
    if (!model_path.empty()) {
        Serializer::load(model_path, &model, &normalizer);
    }
    if (mode == "kernels") std::cout << Inspector::kernels(model);
    else if (mode == "attention") std::cout << Inspector::attention(model);
    else if (mode == "regime") std::cout << Inspector::regime();
    else if (mode == "experts") std::cout << Inspector::experts();
    else if (mode == "symbolic") {
        SymbolicRegressionHead sym;
        std::cout << "Discovered signal formula:\n  " << sym.extract_formula() << "\n";
    } else {
        std::cerr << "Unknown inspect mode: " << mode << "\n";
        return 2;
    }
    return 0;
}

}  // namespace from

