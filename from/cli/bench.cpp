#include "commands.hpp"

#include "model/from_model.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <iostream>

namespace from {

int run_bench(const CliArgs& args) {
    size_t windows = static_cast<size_t>(std::stoull(args.get("--windows", "32")));
    FromModel model(FromConfig{FROM_MAX_FEATURES, 16, 16, 1, 2, 32, 1, 2, 1, 32, 8, 16, 16, 1, 0.1f});
    Tensor<float> x = Tensor<float>::randn({1, 32, FROM_MAX_FEATURES}, 0.0f, 1.0f, 1);
    Timer timer;
    for (size_t i = 0; i < windows; ++i) {
        auto y = model.forward(x, false);
        (void)y;
    }
    double sec = timer.elapsed_seconds();
    std::cout << "Inference throughput: " << static_cast<double>(windows) / std::max(1e-9, sec) << " windows/sec\n";
    std::cout << "Memory usage: bounded by tensor working set in this benchmark\n";
    std::cout << "Per-layer timing: aggregate smoke benchmark\n";
    std::cout << "Accuracy/calibration/Sharpe: require labeled test set\n";
    return 0;
}

}  // namespace from
