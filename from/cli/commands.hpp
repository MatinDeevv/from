#pragma once

#include <string>
#include <vector>

namespace from {

struct CliArgs {
    std::vector<std::string> args;

    std::string get(const std::string& key, const std::string& def = "") const {
        for (size_t i = 0; i + 1 < args.size(); ++i) {
            if (args[i] == key) return args[i + 1];
        }
        return def;
    }

    bool has(const std::string& key) const {
        for (const auto& a : args) if (a == key) return true;
        return false;
    }
};

int run_train(const CliArgs& args);
int run_train_full(const CliArgs& args);
int run_infer(const CliArgs& args);
int run_backtest(const CliArgs& args);
int run_cost_analysis(const CliArgs& args);
int run_validate_adversarial(const CliArgs& args);
int run_verify_checkpoint(const CliArgs& args);
int run_walkforward(const CliArgs& args);
int run_wfdeep(const CliArgs& args);
int run_inspect(const CliArgs& args);
int run_bench(const CliArgs& args);
int run_test(const CliArgs& args);

}  // namespace from

