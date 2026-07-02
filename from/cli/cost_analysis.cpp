#include "commands.hpp"

#include <array>
#include <iostream>
#include <string>

namespace from {
namespace {

struct CostScenario {
    const char* name;
    const char* spread_usd;
    const char* commission_per_lot;
    const char* slippage_usd;
};

}  // namespace

int run_cost_analysis(const CliArgs& args) {
    if (args.has("--help")) {
        std::cout << "Usage: from cost-analysis --data <path> --model <path> [backtest options]\n"
                  << "Runs optimistic, realistic, and conservative USD execution-cost scenarios.\n";
        return 0;
    }

    constexpr std::array<CostScenario, 3> kScenarios{{
        {"optimistic", "0.10", "3.50", "0.02"},
        {"realistic", "0.30", "3.50", "0.05"},
        {"conservative", "0.60", "3.50", "0.10"},
    }};

    for (const CostScenario& scenario : kScenarios) {
        CliArgs scenario_args = args;
        scenario_args.args.insert(scenario_args.args.end(), {
            "--spread-usd", scenario.spread_usd,
            "--commission-per-lot", scenario.commission_per_lot,
            "--slippage-usd", scenario.slippage_usd,
        });
        if (std::string(scenario.name) == "realistic") scenario_args.args.emplace_back("--realistic-scenario");
        std::cout << "\n================ COST SCENARIO: " << scenario.name << " ================\n";
        const int result = run_backtest(scenario_args);
        if (result != 0) return result;
    }
    return 0;
}

}  // namespace from
