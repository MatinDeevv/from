#include "commands.hpp"

#include <exception>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN);
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif
    if (argc < 2 || std::string(argv[1]) == "--help") {
        std::cout << "Usage: from <command> [options]\n\n"
                  << "Commands:\n"
                  << "  train                 Train the SequenceModel.\n"
                  << "  train-full            Train the FromModel pipeline.\n"
                  << "  infer                 Produce model signals from tick data.\n"
                  << "  backtest              Run one configured after-cost backtest.\n"
                  << "  cost-analysis         Run optimistic, realistic, and conservative costs.\n"
                  << "  walkforward, wfdeep   Run walk-forward validation.\n"
                  << "  validate-adversarial  Run adversarial validation.\n"
                  << "  verify-checkpoint     Validate a FROM v1 checkpoint header.\n"
                  << "  inspect, bench, test  Diagnostics and test commands.\n";
        return argc < 2 ? 2 : 0;
    }
    from::CliArgs args;
    for (int i = 2; i < argc; ++i) args.args.emplace_back(argv[i]);
    std::string cmd = argv[1];
    try {
        if (cmd == "train") return from::run_train(args);
        if (cmd == "train-full") return from::run_train_full(args);
        if (cmd == "infer") return from::run_infer(args);
        if (cmd == "backtest") return from::run_backtest(args);
        if (cmd == "cost-analysis") return from::run_cost_analysis(args);
        if (cmd == "validate-adversarial") return from::run_validate_adversarial(args);
        if (cmd == "verify-checkpoint") return from::run_verify_checkpoint(args);
        if (cmd == "walkforward") return from::run_walkforward(args);
        if (cmd == "wfdeep") return from::run_wfdeep(args);
        if (cmd == "inspect") return from::run_inspect(args);
        if (cmd == "bench") return from::run_bench(args);
        if (cmd == "test") return from::run_test(args);
        std::cerr << "Unknown command: " << cmd << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "from: " << e.what() << "\n";
        return 1;
    }
}
