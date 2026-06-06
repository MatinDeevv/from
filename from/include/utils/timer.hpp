#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace from {

class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_ = clock::now();

public:
    void reset() { start_ = clock::now(); }

    double elapsed_seconds() const {
        return std::chrono::duration<double>(clock::now() - start_).count();
    }
};

class Profiler {
    std::unordered_map<std::string, double> totals_;
    std::unordered_map<std::string, Timer> active_;

public:
    void start(const std::string& name) { active_[name].reset(); }
    void stop(const std::string& name) { totals_[name] += active_[name].elapsed_seconds(); }
    const std::unordered_map<std::string, double>& totals() const { return totals_; }
};

}  // namespace from

