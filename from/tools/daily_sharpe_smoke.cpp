#include "wf_metrics.hpp"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    constexpr int64_t kMsPerDay = 86400000LL;
    std::vector<std::pair<int64_t, float>> pnls;
    for (int64_t day = 0; day < 252; ++day) {
        pnls.emplace_back(day * kMsPerDay, day % 2 == 0 ? 1.1f : -0.9f);
    }
    const double expected = 0.1 * std::sqrt(252.0);
    const double actual = from::wfm::daily_sharpe(pnls);
    if (std::abs(actual - expected) > 1.0e-5) return 1;
    std::cout << "daily Sharpe smoke passed: " << actual << "\n";
    return 0;
}
