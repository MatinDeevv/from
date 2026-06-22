#pragma once
// ============================================================================
// sample_weights.hpp — Lopez de Prado sample-uniqueness + time-decay weights.
//
// Overlapping windows are NOT independent observations. Two samples whose
// triple-barrier label spans overlap share future ticks, so counting both at
// full weight inflates the effective sample size and biases the fit toward
// whatever regime happened to produce the most overlap. The standard fix
// (Advances in Financial Machine Learning, ch. 4):
//
//   1. Each label i occupies a span [entry_i, entry_i + horizon) of ticks.
//   2. Concurrency c(t) = number of label spans covering tick t.
//   3. Uniqueness u_i = mean over the span of 1 / c(t)   (in (0, 1]).
//   4. Optional time-decay so recent samples count more than stale ones.
//
// Weights are returned normalized to mean 1.0 so they slot into an existing
// class-weighted loss without changing its overall scale.
//
// Header-only, no GPU, O(n log n). `entries` = GLOBAL monotonic tick index of
// each sample's window start (Sample::t_index); `horizon` = label span length
// in ticks (the Windower horizon). entries need NOT be pre-sorted.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <vector>

namespace from {

// Average concurrency-based uniqueness per sample, in (0, 1]. Returns one
// weight per input entry, in the SAME order as `entries` (not sorted order).
inline std::vector<float> compute_uniqueness(const std::vector<long long>& entries,
                                             long long horizon) {
    const size_t n = entries.size();
    std::vector<float> uniq(n, 1.0f);
    if (n == 0 || horizon <= 0) return uniq;

    // Sort sample indices by entry tick so the sweep is monotonic.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return entries[a] < entries[b]; });

    // Coordinate-compress all span endpoints into elementary intervals.
    // Each span is [s, s+horizon). Endpoints define breakpoints; between two
    // consecutive breakpoints concurrency is constant.
    std::vector<long long> pts;
    pts.reserve(2 * n);
    for (size_t i = 0; i < n; ++i) {
        pts.push_back(entries[i]);
        pts.push_back(entries[i] + horizon);
    }
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    const size_t m = pts.size();
    if (m < 2) return uniq;  // degenerate: all spans identical point

    // Difference array over elementary intervals [pts[k], pts[k+1]).
    // diff[k] += 1 at span start, -= 1 at span end.
    std::vector<long long> diff(m, 0);
    auto idx_of = [&](long long v) -> size_t {
        return static_cast<size_t>(
            std::lower_bound(pts.begin(), pts.end(), v) - pts.begin());
    };
    for (size_t i = 0; i < n; ++i) {
        diff[idx_of(entries[i])] += 1;
        diff[idx_of(entries[i] + horizon)] -= 1;
    }

    // Concurrency per elementary interval + prefix sum of (len / concurrency)
    // so a span's average 1/c is a single O(1) range query.
    std::vector<double> prefix(m, 0.0);  // prefix[k] = sum over intervals < k of len/c
    long long conc = 0;
    for (size_t k = 0; k + 1 < m; ++k) {
        conc += diff[k];
        double len = static_cast<double>(pts[k + 1] - pts[k]);
        double contrib = (conc > 0) ? len / static_cast<double>(conc) : 0.0;
        prefix[k + 1] = prefix[k] + contrib;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t ks = idx_of(entries[i]);
        size_t ke = idx_of(entries[i] + horizon);
        double span_integral = prefix[ke] - prefix[ks];          // sum len/c over span
        double u = span_integral / static_cast<double>(horizon);  // average 1/c
        if (u <= 0.0) u = 1.0 / static_cast<double>(std::max<long long>(1, conc));
        if (u > 1.0) u = 1.0;
        uniq[i] = static_cast<float>(u);
    }
    return uniq;
}

// Linear time-decay in chronological order: oldest sample gets `tail` (in
// (0, 1]), newest gets 1.0, interpolated by rank. tail=1.0 disables decay.
// `entries` order must match `out` order; ranking is by entry tick.
inline std::vector<float> compute_time_decay(const std::vector<long long>& entries,
                                             float tail = 0.5f) {
    const size_t n = entries.size();
    std::vector<float> w(n, 1.0f);
    if (n <= 1 || tail >= 1.0f) return w;
    if (tail < 0.0f) tail = 0.0f;

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return entries[a] < entries[b]; });

    const double denom = static_cast<double>(n - 1);
    for (size_t r = 0; r < n; ++r) {
        double frac = static_cast<double>(r) / denom;  // 0 oldest .. 1 newest
        w[order[r]] = static_cast<float>(tail + (1.0 - tail) * frac);
    }
    return w;
}

// Combined per-sample weight = uniqueness * time-decay, normalized to mean 1.0
// so it composes with class weights without rescaling the loss. `tail < 1.0`
// enables time-decay; `tail = 1.0` => pure uniqueness.
inline std::vector<float> compute_sample_weights(const std::vector<long long>& entries,
                                                 long long horizon,
                                                 float tail = 1.0f) {
    std::vector<float> u = compute_uniqueness(entries, horizon);
    if (tail < 1.0f) {
        std::vector<float> d = compute_time_decay(entries, tail);
        for (size_t i = 0; i < u.size(); ++i) u[i] *= d[i];
    }
    double sum = 0.0;
    for (float v : u) sum += v;
    if (sum > 0.0) {
        double scale = static_cast<double>(u.size()) / sum;
        for (float& v : u) v = static_cast<float>(v * scale);
    }
    return u;
}

}  // namespace from
