#include "data/parquet_reader.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace from;

namespace {

struct Args {
    std::vector<std::string> xs;
    std::string get(const std::string& k, const std::string& d = "") const {
        for (size_t i = 0; i + 1 < xs.size(); ++i) if (xs[i] == k) return xs[i + 1];
        return d;
    }
};

struct Bar {
    int64_t bucket = 0;
    double open = 0, high = 0, low = 0, close = 0;
    double spread_sum = 0.0;
    double ask_vol_sum = 0.0;
    double bid_vol_sum = 0.0;
    uint32_t ticks = 0;
};

struct Row {
    int64_t bucket = 0;
    std::array<float, 24> x{};
    double move_pips = 0.0;
    double cost_pips = 0.0;
};

struct Rule {
    size_t dim = 0;
    double threshold = 0.0;
    int mode = 1;
};

struct Stats {
    size_t n = 0;
    size_t trades = 0;
    size_t wins = 0;
    double pnl = 0.0;
    double avg = 0.0;
    double sharpe = 0.0;
    double max_dd = 0.0;
    double trades_per_day = 0.0;
};

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '"') o += "\\\"";
        else o += c;
    }
    return o;
}

int signal(const Row& r, const Rule& rule) {
    double z = r.x[rule.dim];
    switch (rule.mode) {
        case 1: return z > rule.threshold ? 1 : 0;
        case 2: return z > rule.threshold ? -1 : 0;
        case 3: return z < -rule.threshold ? 1 : 0;
        case 4: return z < -rule.threshold ? -1 : 0;
        case 5:
            if (z > rule.threshold) return 1;
            if (z < -rule.threshold) return -1;
            return 0;
        case 6:
            if (z > rule.threshold) return -1;
            if (z < -rule.threshold) return 1;
            return 0;
        default: return 0;
    }
}

Stats eval_rule(const std::vector<Row>& rows, size_t lo, size_t hi, const Rule& rule, double cost_mult) {
    Stats s;
    hi = std::min(hi, rows.size());
    lo = std::min(lo, hi);
    s.n = hi - lo;
    std::vector<double> rets;
    double eq = 0.0, peak = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        int dir = signal(rows[i], rule);
        if (dir == 0) continue;
        double pnl = dir * rows[i].move_pips - cost_mult * rows[i].cost_pips;
        ++s.trades;
        if (pnl > 0) ++s.wins;
        s.pnl += pnl;
        rets.push_back(pnl);
        eq += pnl;
        peak = std::max(peak, eq);
        s.max_dd = std::max(s.max_dd, peak - eq);
    }
    if (!rets.empty()) {
        s.avg = s.pnl / static_cast<double>(rets.size());
        double var = 0.0;
        for (double v : rets) {
            double d = v - s.avg;
            var += d * d;
        }
        double sd = rets.size() > 1 ? std::sqrt(var / static_cast<double>(rets.size() - 1)) : 0.0;
        s.sharpe = sd > 0.0 ? s.avg / sd * std::sqrt(252.0) : 0.0;
    }
    if (hi > lo && rows[hi - 1].bucket > rows[lo].bucket) {
        double days = static_cast<double>(rows[hi - 1].bucket - rows[lo].bucket) / 86400000.0;
        if (days > 0.0) s.trades_per_day = static_cast<double>(s.trades) / days;
    }
    return s;
}

double mean(const std::vector<Bar>& b, size_t i, size_t n, auto fn) {
    if (i == 0) return 0.0;
    size_t lo = i > n ? i - n : 0;
    double s = 0.0;
    size_t c = 0;
    for (size_t k = lo; k < i; ++k) {
        s += fn(b[k]);
        ++c;
    }
    return c ? s / static_cast<double>(c) : 0.0;
}

std::vector<Row> make_rows(std::vector<Bar>& bars, int horizon_bars) {
    std::vector<Row> rows;
    if (bars.size() < 80 + static_cast<size_t>(horizon_bars)) return rows;
    rows.reserve(bars.size());
    for (size_t i = 64; i + static_cast<size_t>(horizon_bars) < bars.size(); ++i) {
        const Bar& p = bars[i - 1];
        const Bar& c = bars[i];
        double close = c.close;
        double ret1 = close - p.close;
        double ret3 = close - bars[i - 3].close;
        double ret6 = close - bars[i - 6].close;
        double ret12 = close - bars[i - 12].close;
        double ret24 = close - bars[i - 24].close;
        double ret48 = close - bars[i - 48].close;
        double range = c.high - c.low;
        double range12 = mean(bars, i, 12, [](const Bar& x) { return x.high - x.low; });
        double range48 = mean(bars, i, 48, [](const Bar& x) { return x.high - x.low; });
        double ma6 = mean(bars, i, 6, [](const Bar& x) { return x.close; });
        double ma24 = mean(bars, i, 24, [](const Bar& x) { return x.close; });
        double ma48 = mean(bars, i, 48, [](const Bar& x) { return x.close; });
        double spread = c.ticks ? c.spread_sum / c.ticks : 0.0;
        double spread24 = mean(bars, i, 24, [](const Bar& x) { return x.ticks ? x.spread_sum / x.ticks : 0.0; });
        double ticks24 = mean(bars, i, 24, [](const Bar& x) { return static_cast<double>(x.ticks); });
        double vol_imb = (c.ask_vol_sum - c.bid_vol_sum) / (c.ask_vol_sum + c.bid_vol_sum + 1e-8);
        double pos = range > 1e-8 ? (close - c.low) / range : 0.5;

        Row r;
        r.bucket = c.bucket;
        r.move_pips = (bars[i + horizon_bars].close - close) * 100.0;
        r.cost_pips = std::max(0.0, spread * 100.0);
        r.x = {
            static_cast<float>(ret1 * 100.0),
            static_cast<float>(ret3 * 100.0),
            static_cast<float>(ret6 * 100.0),
            static_cast<float>(ret12 * 100.0),
            static_cast<float>(ret24 * 100.0),
            static_cast<float>(ret48 * 100.0),
            static_cast<float>((close - ma6) * 100.0),
            static_cast<float>((close - ma24) * 100.0),
            static_cast<float>((close - ma48) * 100.0),
            static_cast<float>((ma6 - ma24) * 100.0),
            static_cast<float>((ma24 - ma48) * 100.0),
            static_cast<float>(range * 100.0),
            static_cast<float>(range12 * 100.0),
            static_cast<float>(range48 * 100.0),
            static_cast<float>(range48 > 1e-8 ? range / range48 : 0.0),
            static_cast<float>(pos),
            static_cast<float>(spread * 100.0),
            static_cast<float>(spread24 > 1e-8 ? spread / spread24 : 1.0),
            static_cast<float>(c.ticks),
            static_cast<float>(ticks24 > 1e-8 ? c.ticks / ticks24 : 1.0),
            static_cast<float>(vol_imb),
            static_cast<float>(std::sin((c.bucket % 86400000LL) / 86400000.0 * 6.28318530718)),
            static_cast<float>(std::cos((c.bucket % 86400000LL) / 86400000.0 * 6.28318530718)),
            static_cast<float>((ret24 != 0.0 && ret6 != 0.0 && (ret24 > 0) == (ret6 > 0)) ? 1.0 : -1.0)
        };
        rows.push_back(r);
    }
    return rows;
}

void standardize(std::vector<Row>& rows, size_t train_end) {
    std::array<double, 24> m{}, s{};
    for (double& v : s) v = 1.0;
    train_end = std::min(train_end, rows.size());
    for (size_t i = 0; i < train_end; ++i) for (size_t d = 0; d < 24; ++d) m[d] += rows[i].x[d];
    for (double& v : m) v /= static_cast<double>(std::max<size_t>(1, train_end));
    for (size_t i = 0; i < train_end; ++i) {
        for (size_t d = 0; d < 24; ++d) {
            double diff = rows[i].x[d] - m[d];
            s[d] += diff * diff;
        }
    }
    for (double& v : s) v = std::sqrt(v / static_cast<double>(std::max<size_t>(1, train_end)) + 1e-8);
    for (Row& r : rows) {
        for (size_t d = 0; d < 24; ++d) r.x[d] = static_cast<float>(std::clamp((r.x[d] - m[d]) / s[d], -8.0, 8.0));
    }
}

void add_tick(std::vector<Bar>& bars, int64_t bucket, double mid, double spread, float av, float bv) {
    if (bars.empty() || bars.back().bucket != bucket) {
        Bar b;
        b.bucket = bucket;
        b.open = b.high = b.low = b.close = mid;
        bars.push_back(b);
    }
    Bar& b = bars.back();
    b.high = std::max(b.high, mid);
    b.low = std::min(b.low, mid);
    b.close = mid;
    b.spread_sum += spread;
    b.ask_vol_sum += av;
    b.bid_vol_sum += bv;
    ++b.ticks;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) args.xs.emplace_back(argv[i]);
    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string report = args.get("--report", "bar_tf_report.json");
    int tf_min = std::stoi(args.get("--tf-min", "5"));
    int horizon_bars = std::stoi(args.get("--horizon-bars", "1"));
    size_t skip_ticks = static_cast<size_t>(std::stoull(args.get("--skip-ticks", "0")));
    size_t max_ticks = static_cast<size_t>(std::stoull(args.get("--ticks", "10000000")));
    size_t chunk_size = static_cast<size_t>(std::stoull(args.get("--chunk-size", "1000000")));
    double min_tpd = std::stod(args.get("--min-trades-day", "15"));
    double max_tpd = std::stod(args.get("--max-trades-day", "30"));
    double cost_mult = std::stod(args.get("--cost-mult", "1.0"));

    Timer timer;
    ParquetReader reader(data);
    size_t total_rows = reader.total_rows();
    skip_ticks = std::min(skip_ticks, total_rows);
    size_t eval_ticks = std::min(max_ticks, total_rows - skip_ticks);
    int64_t tf_ms = static_cast<int64_t>(tf_min) * 60LL * 1000LL;
    std::vector<Bar> bars;

    size_t skipped = 0;
    while (reader.has_next_chunk() && skipped < skip_ticks) {
        TickChunk t = reader.read_chunk(std::min(chunk_size, skip_ticks - skipped));
        skipped += t.size;
    }
    size_t read = 0;
    std::cout << "BAR TF SEARCH tf=" << tf_min << "m horizon=" << horizon_bars << " bars\n";
    while (reader.has_next_chunk() && read < eval_ticks) {
        TickChunk t = reader.read_chunk(std::min(chunk_size, eval_ticks - read));
        if (t.size == 0) break;
        read += t.size;
        for (size_t i = 0; i < t.size; ++i) {
            int64_t bucket = (t.time_ms[i] / tf_ms) * tf_ms;
            double mid = t.mid[i] != 0.0 ? t.mid[i] : (t.ask[i] + t.bid[i]) * 0.5;
            add_tick(bars, bucket, mid, t.ask[i] - t.bid[i], t.ask_vol[i], t.bid_vol[i]);
        }
        if ((read / 1000000) != ((read - t.size) / 1000000)) {
            std::cout << "  read " << (read / 1000000) << "M ticks bars=" << bars.size() << "\n";
        }
    }
    std::vector<Row> rows = make_rows(bars, horizon_bars);
    if (rows.size() < 300) {
        std::cerr << "not enough rows\n";
        return 2;
    }
    size_t train_end = rows.size() * 60 / 100;
    size_t val_begin = std::min(rows.size(), train_end + static_cast<size_t>(horizon_bars));
    size_t val_end = rows.size() * 80 / 100;
    size_t test_begin = std::min(rows.size(), val_end + static_cast<size_t>(horizon_bars));
    standardize(rows, train_end);

    std::vector<double> thresholds{0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0};
    struct Candidate { Rule r; Stats tr, va, te, stress; };
    std::vector<Candidate> cs;
    for (size_t d = 0; d < 24; ++d) {
        for (double th : thresholds) {
            for (int mode : {1,2,3,4,5,6}) {
                Rule r{d, th, mode};
                Stats tr = eval_rule(rows, 0, train_end, r, cost_mult);
                Stats va = eval_rule(rows, val_begin, val_end, r, cost_mult);
                if (tr.pnl <= 0 || va.pnl <= 0) continue;
                if (tr.trades_per_day < min_tpd || tr.trades_per_day > max_tpd) continue;
                if (va.trades_per_day < min_tpd || va.trades_per_day > max_tpd) continue;
                Stats te = eval_rule(rows, test_begin, rows.size(), r, cost_mult);
                Stats st = eval_rule(rows, test_begin, rows.size(), r, 2.0);
                cs.push_back({r, tr, va, te, st});
            }
        }
    }
    std::sort(cs.begin(), cs.end(), [](const Candidate& a, const Candidate& b) {
        return std::min({a.tr.sharpe, a.va.sharpe, a.te.sharpe}) > std::min({b.tr.sharpe, b.va.sharpe, b.te.sharpe});
    });
    bool found = !cs.empty() && cs[0].te.pnl > 0 && cs[0].stress.pnl > 0 &&
                 cs[0].te.trades_per_day >= min_tpd && cs[0].te.trades_per_day <= max_tpd;

    std::filesystem::create_directories(std::filesystem::path(report).parent_path());
    std::ofstream os(report);
    os << std::fixed << std::setprecision(8);
    os << "{\n";
    os << "  \"verdict\":\"" << (found ? "CANDIDATE_FOUND" : "NO_EDGE_FOUND") << "\",\n";
    os << "  \"data\":{\"path\":\"" << esc(data) << "\",\"skip_ticks\":" << skipped
       << ",\"evaluated_ticks\":" << read << ",\"bars\":" << bars.size()
       << ",\"rows\":" << rows.size() << ",\"tf_min\":" << tf_min
       << ",\"horizon_bars\":" << horizon_bars << ",\"cost_mult\":" << cost_mult
       << ",\"target_min_trades_day\":" << min_tpd << ",\"target_max_trades_day\":" << max_tpd << "},\n";
    os << "  \"candidate_count\":" << cs.size() << ",\n";
    os << "  \"best\":{";
    if (!cs.empty()) {
        const auto& c = cs[0];
        os << "\"dim\":" << c.r.dim << ",\"threshold\":" << c.r.threshold << ",\"mode\":" << c.r.mode
           << ",\"train_pnl\":" << c.tr.pnl << ",\"train_sharpe\":" << c.tr.sharpe
           << ",\"train_trades_day\":" << c.tr.trades_per_day
           << ",\"val_pnl\":" << c.va.pnl << ",\"val_sharpe\":" << c.va.sharpe
           << ",\"val_trades_day\":" << c.va.trades_per_day
           << ",\"test_pnl\":" << c.te.pnl << ",\"test_sharpe\":" << c.te.sharpe
           << ",\"test_trades_day\":" << c.te.trades_per_day
           << ",\"test_2x_cost_pnl\":" << c.stress.pnl
           << ",\"test_2x_cost_sharpe\":" << c.stress.sharpe;
    }
    os << "},\n";
    os << "  \"top\":[\n";
    size_t keep = std::min<size_t>(10, cs.size());
    for (size_t i = 0; i < keep; ++i) {
        const auto& c = cs[i];
        os << "    {\"rank\":" << (i + 1) << ",\"dim\":" << c.r.dim
           << ",\"threshold\":" << c.r.threshold << ",\"mode\":" << c.r.mode
           << ",\"train_sharpe\":" << c.tr.sharpe << ",\"val_sharpe\":" << c.va.sharpe
           << ",\"test_sharpe\":" << c.te.sharpe << ",\"test_pnl\":" << c.te.pnl
           << ",\"test_trades_day\":" << c.te.trades_per_day
           << ",\"test_2x_cost_pnl\":" << c.stress.pnl << "}"
           << (i + 1 == keep ? "\n" : ",\n");
    }
    os << "  ],\n";
    os << "  \"elapsed_seconds\":" << timer.elapsed_seconds() << "\n";
    os << "}\n";

    std::cout << "REPORT " << report << "\n";
    std::cout << "VERDICT " << (found ? "CANDIDATE_FOUND" : "NO_EDGE_FOUND") << "\n";
    if (!cs.empty()) {
        const auto& c = cs[0];
        std::cout << "best dim=" << c.r.dim << " th=" << c.r.threshold << " mode=" << c.r.mode
                  << " val_tpd=" << c.va.trades_per_day << " test_tpd=" << c.te.trades_per_day
                  << " test_pnl=" << c.te.pnl << " test_2x=" << c.stress.pnl << "\n";
    }
    return found ? 0 : 3;
}
