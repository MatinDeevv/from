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
    double open = 0.0, high = 0.0, low = 0.0, close = 0.0;
    double spread_sum = 0.0;
    double av_sum = 0.0, bv_sum = 0.0;
    uint32_t ticks = 0;
};

struct Row {
    int64_t ts = 0;
    double entry = 0.0;
    double spread_pips = 0.0;
    std::array<float, 32> x{};
};

struct Rule {
    int family = 0;
    double th_a = 0.0;
    double th_b = 0.0;
    double rr = 1.5;
    int max_hold = 6;
};

struct Trade {
    double pnl = 0.0;
    int64_t ts = 0;
};

struct Stats {
    size_t rows = 0;
    size_t trades = 0;
    size_t wins = 0;
    double pnl = 0.0;
    double avg = 0.0;
    double sharpe = 0.0;
    double max_dd = 0.0;
    double trades_day = 0.0;
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
    b.av_sum += av;
    b.bv_sum += bv;
    ++b.ticks;
}

double avg_range(const std::vector<Bar>& b, size_t i, size_t n) {
    if (i == 0) return 0.0;
    size_t lo = i > n ? i - n : 0;
    double s = 0.0;
    size_t c = 0;
    for (size_t k = lo; k < i; ++k) {
        s += b[k].high - b[k].low;
        ++c;
    }
    return c ? s / static_cast<double>(c) : 0.0;
}

double avg_close(const std::vector<Bar>& b, size_t i, size_t n) {
    if (i == 0) return 0.0;
    size_t lo = i > n ? i - n : 0;
    double s = 0.0;
    size_t c = 0;
    for (size_t k = lo; k < i; ++k) {
        s += b[k].close;
        ++c;
    }
    return c ? s / static_cast<double>(c) : 0.0;
}

double avg_spread(const std::vector<Bar>& b, size_t i, size_t n) {
    if (i == 0) return 0.0;
    size_t lo = i > n ? i - n : 0;
    double s = 0.0;
    size_t c = 0;
    for (size_t k = lo; k < i; ++k) {
        if (b[k].ticks) {
            s += b[k].spread_sum / b[k].ticks;
            ++c;
        }
    }
    return c ? s / static_cast<double>(c) : 0.0;
}

int hour_utc(int64_t ts) {
    int64_t day = 86400000LL;
    int64_t t = ((ts % day) + day) % day;
    return static_cast<int>(t / 3600000LL);
}

bool liquid_session(int64_t ts) {
    int h = hour_utc(ts);
    return (h >= 7 && h < 17);
}

std::vector<Row> make_rows(const std::vector<Bar>& b5) {
    std::vector<Row> rows;
    if (b5.size() < 300) return rows;
    rows.reserve(b5.size());
    for (size_t i = 240; i + 24 < b5.size(); ++i) {
        const Bar& c = b5[i];
        double close = c.close;
        double ma3 = avg_close(b5, i, 3);
        double ma12 = avg_close(b5, i, 12);
        double ma48 = avg_close(b5, i, 48);
        double ma144 = avg_close(b5, i, 144);
        double ma12_prev = avg_close(b5, i > 12 ? i - 12 : i, 12);
        double ma48_prev = avg_close(b5, i > 12 ? i - 12 : i, 48);
        double r1 = close - b5[i - 1].close;
        double r3 = close - b5[i - 3].close;
        double r12 = close - b5[i - 12].close;
        double r48 = close - b5[i - 48].close;
        double r144 = close - b5[i - 144].close;
        double range = c.high - c.low;
        double atr12 = avg_range(b5, i, 12);
        double atr48 = avg_range(b5, i, 48);
        double atr144 = avg_range(b5, i, 144);
        double spread = c.ticks ? c.spread_sum / c.ticks : 0.0;
        double spread48 = avg_spread(b5, i, 48);
        double vol_imb = (c.av_sum - c.bv_sum) / (c.av_sum + c.bv_sum + 1e-8);
        double pos = range > 1e-8 ? (close - c.low) / range : 0.5;
        int h = hour_utc(c.bucket);

        Row r;
        r.ts = c.bucket;
        r.entry = close;
        r.spread_pips = std::max(0.0, spread * 100.0);
        r.x = {
            static_cast<float>(r1 * 100.0),
            static_cast<float>(r3 * 100.0),
            static_cast<float>(r12 * 100.0),
            static_cast<float>(r48 * 100.0),
            static_cast<float>(r144 * 100.0),
            static_cast<float>((close - ma3) * 100.0),
            static_cast<float>((close - ma12) * 100.0),
            static_cast<float>((close - ma48) * 100.0),
            static_cast<float>((close - ma144) * 100.0),
            static_cast<float>((ma12 - ma48) * 100.0),
            static_cast<float>((ma48 - ma144) * 100.0),
            static_cast<float>((ma12 - ma12_prev) * 100.0),
            static_cast<float>((ma48 - ma48_prev) * 100.0),
            static_cast<float>(range * 100.0),
            static_cast<float>(atr12 * 100.0),
            static_cast<float>(atr48 * 100.0),
            static_cast<float>(atr144 * 100.0),
            static_cast<float>(atr48 > 1e-8 ? range / atr48 : 0.0),
            static_cast<float>(atr144 > 1e-8 ? atr48 / atr144 : 0.0),
            static_cast<float>(pos),
            static_cast<float>(r.spread_pips),
            static_cast<float>(spread48 > 1e-8 ? spread / spread48 : 1.0),
            static_cast<float>(c.ticks),
            static_cast<float>(vol_imb),
            static_cast<float>(liquid_session(c.bucket) ? 1.0 : 0.0),
            static_cast<float>(h >= 7 && h < 12 ? 1.0 : 0.0),
            static_cast<float>(h >= 13 && h < 17 ? 1.0 : 0.0),
            static_cast<float>(std::sin(h / 24.0 * 6.28318530718)),
            static_cast<float>(std::cos(h / 24.0 * 6.28318530718)),
            static_cast<float>(r12 * r48 >= 0 ? 1.0 : -1.0),
            static_cast<float>(close > ma48 ? 1.0 : -1.0),
            static_cast<float>(close > ma144 ? 1.0 : -1.0)
        };
        rows.push_back(r);
    }
    return rows;
}

void standardize(std::vector<Row>& rows, size_t train_end) {
    std::array<double, 32> mean{}, sd{};
    for (double& v : sd) v = 1.0;
    train_end = std::min(train_end, rows.size());
    for (size_t i = 0; i < train_end; ++i) for (size_t d = 0; d < 32; ++d) mean[d] += rows[i].x[d];
    for (double& v : mean) v /= static_cast<double>(std::max<size_t>(1, train_end));
    for (size_t i = 0; i < train_end; ++i) {
        for (size_t d = 0; d < 32; ++d) {
            double diff = rows[i].x[d] - mean[d];
            sd[d] += diff * diff;
        }
    }
    for (double& v : sd) v = std::sqrt(v / static_cast<double>(std::max<size_t>(1, train_end)) + 1e-8);
    for (Row& r : rows) {
        for (size_t d = 0; d < 32; ++d) {
            r.x[d] = static_cast<float>(std::clamp((r.x[d] - mean[d]) / sd[d], -8.0, 8.0));
        }
    }
}

int direction(const Row& r, const Rule& rule) {
    // Hypothesis families:
    // 0: 1h/15m continuation after 5m pullback.
    // 1: volatility compression breakout.
    // 2: exhaustion mean reversion.
    // 3: session trend continuation.
    // 4: liquidity-taking toxicity: imbalance with non-widening spread.
    // 5: adverse-selection underpriced spread: imbalance plus tight spread.
    // 6: tick-rate shock continuation.
    // 7: activity burst without displacement fades.
    // 8: liquidity-adjusted momentum.
    // 9: fragile-market imbalance continuation.
    // 10: short-term positive autocorrelation regime.
    // 11: short-term negative autocorrelation regime.
    // 12: London risk-transfer directional drift.
    // 13: NY risk-transfer directional drift.
    // 14: spread normalization after shock, continuation.
    // 15: spread widening after displacement, fade exhaustion.
    // 16: volume-clock imbalance.
    // 17: high-vol trend filter.
    // 18: low-vol mean-reversion filter.
    // 19: close-location flow confirmation.
    if (r.x[24] < 0.0f) return 0;
    if (rule.family == 0) {
        bool up_ctx = r.x[10] > rule.th_a && r.x[12] > 0.0f;
        bool dn_ctx = r.x[10] < -rule.th_a && r.x[12] < 0.0f;
        if (up_ctx && r.x[6] < -rule.th_b) return 1;
        if (dn_ctx && r.x[6] > rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 1) {
        bool compressed = r.x[18] < -rule.th_a && r.x[21] < rule.th_b;
        if (!compressed) return 0;
        if (r.x[2] > 0.0f && r.x[9] > 0.0f) return 1;
        if (r.x[2] < 0.0f && r.x[9] < 0.0f) return -1;
        return 0;
    }
    if (rule.family == 2) {
        if (r.x[3] > rule.th_a && r.x[19] > rule.th_b) return -1;
        if (r.x[3] < -rule.th_a && r.x[19] < -rule.th_b) return 1;
        return 0;
    }
    if (rule.family == 3) {
        if (r.x[25] < 0.0f && r.x[26] < 0.0f) return 0;
        if (r.x[9] > rule.th_a && r.x[2] > rule.th_b) return 1;
        if (r.x[9] < -rule.th_a && r.x[2] < -rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 4) {
        if (r.x[23] > rule.th_a && r.x[21] < rule.th_b) return 1;
        if (r.x[23] < -rule.th_a && r.x[21] < rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 5) {
        if (r.x[23] > rule.th_a && r.x[20] < rule.th_b && r.x[17] > 0.0f) return 1;
        if (r.x[23] < -rule.th_a && r.x[20] < rule.th_b && r.x[17] > 0.0f) return -1;
        return 0;
    }
    if (rule.family == 6) {
        if (r.x[22] > rule.th_a && r.x[2] > rule.th_b && r.x[19] > 0.5f) return 1;
        if (r.x[22] > rule.th_a && r.x[2] < -rule.th_b && r.x[19] < -0.5f) return -1;
        return 0;
    }
    if (rule.family == 7) {
        if (r.x[22] > rule.th_a && std::abs(r.x[2]) < rule.th_b && r.x[19] > 0.5f) return -1;
        if (r.x[22] > rule.th_a && std::abs(r.x[2]) < rule.th_b && r.x[19] < -0.5f) return 1;
        return 0;
    }
    if (rule.family == 8) {
        if (r.x[2] > rule.th_a && r.x[21] < rule.th_b && r.x[22] > 0.0f) return 1;
        if (r.x[2] < -rule.th_a && r.x[21] < rule.th_b && r.x[22] > 0.0f) return -1;
        return 0;
    }
    if (rule.family == 9) {
        if (r.x[18] > rule.th_a && r.x[23] > rule.th_b) return 1;
        if (r.x[18] > rule.th_a && r.x[23] < -rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 10) {
        if (r.x[29] > 0.0f && r.x[2] > rule.th_a && r.x[17] > rule.th_b) return 1;
        if (r.x[29] > 0.0f && r.x[2] < -rule.th_a && r.x[17] > rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 11) {
        if (r.x[29] < 0.0f && r.x[2] > rule.th_a && r.x[19] > rule.th_b) return -1;
        if (r.x[29] < 0.0f && r.x[2] < -rule.th_a && r.x[19] < -rule.th_b) return 1;
        return 0;
    }
    if (rule.family == 12) {
        if (r.x[25] < 0.0f) return 0;
        if (r.x[9] > rule.th_a && r.x[23] > rule.th_b) return 1;
        if (r.x[9] < -rule.th_a && r.x[23] < -rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 13) {
        if (r.x[26] < 0.0f) return 0;
        if (r.x[3] > rule.th_a && r.x[19] > rule.th_b) return -1;
        if (r.x[3] < -rule.th_a && r.x[19] < -rule.th_b) return 1;
        return 0;
    }
    if (rule.family == 14) {
        if (r.x[21] < -rule.th_a && r.x[2] > rule.th_b) return 1;
        if (r.x[21] < -rule.th_a && r.x[2] < -rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 15) {
        if (r.x[21] > rule.th_a && r.x[3] > rule.th_b) return -1;
        if (r.x[21] > rule.th_a && r.x[3] < -rule.th_b) return 1;
        return 0;
    }
    if (rule.family == 16) {
        if (r.x[22] > rule.th_a && r.x[23] > rule.th_b && r.x[21] < 1.0f) return 1;
        if (r.x[22] > rule.th_a && r.x[23] < -rule.th_b && r.x[21] < 1.0f) return -1;
        return 0;
    }
    if (rule.family == 17) {
        if (r.x[18] > rule.th_a && r.x[9] > 0.0f && r.x[2] > rule.th_b) return 1;
        if (r.x[18] > rule.th_a && r.x[9] < 0.0f && r.x[2] < -rule.th_b) return -1;
        return 0;
    }
    if (rule.family == 18) {
        if (r.x[18] < -rule.th_a && r.x[6] > rule.th_b) return -1;
        if (r.x[18] < -rule.th_a && r.x[6] < -rule.th_b) return 1;
        return 0;
    }
    if (rule.family == 19) {
        if (r.x[23] > rule.th_a && r.x[19] > rule.th_b && r.x[2] > 0.0f) return 1;
        if (r.x[23] < -rule.th_a && r.x[19] < -rule.th_b && r.x[2] < 0.0f) return -1;
        return 0;
    }
    return 0;
}

std::vector<Trade> trades_for(const std::vector<Row>& rows, size_t lo, size_t hi, const Rule& rule, double cost_mult) {
    std::vector<Trade> trades;
    hi = std::min(hi, rows.size());
    for (size_t i = lo; i < hi; ++i) {
        int dir = direction(rows[i], rule);
        if (dir == 0) continue;
        double stop = rows[i].spread_pips * 2.0 + 8.0;
        double target = stop * rule.rr;
        double exit_pnl = 0.0;
        size_t end = std::min(rows.size() - 1, i + static_cast<size_t>(rule.max_hold));
        for (size_t j = i + 1; j <= end; ++j) {
            double move = (rows[j].entry - rows[i].entry) * 100.0 * dir;
            if (move >= target) { exit_pnl = target; break; }
            if (move <= -stop) { exit_pnl = -stop; break; }
            if (j == end) exit_pnl = move;
        }
        exit_pnl -= cost_mult * rows[i].spread_pips;
        trades.push_back({exit_pnl, rows[i].ts});
    }
    return trades;
}

Stats stats_for(const std::vector<Row>& rows, size_t lo, size_t hi, const Rule& rule, double cost_mult) {
    Stats s;
    s.rows = hi > lo ? hi - lo : 0;
    auto trades = trades_for(rows, lo, hi, rule, cost_mult);
    s.trades = trades.size();
    std::vector<double> rets;
    double eq = 0.0, peak = 0.0;
    for (const auto& t : trades) {
        if (t.pnl > 0) ++s.wins;
        s.pnl += t.pnl;
        rets.push_back(t.pnl);
        eq += t.pnl;
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
    if (hi > lo && rows[hi - 1].ts > rows[lo].ts) {
        double days = static_cast<double>(rows[hi - 1].ts - rows[lo].ts) / 86400000.0;
        if (days > 0.0) s.trades_day = static_cast<double>(s.trades) / days;
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) args.xs.emplace_back(argv[i]);
    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string report = args.get("--report", "mtf_hypothesis_report.json");
    size_t skip_ticks = static_cast<size_t>(std::stoull(args.get("--skip-ticks", "0")));
    size_t max_ticks = static_cast<size_t>(std::stoull(args.get("--ticks", "60000000")));
    size_t chunk_size = static_cast<size_t>(std::stoull(args.get("--chunk-size", "1000000")));
    double min_tpd = std::stod(args.get("--min-trades-day", "15"));
    double max_tpd = std::stod(args.get("--max-trades-day", "30"));
    double cost_mult = std::stod(args.get("--cost-mult", "1.0"));
    bool all_sessions = args.get("--all-sessions", "0") == "1";

    Timer timer;
    ParquetReader reader(data);
    size_t total_rows = reader.total_rows();
    skip_ticks = std::min(skip_ticks, total_rows);
    size_t eval_ticks = std::min(max_ticks, total_rows - skip_ticks);
    std::vector<Bar> bars;
    size_t skipped = 0;
    while (reader.has_next_chunk() && skipped < skip_ticks) {
        TickChunk t = reader.read_chunk(std::min(chunk_size, skip_ticks - skipped));
        skipped += t.size;
    }
    size_t read = 0;
    std::cout << "MTF HYPOTHESIS 5m entry, 15m/1h context\n";
    while (reader.has_next_chunk() && read < eval_ticks) {
        TickChunk t = reader.read_chunk(std::min(chunk_size, eval_ticks - read));
        if (t.size == 0) break;
        read += t.size;
        for (size_t i = 0; i < t.size; ++i) {
            int64_t bucket = (t.time_ms[i] / 300000LL) * 300000LL;
            double mid = t.mid[i] != 0.0 ? t.mid[i] : (t.ask[i] + t.bid[i]) * 0.5;
            add_tick(bars, bucket, mid, t.ask[i] - t.bid[i], t.ask_vol[i], t.bid_vol[i]);
        }
        if ((read / 1000000) != ((read - t.size) / 1000000)) {
            std::cout << "  read " << (read / 1000000) << "M ticks bars=" << bars.size() << "\n";
        }
    }
    std::vector<Row> rows = make_rows(bars);
    if (rows.size() < 1000) {
        std::cerr << "not enough rows\n";
        return 2;
    }
    size_t train_end = rows.size() * 60 / 100;
    size_t val_begin = train_end + 24;
    size_t val_end = rows.size() * 80 / 100;
    size_t test_begin = val_end + 24;
    standardize(rows, train_end);
    if (all_sessions) {
        for (Row& row : rows) row.x[24] = 1.0f;
    }

    struct Candidate { Rule r; Stats tr, va, te, stress; };
    std::vector<Candidate> cs;
    std::vector<double> ths{0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};
    std::vector<double> rrs{1.0, 1.25, 1.5, 2.0};
    std::vector<int> holds{3, 6, 12, 24};
    for (int fam = 0; fam < 20; ++fam) {
        for (double a : ths) {
            for (double b : ths) {
                for (double rr : rrs) {
                    for (int hold : holds) {
                        Rule r{fam, a, b, rr, hold};
                        Stats tr = stats_for(rows, 0, train_end, r, cost_mult);
                        Stats va = stats_for(rows, val_begin, val_end, r, cost_mult);
                        if (tr.pnl <= 0.0 || va.pnl <= 0.0) continue;
                        if (tr.trades_day < min_tpd || tr.trades_day > max_tpd) continue;
                        if (va.trades_day < min_tpd || va.trades_day > max_tpd) continue;
                        Stats te = stats_for(rows, test_begin, rows.size(), r, cost_mult);
                        Stats st = stats_for(rows, test_begin, rows.size(), r, 2.0);
                        cs.push_back({r, tr, va, te, st});
                    }
                }
            }
        }
    }
    std::sort(cs.begin(), cs.end(), [](const Candidate& a, const Candidate& b) {
        return std::min({a.tr.sharpe, a.va.sharpe, a.te.sharpe}) > std::min({b.tr.sharpe, b.va.sharpe, b.te.sharpe});
    });
    bool found = !cs.empty() && cs[0].te.pnl > 0.0 && cs[0].stress.pnl > 0.0 &&
                 cs[0].te.trades_day >= min_tpd && cs[0].te.trades_day <= max_tpd;

    std::filesystem::create_directories(std::filesystem::path(report).parent_path());
    std::ofstream os(report);
    os << std::fixed << std::setprecision(8);
    os << "{\n";
    os << "  \"verdict\":\"" << (found ? "CANDIDATE_FOUND" : "NO_EDGE_FOUND") << "\",\n";
    os << "  \"data\":{\"path\":\"" << esc(data) << "\",\"skip_ticks\":" << skipped
       << ",\"evaluated_ticks\":" << read << ",\"bars_5m\":" << bars.size()
       << ",\"rows\":" << rows.size() << ",\"target_min_trades_day\":" << min_tpd
       << ",\"target_max_trades_day\":" << max_tpd << ",\"cost_mult\":" << cost_mult << "},\n";
    os << "  \"candidate_count\":" << cs.size() << ",\n";
    os << "  \"best\":{";
    if (!cs.empty()) {
        const auto& c = cs[0];
        os << "\"family\":" << c.r.family << ",\"th_a\":" << c.r.th_a << ",\"th_b\":" << c.r.th_b
           << ",\"rr\":" << c.r.rr << ",\"max_hold\":" << c.r.max_hold
           << ",\"train_pnl\":" << c.tr.pnl << ",\"train_sharpe\":" << c.tr.sharpe
           << ",\"train_trades_day\":" << c.tr.trades_day
           << ",\"val_pnl\":" << c.va.pnl << ",\"val_sharpe\":" << c.va.sharpe
           << ",\"val_trades_day\":" << c.va.trades_day
           << ",\"test_pnl\":" << c.te.pnl << ",\"test_sharpe\":" << c.te.sharpe
           << ",\"test_trades_day\":" << c.te.trades_day
           << ",\"test_2x_cost_pnl\":" << c.stress.pnl
           << ",\"test_2x_cost_sharpe\":" << c.stress.sharpe
           << ",\"test_2x_cost_trades_day\":" << c.stress.trades_day;
    }
    os << "},\n";
    os << "  \"top\":[\n";
    size_t keep = std::min<size_t>(10, cs.size());
    for (size_t i = 0; i < keep; ++i) {
        const auto& c = cs[i];
        os << "    {\"rank\":" << (i + 1) << ",\"family\":" << c.r.family
           << ",\"th_a\":" << c.r.th_a << ",\"th_b\":" << c.r.th_b
           << ",\"rr\":" << c.r.rr << ",\"max_hold\":" << c.r.max_hold
           << ",\"train_sharpe\":" << c.tr.sharpe << ",\"val_sharpe\":" << c.va.sharpe
           << ",\"test_sharpe\":" << c.te.sharpe << ",\"test_pnl\":" << c.te.pnl
           << ",\"test_trades_day\":" << c.te.trades_day
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
        std::cout << "best family=" << c.r.family << " a=" << c.r.th_a << " b=" << c.r.th_b
                  << " rr=" << c.r.rr << " hold=" << c.r.max_hold
                  << " val_tpd=" << c.va.trades_day << " test_tpd=" << c.te.trades_day
                  << " test_pnl=" << c.te.pnl << " test_2x=" << c.stress.pnl << "\n";
    }
    return found ? 0 : 3;
}
