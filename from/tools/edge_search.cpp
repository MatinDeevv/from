#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "model/sequence_model.hpp"
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
#include <numeric>
#include <random>
#include <sstream>
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

struct Row {
    std::vector<float> x;
    double move_pips = 0.0;
    double spread_pips = 0.0;
    int truth = 0;
    int env = 0;
};

struct Model {
    size_t dim = 0;
    double threshold = 0.0;
    int mode = 1; // 1 long high, 2 short high, 3 long low, 4 short low, 5 trend, 6 reversal
    double train_sharpe = 0.0;
    double val_sharpe = 0.0;
    double test_sharpe = 0.0;
    double train_pnl = 0.0;
    double val_pnl = 0.0;
    double test_pnl = 0.0;
    size_t train_trades = 0;
    size_t val_trades = 0;
    size_t test_trades = 0;
};

struct Combo {
    Model a;
    Model b;
    double train_sharpe = 0.0;
    double val_sharpe = 0.0;
    double test_sharpe = 0.0;
    double train_pnl = 0.0;
    double val_pnl = 0.0;
    double test_pnl = 0.0;
    size_t train_trades = 0;
    size_t val_trades = 0;
    size_t test_trades = 0;
};

struct Stats {
    size_t n = 0;
    size_t trades = 0;
    size_t wins = 0;
    double pnl = 0.0;
    double avg = 0.0;
    double sharpe = 0.0;
    double max_dd = 0.0;
};

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    return out;
}

Stats eval_model(const std::vector<Row>& rows, size_t lo, size_t hi, const Model& m, double cost_mult = 1.0) {
    Stats s;
    hi = std::min(hi, rows.size());
    lo = std::min(lo, hi);
    s.n = hi - lo;
    std::vector<double> rets;
    rets.reserve(s.n);
    double eq = 0.0, peak = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        const Row& r = rows[i];
        double z = r.x[m.dim];
        int dir = 0;
        switch (m.mode) {
            case 1: if (z > m.threshold) dir = 1; break;
            case 2: if (z > m.threshold) dir = -1; break;
            case 3: if (z < -m.threshold) dir = 1; break;
            case 4: if (z < -m.threshold) dir = -1; break;
            case 5:
                if (z > m.threshold) dir = 1;
                else if (z < -m.threshold) dir = -1;
                break;
            case 6:
                if (z > m.threshold) dir = -1;
                else if (z < -m.threshold) dir = 1;
                break;
            default: break;
        }
        if (dir == 0) continue;
        double pnl = static_cast<double>(dir) * r.move_pips - cost_mult * r.spread_pips;
        ++s.trades;
        if (pnl > 0.0) ++s.wins;
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
        s.sharpe = sd > 0.0 ? (s.avg / sd) * std::sqrt(252.0) : 0.0;
    }
    return s;
}

int signal_dir(const Row& r, const Model& m) {
    double z = r.x[m.dim];
    switch (m.mode) {
        case 1: return z > m.threshold ? 1 : 0;
        case 2: return z > m.threshold ? -1 : 0;
        case 3: return z < -m.threshold ? 1 : 0;
        case 4: return z < -m.threshold ? -1 : 0;
        case 5:
            if (z > m.threshold) return 1;
            if (z < -m.threshold) return -1;
            return 0;
        case 6:
            if (z > m.threshold) return -1;
            if (z < -m.threshold) return 1;
            return 0;
        default: return 0;
    }
}

Stats eval_combo(const std::vector<Row>& rows, size_t lo, size_t hi, const Combo& c, double cost_mult = 1.0) {
    Stats s;
    hi = std::min(hi, rows.size());
    lo = std::min(lo, hi);
    s.n = hi - lo;
    std::vector<double> rets;
    rets.reserve(s.n);
    double eq = 0.0, peak = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        const Row& r = rows[i];
        int da = signal_dir(r, c.a);
        int db = signal_dir(r, c.b);
        if (da == 0 || da != db) continue;
        double pnl = static_cast<double>(da) * r.move_pips - cost_mult * r.spread_pips;
        ++s.trades;
        if (pnl > 0.0) ++s.wins;
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
        s.sharpe = sd > 0.0 ? (s.avg / sd) * std::sqrt(252.0) : 0.0;
    }
    return s;
}

void standardize(std::vector<Row>& rows, size_t train_end) {
    std::vector<double> mean(SEQ_SUMMARY_DIM, 0.0), sd(SEQ_SUMMARY_DIM, 1.0);
    train_end = std::min(train_end, rows.size());
    for (size_t i = 0; i < train_end; ++i) {
        for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) mean[d] += rows[i].x[d];
    }
    for (double& m : mean) m /= static_cast<double>(std::max<size_t>(1, train_end));
    for (size_t i = 0; i < train_end; ++i) {
        for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
            double diff = rows[i].x[d] - mean[d];
            sd[d] += diff * diff;
        }
    }
    for (double& s : sd) s = std::sqrt(s / static_cast<double>(std::max<size_t>(1, train_end)) + 1e-8);
    for (Row& r : rows) {
        for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
            double z = (r.x[d] - mean[d]) / sd[d];
            r.x[d] = static_cast<float>(std::clamp(z, -8.0, 8.0));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) args.xs.emplace_back(argv[i]);
    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string report = args.get("--report", "edge_search_report.json");
    size_t max_ticks = static_cast<size_t>(std::stoull(args.get("--ticks", "10000000")));
    size_t skip_ticks = static_cast<size_t>(std::stoull(args.get("--skip-ticks", "0")));
    size_t chunk_size = static_cast<size_t>(std::stoull(args.get("--chunk-size", "500000")));
    size_t min_trades = static_cast<size_t>(std::stoull(args.get("--min-trades", "200")));
    double cost_mult = std::stod(args.get("--cost-mult", "1.0"));
    size_t window = static_cast<size_t>(std::stoull(args.get("--window", "512")));
    size_t stride = static_cast<size_t>(std::stoull(args.get("--stride", "64")));
    size_t horizon = static_cast<size_t>(std::stoull(args.get("--horizon", "256")));
    float label_threshold = std::stof(args.get("--direction-threshold", "2.0"));

    Timer timer;
    ParquetReader reader(data);
    TickProcessor processor;
    Windower windower(window, stride, horizon, label_threshold);
    size_t total_rows = reader.total_rows();
    skip_ticks = std::min(skip_ticks, total_rows);
    size_t eval_ticks = std::min(max_ticks, total_rows - skip_ticks);
    std::vector<Row> rows;
    rows.reserve(eval_ticks / 64);

    std::cout << "EDGE SEARCH\n";
    std::cout << "  data: " << data << "\n";
    std::cout << "  skip/ticks: " << skip_ticks << " / " << eval_ticks << " / " << total_rows << "\n";
    std::cout << "  window/stride/horizon: " << window << "/" << stride << "/" << horizon << "\n";

    size_t read = 0;
    size_t skipped = 0;
    while (reader.has_next_chunk() && skipped < skip_ticks) {
        TickChunk ticks = reader.read_chunk(std::min(chunk_size, skip_ticks - skipped));
        if (ticks.size == 0) break;
        skipped += ticks.size;
    }
    while (reader.has_next_chunk() && read < eval_ticks) {
        TickChunk ticks = reader.read_chunk(std::min(chunk_size, eval_ticks - read));
        if (ticks.size == 0) break;
        read += ticks.size;
        FeatureChunk features = processor.process(ticks);
        std::vector<Sample> samples = windower.add(features);
        for (const Sample& s : samples) {
            Row r;
            r.x.resize(SEQ_SUMMARY_DIM);
            MultiScaleSummarizer::summarize(s, r.x.data());
            r.move_pips = (s.exit_mid - s.entry_mid) * 100.0;
            r.spread_pips = std::max(0.0, s.entry_spread * 100.0);
            size_t truth_cls = 0;
            for (size_t c = 1; c < 3; ++c) if (s.y_dir[c] > s.y_dir[truth_cls]) truth_cls = c;
            r.truth = truth_cls == 0 ? 1 : (truth_cls == 2 ? -1 : 0);
            r.env = s.env_id;
            rows.push_back(std::move(r));
        }
        if ((read / 1000000) != ((read - ticks.size) / 1000000)) {
            std::cout << "  read " << (read / 1000000) << "M ticks, samples=" << rows.size() << "\n";
        }
    }
    if (rows.size() < 1000) {
        std::cerr << "not enough samples\n";
        return 2;
    }

    size_t purge = 20;
    size_t train_end = rows.size() * 60 / 100;
    size_t val_begin = std::min(rows.size(), train_end + purge);
    size_t val_end = rows.size() * 80 / 100;
    size_t test_begin = std::min(rows.size(), val_end + purge);
    standardize(rows, train_end);

    std::vector<double> thresholds{0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0};
    std::vector<Model> candidates;
    std::vector<Model> gross_candidates;
    for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
        for (double th : thresholds) {
            for (int mode : {1, 2, 3, 4, 5, 6}) {
                Model m;
                m.dim = d;
                m.threshold = th;
                m.mode = mode;
                Stats tr = eval_model(rows, 0, train_end, m, cost_mult);
                Stats va = eval_model(rows, val_begin, val_end, m, cost_mult);
                if (tr.trades < min_trades || va.trades < min_trades) continue;
                m.train_sharpe = tr.sharpe;
                m.val_sharpe = va.sharpe;
                m.train_pnl = tr.pnl;
                m.val_pnl = va.pnl;
                m.train_trades = tr.trades;
                m.val_trades = va.trades;
                Stats tr_gross = eval_model(rows, 0, train_end, m, 0.0);
                Stats va_gross = eval_model(rows, val_begin, val_end, m, 0.0);
                if (tr_gross.trades >= min_trades && va_gross.trades >= min_trades &&
                    tr_gross.pnl > 0.0 && va_gross.pnl > 0.0) {
                    Model g = m;
                    g.train_sharpe = tr_gross.sharpe;
                    g.val_sharpe = va_gross.sharpe;
                    g.train_pnl = tr_gross.pnl;
                    g.val_pnl = va_gross.pnl;
                    g.train_trades = tr_gross.trades;
                    g.val_trades = va_gross.trades;
                    gross_candidates.push_back(g);
                }
                if (tr.pnl > 0.0 && va.pnl > 0.0) candidates.push_back(m);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Model& a, const Model& b) {
        return std::min(a.train_sharpe, a.val_sharpe) > std::min(b.train_sharpe, b.val_sharpe);
    });

    size_t keep = std::min<size_t>(25, candidates.size());
    for (size_t i = 0; i < keep; ++i) {
        Stats te = eval_model(rows, test_begin, rows.size(), candidates[i], cost_mult);
        candidates[i].test_sharpe = te.sharpe;
        candidates[i].test_pnl = te.pnl;
        candidates[i].test_trades = te.trades;
    }
    std::sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep), [](const Model& a, const Model& b) {
        return std::min({a.train_sharpe, a.val_sharpe, a.test_sharpe}) > std::min({b.train_sharpe, b.val_sharpe, b.test_sharpe});
    });

    bool found = false;
    Model best;
    Combo best_combo;
    bool combo_found = false;
    Stats best_test, best_stress;
    if (keep > 0) {
        best = candidates[0];
        best_test = eval_model(rows, test_begin, rows.size(), best, cost_mult);
        best_stress = eval_model(rows, test_begin, rows.size(), best, 2.0);
        found = best.train_pnl > 0.0 && best.val_pnl > 0.0 && best_test.pnl > 0.0 &&
                best_stress.pnl > 0.0 && best.train_trades >= min_trades &&
                best.val_trades >= min_trades && best_test.trades >= min_trades;
    }

    std::sort(gross_candidates.begin(), gross_candidates.end(), [](const Model& a, const Model& b) {
        return std::min(a.train_sharpe, a.val_sharpe) > std::min(b.train_sharpe, b.val_sharpe);
    });
    size_t combo_pool = std::min<size_t>(80, gross_candidates.size());
    std::vector<Combo> combos;
    for (size_t i = 0; i < combo_pool; ++i) {
        for (size_t j = i + 1; j < combo_pool; ++j) {
            if (gross_candidates[i].dim == gross_candidates[j].dim) continue;
            Combo c{gross_candidates[i], gross_candidates[j]};
            Stats tr = eval_combo(rows, 0, train_end, c, cost_mult);
            Stats va = eval_combo(rows, val_begin, val_end, c, cost_mult);
            if (tr.trades < min_trades || va.trades < min_trades) continue;
            if (tr.pnl <= 0.0 || va.pnl <= 0.0) continue;
            c.train_sharpe = tr.sharpe;
            c.val_sharpe = va.sharpe;
            c.train_pnl = tr.pnl;
            c.val_pnl = va.pnl;
            c.train_trades = tr.trades;
            c.val_trades = va.trades;
            combos.push_back(c);
        }
    }
    if (!combos.empty()) {
        std::sort(combos.begin(), combos.end(), [](const Combo& a, const Combo& b) {
            return std::min(a.train_sharpe, a.val_sharpe) > std::min(b.train_sharpe, b.val_sharpe);
        });
        best_combo = combos[0];
        Stats ct = eval_combo(rows, test_begin, rows.size(), best_combo, cost_mult);
        Stats cs = eval_combo(rows, test_begin, rows.size(), best_combo, 2.0);
        best_combo.test_pnl = ct.pnl;
        best_combo.test_sharpe = ct.sharpe;
        best_combo.test_trades = ct.trades;
        combo_found = ct.pnl > 0.0 && cs.pnl > 0.0 && ct.trades >= min_trades;
        found = found || combo_found;
    }

    std::filesystem::create_directories(std::filesystem::path(report).parent_path());
    std::ofstream os(report);
    os << std::fixed << std::setprecision(8);
    os << "{\n";
    os << "  \"verdict\":\"" << (found ? "CANDIDATE_FOUND" : "NO_EDGE_FOUND") << "\",\n";
    os << "  \"data\":{\"path\":\"" << json_escape(data) << "\",\"total_rows\":" << total_rows
       << ",\"skip_ticks\":" << skipped
       << ",\"evaluated_ticks\":" << read << ",\"samples\":" << rows.size()
       << ",\"cost_mult\":" << cost_mult
       << ",\"window\":" << window << ",\"stride\":" << stride
       << ",\"horizon\":" << horizon << "},\n";
    os << "  \"split\":{\"train\":" << train_end << ",\"val_begin\":" << val_begin
       << ",\"val_end\":" << val_end << ",\"test_begin\":" << test_begin << "},\n";
    os << "  \"candidate_count\":" << candidates.size() << ",\n";
    os << "  \"gross_candidate_count\":" << gross_candidates.size() << ",\n";
    os << "  \"combo_candidate_count\":" << combos.size() << ",\n";
    os << "  \"best\":{";
    if (keep > 0) {
        os << "\"dim\":" << best.dim << ",\"threshold\":" << best.threshold << ",\"mode\":" << best.mode
           << ",\"train_pnl\":" << best.train_pnl << ",\"train_sharpe\":" << best.train_sharpe
           << ",\"train_trades\":" << best.train_trades
           << ",\"val_pnl\":" << best.val_pnl << ",\"val_sharpe\":" << best.val_sharpe
           << ",\"val_trades\":" << best.val_trades
           << ",\"test_pnl\":" << best_test.pnl << ",\"test_sharpe\":" << best_test.sharpe
           << ",\"test_trades\":" << best_test.trades
           << ",\"test_2x_cost_pnl\":" << best_stress.pnl
           << ",\"test_2x_cost_sharpe\":" << best_stress.sharpe;
    }
    os << "},\n";
    os << "  \"best_combo\":{";
    if (!combos.empty()) {
        Stats ct = eval_combo(rows, test_begin, rows.size(), best_combo, cost_mult);
        Stats cs = eval_combo(rows, test_begin, rows.size(), best_combo, 2.0);
        os << "\"a_dim\":" << best_combo.a.dim << ",\"a_threshold\":" << best_combo.a.threshold
           << ",\"a_mode\":" << best_combo.a.mode
           << ",\"b_dim\":" << best_combo.b.dim << ",\"b_threshold\":" << best_combo.b.threshold
           << ",\"b_mode\":" << best_combo.b.mode
           << ",\"train_pnl\":" << best_combo.train_pnl << ",\"train_sharpe\":" << best_combo.train_sharpe
           << ",\"train_trades\":" << best_combo.train_trades
           << ",\"val_pnl\":" << best_combo.val_pnl << ",\"val_sharpe\":" << best_combo.val_sharpe
           << ",\"val_trades\":" << best_combo.val_trades
           << ",\"test_pnl\":" << ct.pnl << ",\"test_sharpe\":" << ct.sharpe
           << ",\"test_trades\":" << ct.trades
           << ",\"test_2x_cost_pnl\":" << cs.pnl << ",\"test_2x_cost_sharpe\":" << cs.sharpe;
    }
    os << "},\n";
    os << "  \"top\":[\n";
    for (size_t i = 0; i < keep; ++i) {
        Stats te = eval_model(rows, test_begin, rows.size(), candidates[i], cost_mult);
        Stats st = eval_model(rows, test_begin, rows.size(), candidates[i], 2.0);
        os << "    {\"rank\":" << (i + 1) << ",\"dim\":" << candidates[i].dim
           << ",\"threshold\":" << candidates[i].threshold << ",\"mode\":" << candidates[i].mode
           << ",\"train_sharpe\":" << candidates[i].train_sharpe
           << ",\"val_sharpe\":" << candidates[i].val_sharpe
           << ",\"test_sharpe\":" << te.sharpe
           << ",\"test_pnl\":" << te.pnl
           << ",\"test_2x_cost_pnl\":" << st.pnl << "}"
           << (i + 1 == keep ? "\n" : ",\n");
    }
    os << "  ],\n";
    os << "  \"elapsed_seconds\":" << timer.elapsed_seconds() << "\n";
    os << "}\n";

    std::cout << "REPORT " << report << "\n";
    std::cout << "VERDICT " << (found ? "CANDIDATE_FOUND" : "NO_EDGE_FOUND") << "\n";
    if (keep > 0) {
        std::cout << "best dim=" << best.dim << " th=" << best.threshold << " mode=" << best.mode
                  << " train_pnl=" << best.train_pnl << " val_pnl=" << best.val_pnl
                  << " test_pnl=" << best_test.pnl << " test_2x=" << best_stress.pnl << "\n";
    }
    if (!combos.empty()) {
        Stats ct = eval_combo(rows, test_begin, rows.size(), best_combo, cost_mult);
        Stats cs = eval_combo(rows, test_begin, rows.size(), best_combo, 2.0);
        std::cout << "best_combo a=" << best_combo.a.dim << "/" << best_combo.a.threshold << "/" << best_combo.a.mode
                  << " b=" << best_combo.b.dim << "/" << best_combo.b.threshold << "/" << best_combo.b.mode
                  << " train_pnl=" << best_combo.train_pnl << " val_pnl=" << best_combo.val_pnl
                  << " test_pnl=" << ct.pnl << " test_2x=" << cs.pnl << "\n";
    }
    return found ? 0 : 3;
}
