#include "commands.hpp"

#include "data/normalizer.hpp"
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

namespace from {
namespace {

size_t arg_size(const CliArgs& args, const std::string& key, size_t def) {
    std::string v = args.get(key, "");
    return v.empty() ? def : static_cast<size_t>(std::stoull(v));
}

float arg_float(const CliArgs& args, const std::string& key, float def) {
    std::string v = args.get(key, "");
    return v.empty() ? def : std::stof(v);
}

struct EvalRow {
    int pred = 0;
    int truth = 0;
    int momentum = 0;
    int contrarian = 0;
    int env = 0;
    int session = 0;
    float conf = 0.0f;
    double pnl = 0.0;
    double pnl_2x = 0.0;
    double base_mom_pnl = 0.0;
    double base_contra_pnl = 0.0;
    double spread_pips = 0.0;
};

struct Stats {
    size_t n = 0;
    size_t trades = 0;
    size_t correct = 0;
    size_t wins = 0;
    double pnl = 0.0;
    double mean = 0.0;
    double stdev = 0.0;
    double sharpe = 0.0;
    double max_dd = 0.0;
};

int cls_to_dir(size_t cls) {
    if (cls == 0) return 1;
    if (cls == 2) return -1;
    return 0;
}

int session_from_ms(int64_t ts) {
    int64_t ms_day = 24LL * 60LL * 60LL * 1000LL;
    int64_t t = ((ts % ms_day) + ms_day) % ms_day;
    int hour = static_cast<int>(t / (60LL * 60LL * 1000LL));
    if (hour >= 7 && hour < 12) return 1;      // London
    if (hour >= 13 && hour < 17) return 2;     // New York
    if (hour >= 12 && hour < 13) return 3;     // overlap
    if (hour >= 23 || hour < 4) return 4;      // Asia
    return 0;
}

const char* session_name(int s) {
    switch (s) {
        case 1: return "london";
        case 2: return "newyork";
        case 3: return "overlap";
        case 4: return "asian";
        default: return "other";
    }
}

Stats compute_stats(const std::vector<EvalRow>& rows, bool use_2x_cost = false,
                    bool use_momentum = false, bool use_contrarian = false) {
    Stats s;
    s.n = rows.size();
    std::vector<double> returns;
    returns.reserve(rows.size());
    double equity = 0.0;
    double peak = 0.0;
    for (const auto& r : rows) {
        int dir = r.pred;
        double pnl = use_2x_cost ? r.pnl_2x : r.pnl;
        if (use_momentum) {
            dir = r.momentum;
            pnl = r.base_mom_pnl;
        } else if (use_contrarian) {
            dir = r.contrarian;
            pnl = r.base_contra_pnl;
        }
        if (dir != 0) {
            ++s.trades;
            if (dir == r.truth) ++s.correct;
            if (pnl > 0.0) ++s.wins;
            s.pnl += pnl;
            returns.push_back(pnl);
            equity += pnl;
            peak = std::max(peak, equity);
            s.max_dd = std::max(s.max_dd, peak - equity);
        }
    }
    if (!returns.empty()) {
        s.mean = s.pnl / static_cast<double>(returns.size());
        double var = 0.0;
        for (double x : returns) {
            double d = x - s.mean;
            var += d * d;
        }
        s.stdev = returns.size() > 1 ? std::sqrt(var / static_cast<double>(returns.size() - 1)) : 0.0;
        s.sharpe = s.stdev > 0.0 ? (s.mean / s.stdev) * std::sqrt(252.0) : 0.0;
    }
    return s;
}

double quantile(std::vector<double> xs, double q) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    double pos = q * static_cast<double>(xs.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    double w = pos - static_cast<double>(lo);
    return xs[lo] * (1.0 - w) + xs[hi] * w;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

void write_stats_json(std::ostream& os, const char* name, const Stats& s, bool comma) {
    os << "    \"" << name << "\": {"
       << "\"samples\":" << s.n
       << ",\"trades\":" << s.trades
       << ",\"accuracy\":" << (s.trades ? static_cast<double>(s.correct) / static_cast<double>(s.trades) : 0.0)
       << ",\"win_rate\":" << (s.trades ? static_cast<double>(s.wins) / static_cast<double>(s.trades) : 0.0)
       << ",\"pnl_pips\":" << s.pnl
       << ",\"avg_pnl_pips\":" << s.mean
       << ",\"max_drawdown_pips\":" << s.max_dd
       << ",\"sharpe\":" << s.sharpe
       << "}" << (comma ? "," : "") << "\n";
}

}  // namespace

int run_validate_adversarial(const CliArgs& args) {
    std::string model_path = args.get("--model", "weights_best.from");
    std::string data_path = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string report_path = args.get("--report", "adversarial_validation_report.json");
    std::string data_sha256 = args.get("--data-sha256", "");
    size_t max_ticks = arg_size(args, "--ticks", 2000000);
    size_t chunk_size = arg_size(args, "--chunk-size", 500000);
    size_t window = arg_size(args, "--window", 512);
    size_t stride = arg_size(args, "--stride", 64);
    size_t horizon = arg_size(args, "--horizon", 256);
    float threshold_mult = arg_float(args, "--direction-threshold", 2.0f);
    float conf_threshold = arg_float(args, "--confidence", 0.55f);

    require(std::filesystem::exists(model_path), "Model not found: " + model_path);
    require(std::filesystem::exists(data_path), "Data not found: " + data_path);

    SequenceModel model(0.00005f, 42);
    require(SequenceModelIO::load(model_path, model), "Model is not a loadable FSQ2/FSQ3 sequence checkpoint: " + model_path);
    require(model.feat_norm_ready, "Checkpoint has no second-pass feature normalization; validation would be invalid.");

    ParquetReader reader(data_path);
    TickProcessor processor;
    Normalizer normalizer(FROM_MAX_FEATURES);
    Windower windower(window, stride, horizon, threshold_mult);

    const uintmax_t file_size = std::filesystem::file_size(data_path);
    const size_t total_rows = reader.total_rows();
    const size_t eval_ticks = std::min(max_ticks, total_rows);

    std::cout << "ADVERSARIAL VALIDATION\n";
    std::cout << "  data:  " << data_path << "\n";
    std::cout << "  model: " << model_path << "\n";
    std::cout << "  ticks: " << eval_ticks << " / " << total_rows << "\n";

    size_t rows_read = 0;
    size_t duplicate_ts = 0;
    size_t backward_ts = 0;
    size_t large_gaps = 0;
    int64_t prev_ts = std::numeric_limits<int64_t>::min();
    double max_gap_sec = 0.0;
    double spread_sum = 0.0;
    size_t spread_n = 0;
    size_t feature_checks = 0;
    size_t feature_mismatches = 0;
    std::array<size_t, 3> label_counts{0, 0, 0};
    std::array<size_t, 3> pred_counts{0, 0, 0};

    std::vector<EvalRow> rows;
    rows.reserve(eval_ticks / std::max<size_t>(stride, 1));

    Timer timer;
    while (reader.has_next_chunk() && rows_read < eval_ticks) {
        size_t want = std::min(chunk_size, eval_ticks - rows_read);
        TickChunk ticks = reader.read_chunk(want);
        if (ticks.size == 0) break;
        rows_read += ticks.size;

        for (size_t i = 0; i < ticks.size; ++i) {
            int64_t ts = ticks.time_ms[i];
            if (prev_ts != std::numeric_limits<int64_t>::min()) {
                if (ts == prev_ts) ++duplicate_ts;
                if (ts < prev_ts) ++backward_ts;
                double gap = static_cast<double>(ts - prev_ts) / 1000.0;
                if (gap > 60.0) ++large_gaps;
                max_gap_sec = std::max(max_gap_sec, gap);
            }
            prev_ts = ts;
        }

        FeatureChunk features = processor.process(ticks);

        size_t check_n = std::min<size_t>(ticks.size, 1024);
        for (size_t i = 0; i < check_n; ++i) {
            double ask = ticks.ask[i];
            double bid = ticks.bid[i];
            double mid = ticks.mid[i] != 0.0 ? ticks.mid[i] : (ask + bid) * 0.5;
            double av = ticks.ask_vol[i];
            double bv = ticks.bid_vol[i];
            double spread = ask - bid;
            double norm_spread = spread / (mid + 1e-5);
            double ofi = (av - bv) / (av + bv + 1e-8);
            const std::array<double, 5> expected{ask, bid, mid, spread, norm_spread};
            const std::array<size_t, 5> idx{FROM_FEAT_ASK, FROM_FEAT_BID, FROM_FEAT_MID, FROM_FEAT_SPREAD, FROM_FEAT_NORM_SPREAD};
            for (size_t k = 0; k < expected.size(); ++k) {
                ++feature_checks;
                double got = features.features.at(i, idx[k]);
                double tol = std::max(1e-5, std::abs(expected[k]) * 1e-6);
                if (std::abs(got - expected[k]) > tol) ++feature_mismatches;
            }
            ++feature_checks;
            if (std::abs(features.features.at(i, FROM_FEAT_OFI) - ofi) > 1e-5) ++feature_mismatches;
        }

        features.features = normalizer.normalize_chunk(features.features, false);
        std::vector<Sample> samples = windower.add(features);

        for (const Sample& s : samples) {
            float summary[SEQ_SUMMARY_DIM];
            MultiScaleSummarizer::summarize(s, summary);
            model.apply_feat_norm(summary, 1);
            float logits[SEQ_NUM_CLASSES];
            float probs[SEQ_NUM_CLASSES];
            model.forward(summary, 1, logits, false);
            SequenceModel::softmax(logits, 1, probs);
            size_t cls = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                if (probs[c] > probs[cls]) cls = c;
            }

            size_t truth_cls = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                if (s.y_dir[c] > s.y_dir[truth_cls]) truth_cls = c;
            }
            label_counts[truth_cls]++;
            pred_counts[cls]++;

            int pred = cls_to_dir(cls);
            float conf = probs[cls];
            if (conf < conf_threshold) pred = 0;
            int truth = cls_to_dir(truth_cls);
            double move_pips = (s.exit_mid - s.entry_mid) * 100.0;
            double spread_pips = std::max(0.0, s.entry_spread * 100.0);
            spread_sum += spread_pips;
            ++spread_n;
            int mom = 0;
            double first_mid = s.X.at(0, FROM_FEAT_MID);
            if (s.entry_mid > first_mid) mom = 1;
            else if (s.entry_mid < first_mid) mom = -1;

            EvalRow r;
            r.pred = pred;
            r.truth = truth;
            r.momentum = mom;
            r.contrarian = -mom;
            r.env = s.env_id;
            r.session = session_from_ms(features.time_ms.empty() ? 0 : features.time_ms.back());
            r.conf = conf;
            r.pnl = pred == 0 ? 0.0 : static_cast<double>(pred) * move_pips - spread_pips;
            r.pnl_2x = pred == 0 ? 0.0 : static_cast<double>(pred) * move_pips - 2.0 * spread_pips;
            r.base_mom_pnl = mom == 0 ? 0.0 : static_cast<double>(mom) * move_pips - spread_pips;
            r.base_contra_pnl = mom == 0 ? 0.0 : static_cast<double>(-mom) * move_pips - spread_pips;
            r.spread_pips = spread_pips;
            rows.push_back(r);
        }

        if ((rows_read / 1000000) != ((rows_read - ticks.size) / 1000000)) {
            std::cout << "  read " << (rows_read / 1000000) << "M ticks, samples=" << rows.size() << "\n";
        }
    }

    require(!rows.empty(), "No validation samples produced; lower window/horizon or provide more ticks.");

    Stats model_stats = compute_stats(rows);
    Stats stress_stats = compute_stats(rows, true);
    Stats momentum_stats = compute_stats(rows, false, true);
    Stats contra_stats = compute_stats(rows, false, false, true);

    std::vector<double> fold_sharpes;
    std::vector<double> fold_pnls;
    const size_t folds = std::min<size_t>(10, std::max<size_t>(2, rows.size() / 200));
    for (size_t f = 0; f < folds; ++f) {
        size_t lo = rows.size() * f / folds;
        size_t hi = rows.size() * (f + 1) / folds;
        std::vector<EvalRow> part(rows.begin() + static_cast<std::ptrdiff_t>(lo),
                                  rows.begin() + static_cast<std::ptrdiff_t>(hi));
        Stats fs = compute_stats(part);
        fold_sharpes.push_back(fs.sharpe);
        fold_pnls.push_back(fs.pnl);
    }
    double worst_decile_sharpe = quantile(fold_sharpes, 0.10);
    double median_fold_sharpe = quantile(fold_sharpes, 0.50);

    std::mt19937 rng(12345);
    std::vector<int> labels;
    labels.reserve(rows.size());
    for (const auto& r : rows) labels.push_back(r.truth);
    size_t perm_trials = 100;
    size_t perm_better_acc = 0;
    double observed_acc = model_stats.trades ? static_cast<double>(model_stats.correct) / static_cast<double>(model_stats.trades) : 0.0;
    for (size_t t = 0; t < perm_trials; ++t) {
        std::shuffle(labels.begin(), labels.end(), rng);
        size_t correct = 0;
        size_t trades = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].pred == 0) continue;
            ++trades;
            if (rows[i].pred == labels[i]) ++correct;
        }
        double acc = trades ? static_cast<double>(correct) / static_cast<double>(trades) : 0.0;
        if (acc >= observed_acc) ++perm_better_acc;
    }
    double permutation_p = (static_cast<double>(perm_better_acc) + 1.0) / (static_cast<double>(perm_trials) + 1.0);

    std::array<Stats, 7> regime_stats;
    for (int e = 0; e < 7; ++e) {
        std::vector<EvalRow> part;
        for (const auto& r : rows) if (r.env == e) part.push_back(r);
        regime_stats[e] = compute_stats(part);
    }
    std::array<Stats, 5> session_stats;
    for (int s = 0; s < 5; ++s) {
        std::vector<EvalRow> part;
        for (const auto& r : rows) if (r.session == s) part.push_back(r);
        session_stats[s] = compute_stats(part);
    }

    bool killed = false;
    std::vector<std::string> kill_reasons;
    auto kill = [&](const std::string& reason) {
        killed = true;
        kill_reasons.push_back(reason);
    };
    if (feature_mismatches > 0) kill("independent feature recomputation mismatch");
    if (backward_ts > 0 || duplicate_ts > 0) kill("timestamp order/duplicate failure");
    if (worst_decile_sharpe <= 0.0) kill("CPCV-style worst-decile Sharpe is non-positive");
    if (model_stats.trades == 0) kill("model produced no confident trades");
    if (stress_stats.pnl <= 0.0 || stress_stats.sharpe <= 0.0) kill("edge does not survive 2x spread/slippage stress");
    if (momentum_stats.pnl >= model_stats.pnl) kill("model does not beat dumb momentum baseline");
    if (contra_stats.pnl >= model_stats.pnl) kill("model does not beat dumb contrarian baseline");
    if (permutation_p > 0.10) kill("label permutation does not distinguish model accuracy from shuffled labels");
    size_t positive_regimes = 0;
    for (const auto& rs : regime_stats) if (rs.trades > 0 && rs.pnl > 0.0) ++positive_regimes;
    if (positive_regimes <= 1) kill("performance is concentrated in at most one regime");

    std::filesystem::create_directories(std::filesystem::path(report_path).parent_path());
    std::ofstream os(report_path);
    require(static_cast<bool>(os), "Cannot write report: " + report_path);
    os << std::fixed << std::setprecision(8);
    os << "{\n";
    os << "  \"verdict\":\"" << (killed ? "KILL" : "SURVIVES_LIMITED_GAUNTLET") << "\",\n";
    os << "  \"data\":{"
       << "\"path\":\"" << json_escape(data_path) << "\","
       << "\"sha256\":\"" << json_escape(data_sha256) << "\","
       << "\"file_size_bytes\":" << file_size << ","
       << "\"total_rows\":" << total_rows << ","
       << "\"evaluated_ticks\":" << rows_read << ","
       << "\"samples\":" << rows.size() << "},\n";
    os << "  \"model\":{"
       << "\"path\":\"" << json_escape(model_path) << "\","
       << "\"feat_norm_ready\":" << (model.feat_norm_ready ? "true" : "false") << ","
       << "\"confidence_threshold\":" << conf_threshold << "},\n";
    os << "  \"data_integrity\":{"
       << "\"duplicate_timestamps\":" << duplicate_ts << ","
       << "\"backward_timestamps\":" << backward_ts << ","
       << "\"large_gaps_over_60s\":" << large_gaps << ","
       << "\"max_gap_seconds\":" << max_gap_sec << ","
       << "\"feature_checks\":" << feature_checks << ","
       << "\"feature_mismatches\":" << feature_mismatches << ","
       << "\"mean_spread_pips\":" << (spread_n ? spread_sum / static_cast<double>(spread_n) : 0.0) << "},\n";
    os << "  \"label_counts\":{\"long\":" << label_counts[0] << ",\"flat\":" << label_counts[1] << ",\"short\":" << label_counts[2] << "},\n";
    os << "  \"pred_counts\":{\"long\":" << pred_counts[0] << ",\"flat\":" << pred_counts[1] << ",\"short\":" << pred_counts[2] << "},\n";
    os << "  \"metrics\":{\n";
    write_stats_json(os, "model", model_stats, true);
    write_stats_json(os, "model_2x_cost", stress_stats, true);
    write_stats_json(os, "momentum_baseline", momentum_stats, true);
    write_stats_json(os, "contrarian_baseline", contra_stats, false);
    os << "  },\n";
    os << "  \"cpcv_style\":{\"folds\":" << folds
       << ",\"worst_decile_sharpe\":" << worst_decile_sharpe
       << ",\"median_fold_sharpe\":" << median_fold_sharpe << "},\n";
    os << "  \"permutation\":{\"trials\":" << perm_trials
       << ",\"observed_trade_accuracy\":" << observed_acc
       << ",\"p_value\":" << permutation_p << "},\n";
    os << "  \"regimes\":[\n";
    for (size_t i = 0; i < regime_stats.size(); ++i) {
        os << "    {\"env\":" << i << ",\"trades\":" << regime_stats[i].trades
           << ",\"pnl_pips\":" << regime_stats[i].pnl
           << ",\"sharpe\":" << regime_stats[i].sharpe << "}"
           << (i + 1 == regime_stats.size() ? "\n" : ",\n");
    }
    os << "  ],\n";
    os << "  \"sessions\":[\n";
    for (size_t i = 0; i < session_stats.size(); ++i) {
        os << "    {\"session\":\"" << session_name(static_cast<int>(i)) << "\",\"trades\":" << session_stats[i].trades
           << ",\"pnl_pips\":" << session_stats[i].pnl
           << ",\"sharpe\":" << session_stats[i].sharpe << "}"
           << (i + 1 == session_stats.size() ? "\n" : ",\n");
    }
    os << "  ],\n";
    os << "  \"kill_reasons\":[";
    for (size_t i = 0; i < kill_reasons.size(); ++i) {
        os << "\"" << kill_reasons[i] << "\"" << (i + 1 == kill_reasons.size() ? "" : ",");
    }
    os << "],\n";
    os << "  \"limitations\":["
       << "\"This is a fixed-checkpoint adversarial validation, not nested retraining CPCV.\","
       << "\"DSR/PBO/SPA are approximated by fold distribution, baselines, and permutation because the repo does not track the full configuration trial universe.\","
       << "\"Feature importance stability is not available for this hand-written MLP checkpoint format.\""
       << "],\n";
    os << "  \"elapsed_seconds\":" << timer.elapsed_seconds() << "\n";
    os << "}\n";

    std::cout << "REPORT " << report_path << "\n";
    std::cout << "VERDICT " << (killed ? "KILL" : "SURVIVES_LIMITED_GAUNTLET") << "\n";
    std::cout << "model pnl=" << model_stats.pnl << " sharpe=" << model_stats.sharpe
              << " trades=" << model_stats.trades << " worst_decile_sharpe=" << worst_decile_sharpe << "\n";
    if (killed) {
        for (const auto& r : kill_reasons) std::cout << "KILL: " << r << "\n";
    }
    return killed ? 3 : 0;
}

}  // namespace from
