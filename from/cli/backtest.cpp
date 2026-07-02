#include "commands.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "io/artifact.hpp"
#include "model/sequence_model.hpp"
#include "model/meta_labeler.hpp"
#include "utils/config_parser.hpp"
#include "utils/timer.hpp"
#include "wf_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace from {

// ============================================================================
// backtest — honest, after-cost standalone backtest of a trained SequenceModel
// artifact, scored on the SAME triple-barrier labels the walk-forward harness
// trades. Adds three things the old delta-threshold backtest lacked:
//
//   1. Temperature calibration. The confidence gate is meaningless unless the
//      probabilities are calibrated. We fit T on the first --calib-frac of the
//      eval stream (in-sample to calibration only) and apply it to the rest,
//      reporting ECE before/after.
//   2. Position sizing. Confidence-proportional size capped at a fraction of the
//      Kelly stake estimated from the calibration slice (no look-ahead). Cost
//      scales with size, so over-trading is penalized.
//   3. Honest metrics. PnL/winrate/PF/Kelly/maxDD + overlap-deflated N_eff, CI,
//      Probabilistic & Deflated Sharpe, and a per-regime breakdown.
//
// First-pass normalization: load the artifact's norm1.bin if present (correct);
// otherwise fit Welford on a warmup prefix and freeze (those ticks don't trade).
// ============================================================================

namespace {

// Argmax of a 3-class row.
inline int argmax3(const float* p) {
    int m = 0;
    if (p[1] > p[m]) m = 1;
    if (p[2] > p[m]) m = 2;
    return m;
}

// Per-trade record kept for calibration, sizing, and post-hoc scoring.
struct Pred {
    float logit[3];
    uint8_t tb_label;   // 0 UP / 1 NEUTRAL / 2 DOWN
    uint8_t y_label;    // naive delta-threshold label (for the accuracy gap)
    float ret;          // tb_ret  (signed return entry->first-touch)
    float cost;         // tb_cost (one round-trip spread, return units)
    double entry_mid;   // price denominator for configured USD execution costs
    int64_t entry_time_ms;
    int env_id;
};

// Softmax of a temperature-scaled logit row into probs[3].
inline void softmax_T(const float* logit, float T, float* probs) {
    float z0 = logit[0] / T, z1 = logit[1] / T, z2 = logit[2] / T;
    float mx = std::max({z0, z1, z2});
    float e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx), e2 = std::exp(z2 - mx);
    float s = e0 + e1 + e2 + 1e-20f;
    probs[0] = e0 / s; probs[1] = e1 / s; probs[2] = e2 / s;
}

// Fit a single temperature by gradient descent on NLL over a calibration slice.
float fit_temperature(const std::vector<Pred>& preds, size_t lo, size_t hi) {
    float T = 1.0f;
    if (hi <= lo) return T;
    for (int step = 0; step < 300; ++step) {
        double grad = 0.0;
        for (size_t i = lo; i < hi; ++i) {
            float p[3];
            softmax_T(preds[i].logit, T, p);
            int y = preds[i].tb_label;
            // d NLL / dT for temperature scaling: sum_c (p_c - 1{c=y}) * logit_c / T^2
            for (int c = 0; c < 3; ++c) {
                float ind = (c == y) ? 1.0f : 0.0f;
                grad += (p[c] - ind) * preds[i].logit[c];
            }
        }
        grad /= (T * T) * static_cast<double>(hi - lo);
        T -= 0.05f * static_cast<float>(grad);
        if (T < 0.05f) T = 0.05f;
        if (T > 20.0f) T = 20.0f;
    }
    return T;
}

// Expected Calibration Error over a slice at temperature T (10 bins).
float ece(const std::vector<Pred>& preds, size_t lo, size_t hi, float T, size_t bins = 10) {
    std::vector<double> conf_sum(bins, 0.0), acc_sum(bins, 0.0), cnt(bins, 0.0);
    for (size_t i = lo; i < hi; ++i) {
        float p[3]; softmax_T(preds[i].logit, T, p);
        int pred = argmax3(p);
        float conf = p[pred];
        size_t b = std::min(bins - 1, static_cast<size_t>(conf * bins));
        conf_sum[b] += conf;
        acc_sum[b] += (pred == preds[i].tb_label) ? 1.0 : 0.0;
        cnt[b] += 1.0;
    }
    double total = 0.0; for (double c : cnt) total += c;
    if (total <= 0.0) return 0.0;
    double e = 0.0;
    for (size_t b = 0; b < bins; ++b)
        if (cnt[b] > 0.0)
            e += (cnt[b] / total) * std::abs(conf_sum[b] / cnt[b] - acc_sum[b] / cnt[b]);
    return static_cast<float>(e);
}

const char* regime_name(int env) {
    switch (env) {
        case 0: return "london";   case 1: return "newyork"; case 2: return "asian";
        case 3: return "high-vol"; case 4: return "low-vol"; case 5: return "easy";
        default: return "hard";
    }
}

}  // namespace

int run_backtest(const CliArgs& args) {
    std::string model_path = args.get("--model", "weights_best.from");
    std::string data_path  = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string norm1_path = args.get("--norm1", "");  // empty => auto-detect / warmup-fit
    size_t ticks       = static_cast<size_t>(std::stoull(args.get("--ticks", "5000000")));

    size_t window      = static_cast<size_t>(std::stoull(args.get("--window", "512")));
    size_t stride      = static_cast<size_t>(std::stoull(args.get("--stride", "64")));
    size_t horizon     = static_cast<size_t>(std::stoull(args.get("--horizon", "256")));
    float  dir_thresh  = std::stof(args.get("--direction-threshold", "2.0"));
    float  barrier_k   = std::stof(args.get("--barrier-k", "1.0"));
    float  cost_mult   = std::stof(args.get("--cost-mult", "1.5"));

    Config config;
    const std::string config_path = args.get("--config", "config.toml");
    if (std::filesystem::exists(config_path)) config.load(config_path);
    const float lot_size = config.get_float("costs.lot_size", 100.0f);
    const float commission_per_lot = config.get_float("costs.commission_per_lot", 3.50f);
    const float slippage_usd = config.get_float("costs.slippage_usd", 0.05f);

    float  conf_gate     = std::stof(args.get("--conf-gate", "0.50"));
    const float commission_per_lot_arg = std::stof(args.get("--commission-per-lot", std::to_string(commission_per_lot)));
    const float slippage_usd_arg = std::stof(args.get("--slippage-usd", std::to_string(slippage_usd)));
    const bool use_fixed_spread = args.has("--spread-usd");
    const float spread_usd_arg = std::stof(args.get("--spread-usd", "0"));
    float  calib_frac    = std::stof(args.get("--calib-frac", "0.20"));      // eval prefix used to fit T + Kelly
    float  kelly_frac    = std::stof(args.get("--kelly-frac", "0.25"));      // fractional-Kelly cap
    float  size_cap      = std::stof(args.get("--size-cap", "1.0"));         // max position units
    size_t n_trials      = static_cast<size_t>(std::stoull(args.get("--trials", "1")));  // for DSR
    bool   no_calib      = args.has("--no-calib");
    bool   use_meta      = args.has("--meta");
    float  meta_gate     = std::stof(args.get("--meta-gate", "0.50"));
    float  warmup_frac   = std::stof(args.get("--warmup-frac", "0.05"));     // only if norm1 must be fit

    require(lot_size > 0.0f && commission_per_lot_arg >= 0.0f && slippage_usd_arg >= 0.0f &&
                (!use_fixed_spread || spread_usd_arg >= 0.0f),
            "Execution costs must be non-negative and lot_size must be positive");
    auto execution_cost = [&](const Pred& p) {
        const float price = static_cast<float>(p.entry_mid);
        require(price > 0.0f, "Cannot apply USD execution costs to a non-positive entry price");
        const float spread_cost = use_fixed_spread ? spread_usd_arg / price : p.cost;
        return spread_cost + (commission_per_lot_arg / lot_size + slippage_usd_arg) / price;
    };

    require(std::filesystem::exists(model_path), "Model not found: " + model_path);
    require(std::filesystem::exists(data_path), "Data not found: " + data_path);

    // ---- Load model artifact (FSQ3 carries the 2nd-pass feat_norm) ----
    SequenceModel model(0.00005f, 42);
    require(SequenceModelIO::load(model_path, model), "Failed to load model: " + model_path);
    require(model.feat_norm_ready,
            "Model has no 2nd-pass feat_norm; retrain with current code.");

    // ---- First-pass Welford normalizer ----
    Normalizer norm1(FROM_MAX_FEATURES);
    bool norm1_loaded = false;
    if (norm1_path.empty()) {
        // Auto-detect: sibling norm1.bin, or "<model>.norm1".
        std::filesystem::path mp(model_path);
        std::filesystem::path cand1 = mp.parent_path() / "norm1.bin";
        std::filesystem::path cand2 = mp.string() + ".norm1";
        if (std::filesystem::exists(cand1)) norm1_path = cand1.string();
        else if (std::filesystem::exists(cand2)) norm1_path = cand2.string();
    }
    if (!norm1_path.empty() && std::filesystem::exists(norm1_path)) {
        norm1_loaded = io::load_norm1(norm1_path, norm1);
        if (norm1_loaded) norm1.freeze();
    }

    ParquetReader reader(data_path);
    TickProcessor processor;
    Windower windower(window, stride, horizon, dir_thresh);
    windower.set_barriers(barrier_k, barrier_k, cost_mult);

    size_t total_rows = reader.total_rows();
    size_t eval_ticks = std::min(ticks, total_rows);
    size_t skip_rows  = total_rows > eval_ticks ? total_rows - eval_ticks : 0;

    std::cout << "=======================================================================\n"
              << "BACKTEST (triple-barrier, after-cost): " << model_path << "\n"
              << "=======================================================================\n"
              << "Data:        " << data_path << " (" << (total_rows / 1000000) << "M rows)\n"
              << "Eval window: last " << (eval_ticks / 1000000) << "M ticks\n"
              << "norm1:       " << (norm1_loaded ? norm1_path : std::string("(warmup-fit)")) << "\n"
              << "conf-gate:   " << conf_gate << "   calib-frac: " << calib_frac
              << "   kelly-frac: " << kelly_frac << "\n"
              << "costs:       " << (use_fixed_spread ? std::string("$") + std::to_string(spread_usd_arg) : std::string("observed spread"))
              << " + $" << commission_per_lot_arg << "/lot commission + $"
              << slippage_usd_arg << " slippage (lot=" << lot_size << ")\n"
              << "=======================================================================\n\n";

    // ---- Skip to eval region; if norm1 unknown, fit Welford on this prefix ----
    size_t rows_read = 0;
    while (rows_read < skip_rows && reader.has_next_chunk()) {
        size_t cs = std::min<size_t>(2000000, skip_rows - rows_read);
        TickChunk tc = reader.read_chunk(cs);
        if (tc.size == 0) break;
        rows_read += tc.size;
        FeatureChunk f = processor.process(tc);
        // If we have norm1, keep it frozen; else fit Welford on the (pre-eval) tail.
        f.features = norm1.normalize_chunk(f.features, !norm1_loaded);
        windower.add(f);  // keep windower warm so the first eval sample is valid
    }
    if (!norm1_loaded) { norm1.freeze(); }
    std::cout << "Skipped " << (rows_read / 1000000) << "M rows to reach eval window"
              << (norm1_loaded ? ".\n" : " (Welford fit on prefix, then frozen).\n");

    // ---- Stream eval region; collect one Pred per emitted sample ----
    std::vector<Pred> preds;
    preds.reserve(eval_ticks / std::max<size_t>(1, stride));
    float logits[3];
    Timer t;
    size_t eval_read = 0;
    while (reader.has_next_chunk() && eval_read < eval_ticks) {
        size_t cs = std::min<size_t>(2000000, eval_ticks - eval_read);
        TickChunk tc = reader.read_chunk(cs);
        if (tc.size == 0) break;
        eval_read += tc.size;

        FeatureChunk f = processor.process(tc);
        f.features = norm1.normalize_chunk(f.features, false);  // frozen at inference
        std::vector<Sample> samples = windower.add(f);

        for (auto& s : samples) {
            float summary[SEQ_SUMMARY_DIM];
            MultiScaleSummarizer::summarize(s, summary);
            model.apply_feat_norm(summary, 1);
            model.forward(summary, 1, logits, false);

            Pred p;
            p.logit[0] = logits[0]; p.logit[1] = logits[1]; p.logit[2] = logits[2];
            p.tb_label = static_cast<uint8_t>(s.tb_label);
            uint8_t yl = 1;
            if (s.y_dir[0] > s.y_dir[1] && s.y_dir[0] > s.y_dir[2]) yl = 0;
            else if (s.y_dir[2] > s.y_dir[0] && s.y_dir[2] > s.y_dir[1]) yl = 2;
            p.y_label = yl;
            p.ret = s.tb_ret;
            p.cost = s.tb_cost;
            p.entry_mid = s.entry_mid;
            p.entry_time_ms = s.entry_time_ms;
            p.env_id = s.env_id;
            preds.push_back(p);
        }
    }
    double elapsed = t.elapsed_seconds();
    require(!preds.empty(), "No samples produced in eval window.");

    // ---- Calibration: fit T on the first calib_frac, test on the rest ----
    size_t n = preds.size();
    size_t calib_hi = std::min(n, static_cast<size_t>(static_cast<double>(n) * calib_frac));
    if (calib_hi < 50) calib_hi = std::min<size_t>(n, 50);  // need a minimum to fit T
    float T = 1.0f;
    float ece_before = ece(preds, calib_hi, n, 1.0f);
    if (!no_calib) {
        T = fit_temperature(preds, 0, calib_hi);
    }
    float ece_after = ece(preds, calib_hi, n, T);

    // ---- Kelly stake from the calibration slice (no look-ahead) ----
    // Gate calib trades at conf_gate (calibrated), realize unit-sized nets, take
    // the Kelly fraction of that slice -> cap the test sizing at kelly_frac * f*.
    std::vector<float> calib_nets;
    std::vector<float>   meta_X;   // [m x META_FEATURE_DIM] gated calib signals
    std::vector<uint8_t> meta_y;   // 1 if that signal was profitable after cost
    for (size_t i = 0; i < calib_hi; ++i) {
        float p[3]; softmax_T(preds[i].logit, T, p);
        int pred = argmax3(p);
        if (pred == 1 || p[pred] < conf_gate) continue;
        float dir = (pred == 0) ? 1.0f : -1.0f;
        float tc = execution_cost(preds[i]);
        float net = dir * preds[i].ret - tc;
        calib_nets.push_back(net);
        if (use_meta) {
            float fr[META_FEATURE_DIM];
            meta_features(p, preds[i].env_id, fr);
            meta_X.insert(meta_X.end(), fr, fr + META_FEATURE_DIM);
            meta_y.push_back(net > 0.0f ? 1 : 0);
        }
    }

    MetaLabeler meta;
    if (use_meta && meta_y.size() >= 30) {
        meta.fit(meta_X, meta_y, META_FEATURE_DIM);
        std::cout << "[META] trained on " << meta_y.size() << " gated calib signals\n";
    } else if (use_meta) {
        std::cout << "\033[33m[META] too few calib signals (" << meta_y.size()
                  << "); meta-gate disabled\033[0m\n";
    }
    double overlap_block = static_cast<double>(window + horizon) /
                           static_cast<double>(stride > 0 ? stride : 1);
    wfm::TradeStats calib_stats = wfm::compute_stats(calib_nets, overlap_block);
    double kelly_star = std::max(0.0, calib_stats.kelly);
    double size_base = std::min(static_cast<double>(size_cap), kelly_frac * kelly_star);
    if (size_base <= 0.0) size_base = std::min<double>(size_cap, 0.10);  // floor so test still trades

    // ---- Test region: unit-sized AND confidence-sized nets, overall + per regime ----
    std::vector<float> unit_nets, sized_nets;
    std::vector<std::pair<int64_t, float>> timed_unit_nets;
    std::map<int, std::vector<float>> regime_nets;
    size_t tb_correct = 0, y_correct = 0, scored = 0;
    size_t n_long = 0, n_short = 0, n_flat = 0;
    for (size_t i = calib_hi; i < n; ++i) {
        float p[3]; softmax_T(preds[i].logit, T, p);
        int pred = argmax3(p);
        ++scored;
        if (pred == preds[i].tb_label) ++tb_correct;
        if (pred == preds[i].y_label)  ++y_correct;

        if (pred == 1 || p[pred] < conf_gate) { ++n_flat; continue; }
        if (pred == 0) ++n_long; else ++n_short;

        float dir = (pred == 0) ? 1.0f : -1.0f;
        float tc = execution_cost(preds[i]);
        float unit = dir * preds[i].ret - tc;

        // Confidence-proportional size in [0,1], scaled by the Kelly-capped base.
        float conf_scale = (p[pred] - conf_gate) / (1.0f - conf_gate + 1e-6f);
        conf_scale = std::min(1.0f, std::max(0.0f, conf_scale));
        float size = static_cast<float>(size_base) * conf_scale;
        float sized = size * (dir * preds[i].ret - tc);  // cost scales with size

        unit_nets.push_back(unit);
        timed_unit_nets.emplace_back(preds[i].entry_time_ms, unit);
        sized_nets.push_back(sized);
        regime_nets[preds[i].env_id].push_back(unit);
    }

    // ---- Stats ----
    wfm::TradeStats unit_st  = wfm::compute_stats(unit_nets, overlap_block);
    wfm::TradeStats sized_st = wfm::compute_stats(sized_nets, overlap_block);
    wfm::SharpeRobustness rob = wfm::sharpe_robustness(unit_nets, unit_st.n_eff, n_trials);
    const double daily_sharpe = wfm::daily_sharpe(timed_unit_nets);

    double tb_acc = scored ? static_cast<double>(tb_correct) / scored : 0.0;
    double y_acc  = scored ? static_cast<double>(y_correct) / scored : 0.0;

    std::cout << std::fixed;
    std::cout << "Duration:      " << std::setprecision(1) << elapsed << "s\n"
              << "Samples:       " << n << " (calib=" << calib_hi << ", test=" << (n - calib_hi) << ")\n"
              << "Temperature:   " << std::setprecision(3) << T
              << (no_calib ? "  (calibration OFF)" : "") << "\n"
              << "ECE:           " << std::setprecision(4) << ece_before
              << " -> " << ece_after << " (lower = better)\n"
              << "Accuracy(tb):  " << std::setprecision(2) << (tb_acc * 100.0) << "%   "
              << "Accuracy(naive y): " << (y_acc * 100.0) << "%\n"
              << "Signals:       long=" << n_long << " short=" << n_short << " flat=" << n_flat << "\n"
              << "Kelly f* (calib): " << std::setprecision(3) << kelly_star
              << "  -> size_base=" << size_base << "\n\n";

    std::ostringstream hdr; wfm::print_stats_header(hdr); std::cout << hdr.str();
    { std::ostringstream r; wfm::print_stats_row(r, "unit",  unit_st);  std::cout << r.str(); }
    { std::ostringstream r; wfm::print_stats_row(r, "sized", sized_st); std::cout << r.str(); }

    std::cout << "\nPer-regime (unit-sized, after-cost):\n";
    std::cout << hdr.str();
    for (auto& [env, nets] : regime_nets) {
        wfm::TradeStats rs = wfm::compute_stats(nets, overlap_block);
        std::ostringstream r; wfm::print_stats_row(r, regime_name(env), rs); std::cout << r.str();
    }

    std::cout << "\nRobustness (unit-sized):\n"
              << "  Sharpe (per-trade): " << std::setprecision(4) << rob.sharpe
              << "   skew=" << rob.skew << "   kurt=" << rob.kurt << "\n"
              << "  Sharpe (daily UTC): " << std::setprecision(4) << daily_sharpe << "\n"
              << "  PSR  P(SR>0):       " << std::setprecision(4) << rob.psr << "\n"
              << "  DSR  P(SR>SR*)      " << std::setprecision(4) << rob.dsr
              << "   (SR*=" << rob.sr_star << ", trials=" << n_trials << ")\n"
              << "=======================================================================\n";

    if (args.has("--realistic-scenario") && unit_st.edge < 0.0) {
        std::cout << "WARNING: REALISTIC COST SCENARIO HAS NEGATIVE NET P&L; NOT PROFITABLE.\n";
    }

    return 0;
}

}  // namespace from
