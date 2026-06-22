// ============================================================================
// walkforward.cpp — Medallion-Lite Phase 1
//
// Anchored, PURGED + EMBARGOED walk-forward CV with an untouched final holdout.
// Trains the EXISTING SequenceModel (243-dim summary MLP) via GpuTrainer per
// fold and reports honest, AFTER-COST metrics from the triple-barrier labels.
//
// This file is intentionally self-contained (no shared globals with train.cpp).
// It reuses train.cpp's data-loading / normalization / GpuTrainer patterns but
// does not modify train.cpp — the sweep must keep working.
//
// Class convention everywhere: 0=UP, 1=NEUTRAL, 2=DOWN.
// ============================================================================
#include "commands.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "data/sample_weights.hpp"
#include "model/sequence_model.hpp"
#include "io/artifact.hpp"
#include "utils/config_parser.hpp"
#include "utils/timer.hpp"

#ifdef FROM_CUDA
#include "cuda/gpu_trainer.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#endif

namespace from {
namespace {

void wf_enable_ansi_colors() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(h, &mode);
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

size_t arg_size(const CliArgs& args, const std::string& key, size_t def) {
    std::string v = args.get(key, "");
    return v.empty() ? def : static_cast<size_t>(std::stoull(v));
}

float arg_float(const CliArgs& args, const std::string& key, float def) {
    std::string v = args.get(key, "");
    return v.empty() ? def : std::stof(v);
}

// ----------------------------------------------------------------------------
// After-cost trade metrics computed from a vector of net per-trade returns.
// Everything here is NaN-safe: zero trades -> all-zero struct.
// ----------------------------------------------------------------------------
struct TradeStats {
    size_t trades = 0;
    double winrate = 0.0;        // fraction with net > 0
    double edge = 0.0;           // mean net per trade (return units)
    double profit_factor = 0.0;  // sum(+net) / sum(-net)
    double kelly = 0.0;          // p - (1-p)/b,  b = avg_win / avg_loss
    double max_drawdown = 0.0;   // on the cumulative-sum equity curve (return units)
    double t_stat = 0.0;         // mean / se,  se = sd / sqrt(N_eff)
    double sharpe = 0.0;         // mean / std  (per-trade Sharpe)
    // --- effective-sample-size aware significance (overlapping-trade correction) ---
    double n_eff = 0.0;          // trades / block  (>= 1 when trades >= 1)
    double se = 0.0;             // sd(net) / sqrt(N_eff)
    double ci_lo = 0.0;          // mean - 1.96*se   (edge 95% CI lower bound)
    double ci_hi = 0.0;          // mean + 1.96*se   (edge 95% CI upper bound)
};

// Per-trade detail row for the optional --dump-trades holdout export. Captured in
// eval_block ONLY when a sink pointer is passed (holdout path); the `net` field is
// the SAME value pushed into out_nets, so the dump cannot diverge from the metrics.
struct TradeRow {
    size_t  idx = 0;       // running trade ordinal (0-based, time order)
    int64_t t_index = 0;   // GLOBAL entry tick index (all_t[gidx])
    int     dir = 0;       // +1 long (pred==UP), -1 short (pred==DOWN)
    float   ret = 0.0f;    // raw signed realized return (all_ret[gidx])
    float   cost = 0.0f;   // total per-trade cost actually subtracted
    float   net = 0.0f;    // dir*ret - cost  (== value in out_nets)
};

// nets must already be in time order for a meaningful equity / drawdown curve.
// `block` is the overlap factor (reach/stride): consecutive samples share future
// path, so N_eff = trades / block is used for se / t_stat / CI (NOT raw count).
TradeStats compute_stats(const std::vector<float>& nets, double block) {
    TradeStats st;
    st.trades = nets.size();
    if (st.trades == 0) return st;

    double sum = 0.0, gross_profit = 0.0, gross_loss = 0.0;
    size_t wins = 0;
    for (float v : nets) {
        sum += v;
        if (v > 0.0f) { ++wins; gross_profit += v; }
        else          { gross_loss += -static_cast<double>(v); }
    }
    double n = static_cast<double>(st.trades);
    double mean = sum / n;
    st.edge = mean;
    st.winrate = static_cast<double>(wins) / n;

    // Effective sample size: overlapping trades are not independent. Deflate the
    // raw count by the overlap block so the standard error is honest.
    if (block < 1.0) block = 1.0;
    st.n_eff = n / block;
    if (st.n_eff < 1.0) st.n_eff = 1.0;

    // Variance (population) for Sharpe / t-stat
    double var = 0.0;
    for (float v : nets) {
        double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    var /= n;
    double sd = std::sqrt(var);
    if (sd > 1e-12 && st.trades >= 2) {
        st.sharpe = mean / sd;
        st.se = sd / std::sqrt(st.n_eff);
        st.t_stat = (st.se > 1e-300) ? (mean / st.se) : 0.0;
    } else {
        // zero-variance or single trade -> no meaningful dispersion / significance.
        st.sharpe = 0.0;
        st.se = 0.0;
        st.t_stat = 0.0;
    }
    // 95% CI on edge using the effective standard error.
    st.ci_lo = mean - 1.96 * st.se;
    st.ci_hi = mean + 1.96 * st.se;

    // Profit factor
    if (gross_loss > 1e-12) st.profit_factor = gross_profit / gross_loss;
    else                    st.profit_factor = (gross_profit > 0.0) ? 999.0 : 0.0;

    // Kelly: p - (1-p)/b, b = avg_win/avg_loss
    double losses = n - static_cast<double>(wins);
    double avg_win  = (wins   > 0)   ? gross_profit / static_cast<double>(wins)   : 0.0;
    double avg_loss = (losses > 0.0) ? gross_loss   / losses                      : 0.0;
    if (avg_loss > 1e-12) {
        double b = avg_win / avg_loss;
        st.kelly = st.winrate - (1.0 - st.winrate) / (b + 1e-12);
    } else {
        st.kelly = 0.0;
    }

    // Max drawdown on cumulative equity curve (return units, additive)
    double equity = 0.0, peak = 0.0, max_dd = 0.0;
    for (float v : nets) {
        equity += v;
        if (equity > peak) peak = equity;
        double dd = peak - equity;
        if (dd > max_dd) max_dd = dd;
    }
    st.max_drawdown = max_dd;
    return st;
}

// Render one stats row to a stream (table body line).
void print_stats_row(std::ostream& os, const std::string& name, const TradeStats& s) {
    os << std::left << std::setw(14) << name << std::right
       << std::setw(9)  << s.trades
       << std::setw(10) << std::fixed << std::setprecision(1) << s.n_eff
       << std::setw(10) << std::setprecision(3) << s.winrate
       << std::setw(12) << std::setprecision(6) << s.edge
       << std::setw(9)  << std::setprecision(2) << s.profit_factor
       << std::setw(9)  << std::setprecision(3) << s.kelly
       << std::setw(11) << std::setprecision(6) << s.max_drawdown
       << std::setw(11) << std::setprecision(6) << s.se
       << std::setw(9)  << std::setprecision(2) << s.t_stat
       << std::setw(9)  << std::setprecision(3) << s.sharpe
       << std::setw(13) << std::setprecision(6) << s.ci_lo
       << std::setw(13) << std::setprecision(6) << s.ci_hi
       << "\n";
}

void print_stats_header(std::ostream& os) {
    os << std::left << std::setw(14) << "segment" << std::right
       << std::setw(9)  << "trades"
       << std::setw(10) << "N_eff"
       << std::setw(10) << "winrate"
       << std::setw(12) << "edge"
       << std::setw(9)  << "PF"
       << std::setw(9)  << "kelly"
       << std::setw(11) << "maxDD"
       << std::setw(11) << "se"
       << std::setw(9)  << "t_stat"
       << std::setw(9)  << "sharpe"
       << std::setw(13) << "ci95_lo"
       << std::setw(13) << "ci95_hi"
       << "\n";
    os << std::string(128, '-') << "\n";
}

}  // namespace

// ============================================================================
int run_walkforward(const CliArgs& args) {
    wf_enable_ansi_colors();

    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string config_path = args.get("--config", "config.toml");
    std::string report_path = args.get("--output", "walkforward_report.txt");

    Config cfg;
    if (!config_path.empty() && std::filesystem::exists(config_path)) cfg.load(config_path);

    // ---- CLI flags (mirror train.cpp + walk-forward specific) ----
    size_t chunk_size  = arg_size(args, "--chunk-size", cfg.get_size("data.chunk_size", 50000000));
    size_t window      = arg_size(args, "--window", cfg.get_size("data.window_size", 512));
    size_t stride      = arg_size(args, "--stride", cfg.get_size("data.stride", 64));
    size_t horizon     = arg_size(args, "--horizon", cfg.get_size("data.horizon", 256));
    size_t batch_size  = arg_size(args, "--batch-size", cfg.get_size("training.batch_size", 32));
    size_t max_steps   = arg_size(args, "--max-steps", 5000);
    size_t epochs      = arg_size(args, "--epochs", 12);   // sane epoch budget (caps steps/fold)
    size_t validate_every = arg_size(args, "--validate-every", 1000);
    float  lr          = arg_float(args, "--lr", 0.0003f);
    // ---- Cost model: total cost per trade = tb_cost*(1+slippage_mult) + commission ----
    float  commission     = arg_float(args, "--commission", 0.00003f);   // return units, per trade
    float  slippage_mult  = arg_float(args, "--slippage-mult", 0.5f);    // fraction of spread cost added as slippage
    float  dir_threshold = arg_float(args, "--direction-threshold",
                                     cfg.get_float("data.direction_threshold", 2.0f));
    size_t max_samples = arg_size(args, "--max-samples", 500000);
    uint64_t freeze_after = static_cast<uint64_t>(cfg.get_size("data.normalize_freeze_after", 100000));

    size_t folds       = arg_size(args, "--folds", 8);
    float  holdout_frac = arg_float(args, "--holdout-frac", 0.15f);
    float  embargo_frac = arg_float(args, "--embargo-frac", 0.01f);
    float  conf_gate    = arg_float(args, "--conf-gate", 0.50f);
    float  barrier_k    = arg_float(args, "--barrier-k", 1.0f);   // barrier in horizon-vol units
    float  cost_mult    = arg_float(args, "--cost-mult", 1.5f);   // min barrier in round-trip-spread units

    // ---- Sample weighting (Lopez de Prado uniqueness + optional time-decay) ----
    // Overlapping triple-barrier labels are not independent; weighting by
    // average uniqueness (and resampling the train slice by it) removes the
    // overlap bias the fixed-stride windower introduces. --weight-tail < 1.0
    // adds linear time-decay (oldest sample weight = tail, newest = 1.0).
    bool   use_uniqueness = !args.has("--no-uniqueness");
    float  weight_tail    = arg_float(args, "--weight-tail", 1.0f);

    // ---- Artifact / reproducibility flags ----
    std::string model_dir = args.get("--model-dir", "registry");
    uint32_t seed = static_cast<uint32_t>(arg_size(args, "--seed", 42));
    // --dump-trades (default OFF): when set, the FINAL HOLDOUT eval also persists
    // per-trade detail (holdout_trades.csv) + per-fold edges (fold_edges.csv) under
    // the artifact dir, for a later top-N PBO/CSCV + equity-curve re-run.
    bool dump_trades = args.has("--dump-trades");

    if (folds < 1) folds = 1;
    if (holdout_frac < 0.0f) holdout_frac = 0.0f;
    if (holdout_frac > 0.9f) holdout_frac = 0.9f;
    if (embargo_frac < 0.0f) embargo_frac = 0.0f;

    // ========================================================================
    // PHASE 1: Load data — FTC3 cache (instant) or parquet -> summary pipeline.
    //   Contiguous arrays, all indexed by sample (time order == index order):
    //     all_summaries[N*SEQ_SUMMARY_DIM], all_label(uint8), all_ret(float),
    //     all_cost(float), all_w(float), all_t(int64)
    // ========================================================================
    std::vector<float>    all_summaries;   // [N x SEQ_SUMMARY_DIM]
    std::vector<uint8_t>  all_label;       // triple-barrier class 0/1/2
    std::vector<float>    all_ret;         // tb_ret  (signed realized return entry->first-touch)
    std::vector<float>    all_cost;        // tb_cost (cost in return units)
    std::vector<float>    all_w;           // uniqueness_w (P1: 1.0)
    std::vector<int64_t>  all_t;           // t_index (GLOBAL entry tick index)
    size_t N = 0;

    // First-pass (Welford) Normalizer. Hoisted to function scope so it survives
    // BOTH the build path (where it is fit) and the cache path (where its state is
    // reloaded from a sidecar). This is what gets saved to <id>/norm1.bin — it is
    // the normalization a reloaded model needs to turn RAW features into the inputs
    // the summary pipeline saw at train time (audit A1).
    Normalizer norm1(FROM_MAX_FEATURES);
    bool norm1_ready = false;

    std::ostringstream cache_key;
    cache_key << data << ".w" << window << "_s" << stride << "_h" << horizon
              << "_t" << std::fixed << std::setprecision(2) << dir_threshold
              << "_bk" << std::setprecision(2) << barrier_k
              << "_cm" << std::setprecision(2) << cost_mult
              << "_n" << max_samples << ".wf";
    std::string cache_path = cache_key.str();
    std::string norm_sidecar_path = cache_path + ".norm1";  // first-pass Normalizer state

    Timer load_timer;
    bool cache_loaded = false;

    // ---- Try FTC3 cache ----
    if (std::filesystem::exists(cache_path)) {
        std::ifstream cache(cache_path, std::ios::binary);
        if (cache) {
            char magic[4];
            cache.read(magic, 4);
            if (std::memcmp(magic, "FTC3", 4) == 0) {
                uint64_t n = 0, dim = 0;
                cache.read(reinterpret_cast<char*>(&n), 8);
                cache.read(reinterpret_cast<char*>(&dim), 8);
                if (dim == SEQ_SUMMARY_DIM && n == max_samples) {
                    // Strict match only (same rule as train.cpp): the filename encodes
                    // _n=max_samples, so an exact match is the norm. A partial read of a
                    // larger file would desync the block layout into garbage.
                    N = static_cast<size_t>(n);
                    all_summaries.resize(N * SEQ_SUMMARY_DIM);
                    all_label.resize(N);
                    all_ret.resize(N);
                    all_cost.resize(N);
                    all_w.resize(N);
                    all_t.resize(N);
                    cache.read(reinterpret_cast<char*>(all_summaries.data()),
                               static_cast<std::streamsize>(N * SEQ_SUMMARY_DIM * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_label.data()),
                               static_cast<std::streamsize>(N));
                    cache.read(reinterpret_cast<char*>(all_ret.data()),
                               static_cast<std::streamsize>(N * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_cost.data()),
                               static_cast<std::streamsize>(N * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_w.data()),
                               static_cast<std::streamsize>(N * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_t.data()),
                               static_cast<std::streamsize>(N * sizeof(int64_t)));
                    if (cache) {
                        cache_loaded = true;
                        std::cout << "\033[32m[CACHE] Loaded " << N << " samples in "
                                  << std::fixed << std::setprecision(2) << load_timer.elapsed_seconds()
                                  << "s from " << cache_path << "\033[0m" << std::endl;
                        // ---- Load first-pass Normalizer sidecar (NRM1) if present ----
                        std::ifstream nin(norm_sidecar_path, std::ios::binary);
                        if (nin) {
                            char nmagic[4];
                            nin.read(nmagic, 4);
                            uint64_t ndims = 0;
                            nin.read(reinterpret_cast<char*>(&ndims), 8);
                            if (std::memcmp(nmagic, "NRM1", 4) == 0 && ndims > 0 && ndims <= 1024) {
                                std::vector<double> nmean(ndims), nm2(ndims);
                                std::vector<uint64_t> ncount(ndims);
                                for (double& v : nmean)  nin.read(reinterpret_cast<char*>(&v), 8);
                                for (double& v : nm2)    nin.read(reinterpret_cast<char*>(&v), 8);
                                for (uint64_t& v : ncount) nin.read(reinterpret_cast<char*>(&v), 8);
                                uint8_t nfrozen = 0;
                                nin.read(reinterpret_cast<char*>(&nfrozen), 1);
                                if (nin) {
                                    norm1.set_state(std::move(nmean), std::move(nm2),
                                                    std::move(ncount), nfrozen != 0);
                                    norm1_ready = true;
                                }
                            }
                        }
                        if (!norm1_ready) {
                            std::cout << "\033[33m[NORM] no first-pass Normalizer sidecar ("
                                      << norm_sidecar_path << "); norm1.bin will be the "
                                         "untrained default for this cached run\033[0m" << std::endl;
                        }
                    }
                } else if (dim == SEQ_SUMMARY_DIM && n > 0 && n != max_samples) {
                    std::cout << "\033[33m[CACHE] Stale: has " << n << " samples but need "
                              << max_samples << " — regenerating...\033[0m" << std::endl;
                } else if (dim != SEQ_SUMMARY_DIM && n > 0) {
                    std::cout << "\033[33m[CACHE] Stale: dim=" << dim << " but SEQ_SUMMARY_DIM="
                              << SEQ_SUMMARY_DIM << " — features changed, regenerating...\033[0m" << std::endl;
                }
            }
        }
    }

    // ---- Build from parquet if no usable cache ----
    if (!cache_loaded) {
        Normalizer& normalizer = norm1;  // fit the function-scope first-pass Normalizer
        ParquetReader reader(data);
        TickProcessor processor;
        Windower windower(window, stride, horizon, dir_threshold);
        windower.set_barriers(barrier_k, barrier_k, cost_mult);
        size_t total_rows = reader.total_rows();
        size_t rows_loaded = 0;

        std::cout << "============================================================\n";
        std::cout << " walkforward: processing parquet -> FTC3 cache\n";
        std::cout << " " << (total_rows / 1000000) << "M rows (next run will be instant)\n";
        std::cout << "============================================================" << std::endl;

        all_summaries.reserve(max_samples * SEQ_SUMMARY_DIM);
        all_label.reserve(max_samples);
        all_ret.reserve(max_samples);
        all_cost.reserve(max_samples);
        all_w.reserve(max_samples);
        all_t.reserve(max_samples);

        while (reader.has_next_chunk() && N < max_samples) {
            TickChunk ticks = reader.read_chunk(chunk_size);
            if (ticks.size == 0) break;
            rows_loaded += ticks.size;

            FeatureChunk features = processor.process(ticks);
            bool update_norm = normalizer.count()[0] < freeze_after;
            features.features = normalizer.normalize_chunk(features.features, update_norm);
            if (normalizer.count()[0] >= freeze_after) normalizer.freeze();

            std::vector<Sample> samples = windower.add(features);
            for (auto& s : samples) {
                if (N >= max_samples) break;
                SequenceModel::precompute_summary(s);
                size_t offset = all_summaries.size();
                all_summaries.resize(offset + SEQ_SUMMARY_DIM);
                std::memcpy(all_summaries.data() + offset, s.summary.data(),
                            SEQ_SUMMARY_DIM * sizeof(float));

                // Honest triple-barrier labels/returns straight from the windower contract.
                all_label.push_back(static_cast<uint8_t>(s.tb_label));
                all_ret.push_back(s.tb_ret);
                all_cost.push_back(s.tb_cost);
                all_w.push_back(s.uniqueness_w);
                all_t.push_back(static_cast<int64_t>(s.t_index));
                ++N;
            }

            if (rows_loaded % 10000000 == 0) {
                std::cout << "  [LOAD] " << (rows_loaded / 1000000) << "M/" << (total_rows / 1000000)
                          << "M rows -> " << N << " samples" << std::endl;
            }
        }

        // ---- Save FTC3 cache (symmetric to the reader above) ----
        {
            std::ofstream cache(cache_path, std::ios::binary);
            char magic[4] = {'F', 'T', 'C', '3'};
            cache.write(magic, 4);
            uint64_t n = N, dim = SEQ_SUMMARY_DIM;
            cache.write(reinterpret_cast<const char*>(&n), 8);
            cache.write(reinterpret_cast<const char*>(&dim), 8);
            cache.write(reinterpret_cast<const char*>(all_summaries.data()),
                        static_cast<std::streamsize>(N * SEQ_SUMMARY_DIM * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_label.data()),
                        static_cast<std::streamsize>(N));
            cache.write(reinterpret_cast<const char*>(all_ret.data()),
                        static_cast<std::streamsize>(N * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_cost.data()),
                        static_cast<std::streamsize>(N * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_w.data()),
                        static_cast<std::streamsize>(N * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_t.data()),
                        static_cast<std::streamsize>(N * sizeof(int64_t)));
            std::cout << "[CACHE] Saved " << cache_path << " (" << N << " samples)" << std::endl;
        }

        // ---- Save first-pass Normalizer sidecar (NRM1) alongside the cache, so a
        //      future cache-only run can still reproduce the RAW->normalized mapping. ----
        norm1_ready = true;
        if (from::io::save_norm1(norm_sidecar_path, norm1)) {
            std::cout << "[CACHE] Saved " << norm_sidecar_path << " (first-pass Normalizer)"
                      << std::endl;
        } else {
            std::cout << "\033[33m[WARN] could not write Normalizer sidecar "
                      << norm_sidecar_path << "\033[0m" << std::endl;
        }
    }

    if (N == 0) {
        std::cout << "[ERROR] No samples loaded.\n";
        return 1;
    }
    if (N < batch_size * 4) {
        std::cout << "[ERROR] Not enough samples (" << N << ") for walk-forward.\n";
        return 1;
    }

    // ========================================================================
    // PHASE 1.5: Holdout reservation + normalization region
    //   Reserve the LAST holdout_frac (by index == by time) as HOLDOUT. It is
    //   NEVER used for any fold training or selection. H = first non-holdout idx.
    // ========================================================================
    size_t holdout_start = N - static_cast<size_t>(static_cast<double>(N) * holdout_frac);
    if (holdout_start >= N) holdout_start = N;             // holdout_frac==0 -> no holdout
    if (holdout_start < batch_size * 2 && holdout_frac > 0.0f) {
        std::cout << "[WARN] holdout would leave too little training data; clamping holdout to 0.\n";
        holdout_start = N;
    }
    size_t H = holdout_start;  // non-holdout region is [0, H)

    // Sanity: ensure all_t is monotonic non-decreasing (purge index math relies on it).
    bool t_monotonic = true;
    for (size_t i = 1; i < N; ++i) {
        if (all_t[i] < all_t[i - 1]) { t_monotonic = false; break; }
    }
    if (!t_monotonic) {
        std::cout << "\033[33m[WARN] all_t is not monotonic; purge index math assumes time order.\033[0m\n";
    }

    // ---- Sample-uniqueness weights (overwrite the 1.0 placeholder in all_w) ----
    // Label span = horizon ticks from entry; weight = avg(1/concurrency) over the
    // span, optionally x linear time-decay. Used to importance-resample each fold's
    // training slice so overlapping windows stop being double-counted.
    if (use_uniqueness) {
        std::vector<long long> entries(N);
        for (size_t i = 0; i < N; ++i) entries[i] = static_cast<long long>(all_t[i]);
        std::vector<float> w = compute_sample_weights(entries, static_cast<long long>(horizon),
                                                      weight_tail);
        double wmin = 1e30, wmax = -1e30, wsum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            all_w[i] = w[i];
            wsum += w[i];
            if (w[i] < wmin) wmin = w[i];
            if (w[i] > wmax) wmax = w[i];
        }
        double wmean = (N > 0) ? wsum / static_cast<double>(N) : 0.0;
        std::cout << "\033[32m[WEIGHTS] uniqueness on (tail=" << std::fixed << std::setprecision(2)
                  << weight_tail << "): mean=" << std::setprecision(3) << wmean
                  << " min=" << wmin << " max=" << wmax
                  << " -> N_eff_uniq~=" << std::setprecision(0) << (wmean > 0 ? N : 0) << "\033[0m\n";
    } else {
        std::cout << "\033[33m[WEIGHTS] uniqueness OFF (--no-uniqueness); train slices uniform\033[0m\n";
    }

    // ========================================================================
    // PHASE 1.6: Second-pass z-score normalization.
    //   Mean/std from the FIRST ~200k samples of the NON-holdout region only, so
    //   there is no holdout leakage into normalization. Clip [-5,5]. Applied to
    //   ALL samples (incl. holdout) using the train-derived statistics.
    // ========================================================================
    std::vector<float> feat_mean(SEQ_SUMMARY_DIM, 0.0f);
    std::vector<float> feat_std(SEQ_SUMMARY_DIM, 1.0f);
    {
        size_t norm_n = std::min(H, static_cast<size_t>(200000));
        if (norm_n == 0) norm_n = std::min(N, static_cast<size_t>(200000));
        for (size_t i = 0; i < norm_n; ++i) {
            const float* row = all_summaries.data() + i * SEQ_SUMMARY_DIM;
            for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) feat_mean[d] += row[d];
        }
        float inv_n = 1.0f / static_cast<float>(norm_n);
        for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) feat_mean[d] *= inv_n;

        for (size_t i = 0; i < norm_n; ++i) {
            const float* row = all_summaries.data() + i * SEQ_SUMMARY_DIM;
            for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
                float diff = row[d] - feat_mean[d];
                feat_std[d] += diff * diff;
            }
        }
        for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
            feat_std[d] = std::sqrt(feat_std[d] * inv_n + 1e-8f);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t ii = 0; ii < static_cast<int64_t>(N); ++ii) {
            size_t i = static_cast<size_t>(ii);
            float* row = all_summaries.data() + i * SEQ_SUMMARY_DIM;
            for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
                row[d] = (row[d] - feat_mean[d]) / feat_std[d];
                if (row[d] >  5.0f) row[d] =  5.0f;
                if (row[d] < -5.0f) row[d] = -5.0f;
            }
        }
        std::cout << "\033[32m[NORM] z-score from first " << norm_n
                  << " NON-holdout samples (clip [-5,5])\033[0m" << std::endl;
    }

    // ========================================================================
    // PHASE 2: Fold geometry over the non-holdout region [0, H).
    //   warmup = H/(K+1). Val blocks evenly tile [warmup, H). Fold i (1..K)
    //   trains on [0, t_i) and validates on [t_i, t_{i+1}).
    // ========================================================================
    size_t K = folds;
    size_t warmup = H / (K + 1);
    if (warmup < batch_size) warmup = std::min(H, batch_size);  // guard tiny H
    size_t embargo = static_cast<size_t>(static_cast<double>(N) * embargo_frac);

    // ---- Overlap block for effective sample size ----
    //   Consecutive validation samples are stride ticks apart but each label reads
    //   reach = window + horizon ticks of future path, so ~reach/stride consecutive
    //   trades share the same path and are NOT independent. N_eff = trades / block.
    size_t reach = window + horizon;
    double overlap_block = static_cast<double>(reach) / static_cast<double>(stride > 0 ? stride : 1);
    if (overlap_block < 1.0) overlap_block = 1.0;

    // Val-block boundaries: t_1 = warmup, then evenly spaced up to H.
    // block k (0-based) -> [val_lo[k], val_hi[k]).
    std::vector<size_t> val_lo(K), val_hi(K);
    {
        double span = static_cast<double>(H > warmup ? H - warmup : 0);
        for (size_t k = 0; k < K; ++k) {
            double lo = static_cast<double>(warmup) + span * (static_cast<double>(k)     / static_cast<double>(K));
            double hi = static_cast<double>(warmup) + span * (static_cast<double>(k + 1) / static_cast<double>(K));
            val_lo[k] = static_cast<size_t>(lo);
            val_hi[k] = static_cast<size_t>(hi);
            if (k + 1 == K) val_hi[k] = H;  // last block ends exactly at H
            if (val_hi[k] > H) val_hi[k] = H;
            if (val_lo[k] > H) val_lo[k] = H;
        }
    }

    // ---- Report banner ----
    std::ostringstream rpt;  // accumulate the report; mirrored to stdout
    auto emit = [&](const std::string& line) { std::cout << line << "\n"; rpt << line << "\n"; };

    {
        std::ostringstream b;
        b << "============================================================\n"
          << " MEDALLION-LITE P1 — PURGED + EMBARGOED WALK-FORWARD\n"
          << "============================================================\n"
          << " data            : " << data << "\n"
          << " samples (N)     : " << N << "\n"
          << " holdout region  : [" << H << ", " << N << ")  (" << (N - H) << " samples, frac="
          << std::fixed << std::setprecision(3) << holdout_frac << ")\n"
          << " non-holdout (H) : [0, " << H << ")\n"
          << " folds (K)       : " << K << "\n"
          << " warmup          : " << warmup << "\n"
          << " embargo         : " << embargo << " samples (frac=" << embargo_frac << ")\n"
          << " horizon         : " << horizon << " ticks\n"
          << " conf_gate       : " << conf_gate << "\n"
          << " window/stride   : " << window << "/" << stride << "  dir_threshold=" << dir_threshold << "\n"
          << " batch/lr/steps  : " << batch_size << "/" << lr << "/" << max_steps << "\n"
          << " epoch budget    : " << epochs << " epochs (steps/fold capped at min(max_steps, epochs*ceil(train/batch)))\n"
          << " cost model      : tb_cost*(1+slippage_mult) + commission"
          << "  [commission=" << std::setprecision(6) << commission
          << ", slippage_mult=" << std::setprecision(3) << slippage_mult << "]\n"
          << " overlap block   : reach(=" << reach << ")/stride(=" << stride
          << ") = " << std::setprecision(2) << overlap_block << "  -> N_eff = trades/block\n"
          << " label dist (all): [" ;
        size_t cls[3] = {0,0,0};
        for (size_t i = 0; i < N; ++i) cls[all_label[i] < 3 ? all_label[i] : 1]++;
        b << cls[0] << "," << cls[1] << "," << cls[2] << "]";
        emit(b.str());
    }

    // ---- Class weights helper (inverse freq over a training slice; cap 4x) ----
    auto class_weights_for = [&](const std::vector<uint8_t>& labels, float out_w[3]) {
        size_t cnt[3] = {0,0,0};
        for (uint8_t l : labels) cnt[l < 3 ? l : 1]++;
        float total = static_cast<float>(labels.size());
        if (total < 1.0f) { out_w[0]=out_w[1]=out_w[2]=1.0f; return; }
        for (int c = 0; c < 3; ++c) {
            float freq = static_cast<float>(cnt[c]) / total;
            out_w[c] = 1.0f / (freq + 1e-8f);
        }
        float mean_w = (out_w[0] + out_w[1] + out_w[2]) / 3.0f;
        for (int c = 0; c < 3; ++c) out_w[c] /= (mean_w + 1e-12f);
        float max_w = std::max({out_w[0], out_w[1], out_w[2]});
        for (int c = 0; c < 3; ++c) out_w[c] = std::max(out_w[c], max_w / 4.0f);
    };

    // ------------------------------------------------------------------------
    // Train a model on an explicit list of training indices (purged) and write
    // the resulting weights into `model`. Uses GpuTrainer when CUDA is present,
    // else a CPU fallback that reuses SequenceModel forward/backward.
    // Returns true on success. The training slice is gathered into contiguous
    // temp buffers so GpuTrainer sees train_count == slice size.
    // ------------------------------------------------------------------------
    auto train_on_indices = [&](const std::vector<size_t>& train_idx, uint32_t seed,
                                SequenceModel& model, const std::string& tag) -> bool {
        size_t train_count = train_idx.size();
        if (train_count < batch_size) {
            emit("  [SKIP] " + tag + ": training slice (" + std::to_string(train_count) +
                 ") < batch_size");
            return false;
        }

        // ---- Sane epoch budget: cap steps so we don't run hundreds of epochs ----
        //   steps_per_epoch = ceil(train_count / batch_size);
        //   capped_steps = min(max_steps, epochs * steps_per_epoch).
        // --max-steps remains a hard upper bound.
        size_t steps_per_epoch = (train_count + batch_size - 1) / batch_size;
        size_t epoch_cap = epochs * steps_per_epoch;
        size_t capped_steps = std::min(max_steps, epoch_cap);
        if (capped_steps < 1) capped_steps = 1;
        {
            std::ostringstream es;
            double eff_epochs = static_cast<double>(capped_steps) /
                                static_cast<double>(steps_per_epoch > 0 ? steps_per_epoch : 1);
            es << "  " << tag << ": steps=" << capped_steps
               << " (epochs~=" << std::fixed << std::setprecision(1) << eff_epochs
               << ", steps/epoch=" << steps_per_epoch
               << ", max_steps=" << max_steps << ")";
            emit(es.str());
        }

        // Importance-resample the training slice by sample uniqueness. Drawing
        // train_count samples with replacement, with probability proportional to
        // all_w, makes overlapping (low-uniqueness) windows appear proportionally
        // less often -- a sequential-bootstrap that both the GPU kernel (uniform
        // internal sampling) and the CPU path then see for free. Deterministic in
        // `seed` for reproducibility. Disabled by --no-uniqueness.
        std::vector<size_t> eff_idx;
        const std::vector<size_t>* gather_idx = &train_idx;
        if (use_uniqueness && train_count > 0) {
            std::vector<double> cdf(train_count);
            double acc = 0.0;
            for (size_t i = 0; i < train_count; ++i) {
                double w = static_cast<double>(all_w[train_idx[i]]);
                if (!(w > 0.0)) w = 1e-6;
                acc += w;
                cdf[i] = acc;
            }
            if (acc > 0.0) {
                eff_idx.resize(train_count);
                uint32_t rs = seed * 2654435761u + 1u;
                auto next = [&]() -> double {
                    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
                    return (static_cast<double>(rs) / 4294967296.0);
                };
                for (size_t i = 0; i < train_count; ++i) {
                    double r = next() * acc;
                    size_t lo = static_cast<size_t>(
                        std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
                    if (lo >= train_count) lo = train_count - 1;
                    eff_idx[i] = train_idx[lo];
                }
                gather_idx = &eff_idx;
            }
        }

        // Gather contiguous training arrays.
        std::vector<float>   tr_sum(train_count * SEQ_SUMMARY_DIM);
        std::vector<uint8_t> tr_lab(train_count);
        for (size_t i = 0; i < train_count; ++i) {
            size_t g = (*gather_idx)[i];
            std::memcpy(tr_sum.data() + i * SEQ_SUMMARY_DIM,
                        all_summaries.data() + g * SEQ_SUMMARY_DIM,
                        SEQ_SUMMARY_DIM * sizeof(float));
            tr_lab[i] = all_label[g];
        }

        float cw[3];
        class_weights_for(tr_lab, cw);

        // Persist normalization into the model for downstream forward() calls.
        model.feat_mean = feat_mean;
        model.feat_std = feat_std;
        model.feat_norm_ready = false;  // summaries are ALREADY normalized in-place

        bool use_gpu = false;
#ifdef FROM_CUDA
        cuda::GpuTrainer gpu;
        use_gpu = gpu.initialize(static_cast<int>(batch_size), tr_sum, tr_lab, train_count,
                                 model.w1.data(), model.b1.data(),
                                 model.w2.data(), model.b2.data(),
                                 model.w3.data(), model.b3.data());
        if (use_gpu) {
            gpu.set_class_weights(cw);
            size_t warmup_steps = std::min<size_t>(2000, capped_steps / 4 + 1);
            float base_lr = lr;
            size_t step = 0;
            size_t burst = 100;
            while (step < capped_steps) {
                size_t this_burst = std::min(burst, capped_steps - step);
                for (size_t b = 0; b < this_burst; ++b) {
                    size_t st = step + b;
                    float cur_lr = (st < warmup_steps)
                        ? base_lr * static_cast<float>(st + 1) / static_cast<float>(warmup_steps)
                        : base_lr;
                    uint32_t step_seed = static_cast<uint32_t>(st + 1) * 2654435761u + seed;
                    gpu.train_step_gpu_only(static_cast<int>(batch_size), cur_lr, step_seed);
                }
                step += this_burst;
                if (validate_every > 0 && step % validate_every == 0) {
                    float acc = 0.0f;
                    float loss = gpu.sync_metrics(static_cast<int>(batch_size), &acc);
                    std::cout << "    [" << tag << " step " << std::setw(6) << step
                              << "] loss=" << std::fixed << std::setprecision(4) << loss
                              << " acc=" << std::setprecision(3) << acc << "\r" << std::flush;
                }
            }
            std::cout << std::endl;
            gpu.download_weights(model.w1.data(), model.b1.data(),
                                 model.w2.data(), model.b2.data(),
                                 model.w3.data(), model.b3.data());
            return true;
        }
#endif
        if (!use_gpu) {
            // CPU fallback: class-weighted CE, grad-clipped Adam (model.backward clips).
            std::vector<float> batch_input(batch_size * SEQ_SUMMARY_DIM);
            std::vector<float> batch_logits(batch_size * SEQ_NUM_CLASSES);
            std::vector<float> batch_probs(batch_size * SEQ_NUM_CLASSES);
            std::vector<float> batch_grad(batch_size * SEQ_NUM_CLASSES);
            uint32_t xs = seed + 1;
            auto xorshift = [&]() -> uint32_t {
                xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs;
            };
            size_t warmup_steps = std::min<size_t>(2000, capped_steps / 4 + 1);
            float base_lr = lr;
            std::vector<uint32_t> idx(batch_size);
            for (size_t step = 0; step < capped_steps; ++step) {
                model.lr = (step < warmup_steps)
                    ? base_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps)
                    : base_lr;
                for (size_t i = 0; i < batch_size; ++i) {
                    idx[i] = xorshift() % static_cast<uint32_t>(train_count);
                    std::memcpy(batch_input.data() + i * SEQ_SUMMARY_DIM,
                                tr_sum.data() + idx[i] * SEQ_SUMMARY_DIM,
                                SEQ_SUMMARY_DIM * sizeof(float));
                }
                model.forward(batch_input.data(), batch_size, batch_logits.data(), true);
                SequenceModel::softmax(batch_logits.data(), batch_size, batch_probs.data());
                float inv_b = 1.0f / static_cast<float>(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    uint8_t truth = tr_lab[idx[i]];
                    float w = cw[truth];
                    for (size_t c = 0; c < SEQ_NUM_CLASSES; ++c) {
                        float target = (c == truth) ? 1.0f : 0.0f;
                        batch_grad[i * SEQ_NUM_CLASSES + c] =
                            w * (batch_probs[i * SEQ_NUM_CLASSES + c] - target) * inv_b;
                    }
                }
                model.backward(batch_grad.data(), batch_size);
            }
            return true;
        }
        return false;
    };

    // ------------------------------------------------------------------------
    // Evaluate a trained model on a val block [lo, hi) and append confidence-
    // gated, after-cost trade nets (in time order) to `out_nets`.
    //   trade iff pred != NEUTRAL and max_prob >= conf_gate
    //   total_cost = tb_cost*(1+slippage_mult) + commission   (spread + slippage + commission)
    //   net = dir_sign*tb_ret - total_cost,  dir_sign = (pred==0 ? +1 : -1)
    // ------------------------------------------------------------------------
    // `trade_sink` (optional, default nullptr) captures per-trade detail rows in the
    // SAME gated loop that builds out_nets — used only by the --dump-trades holdout
    // path. When nullptr, behavior is byte-identical to before.
    auto eval_block = [&](SequenceModel& model, size_t lo, size_t hi,
                          std::vector<float>& out_nets,
                          std::vector<TradeRow>* trade_sink = nullptr) {
        if (hi <= lo) return;
        size_t bs = batch_size;
        std::vector<float> in(bs * SEQ_SUMMARY_DIM);
        std::vector<float> logits(bs * SEQ_NUM_CLASSES);
        std::vector<float> probs(bs * SEQ_NUM_CLASSES);
        size_t i = lo;
        while (i < hi) {
            size_t cur = std::min(bs, hi - i);
            for (size_t j = 0; j < cur; ++j) {
                std::memcpy(in.data() + j * SEQ_SUMMARY_DIM,
                            all_summaries.data() + (i + j) * SEQ_SUMMARY_DIM,
                            SEQ_SUMMARY_DIM * sizeof(float));
            }
            model.forward(in.data(), cur, logits.data(), false);
            SequenceModel::softmax(logits.data(), cur, probs.data());
            for (size_t j = 0; j < cur; ++j) {
                size_t pred = 0;
                float mx = probs[j * SEQ_NUM_CLASSES];
                for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                    if (probs[j * SEQ_NUM_CLASSES + c] > mx) {
                        mx = probs[j * SEQ_NUM_CLASSES + c];
                        pred = c;
                    }
                }
                if (pred != 1 && mx >= conf_gate) {
                    size_t gidx = i + j;
                    float dir_sign = (pred == 0) ? 1.0f : -1.0f;
                    // Full cost model: spread + slippage (fraction of spread) + commission.
                    float total_cost = all_cost[gidx] * (1.0f + slippage_mult) + commission;
                    float net = dir_sign * all_ret[gidx] - total_cost;
                    out_nets.push_back(net);
                    if (trade_sink) {
                        TradeRow tr;
                        tr.idx     = trade_sink->size();
                        tr.t_index = all_t[gidx];
                        tr.dir     = (pred == 0) ? 1 : -1;
                        tr.ret     = all_ret[gidx];
                        tr.cost    = total_cost;
                        tr.net     = net;  // identical to the value pushed into out_nets
                        trade_sink->push_back(tr);
                    }
                }
            }
            i += cur;
        }
    };

    // ========================================================================
    // PHASE 3: Run K anchored-expanding folds with PURGE + EMBARGO.
    //   For fold i validating on [val_lo, val_hi):
    //     val_start_t = all_t[val_lo]
    //     PURGE: drop training sample j where all_t[j] + horizon >= val_start_t
    //     EMBARGO: also drop the `embargo` samples immediately AFTER each PRIOR
    //              and current val block (excluded from later training too).
    //   Training pool is [0, val_lo) anchored-expanding, minus purge/embargo.
    // ========================================================================
    std::vector<float> pooled_oof;  // out-of-fold trade nets, in time order
    std::vector<TradeStats> fold_stats;

    emit("");
    emit("------------------------------------------------------------");
    emit(" PER-FOLD (out-of-fold, after-cost)");
    emit("------------------------------------------------------------");
    {
        std::ostringstream hs; print_stats_header(hs); std::cout << hs.str(); rpt << hs.str();
    }

    for (size_t k = 0; k < K; ++k) {
        size_t lo = val_lo[k];
        size_t hi = val_hi[k];

        // Guard tiny folds: skip if val block < batch_size.
        if (hi <= lo || (hi - lo) < batch_size) {
            std::ostringstream s;
            s << " fold " << (k + 1) << "/" << K << ": val block ["
              << lo << "," << hi << ") too small (<" << batch_size << ") — skipped";
            emit(s.str());
            fold_stats.push_back(TradeStats{});  // placeholder zero row
            continue;
        }

        int64_t val_start_t = all_t[lo];

        // Build purged + embargoed training index list over the anchored-expanding
        // window [0, lo). Prior val blocks ARE legitimate training data here (they
        // are in the past relative to this fold) — anchored-EXPANDING means [0, t_i).
        // We only drop:
        //   PURGE   — samples whose label horizon reaches into this val block, and
        //   EMBARGO — the `embargo` samples immediately AFTER each prior val block
        //             (these straddle a regime boundary the model already validated).
        std::vector<size_t> train_idx;
        train_idx.reserve(lo);
        // Reach = window + horizon: t_index is the window START, but the label's
        // barrier scan reads ticks out to window+horizon-1 past entry. Purging only
        // `horizon` would leak the trailing `window` ticks across the boundary.
        int64_t hzn = static_cast<int64_t>(window + horizon);
        for (size_t j = 0; j < lo; ++j) {
            // PURGE: label horizon overlaps the (current) val block boundary.
            if (all_t[j] + hzn >= val_start_t) continue;
            // EMBARGO: exclude the `embargo` samples immediately AFTER each PRIOR
            // val block [val_lo[p], val_hi[p]) -> embargo region [val_hi[p], val_hi[p]+embargo).
            bool embargoed = false;
            for (size_t p = 0; p < k; ++p) {
                size_t e_lo = val_hi[p];
                size_t e_hi = val_hi[p] + embargo;
                if (j >= e_lo && j < e_hi) { embargoed = true; break; }
            }
            if (embargoed) continue;
            train_idx.push_back(j);
        }

        std::ostringstream tg; tg << "fold" << (k + 1);
        std::cout << " fold " << (k + 1) << "/" << K
                  << " train=" << train_idx.size()
                  << " val=[" << lo << "," << hi << ") (" << (hi - lo) << ")"
                  << " val_start_t=" << val_start_t << std::endl;

        SequenceModel model(lr, 42u + static_cast<uint32_t>(k));
        if (!train_on_indices(train_idx, 42u + static_cast<uint32_t>(k), model, tg.str())) {
            fold_stats.push_back(TradeStats{});
            continue;
        }

        std::vector<float> fold_nets;
        eval_block(model, lo, hi, fold_nets);
        TradeStats fs = compute_stats(fold_nets, overlap_block);
        fold_stats.push_back(fs);

        // Append to pooled OOF in time order (folds processed in time order).
        pooled_oof.insert(pooled_oof.end(), fold_nets.begin(), fold_nets.end());

        std::ostringstream row;
        print_stats_row(row, "fold" + std::to_string(k + 1), fs);
        std::cout << row.str(); rpt << row.str();
    }

    // ========================================================================
    // PHASE 4: Pooled out-of-fold + Holdout.
    //   Holdout model is trained on ALL non-holdout samples [0, H) (purged
    //   against the holdout boundary + embargo), then evaluated on [H, N).
    // ========================================================================
    TradeStats pooled_stats = compute_stats(pooled_oof, overlap_block);

    emit("");
    emit("------------------------------------------------------------");
    emit(" POOLED OUT-OF-FOLD (after-cost)");
    emit("------------------------------------------------------------");
    {
        std::ostringstream hs; print_stats_header(hs); std::cout << hs.str(); rpt << hs.str();
        std::ostringstream row; print_stats_row(row, "pooled_oof", pooled_stats);
        std::cout << row.str(); rpt << row.str();
    }

    // ---- Holdout ----
    TradeStats holdout_stats;
    bool holdout_evaluated = false;
    // Hoisted so the trained FINAL holdout model can be serialized as an artifact.
    // Seeded from --seed so the saved model is reproducible and fleet-distinct.
    SequenceModel holdout_model(lr, seed);
    bool holdout_model_trained = false;
    // Per-trade holdout detail, captured only when --dump-trades is set (else empty
    // and never written). Lives at function scope so PHASE 6 can persist it.
    std::vector<TradeRow> holdout_trades;
    if (H < N && (N - H) >= batch_size) {
        int64_t holdout_start_t = all_t[H];
        // Reach = window + horizon: t_index is the window START, but the label's
        // barrier scan reads ticks out to window+horizon-1 past entry. Purging only
        // `horizon` would leak the trailing `window` ticks across the boundary.
        int64_t hzn = static_cast<int64_t>(window + horizon);

        std::vector<size_t> train_idx;
        train_idx.reserve(H);
        for (size_t j = 0; j < H; ++j) {
            // PURGE against the holdout boundary.
            if (all_t[j] + hzn >= holdout_start_t) continue;
            // EMBARGO immediately before holdout: drop [H-embargo, H).
            if (embargo > 0 && j + embargo >= H) continue;
            train_idx.push_back(j);
        }

        std::cout << "\n [HOLDOUT] training final model on " << train_idx.size()
                  << " non-holdout samples (purged), eval on [" << H << "," << N << ")\n";

        if (train_on_indices(train_idx, seed, holdout_model, "holdout")) {
            std::vector<float> hnets;
            // Capture per-trade detail only when dumping; same gated loop, same nets.
            eval_block(holdout_model, H, N, hnets,
                       dump_trades ? &holdout_trades : nullptr);
            holdout_stats = compute_stats(hnets, overlap_block);
            holdout_evaluated = true;
            holdout_model_trained = true;
        }
    } else {
        std::cout << "\n [HOLDOUT] none (holdout_frac=0 or block too small)\n";
    }

    emit("");
    emit("------------------------------------------------------------");
    emit(" FINAL HOLDOUT (untouched; after-cost)");
    emit("------------------------------------------------------------");
    {
        std::ostringstream hs; print_stats_header(hs); std::cout << hs.str(); rpt << hs.str();
        std::ostringstream row; print_stats_row(row, "holdout", holdout_stats);
        std::cout << row.str(); rpt << row.str();
    }

    // ========================================================================
    // PHASE 5: Deflation note + final verdict.
    //   Crude deflation: with K configs tried, require |t_stat| > sqrt(2*ln(K)).
    //   t_stat here is the N_eff-deflated statistic (compute_stats uses N_eff for se).
    // ========================================================================
    double defl_thresh = std::sqrt(2.0 * std::log(static_cast<double>(std::max<size_t>(2, K))));
    emit("");
    emit("------------------------------------------------------------");
    {
        std::ostringstream d;
        d << " DEFLATION: n_configs=" << K
          << "  crude threshold |t_stat| > sqrt(2*ln(max(2,K))) = "
          << std::fixed << std::setprecision(3) << defl_thresh
          << "  (t_stat uses N_eff)\n"
          << "   pooled_oof  t_stat=" << std::setprecision(3) << pooled_stats.t_stat
          << " (N_eff=" << std::setprecision(1) << pooled_stats.n_eff << ")"
          << (std::fabs(pooled_stats.t_stat) > defl_thresh ? "  (passes)" : "  (FAILS)") << "\n"
          << "   holdout     t_stat=" << std::setprecision(3) << holdout_stats.t_stat
          << " (N_eff=" << std::setprecision(1) << holdout_stats.n_eff << ")"
          << (std::fabs(holdout_stats.t_stat) > defl_thresh ? "  (passes)" : "  (FAILS)");
        emit(d.str());
    }

    // ------------------------------------------------------------------------
    // Verdict: edge is "yes" ONLY if the UNTOUCHED holdout passes ALL of:
    //   (a) edge CI lower bound > 0  (mean - 1.96*se > 0),
    //   (b) kelly > 0,
    //   (c) N_eff >= 30 (independent trades; else "insufficient independent trades"),
    //   (d) profit_factor > 1.0.
    // ------------------------------------------------------------------------
    bool cond_ci     = holdout_evaluated && (holdout_stats.ci_lo > 0.0);
    bool cond_kelly  = holdout_evaluated && (holdout_stats.kelly > 0.0);
    bool cond_neff   = holdout_evaluated && (holdout_stats.n_eff >= 30.0);
    bool cond_pf     = holdout_evaluated && (holdout_stats.profit_factor > 1.0);
    bool edge_yes    = cond_ci && cond_kelly && cond_neff && cond_pf;

    emit("------------------------------------------------------------");
    {
        auto mark = [](bool ok) { return ok ? "PASS" : "FAIL"; };
        std::ostringstream c;
        c << " HOLDOUT GATE (all required):\n";
        if (!holdout_evaluated) {
            c << "   [FAIL] no holdout evaluated (holdout_frac=0 or block too small)\n";
        } else {
            c << "   [" << mark(cond_ci)    << "] edge CI95 lower bound > 0   (ci_lo="
              << std::fixed << std::setprecision(6) << holdout_stats.ci_lo << ")\n"
              << "   [" << mark(cond_kelly) << "] kelly > 0                   (kelly="
              << std::setprecision(3) << holdout_stats.kelly << ")\n"
              << "   [" << mark(cond_neff)  << "] N_eff >= 30                 (N_eff="
              << std::setprecision(1) << holdout_stats.n_eff
              << (cond_neff ? "" : "  insufficient independent trades") << ")\n"
              << "   [" << mark(cond_pf)    << "] profit_factor > 1.0         (PF="
              << std::setprecision(2) << holdout_stats.profit_factor << ")";
        }
        emit(c.str());
    }

    emit("------------------------------------------------------------");
    {
        std::ostringstream v;
        v << "EDGE: " << (edge_yes ? "yes" : "no")
          << " (holdout PF=" << std::fixed << std::setprecision(2) << holdout_stats.profit_factor
          << ", kelly=" << std::setprecision(3) << holdout_stats.kelly
          << ", t=" << std::setprecision(3) << holdout_stats.t_stat
          << ", N_eff=" << std::setprecision(1) << holdout_stats.n_eff
          << ", edge_ci95=[" << std::setprecision(6) << holdout_stats.ci_lo
          << ", " << holdout_stats.ci_hi << "])";
        emit(v.str());
    }
    emit("============================================================");

    // ---- Write report file ----
    {
        std::ofstream out(report_path);
        if (out) {
            out << rpt.str();
            std::cout << "\n[REPORT] written to " << report_path << std::endl;
        } else {
            std::cout << "\n[WARN] could not write report to " << report_path << std::endl;
        }
    }

    // ========================================================================
    // PHASE 6: Committee-ready ARTIFACT. Self-contained <model_dir>/<id>/ with
    //   weights (FSQ3) + first-pass Normalizer (NRM1, fixes A1) + meta.json +
    //   report.txt, plus one appended manifest.csv row. Only when the FINAL
    //   holdout model was actually trained (else there is nothing to ship).
    // ========================================================================
    if (holdout_model_trained) {
        namespace fs = std::filesystem;
        std::string id = from::io::make_artifact_id("mlp", horizon, cost_mult,
                                                    barrier_k, conf_gate, seed);
        fs::path adir = fs::path(model_dir) / id;
        if (!from::io::ensure_dir(adir)) {
            std::cout << "\033[33m[ARTIFACT] could not create dir " << adir.string()
                      << " — skipping artifact\033[0m" << std::endl;
        } else {
            // 1a. Weights (FSQ3): summary MLP weights + second-pass feat_mean/std.
            //     Enable feat_norm flag so the second-pass z-score is persisted.
            holdout_model.feat_mean = feat_mean;
            holdout_model.feat_std = feat_std;
            holdout_model.feat_norm_ready = true;
            SequenceModelIO::save(holdout_model, (adir / "model.from").string());

            // 1b. First-pass Welford Normalizer (NRM1) — fixes audit A1.
            if (!from::io::save_norm1(adir / "norm1.bin", norm1)) {
                std::cout << "\033[33m[ARTIFACT] could not write norm1.bin\033[0m" << std::endl;
            }
            if (!norm1_ready) {
                std::cout << "\033[33m[ARTIFACT] WARNING: norm1 state was not available "
                             "(cache run without sidecar); norm1.bin holds default state\033[0m"
                          << std::endl;
            }

            // 2. meta.json
            from::io::ArtifactMeta meta;
            meta.id = id;
            meta.arch = "mlp";
            meta.data = data;
            meta.window = window;
            meta.stride = stride;
            meta.horizon = horizon;
            meta.direction_threshold = dir_threshold;
            meta.barrier_k = barrier_k;
            meta.cost_mult = cost_mult;
            meta.conf_gate = conf_gate;
            meta.seed = seed;
            meta.layers = "";  // summary MLP: fixed 243->256->128->3
            meta.max_samples = max_samples;
            meta.folds = folds;
            meta.holdout_frac = holdout_frac;
            meta.embargo_frac = embargo_frac;
            meta.commission = commission;
            meta.slippage_mult = slippage_mult;
            auto fill = [](from::io::ArtifactMetrics& d, const TradeStats& s) {
                d.trades = s.trades; d.n_eff = s.n_eff; d.winrate = s.winrate;
                d.edge = s.edge; d.profit_factor = s.profit_factor; d.kelly = s.kelly;
                d.max_drawdown = s.max_drawdown; d.t_stat = s.t_stat; d.sharpe = s.sharpe;
                d.ci_lo = s.ci_lo; d.ci_hi = s.ci_hi;
            };
            fill(meta.holdout, holdout_stats);
            fill(meta.oof, pooled_stats);
            meta.edge_verdict = edge_yes;  // same gate: ci_lo>0 && kelly>0 && n_eff>=30 && pf>1
            if (!from::io::write_meta_json(adir / "meta.json", meta)) {
                std::cout << "\033[33m[ARTIFACT] could not write meta.json\033[0m" << std::endl;
            }

            // 3. report.txt (full text report; --output still written above).
            {
                std::ofstream rout(adir / "report.txt");
                if (rout) rout << rpt.str();
            }

            // 3b. --dump-trades ONLY: persist per-trade holdout PnL + per-fold edges
            //     for a later PBO/CSCV + equity-curve re-run. Reuses the already-
            //     computed holdout_trades (captured above) and fold_stats (PHASE 3).
            if (dump_trades) {
                // holdout_trades.csv: idx,t_index,dir,ret,cost,net,equity
                //   equity = running cumsum of net (additive return-unit equity curve).
                std::ofstream tcsv(adir / "holdout_trades.csv");
                if (tcsv) {
                    tcsv << "idx,t_index,dir,ret,cost,net,equity\n";
                    tcsv << std::fixed << std::setprecision(8);
                    double equity = 0.0;
                    for (const TradeRow& tr : holdout_trades) {
                        equity += static_cast<double>(tr.net);
                        tcsv << tr.idx << ',' << tr.t_index << ',' << tr.dir << ','
                             << tr.ret << ',' << tr.cost << ',' << tr.net << ','
                             << equity << '\n';
                    }
                } else {
                    std::cout << "\033[33m[ARTIFACT] could not write holdout_trades.csv\033[0m"
                              << std::endl;
                }
                // fold_edges.csv: fold,trades,n_eff,edge,pf,t_stat (one row per WF fold).
                std::ofstream fcsv(adir / "fold_edges.csv");
                if (fcsv) {
                    fcsv << "fold,trades,n_eff,edge,pf,t_stat\n";
                    fcsv << std::fixed << std::setprecision(8);
                    for (size_t k = 0; k < fold_stats.size(); ++k) {
                        const TradeStats& s = fold_stats[k];
                        fcsv << (k + 1) << ',' << s.trades << ',' << s.n_eff << ','
                             << s.edge << ',' << s.profit_factor << ',' << s.t_stat << '\n';
                    }
                } else {
                    std::cout << "\033[33m[ARTIFACT] could not write fold_edges.csv\033[0m"
                              << std::endl;
                }
                std::cout << "\033[32m[ARTIFACT] --dump-trades: wrote holdout_trades.csv ("
                          << holdout_trades.size() << " trades) + fold_edges.csv ("
                          << fold_stats.size() << " folds)\033[0m" << std::endl;
            }

            // 4. Append one CSV row to <model_dir>/manifest.csv (header on create).
            fs::path manifest = fs::path(model_dir) / "manifest.csv";
            if (!from::io::append_manifest_row(manifest, meta, adir.string())) {
                std::cout << "\033[33m[ARTIFACT] could not append manifest row\033[0m" << std::endl;
            }

            std::cout << "\033[32m[ARTIFACT] wrote " << adir.string()
                      << " (model.from, norm1.bin, meta.json, report.txt) + manifest row\033[0m"
                      << std::endl;
        }
    } else {
        std::cout << "\n[ARTIFACT] no final holdout model trained — nothing to save\n";
    }

    return 0;
}

}  // namespace from
