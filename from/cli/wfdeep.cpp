// ============================================================================
// wfdeep.cpp — Medallion-Lite Phase 2 (RAW per-tick window deep MLP)
//
// Same honest machinery as walkforward.cpp (PURGED + EMBARGOED walk-forward CV +
// untouched holdout + triple-barrier labels + after-cost N_eff/CI/deflation), but
// the MODEL consumes the RAW per-tick window [window x 27] instead of the 243-dim
// summary. The per-tick feature stream stays RESIDENT on the GPU once; each
// sample's window is gathered on-the-fly from a per-sample entry tick index.
//
// Data path: a new FTD1 cache stores the contiguous NORMALIZED tick feature stream
// + per-sample window-start offsets + the SAME triple-barrier arrays walkforward
// uses, so every purge / split / metric / verdict is byte-identical logic.
//
// Class convention everywhere: 0=UP, 1=NEUTRAL, 2=DOWN.
// ============================================================================
#include "commands.hpp"
#include "wf_metrics.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "io/artifact.hpp"
#include "utils/config_parser.hpp"
#include "utils/timer.hpp"

#ifdef FROM_CUDA
#include "cuda/deep_mlp_trainer.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#endif

namespace from {
namespace {

using wfm::TradeStats;
using wfm::compute_stats;
using wfm::print_stats_row;
using wfm::print_stats_header;
using wfm::arg_size;
using wfm::arg_float;

void wfd_enable_ansi_colors() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(h, &mode);
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

// Per-trade detail row for the optional --dump-trades holdout export. Captured in
// eval_block ONLY when a sink pointer is passed (holdout path); the `net` field is
// the SAME value pushed into out_nets, so the dump cannot diverge from the metrics.
struct TradeRow {
    size_t  idx = 0;       // running trade ordinal (0-based, time order)
    int64_t t_index = 0;   // GLOBAL entry tick index (all_t[g] == entry_off[g])
    int     dir = 0;       // +1 long (pred==UP), -1 short (pred==DOWN)
    float   ret = 0.0f;    // raw signed realized return (all_ret[g])
    float   cost = 0.0f;   // total per-trade cost actually subtracted
    float   net = 0.0f;    // dir*ret - cost  (== value in out_nets)
};

// Parse "2048,1024,512" -> {2048,1024,512}. Drops non-positive / empty tokens.
std::vector<int> parse_layers(const std::string& s, const std::vector<int>& def) {
    if (s.empty()) return def;
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        tok = tok.substr(a, b - a + 1);
        if (tok.empty()) continue;
        int v = std::stoi(tok);
        if (v > 0) out.push_back(v);
    }
    return out.empty() ? def : out;
}

}  // namespace

// ============================================================================
int run_wfdeep(const CliArgs& args) {
    wfd_enable_ansi_colors();

    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string config_path = args.get("--config", "config.toml");
    std::string report_path = args.get("--output", "wfdeep_report.txt");

    Config cfg;
    if (!config_path.empty() && std::filesystem::exists(config_path)) cfg.load(config_path);

    // ---- CLI flags ----
    size_t chunk_size  = arg_size(args, "--chunk-size", cfg.get_size("data.chunk_size", 50000000));
    size_t window      = arg_size(args, "--window", 256);            // D_in = window*27
    size_t stride      = arg_size(args, "--stride", cfg.get_size("data.stride", 64));
    size_t horizon     = arg_size(args, "--horizon", cfg.get_size("data.horizon", 256));
    size_t batch_size  = arg_size(args, "--batch-size", 8192);
    size_t max_steps   = arg_size(args, "--max-steps", 5000);
    size_t epochs      = arg_size(args, "--epochs", 12);
    size_t validate_every = arg_size(args, "--validate-every", 1000);
    float  lr          = arg_float(args, "--lr", 0.0003f);
    float  commission     = arg_float(args, "--commission", 0.00003f);
    float  slippage_mult  = arg_float(args, "--slippage-mult", 0.5f);
    float  dir_threshold = arg_float(args, "--direction-threshold",
                                     cfg.get_float("data.direction_threshold", 2.0f));
    size_t max_samples = arg_size(args, "--max-samples", 500000);
    uint64_t freeze_after = static_cast<uint64_t>(cfg.get_size("data.normalize_freeze_after", 100000));

    size_t folds       = arg_size(args, "--folds", 8);
    float  holdout_frac = arg_float(args, "--holdout-frac", 0.15f);
    float  embargo_frac = arg_float(args, "--embargo-frac", 0.01f);
    float  conf_gate    = arg_float(args, "--conf-gate", 0.50f);
    float  barrier_k    = arg_float(args, "--barrier-k", 1.0f);
    float  cost_mult    = arg_float(args, "--cost-mult", 1.5f);

    std::vector<int> hidden = parse_layers(args.get("--layers", ""), {2048, 1024, 512});

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
    if (window < 1) window = 1;

    const size_t FEAT = static_cast<size_t>(FROM_MAX_FEATURES);  // 27

    // ========================================================================
    // PHASE 1: Load data — FTD1 cache (instant) or parquet -> raw-stream build.
    //   Per-sample arrays (time order == index order):
    //     entry_off[N] (uint32 window-start ROW into tickfeat), tb_label/ret/cost,
    //     all_t[N] (== entry_off, kept as int64 for parity with walkforward math).
    //   Plus the contiguous per-tick feature stream tickfeat[n_ticks*27].
    // ========================================================================
    std::vector<float>    tickfeat;        // [n_ticks x 27] row-major (normalized)
    std::vector<uint32_t> entry_off;       // [N] window-start ROW offset into tickfeat
    std::vector<uint8_t>  all_label;       // [N]
    std::vector<float>    all_ret;         // [N] tb_ret
    std::vector<float>    all_cost;        // [N] tb_cost
    std::vector<int64_t>  all_t;           // [N] == entry_off (purge index)
    size_t N = 0;
    size_t n_ticks = 0;

    std::ostringstream cache_key;
    cache_key << data << ".w" << window << "_s" << stride << "_h" << horizon
              << "_t" << std::fixed << std::setprecision(2) << dir_threshold
              << "_bk" << std::setprecision(2) << barrier_k
              << "_cm" << std::setprecision(2) << cost_mult
              << "_n" << max_samples << ".wfd";
    std::string cache_path = cache_key.str();

    Timer load_timer;
    bool cache_loaded = false;

    // ---- Try FTD1 cache ----
    if (std::filesystem::exists(cache_path)) {
        std::ifstream cache(cache_path, std::ios::binary);
        if (cache) {
            char magic[4];
            cache.read(magic, 4);
            if (std::memcmp(magic, "FTD1", 4) == 0) {
                uint64_t n = 0, nt = 0, feat = 0, win = 0;
                cache.read(reinterpret_cast<char*>(&n), 8);
                cache.read(reinterpret_cast<char*>(&nt), 8);
                cache.read(reinterpret_cast<char*>(&feat), 8);
                cache.read(reinterpret_cast<char*>(&win), 8);
                if (feat == FEAT && win == window && n > 0 && n <= max_samples && nt > 0) {
                    N = static_cast<size_t>(n);
                    n_ticks = static_cast<size_t>(nt);
                    tickfeat.resize(n_ticks * FEAT);
                    entry_off.resize(N);
                    all_label.resize(N);
                    all_ret.resize(N);
                    all_cost.resize(N);
                    all_t.resize(N);
                    cache.read(reinterpret_cast<char*>(tickfeat.data()),
                               static_cast<std::streamsize>(n_ticks * FEAT * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(entry_off.data()),
                               static_cast<std::streamsize>(N * sizeof(uint32_t)));
                    cache.read(reinterpret_cast<char*>(all_label.data()),
                               static_cast<std::streamsize>(N));
                    cache.read(reinterpret_cast<char*>(all_ret.data()),
                               static_cast<std::streamsize>(N * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_cost.data()),
                               static_cast<std::streamsize>(N * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_t.data()),
                               static_cast<std::streamsize>(N * sizeof(int64_t)));
                    if (cache) {
                        cache_loaded = true;
                        std::cout << "\033[32m[CACHE] Loaded " << N << " samples / " << n_ticks
                                  << " ticks in " << std::fixed << std::setprecision(2)
                                  << load_timer.elapsed_seconds() << "s from " << cache_path
                                  << "\033[0m" << std::endl;
                    }
                } else if (n > 0 && (feat != FEAT || win != window)) {
                    std::cout << "\033[33m[CACHE] Stale: feat=" << feat << " window=" << win
                              << " but need feat=" << FEAT << " window=" << window
                              << " — regenerating...\033[0m" << std::endl;
                } else if (n > 0 && n != max_samples) {
                    std::cout << "\033[33m[CACHE] Stale: has " << n << " samples but need "
                              << max_samples << " — regenerating...\033[0m" << std::endl;
                }
            }
        }
    }

    // ---- Build from parquet if no usable cache ----
    if (!cache_loaded) {
        tickfeat.clear(); entry_off.clear(); all_label.clear();
        all_ret.clear(); all_cost.clear(); all_t.clear(); N = 0; n_ticks = 0;

        Normalizer normalizer(FROM_MAX_FEATURES);
        ParquetReader reader(data);
        TickProcessor processor;
        Windower windower(window, stride, horizon, dir_threshold);
        windower.set_barriers(barrier_k, barrier_k, cost_mult);
        size_t total_rows = reader.total_rows();
        size_t rows_loaded = 0;

        std::cout << "============================================================\n";
        std::cout << " wfdeep: processing parquet -> FTD1 raw-stream cache\n";
        std::cout << " " << (total_rows / 1000000) << "M rows (next run will be instant)\n";
        std::cout << "============================================================" << std::endl;

        entry_off.reserve(max_samples);
        all_label.reserve(max_samples);
        all_ret.reserve(max_samples);
        all_cost.reserve(max_samples);
        all_t.reserve(max_samples);

        while (reader.has_next_chunk() && N < max_samples) {
            TickChunk ticks = reader.read_chunk(chunk_size);
            if (ticks.size == 0) break;
            rows_loaded += ticks.size;

            FeatureChunk features = processor.process(ticks);
            bool update_norm = normalizer.count()[0] < freeze_after;
            features.features = normalizer.normalize_chunk(features.features, update_norm);
            if (normalizer.count()[0] >= freeze_after) normalizer.freeze();

            // Append every normalized feature row in feed order BEFORE windowing, so
            // tickfeat row g == windower tick index g == Sample.t_index for windows
            // starting at g. The windower starts counting at the first fed tick (row 0).
            const float* fp = features.features.data_ptr();  // contiguous [size x 27]
            tickfeat.insert(tickfeat.end(), fp, fp + features.size * FEAT);

            std::vector<Sample> samples = windower.add(features);
            for (auto& s : samples) {
                if (N >= max_samples) break;
                entry_off.push_back(static_cast<uint32_t>(s.t_index));  // window-start tick index
                all_label.push_back(static_cast<uint8_t>(s.tb_label));
                all_ret.push_back(s.tb_ret);
                all_cost.push_back(s.tb_cost);
                all_t.push_back(static_cast<int64_t>(s.t_index));
                ++N;
            }

            if (rows_loaded % 10000000 == 0) {
                std::cout << "  [LOAD] " << (rows_loaded / 1000000) << "M/" << (total_rows / 1000000)
                          << "M rows -> " << N << " samples" << std::endl;
            }
        }
        n_ticks = tickfeat.size() / FEAT;

        // ---- DEBUG parity check (first few samples): the on-the-fly gather slice
        //      tickfeat[entry_off*27 .. +window*27) must equal what the windower used.
        //      (We can't re-read Sample.X here since samples are consumed, but the
        //      invariant is: entry_off[i] < n_ticks AND entry_off[i]+window <= n_ticks.)
        for (size_t i = 0; i < std::min<size_t>(N, 8); ++i) {
            if (static_cast<size_t>(entry_off[i]) + window > n_ticks) {
                std::cout << "\033[33m[WARN] sample " << i << " window exceeds tick stream "
                          << "(off=" << entry_off[i] << ", window=" << window
                          << ", n_ticks=" << n_ticks << ")\033[0m" << std::endl;
            }
        }

        // ---- Save FTD1 cache (symmetric to the reader above) ----
        {
            std::ofstream cache(cache_path, std::ios::binary);
            char magic[4] = {'F', 'T', 'D', '1'};
            cache.write(magic, 4);
            uint64_t n = N, nt = n_ticks, feat = FEAT, win = window;
            cache.write(reinterpret_cast<const char*>(&n), 8);
            cache.write(reinterpret_cast<const char*>(&nt), 8);
            cache.write(reinterpret_cast<const char*>(&feat), 8);
            cache.write(reinterpret_cast<const char*>(&win), 8);
            cache.write(reinterpret_cast<const char*>(tickfeat.data()),
                        static_cast<std::streamsize>(n_ticks * FEAT * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(entry_off.data()),
                        static_cast<std::streamsize>(N * sizeof(uint32_t)));
            cache.write(reinterpret_cast<const char*>(all_label.data()),
                        static_cast<std::streamsize>(N));
            cache.write(reinterpret_cast<const char*>(all_ret.data()),
                        static_cast<std::streamsize>(N * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_cost.data()),
                        static_cast<std::streamsize>(N * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_t.data()),
                        static_cast<std::streamsize>(N * sizeof(int64_t)));
            std::cout << "[CACHE] Saved " << cache_path << " (" << N << " samples, "
                      << n_ticks << " ticks)" << std::endl;
        }
    }

    if (N == 0 || n_ticks == 0) {
        std::cout << "[ERROR] No samples loaded.\n";
        return 1;
    }
    if (N < batch_size * 4) {
        std::cout << "[ERROR] Not enough samples (" << N << ") for walk-forward.\n";
        return 1;
    }

    // ========================================================================
    // PHASE 1.5: Holdout reservation. Reserve the LAST holdout_frac (by index ==
    //   by time) as HOLDOUT, never used for any fold training/selection.
    // ========================================================================
    size_t holdout_start = N - static_cast<size_t>(static_cast<double>(N) * holdout_frac);
    if (holdout_start >= N) holdout_start = N;
    if (holdout_start < batch_size * 2 && holdout_frac > 0.0f) {
        std::cout << "[WARN] holdout would leave too little training data; clamping holdout to 0.\n";
        holdout_start = N;
    }
    size_t H = holdout_start;

    // Sanity: entry_off / all_t must be monotonic non-decreasing.
    bool t_monotonic = true;
    for (size_t i = 1; i < N; ++i) {
        if (all_t[i] < all_t[i - 1]) { t_monotonic = false; break; }
    }
    if (!t_monotonic) {
        std::cout << "\033[33m[WARN] entry_off is not monotonic; purge index math assumes "
                     "time order.\033[0m\n";
    }

    // ========================================================================
    // PHASE 1.6: Second-pass per-tick-feature normalization.
    //   Fit per-feature mean/std on NON-HOLDOUT ticks only (rows [0, tick_H)),
    //   capped to 20M rows. Apply to ALL ticks in place with clip [-8,8].
    //   tick_H = first holdout window-start = entry_off[H] (or n_ticks if no holdout).
    // ========================================================================
    // Hoisted so the per-tick normalization (mean[27]/std[27]) can be embedded in
    // the saved deep.bin artifact — a future loader needs it to reproduce the exact
    // RAW->normalized window inputs the model trained on (audit A1 for the deep path).
    std::vector<float> pertick_mean(FEAT, 0.0f);
    std::vector<float> pertick_std(FEAT, 1.0f);
    {
        size_t tick_H = (H < N) ? static_cast<size_t>(entry_off[H]) : n_ticks;
        if (tick_H > n_ticks) tick_H = n_ticks;
        size_t norm_ticks = std::min<size_t>(tick_H, static_cast<size_t>(20000000));
        if (norm_ticks == 0) norm_ticks = std::min<size_t>(n_ticks, static_cast<size_t>(20000000));

        std::vector<double> mean(FEAT, 0.0), std_(FEAT, 0.0);
        for (size_t r = 0; r < norm_ticks; ++r) {
            const float* row = tickfeat.data() + r * FEAT;
            for (size_t d = 0; d < FEAT; ++d) mean[d] += row[d];
        }
        double inv_n = 1.0 / static_cast<double>(norm_ticks);
        for (size_t d = 0; d < FEAT; ++d) mean[d] *= inv_n;
        for (size_t r = 0; r < norm_ticks; ++r) {
            const float* row = tickfeat.data() + r * FEAT;
            for (size_t d = 0; d < FEAT; ++d) {
                double diff = static_cast<double>(row[d]) - mean[d];
                std_[d] += diff * diff;
            }
        }
        std::vector<float> meanf(FEAT), invstd(FEAT);
        for (size_t d = 0; d < FEAT; ++d) {
            double sd = std::sqrt(std_[d] * inv_n + 1e-8);
            meanf[d] = static_cast<float>(mean[d]);
            invstd[d] = static_cast<float>(1.0 / (sd > 0.0 ? sd : 1.0));
            // Persist actual mean/std (NOT inv-std) for the saved artifact.
            pertick_mean[d] = static_cast<float>(mean[d]);
            pertick_std[d] = static_cast<float>(sd > 0.0 ? sd : 1.0);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t rr = 0; rr < static_cast<int64_t>(n_ticks); ++rr) {
            float* row = tickfeat.data() + static_cast<size_t>(rr) * FEAT;
            for (size_t d = 0; d < FEAT; ++d) {
                float z = (row[d] - meanf[d]) * invstd[d];
                if (z >  8.0f) z =  8.0f;
                if (z < -8.0f) z = -8.0f;
                row[d] = z;
            }
        }
        std::cout << "\033[32m[NORM] per-tick z-score from first " << norm_ticks
                  << " NON-holdout ticks (clip [-8,8])\033[0m" << std::endl;
    }

    // ========================================================================
    // PHASE 2: Fold geometry over the non-holdout region [0, H).
    // ========================================================================
    size_t K = folds;
    size_t warmup = H / (K + 1);
    if (warmup < batch_size) warmup = std::min(H, batch_size);
    size_t embargo = static_cast<size_t>(static_cast<double>(N) * embargo_frac);

    size_t reach = window + horizon;
    double overlap_block = static_cast<double>(reach) / static_cast<double>(stride > 0 ? stride : 1);
    if (overlap_block < 1.0) overlap_block = 1.0;

    std::vector<size_t> val_lo(K), val_hi(K);
    {
        double span = static_cast<double>(H > warmup ? H - warmup : 0);
        for (size_t k = 0; k < K; ++k) {
            double lo = static_cast<double>(warmup) + span * (static_cast<double>(k)     / static_cast<double>(K));
            double hi = static_cast<double>(warmup) + span * (static_cast<double>(k + 1) / static_cast<double>(K));
            val_lo[k] = static_cast<size_t>(lo);
            val_hi[k] = static_cast<size_t>(hi);
            if (k + 1 == K) val_hi[k] = H;
            if (val_hi[k] > H) val_hi[k] = H;
            if (val_lo[k] > H) val_lo[k] = H;
        }
    }

    std::ostringstream rpt;
    auto emit = [&](const std::string& line) { std::cout << line << "\n"; rpt << line << "\n"; };

    {
        std::ostringstream b;
        b << "============================================================\n"
          << " MEDALLION-LITE P2 — RAW-WINDOW DEEP MLP WALK-FORWARD\n"
          << "============================================================\n"
          << " data            : " << data << "\n"
          << " samples (N)     : " << N << "\n"
          << " ticks resident  : " << n_ticks << "  (" << (n_ticks * FEAT * 4 / 1048576) << "MB)\n"
          << " holdout region  : [" << H << ", " << N << ")  (" << (N - H) << " samples, frac="
          << std::fixed << std::setprecision(3) << holdout_frac << ")\n"
          << " non-holdout (H) : [0, " << H << ")\n"
          << " folds (K)       : " << K << "\n"
          << " warmup          : " << warmup << "\n"
          << " embargo         : " << embargo << " samples (frac=" << embargo_frac << ")\n"
          << " window/stride   : " << window << "/" << stride << "  (D_in=" << (window * FEAT) << ")\n"
          << " horizon         : " << horizon << " ticks\n"
          << " layers          : ";
        for (size_t i = 0; i < hidden.size(); ++i) b << hidden[i] << (i + 1 < hidden.size() ? "," : "");
        b << " (hidden) -> 3\n"
          << " conf_gate       : " << conf_gate << "\n"
          << " batch/lr/steps  : " << batch_size << "/" << lr << "/" << max_steps << "\n"
          << " epoch budget    : " << epochs << " epochs (steps/fold capped at min(max_steps, epochs*ceil(train/batch)))\n"
          << " cost model      : tb_cost*(1+slippage_mult) + commission"
          << "  [commission=" << std::setprecision(6) << commission
          << ", slippage_mult=" << std::setprecision(3) << slippage_mult << "]\n"
          << " overlap block   : reach(=" << reach << ")/stride(=" << stride
          << ") = " << std::setprecision(2) << overlap_block << "  -> N_eff = trades/block\n"
          << " label dist (all): [";
        size_t cls[3] = {0, 0, 0};
        for (size_t i = 0; i < N; ++i) cls[all_label[i] < 3 ? all_label[i] : 1]++;
        b << cls[0] << "," << cls[1] << "," << cls[2] << "]";
        emit(b.str());
    }

    // Class weights (inverse freq over a training slice; cap 4x) — same as walkforward.
    auto class_weights_for = [&](const std::vector<uint8_t>& labels, float out_w[3]) {
        size_t cnt[3] = {0, 0, 0};
        for (uint8_t l : labels) cnt[l < 3 ? l : 1]++;
        float total = static_cast<float>(labels.size());
        if (total < 1.0f) { out_w[0] = out_w[1] = out_w[2] = 1.0f; return; }
        for (int c = 0; c < 3; ++c) {
            float freq = static_cast<float>(cnt[c]) / total;
            out_w[c] = 1.0f / (freq + 1e-8f);
        }
        float mean_w = (out_w[0] + out_w[1] + out_w[2]) / 3.0f;
        for (int c = 0; c < 3; ++c) out_w[c] /= (mean_w + 1e-12f);
        float max_w = std::max({out_w[0], out_w[1], out_w[2]});
        for (int c = 0; c < 3; ++c) out_w[c] = std::max(out_w[c], max_w / 4.0f);
    };

#ifndef FROM_CUDA
    (void)model_dir; (void)seed; (void)dump_trades;  // artifact saving is GPU-only (deep.bin needs the trainer)
    emit("");
    emit("[ERROR] wfdeep requires the CUDA backend (raw-window deep MLP is GPU-only).");
    {
        std::ofstream out(report_path);
        if (out) out << rpt.str();
    }
    return 1;
#else
    // ========================================================================
    // GPU trainer setup — the 10GB tick stream is uploaded ONCE and shared
    // across folds. Per fold we only re-bind the (small) offset/label subset.
    // ========================================================================
    cuda::DeepMlpConfig dcfg;
    dcfg.window = static_cast<int>(window);
    dcfg.feat = FROM_MAX_FEATURES;
    dcfg.hidden = hidden;
    dcfg.out = 3;
    dcfg.batch_cap = static_cast<int>(batch_size);

    cuda::DeepMlpTrainer trainer;
    bool trainer_ready = false;

    // Filter out any sample whose window would exceed the (possibly truncated)
    // resident tick stream. We must know n_ticks_dev BEFORE that, so initialize
    // with an empty fold first, then validate offsets against n_ticks_dev().
    // We use a dummy 1-sample subset for the very first init (replaced per fold).
    auto sample_window_ok = [&](size_t g, uint32_t n_ticks_dev) -> bool {
        return static_cast<uint64_t>(entry_off[g]) + window <= static_cast<uint64_t>(n_ticks_dev);
    };

    // Train on an explicit list of (purged) training sample indices. First call
    // initializes the trainer (uploading the resident stream); later calls re-bind.
    auto train_on_indices = [&](const std::vector<size_t>& train_idx, uint32_t seed,
                                const std::string& tag) -> bool {
        size_t train_count = train_idx.size();
        if (train_count < batch_size) {
            emit("  [SKIP] " + tag + ": training slice (" + std::to_string(train_count) +
                 ") < batch_size");
            return false;
        }

        // Initialize the trainer on first call (uploads the resident stream, which may
        // be VRAM-truncated). We then know the device tick cap and can drop any sample
        // whose window exceeds it from the fold subset (the gather kernel also bounds-
        // guards, but filtering keeps the training distribution honest).
        if (!trainer_ready) {
            // Bootstrap init with the first sample's offset/label (replaced below).
            uint32_t boot_off = entry_off[train_idx[0]];
            uint8_t  boot_lab = all_label[train_idx[0]];
            trainer_ready = trainer.initialize(dcfg, tickfeat.data(), n_ticks,
                                               &boot_off, &boot_lab, 1, seed);
            if (!trainer_ready) {
                emit("  [ERROR] DeepMlpTrainer.initialize failed (no GPU / VRAM).");
                return false;
            }
        }
        uint32_t ntd = trainer.n_ticks_dev();

        // Build contiguous fold subset (offsets + labels), filtered to window-valid.
        std::vector<uint32_t> off_sub;
        std::vector<uint8_t>  lab_sub;
        off_sub.reserve(train_count);
        lab_sub.reserve(train_count);
        for (size_t i = 0; i < train_count; ++i) {
            size_t g = train_idx[i];
            if (!sample_window_ok(g, ntd)) continue;
            off_sub.push_back(entry_off[g]);
            lab_sub.push_back(all_label[g]);
        }
        if (off_sub.size() < batch_size) {
            emit("  [SKIP] " + tag + ": window-valid training slice (" +
                 std::to_string(off_sub.size()) + ") < batch_size");
            return false;
        }
        size_t valid_count = off_sub.size();
        trainer.reset_for_fold(off_sub.data(), lab_sub.data(), valid_count, seed);

        float cw[3];
        class_weights_for(lab_sub, cw);
        trainer.set_class_weights(cw);
        train_count = valid_count;  // step budget uses the actual trained count

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

        size_t warmup_steps = std::min<size_t>(2000, capped_steps / 4 + 1);
        float base_lr = lr;
        for (size_t step = 0; step < capped_steps; ++step) {
            float cur_lr = (step < warmup_steps)
                ? base_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps)
                : base_lr;
            uint32_t step_seed = static_cast<uint32_t>(step + 1) * 2654435761u + seed;
            trainer.train_step_gpu_only(static_cast<int>(batch_size), cur_lr, step_seed);
            if (validate_every > 0 && (step + 1) % validate_every == 0) {
                float acc = 0.0f;
                float loss = trainer.sync_metrics(static_cast<int>(batch_size), &acc);
                std::cout << "    [" << tag << " step " << std::setw(6) << (step + 1)
                          << "] loss=" << std::fixed << std::setprecision(4) << loss
                          << " acc=" << std::setprecision(3) << acc << "\r" << std::flush;
            }
        }
        std::cout << std::endl;
        return true;
    };

    // Evaluate a trained model on a val block [lo, hi): forward each sample's
    // window, gate (pred != NEUTRAL && max_prob >= conf_gate), apply the SAME
    // after-cost net as walkforward. Offsets are ABSOLUTE entry_off[g].
    // `trade_sink` (optional, default nullptr) captures per-trade detail rows in the
    // SAME gated loop that builds out_nets — used only by the --dump-trades holdout
    // path. When nullptr, behavior is byte-identical to before.
    auto eval_block = [&](size_t lo, size_t hi, std::vector<float>& out_nets,
                          std::vector<TradeRow>* trade_sink = nullptr) {
        if (hi <= lo) return;
        uint32_t ntd = trainer.n_ticks_dev();
        // Gather only window-valid samples; keep their global index to map ret/cost.
        std::vector<uint32_t> offs;
        std::vector<size_t>   gidx;
        offs.reserve(hi - lo);
        gidx.reserve(hi - lo);
        for (size_t g = lo; g < hi; ++g) {
            if (!sample_window_ok(g, ntd)) continue;
            offs.push_back(entry_off[g]);
            gidx.push_back(g);
        }
        if (offs.empty()) return;

        std::vector<float> logits(offs.size() * 3);
        trainer.eval_offsets(offs.data(), static_cast<int>(offs.size()), logits.data());

        for (size_t j = 0; j < offs.size(); ++j) {
            const float* p = logits.data() + j * 3;
            size_t pred = 0;
            float mx = p[0];
            for (size_t c = 1; c < 3; ++c) {
                if (p[c] > mx) { mx = p[c]; pred = c; }
            }
            if (pred != 1 && mx >= conf_gate) {
                size_t g = gidx[j];
                float dir_sign = (pred == 0) ? 1.0f : -1.0f;
                float total_cost = all_cost[g] * (1.0f + slippage_mult) + commission;
                float net = dir_sign * all_ret[g] - total_cost;
                out_nets.push_back(net);
                if (trade_sink) {
                    TradeRow tr;
                    tr.idx     = trade_sink->size();
                    tr.t_index = all_t[g];
                    tr.dir     = (pred == 0) ? 1 : -1;
                    tr.ret     = all_ret[g];
                    tr.cost    = total_cost;
                    tr.net     = net;  // identical to the value pushed into out_nets
                    trade_sink->push_back(tr);
                }
            }
        }
    };

    // ========================================================================
    // PHASE 3: K anchored-expanding folds with PURGE + EMBARGO (== walkforward).
    // ========================================================================
    std::vector<float> pooled_oof;
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

        if (hi <= lo || (hi - lo) < batch_size) {
            std::ostringstream s;
            s << " fold " << (k + 1) << "/" << K << ": val block ["
              << lo << "," << hi << ") too small (<" << batch_size << ") — skipped";
            emit(s.str());
            fold_stats.push_back(TradeStats{});
            continue;
        }

        int64_t val_start_t = all_t[lo];
        std::vector<size_t> train_idx;
        train_idx.reserve(lo);
        // Reach = window + horizon (t_index is the window START; barrier scan reads
        // out to window+horizon-1). Purging only horizon would leak the window tail.
        int64_t hzn = static_cast<int64_t>(window + horizon);
        for (size_t j = 0; j < lo; ++j) {
            if (all_t[j] + hzn >= val_start_t) continue;  // PURGE
            bool embargoed = false;
            for (size_t p = 0; p < k; ++p) {
                size_t e_lo = val_hi[p];
                size_t e_hi = val_hi[p] + embargo;
                if (j >= e_lo && j < e_hi) { embargoed = true; break; }
            }
            if (embargoed) continue;  // EMBARGO
            train_idx.push_back(j);
        }

        std::ostringstream tg; tg << "fold" << (k + 1);
        std::cout << " fold " << (k + 1) << "/" << K
                  << " train=" << train_idx.size()
                  << " val=[" << lo << "," << hi << ") (" << (hi - lo) << ")"
                  << " val_start_t=" << val_start_t << std::endl;

        if (!train_on_indices(train_idx, 42u + static_cast<uint32_t>(k), tg.str())) {
            fold_stats.push_back(TradeStats{});
            continue;
        }

        std::vector<float> fold_nets;
        eval_block(lo, hi, fold_nets);
        TradeStats fs = compute_stats(fold_nets, overlap_block);
        fold_stats.push_back(fs);
        pooled_oof.insert(pooled_oof.end(), fold_nets.begin(), fold_nets.end());

        std::ostringstream row;
        print_stats_row(row, "fold" + std::to_string(k + 1), fs);
        std::cout << row.str(); rpt << row.str();
    }

    // ========================================================================
    // PHASE 4: Pooled OOF + Holdout.
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

    TradeStats holdout_stats;
    bool holdout_evaluated = false;
    bool holdout_model_trained = false;  // trainer holds the FINAL holdout weights
    // Per-trade holdout detail, captured only when --dump-trades is set (else empty
    // and never written). Lives at function scope so PHASE 6 can persist it.
    std::vector<TradeRow> holdout_trades;
    if (H < N && (N - H) >= batch_size) {
        int64_t holdout_start_t = all_t[H];
        int64_t hzn = static_cast<int64_t>(window + horizon);

        std::vector<size_t> train_idx;
        train_idx.reserve(H);
        for (size_t j = 0; j < H; ++j) {
            if (all_t[j] + hzn >= holdout_start_t) continue;        // PURGE vs holdout
            if (embargo > 0 && j + embargo >= H) continue;          // EMBARGO before holdout
            train_idx.push_back(j);
        }

        std::cout << "\n [HOLDOUT] training final model on " << train_idx.size()
                  << " non-holdout samples (purged), eval on [" << H << "," << N << ")\n";

        // Seed the He-init from --seed so the saved deep model is reproducible
        // and distinct across fleet workers.
        if (train_on_indices(train_idx, seed, "holdout")) {
            std::vector<float> hnets;
            // Capture per-trade detail only when dumping; same gated loop, same nets.
            eval_block(H, N, hnets, dump_trades ? &holdout_trades : nullptr);
            holdout_stats = compute_stats(hnets, overlap_block);
            holdout_evaluated = true;
            holdout_model_trained = true;  // trainer now holds the holdout weights
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
    // PHASE 5: Deflation note + final verdict (== walkforward).
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
    //   deep.bin (all layer W/b + per-tick mean[27]/std[27], fixes A1 for deep) +
    //   meta.json + report.txt, plus one appended manifest.csv row. Only when the
    //   FINAL holdout model was actually trained (trainer holds those weights).
    // ========================================================================
    if (holdout_model_trained) {
        namespace fs = std::filesystem;
        std::string id = from::io::make_artifact_id("deep", horizon, cost_mult,
                                                    barrier_k, conf_gate, seed);
        fs::path adir = fs::path(model_dir) / id;
        if (!from::io::ensure_dir(adir)) {
            std::cout << "\033[33m[ARTIFACT] could not create dir " << adir.string()
                      << " — skipping artifact\033[0m" << std::endl;
        } else {
            // 1. Weights (DEEP): all layer W/b downloaded from GPU + per-tick norm.
            if (!trainer.save((adir / "deep.bin").string(), pertick_mean, pertick_std)) {
                std::cout << "\033[33m[ARTIFACT] could not write deep.bin\033[0m" << std::endl;
            }

            // 2. meta.json (layers spec recorded for reconstruction).
            std::ostringstream layers_ss;
            for (size_t i = 0; i < hidden.size(); ++i)
                layers_ss << hidden[i] << (i + 1 < hidden.size() ? "," : "");

            from::io::ArtifactMeta meta;
            meta.id = id;
            meta.arch = "deep";
            meta.data = data;
            meta.window = window;
            meta.stride = stride;
            meta.horizon = horizon;
            meta.direction_threshold = dir_threshold;
            meta.barrier_k = barrier_k;
            meta.cost_mult = cost_mult;
            meta.conf_gate = conf_gate;
            meta.seed = seed;
            meta.layers = layers_ss.str();
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
                      << " (deep.bin, meta.json, report.txt) + manifest row\033[0m"
                      << std::endl;
        }
    } else {
        std::cout << "\n[ARTIFACT] no final holdout model trained — nothing to save\n";
    }

    return 0;
#endif  // FROM_CUDA
}

}  // namespace from
