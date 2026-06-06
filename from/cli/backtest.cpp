#include "commands.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "model/from_model.hpp"
#include "model/serializer.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace from {

int run_backtest(const CliArgs& args) {
    std::string model_path = args.get("--model", "weights.from");
    std::string data_path = args.get("--data", "XAUUSD_ticks_all.parquet");
    size_t ticks = static_cast<size_t>(std::stoull(args.get("--ticks", "5000000")));
    float conf_threshold = std::stof(args.get("--threshold", "0.55"));
    float spread_cost = std::stof(args.get("--spread-cost", "3.0"));

    require(std::filesystem::exists(model_path), "Model not found: " + model_path);
    require(std::filesystem::exists(data_path), "Data not found: " + data_path);

    // Load model
    FromConfig cfg;
    cfg.conv_out = 256;
    cfg.lstm_hidden = 512;
    cfg.lstm_layers = 3;
    cfg.attention_heads = 8;
    cfg.attention_d_ff = 2048;
    cfg.tft_blocks = 3;
    cfg.moe_experts = 16;
    cfg.moe_top_k = 2;
    cfg.moe_expert_dim = 512;
    cfg.hyperbolic_dim = 64;
    cfg.kalman_state_dim = 16;
    cfg.ode_hidden = 128;
    cfg.ode_steps = 10;
    cfg.dropout = 0.0f;

    std::cout << "Loading model: " << model_path << "..." << std::flush;
    FromModel model(cfg);
    Normalizer normalizer(FROM_MAX_FEATURES);
    Serializer::load(model_path, &model, &normalizer);
    std::cout << " done (" << model.parameter_count() << " params)\n\n";

    // Stream data and generate predictions
    ParquetReader reader(data_path);
    TickProcessor processor;
    Windower windower(512, 64, 128);

    size_t total_rows = reader.total_rows();
    size_t eval_ticks = std::min(ticks, total_rows);
    // Use last portion of data for out-of-sample backtest
    size_t skip_rows = total_rows > eval_ticks ? total_rows - eval_ticks : 0;

    std::cout << "=======================================================================\n";
    std::cout << "BACKTEST: " << model_path << "\n";
    std::cout << "=======================================================================\n";
    std::cout << "Data:          " << data_path << " (" << (total_rows/1000000) << "M rows)\n";
    std::cout << "Eval window:   last " << (eval_ticks/1000000) << "M ticks\n";
    std::cout << "Confidence:    " << conf_threshold << "\n";
    std::cout << "Spread cost:   " << spread_cost << " pips per trade\n";
    std::cout << "=======================================================================\n\n";

    // Skip to eval portion
    size_t rows_read = 0;
    while (rows_read < skip_rows && reader.has_next_chunk()) {
        size_t chunk_sz = std::min<size_t>(2000000, skip_rows - rows_read);
        TickChunk ticks_chunk = reader.read_chunk(chunk_sz);
        rows_read += ticks_chunk.size;
        // Process to keep normalizer/processor state consistent
        FeatureChunk features = processor.process(ticks_chunk);
        normalizer.normalize_chunk(features.features, false);
        windower.add(features);
    }
    std::cout << "Skipped " << (rows_read/1000000) << "M rows to reach eval window.\n";

    // Run backtest
    struct Trade {
        int direction;  // +1 long, -1 short
        double entry_mid;
        double exit_mid;
        double pnl;
    };

    std::vector<Trade> trades;
    size_t signals_total = 0;
    size_t signals_long = 0;
    size_t signals_short = 0;
    size_t signals_flat = 0;
    size_t correct = 0;
    size_t total_preds = 0;

    Timer t;
    size_t eval_read = 0;

    while (reader.has_next_chunk() && eval_read < eval_ticks) {
        size_t chunk_sz = std::min<size_t>(2000000, eval_ticks - eval_read);
        TickChunk ticks_chunk = reader.read_chunk(chunk_sz);
        if (ticks_chunk.size == 0) break;
        eval_read += ticks_chunk.size;

        FeatureChunk features = processor.process(ticks_chunk);
        features.features = normalizer.normalize_chunk(features.features, false);

        std::vector<Sample> samples = windower.add(features);
        if (samples.empty()) continue;

        // Batch inference
        size_t batch_sz = std::min<size_t>(samples.size(), 64);
        for (size_t i = 0; i + batch_sz <= samples.size(); i += batch_sz) {
            Tensor<float> batch_x({batch_sz, 512, FROM_MAX_FEATURES});
            for (size_t b = 0; b < batch_sz; ++b) {
                for (size_t tt = 0; tt < 512; ++tt) {
                    for (size_t d = 0; d < FROM_MAX_FEATURES; ++d) {
                        batch_x.at(b, tt, d) = samples[i + b].X.at(tt, d);
                    }
                }
            }

            FromOutput out = model.forward(batch_x, false);

            for (size_t b = 0; b < batch_sz; ++b) {
                float p_long = out.probs_dir.at(b, 0);
                float p_flat = out.probs_dir.at(b, 1);
                float p_short = out.probs_dir.at(b, 2);

                float max_prob = std::max({p_long, p_flat, p_short});
                int pred_dir = 0;
                if (p_long == max_prob) pred_dir = 1;
                else if (p_short == max_prob) pred_dir = -1;

                // Ground truth from sample labels
                const auto& s = samples[i + b];
                int truth_dir = 0;
                if (s.y_dir[0] > s.y_dir[1] && s.y_dir[0] > s.y_dir[2]) truth_dir = 1;
                else if (s.y_dir[2] > s.y_dir[1] && s.y_dir[2] > s.y_dir[0]) truth_dir = -1;

                ++total_preds;
                if (pred_dir == truth_dir) ++correct;

                ++signals_total;
                if (max_prob < conf_threshold || pred_dir == 0) {
                    ++signals_flat;
                    continue;
                }

                if (pred_dir == 1) ++signals_long;
                else ++signals_short;

                // Simulate trade: PnL from actual price delta in pips
                // XAUUSD: 1 pip = $0.01, so price_delta * 100 = pips
                double price_delta_pips = (s.exit_mid - s.entry_mid) * 100.0;
                double pnl_net = static_cast<double>(pred_dir) * price_delta_pips - spread_cost;

                trades.push_back({pred_dir, s.entry_mid, s.exit_mid, pnl_net});
            }
        }

        if ((eval_read / 1000000) != ((eval_read - ticks_chunk.size) / 1000000)) {
            std::cout << "  [" << (eval_read/1000000) << "M/" << (eval_ticks/1000000) << "M] "
                      << trades.size() << " trades, "
                      << total_preds << " predictions\n";
        }
    }

    double elapsed = t.elapsed_seconds();

    // Compute stats
    double total_pnl = 0.0;
    double max_drawdown = 0.0;
    double peak = 0.0;
    size_t wins = 0;
    std::vector<double> equity;
    equity.reserve(trades.size() + 1);
    equity.push_back(0.0);

    for (const auto& tr : trades) {
        total_pnl += tr.pnl;
        equity.push_back(total_pnl);
        if (total_pnl > peak) peak = total_pnl;
        double dd = peak - total_pnl;
        if (dd > max_drawdown) max_drawdown = dd;
        if (tr.pnl > 0.0) ++wins;
    }

    double win_rate = trades.empty() ? 0.0 : static_cast<double>(wins) / static_cast<double>(trades.size());
    double avg_pnl = trades.empty() ? 0.0 : total_pnl / static_cast<double>(trades.size());
    double accuracy = total_preds > 0 ? static_cast<double>(correct) / static_cast<double>(total_preds) : 0.0;

    // Sharpe (assume daily steps, rough)
    double mean_ret = avg_pnl;
    double var_ret = 0.0;
    for (const auto& tr : trades) {
        double diff = tr.pnl - mean_ret;
        var_ret += diff * diff;
    }
    double std_ret = trades.size() > 1 ? std::sqrt(var_ret / static_cast<double>(trades.size() - 1)) : 1.0;
    double sharpe = std_ret > 0.0 ? (mean_ret / std_ret) * std::sqrt(252.0) : 0.0;

    std::cout << "\n=======================================================================\n";
    std::cout << "BACKTEST RESULTS\n";
    std::cout << "=======================================================================\n";
    std::cout << std::fixed;
    std::cout << "Duration:      " << std::setprecision(1) << elapsed << "s\n";
    std::cout << "Predictions:   " << total_preds << "\n";
    std::cout << "Accuracy:      " << std::setprecision(2) << (accuracy * 100.0) << "%\n";
    std::cout << "Signals:       " << signals_total << " (long=" << signals_long
              << " short=" << signals_short << " flat=" << signals_flat << ")\n";
    std::cout << "Trades:        " << trades.size() << "\n";
    std::cout << "Win rate:      " << std::setprecision(2) << (win_rate * 100.0) << "%\n";
    std::cout << "Total PnL:     " << std::setprecision(2) << total_pnl << " pips (spread-adjusted)\n";
    std::cout << "Avg PnL/trade: " << std::setprecision(4) << avg_pnl << " pips\n";
    std::cout << "Max Drawdown:  " << std::setprecision(2) << max_drawdown << " pips\n";
    std::cout << "Sharpe (ann):  " << std::setprecision(3) << sharpe << "\n";
    std::cout << "Spread cost:   " << std::setprecision(1) << spread_cost << " pips/trade\n";
    std::cout << "=======================================================================\n";

    return 0;
}

}  // namespace from
