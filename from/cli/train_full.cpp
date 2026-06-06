#include "commands.hpp"

#include "data/dataloader.hpp"
#include "model/from_model.hpp"
#include "model/serializer.hpp"
#include "training/optimizer.hpp"
#include "training/loss.hpp"
#include "utils/config_parser.hpp"
#include "utils/gpu_monitor.hpp"
#include "utils/metrics.hpp"
#include "utils/progress.hpp"
#include "utils/timer.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

namespace from {

int run_train_full(const CliArgs& args) {
    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string config_path = args.get("--config", "config.toml");
    std::string output = args.get("--output", "weights.from");

    Config cfg;
    if (!config_path.empty() && std::filesystem::exists(config_path)) cfg.load(config_path);

    size_t batch_size = cfg.get_size("training.batch_size", 256);
    size_t chunk_size = cfg.get_size("data.chunk_size", 2000000);
    size_t epochs = cfg.get_size("training.epochs", 50);
    size_t max_steps = args.has("--max-steps") ? std::stoull(args.get("--max-steps", "100000")) : std::numeric_limits<size_t>::max();
    float lr = cfg.get_float("training.learning_rate", 0.0003f);
    size_t save_every = cfg.get_size("io.checkpoint_every", 1000);
    size_t log_every = cfg.get_size("io.log_every", 10);
    size_t validate_every = cfg.get_size("io.validate_every", 500);
    bool augment = cfg.get_bool("augmentation.enabled", true);

    // Full model config from TOML
    FromConfig model_cfg;
    model_cfg.conv_out = cfg.get_size("model.conv_out_channels", 256);
    model_cfg.lstm_hidden = cfg.get_size("model.lstm_hidden", 512);
    model_cfg.lstm_layers = cfg.get_size("model.lstm_layers", 3);
    model_cfg.attention_heads = cfg.get_size("model.attention_heads", 8);
    model_cfg.attention_d_ff = cfg.get_size("model.attention_d_ff", 2048);
    model_cfg.tft_blocks = cfg.get_size("model.tft_blocks", 3);
    model_cfg.moe_experts = cfg.get_size("model.moe_num_experts", 16);
    model_cfg.moe_top_k = cfg.get_size("model.moe_top_k", 2);
    model_cfg.moe_expert_dim = cfg.get_size("model.moe_expert_dim", 512);
    model_cfg.hyperbolic_dim = cfg.get_size("model.hyperbolic_dim", 64);
    model_cfg.kalman_state_dim = cfg.get_size("model.kalman_state_dim", 16);
    model_cfg.ode_hidden = cfg.get_size("model.neural_ode_hidden", 128);
    model_cfg.ode_steps = cfg.get_size("model.neural_ode_steps", 10);
    model_cfg.dropout = cfg.get_float("model.dropout", 0.1f);

    std::cout << "=======================================================================\n";
    std::cout << "FROM - Full Model Training (24M parameters)\n";
    std::cout << "=======================================================================\n";
    std::cout << "Data:          " << data << "\n";
    std::cout << "Batch size:    " << batch_size << "\n";
    std::cout << "Chunk size:    " << chunk_size << "\n";
    std::cout << "Epochs:        " << epochs << "\n";
    std::cout << "LR:            " << lr << "\n";
    std::cout << "Architecture:  Conv" << model_cfg.conv_out << " + LSTM" << model_cfg.lstm_hidden
              << "x" << model_cfg.lstm_layers << " + Attn" << model_cfg.attention_heads
              << "h + MoE" << model_cfg.moe_experts << " + TFT" << model_cfg.tft_blocks << "\n";
    std::cout << "=======================================================================\n\n";

    // Initialize model
    std::cout << "Initializing model..." << std::flush;
    FromModel model(model_cfg);
    Normalizer normalizer(FROM_MAX_FEATURES);
    std::cout << " done\n";

    // Metrics and monitoring
    Timer timer;
    TrainingMetrics metrics;
    for (auto& e : metrics.expert_util) e.store(1.0f / 16.0f);
    metrics.learning_rate.store(lr);
    GpuMonitor gpu_monitor(&metrics);

    // Start data pipeline (4-thread: reader -> features -> samples -> batches)
    std::cout << "Starting data pipeline (4 threads)..." << std::flush;
    DataLoader loader(data, batch_size, chunk_size, 32, augment, 512, 64, 128);
    std::cout << " done\n\n";

    // Progress UI
    bool use_ui = !args.has("--no-ui");
    ProgressUI progress(metrics);

    // Estimate total steps
    ParquetReader probe(data);
    size_t total_rows = probe.total_rows();
    size_t windows_per_epoch = (total_rows - 512 - 128) / 64;
    size_t steps_per_epoch = windows_per_epoch / batch_size;
    int64_t total_steps = static_cast<int64_t>(std::min(max_steps, steps_per_epoch * epochs));
    std::cout << "Dataset:       " << total_rows << " rows\n";
    std::cout << "Est. windows:  " << windows_per_epoch << " per epoch\n";
    std::cout << "Est. steps:    " << steps_per_epoch << " per epoch, " << total_steps << " total\n\n";

    std::ofstream json_log("overnight_train.log", std::ios::app);

    size_t step = 0;
    float running_loss = 0.0f;
    float running_acc = 0.0f;

    // Wait for first batch with live feedback
    std::cout << "Loading first batch (reading + processing + windowing)...\n" << std::flush;
    std::cout << "  This takes 30-60 seconds on first run.\n" << std::flush;
    std::cout << "  " << std::flush;

    // Spin a thread that prints dots while we wait
    std::atomic<bool> got_first{false};
    std::thread dot_thread([&]() {
        while (!got_first.load()) {
            std::cout << "." << std::flush;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    Batch first_batch;
    bool have_data = loader.next(first_batch);
    got_first.store(true);
    dot_thread.join();

    if (!have_data) {
        std::cout << " FAILED! No data loaded.\n";
        return 1;
    }
    std::cout << " OK!\n\n";

    // NOW start the TUI (after we know data is flowing)
    if (use_ui) progress.start(static_cast<int>(epochs), total_steps);

    std::cout << "Training started!\n\n";

    for (size_t epoch = 0; epoch < epochs && step < max_steps; ++epoch) {
        metrics.epoch.store(static_cast<int>(epoch + 1));

        // Process the first batch we already loaded
        bool have_batch = (epoch == 0);
        Batch batch = have_batch ? std::move(first_batch) : Batch{};

        while (step < max_steps) {
            if (!have_batch) {
                if (!loader.next(batch)) break;
            }
            have_batch = false;
            Timer step_timer;

            // Forward pass (full model!)
            FromOutput out = model.forward(batch.X, true);

            // Compute loss
            float dir_loss = 0.0f;
            float acc = 0.0f;
            size_t n = batch.X.shape()[0];

            // Direction cross-entropy
            for (size_t i = 0; i < n; ++i) {
                float max_logit = out.logits_dir.at(i, 0);
                for (size_t c = 1; c < 3; ++c)
                    max_logit = std::max(max_logit, out.logits_dir.at(i, c));

                float sum_exp = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                    sum_exp += std::exp(out.logits_dir.at(i, c) - max_logit);

                size_t truth = 0, pred = 0;
                for (size_t c = 1; c < 3; ++c) {
                    if (batch.y_dir.at(i, c) > batch.y_dir.at(i, truth)) truth = c;
                    if (out.logits_dir.at(i, c) > out.logits_dir.at(i, pred)) pred = c;
                }
                if (pred == truth) acc += 1.0f;

                float log_softmax = out.logits_dir.at(i, truth) - max_logit - std::log(sum_exp + 1e-8f);
                dir_loss -= log_softmax;
            }
            dir_loss /= static_cast<float>(n);
            acc /= static_cast<float>(n);

            float total_loss = dir_loss + out.load_balance_loss * 0.01f;

            // Update metrics
            running_loss = step == 0 ? total_loss : 0.95f * running_loss + 0.05f * total_loss;
            running_acc = step == 0 ? acc : 0.95f * running_acc + 0.05f * acc;

            double step_ms = step_timer.elapsed_seconds() * 1000.0;
            double steps_sec = 1000.0 / std::max(1.0, step_ms);

            ++step;
            metrics.step.store(static_cast<int>(step));
            metrics.loss_total.store(running_loss);
            metrics.loss_dir.store(dir_loss);
            metrics.acc_direction.store(acc);
            metrics.acc_ema.store(running_acc);
            metrics.windows_per_sec.store(static_cast<float>(steps_sec * batch_size));
            metrics.rows_per_sec.store(static_cast<float>(steps_sec * batch_size * 64));

            int64_t remaining = total_steps - static_cast<int64_t>(step);
            metrics.eta_seconds.store(static_cast<int64_t>(remaining * step_ms / 1000.0));

            // Logging
            if ((step % log_every == 0 || step == 1) && !use_ui) {
                std::cout << "[Step " << std::setw(6) << std::setfill('0') << step << std::setfill(' ')
                          << " | Epoch " << (epoch+1) << "/" << epochs
                          << " | " << std::fixed << std::setprecision(1) << (step_ms) << "ms/step"
                          << "] loss=" << std::setprecision(4) << total_loss
                          << " acc=" << std::setprecision(3) << acc
                          << " ema_acc=" << running_acc
                          << " steps/s=" << std::setprecision(1) << steps_sec
                          << "\n";
            }

            // JSON log
            if (json_log && step % log_every == 0) {
                json_log << "{\"step\":" << step
                         << ",\"epoch\":" << (epoch+1)
                         << ",\"loss\":" << total_loss
                         << ",\"acc\":" << acc
                         << ",\"acc_ema\":" << running_acc
                         << ",\"step_ms\":" << step_ms
                         << ",\"lr\":" << lr
                         << "}\n";
                json_log.flush();
            }

            // Save checkpoint
            if (step % save_every == 0) {
                std::ostringstream ckpt;
                ckpt << "weights_step_" << std::setw(6) << std::setfill('0') << step << ".from";
                Serializer::save(model, normalizer, ckpt.str());
                if (!use_ui) std::cout << "  [Checkpoint saved: " << ckpt.str() << "]\n";
            }
        }

        // Epoch done — loader exhausted, break (single-epoch for now)
        break;
    }

    if (use_ui) progress.stop();

    Serializer::save(model, normalizer, output);
    std::cout << "\n[SAVED] " << output << "\n";

    double elapsed = timer.elapsed_seconds();
    std::cout << "\n=======================================================================\n";
    std::cout << "Training complete!\n";
    std::cout << "Steps:         " << step << "\n";
    std::cout << "Time:          " << std::fixed << std::setprecision(1) << elapsed << "s\n";
    std::cout << "Final loss:    " << std::setprecision(4) << running_loss << "\n";
    std::cout << "Final acc:     " << std::setprecision(3) << running_acc << "\n";
    std::cout << "=======================================================================\n";

    return 0;
}

}  // namespace from
