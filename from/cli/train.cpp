#include "commands.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "model/direction_model.hpp"
#include "model/sequence_model.hpp"
#include "utils/config_parser.hpp"
#include "utils/timer.hpp"

#ifdef FROM_CUDA
#include "cuda/gpu_trainer.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#define NOMINMAX
#include <windows.h>
#else
#include <x86intrin.h>
#endif

namespace from {
namespace {

void enable_ansi_colors() {
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

}  // namespace

int run_train(const CliArgs& args) {
    enable_ansi_colors();
    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string config_path = args.get("--config", "config.toml");
    bool use_linear = args.has("--linear");
    bool use_ensemble = args.has("--ensemble");

    Config cfg;
    if (!config_path.empty() && std::filesystem::exists(config_path)) cfg.load(config_path);

    size_t chunk_size = arg_size(args, "--chunk-size", cfg.get_size("data.chunk_size", 2000000));
    size_t window = arg_size(args, "--window", cfg.get_size("data.window_size", 512));
    size_t stride = arg_size(args, "--stride", cfg.get_size("data.stride", 64));
    size_t horizon = arg_size(args, "--horizon", cfg.get_size("data.horizon", 256));
    size_t batch_size = arg_size(args, "--batch-size", cfg.get_size("training.batch_size", 32));
    size_t max_steps = arg_size(args, "--max-steps", 5000);
    size_t save_every = arg_size(args, "--save-every", 500);
    size_t validate_every = arg_size(args, "--validate-every", 500);
    float lr = arg_float(args, "--lr", 0.0003f);
    uint64_t freeze_after = static_cast<uint64_t>(cfg.get_size("data.normalize_freeze_after", 100000));
    float dir_threshold = cfg.get_float("data.direction_threshold", 2.0f);

    // ====================================================================
    // PHASE 1: Load data — use binary cache if available (instant), else
    // process parquet and save cache for next time
    // ====================================================================
    std::vector<float> all_summaries;  // [N × 176] contiguous
    std::vector<uint8_t> all_labels;   // argmax of y_dir per sample
    size_t num_samples = 0;

    size_t max_samples = arg_size(args, "--max-samples", 500000);
    std::string cache_path = data + ".cache";

    Timer load_timer;

    // Try loading binary cache first
    bool cache_loaded = false;
    if (std::filesystem::exists(cache_path)) {
        std::ifstream cache(cache_path, std::ios::binary);
        if (cache) {
            char magic[4];
            cache.read(magic, 4);
            if (std::memcmp(magic, "FTC1", 4) == 0) {
                uint64_t n = 0, dim = 0;
                cache.read(reinterpret_cast<char*>(&n), 8);
                cache.read(reinterpret_cast<char*>(&dim), 8);
                if (dim == SEQ_SUMMARY_DIM && n > 0 && n >= max_samples) {
                    num_samples = static_cast<size_t>(n);
                    if (num_samples > max_samples) num_samples = max_samples;
                    all_summaries.resize(num_samples * SEQ_SUMMARY_DIM);
                    all_labels.resize(num_samples);
                    cache.read(reinterpret_cast<char*>(all_summaries.data()),
                              static_cast<std::streamsize>(num_samples * SEQ_SUMMARY_DIM * sizeof(float)));
                    cache.read(reinterpret_cast<char*>(all_labels.data()),
                              static_cast<std::streamsize>(num_samples));
                    if (cache) {
                        cache_loaded = true;
                        double t = load_timer.elapsed_seconds();
                        std::cout << "\033[32m[CACHE] Loaded " << num_samples << " samples in "
                                  << std::fixed << std::setprecision(2) << t << "s from " << cache_path << "\033[0m" << std::endl;
                    }
                } else if (dim == SEQ_SUMMARY_DIM && n > 0 && n < max_samples) {
                    std::cout << "\033[33m[CACHE] Stale: has " << n << " samples but need " << max_samples
                              << " — regenerating...\033[0m" << std::endl;
                } else if (dim != SEQ_SUMMARY_DIM && n > 0) {
                    std::cout << "\033[33m[CACHE] Stale: dim=" << dim << " but SEQ_SUMMARY_DIM=" << SEQ_SUMMARY_DIM
                              << " — features changed, regenerating cache...\033[0m" << std::endl;
                }
            }
        }
    }

    if (!cache_loaded) {
        Normalizer normalizer(FROM_MAX_FEATURES);
        ParquetReader reader(data);
        TickProcessor processor;
        Windower windower(window, stride, horizon, dir_threshold);
        size_t total_rows = reader.total_rows();
        size_t rows_loaded = 0;

        std::cout << "============================================================\n";
        std::cout << " First run: processing parquet → binary cache\n";
        std::cout << " " << (total_rows / 1000000) << "M rows (next run will be instant)\n";
        std::cout << "============================================================" << std::endl;

        all_summaries.reserve(max_samples * SEQ_SUMMARY_DIM);
        all_labels.reserve(max_samples);

        while (reader.has_next_chunk() && num_samples < max_samples) {
            TickChunk ticks = reader.read_chunk(chunk_size);
            if (ticks.size == 0) break;
            rows_loaded += ticks.size;

            FeatureChunk features = processor.process(ticks);
            bool update_norm = normalizer.count()[0] < freeze_after;
            features.features = normalizer.normalize_chunk(features.features, update_norm);
            if (normalizer.count()[0] >= freeze_after) normalizer.freeze();

            std::vector<Sample> samples = windower.add(features);
            for (auto& s : samples) {
                if (num_samples >= max_samples) break;
                SequenceModel::precompute_summary(s);
                size_t offset = all_summaries.size();
                all_summaries.resize(offset + SEQ_SUMMARY_DIM);
                std::memcpy(all_summaries.data() + offset, s.summary.data(), SEQ_SUMMARY_DIM * sizeof(float));
                uint8_t label = 1;
                if (s.y_dir[0] > s.y_dir[1] && s.y_dir[0] > s.y_dir[2]) label = 0;
                else if (s.y_dir[2] > s.y_dir[0] && s.y_dir[2] > s.y_dir[1]) label = 2;
                all_labels.push_back(label);
                ++num_samples;
            }

            if (rows_loaded % 10000000 == 0) {
                std::cout << "  [LOAD] " << (rows_loaded / 1000000) << "M/" << (total_rows / 1000000) << "M rows → "
                          << num_samples << " samples" << std::endl;
            }
        }

        // Save binary cache for instant loading next time
        {
            std::ofstream cache(cache_path, std::ios::binary);
            char magic[4] = {'F', 'T', 'C', '1'};
            cache.write(magic, 4);
            uint64_t n = num_samples, dim = SEQ_SUMMARY_DIM;
            cache.write(reinterpret_cast<const char*>(&n), 8);
            cache.write(reinterpret_cast<const char*>(&dim), 8);
            cache.write(reinterpret_cast<const char*>(all_summaries.data()),
                       static_cast<std::streamsize>(num_samples * SEQ_SUMMARY_DIM * sizeof(float)));
            cache.write(reinterpret_cast<const char*>(all_labels.data()),
                       static_cast<std::streamsize>(num_samples));
            std::cout << "[CACHE] Saved " << cache_path << " for instant loading next time" << std::endl;
        }
    }

    double load_secs = load_timer.elapsed_seconds();
    size_t data_mb = num_samples * SEQ_SUMMARY_DIM * 4 / 1048576;
    if (!cache_loaded) {
        std::cout << "\n[LOADED] " << num_samples << " samples in "
                  << std::fixed << std::setprecision(1) << load_secs << "s ("
                  << data_mb << " MB in RAM)" << std::endl;
    }

    if (num_samples < batch_size * 2) {
        std::cout << "[ERROR] Not enough samples to train\n";
        return 1;
    }

    // Temporal split: train on first 90%, validate on last 10% (forward-looking)
    size_t val_count = num_samples / 10;
    if (val_count < batch_size) val_count = batch_size;
    size_t train_count = num_samples - val_count;

    // Print class distribution
    size_t train_cls[3] = {0,0,0}, val_cls[3] = {0,0,0};
    for (size_t i = 0; i < train_count; ++i) train_cls[all_labels[i]]++;
    for (size_t i = train_count; i < num_samples; ++i) val_cls[all_labels[i]]++;
    std::cout << "[SPLIT] train=" << train_count << " val=" << val_count
              << " | train_dist=[" << train_cls[0] << "," << train_cls[1] << "," << train_cls[2] << "]"
              << " val_dist=[" << val_cls[0] << "," << val_cls[1] << "," << val_cls[2] << "]" << std::endl;

    // ====================================================================
    // PHASE 1.5: Feature normalization (z-score from training data)
    // Without this, raw price features (~2000) dominate tiny features (~0.001)
    // ====================================================================
    std::vector<float> feat_mean(SEQ_SUMMARY_DIM, 0.0f);
    std::vector<float> feat_std(SEQ_SUMMARY_DIM, 1.0f);
    {
        // Compute mean over training set
        size_t norm_n = std::min(train_count, static_cast<size_t>(200000));
        for (size_t i = 0; i < norm_n; ++i) {
            const float* row = all_summaries.data() + i * SEQ_SUMMARY_DIM;
            for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) feat_mean[d] += row[d];
        }
        float inv_n = 1.0f / static_cast<float>(norm_n);
        for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) feat_mean[d] *= inv_n;

        // Compute std
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

        // Apply z-score to ALL data (train + val)
        for (size_t i = 0; i < num_samples; ++i) {
            float* row = all_summaries.data() + i * SEQ_SUMMARY_DIM;
            for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
                row[d] = (row[d] - feat_mean[d]) / feat_std[d];
                // Clip to [-5, 5] to avoid extreme outliers
                if (row[d] > 5.0f) row[d] = 5.0f;
                if (row[d] < -5.0f) row[d] = -5.0f;
            }
        }
        std::cout << "\033[32m[NORM] Z-score normalization applied (clip [-5,5])\033[0m" << std::endl;
    }

    // ====================================================================
    // PHASE 1.6: Compute class weights (inverse frequency)
    // This prevents the model from just predicting the majority class
    // ====================================================================
    float class_weight[3];
    {
        float total = static_cast<float>(train_count);
        for (int c = 0; c < 3; ++c) {
            float freq = static_cast<float>(train_cls[c]) / total;
            class_weight[c] = 1.0f / (3.0f * freq + 1e-8f);
        }
        // Normalize so mean weight = 1
        float mean_w = (class_weight[0] + class_weight[1] + class_weight[2]) / 3.0f;
        for (int c = 0; c < 3; ++c) class_weight[c] /= mean_w;
        std::cout << "\033[33m[WEIGHTS] class_w=[" << std::setprecision(3)
                  << class_weight[0] << "," << class_weight[1] << "," << class_weight[2] << "]\033[0m" << std::endl;
    }

    // ====================================================================
    // PHASE 2: TRAINING
    // ====================================================================
    if (use_linear) {
        std::cout << "[LINEAR] Not supported in turbo mode, use --no-turbo\n";
        return 1;
    }

    auto train_one = [&](uint32_t seed_val) -> float {
        SequenceModel model(lr, seed_val);
        // Store second-pass normalization into model for serialization
        model.feat_mean = feat_mean;
        model.feat_std = feat_std;
        model.feat_norm_ready = true;
        std::string prefix = use_ensemble ? "weights_seed" + std::to_string(seed_val) : "weights";
        std::string best_path = prefix + "_best.from";

        std::string resume = args.get("--model", "");
        if (!use_ensemble && !resume.empty() && std::filesystem::exists(resume)) {
            if (SequenceModelIO::load(resume, model)) {
                model.lr = lr;
                std::cout << "[RESUME] " << resume << "\n";
            }
        }

        std::cout << "[TRAIN] seed=" << seed_val << " params=" << model.total_params
                  << " lr=" << std::scientific << std::setprecision(1) << lr << std::fixed
                  << " batch=" << batch_size
                  << " train=" << train_count << " val=" << val_count << "\n";

        // ============================================================
        // Try GPU path first
        // ============================================================
        bool use_gpu = false;
#ifdef FROM_CUDA
        cuda::GpuTrainer gpu;
        use_gpu = gpu.initialize(static_cast<int>(batch_size), all_summaries, all_labels,
                                 model.w1.data(), model.b1.data(),
                                 model.w2.data(), model.b2.data(),
                                 model.w3.data(), model.b3.data());
        if (use_gpu) {
            gpu.set_class_weights(class_weight);
            std::cout << "\033[32m[MODE] GPU cuBLAS — batch=" << batch_size << " target 2M+ samples/sec\033[0m\n" << std::endl;
        }
#endif
        if (!use_gpu) {
            std::cout << "[MODE] CPU AVX2 — target 1,500+ steps/sec\n" << std::endl;
        }

        // Pre-allocate ALL working buffers
        std::vector<float> batch_input(batch_size * SEQ_SUMMARY_DIM);
        std::vector<float> batch_logits(batch_size * SEQ_NUM_CLASSES);
        std::vector<float> batch_probs(batch_size * SEQ_NUM_CLASSES);
        std::vector<float> batch_grad_logits(batch_size * SEQ_NUM_CLASSES);
        std::vector<uint32_t> batch_indices(batch_size);

        // Fast pseudo-random index via xorshift
        uint32_t xor_state = seed_val + 1;
        auto xorshift = [&]() -> uint32_t {
            xor_state ^= xor_state << 13;
            xor_state ^= xor_state >> 17;
            xor_state ^= xor_state << 5;
            return xor_state;
        };

        size_t step = 0;
        float best_val = 0.0f, best_val_loss = 1e9f, ema = 0.0f;
        float best_edge = -1.0f;
        size_t epoch = 0;
        size_t samples_seen = 0;
        float base_lr = lr;
        size_t warmup_steps = 2000;
        size_t cosine_period = 50000;  // cosine annealing period

        Timer total_timer;
        Timer speed_timer;
        size_t speed_steps = 0;

        // Color helpers
        auto col = [](float val, float good, float bad) -> const char* {
            if (val <= good) return "\033[32m";
            if (val >= bad) return "\033[31m";
            return "\033[33m";
        };
        auto col_hi = [](float val, float good, float bad) -> const char* {
            if (val >= good) return "\033[32m";
            if (val <= bad) return "\033[31m";
            return "\033[33m";
        };
        const char* RST = "\033[0m";

        // Burst size: how many steps to queue at once on GPU before syncing
        size_t burst = 100;
        if (validate_every > 0 && validate_every < burst) burst = validate_every;
        if (save_every > 0 && save_every < burst) burst = save_every;

        while (step < max_steps) {
            // Learning rate: warmup then cosine annealing with warm restarts
            if (step < warmup_steps) {
                lr = base_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
            } else {
                float progress = static_cast<float>((step - warmup_steps) % cosine_period)
                               / static_cast<float>(cosine_period);
                lr = base_lr * 0.5f * (1.0f + std::cos(3.14159265f * progress));
                if (lr < base_lr * 0.01f) lr = base_lr * 0.01f;
            }
            float loss = 0.0f, acc = 0.0f;

#ifdef FROM_CUDA
            if (use_gpu) {
                // Compute how many steps to do in this burst
                size_t steps_remaining = max_steps - step;
                size_t this_burst = std::min(burst, steps_remaining);
                size_t next_print = ((step / 1000) + 1) * 1000;
                if (next_print - step < this_burst) this_burst = next_print - step;
                if (validate_every > 0) {
                    size_t next_val = ((step / validate_every) + 1) * validate_every;
                    if (next_val - step < this_burst) this_burst = next_val - step;
                }
                if (save_every > 0) {
                    size_t next_save = ((step / save_every) + 1) * save_every;
                    if (next_save - step < this_burst) this_burst = next_save - step;
                }
                if (step < 3) this_burst = 1;

                for (size_t b = 0; b < this_burst; ++b) {
                    uint32_t seed = static_cast<uint32_t>(step + b + 1) * 2654435761u + 42u;
                    gpu.train_step_gpu_only(static_cast<int>(batch_size), lr, seed);
                }
                loss = gpu.sync_metrics(static_cast<int>(batch_size), &acc);

                step += this_burst;
                speed_steps += this_burst;
                samples_seen += this_burst * batch_size;
                epoch = samples_seen / train_count;
                model.last_grad_norm = gpu.last_grad_norm;
            } else
#endif
            {
                // CPU path with CLASS-WEIGHTED cross-entropy loss
                for (size_t i = 0; i < batch_size; ++i) {
                    batch_indices[i] = xorshift() % static_cast<uint32_t>(train_count);
                }
                samples_seen += batch_size;
                epoch = samples_seen / train_count;

                for (size_t i = 0; i < batch_size; ++i) {
                    std::memcpy(batch_input.data() + i * SEQ_SUMMARY_DIM,
                               all_summaries.data() + batch_indices[i] * SEQ_SUMMARY_DIM,
                               SEQ_SUMMARY_DIM * sizeof(float));
                }
                model.forward(batch_input.data(), batch_size, batch_logits.data(), true);
                SequenceModel::softmax(batch_logits.data(), batch_size, batch_probs.data());

                size_t correct = 0;
                float inv_n = 1.0f / static_cast<float>(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    uint8_t truth = all_labels[batch_indices[i]];
                    size_t pred = 0;
                    for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                        if (batch_probs[i * SEQ_NUM_CLASSES + c] > batch_probs[i * SEQ_NUM_CLASSES + pred]) pred = c;
                    }
                    if (pred == truth) ++correct;
                    float w = class_weight[truth];
                    loss -= w * std::log(batch_probs[i * SEQ_NUM_CLASSES + truth] + FROM_EPS_F);
                    for (size_t c = 0; c < SEQ_NUM_CLASSES; ++c) {
                        float target = (c == truth) ? 1.0f : 0.0f;
                        batch_grad_logits[i * SEQ_NUM_CLASSES + c] = w * (batch_probs[i * SEQ_NUM_CLASSES + c] - target) * inv_n;
                    }
                }
                model.backward(batch_grad_logits.data(), batch_size);
                acc = static_cast<float>(correct) / static_cast<float>(batch_size);
                loss *= inv_n;
                ++step;
                ++speed_steps;
            }

            if (acc > 0.0f) ema = (ema < 0.01f) ? acc : 0.99f * ema + 0.01f * acc;

            if (loss > 10.0f && step > 500) {
                std::cout << "\033[31m[EXPLODE] loss=" << loss << " → lr/=2\033[0m\n";
                if (std::filesystem::exists(best_path)) SequenceModelIO::load(best_path, model);
                base_lr *= 0.5f;
                lr = base_lr;
                continue;
            }

            // Print every 1000 steps (or first 3)
            if (step % 1000 == 0 || step <= 3) {
                double elapsed = speed_timer.elapsed_seconds();
                double sps = (elapsed > 0.001) ? static_cast<double>(speed_steps) / elapsed : 0.0;

                double samples_per_sec = sps * static_cast<double>(batch_size);
                std::cout << "\033[36m[" << std::setw(7) << step << "]\033[0m"
                          << " loss=" << col(loss, 0.9f, 1.2f) << std::fixed << std::setprecision(4) << loss << RST
                          << " acc=" << col_hi(acc, 0.45f, 0.35f) << std::setprecision(3) << acc << RST
                          << " ema=" << col_hi(ema, 0.45f, 0.35f) << ema << RST
                          << " lr=" << std::scientific << std::setprecision(1) << lr << std::fixed
                          << " | " << col_hi(static_cast<float>(sps), 2000.f, 500.f) << std::setprecision(0) << sps << " steps/s" << RST
                          << " " << col_hi(static_cast<float>(samples_per_sec / 1e6), 1.0f, 0.2f) << std::setprecision(2) << (samples_per_sec / 1e6) << "M smp/s" << RST
                          << " ep=" << epoch
                          << " best_edge=" << col_hi(best_edge, 0.02f, 0.0f) << std::setprecision(4) << best_edge << RST
                          << std::endl;
                speed_timer = Timer();
                speed_steps = 0;
            }

            // ============================================================
            // TRADING-RELEVANT VALIDATION
            // ============================================================
            if (validate_every > 0 && step % validate_every == 0) {
#ifdef FROM_CUDA
                if (use_gpu) gpu.download_weights(model.w1.data(), model.b1.data(),
                                                   model.w2.data(), model.b2.data(),
                                                   model.w3.data(), model.b3.data());
#endif
                size_t val_correct = 0;
                float val_loss = 0.0f;
                size_t val_batches = val_count / batch_size;

                // Per-class tracking for trading metrics
                size_t pred_count[3] = {0,0,0};
                size_t pred_correct[3] = {0,0,0};
                size_t conf_trades = 0, conf_correct = 0;
                float conf_threshold = 0.50f;  // Only trade when >50% confident
                float total_edge = 0.0f;

                for (size_t vb = 0; vb < val_batches; ++vb) {
                    size_t vstart = train_count + vb * batch_size;
                    for (size_t i = 0; i < batch_size; ++i) {
                        std::memcpy(batch_input.data() + i * SEQ_SUMMARY_DIM,
                                   all_summaries.data() + (vstart + i) * SEQ_SUMMARY_DIM,
                                   SEQ_SUMMARY_DIM * sizeof(float));
                    }
                    model.forward(batch_input.data(), batch_size, batch_logits.data(), false);
                    SequenceModel::softmax(batch_logits.data(), batch_size, batch_probs.data());
                    for (size_t i = 0; i < batch_size; ++i) {
                        uint8_t truth = all_labels[vstart + i];
                        size_t pred = 0;
                        float max_prob = batch_probs[i * SEQ_NUM_CLASSES];
                        for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) {
                            if (batch_probs[i * SEQ_NUM_CLASSES + c] > max_prob) {
                                max_prob = batch_probs[i * SEQ_NUM_CLASSES + c];
                                pred = c;
                            }
                        }
                        if (pred == truth) ++val_correct;
                        float w = class_weight[truth];
                        val_loss -= w * std::log(batch_probs[i * SEQ_NUM_CLASSES + truth] + FROM_EPS_F);

                        // Trading metrics: only count UP/DOWN predictions (not neutral)
                        pred_count[pred]++;
                        if (pred == truth) pred_correct[pred]++;

                        // Confidence-gated trading
                        if (pred != 1 && max_prob >= conf_threshold) {
                            conf_trades++;
                            if (pred == truth) {
                                conf_correct++;
                                total_edge += max_prob;
                            } else {
                                total_edge -= (1.0f - max_prob);
                            }
                        }
                    }
                }
                float va = static_cast<float>(val_correct) / static_cast<float>(val_batches * batch_size);
                val_loss /= static_cast<float>(val_batches * batch_size);

                // Compute directional accuracy (UP + DOWN only, ignoring NEUTRAL predictions)
                size_t dir_preds = pred_count[0] + pred_count[2];
                size_t dir_correct = pred_correct[0] + pred_correct[2];
                float dir_acc = dir_preds > 0 ? static_cast<float>(dir_correct) / static_cast<float>(dir_preds) : 0.0f;

                // Confidence-gated win rate
                float conf_winrate = conf_trades > 0 ? static_cast<float>(conf_correct) / static_cast<float>(conf_trades) : 0.0f;
                float edge = conf_trades > 0 ? total_edge / static_cast<float>(conf_trades) : 0.0f;
                size_t conf_wrong = conf_trades - conf_correct;

                // Precision per class
                float prec_up = pred_count[0] > 0 ? static_cast<float>(pred_correct[0]) / static_cast<float>(pred_count[0]) : 0.0f;
                float prec_dn = pred_count[2] > 0 ? static_cast<float>(pred_correct[2]) / static_cast<float>(pred_count[2]) : 0.0f;

                // Profit factor: gross_profit / gross_loss
                float gross_profit = (conf_correct > 0) ? total_edge : 0.0f;
                float gross_loss_val = (conf_wrong > 0) ? static_cast<float>(conf_wrong) - total_edge : 1e-6f;
                float profit_factor = (gross_loss_val > 1e-6f) ? std::abs(gross_profit) / gross_loss_val : 0.0f;

                // Kelly fraction: p - (1-p)/b where b = avg_win/avg_loss
                float avg_win = (conf_correct > 0) ? total_edge / static_cast<float>(conf_correct) : 0.0f;
                float avg_loss = (conf_wrong > 0) ? (static_cast<float>(conf_wrong) * 1.0f - total_edge) / static_cast<float>(conf_wrong) : 1.0f;
                float kelly = (avg_loss > 1e-6f)
                    ? conf_winrate - (1.0f - conf_winrate) / (avg_win / avg_loss + 1e-6f)
                    : 0.0f;

                bool improved = (edge > best_edge && conf_trades > 50);
                if (improved || (best_edge < 0.0f && val_loss < best_val_loss)) {
                    if (edge > best_edge && conf_trades > 50) best_edge = edge;
                    best_val = va;
                    best_val_loss = val_loss;
                    SequenceModelIO::save(model, best_path);
                }

                std::cout << "\033[35m[VAL] loss=" << std::setprecision(4) << val_loss
                          << " acc=" << std::setprecision(3) << va
                          << " dir_acc=" << col_hi(dir_acc, 0.52f, 0.48f) << dir_acc << RST
                          << "\033[35m | trades=" << conf_trades
                          << " winrate=" << col_hi(conf_winrate, 0.55f, 0.50f) << std::setprecision(3) << conf_winrate << RST
                          << "\033[35m edge=" << col_hi(edge, 0.02f, 0.0f) << std::setprecision(4) << edge << RST
                          << " pf=" << col_hi(profit_factor, 1.3f, 1.0f) << std::setprecision(2) << profit_factor << RST
                          << " kelly=" << col_hi(kelly, 0.05f, 0.0f) << std::setprecision(3) << kelly << RST
                          << "\033[35m | prec_UP=" << std::setprecision(3) << prec_up
                          << " prec_DN=" << prec_dn
                          << " preds=[" << pred_count[0] << "," << pred_count[1] << "," << pred_count[2] << "]" << RST << "\n";
                if (improved) {
                    std::cout << "\033[32m[BEST] edge=" << std::setprecision(4) << best_edge
                              << " → " << best_path << RST << "\n";
                }
            }

            // Save checkpoint
            if (save_every > 0 && step % save_every == 0) {
                std::ostringstream nm;
                nm << prefix << "_step_" << std::setw(6) << std::setfill('0') << step << ".from";
#ifdef FROM_CUDA
                if (use_gpu) gpu.download_weights(model.w1.data(), model.b1.data(),
                                                  model.w2.data(), model.b2.data(),
                                                  model.w3.data(), model.b3.data());
#endif
                SequenceModelIO::save(model, nm.str());
                std::cout << "\033[33m[SAVE] " << nm.str() << RST << "\n";
            }
        }

        std::string final_path = prefix + ".from";
#ifdef FROM_CUDA
        if (use_gpu) gpu.download_weights(model.w1.data(), model.b1.data(),
                                          model.w2.data(), model.b2.data(),
                                          model.w3.data(), model.b3.data());
#endif
        SequenceModelIO::save(model, final_path);
        double total_secs = total_timer.elapsed_seconds();
        std::cout << "\n\033[32m[DONE] seed=" << seed_val << " best_edge=" << std::setprecision(4) << best_edge
                  << " steps=" << step << " time=" << std::setprecision(1) << total_secs << "s"
                  << " avg=" << std::setprecision(0) << (static_cast<double>(step) / total_secs) << " steps/s\033[0m\n";
        return best_edge;
    };

    if (use_ensemble) {
        std::cout << "========================================\n";
        std::cout << "ENSEMBLE: 3 models (42, 137, 2718)\n";
        std::cout << "========================================\n\n";

        for (uint32_t seed : Ensemble::SEEDS) {
            train_one(seed);
            std::cout << "\n";
        }
        std::cout << "[ENSEMBLE DONE] All 3 saved.\n";
    } else {
        train_one(42);
    }

    return 0;
}

}  // namespace from
