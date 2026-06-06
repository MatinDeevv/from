#pragma once

#include "utils/metrics.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace from {

class ProgressUI {
    TrainingMetrics& metrics_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    int total_epochs_ = 1;
    int64_t total_steps_ = 1;
    std::chrono::steady_clock::time_point start_time_;

    // Make bars that actually update!
    static std::string bar(float frac, int width = 40) {
        frac = std::max(0.0f, std::min(1.0f, frac));
        int filled = static_cast<int>(frac * static_cast<float>(width));
        std::string s = "[";
        for (int i = 0; i < width; ++i) {
            if (i < filled) s += "=";
            else if (i == filled && filled < width) s += ">";
            else s += " ";
        }
        s += "]";
        return s;
    }

    static std::string mini_bar(float frac, int width = 10) {
        frac = std::max(0.0f, std::min(1.0f, frac));
        int filled = static_cast<int>(frac * static_cast<float>(width));
        std::string s;
        for (int i = 0; i < width; ++i) s += (i < filled) ? '|' : '.';
        return s;
    }

    static std::string trend(float v) {
        if (v > 0.02f) return "^";
        if (v < -0.02f) return "v";
        return "~";
    }

    static std::string format_time(int64_t seconds) {
        if (seconds < 0) seconds = 0;
        if (seconds < 60) return std::to_string(seconds) + "s";
        if (seconds < 3600) {
            int m = seconds / 60;
            int s = seconds % 60;
            return std::to_string(m) + "m " + std::to_string(s) + "s";
        }
        int h = seconds / 3600;
        int m = (seconds % 3600) / 60;
        return std::to_string(h) + "h " + std::to_string(m) + "m";
    }

public:
    explicit ProgressUI(TrainingMetrics& metrics) : metrics_(metrics) {
#ifdef _WIN32
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(h, &mode)) {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
#endif
        start_time_ = std::chrono::steady_clock::now();
    }

    void start(int total_epochs, int64_t total_steps) {
        total_epochs_ = std::max(1, total_epochs);
        total_steps_ = std::max<int64_t>(1, total_steps);
        start_time_ = std::chrono::steady_clock::now();
        running_.store(true);

        thread_ = std::thread([this]() {
            float prev_loss = metrics_.loss_total.load();
            float prev_acc = metrics_.acc_direction.load();

            while (running_.load()) {
                // Read all metrics (these update from training thread)
                int step = metrics_.step.load();
                int epoch = metrics_.epoch.load();

                // Calculate progress (FIX: actually use current step!)
                float step_progress = (total_steps_ > 0)
                    ? static_cast<float>(step) / static_cast<float>(total_steps_)
                    : 0.0f;

                float epoch_progress = (total_epochs_ > 0)
                    ? static_cast<float>(epoch - 1) / static_cast<float>(total_epochs_)
                    : 0.0f;

                // Load metrics
                float loss = metrics_.loss_total.load();
                float acc = metrics_.acc_direction.load();
                float loss_dir = metrics_.loss_dir.load();
                float loss_quant = metrics_.loss_quant.load();
                float loss_vol = metrics_.loss_vol.load();
                float loss_spread = metrics_.loss_spread.load();
                float acc_ema = metrics_.acc_ema.load();
                float acc_val = metrics_.acc_val.load();
                float acc_val_best = metrics_.acc_val_best.load();
                float ood_rate = metrics_.ood_rate.load();
                float gpu = metrics_.gpu_util.load();
                float vram = metrics_.gpu_vram_used_gb.load();
                float cpu = metrics_.cpu_util.load();
                float ram = metrics_.ram_used_gb.load();
                float lr = metrics_.learning_rate.load();
                float grad = metrics_.grad_norm.load();
                int64_t eta = metrics_.eta_seconds.load();
                int64_t best_step = metrics_.step_val_best.load();
                float windows_sec = metrics_.windows_per_sec.load();
                float rows_sec = metrics_.rows_per_sec.load();
                int64_t rows = metrics_.rows_read.load();

                // Calculate elapsed time
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

                // Build frame
                std::ostringstream os;

                // Clear screen and reset cursor
                os << "\033[2J\033[H";

                // Title bar
                os << "================================================================================\n";
                os << "  FROM - Neural Market Intelligence Engine - MAXIMUM SPEED MODE\n";
                os << "================================================================================\n";
                os << "\n";

                // Step and time info
                os << "  Step: " << std::setw(7) << step << " / " << total_steps_
                   << "    Epoch: " << epoch << " / " << total_epochs_
                   << "    Elapsed: " << format_time(elapsed)
                   << "    ETA: " << format_time(eta) << "\n";
                os << "\n";

                // Progress bars (these should update now!)
                os << "  Step Progress  " << bar(step_progress, 50) << " "
                   << std::fixed << std::setprecision(1) << (step_progress * 100.0f) << "%\n";
                os << "  Epoch Progress " << bar(epoch_progress, 50) << " "
                   << std::fixed << std::setprecision(1) << (epoch_progress * 100.0f) << "%\n";
                os << "\n";

                os << "--------------------------------------------------------------------------------\n";
                os << "  LOSS & ACCURACY\n";
                os << "--------------------------------------------------------------------------------\n";

                // Loss metrics
                os << "  Total Loss      " << std::setw(8) << std::setprecision(4) << loss
                   << " " << trend(prev_loss - loss)
                   << "      Direction Acc   " << std::setw(6) << std::setprecision(2) << (acc * 100.0f) << "% "
                   << trend(acc - prev_acc) << "\n";

                os << "  Direction       " << std::setw(8) << std::setprecision(4) << loss_dir
                   << "          EMA Accuracy    " << std::setw(6) << std::setprecision(2) << (acc_ema * 100.0f) << "%\n";

                os << "  Quantile        " << std::setw(8) << std::setprecision(4) << loss_quant
                   << "          Val Accuracy    " << std::setw(6) << std::setprecision(2) << (acc_val * 100.0f) << "%\n";

                os << "  Volatility      " << std::setw(8) << std::setprecision(4) << loss_vol
                   << "          Best Val Acc    " << std::setw(6) << std::setprecision(2) << (acc_val_best * 100.0f)
                   << "% (step " << best_step << ")\n";

                os << "  Spread          " << std::setw(8) << std::setprecision(4) << loss_spread
                   << "          OOD Rate        " << std::setw(6) << std::setprecision(2) << (ood_rate * 100.0f) << "%\n";
                os << "\n";

                os << "--------------------------------------------------------------------------------\n";
                os << "  HARDWARE UTILIZATION\n";
                os << "--------------------------------------------------------------------------------\n";

                // GPU metrics with bars
                os << "  GPU Util    " << mini_bar(gpu / 100.0f, 20) << " "
                   << std::setw(5) << std::setprecision(1) << gpu << "%";
                if (gpu < 70.0f) os << "  [WARNING: GPU underutilized!]";
                os << "\n";

                os << "  GPU VRAM    " << mini_bar(vram / 4.0f, 20) << " "
                   << std::setw(4) << std::setprecision(2) << vram << " / 4.0 GB";
                if (vram < 2.0f) os << "  [WARNING: VRAM underutilized!]";
                os << "\n";

                os << "  CPU Util    " << mini_bar(cpu / 100.0f, 20) << " "
                   << std::setw(5) << std::setprecision(1) << cpu << "%";
                if (cpu < 60.0f) os << "  [WARNING: CPU underutilized!]";
                os << "\n";

                os << "  RAM Usage   " << mini_bar(ram / 16.0f, 20) << " "
                   << std::setw(5) << std::setprecision(2) << ram << " GB\n";
                os << "\n";

                os << "--------------------------------------------------------------------------------\n";
                os << "  THROUGHPUT\n";
                os << "--------------------------------------------------------------------------------\n";

                os << "  Windows/sec     " << std::setw(12) << std::setprecision(0) << std::fixed << windows_sec << "\n";
                os << "  Rows/sec        " << std::setw(12) << std::setprecision(0) << std::fixed << rows_sec << "\n";
                os << "  Rows processed  " << std::setw(12) << rows << "\n";
                os << "\n";

                os << "--------------------------------------------------------------------------------\n";
                os << "  TRAINING HYPERPARAMETERS\n";
                os << "--------------------------------------------------------------------------------\n";

                os << "  Learning Rate   " << std::setw(10) << std::setprecision(6) << std::scientific << lr << "\n";
                os << "  Gradient Norm   " << std::setw(10) << std::setprecision(4) << std::fixed << grad << "\n";
                os << "\n";

                os << "================================================================================\n";

                // Write frame
                std::string frame = os.str();
                fwrite(frame.data(), 1, frame.size(), stdout);
                fflush(stdout);

                // Store for trend calculation
                prev_loss = loss;
                prev_acc = acc;

                // Update every 500ms
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }

    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
    }

    ~ProgressUI() { stop(); }
};

}  // namespace from
