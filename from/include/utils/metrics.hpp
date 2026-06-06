#pragma once

#include <atomic>
#include <cstdint>

namespace from {

struct TrainingMetrics {
    std::atomic<float> loss_total{0};
    std::atomic<float> loss_dir{0};
    std::atomic<float> loss_quant{0};
    std::atomic<float> loss_vol{0};
    std::atomic<float> loss_spread{0};
    std::atomic<float> loss_irm{0};
    std::atomic<float> acc_direction{0};
    std::atomic<float> acc_ema{0};
    std::atomic<float> acc_val{0};
    std::atomic<float> acc_val_best{0};
    std::atomic<int> step_val_best{0};
    std::atomic<float> uncertainty_epistemic{0};
    std::atomic<float> uncertainty_aleatoric{0};
    std::atomic<float> ood_rate{0};
    std::atomic<float> gpu_util{0};
    std::atomic<float> gpu_vram_used_gb{0};
    std::atomic<float> cpu_util{0};
    std::atomic<float> ram_used_gb{0};
    std::atomic<float> windows_per_sec{0};
    std::atomic<float> rows_per_sec{0};
    std::atomic<float> learning_rate{0};
    std::atomic<float> grad_norm{0};
    std::atomic<int64_t> rows_read{0};
    std::atomic<int> step{0};
    std::atomic<int> epoch{0};
    std::atomic<float> expert_util[16];
    std::atomic<int64_t> eta_seconds{0};
};

}  // namespace from
