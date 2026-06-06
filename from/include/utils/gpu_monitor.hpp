#pragma once

#include "utils/metrics.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>

namespace from {

class GpuMonitor {
    TrainingMetrics* metrics_ = nullptr;
    std::thread monitor_thread_;
    std::atomic<bool> running_{false};

    static bool query_nvidia_smi(float* util, float* vram_gb) {
        FILE* pipe = _popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>NUL", "r");
        if (!pipe) return false;
        char buf[256] = {};
        bool ok = fgets(buf, sizeof(buf), pipe) != nullptr;
        int rc = _pclose(pipe);
        if (!ok || rc != 0) return false;
        std::string s(buf);
        for (char& c : s) if (c == ',') c = ' ';
        std::stringstream ss(s);
        float mem_mb = 0.0f;
        ss >> *util >> mem_mb;
        *vram_gb = mem_mb / 1024.0f;
        return ss.good() || ss.eof();
    }

public:
    explicit GpuMonitor(TrainingMetrics* metrics = nullptr) : metrics_(metrics) {
        if (!metrics_) return;
        running_.store(true);
        monitor_thread_ = std::thread([this]() {
            bool available = true;
            while (running_.load() && available) {
                float util = 0.0f;
                float vram = 0.0f;
                if (query_nvidia_smi(&util, &vram)) {
                    metrics_->gpu_util.store(util);
                    metrics_->gpu_vram_used_gb.store(vram);
                } else {
                    available = false;
                }
                if (available)
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }

    ~GpuMonitor() {
        running_.store(false);
        if (monitor_thread_.joinable()) monitor_thread_.join();
    }

    float utilization_percent() const { return metrics_ ? metrics_->gpu_util.load() : 0.0f; }
    float vram_used_gb() const { return metrics_ ? metrics_->gpu_vram_used_gb.load() : 0.0f; }
    float temperature_celsius() const { return 0.0f; }
    float power_watts() const { return 0.0f; }
};

}  // namespace from
