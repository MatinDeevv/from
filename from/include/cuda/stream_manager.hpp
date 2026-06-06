#pragma once

#include "cuda/device_memory.hpp"

#include <stdexcept>

namespace from::cuda {

#ifdef FROM_CUDA

class StreamManager {
    cudaStream_t compute_{};
    cudaStream_t h2d_{};
    cudaStream_t d2h_{};

public:
    StreamManager() {
        if (cudaStreamCreate(&compute_) != cudaSuccess ||
            cudaStreamCreate(&h2d_) != cudaSuccess ||
            cudaStreamCreate(&d2h_) != cudaSuccess) {
            throw std::runtime_error("Could not create CUDA streams");
        }
    }
    ~StreamManager() {
        cudaStreamDestroy(compute_);
        cudaStreamDestroy(h2d_);
        cudaStreamDestroy(d2h_);
    }
    cudaStream_t compute() const { return compute_; }
    cudaStream_t h2d() const { return h2d_; }
    cudaStream_t d2h() const { return d2h_; }
    void synchronize_all() {
        cudaStreamSynchronize(compute_);
        cudaStreamSynchronize(h2d_);
        cudaStreamSynchronize(d2h_);
    }
    void synchronize_compute() { cudaStreamSynchronize(compute_); }
};

struct PinnedBatch {
    float* X = nullptr;
    float* y_dir = nullptr;
    float* y_mag = nullptr;
    float* y_vol = nullptr;
    float* y_spread = nullptr;
    size_t batch_size = 0;
    size_t window = 0;
    size_t features = 0;

    void allocate(size_t B, size_t W, size_t F) {
        free();
        batch_size = B;
        window = W;
        features = F;
        cudaMallocHost(&X, B * W * F * sizeof(float));
        cudaMallocHost(&y_dir, B * 3 * sizeof(float));
        cudaMallocHost(&y_mag, B * sizeof(float));
        cudaMallocHost(&y_vol, B * sizeof(float));
        cudaMallocHost(&y_spread, B * sizeof(float));
    }
    void free() {
        if (X) cudaFreeHost(X);
        if (y_dir) cudaFreeHost(y_dir);
        if (y_mag) cudaFreeHost(y_mag);
        if (y_vol) cudaFreeHost(y_vol);
        if (y_spread) cudaFreeHost(y_spread);
        X = y_dir = y_mag = y_vol = y_spread = nullptr;
    }
    ~PinnedBatch() { free(); }
};

#else

class StreamManager {
public:
    cudaStream_t compute() const { return nullptr; }
    cudaStream_t h2d() const { return nullptr; }
    cudaStream_t d2h() const { return nullptr; }
    void synchronize_all() {}
    void synchronize_compute() {}
};

struct PinnedBatch {
    float* X = nullptr;
    float* y_dir = nullptr;
    float* y_mag = nullptr;
    float* y_vol = nullptr;
    float* y_spread = nullptr;
    size_t batch_size = 0;
    void allocate(size_t, size_t, size_t) { throw std::runtime_error("CUDA pinned memory is not compiled"); }
    void free() {}
};

#endif

}  // namespace from::cuda
