#pragma once

#include "cuda/device_memory.hpp"

#include <stdexcept>

namespace from::cuda {

#ifdef FROM_CUDA
void gemm(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream = 0);
void softmax(float* x, int rows, int cols, cudaStream_t stream = 0);
void softmax_inplace(float* x, int rows, int cols, cudaStream_t stream = 0);
void cross_entropy_loss(const float* probs, const float* targets, float* losses, int batch, int classes, cudaStream_t stream = 0);
void sgd_update(float* params, const float* grads, float lr, int n, cudaStream_t stream = 0);
void adam_step(float* params, float* m, float* v, const float* grads, float lr, float beta1, float beta2,
               float eps, float weight_decay, float bc1, float bc2, int n, cudaStream_t stream = 0);
#else
inline void gemm(const float*, const float*, float*, int, int, int, cudaStream_t = nullptr) {
    throw std::runtime_error("CUDA backend is not compiled");
}
inline void softmax(float*, int, int, cudaStream_t = nullptr) {
    throw std::runtime_error("CUDA backend is not compiled");
}
inline void adam_step(float*, float*, float*, const float*, float, float, float, float, float, float, float, int,
                      cudaStream_t = nullptr) {
    throw std::runtime_error("CUDA backend is not compiled");
}
#endif

}  // namespace from::cuda
