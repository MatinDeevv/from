#pragma once

/*
 * cuBLAS-accelerated trainer for the full FROM model.
 * All matmuls on GPU via cuBLAS. All activations via custom kernels.
 * Upload batch once → entire forward/backward on GPU → download loss.
 *
 * Target: 1000+ steps/sec on RTX 3050.
 */

#ifdef FROM_CUDA

#include "common.h"
#include "data/dataloader.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace from::cuda {

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline void check_cublas(cublasStatus_t err, const char* msg) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": cuBLAS error " + std::to_string(err));
    }
}

// Activation kernels (defined in kernels.cu)
void relu_forward(float* x, int n, cudaStream_t stream);
void relu_backward(float* grad, const float* x, int n, cudaStream_t stream);
void softmax_inplace(float* x, int rows, int cols, cudaStream_t stream);
void cross_entropy_loss(const float* probs, const float* targets, float* losses, int batch, int classes, cudaStream_t stream);
void sgd_update(float* params, const float* grads, float lr, int n, cudaStream_t stream);

// Simple linear model on GPU: z = X @ W^T + b, then softmax
// This is the DirectionModel but running ENTIRELY on GPU
class CublasTrainer {
    cublasHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Model params on device
    float* d_W_ = nullptr;       // [out_dim, in_dim] = [3, 32]
    float* d_b_ = nullptr;       // [3]

    // Workspace on device (per-batch)
    float* d_X_ = nullptr;       // [batch, in_dim]
    float* d_Y_ = nullptr;       // [batch, 3] targets
    float* d_logits_ = nullptr;  // [batch, 3]
    float* d_probs_ = nullptr;   // [batch, 3]
    float* d_loss_ = nullptr;    // [batch]
    float* d_grad_W_ = nullptr;  // [3, in_dim]
    float* d_grad_b_ = nullptr;  // [3]

    // Host staging
    float* h_loss_ = nullptr;    // pinned
    float* h_probs_ = nullptr;   // pinned

    int in_dim_ = 0;
    int out_dim_ = 3;
    int batch_cap_ = 0;
    bool ready_ = false;

    void alloc_batch(int batch) {
        if (batch <= batch_cap_) return;
        if (d_X_) cudaFree(d_X_);
        if (d_Y_) cudaFree(d_Y_);
        if (d_logits_) cudaFree(d_logits_);
        if (d_probs_) cudaFree(d_probs_);
        if (d_loss_) cudaFree(d_loss_);
        if (h_loss_) cudaFreeHost(h_loss_);
        if (h_probs_) cudaFreeHost(h_probs_);

        batch_cap_ = batch;
        check_cuda(cudaMalloc(&d_X_, batch * in_dim_ * sizeof(float)), "alloc X");
        check_cuda(cudaMalloc(&d_Y_, batch * out_dim_ * sizeof(float)), "alloc Y");
        check_cuda(cudaMalloc(&d_logits_, batch * out_dim_ * sizeof(float)), "alloc logits");
        check_cuda(cudaMalloc(&d_probs_, batch * out_dim_ * sizeof(float)), "alloc probs");
        check_cuda(cudaMalloc(&d_loss_, batch * sizeof(float)), "alloc loss");
        check_cuda(cudaMallocHost(&h_loss_, batch * sizeof(float)), "alloc host loss");
        check_cuda(cudaMallocHost(&h_probs_, batch * out_dim_ * sizeof(float)), "alloc host probs");
    }

public:
    CublasTrainer() = default;

    bool initialize(int in_dim, int out_dim, const float* weights, const float* bias) {
        in_dim_ = in_dim;
        out_dim_ = out_dim;

        // Init CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) return false;

        check_cuda(cudaSetDevice(0), "set device");
        check_cuda(cudaStreamCreate(&stream_), "create stream");

        cublasStatus_t st = cublasCreate(&handle_);
        if (st != CUBLAS_STATUS_SUCCESS) return false;
        cublasSetStream(handle_, stream_);

        // Allocate model params
        check_cuda(cudaMalloc(&d_W_, out_dim * in_dim * sizeof(float)), "alloc W");
        check_cuda(cudaMalloc(&d_b_, out_dim * sizeof(float)), "alloc b");
        check_cuda(cudaMalloc(&d_grad_W_, out_dim * in_dim * sizeof(float)), "alloc grad_W");
        check_cuda(cudaMalloc(&d_grad_b_, out_dim * sizeof(float)), "alloc grad_b");

        // Upload initial weights
        check_cuda(cudaMemcpy(d_W_, weights, out_dim * in_dim * sizeof(float), cudaMemcpyHostToDevice), "upload W");
        check_cuda(cudaMemcpy(d_b_, bias, out_dim * sizeof(float), cudaMemcpyHostToDevice), "upload b");

        ready_ = true;
        return true;
    }

    bool available() const { return ready_; }

    // Train one batch ENTIRELY on GPU. Returns loss.
    // summaries: [batch, in_dim] pre-computed on CPU
    // labels: [batch, out_dim] one-hot
    float train_step(const float* summaries, const float* labels, int batch, float lr, float* accuracy) {
        alloc_batch(batch);

        // Upload batch to GPU (async)
        check_cuda(cudaMemcpyAsync(d_X_, summaries, batch * in_dim_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_), "upload X");
        check_cuda(cudaMemcpyAsync(d_Y_, labels, batch * out_dim_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_), "upload Y");

        // Forward: logits = X @ W^T + b
        // cuBLAS: C = alpha * A * B + beta * C
        // We want: logits[batch, 3] = X[batch, 32] @ W^T[32, 3]
        // cuBLAS is column-major, so we compute: logits^T = W @ X^T
        // Which gives us logits in row-major as: logits = X @ W^T
        float alpha = 1.0f, beta = 0.0f;

        // First set logits = bias (broadcast)
        // Copy bias to each row of logits
        for (int i = 0; i < batch; i++) {
            check_cuda(cudaMemcpyAsync(d_logits_ + i * out_dim_, d_b_, out_dim_ * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream_), "copy bias");
        }

        // logits += X @ W^T
        // cublasSgemm: C(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n)
        // But cuBLAS is col-major! So we swap: C^T = B^T @ A^T
        // Want: logits(batch, 3) = X(batch, 32) @ W^T(32, 3)
        // Col-major: logits^T(3, batch) = W(3, 32) @ X^T(32, batch)
        beta = 1.0f;  // Add to existing bias
        check_cublas(cublasSgemm(handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            out_dim_,    // m = rows of output (col-major) = out_dim
            batch,       // n = cols of output = batch
            in_dim_,     // k = inner dim
            &alpha,
            d_W_, out_dim_,     // A = W [out_dim, in_dim], lda = out_dim
            d_X_, in_dim_,      // B = X^T [in_dim, batch], ldb = in_dim
            &beta,
            d_logits_, out_dim_ // C = logits^T [out_dim, batch], ldc = out_dim
        ), "gemm forward");

        // Softmax
        softmax_inplace(d_logits_, batch, out_dim_, stream_);

        // Copy to probs (keep logits for backward)
        check_cuda(cudaMemcpyAsync(d_probs_, d_logits_, batch * out_dim_ * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream_), "copy probs");

        // Compute loss
        cross_entropy_loss(d_probs_, d_Y_, d_loss_, batch, out_dim_, stream_);

        // Backward: grad_logits = probs - targets (for softmax + cross-entropy)
        // d_logits_ now holds probs, subtract targets to get gradient
        // grad = probs - Y, already in d_logits after softmax
        // We need: d_logits_ = d_probs_ - d_Y_
        // Use cublasSaxpy: y = alpha * x + y → d_logits_ = -1 * d_Y_ + d_probs_
        check_cuda(cudaMemcpyAsync(d_logits_, d_probs_, batch * out_dim_ * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream_), "copy for backward");
        float neg_one = -1.0f;
        check_cublas(cublasSaxpy(handle_, batch * out_dim_, &neg_one, d_Y_, 1, d_logits_, 1), "grad subtract");

        // grad_W = grad_logits^T @ X / batch
        // Col-major: grad_W^T(in_dim, out_dim) = X^T(in_dim, batch) @ grad(batch, out_dim)
        float scale = 1.0f / static_cast<float>(batch);
        beta = 0.0f;
        check_cublas(cublasSgemm(handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            out_dim_,    // m
            in_dim_,     // n
            batch,       // k
            &scale,
            d_logits_, out_dim_,  // grad^T [out_dim, batch]
            d_X_, in_dim_,        // X^T [in_dim, batch]
            &beta,
            d_grad_W_, out_dim_   // grad_W^T [out_dim, in_dim]
        ), "gemm backward W");

        // grad_b = mean(grad_logits, dim=0)
        // Sum columns: grad_b = grad^T @ ones / batch
        // Simpler: just sum each row of grad (which is [out_dim, batch] in col-major)
        // Use cublasSgemv: y = alpha * A * x + beta * y
        std::vector<float> ones_host(batch, 1.0f / batch);
        float* d_ones;
        check_cuda(cudaMalloc(&d_ones, batch * sizeof(float)), "alloc ones");
        check_cuda(cudaMemcpyAsync(d_ones, ones_host.data(), batch * sizeof(float),
                   cudaMemcpyHostToDevice, stream_), "upload ones");
        beta = 0.0f;
        check_cublas(cublasSgemv(handle_,
            CUBLAS_OP_N,
            out_dim_, batch,
            &alpha,
            d_logits_, out_dim_,
            d_ones, 1,
            &beta,
            d_grad_b_, 1
        ), "gemv backward b");
        cudaFree(d_ones);

        // SGD update: W -= lr * grad_W, b -= lr * grad_b
        float neg_lr = -lr;
        check_cublas(cublasSaxpy(handle_, out_dim_ * in_dim_, &neg_lr, d_grad_W_, 1, d_W_, 1), "update W");
        check_cublas(cublasSaxpy(handle_, out_dim_, &neg_lr, d_grad_b_, 1, d_b_, 1), "update b");

        // Download loss and probs for metrics
        check_cuda(cudaMemcpyAsync(h_loss_, d_loss_, batch * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_), "download loss");
        check_cuda(cudaMemcpyAsync(h_probs_, d_probs_, batch * out_dim_ * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_), "download probs");

        cudaStreamSynchronize(stream_);

        // Compute metrics on CPU (tiny)
        float total_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < batch; i++) {
            total_loss += h_loss_[i];
            int pred = 0, truth = 0;
            for (int c = 1; c < out_dim_; c++) {
                if (h_probs_[i * out_dim_ + c] > h_probs_[i * out_dim_ + pred]) pred = c;
                if (labels[i * out_dim_ + c] > labels[i * out_dim_ + truth]) truth = c;
            }
            if (pred == truth) correct++;
        }

        *accuracy = static_cast<float>(correct) / static_cast<float>(batch);
        return total_loss / static_cast<float>(batch);
    }

    // Download current weights back to host
    void download_weights(float* weights, float* bias) {
        cudaMemcpy(weights, d_W_, out_dim_ * in_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias, d_b_, out_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

    ~CublasTrainer() {
        if (handle_) cublasDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
        cudaFree(d_W_); cudaFree(d_b_);
        cudaFree(d_X_); cudaFree(d_Y_);
        cudaFree(d_logits_); cudaFree(d_probs_);
        cudaFree(d_loss_);
        cudaFree(d_grad_W_); cudaFree(d_grad_b_);
        if (h_loss_) cudaFreeHost(h_loss_);
        if (h_probs_) cudaFreeHost(h_probs_);
    }
};

}  // namespace from::cuda

#endif  // FROM_CUDA
