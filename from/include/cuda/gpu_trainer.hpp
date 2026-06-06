#pragma once

#ifdef FROM_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/sequence_model.hpp"

namespace from::cuda {

#define GPU_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); exit(1); } } while(0)

// CUDA kernels declared here, defined in gpu_trainer_kernels.cu
void gpu_relu_forward(float* x, char* mask, int n, cudaStream_t s);
void gpu_relu_backward(float* grad, const char* mask, int n, cudaStream_t s);
void gpu_add_bias(float* out, const float* bias, int rows, int cols, cudaStream_t s);
void gpu_softmax(float* x, int rows, int cols, cudaStream_t s);
void gpu_cross_entropy_grad(float* grad, const uint8_t* labels, int batch, int classes, cudaStream_t s);
void gpu_cross_entropy_grad_weighted(float* grad, const uint8_t* labels, const float* class_weights,
                                      int batch, int classes, cudaStream_t s);
void gpu_adam_update(float* w, const float* g, float* m, float* v,
                    float lr, float beta1, float beta2, float eps, float bc1, float bc2,
                    int n, cudaStream_t s);
void gpu_grad_clip(float* data, int n, float max_norm, float* norm_out, cudaStream_t s);
void gpu_compute_metrics(const float* probs, const uint8_t* labels, float* loss_out, int* correct_out,
                         int batch, int classes, cudaStream_t s);
void gpu_gather_batch(const float* all_summaries, const uint8_t* all_labels,
                      const uint32_t* indices, float* out_input, uint8_t* out_labels,
                      int batch, int dim, cudaStream_t s);
void gpu_bias_grad(const float* grad, float* gb, int batch, int dim, cudaStream_t s);
void gpu_rand_indices(uint32_t* indices, int batch, uint32_t n_samples, uint32_t seed, cudaStream_t s);
void gpu_fused_train(
    float* w1, float* b1, float* w2, float* b2, float* w3, float* b3,
    float* mw1, float* vw1, float* mb1, float* vb1,
    float* mw2, float* vw2, float* mb2, float* vb2,
    float* mw3, float* vw3, float* mb3, float* vb3,
    const float* all_data, const uint8_t* all_labels,
    float* h1_buf, float* h2_buf, float* logits_buf,
    float* grad3_buf, float* grad2_buf, float* grad1_buf,
    float* gw1, float* gb1, float* gw2, float* gb2, float* gw3, float* gb3,
    uint32_t n_samples, uint32_t batch, uint32_t seed,
    float lr, float bc1, float bc2,
    float* loss_out, int* correct_out, cudaStream_t stream);

// Full 3-layer MLP trainer on GPU: 176 → 256 → 128 → 3
// Entire forward+backward+adam on GPU. Only upload batch, download loss.
class GpuTrainer {
    static constexpr int IN = static_cast<int>(SEQ_SUMMARY_DIM);  // 243
    static constexpr int H1 = 256;
    static constexpr int H2 = 128;
    static constexpr int OUT = 3;

    cublasHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Model weights on GPU
    float *d_w1, *d_b1;  // [H1, IN], [H1]
    float *d_w2, *d_b2;  // [H2, H1], [H2]
    float *d_w3, *d_b3;  // [OUT, H2], [OUT]

    // Adam state on GPU
    float *d_mw1, *d_vw1, *d_mb1, *d_vb1;
    float *d_mw2, *d_vw2, *d_mb2, *d_vb2;
    float *d_mw3, *d_vw3, *d_mb3, *d_vb3;

    // Gradients on GPU
    float *d_gw1, *d_gb1;
    float *d_gw2, *d_gb2;
    float *d_gw3, *d_gb3;

    // Activations (batch workspace)
    float *d_input;   // [batch, IN]
    float *d_h1;      // [batch, H1]
    float *d_h2;      // [batch, H2]
    float *d_logits;  // [batch, OUT]
    char  *d_mask1;   // [batch, H1] relu mask
    char  *d_mask2;   // [batch, H2] relu mask

    // Gradient workspace
    float *d_grad3;   // [batch, OUT]
    float *d_grad2;   // [batch, H2]
    float *d_grad1;   // [batch, H1]

    // Labels on GPU
    uint8_t *d_labels; // [batch]

    // Host pinned buffers for metrics
    float *h_loss;
    int   *h_correct;
    float *d_loss_scalar;
    int   *d_correct_scalar;

    // Training data on GPU (all of it!)
    float   *d_all_summaries = nullptr;  // [N, 176]
    uint8_t *d_all_labels = nullptr;     // [N]
    uint32_t *d_indices = nullptr;       // [batch] random indices
    float   *d_class_weights = nullptr;  // [3] class weights for weighted CE

    int batch_cap_ = 0;
    int adam_t_ = 0;
    size_t n_samples_ = 0;
    bool ready_ = false;
    bool use_weighted_ce_ = false;

    void alloc(float** p, size_t n) { GPU_CHECK(cudaMalloc(p, n * sizeof(float))); GPU_CHECK(cudaMemset(*p, 0, n * sizeof(float))); }

public:
    float last_grad_norm = 0.0f;

    GpuTrainer() = default;

    bool initialize(int batch_size, const std::vector<float>& summaries, const std::vector<uint8_t>& labels,
                    const float* w1, const float* b1, const float* w2, const float* b2,
                    const float* w3, const float* b3) {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (dev_count == 0) return false;

        GPU_CHECK(cudaSetDevice(0));
        GPU_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream_));

        n_samples_ = labels.size();
        batch_cap_ = batch_size;

        // Allocate weights
        alloc(&d_w1, H1 * IN); alloc(&d_b1, H1);
        alloc(&d_w2, H2 * H1); alloc(&d_b2, H2);
        alloc(&d_w3, OUT * H2); alloc(&d_b3, OUT);

        // Upload weights (stored row-major on CPU: W[out, in])
        // cuBLAS wants col-major, so we upload as-is and use appropriate ops
        GPU_CHECK(cudaMemcpy(d_w1, w1, H1 * IN * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_b1, b1, H1 * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_w2, w2, H2 * H1 * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_b2, b2, H2 * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_w3, w3, OUT * H2 * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_b3, b3, OUT * sizeof(float), cudaMemcpyHostToDevice));

        // Adam state
        alloc(&d_mw1, H1*IN); alloc(&d_vw1, H1*IN); alloc(&d_mb1, H1); alloc(&d_vb1, H1);
        alloc(&d_mw2, H2*H1); alloc(&d_vw2, H2*H1); alloc(&d_mb2, H2); alloc(&d_vb2, H2);
        alloc(&d_mw3, OUT*H2); alloc(&d_vw3, OUT*H2); alloc(&d_mb3, OUT); alloc(&d_vb3, OUT);

        // Gradients
        alloc(&d_gw1, H1*IN); alloc(&d_gb1, H1);
        alloc(&d_gw2, H2*H1); alloc(&d_gb2, H2);
        alloc(&d_gw3, OUT*H2); alloc(&d_gb3, OUT);

        // Batch workspace
        alloc(&d_input, batch_size * IN);
        alloc(&d_h1, batch_size * H1);
        alloc(&d_h2, batch_size * H2);
        alloc(&d_logits, batch_size * OUT);
        GPU_CHECK(cudaMalloc(&d_mask1, batch_size * H1)); GPU_CHECK(cudaMemset(d_mask1, 0, batch_size * H1));
        GPU_CHECK(cudaMalloc(&d_mask2, batch_size * H2)); GPU_CHECK(cudaMemset(d_mask2, 0, batch_size * H2));
        alloc(&d_grad3, batch_size * OUT);
        alloc(&d_grad2, batch_size * H2);
        alloc(&d_grad1, batch_size * H1);
        GPU_CHECK(cudaMalloc(&d_labels, batch_size));
        GPU_CHECK(cudaMalloc(&d_indices, batch_size * sizeof(uint32_t)));

        // Metrics
        GPU_CHECK(cudaMallocHost(&h_loss, sizeof(float)));
        GPU_CHECK(cudaMallocHost(&h_correct, sizeof(int)));
        GPU_CHECK(cudaMalloc(&d_loss_scalar, sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_correct_scalar, sizeof(int)));

        // Upload training data to GPU — cap to what VRAM can hold
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t data_bytes = n_samples_ * IN * sizeof(float) + n_samples_;
        size_t safe_limit = free_mem > 512*1024*1024 ? free_mem - 512*1024*1024 : 0;
        if (data_bytes > safe_limit && safe_limit > 0) {
            n_samples_ = safe_limit / (IN * sizeof(float) + 1);
            printf("\033[33m[GPU] VRAM limit: capping to %zu samples (%.0fMB free)\033[0m\n",
                   n_samples_, (float)free_mem / 1048576.0f);
        }
        GPU_CHECK(cudaMalloc(&d_all_summaries, n_samples_ * IN * sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_all_labels, n_samples_));
        GPU_CHECK(cudaMemcpy(d_all_summaries, summaries.data(), n_samples_ * IN * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_all_labels, labels.data(), n_samples_, cudaMemcpyHostToDevice));

        ready_ = true;

        // Print GPU info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t data_mb = n_samples_ * IN * 4 / 1048576;
        printf("\033[35m[GPU] %s | %zuMB VRAM used | %zu samples on device\033[0m\n",
               prop.name, data_mb, n_samples_);
        return true;
    }

    bool available() const { return ready_; }

    void set_class_weights(const float* weights) {
        if (!ready_) return;
        GPU_CHECK(cudaMalloc(&d_class_weights, OUT * sizeof(float)));
        GPU_CHECK(cudaMemcpy(d_class_weights, weights, OUT * sizeof(float), cudaMemcpyHostToDevice));
        use_weighted_ce_ = true;
    }

    // Fully GPU-resident step: indices generated on device. Zero host interaction.
    void train_step_gpu_only(int batch, float lr, uint32_t step_seed) {
        ++adam_t_;
        float inv_batch = 1.0f / static_cast<float>(batch);
        float bc1 = 1.0f - powf(0.9f, static_cast<float>(adam_t_));
        float bc2 = 1.0f - powf(0.999f, static_cast<float>(adam_t_));

        gpu_rand_indices(d_indices, batch, static_cast<uint32_t>(n_samples_), step_seed, stream_);
        gpu_gather_batch(d_all_summaries, d_all_labels, d_indices, d_input, d_labels, batch, IN, stream_);

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H1, batch, IN, &alpha, d_w1, IN, d_input, IN, &beta, d_h1, H1));
        gpu_add_bias(d_h1, d_b1, batch, H1, stream_);
        gpu_relu_forward(d_h1, d_mask1, batch * H1, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H2, batch, H1, &alpha, d_w2, H1, d_h1, H1, &beta, d_h2, H2));
        gpu_add_bias(d_h2, d_b2, batch, H2, stream_);
        gpu_relu_forward(d_h2, d_mask2, batch * H2, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            OUT, batch, H2, &alpha, d_w3, H2, d_h2, H2, &beta, d_logits, OUT));
        gpu_add_bias(d_logits, d_b3, batch, OUT, stream_);

        gpu_softmax(d_logits, batch, OUT, stream_);
        GPU_CHECK(cudaMemcpyAsync(d_grad3, d_logits, batch * OUT * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        if (use_weighted_ce_) {
            gpu_cross_entropy_grad_weighted(d_grad3, d_labels, d_class_weights, batch, OUT, stream_);
        } else {
            gpu_cross_entropy_grad(d_grad3, d_labels, batch, OUT, stream_);
        }
        CUBLAS_CHECK(cublasSscal(handle_, batch * OUT, &inv_batch, d_grad3, 1));

        beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H2, OUT, batch, &alpha, d_h2, H2, d_grad3, OUT, &beta, d_gw3, H2));
        gpu_bias_grad(d_grad3, d_gb3, batch, OUT, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H2, batch, OUT, &alpha, d_w3, H2, d_grad3, OUT, &beta, d_grad2, H2));
        gpu_relu_backward(d_grad2, d_mask2, batch * H2, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H1, H2, batch, &alpha, d_h1, H1, d_grad2, H2, &beta, d_gw2, H1));
        gpu_bias_grad(d_grad2, d_gb2, batch, H2, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H1, batch, H2, &alpha, d_w2, H1, d_grad2, H2, &beta, d_grad1, H1));
        gpu_relu_backward(d_grad1, d_mask1, batch * H1, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            IN, H1, batch, &alpha, d_input, IN, d_grad1, H1, &beta, d_gw1, IN));
        gpu_bias_grad(d_grad1, d_gb1, batch, H1, stream_);

        gpu_adam_update(d_w1, d_gw1, d_mw1, d_vw1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1*IN, stream_);
        gpu_adam_update(d_b1, d_gb1, d_mb1, d_vb1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1, stream_);
        gpu_adam_update(d_w2, d_gw2, d_mw2, d_vw2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2*H1, stream_);
        gpu_adam_update(d_b2, d_gb2, d_mb2, d_vb2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2, stream_);
        gpu_adam_update(d_w3, d_gw3, d_mw3, d_vw3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT*H2, stream_);
        gpu_adam_update(d_b3, d_gb3, d_mb3, d_vb3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT, stream_);
    }

    // Train one step, no sync (fire-and-forget). Call sync_metrics() when you need loss/acc.
    void train_step_async(const uint32_t* host_indices, int batch, float lr) {
        float inv_batch = 1.0f / static_cast<float>(batch);
        ++adam_t_;
        float bc1 = 1.0f - powf(0.9f, static_cast<float>(adam_t_));
        float bc2 = 1.0f - powf(0.999f, static_cast<float>(adam_t_));

        GPU_CHECK(cudaMemcpyAsync(d_indices, host_indices, batch * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_));
        gpu_gather_batch(d_all_summaries, d_all_labels, d_indices, d_input, d_labels, batch, IN, stream_);

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H1, batch, IN, &alpha, d_w1, IN, d_input, IN, &beta, d_h1, H1));
        gpu_add_bias(d_h1, d_b1, batch, H1, stream_);
        gpu_relu_forward(d_h1, d_mask1, batch * H1, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H2, batch, H1, &alpha, d_w2, H1, d_h1, H1, &beta, d_h2, H2));
        gpu_add_bias(d_h2, d_b2, batch, H2, stream_);
        gpu_relu_forward(d_h2, d_mask2, batch * H2, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            OUT, batch, H2, &alpha, d_w3, H2, d_h2, H2, &beta, d_logits, OUT));
        gpu_add_bias(d_logits, d_b3, batch, OUT, stream_);

        gpu_softmax(d_logits, batch, OUT, stream_);
        GPU_CHECK(cudaMemcpyAsync(d_grad3, d_logits, batch * OUT * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        if (use_weighted_ce_) {
            gpu_cross_entropy_grad_weighted(d_grad3, d_labels, d_class_weights, batch, OUT, stream_);
        } else {
            gpu_cross_entropy_grad(d_grad3, d_labels, batch, OUT, stream_);
        }
        CUBLAS_CHECK(cublasSscal(handle_, batch * OUT, &inv_batch, d_grad3, 1));

        beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H2, OUT, batch, &alpha, d_h2, H2, d_grad3, OUT, &beta, d_gw3, H2));
        gpu_bias_grad(d_grad3, d_gb3, batch, OUT, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H2, batch, OUT, &alpha, d_w3, H2, d_grad3, OUT, &beta, d_grad2, H2));
        gpu_relu_backward(d_grad2, d_mask2, batch * H2, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H1, H2, batch, &alpha, d_h1, H1, d_grad2, H2, &beta, d_gw2, H1));
        gpu_bias_grad(d_grad2, d_gb2, batch, H2, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H1, batch, H2, &alpha, d_w2, H1, d_grad2, H2, &beta, d_grad1, H1));
        gpu_relu_backward(d_grad1, d_mask1, batch * H1, stream_);

        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            IN, H1, batch, &alpha, d_input, IN, d_grad1, H1, &beta, d_gw1, IN));
        gpu_bias_grad(d_grad1, d_gb1, batch, H1, stream_);

        gpu_adam_update(d_w1, d_gw1, d_mw1, d_vw1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1*IN, stream_);
        gpu_adam_update(d_b1, d_gb1, d_mb1, d_vb1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1, stream_);
        gpu_adam_update(d_w2, d_gw2, d_mw2, d_vw2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2*H1, stream_);
        gpu_adam_update(d_b2, d_gb2, d_mb2, d_vb2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2, stream_);
        gpu_adam_update(d_w3, d_gw3, d_mw3, d_vw3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT*H2, stream_);
        gpu_adam_update(d_b3, d_gb3, d_mb3, d_vb3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT, stream_);
    }

    // Fused kernel: entire forward+backward+adam in 1 kernel + 6 adam kernels.
    // Minimizes launch overhead. Best for small batch (32-128).
    float train_fused(int batch, float lr, float* accuracy) {
        ++adam_t_;
        float bc1 = 1.0f - powf(0.9f, static_cast<float>(adam_t_));
        float bc2 = 1.0f - powf(0.999f, static_cast<float>(adam_t_));
        uint32_t seed = static_cast<uint32_t>(adam_t_) * 2654435761u + 42u;

        gpu_fused_train(d_w1, d_b1, d_w2, d_b2, d_w3, d_b3,
                       d_mw1, d_vw1, d_mb1, d_vb1,
                       d_mw2, d_vw2, d_mb2, d_vb2,
                       d_mw3, d_vw3, d_mb3, d_vb3,
                       d_all_summaries, d_all_labels,
                       d_h1, d_h2, d_logits,
                       d_grad3, d_grad2, d_grad1,
                       d_gw1, d_gb1, d_gw2, d_gb2, d_gw3, d_gb3,
                       static_cast<uint32_t>(n_samples_), batch, seed,
                       lr, bc1, bc2,
                       d_loss_scalar, d_correct_scalar, stream_);

        GPU_CHECK(cudaMemcpyAsync(h_loss, d_loss_scalar, sizeof(float), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaMemcpyAsync(h_correct, d_correct_scalar, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaStreamSynchronize(stream_));
        *accuracy = static_cast<float>(*h_correct) / static_cast<float>(batch);
        return *h_loss;
    }

    // Fused kernel without sync - fire and forget
    void train_fused_async(int batch, float lr) {
        ++adam_t_;
        float bc1 = 1.0f - powf(0.9f, static_cast<float>(adam_t_));
        float bc2 = 1.0f - powf(0.999f, static_cast<float>(adam_t_));
        uint32_t seed = static_cast<uint32_t>(adam_t_) * 2654435761u + 42u;

        gpu_fused_train(d_w1, d_b1, d_w2, d_b2, d_w3, d_b3,
                       d_mw1, d_vw1, d_mb1, d_vb1,
                       d_mw2, d_vw2, d_mb2, d_vb2,
                       d_mw3, d_vw3, d_mb3, d_vb3,
                       d_all_summaries, d_all_labels,
                       d_h1, d_h2, d_logits,
                       d_grad3, d_grad2, d_grad1,
                       d_gw1, d_gb1, d_gw2, d_gb2, d_gw3, d_gb3,
                       static_cast<uint32_t>(n_samples_), batch, seed,
                       lr, bc1, bc2,
                       d_loss_scalar, d_correct_scalar, stream_);
    }

    // Train N steps in a burst — queue all into stream, sync once at end.
    // Returns metrics from the LAST step only.
    float train_burst(int n_steps, int batch, float lr, float* accuracy) {
        for (int s = 0; s < n_steps - 1; ++s) {
            uint32_t seed = static_cast<uint32_t>(adam_t_ + 1) * 2654435761u + 42u;
            train_step_gpu_only(batch, lr, seed);
        }
        // Last step with metrics
        uint32_t seed = static_cast<uint32_t>(adam_t_ + 1) * 2654435761u + 42u;
        ++adam_t_;
        float inv_batch = 1.0f / static_cast<float>(batch);
        float bc1 = 1.0f - powf(0.9f, static_cast<float>(adam_t_));
        float bc2 = 1.0f - powf(0.999f, static_cast<float>(adam_t_));

        gpu_rand_indices(d_indices, batch, static_cast<uint32_t>(n_samples_), seed, stream_);
        gpu_gather_batch(d_all_summaries, d_all_labels, d_indices, d_input, d_labels, batch, IN, stream_);

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H1, batch, IN, &alpha, d_w1, IN, d_input, IN, &beta, d_h1, H1));
        gpu_add_bias(d_h1, d_b1, batch, H1, stream_);
        gpu_relu_forward(d_h1, d_mask1, batch * H1, stream_);
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H2, batch, H1, &alpha, d_w2, H1, d_h1, H1, &beta, d_h2, H2));
        gpu_add_bias(d_h2, d_b2, batch, H2, stream_);
        gpu_relu_forward(d_h2, d_mask2, batch * H2, stream_);
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            OUT, batch, H2, &alpha, d_w3, H2, d_h2, H2, &beta, d_logits, OUT));
        gpu_add_bias(d_logits, d_b3, batch, OUT, stream_);

        gpu_softmax(d_logits, batch, OUT, stream_);
        gpu_compute_metrics(d_logits, d_labels, d_loss_scalar, d_correct_scalar, batch, OUT, stream_);

        GPU_CHECK(cudaMemcpyAsync(d_grad3, d_logits, batch * OUT * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        if (use_weighted_ce_) {
            gpu_cross_entropy_grad_weighted(d_grad3, d_labels, d_class_weights, batch, OUT, stream_);
        } else {
            gpu_cross_entropy_grad(d_grad3, d_labels, batch, OUT, stream_);
        }
        CUBLAS_CHECK(cublasSscal(handle_, batch * OUT, &inv_batch, d_grad3, 1));

        beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H2, OUT, batch, &alpha, d_h2, H2, d_grad3, OUT, &beta, d_gw3, H2));
        gpu_bias_grad(d_grad3, d_gb3, batch, OUT, stream_);
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H2, batch, OUT, &alpha, d_w3, H2, d_grad3, OUT, &beta, d_grad2, H2));
        gpu_relu_backward(d_grad2, d_mask2, batch * H2, stream_);
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H1, H2, batch, &alpha, d_h1, H1, d_grad2, H2, &beta, d_gw2, H1));
        gpu_bias_grad(d_grad2, d_gb2, batch, H2, stream_);
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H1, batch, H2, &alpha, d_w2, H1, d_grad2, H2, &beta, d_grad1, H1));
        gpu_relu_backward(d_grad1, d_mask1, batch * H1, stream_);
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            IN, H1, batch, &alpha, d_input, IN, d_grad1, H1, &beta, d_gw1, IN));
        gpu_bias_grad(d_grad1, d_gb1, batch, H1, stream_);

        gpu_adam_update(d_w1, d_gw1, d_mw1, d_vw1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1*IN, stream_);
        gpu_adam_update(d_b1, d_gb1, d_mb1, d_vb1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1, stream_);
        gpu_adam_update(d_w2, d_gw2, d_mw2, d_vw2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2*H1, stream_);
        gpu_adam_update(d_b2, d_gb2, d_mb2, d_vb2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2, stream_);
        gpu_adam_update(d_w3, d_gw3, d_mw3, d_vw3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT*H2, stream_);
        gpu_adam_update(d_b3, d_gb3, d_mb3, d_vb3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT, stream_);

        GPU_CHECK(cudaMemcpyAsync(h_loss, d_loss_scalar, sizeof(float), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaMemcpyAsync(h_correct, d_correct_scalar, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaStreamSynchronize(stream_));
        *accuracy = static_cast<float>(*h_correct) / static_cast<float>(batch);
        return *h_loss;
    }

    // Sync and get metrics from the last step that computed them
    float sync_metrics(int batch, float* accuracy) {
        gpu_compute_metrics(d_logits, d_labels, d_loss_scalar, d_correct_scalar, batch, OUT, stream_);
        GPU_CHECK(cudaMemcpyAsync(h_loss, d_loss_scalar, sizeof(float), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaMemcpyAsync(h_correct, d_correct_scalar, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaStreamSynchronize(stream_));
        *accuracy = static_cast<float>(*h_correct) / static_cast<float>(batch);
        return *h_loss;
    }

    // Train one batch entirely on GPU. Returns loss.
    float train_step(const uint32_t* host_indices, int batch, float lr, float* accuracy) {
        float inv_batch = 1.0f / static_cast<float>(batch);
        ++adam_t_;
        float bc1 = 1.0f - powf(0.9f, static_cast<float>(adam_t_));
        float bc2 = 1.0f - powf(0.999f, static_cast<float>(adam_t_));

        // Upload batch indices and gather input+labels on GPU
        GPU_CHECK(cudaMemcpyAsync(d_indices, host_indices, batch * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_));

        // Gather: d_input[i] = d_all_summaries[indices[i]], d_labels[i] = d_all_labels[indices[i]]
        // We'll do this with a simple kernel (declared in gpu_trainer_kernels.cu)
        gpu_gather_batch(d_all_summaries, d_all_labels, d_indices, d_input, d_labels, batch, IN, stream_);

        // ===== FORWARD =====
        // All matrices stored row-major. For row-major C[M,N] = A[M,K] @ B[K,N]:
        // cuBLAS: cublasSgemm(N, N, N, M, K, alpha, B, N, A, K, beta, C, N)
        // For C[M,N] = A[M,K] @ B^T where B is [N,K] row-major:
        // cuBLAS: cublasSgemm(T, N, N, M, K, alpha, B, K, A, K, beta, C, N)
        float alpha = 1.0f, beta = 0.0f;

        // h1[batch, H1] = input[batch, IN] @ W1[H1, IN]^T
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H1, batch, IN, &alpha,
            d_w1, IN, d_input, IN, &beta, d_h1, H1));
        gpu_add_bias(d_h1, d_b1, batch, H1, stream_);
        gpu_relu_forward(d_h1, d_mask1, batch * H1, stream_);

        // h2[batch, H2] = h1[batch, H1] @ W2[H2, H1]^T
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            H2, batch, H1, &alpha,
            d_w2, H1, d_h1, H1, &beta, d_h2, H2));
        gpu_add_bias(d_h2, d_b2, batch, H2, stream_);
        gpu_relu_forward(d_h2, d_mask2, batch * H2, stream_);

        // logits[batch, OUT] = h2[batch, H2] @ W3[OUT, H2]^T
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            OUT, batch, H2, &alpha,
            d_w3, H2, d_h2, H2, &beta, d_logits, OUT));
        gpu_add_bias(d_logits, d_b3, batch, OUT, stream_);

        // Softmax + compute grad = (softmax - one_hot) / batch
        gpu_softmax(d_logits, batch, OUT, stream_);
        // d_logits now holds probs

        // Compute loss + accuracy on GPU
        gpu_compute_metrics(d_logits, d_labels, d_loss_scalar, d_correct_scalar, batch, OUT, stream_);

        // grad3 = weighted (probs - one_hot) / batch → stored in d_grad3
        GPU_CHECK(cudaMemcpyAsync(d_grad3, d_logits, batch * OUT * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        if (use_weighted_ce_) {
            gpu_cross_entropy_grad_weighted(d_grad3, d_labels, d_class_weights, batch, OUT, stream_);
        } else {
            gpu_cross_entropy_grad(d_grad3, d_labels, batch, OUT, stream_);
        }
        // Scale by 1/batch
        CUBLAS_CHECK(cublasSscal(handle_, batch * OUT, &inv_batch, d_grad3, 1));

        // ===== BACKWARD =====
        // Row-major trick: C[M,N] = A[M,K] @ B[K,N] →
        //   cublasSgemm(N, N, N, M, K, α, B, N, A, K, β, C, N)
        // For C[M,N] = A^T[M,K] @ B[K,N] where A stored [K,M]:
        //   cublasSgemm(N, T, N, M, K, α, B, N, A, M, β, C, N)
        //   (swap A/B roles and use T on the transposed one)
        beta = 0.0f;

        // gw3[OUT,H2] = grad3^T[OUT,batch] @ h2[batch,H2]
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H2, OUT, batch, &alpha,
            d_h2, H2, d_grad3, OUT, &beta, d_gw3, H2));
        gpu_bias_grad(d_grad3, d_gb3, batch, OUT, stream_);

        // grad2[batch,H2] = grad3[batch,OUT] @ W3[OUT,H2]
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H2, batch, OUT, &alpha,
            d_w3, H2, d_grad3, OUT, &beta, d_grad2, H2));
        gpu_relu_backward(d_grad2, d_mask2, batch * H2, stream_);

        // gw2[H2,H1] = grad2^T[H2,batch] @ h1[batch,H1]
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            H1, H2, batch, &alpha,
            d_h1, H1, d_grad2, H2, &beta, d_gw2, H1));
        gpu_bias_grad(d_grad2, d_gb2, batch, H2, stream_);

        // grad1[batch,H1] = grad2[batch,H2] @ W2[H2,H1]
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            H1, batch, H2, &alpha,
            d_w2, H1, d_grad2, H2, &beta, d_grad1, H1));
        gpu_relu_backward(d_grad1, d_mask1, batch * H1, stream_);

        // gw1[H1,IN] = grad1^T[H1,batch] @ input[batch,IN]
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            IN, H1, batch, &alpha,
            d_input, IN, d_grad1, H1, &beta, d_gw1, IN));
        gpu_bias_grad(d_grad1, d_gb1, batch, H1, stream_);

        // ===== ADAM UPDATE =====
        gpu_adam_update(d_w1, d_gw1, d_mw1, d_vw1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1*IN, stream_);
        gpu_adam_update(d_b1, d_gb1, d_mb1, d_vb1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H1, stream_);
        gpu_adam_update(d_w2, d_gw2, d_mw2, d_vw2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2*H1, stream_);
        gpu_adam_update(d_b2, d_gb2, d_mb2, d_vb2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, H2, stream_);
        gpu_adam_update(d_w3, d_gw3, d_mw3, d_vw3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT*H2, stream_);
        gpu_adam_update(d_b3, d_gb3, d_mb3, d_vb3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, OUT, stream_);

        // Download metrics
        GPU_CHECK(cudaMemcpyAsync(h_loss, d_loss_scalar, sizeof(float), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaMemcpyAsync(h_correct, d_correct_scalar, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaStreamSynchronize(stream_));

        *accuracy = static_cast<float>(*h_correct) / static_cast<float>(batch);
        return *h_loss;
    }

    // Download weights back to CPU model
    void download_weights(float* w1, float* b1, float* w2, float* b2, float* w3, float* b3) {
        GPU_CHECK(cudaMemcpy(w1, d_w1, H1*IN*sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(b1, d_b1, H1*sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(w2, d_w2, H2*H1*sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(b2, d_b2, H2*sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(w3, d_w3, OUT*H2*sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(b3, d_b3, OUT*sizeof(float), cudaMemcpyDeviceToHost));
    }

    ~GpuTrainer() {
        if (!ready_) return;
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
        cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_w3); cudaFree(d_b3);
        cudaFree(d_mw1); cudaFree(d_vw1); cudaFree(d_mb1); cudaFree(d_vb1);
        cudaFree(d_mw2); cudaFree(d_vw2); cudaFree(d_mb2); cudaFree(d_vb2);
        cudaFree(d_mw3); cudaFree(d_vw3); cudaFree(d_mb3); cudaFree(d_vb3);
        cudaFree(d_gw1); cudaFree(d_gb1); cudaFree(d_gw2); cudaFree(d_gb2); cudaFree(d_gw3); cudaFree(d_gb3);
        cudaFree(d_input); cudaFree(d_h1); cudaFree(d_h2); cudaFree(d_logits);
        cudaFree(d_mask1); cudaFree(d_mask2);
        cudaFree(d_grad3); cudaFree(d_grad2); cudaFree(d_grad1);
        cudaFree(d_labels); cudaFree(d_indices);
        cudaFree(d_all_summaries); cudaFree(d_all_labels);
        if (d_class_weights) cudaFree(d_class_weights);
        cudaFree(d_loss_scalar); cudaFree(d_correct_scalar);
        cudaFreeHost(h_loss); cudaFreeHost(h_correct);
    }
};

}  // namespace from::cuda

#endif  // FROM_CUDA
