#pragma once

// ============================================================================
// deep_mlp_trainer.hpp — wfdeep (Medallion-Lite P2)
//
// N-layer GPU MLP that consumes the RAW per-tick window [window x 27] instead of
// the 243-dim summary. The per-tick FEATURE STREAM lives resident on the GPU
// once ([n_ticks x 27]); each sample's window is GATHERED on-the-fly from a
// per-sample ENTRY TICK INDEX (entry_off) via gpu_gather_window.
//
// GEMM/back-prop math is the N-layer generalization of GpuTrainer (3-layer):
//   forward  layer l:  act[l+1][batch,out] = act[l][batch,in] @ W[l][out,in]^T
//   wgrad    layer l:  gW[l][out,in]       = grad[l+1]^T[out,batch] @ act[l][batch,in]
//   igrad    layer l:  grad[l][batch,in]   = grad[l+1][batch,out]   @ W[l][out,in]
// ReLU on every hidden layer, softmax on the output. Adam + global-L2 grad clip,
// all reused verbatim from gpu_trainer_kernels.cu.
//
// Class convention: 0=UP / 1=NEUTRAL / 2=DOWN.
// ============================================================================

#ifdef FROM_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"
#include "cuda/gpu_trainer.hpp"  // GPU_CHECK / CUBLAS_CHECK + reused free kernels

namespace from::cuda {

struct DeepMlpConfig {
    int window = 256;
    int feat = FROM_MAX_FEATURES;       // 27
    std::vector<int> hidden = {2048, 1024, 512};
    int out = 3;
    int batch_cap = 8192;
    float grad_clip = 1.0f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int d_in() const { return window * feat; }   // e.g. 256*27 = 6912
};

class DeepMlpTrainer {
    DeepMlpConfig cfg_;
    cublasHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // dims = [D_in, hidden..., out]; L = number of GEMM layers = hidden.size()+1.
    std::vector<int> dims_;
    int L_ = 0;

    // Per-layer model + Adam + grad buffers (index by layer l in [0, L)).
    std::vector<float*> d_W_, d_b_;
    std::vector<float*> d_mW_, d_vW_, d_mb_, d_vb_;
    std::vector<float*> d_gW_, d_gb_;

    // Activations: d_act_[0] == d_input ([batch, D_in]); d_act_[l] == [batch, dims_[l]]
    // for l in 1..L; d_act_[L] == logits ([batch, out]).
    std::vector<float*> d_act_;
    // ReLU masks for hidden layers: d_mask_[l] valid for l in 1..L-1 ([batch, dims_[l]]).
    std::vector<char*> d_mask_;
    // Backprop deltas: d_grad_[l] valid for l in 1..L ([batch, dims_[l]]).
    std::vector<float*> d_grad_;

    float* d_grad_norm_ = nullptr;

    // Resident per-tick feature stream (shared across folds) + per-fold arrays.
    float*    d_tickfeat_ = nullptr;     // [n_ticks_dev_ * feat]
    uint32_t  n_ticks_dev_ = 0;
    uint32_t* d_entry_off_ = nullptr;    // [n_train_] this fold's window-start offsets
    uint8_t*  d_labels_all_ = nullptr;   // [n_train_] this fold's labels
    uint32_t  n_train_ = 0;

    uint32_t* d_indices_   = nullptr;    // [batch_cap] random sample indices into subset
    uint32_t* d_batch_off_ = nullptr;    // [batch_cap] resolved per-batch window offsets
    uint8_t*  d_batch_lab_ = nullptr;    // [batch_cap] resolved per-batch labels
    float*    d_class_weights_ = nullptr;

    // Pinned metrics scratch.
    float* h_loss_ = nullptr;
    int*   h_correct_ = nullptr;
    float* d_loss_scalar_ = nullptr;
    int*   d_correct_scalar_ = nullptr;

    int  adam_t_ = 0;
    bool ready_ = false;
    bool use_weighted_ce_ = false;

    // Host weight buffers (for He-init upload / fold resets).
    std::vector<std::vector<float>> h_W_, h_b_;

    void alloc_zero(float** p, size_t n) {
        GPU_CHECK(cudaMalloc(p, n * sizeof(float)));
        GPU_CHECK(cudaMemset(*p, 0, n * sizeof(float)));
    }

    // Deterministic xorshift host RNG for He init (seeded per fold).
    static uint32_t xs_next(uint32_t& s) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
    }

    void he_init_host(uint32_t seed) {
        uint32_t s = seed ? seed : 1u;
        for (int l = 0; l < L_; ++l) {
            int out = dims_[l + 1], in = dims_[l];
            // ReLU layers use sqrt(2/in); output layer uses sqrt(1/in).
            float std_dev = (l < L_ - 1) ? std::sqrt(2.0f / (float)in)
                                         : std::sqrt(1.0f / (float)in);
            h_W_[l].assign((size_t)out * in, 0.0f);
            h_b_[l].assign((size_t)out, 0.0f);
            for (size_t i = 0; i < h_W_[l].size(); ++i) {
                // Box-Muller from two uniforms.
                float u1 = (xs_next(s) >> 8) * (1.0f / 16777216.0f);  // (0,1)
                float u2 = (xs_next(s) >> 8) * (1.0f / 16777216.0f);
                if (u1 < 1e-7f) u1 = 1e-7f;
                float r = std::sqrt(-2.0f * std::log(u1));
                float z = r * std::cos(6.2831853f * u2);
                h_W_[l][i] = z * std_dev;
            }
        }
    }

    void upload_weights() {
        for (int l = 0; l < L_; ++l) {
            int out = dims_[l + 1], in = dims_[l];
            GPU_CHECK(cudaMemcpy(d_W_[l], h_W_[l].data(), (size_t)out * in * sizeof(float),
                                 cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_b_[l], h_b_[l].data(), (size_t)out * sizeof(float),
                                 cudaMemcpyHostToDevice));
        }
    }

    void reset_adam() {
        for (int l = 0; l < L_; ++l) {
            size_t wn = (size_t)dims_[l + 1] * dims_[l];
            size_t bn = (size_t)dims_[l + 1];
            cudaMemsetAsync(d_mW_[l], 0, wn * sizeof(float), stream_);
            cudaMemsetAsync(d_vW_[l], 0, wn * sizeof(float), stream_);
            cudaMemsetAsync(d_mb_[l], 0, bn * sizeof(float), stream_);
            cudaMemsetAsync(d_vb_[l], 0, bn * sizeof(float), stream_);
        }
        adam_t_ = 0;
    }

public:
    DeepMlpTrainer() = default;

    bool available() const { return ready_; }

    // Upload the resident tick stream ONCE + set up all layer/Adam/workspace buffers,
    // then bind the first fold's subset. Returns false if no device or VRAM too tight.
    bool initialize(const DeepMlpConfig& cfg,
                    const float* tickfeat, size_t n_ticks,
                    const uint32_t* entry_off_subset, const uint8_t* labels_subset,
                    size_t n_train, uint32_t seed) {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (dev_count == 0) return false;

        cfg_ = cfg;
        GPU_CHECK(cudaSetDevice(0));
        GPU_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream_));

        // dims = [D_in, hidden..., out]
        dims_.clear();
        dims_.push_back(cfg_.d_in());
        for (int h : cfg_.hidden) dims_.push_back(h);
        dims_.push_back(cfg_.out);
        L_ = (int)dims_.size() - 1;

        const int B = cfg_.batch_cap;

        // Per-layer model/Adam/grad buffers.
        d_W_.assign(L_, nullptr);   d_b_.assign(L_, nullptr);
        d_mW_.assign(L_, nullptr);  d_vW_.assign(L_, nullptr);
        d_mb_.assign(L_, nullptr);  d_vb_.assign(L_, nullptr);
        d_gW_.assign(L_, nullptr);  d_gb_.assign(L_, nullptr);
        h_W_.assign(L_, {});        h_b_.assign(L_, {});
        for (int l = 0; l < L_; ++l) {
            size_t wn = (size_t)dims_[l + 1] * dims_[l];
            size_t bn = (size_t)dims_[l + 1];
            alloc_zero(&d_W_[l], wn);  alloc_zero(&d_b_[l], bn);
            alloc_zero(&d_mW_[l], wn); alloc_zero(&d_vW_[l], wn);
            alloc_zero(&d_mb_[l], bn); alloc_zero(&d_vb_[l], bn);
            alloc_zero(&d_gW_[l], wn); alloc_zero(&d_gb_[l], bn);
        }

        // Activations + masks + deltas. d_act_[0] is the gathered input batch.
        d_act_.assign(L_ + 1, nullptr);
        d_mask_.assign(L_ + 1, nullptr);
        d_grad_.assign(L_ + 1, nullptr);
        alloc_zero(&d_act_[0], (size_t)B * dims_[0]);  // [batch, D_in]
        for (int l = 1; l <= L_; ++l) {
            alloc_zero(&d_act_[l], (size_t)B * dims_[l]);
            alloc_zero(&d_grad_[l], (size_t)B * dims_[l]);
            if (l < L_) {  // hidden layers get a relu mask
                GPU_CHECK(cudaMalloc(&d_mask_[l], (size_t)B * dims_[l]));
                GPU_CHECK(cudaMemset(d_mask_[l], 0, (size_t)B * dims_[l]));
            }
        }

        GPU_CHECK(cudaMalloc(&d_grad_norm_, sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_indices_,   (size_t)B * sizeof(uint32_t)));
        GPU_CHECK(cudaMalloc(&d_batch_off_, (size_t)B * sizeof(uint32_t)));
        GPU_CHECK(cudaMalloc(&d_batch_lab_, (size_t)B));

        GPU_CHECK(cudaMallocHost(&h_loss_, sizeof(float)));
        GPU_CHECK(cudaMallocHost(&h_correct_, sizeof(int)));
        GPU_CHECK(cudaMalloc(&d_loss_scalar_, sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_correct_scalar_, sizeof(int)));

        // --- Resident tick stream: cap to VRAM (reserve ~1.5GB headroom) ---
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t reserve = (size_t)1536 * 1024 * 1024;
        size_t safe = free_mem > reserve ? free_mem - reserve : 0;
        size_t want_ticks = n_ticks;
        size_t max_ticks = safe / ((size_t)cfg_.feat * sizeof(float));
        n_ticks_dev_ = (uint32_t)std::min<size_t>(want_ticks, max_ticks);
        if (n_ticks_dev_ == 0) {
            std::fprintf(stderr, "[wfdeep][GPU] not enough VRAM for tick stream\n");
            return false;
        }
        if (n_ticks_dev_ < want_ticks) {
            std::printf("\033[33m[wfdeep][GPU] VRAM cap: tick stream truncated %zu -> %u ticks "
                        "(%.0fMB free)\033[0m\n",
                        want_ticks, n_ticks_dev_, (float)free_mem / 1048576.0f);
        }
        GPU_CHECK(cudaMalloc(&d_tickfeat_, (size_t)n_ticks_dev_ * cfg_.feat * sizeof(float)));
        GPU_CHECK(cudaMemcpy(d_tickfeat_, tickfeat,
                             (size_t)n_ticks_dev_ * cfg_.feat * sizeof(float),
                             cudaMemcpyHostToDevice));

        ready_ = true;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t tf_mb = (size_t)n_ticks_dev_ * cfg_.feat * 4 / 1048576;
        std::printf("\033[35m[wfdeep][GPU] %s | tickstream %zuMB (%u ticks) | D_in=%d | layers=",
                    prop.name, tf_mb, n_ticks_dev_, cfg_.d_in());
        for (size_t i = 0; i < dims_.size(); ++i)
            std::printf("%d%s", dims_[i], i + 1 < dims_.size() ? "x" : "");
        std::printf("\033[0m\n");

        // Bind first fold (uploads He weights + per-fold subset).
        reset_for_fold(entry_off_subset, labels_subset, n_train, seed);
        return true;
    }

    void set_class_weights(const float w[3]) {
        if (!ready_) return;
        if (!d_class_weights_) GPU_CHECK(cudaMalloc(&d_class_weights_, 3 * sizeof(float)));
        GPU_CHECK(cudaMemcpy(d_class_weights_, w, 3 * sizeof(float), cudaMemcpyHostToDevice));
        use_weighted_ce_ = true;
    }

    // Re-bind a new fold's training subset: re-init weights (He, per-fold seed),
    // reset Adam, and re-upload the (cheap) per-fold offset/label arrays. The 10GB
    // resident tick stream is NOT touched.
    void reset_for_fold(const uint32_t* entry_off_subset, const uint8_t* labels_subset,
                        size_t n_train, uint32_t seed) {
        if (!ready_) return;
        he_init_host(seed);
        upload_weights();
        reset_adam();

        if (d_entry_off_)  { cudaFree(d_entry_off_);  d_entry_off_ = nullptr; }
        if (d_labels_all_) { cudaFree(d_labels_all_); d_labels_all_ = nullptr; }
        n_train_ = (uint32_t)n_train;
        if (n_train_ > 0) {
            GPU_CHECK(cudaMalloc(&d_entry_off_, (size_t)n_train_ * sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_labels_all_, (size_t)n_train_));
            GPU_CHECK(cudaMemcpy(d_entry_off_, entry_off_subset,
                                 (size_t)n_train_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_labels_all_, labels_subset, (size_t)n_train_,
                                 cudaMemcpyHostToDevice));
        }
    }

    // --------------------------------------------------------------------
    // Forward over the currently-gathered d_act_[0] (input batch).
    // Leaves logits (softmaxed) in d_act_[L_] and relu masks in d_mask_.
    // --------------------------------------------------------------------
    void forward(int batch, bool train) {
        float alpha = 1.0f, beta = 0.0f;
        for (int l = 0; l < L_; ++l) {
            int in = dims_[l], out = dims_[l + 1];
            // act[l+1][batch,out] = act[l][batch,in] @ W[l][out,in]^T
            CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                out, batch, in, &alpha,
                d_W_[l], in, d_act_[l], in, &beta, d_act_[l + 1], out));
            gpu_add_bias(d_act_[l + 1], d_b_[l], batch, out, stream_);
            if (l < L_ - 1) {
                gpu_relu_forward(d_act_[l + 1], d_mask_[l + 1], batch * out, stream_);
            }
        }
        gpu_softmax(d_act_[L_], batch, cfg_.out, stream_);
        (void)train;
    }

    // --------------------------------------------------------------------
    // One fully-GPU training step: random subset indices -> resolve offsets/labels
    // -> gather windows -> forward -> backward -> clip -> Adam.
    // --------------------------------------------------------------------
    void train_step_gpu_only(int batch, float lr, uint32_t step_seed) {
        if (n_train_ == 0) return;
        ++adam_t_;
        float inv_batch = 1.0f / (float)batch;
        float bc1 = 1.0f - powf(cfg_.beta1, (float)adam_t_);
        float bc2 = 1.0f - powf(cfg_.beta2, (float)adam_t_);

        gpu_rand_indices(d_indices_, batch, n_train_, step_seed, stream_);
        gpu_resolve_offsets(d_entry_off_, d_labels_all_, d_indices_,
                            d_batch_off_, d_batch_lab_, batch, stream_);
        gpu_gather_window(d_tickfeat_, d_batch_off_, d_act_[0],
                          batch, cfg_.window, cfg_.feat, n_ticks_dev_, stream_);

        forward(batch, true);

        // grad[L] = (softmax - onehot)/batch  (weighted if class weights set)
        const int OUT = cfg_.out;
        GPU_CHECK(cudaMemcpyAsync(d_grad_[L_], d_act_[L_], (size_t)batch * OUT * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream_));
        if (use_weighted_ce_) {
            gpu_cross_entropy_grad_weighted(d_grad_[L_], d_batch_lab_, d_class_weights_,
                                            batch, OUT, stream_);
        } else {
            gpu_cross_entropy_grad(d_grad_[L_], d_batch_lab_, batch, OUT, stream_);
        }
        CUBLAS_CHECK(cublasSscal(handle_, batch * OUT, &inv_batch, d_grad_[L_], 1));

        float alpha = 1.0f, beta = 0.0f;
        for (int l = L_ - 1; l >= 0; --l) {
            int in = dims_[l], out = dims_[l + 1];
            // (A) gW[l][out,in] = grad[l+1]^T[out,batch] @ act[l][batch,in]
            CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                in, out, batch, &alpha,
                d_act_[l], in, d_grad_[l + 1], out, &beta, d_gW_[l], in));
            gpu_bias_grad(d_grad_[l + 1], d_gb_[l], batch, out, stream_);

            // (B) grad[l][batch,in] = grad[l+1][batch,out] @ W[l][out,in]  (skip l==0)
            if (l > 0) {
                CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                    in, batch, out, &alpha,
                    d_W_[l], in, d_grad_[l + 1], out, &beta, d_grad_[l], in));
                gpu_relu_backward(d_grad_[l], d_mask_[l], batch * in, stream_);
            }
        }

        // Clip weight grads by global L2 norm (biases unclipped), then Adam.
        for (int l = 0; l < L_; ++l) {
            gpu_grad_clip(d_gW_[l], dims_[l + 1] * dims_[l], cfg_.grad_clip, d_grad_norm_, stream_);
        }
        for (int l = 0; l < L_; ++l) {
            gpu_adam_update(d_W_[l], d_gW_[l], d_mW_[l], d_vW_[l],
                            lr, cfg_.beta1, cfg_.beta2, cfg_.eps, bc1, bc2,
                            dims_[l + 1] * dims_[l], stream_);
            gpu_adam_update(d_b_[l], d_gb_[l], d_mb_[l], d_vb_[l],
                            lr, cfg_.beta1, cfg_.beta2, cfg_.eps, bc1, bc2,
                            dims_[l + 1], stream_);
        }
    }

    // Metrics from the last forward's logits vs last gathered labels.
    float sync_metrics(int batch, float* accuracy) {
        gpu_compute_metrics(d_act_[L_], d_batch_lab_, d_loss_scalar_, d_correct_scalar_,
                            batch, cfg_.out, stream_);
        GPU_CHECK(cudaMemcpyAsync(h_loss_, d_loss_scalar_, sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaMemcpyAsync(h_correct_, d_correct_scalar_, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_));
        GPU_CHECK(cudaStreamSynchronize(stream_));
        *accuracy = (float)(*h_correct_) / (float)batch;
        return *h_loss_;
    }

    // --------------------------------------------------------------------
    // EVAL: forward-only on an explicit list of ABSOLUTE window offsets.
    //   host_offsets[count] -> host_logits[count*3] (softmaxed probabilities).
    // Processes in batches of batch_cap. No labels / no backward.
    // --------------------------------------------------------------------
    void eval_offsets(const uint32_t* host_offsets, int count, float* host_logits) {
        if (count <= 0) return;
        const int B = cfg_.batch_cap;
        const int OUT = cfg_.out;
        int i = 0;
        while (i < count) {
            int cur = std::min(B, count - i);
            GPU_CHECK(cudaMemcpyAsync(d_batch_off_, host_offsets + i,
                                      (size_t)cur * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream_));
            gpu_gather_window(d_tickfeat_, d_batch_off_, d_act_[0],
                              cur, cfg_.window, cfg_.feat, n_ticks_dev_, stream_);
            forward(cur, false);
            GPU_CHECK(cudaMemcpyAsync(host_logits + (size_t)i * OUT, d_act_[L_],
                                      (size_t)cur * OUT * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream_));
            GPU_CHECK(cudaStreamSynchronize(stream_));
            i += cur;
        }
    }

    uint32_t n_ticks_dev() const { return n_ticks_dev_; }

    // --------------------------------------------------------------------
    // Download ALL layer weights/biases from GPU into host-side buffers.
    // out_W[l] -> [dims_[l+1] * dims_[l]] row-major (out,in); out_b[l] -> [dims_[l+1]].
    // out_dims -> the full dims vector [D_in, hidden..., out] (L_+1 entries).
    // Safe to call after training; no-op (returns false) if not ready.
    // --------------------------------------------------------------------
    bool download_weights(std::vector<std::vector<float>>& out_W,
                          std::vector<std::vector<float>>& out_b,
                          std::vector<int>& out_dims) const {
        if (!ready_) return false;
        out_dims = dims_;
        out_W.assign(L_, {});
        out_b.assign(L_, {});
        for (int l = 0; l < L_; ++l) {
            size_t wn = (size_t)dims_[l + 1] * dims_[l];
            size_t bn = (size_t)dims_[l + 1];
            out_W[l].resize(wn);
            out_b[l].resize(bn);
            GPU_CHECK(cudaMemcpy(out_W[l].data(), d_W_[l], wn * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(out_b[l].data(), d_b_[l], bn * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        }
        return true;
    }

    // --------------------------------------------------------------------
    // Save a committee-ready weights file (magic "DEEP"). Downloads all layer
    // W/b from the GPU and appends the per-tick normalization (Phase 1.6) so a
    // future loader can reconstruct the exact RAW->normalized window inputs.
    //
    // Layout (little-endian native float/int32):
    //   char[4]  "DEEP"
    //   int32    version (=1)
    //   int32    n_layers (== L_, number of GEMM layers)
    //   int32    n_dims   (== L_+1)
    //   int32    dims[n_dims]            // [D_in, hidden..., out]
    //   int32    window
    //   int32    feat                    // 27
    //   for l in [0, n_layers):
    //     int32  out_l (= dims[l+1]), int32 in_l (= dims[l])
    //     float  W[l][out_l * in_l]      // row-major (out,in)
    //     float  b[l][out_l]
    //   int32    norm_feat (= feat)
    //   float    pertick_mean[norm_feat] // Phase 1.6 mean[27]
    //   float    pertick_std[norm_feat]  // Phase 1.6 std[27]
    //
    // `pertick_mean`/`pertick_std` must each have cfg_.feat entries. Returns
    // false on download or write failure.
    // --------------------------------------------------------------------
    bool save(const std::string& path,
              const std::vector<float>& pertick_mean,
              const std::vector<float>& pertick_std) const {
        if (!ready_) return false;
        std::vector<std::vector<float>> hW, hb;
        std::vector<int> dims;
        if (!download_weights(hW, hb, dims)) return false;

        std::ofstream out(path, std::ios::binary);
        if (!out) return false;

        const char magic[4] = {'D', 'E', 'E', 'P'};
        out.write(magic, 4);
        int32_t version = 1;
        int32_t n_layers = L_;
        int32_t n_dims = (int32_t)dims.size();
        out.write(reinterpret_cast<const char*>(&version), 4);
        out.write(reinterpret_cast<const char*>(&n_layers), 4);
        out.write(reinterpret_cast<const char*>(&n_dims), 4);
        for (int d : dims) {
            int32_t dv = d;
            out.write(reinterpret_cast<const char*>(&dv), 4);
        }
        int32_t window = cfg_.window;
        int32_t feat = cfg_.feat;
        out.write(reinterpret_cast<const char*>(&window), 4);
        out.write(reinterpret_cast<const char*>(&feat), 4);

        for (int l = 0; l < L_; ++l) {
            int32_t out_l = dims[l + 1], in_l = dims[l];
            out.write(reinterpret_cast<const char*>(&out_l), 4);
            out.write(reinterpret_cast<const char*>(&in_l), 4);
            out.write(reinterpret_cast<const char*>(hW[l].data()),
                      (std::streamsize)hW[l].size() * sizeof(float));
            out.write(reinterpret_cast<const char*>(hb[l].data()),
                      (std::streamsize)hb[l].size() * sizeof(float));
        }

        // Per-tick normalization (Phase 1.6). Pad/truncate to feat for safety.
        int32_t norm_feat = feat;
        out.write(reinterpret_cast<const char*>(&norm_feat), 4);
        for (int d = 0; d < feat; ++d) {
            float v = (d < (int)pertick_mean.size()) ? pertick_mean[(size_t)d] : 0.0f;
            out.write(reinterpret_cast<const char*>(&v), 4);
        }
        for (int d = 0; d < feat; ++d) {
            float v = (d < (int)pertick_std.size()) ? pertick_std[(size_t)d] : 1.0f;
            out.write(reinterpret_cast<const char*>(&v), 4);
        }
        return static_cast<bool>(out);
    }

    ~DeepMlpTrainer() {
        if (!ready_) return;
        cublasDestroy(handle_);
        for (int l = 0; l < L_; ++l) {
            cudaFree(d_W_[l]); cudaFree(d_b_[l]);
            cudaFree(d_mW_[l]); cudaFree(d_vW_[l]); cudaFree(d_mb_[l]); cudaFree(d_vb_[l]);
            cudaFree(d_gW_[l]); cudaFree(d_gb_[l]);
        }
        for (int l = 0; l <= L_; ++l) {
            if (d_act_[l]) cudaFree(d_act_[l]);
            if (l >= 1 && d_grad_[l]) cudaFree(d_grad_[l]);
            if (l >= 1 && l < L_ && d_mask_[l]) cudaFree(d_mask_[l]);
        }
        cudaFree(d_grad_norm_);
        cudaFree(d_indices_); cudaFree(d_batch_off_); cudaFree(d_batch_lab_);
        if (d_class_weights_) cudaFree(d_class_weights_);
        if (d_entry_off_) cudaFree(d_entry_off_);
        if (d_labels_all_) cudaFree(d_labels_all_);
        if (d_tickfeat_) cudaFree(d_tickfeat_);
        cudaFree(d_loss_scalar_); cudaFree(d_correct_scalar_);
        cudaFreeHost(h_loss_); cudaFreeHost(h_correct_);
        cudaStreamDestroy(stream_);
    }
};

}  // namespace from::cuda

#endif  // FROM_CUDA
