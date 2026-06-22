#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>

#include "common.h"

// Mirror SEQ constants from sequence_model.hpp (can't include due to AVX2 intrinsics)
#define SEQ_IN_FEATURES_CU FROM_MAX_FEATURES
#define SEQ_NUM_SCALES_CU 9
#define SEQ_SUMMARY_DIM_CU (SEQ_NUM_SCALES_CU * SEQ_IN_FEATURES_CU)

namespace from::cuda {

// Gather batch from pre-loaded data
__global__ void gather_kernel(const float* __restrict__ all_data, const uint8_t* __restrict__ all_labels,
                              const uint32_t* __restrict__ indices, float* __restrict__ out_data,
                              uint8_t* __restrict__ out_labels, int batch, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    uint32_t idx = indices[i];
    out_labels[i] = all_labels[idx];
    const float* src = all_data + idx * dim;
    float* dst = out_data + i * dim;
    for (int d = 0; d < dim; ++d) dst[d] = src[d];
}

void gpu_gather_batch(const float* all_summaries, const uint8_t* all_labels,
                      const uint32_t* indices, float* out_input, uint8_t* out_labels,
                      int batch, int dim, cudaStream_t s) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    gather_kernel<<<blocks, threads, 0, s>>>(all_summaries, all_labels, indices, out_input, out_labels, batch, dim);
}

// ============================================================================
// wfdeep (P2) — raw per-tick window gather from a resident feature stream.
//
//   tickfeat : [n_ticks x feat] row-major (resident on device, NORMALIZED).
//   batch_off[i] = window-start ROW offset of sample i into tickfeat.
//   out      : [batch x (window*feat)] row-major, timestep-major feature-fastest:
//                out[i, t*feat + d] = tickfeat[(batch_off[i] + t)*feat + d]
//   This flattened layout is the first GEMM's contiguous IN vector. Because j runs
//   feature-fastest then timestep, the source read tickfeat[off*feat + j] is exactly
//   contiguous for j in [0, window*feat) -> fully coalesced on both sides.
//   Bounds guard: if (off + window > n_ticks) the sample window is written as zeros
//   (the host index sets MUST also exclude such samples; this is belt-and-suspenders).
// ============================================================================
__global__ void gather_window_kernel(const float* __restrict__ tickfeat,
                                      const uint32_t* __restrict__ batch_off,
                                      float* __restrict__ out,
                                      int batch, int window, int feat, uint32_t n_ticks) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long d_in = (long long)window * feat;
    long long total = (long long)batch * d_in;
    if (tid >= total) return;
    int i = (int)(tid / d_in);          // sample
    long long j = tid - (long long)i * d_in;  // flat index within window (0 .. window*feat-1)
    uint32_t off = batch_off[i];
    // Bounds guard: window must fit inside the resident stream.
    if ((uint64_t)off + (uint64_t)window > (uint64_t)n_ticks) {
        out[tid] = 0.0f;
        return;
    }
    out[tid] = tickfeat[(long long)off * feat + j];
}

void gpu_gather_window(const float* tickfeat, const uint32_t* batch_off,
                       float* out, int batch, int window, int feat,
                       uint32_t n_ticks, cudaStream_t s) {
    long long total = (long long)batch * window * feat;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    gather_window_kernel<<<(unsigned int)blocks, threads, 0, s>>>(
        tickfeat, batch_off, out, batch, window, feat, n_ticks);
}

// Resolve per-batch window offsets + labels from a fold subset using random
// sample indices into that subset:
//   out_off[i]    = entry_off[indices[i]]
//   out_labels[i] = labels_all[indices[i]]
__global__ void resolve_offsets_kernel(const uint32_t* __restrict__ entry_off,
                                       const uint8_t* __restrict__ labels_all,
                                       const uint32_t* __restrict__ indices,
                                       uint32_t* __restrict__ out_off,
                                       uint8_t* __restrict__ out_labels, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    uint32_t idx = indices[i];
    out_off[i] = entry_off[idx];
    out_labels[i] = labels_all[idx];
}

void gpu_resolve_offsets(const uint32_t* entry_off, const uint8_t* labels_all,
                         const uint32_t* indices, uint32_t* out_off, uint8_t* out_labels,
                         int batch, cudaStream_t s) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    resolve_offsets_kernel<<<blocks, threads, 0, s>>>(entry_off, labels_all, indices,
                                                      out_off, out_labels, batch);
}

// ReLU forward: x = max(0, x), write mask
__global__ void relu_fwd_kernel(float* x, char* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool active = x[i] > 0.0f;
    mask[i] = active ? 1 : 0;
    if (!active) x[i] = 0.0f;
}

void gpu_relu_forward(float* x, char* mask, int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_fwd_kernel<<<blocks, threads, 0, s>>>(x, mask, n);
}

// ReLU backward: grad *= mask
__global__ void relu_bwd_kernel(float* grad, const char* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!mask[i]) grad[i] = 0.0f;
}

void gpu_relu_backward(float* grad, const char* mask, int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_bwd_kernel<<<blocks, threads, 0, s>>>(grad, mask, n);
}

// Add bias: out[row, col] += bias[col]
__global__ void add_bias_kernel(float* out, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    out[idx] += bias[col];
}

void gpu_add_bias(float* out, const float* bias, int rows, int cols, cudaStream_t s) {
    int n = rows * cols;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, s>>>(out, bias, rows, cols);
}

// Softmax per row (data stored col-major from cuBLAS perspective but row-major for us)
// Data is actually [cols, rows] col-major = [rows, cols] row-major
// Each "row" in our sense has stride = cols, starting at row*cols
// But cuBLAS output is col-major [OUT, batch]: element [class, sample] at class + sample*OUT
// So each sample's logits are at stride OUT
__global__ void softmax_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float* r = x + row * cols;
    float mx = -FLT_MAX;
    for (int c = 0; c < cols; ++c) mx = fmaxf(mx, r[c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) { r[c] = expf(r[c] - mx); sum += r[c]; }
    float inv = 1.0f / (sum + 1e-8f);
    for (int c = 0; c < cols; ++c) r[c] *= inv;
}

void gpu_softmax(float* x, int rows, int cols, cudaStream_t s) {
    // x is [cols, rows] col-major = each sample at stride cols
    // Actually cuBLAS stores [OUT, batch] col-major, sample i at x[i*OUT .. i*OUT+OUT-1]
    // That's equivalent to [batch, OUT] row-major with stride OUT
    // So rows=batch, cols=OUT, stride=OUT per sample → just call with batch, OUT
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    softmax_kernel<<<blocks, threads, 0, s>>>(x, rows, cols);
}

// Weighted cross entropy grad: grad[i,c] = w[label] * (prob[i,c] - one_hot[i,c])
// The grad already holds probs at this point. We subtract one-hot and scale by class weight.
__global__ void ce_grad_weighted_kernel(float* grad, const uint8_t* labels,
                                         const float* class_weights, int batch, int classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    uint8_t label = labels[i];
    float w = class_weights[label];
    for (int c = 0; c < classes; ++c) {
        grad[i * classes + c] *= w;
    }
    grad[i * classes + label] -= w;
}

// Unweighted version (backward compat)
__global__ void ce_grad_kernel(float* grad, const uint8_t* labels, int batch, int classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    grad[i * classes + labels[i]] -= 1.0f;
}

void gpu_cross_entropy_grad(float* grad, const uint8_t* labels, int batch, int classes, cudaStream_t s) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    ce_grad_kernel<<<blocks, threads, 0, s>>>(grad, labels, batch, classes);
}

void gpu_cross_entropy_grad_weighted(float* grad, const uint8_t* labels, const float* class_weights,
                                      int batch, int classes, cudaStream_t s) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    ce_grad_weighted_kernel<<<blocks, threads, 0, s>>>(grad, labels, class_weights, batch, classes);
}

// Compute metrics: loss and accuracy from probs + labels
__global__ void metrics_kernel(const float* probs, const uint8_t* labels, float* loss_out, int* correct_out,
                               int batch, int classes) {
    // Single-block reduction for small batch sizes
    extern __shared__ float shared[];
    float* s_loss = shared;
    int* s_correct = (int*)(shared + blockDim.x);

    int tid = threadIdx.x;
    float my_loss = 0.0f;
    int my_correct = 0;

    for (int i = tid; i < batch; i += blockDim.x) {
        uint8_t label = labels[i];
        float p = probs[i * classes + label];
        my_loss -= logf(p + 1e-8f);
        // Find predicted class
        int pred = 0;
        float max_p = probs[i * classes];
        for (int c = 1; c < classes; ++c) {
            if (probs[i * classes + c] > max_p) { max_p = probs[i * classes + c]; pred = c; }
        }
        if (pred == label) my_correct++;
    }

    s_loss[tid] = my_loss;
    s_correct[tid] = my_correct;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_loss[tid] += s_loss[tid + s];
            s_correct[tid] += s_correct[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *loss_out = s_loss[0] / (float)batch;
        *correct_out = s_correct[0];
    }
}

void gpu_compute_metrics(const float* probs, const uint8_t* labels, float* loss_out, int* correct_out,
                         int batch, int classes, cudaStream_t s) {
    int threads = 256;
    size_t shared_sz = threads * sizeof(float) + threads * sizeof(int);
    metrics_kernel<<<1, threads, shared_sz, s>>>(probs, labels, loss_out, correct_out, batch, classes);
}

// Adam update kernel
__global__ void adam_kernel(float* w, const float* g, float* m, float* v,
                           float lr, float beta1, float beta2, float eps, float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float step = lr / bc1;
    float vh = vi / bc2;
    w[i] -= step * mi / (sqrtf(vh) + eps);
}

void gpu_adam_update(float* w, const float* g, float* m, float* v,
                    float lr, float beta1, float beta2, float eps, float bc1, float bc2,
                    int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adam_kernel<<<blocks, threads, 0, s>>>(w, g, m, v, lr, beta1, beta2, eps, bc1, bc2, n);
}

// Gradient clipping by global L2 norm.
// Kernel A: block-reduce sum-of-squares, atomicAdd into accumulator (norm_out holds sumsq).
// Kernel B: read sumsq, compute norm, scale each element if norm>max_norm.
// Kernel C: single thread overwrites *norm_out with the final (pre-clip) L2 norm.
__global__ void grad_clip_sumsq_kernel(const float* __restrict__ data, int n, float* __restrict__ sumsq) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? data[i] : 0.0f;
    sdata[tid] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sumsq, sdata[0]);
}

__global__ void grad_clip_scale_kernel(float* __restrict__ data, int n, float max_norm, const float* __restrict__ sumsq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float norm = sqrtf(*sumsq);
    if (norm > max_norm) {
        data[i] *= max_norm / norm;
    }
}

// Single thread: overwrite *norm_out (which holds sumsq) with the final L2 norm.
__global__ void grad_clip_finalize_kernel(float* __restrict__ norm_out) {
    *norm_out = sqrtf(*norm_out);
}

void gpu_grad_clip(float* data, int n, float max_norm, float* norm_out, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_sz = threads * sizeof(float);
    cudaMemsetAsync(norm_out, 0, sizeof(float), s);
    grad_clip_sumsq_kernel<<<blocks, threads, shared_sz, s>>>(data, n, norm_out);
    grad_clip_scale_kernel<<<blocks, threads, 0, s>>>(data, n, max_norm, norm_out);
    grad_clip_finalize_kernel<<<1, 1, 0, s>>>(norm_out);
}

// Bias gradient: gb[d] = sum over batch of grad[i*dim + d]
__global__ void bias_grad_kernel(const float* grad, float* gb, int batch, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float sum = 0.0f;
    for (int i = 0; i < batch; ++i) sum += grad[i * dim + d];
    gb[d] = sum;
}

void gpu_bias_grad(const float* grad, float* gb, int batch, int dim, cudaStream_t s) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    bias_grad_kernel<<<blocks, threads, 0, s>>>(grad, gb, batch, dim);
}

// Generate random indices on GPU using xorshift
__global__ void rand_indices_kernel(uint32_t* indices, int batch, uint32_t n_samples, uint32_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    // Strong, well-mixed hash of (seed, i) using constants DECOUPLED from the
    // caller's K=2654435761 to break cross-step index-stream correlation.
    uint32_t s = seed ^ ((uint32_t)i * 0x9E3779B9u);
    s ^= s >> 16; s *= 0x7feb352du; s ^= s >> 15; s *= 0x846ca68bu; s ^= s >> 16;
    indices[i] = s % n_samples;
}

void gpu_rand_indices(uint32_t* indices, int batch, uint32_t n_samples, uint32_t seed, cudaStream_t s) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    rand_indices_kernel<<<blocks, threads, 0, s>>>(indices, batch, n_samples, seed);
}

// ============================================================
// FUSED training kernel: entire forward+backward+adam in ONE launch
// One block per output neuron in the largest layer (H1=256).
// Each block processes the entire batch for one neuron's column.
// This eliminates ~20 kernel launches per step.
// ============================================================

// Dimensions from model constants
#define FUSED_IN SEQ_SUMMARY_DIM_CU  // 243
#define FUSED_H1 256
#define FUSED_H2 128
#define FUSED_OUT 3

// Fused forward+backward kernel for the 3-layer MLP
// Approach: each sample in the batch is processed by one thread.
// All threads cooperate to do the matmuls via shared mem.
// For batch <= 64, we can fit everything in registers + shared.
__global__ void fused_train_kernel(
    // Weights (read/write)
    float* __restrict__ w1, float* __restrict__ b1,   // [H1, IN], [H1]
    float* __restrict__ w2, float* __restrict__ b2,   // [H2, H1], [H2]
    float* __restrict__ w3, float* __restrict__ b3,   // [OUT, H2], [OUT]
    // Adam state
    float* __restrict__ mw1, float* __restrict__ vw1, float* __restrict__ mb1, float* __restrict__ vb1,
    float* __restrict__ mw2, float* __restrict__ vw2, float* __restrict__ mb2, float* __restrict__ vb2,
    float* __restrict__ mw3, float* __restrict__ vw3, float* __restrict__ mb3, float* __restrict__ vb3,
    // Data
    const float* __restrict__ all_data, const uint8_t* __restrict__ all_labels,
    // Batch workspace (pre-allocated, [batch, max_dim])
    float* __restrict__ h1_buf, float* __restrict__ h2_buf, float* __restrict__ logits_buf,
    float* __restrict__ grad3_buf, float* __restrict__ grad2_buf, float* __restrict__ grad1_buf,
    // Gradients (weight)
    float* __restrict__ gw1, float* __restrict__ gb1,
    float* __restrict__ gw2, float* __restrict__ gb2,
    float* __restrict__ gw3, float* __restrict__ gb3,
    // Params
    uint32_t n_samples, uint32_t batch, uint32_t seed,
    float lr, float bc1, float bc2,
    // Output
    float* __restrict__ loss_out, int* __restrict__ correct_out
) {
    // This kernel does NOT try to be fancy — it parallelizes across batch samples.
    // Each thread handles one sample's forward pass, then we reduce for backward.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (int)batch) return;

    // Generate random index for this sample
    uint32_t s = seed + tid * 2654435761u;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    uint32_t idx = s % n_samples;

    const float* input = all_data + idx * FUSED_IN;
    uint8_t label = all_labels[idx];

    // Forward layer 1: h1[j] = relu(input @ w1[j]^T + b1[j])
    float* my_h1 = h1_buf + tid * FUSED_H1;
    for (int j = 0; j < FUSED_H1; ++j) {
        float sum = b1[j];
        const float* wrow = w1 + j * FUSED_IN;
        int k = 0;
        for (; k + 3 < FUSED_IN; k += 4) {
            sum += input[k] * wrow[k] + input[k+1] * wrow[k+1]
                 + input[k+2] * wrow[k+2] + input[k+3] * wrow[k+3];
        }
        for (; k < FUSED_IN; ++k) sum += input[k] * wrow[k];
        my_h1[j] = sum > 0.0f ? sum : 0.0f;
    }

    // Forward layer 2: h2[j] = relu(h1 @ w2[j]^T + b2[j])
    float* my_h2 = h2_buf + tid * FUSED_H2;
    for (int j = 0; j < FUSED_H2; ++j) {
        float sum = b2[j];
        const float* wrow = w2 + j * FUSED_H1;
        for (int k = 0; k < FUSED_H1; k += 4) {
            sum += my_h1[k] * wrow[k] + my_h1[k+1] * wrow[k+1]
                 + my_h1[k+2] * wrow[k+2] + my_h1[k+3] * wrow[k+3];
        }
        my_h2[j] = sum > 0.0f ? sum : 0.0f;
    }

    // Forward layer 3: logits[c] = h2 @ w3[c]^T + b3[c], then softmax
    float logits[FUSED_OUT];
    float mx = -1e30f;
    for (int c = 0; c < FUSED_OUT; ++c) {
        float sum = b3[c];
        const float* wrow = w3 + c * FUSED_H2;
        for (int k = 0; k < FUSED_H2; k += 4) {
            sum += my_h2[k] * wrow[k] + my_h2[k+1] * wrow[k+1]
                 + my_h2[k+2] * wrow[k+2] + my_h2[k+3] * wrow[k+3];
        }
        logits[c] = sum;
        if (sum > mx) mx = sum;
    }

    // Softmax
    float probs[FUSED_OUT];
    float expsum = 0.0f;
    for (int c = 0; c < FUSED_OUT; ++c) { probs[c] = expf(logits[c] - mx); expsum += probs[c]; }
    float inv_s = 1.0f / (expsum + 1e-8f);
    for (int c = 0; c < FUSED_OUT; ++c) probs[c] *= inv_s;

    // Loss + accuracy for this sample
    float sample_loss = -logf(probs[label] + 1e-8f);
    int pred = 0;
    if (probs[1] > probs[pred]) pred = 1;
    if (probs[2] > probs[pred]) pred = 2;
    int correct = (pred == label) ? 1 : 0;

    // Atomic reduction for metrics
    atomicAdd(loss_out, sample_loss / (float)batch);
    atomicAdd(correct_out, correct);

    // grad3 = (probs - one_hot) / batch
    float inv_batch = 1.0f / (float)batch;
    float g3[FUSED_OUT];
    for (int c = 0; c < FUSED_OUT; ++c) {
        g3[c] = (probs[c] - (c == label ? 1.0f : 0.0f)) * inv_batch;
    }

    // Accumulate weight gradients for layer 3
    for (int c = 0; c < FUSED_OUT; ++c) {
        atomicAdd(&gb3[c], g3[c]);
        for (int k = 0; k < FUSED_H2; ++k) {
            atomicAdd(&gw3[c * FUSED_H2 + k], g3[c] * my_h2[k]);
        }
    }

    // grad2 = grad3 @ W3, then relu_backward
    float* my_g2 = grad2_buf + tid * FUSED_H2;
    for (int j = 0; j < FUSED_H2; ++j) {
        float sum = 0.0f;
        for (int c = 0; c < FUSED_OUT; ++c) sum += g3[c] * w3[c * FUSED_H2 + j];
        my_g2[j] = (my_h2[j] > 0.0f) ? sum : 0.0f;
    }

    // Accumulate weight gradients for layer 2
    for (int j = 0; j < FUSED_H2; ++j) {
        if (my_g2[j] == 0.0f) continue;
        atomicAdd(&gb2[j], my_g2[j]);
        for (int k = 0; k < FUSED_H1; ++k) {
            atomicAdd(&gw2[j * FUSED_H1 + k], my_g2[j] * my_h1[k]);
        }
    }

    // grad1 = grad2 @ W2, then relu_backward
    float* my_g1 = grad1_buf + tid * FUSED_H1;
    for (int j = 0; j < FUSED_H1; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < FUSED_H2; ++k) sum += my_g2[k] * w2[k * FUSED_H1 + j];
        my_g1[j] = (my_h1[j] > 0.0f) ? sum : 0.0f;
    }

    // Accumulate weight gradients for layer 1
    for (int j = 0; j < FUSED_H1; ++j) {
        if (my_g1[j] == 0.0f) continue;
        atomicAdd(&gb1[j], my_g1[j]);
        for (int k = 0; k < FUSED_IN; ++k) {
            atomicAdd(&gw1[j * FUSED_IN + k], my_g1[j] * input[k]);
        }
    }
}

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
    float* loss_out, int* correct_out, cudaStream_t stream) {

    // Zero gradients and metrics
    cudaMemsetAsync(gw1, 0, FUSED_H1 * FUSED_IN * sizeof(float), stream);
    cudaMemsetAsync(gb1, 0, FUSED_H1 * sizeof(float), stream);
    cudaMemsetAsync(gw2, 0, FUSED_H2 * FUSED_H1 * sizeof(float), stream);
    cudaMemsetAsync(gb2, 0, FUSED_H2 * sizeof(float), stream);
    cudaMemsetAsync(gw3, 0, FUSED_OUT * FUSED_H2 * sizeof(float), stream);
    cudaMemsetAsync(gb3, 0, FUSED_OUT * sizeof(float), stream);
    cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
    cudaMemsetAsync(correct_out, 0, sizeof(int), stream);

    // Launch fused kernel: one thread per sample
    int threads = 64;
    int blocks = (batch + threads - 1) / threads;
    fused_train_kernel<<<blocks, threads, 0, stream>>>(
        w1, b1, w2, b2, w3, b3,
        mw1, vw1, mb1, vb1,
        mw2, vw2, mb2, vb2,
        mw3, vw3, mb3, vb3,
        all_data, all_labels,
        h1_buf, h2_buf, logits_buf,
        grad3_buf, grad2_buf, grad1_buf,
        gw1, gb1, gw2, gb2, gw3, gb3,
        n_samples, batch, seed, lr, bc1, bc2,
        loss_out, correct_out);

    // Adam update on gradients (reuse existing kernel)
    int athreads = 256;
    adam_kernel<<<(FUSED_H1*FUSED_IN+athreads-1)/athreads, athreads, 0, stream>>>(w1, gw1, mw1, vw1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, FUSED_H1*FUSED_IN);
    adam_kernel<<<(FUSED_H1+athreads-1)/athreads, athreads, 0, stream>>>(b1, gb1, mb1, vb1, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, FUSED_H1);
    adam_kernel<<<(FUSED_H2*FUSED_H1+athreads-1)/athreads, athreads, 0, stream>>>(w2, gw2, mw2, vw2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, FUSED_H2*FUSED_H1);
    adam_kernel<<<(FUSED_H2+athreads-1)/athreads, athreads, 0, stream>>>(b2, gb2, mb2, vb2, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, FUSED_H2);
    adam_kernel<<<(FUSED_OUT*FUSED_H2+athreads-1)/athreads, athreads, 0, stream>>>(w3, gw3, mw3, vw3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, FUSED_OUT*FUSED_H2);
    adam_kernel<<<(FUSED_OUT+athreads-1)/athreads, athreads, 0, stream>>>(b3, gb3, mb3, vb3, lr, 0.9f, 0.999f, 1e-8f, bc1, bc2, FUSED_OUT);
}

}  // namespace from::cuda
