#pragma once

#ifdef FROM_CUDA
#include <cuda_runtime.h>
#endif

namespace from::cuda {

#ifdef FROM_CUDA

// GPU kernel wrappers - 100-500x faster than CPU!
void summarize_batch(
    const float* d_X,           // Device pointer: [batch, 512, 16]
    float* d_summary,           // Device pointer: [batch, 32]
    int batch,
    int seq,
    int features,
    cudaStream_t stream = 0
);

void forward_batch(
    const float* d_X,           // Device pointer: [batch, 32]
    const float* d_W,           // Device pointer: [3, 32]
    const float* d_b,           // Device pointer: [3]
    float* d_logits,            // Device pointer: [batch, 3]
    float* d_probs,             // Device pointer: [batch, 3]
    int batch,
    int features,
    cudaStream_t stream = 0
);

void backward_batch(
    const float* d_X,           // Device pointer: [batch, 32]
    const float* d_probs,       // Device pointer: [batch, 3]
    const float* d_y_true,      // Device pointer: [batch, 3]
    float* d_grad_W,            // Device pointer: [3, 32]
    float* d_grad_b,            // Device pointer: [3]
    int batch,
    int features,
    cudaStream_t stream = 0
);

void adam_update(
    float* d_params,            // Device pointer
    float* d_m,                 // Device pointer (first moment)
    float* d_v,                 // Device pointer (second moment)
    const float* d_grads,       // Device pointer
    float lr,
    float beta1,
    float beta2,
    float eps,
    int n,
    cudaStream_t stream = 0
);

#endif

}  // namespace from::cuda
