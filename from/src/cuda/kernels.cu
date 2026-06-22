#include "cuda/kernels.hpp"

#ifdef FROM_CUDA

#include <cuda_runtime.h>

namespace from::cuda {

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                            int M, int K, int N) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// static: internal linkage — gpu_trainer_kernels.cu defines its own from::cuda::softmax_kernel,
// and GNU ld rejects the duplicate external symbol (MSVC tolerated it). Only launched here.
static __global__ void softmax_kernel(float* __restrict__ x, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + r * cols;
    float mx = -1.0e30f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) mx = fmaxf(mx, row[d]);
    for (int offset = 16; offset > 0; offset >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float s = 0.0f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) {
        row[d] = expf(row[d] - mx);
        s += row[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_down_sync(0xffffffff, s, offset);
    s = __shfl_sync(0xffffffff, s, 0);
    for (int d = threadIdx.x; d < cols; d += blockDim.x) row[d] /= s;
}

__global__ void adam_step_kernel(float* __restrict__ params, float* __restrict__ m, float* __restrict__ v,
                                 const float* __restrict__ grads, float lr, float beta1, float beta2, float eps,
                                 float weight_decay, float bc1, float bc2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = grads[i] + weight_decay * params[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    float mh = m[i] / bc1;
    float vh = v[i] / bc2;
    params[i] -= lr * mh / (sqrtf(vh) + eps);
}

__global__ void lstm_gates_kernel(const float* __restrict__ x, const float* __restrict__ h,
                                  const float* __restrict__ Wx, const float* __restrict__ Wh,
                                  const float* __restrict__ bias, float* __restrict__ gates,
                                  int batch, int input_size, int hidden_size) {
    int b = blockIdx.x;
    int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || g >= 4 * hidden_size) return;
    float val = bias[g];
    for (int i = 0; i < input_size; ++i) val += x[b * input_size + i] * Wx[g * input_size + i];
    for (int i = 0; i < hidden_size; ++i) val += h[b * hidden_size + i] * Wh[g * hidden_size + i];
    gates[b * 4 * hidden_size + g] = val;
}

__global__ void lstm_activations_kernel(const float* __restrict__ gates, const float* __restrict__ c_prev,
                                        float* __restrict__ c_next, float* __restrict__ h_next,
                                        int batch, int hidden_size) {
    int b = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || h >= hidden_size) return;
    int base = b * 4 * hidden_size;
    float ig = 1.0f / (1.0f + expf(-gates[base + h]));
    float fg = 1.0f / (1.0f + expf(-gates[base + hidden_size + h]));
    float gg = tanhf(gates[base + 2 * hidden_size + h]);
    float og = 1.0f / (1.0f + expf(-gates[base + 3 * hidden_size + h]));
    float c = fg * c_prev[b * hidden_size + h] + ig * gg;
    c_next[b * hidden_size + h] = c;
    h_next[b * hidden_size + h] = og * tanhf(c);
}

void gemm(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

void softmax(float* x, int rows, int cols, cudaStream_t stream) {
    softmax_kernel<<<rows, 32, 0, stream>>>(x, rows, cols);
}

void adam_step(float* params, float* m, float* v, const float* grads, float lr, float beta1, float beta2, float eps,
               float weight_decay, float bc1, float bc2, int n, cudaStream_t stream) {
    int block = 256;
    adam_step_kernel<<<(n + block - 1) / block, block, 0, stream>>>(params, m, v, grads, lr, beta1, beta2, eps,
                                                                    weight_decay, bc1, bc2, n);
}

// --- Kernels needed by CublasTrainer ---

__global__ void cross_entropy_kernel(const float* probs, const float* targets, float* losses, int batch, int classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    float loss = 0.0f;
    for (int c = 0; c < classes; c++) {
        loss -= targets[i * classes + c] * logf(probs[i * classes + c] + 1e-8f);
    }
    losses[i] = loss;
}

__global__ void sgd_update_kernel(float* params, const float* grads, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    params[i] -= lr * grads[i];
}

void softmax_inplace(float* x, int rows, int cols, cudaStream_t stream) {
    softmax_kernel<<<rows, 32, 0, stream>>>(x, rows, cols);
}

void cross_entropy_loss(const float* probs, const float* targets, float* losses, int batch, int classes, cudaStream_t stream) {
    int block = 256;
    cross_entropy_kernel<<<(batch + block - 1) / block, block, 0, stream>>>(probs, targets, losses, batch, classes);
}

void sgd_update(float* params, const float* grads, float lr, int n, cudaStream_t stream) {
    int block = 256;
    sgd_update_kernel<<<(n + block - 1) / block, block, 0, stream>>>(params, grads, lr, n);
}

}  // namespace from::cuda

#endif
