#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef FROM_CUDA
#include <cuda_runtime.h>
#endif

namespace from::cuda {

#ifdef FROM_CUDA

class DeviceMemoryPool {
    void* pool_ptr_ = nullptr;
    size_t pool_size_ = 3ULL * 1024ULL * 1024ULL * 1024ULL;
    size_t used_ = 0;
    std::mutex mtx_;
    std::vector<std::pair<size_t, size_t>> free_list_;

    DeviceMemoryPool() {
        cudaError_t err = cudaMalloc(&pool_ptr_, pool_size_);
        if (err != cudaSuccess) {
            pool_size_ = 1536ULL * 1024ULL * 1024ULL;
            err = cudaMalloc(&pool_ptr_, pool_size_);
        }
        if (err != cudaSuccess) {
            pool_ptr_ = nullptr;
            pool_size_ = 0;
            throw std::runtime_error("cudaMalloc device pool failed");
        }
        free_list_.push_back({0, pool_size_});
    }

public:
    static DeviceMemoryPool& instance() {
        static DeviceMemoryPool pool;
        return pool;
    }

    ~DeviceMemoryPool() {
        if (pool_ptr_) cudaFree(pool_ptr_);
    }

    float* alloc(size_t n_floats) {
        size_t bytes = n_floats * sizeof(float);
        bytes = (bytes + 255U) & ~size_t{255U};
        std::lock_guard<std::mutex> lock(mtx_);
        for (size_t i = 0; i < free_list_.size(); ++i) {
            auto [off, sz] = free_list_[i];
            if (sz >= bytes) {
                free_list_[i] = {off + bytes, sz - bytes};
                if (free_list_[i].second == 0) free_list_.erase(free_list_.begin() + static_cast<std::ptrdiff_t>(i));
                used_ += bytes;
                return reinterpret_cast<float*>(static_cast<unsigned char*>(pool_ptr_) + off);
            }
        }
        throw std::runtime_error("CUDA device pool exhausted");
    }

    void free(float* ptr, size_t n_floats) {
        if (!ptr) return;
        size_t bytes = n_floats * sizeof(float);
        bytes = (bytes + 255U) & ~size_t{255U};
        size_t off = static_cast<size_t>(reinterpret_cast<unsigned char*>(ptr) - static_cast<unsigned char*>(pool_ptr_));
        std::lock_guard<std::mutex> lock(mtx_);
        free_list_.push_back({off, bytes});
        used_ = used_ > bytes ? used_ - bytes : 0;
        std::sort(free_list_.begin(), free_list_.end());
        std::vector<std::pair<size_t, size_t>> merged;
        for (auto [o, s] : free_list_) {
            if (!merged.empty() && merged.back().first + merged.back().second == o) merged.back().second += s;
            else merged.push_back({o, s});
        }
        free_list_.swap(merged);
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        free_list_.clear();
        free_list_.push_back({0, pool_size_});
        used_ = 0;
    }

    size_t used_bytes() const { return used_; }
    size_t total_bytes() const { return pool_size_; }
    float utilization() const { return pool_size_ ? static_cast<float>(used_) / static_cast<float>(pool_size_) : 0.0f; }
};

struct DeviceTensor {
    float* ptr = nullptr;
    size_t n = 0;

    DeviceTensor() = default;
    explicit DeviceTensor(size_t n_floats) : ptr(DeviceMemoryPool::instance().alloc(n_floats)), n(n_floats) {}
    ~DeviceTensor() { DeviceMemoryPool::instance().free(ptr, n); }
    DeviceTensor(DeviceTensor&& other) noexcept : ptr(other.ptr), n(other.n) { other.ptr = nullptr; other.n = 0; }
    DeviceTensor& operator=(DeviceTensor&& other) noexcept {
        if (this != &other) {
            DeviceMemoryPool::instance().free(ptr, n);
            ptr = other.ptr;
            n = other.n;
            other.ptr = nullptr;
            other.n = 0;
        }
        return *this;
    }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    void upload(const float* host_ptr, size_t n_floats, cudaStream_t stream = 0) {
        if (n_floats > n) throw std::runtime_error("DeviceTensor upload exceeds allocation");
        cudaMemcpyAsync(ptr, host_ptr, n_floats * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    void download(float* host_ptr, size_t n_floats, cudaStream_t stream = 0) const {
        if (n_floats > n) throw std::runtime_error("DeviceTensor download exceeds allocation");
        cudaMemcpyAsync(host_ptr, ptr, n_floats * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    void zero(cudaStream_t stream = 0) { cudaMemsetAsync(ptr, 0, n * sizeof(float), stream); }
};

#else

using cudaStream_t = void*;

class DeviceMemoryPool {
public:
    static DeviceMemoryPool& instance() {
        static DeviceMemoryPool pool;
        return pool;
    }
    float* alloc(size_t) { throw std::runtime_error("CUDA backend is not compiled"); }
    void free(float*, size_t) {}
    void reset() {}
    size_t used_bytes() const { return 0; }
    size_t total_bytes() const { return 0; }
    float utilization() const { return 0.0f; }
};

struct DeviceTensor {
    float* ptr = nullptr;
    size_t n = 0;
    DeviceTensor() = default;
    explicit DeviceTensor(size_t) { throw std::runtime_error("CUDA backend is not compiled"); }
    void upload(const float*, size_t, cudaStream_t = nullptr) {}
    void download(float*, size_t, cudaStream_t = nullptr) const {}
    void zero(cudaStream_t = nullptr) {}
};

#endif

}  // namespace from::cuda
