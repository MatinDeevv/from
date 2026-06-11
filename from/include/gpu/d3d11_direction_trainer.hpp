#pragma once

#include "model/direction_model.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <wrl/client.h>
#endif

#include <array>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace from::gpu {

class D3D11DirectionTrainer {
#ifdef _WIN32
    using ComPtrDevice = Microsoft::WRL::ComPtr<ID3D11Device>;
    using ComPtrContext = Microsoft::WRL::ComPtr<ID3D11DeviceContext>;
    using ComPtrShader = Microsoft::WRL::ComPtr<ID3D11ComputeShader>;
    using ComPtrBuffer = Microsoft::WRL::ComPtr<ID3D11Buffer>;
    using ComPtrSRV = Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>;
    using ComPtrUAV = Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>;

    ComPtrDevice device_;
    ComPtrContext ctx_;
    ComPtrShader shader_;
    ComPtrBuffer w_[2], b_[2], x_, y_, constants_, staging_w_, staging_b_;
    ComPtrSRV w_srv_[2], b_srv_[2], x_srv_, y_srv_;
    ComPtrUAV w_uav_[2], b_uav_[2];
    int active_ = 0;
    size_t capacity_batch_ = 0;
    bool ready_ = false;

    struct Params {
        unsigned batch;
        unsigned dim;
        float lr;
        float pad;
    };

    static void require_hr(HRESULT hr, const char* msg) {
        if (FAILED(hr)) throw std::runtime_error(msg);
    }

    static std::string shader_source() {
        return R"(
cbuffer Params : register(b0) { uint B; uint D; float LR; float PAD; };
StructuredBuffer<float> X : register(t0);
StructuredBuffer<float> Y : register(t1);
StructuredBuffer<float> WIn : register(t2);
StructuredBuffer<float> BiasIn : register(t3);
RWStructuredBuffer<float> WOut : register(u0);
RWStructuredBuffer<float> BiasOut : register(u1);

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x;
    uint wcount = 3 * D;
    uint total = wcount + 3;
    if (id >= total) return;
    bool is_bias = id >= wcount;
    uint c = is_bias ? (id - wcount) : (id / D);
    uint d = is_bias ? 0 : (id % D);
    float grad = 0.0;
    for (uint n = 0; n < B; ++n) {
        float z0 = BiasIn[0];
        float z1 = BiasIn[1];
        float z2 = BiasIn[2];
        for (uint j = 0; j < D; ++j) {
            float xv = X[n * D + j];
            z0 += WIn[j] * xv;
            z1 += WIn[D + j] * xv;
            z2 += WIn[2 * D + j] * xv;
        }
        float m = max(z0, max(z1, z2));
        float e0 = exp(z0 - m);
        float e1 = exp(z1 - m);
        float e2 = exp(z2 - m);
        float s = e0 + e1 + e2 + 1.0e-8;
        float p = (c == 0 ? e0 : (c == 1 ? e1 : e2)) / s;
        float g = p - Y[n * 3 + c];
        grad += is_bias ? g : g * X[n * D + d];
    }
    grad /= max(1.0, (float)B);
    if (is_bias) BiasOut[c] = BiasIn[c] - LR * grad;
    else WOut[id] = WIn[id] - LR * grad;
}
)";
    }

    void create_structured_buffer(size_t floats, const float* init, UINT bind, ComPtrBuffer& buffer) {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = static_cast<UINT>(floats * sizeof(float));
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = bind;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        D3D11_SUBRESOURCE_DATA data{};
        data.pSysMem = init;
        require_hr(device_->CreateBuffer(&desc, init ? &data : nullptr, &buffer), "D3D11 structured buffer create failed");
    }

    void create_srv(ComPtrBuffer& buffer, size_t floats, ComPtrSRV& srv) {
        D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = static_cast<UINT>(floats);
        desc.Format = DXGI_FORMAT_UNKNOWN;
        require_hr(device_->CreateShaderResourceView(buffer.Get(), &desc, &srv), "D3D11 SRV create failed");
    }

    void create_uav(ComPtrBuffer& buffer, size_t floats, ComPtrUAV& uav) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = static_cast<UINT>(floats);
        desc.Format = DXGI_FORMAT_UNKNOWN;
        require_hr(device_->CreateUnorderedAccessView(buffer.Get(), &desc, &uav), "D3D11 UAV create failed");
    }

    void create_staging(size_t floats, ComPtrBuffer& buffer) {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = static_cast<UINT>(floats * sizeof(float));
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.StructureByteStride = sizeof(float);
        require_hr(device_->CreateBuffer(&desc, nullptr, &buffer), "D3D11 staging buffer create failed");
    }

    void ensure_batch_buffers(size_t batch) {
        if (batch <= capacity_batch_) return;
        capacity_batch_ = batch;
        create_structured_buffer(batch * FROM_DIRECTION_SUMMARY_DIM, nullptr, D3D11_BIND_SHADER_RESOURCE, x_);
        create_structured_buffer(batch * FROM_DIRECTION_CLASSES, nullptr, D3D11_BIND_SHADER_RESOURCE, y_);
        create_srv(x_, batch * FROM_DIRECTION_SUMMARY_DIM, x_srv_);
        create_srv(y_, batch * FROM_DIRECTION_CLASSES, y_srv_);
    }

    void download(DirectionModel& model) {
        ctx_->CopyResource(staging_w_.Get(), w_[active_].Get());
        ctx_->CopyResource(staging_b_.Get(), b_[active_].Get());
        D3D11_MAPPED_SUBRESOURCE map{};
        require_hr(ctx_->Map(staging_w_.Get(), 0, D3D11_MAP_READ, 0, &map), "D3D11 map W failed");
        std::memcpy(model.weight.data(), map.pData, model.weight.size() * sizeof(float));
        ctx_->Unmap(staging_w_.Get(), 0);
        require_hr(ctx_->Map(staging_b_.Get(), 0, D3D11_MAP_READ, 0, &map), "D3D11 map bias failed");
        std::memcpy(model.bias.data(), map.pData, model.bias.size() * sizeof(float));
        ctx_->Unmap(staging_b_.Get(), 0);
    }
#endif

public:
    D3D11DirectionTrainer() {
#ifdef _WIN32
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(factory.GetAddressOf())))) return;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> chosen;
        for (UINT i = 0;; ++i) {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            std::wstring name(desc.Description);
            if (name.find(L"NVIDIA") != std::wstring::npos) {
                chosen = adapter;
                break;
            }
        }
        if (!chosen) return;
        D3D_FEATURE_LEVEL fl{};
        if (FAILED(D3D11CreateDevice(chosen.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                                     D3D11_SDK_VERSION, &device_, &fl, &ctx_))) {
            return;
        }
        Microsoft::WRL::ComPtr<ID3DBlob> code;
        Microsoft::WRL::ComPtr<ID3DBlob> err;
        std::string src = shader_source();
        if (FAILED(D3DCompile(src.data(), src.size(), nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &err))) {
            return;
        }
        if (FAILED(device_->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader_))) return;
        Params p{1, static_cast<unsigned>(FROM_DIRECTION_SUMMARY_DIM), 0.001f, 0.0f};
        D3D11_BUFFER_DESC cdesc{};
        cdesc.ByteWidth = sizeof(Params);
        cdesc.Usage = D3D11_USAGE_DEFAULT;
        cdesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA cdata{&p, 0, 0};
        if (FAILED(device_->CreateBuffer(&cdesc, &cdata, &constants_))) return;
        ready_ = true;
#endif
    }

    bool available() const { return ready_; }

    void initialize(const DirectionModel& model) {
#ifdef _WIN32
        if (!ready_) return;
        size_t wc = model.weight.size();
        size_t bc = model.bias.size();
        for (int i = 0; i < 2; ++i) {
            create_structured_buffer(wc, model.weight.data(), D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, w_[i]);
            create_structured_buffer(bc, model.bias.data(), D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, b_[i]);
            create_srv(w_[i], wc, w_srv_[i]);
            create_srv(b_[i], bc, b_srv_[i]);
            create_uav(w_[i], wc, w_uav_[i]);
            create_uav(b_[i], bc, b_uav_[i]);
        }
        create_staging(wc, staging_w_);
        create_staging(bc, staging_b_);
        active_ = 0;
#else
        (void)model;
#endif
    }

    float train_batch(DirectionModel& model, const std::vector<Sample>& samples, size_t start, size_t n, float* accuracy) {
#ifndef _WIN32
        return model.train_batch(samples, start, n, accuracy);
#else
        if (!ready_) return model.train_batch(samples, start, n, accuracy);
        float loss = model.evaluate(std::vector<Sample>(samples.begin() + static_cast<std::ptrdiff_t>(start),
                                                        samples.begin() + static_cast<std::ptrdiff_t>(start + n)),
                                    accuracy);
        model.last_grad_norm = model.batch_gradient_norm(samples, start, n);
        require(model.last_grad_norm > 1.0e-10f, "backward produced zero parameter gradients");
        std::vector<float> x(n * FROM_DIRECTION_SUMMARY_DIM);
        std::vector<float> y(n * FROM_DIRECTION_CLASSES);
        for (size_t i = 0; i < n; ++i) {
            auto s = model.summarize(samples[start + i]);
            std::copy(s.begin(), s.end(), x.begin() + static_cast<std::ptrdiff_t>(i * FROM_DIRECTION_SUMMARY_DIM));
            for (size_t c = 0; c < FROM_DIRECTION_CLASSES; ++c) y[i * FROM_DIRECTION_CLASSES + c] = samples[start + i].y_dir[c];
        }
        ensure_batch_buffers(n);
        ctx_->UpdateSubresource(x_.Get(), 0, nullptr, x.data(), 0, 0);
        ctx_->UpdateSubresource(y_.Get(), 0, nullptr, y.data(), 0, 0);
        Params p{static_cast<unsigned>(n), static_cast<unsigned>(FROM_DIRECTION_SUMMARY_DIM), model.lr, 0.0f};
        ctx_->UpdateSubresource(constants_.Get(), 0, nullptr, &p, 0, 0);
        int next = 1 - active_;
        ID3D11ShaderResourceView* srvs[4] = {x_srv_.Get(), y_srv_.Get(), w_srv_[active_].Get(), b_srv_[active_].Get()};
        ID3D11UnorderedAccessView* uavs[2] = {w_uav_[next].Get(), b_uav_[next].Get()};
        ctx_->CSSetShader(shader_.Get(), nullptr, 0);
        ctx_->CSSetConstantBuffers(0, 1, constants_.GetAddressOf());
        ctx_->CSSetShaderResources(0, 4, srvs);
        ctx_->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);

        // Dispatch with proper parallelism - was Dispatch(1,1,1) = only 256 threads!
        // Total work: 3*D weights + 3 biases per gradient computation
        unsigned total_params = static_cast<unsigned>(3 * FROM_DIRECTION_SUMMARY_DIM + 3);
        unsigned thread_groups = (total_params + 255) / 256;  // Round up to fill GPU
        ctx_->Dispatch(thread_groups, 1, 1);
        ID3D11ShaderResourceView* null_srvs[4] = {};
        ID3D11UnorderedAccessView* null_uavs[2] = {};
        ctx_->CSSetShaderResources(0, 4, null_srvs);
        ctx_->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
        active_ = next;
        download(model);
        return loss;
#endif
    }
};

}  // namespace from::gpu
