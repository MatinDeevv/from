#pragma once

#include "data/normalizer.hpp"
#include "model/from_model.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace from {

class Serializer {
    static void append_bytes(std::vector<uint8_t>& buf, const void* ptr, size_t n) {
        const auto* p = static_cast<const uint8_t*>(ptr);
        buf.insert(buf.end(), p, p + n);
    }

    template <class T>
    static void append_value(std::vector<uint8_t>& buf, const T& v) {
        append_bytes(buf, &v, sizeof(T));
    }

    template <class T>
    static T read_value(const std::vector<uint8_t>& buf, size_t& pos) {
        require(pos + sizeof(T) <= buf.size(), "serializer read past end");
        T v{};
        std::memcpy(&v, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

public:
    static void save(FromModel& model, const Normalizer& normalizer, const std::string& path) {
        std::vector<uint8_t> buf;
        auto params = model.parameters();
        FromHeader header{};
        std::memcpy(header.magic, "FROM", 4);
        header.version = 1;
        header.num_layers = 1;
        header.total_params = model.parameter_count();
        header.creation_time = static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
        header.arch_hash = 0x46524f4dU;
        std::snprintf(header.description, sizeof(header.description), "From neural market engine");
        append_value(buf, header);

        LayerBlock layer{};
        std::snprintf(layer.layer_name, sizeof(layer.layer_name), "from_model");
        layer.num_tensors = params.size();
        append_value(buf, layer);
        for (const auto& p : params) {
            TensorBlock tb{};
            std::snprintf(tb.tensor_name, sizeof(tb.tensor_name), "%s", p.name.c_str());
            tb.ndim = static_cast<uint32_t>(p.value->shape().size());
            for (size_t i = 0; i < p.value->shape().size() && i < 8; ++i) tb.shape[i] = p.value->shape()[i];
            tb.num_elements = p.value->numel();
            tb.dtype = FROM_DTYPE_F32;
            append_value(buf, tb);
            Tensor<float> c = p.value->contiguous();
            append_bytes(buf, c.data_ptr(), c.numel() * sizeof(float));
        }
        uint64_t dims = normalizer.mean().size();
        append_value(buf, dims);
        for (double v : normalizer.mean()) append_value(buf, v);
        for (double v : normalizer.m2()) append_value(buf, v);
        for (uint64_t v : normalizer.count()) append_value(buf, v);
        uint8_t frozen = normalizer.frozen() ? 1 : 0;
        append_value(buf, frozen);
        uint64_t crc = from_crc64(buf.data(), buf.size());
        append_value(buf, crc);
        std::ofstream out(path, std::ios::binary);
        require(static_cast<bool>(out), "Cannot write model file: " + path);
        out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }

    static void load(const std::string& path, FromModel* model, Normalizer* normalizer) {
        std::ifstream in(path, std::ios::binary);
        require(static_cast<bool>(in), "Cannot open model file: " + path);
        std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        require(buf.size() > sizeof(uint64_t) + sizeof(FromHeader), "Model file too small");
        uint64_t stored_crc{};
        std::memcpy(&stored_crc, buf.data() + buf.size() - sizeof(uint64_t), sizeof(uint64_t));
        uint64_t calc_crc = from_crc64(buf.data(), buf.size() - sizeof(uint64_t));
        require(stored_crc == calc_crc, "Model checksum mismatch");
        size_t pos = 0;
        FromHeader header = read_value<FromHeader>(buf, pos);
        require(std::memcmp(header.magic, "FROM", 4) == 0, "Invalid FROM magic");
        require(header.version == 1, "Unsupported FROM version");
        LayerBlock layer = read_value<LayerBlock>(buf, pos);
        auto params = model->parameters();
        size_t layer_tensors = static_cast<size_t>(layer.num_tensors);
        size_t n = std::min(layer_tensors, params.size());
        for (size_t pi = 0; pi < layer_tensors; ++pi) {
            TensorBlock tb = read_value<TensorBlock>(buf, pos);
            size_t bytes = static_cast<size_t>(tb.num_elements) * sizeof(float);
            require(pos + bytes <= buf.size(), "Tensor block exceeds file size");
            if (pi < n && params[pi].value->numel() == tb.num_elements) {
                std::memcpy(params[pi].value->data_ptr(), buf.data() + pos, bytes);
            }
            pos += bytes;
        }
        if (normalizer && pos + sizeof(uint64_t) < buf.size()) {
            uint64_t dims = read_value<uint64_t>(buf, pos);
            require(dims <= 1024, "Normalizer dimension is invalid");
            std::vector<double> mean(static_cast<size_t>(dims)), m2(static_cast<size_t>(dims));
            std::vector<uint64_t> count(static_cast<size_t>(dims));
            for (double& v : mean) v = read_value<double>(buf, pos);
            for (double& v : m2) v = read_value<double>(buf, pos);
            for (uint64_t& v : count) v = read_value<uint64_t>(buf, pos);
            bool frozen = read_value<uint8_t>(buf, pos) != 0;
            normalizer->set_state(std::move(mean), std::move(m2), std::move(count), frozen);
        }
    }
};

}  // namespace from
