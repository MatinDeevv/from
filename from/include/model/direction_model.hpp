#pragma once

#include "common.h"
#include "data/normalizer.hpp"
#include "data/windower.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>

namespace from {

constexpr size_t FROM_DIRECTION_CLASSES = 3;
constexpr size_t FROM_DIRECTION_SUMMARY_DIM = FROM_MAX_FEATURES * 2;

class DirectionModel {
public:
    std::vector<float> weight;
    std::array<float, FROM_DIRECTION_CLASSES> bias{};
    float lr = 0.001f;
    float last_grad_norm = 0.0f;

    explicit DirectionModel(float learning_rate = 0.001f)
        : weight(FROM_DIRECTION_CLASSES * FROM_DIRECTION_SUMMARY_DIM, 0.0f), lr(learning_rate) {
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (float& v : weight) v = dist(rng);
    }

    std::array<float, FROM_DIRECTION_SUMMARY_DIM> summarize(const Sample& s) const {
        std::array<float, FROM_DIRECTION_SUMMARY_DIM> x{};
        size_t seq = s.X.shape()[0];
        for (size_t t = 0; t < seq; ++t) {
            for (size_t d = 0; d < FROM_MAX_FEATURES; ++d) x[d] += s.X.at(t, d);
        }
        for (size_t d = 0; d < FROM_MAX_FEATURES; ++d) {
            x[d] /= static_cast<float>(std::max<size_t>(1, seq));
            x[FROM_MAX_FEATURES + d] = s.X.at(seq - 1, d);
        }
        return x;
    }

    std::array<float, FROM_DIRECTION_CLASSES> logits(const std::array<float, FROM_DIRECTION_SUMMARY_DIM>& x) const {
        std::array<float, FROM_DIRECTION_CLASSES> z{};
        for (size_t c = 0; c < FROM_DIRECTION_CLASSES; ++c) {
            z[c] = bias[c];
            for (size_t d = 0; d < FROM_DIRECTION_SUMMARY_DIM; ++d) {
                z[c] += weight[c * FROM_DIRECTION_SUMMARY_DIM + d] * x[d];
            }
        }
        return z;
    }

    static std::array<float, FROM_DIRECTION_CLASSES> softmax(const std::array<float, FROM_DIRECTION_CLASSES>& z) {
        float m = std::max({z[0], z[1], z[2]});
        std::array<float, FROM_DIRECTION_CLASSES> p{};
        float s = 0.0f;
        for (size_t c = 0; c < FROM_DIRECTION_CLASSES; ++c) {
            p[c] = std::exp(z[c] - m);
            s += p[c];
        }
        for (float& v : p) v /= (s + FROM_EPS_F);
        return p;
    }

    std::array<float, FROM_DIRECTION_CLASSES> predict(const Sample& s) const {
        return softmax(logits(summarize(s)));
    }

    static size_t truth_class(const Sample& sample) {
        size_t truth = 0;
        for (size_t c = 1; c < FROM_DIRECTION_CLASSES; ++c) {
            if (sample.y_dir[c] > sample.y_dir[truth]) truth = c;
        }
        return truth;
    }

    float batch_gradient_norm(const std::vector<Sample>& samples, size_t start, size_t n) const {
        std::vector<float> gw(weight.size(), 0.0f);
        std::array<float, FROM_DIRECTION_CLASSES> gb{};
        for (size_t i = 0; i < n; ++i) {
            const Sample& sample = samples[start + i];
            auto x = summarize(sample);
            auto p = softmax(logits(x));
            for (size_t c = 0; c < FROM_DIRECTION_CLASSES; ++c) {
                float g = p[c] - sample.y_dir[c];
                gb[c] += g;
                for (size_t d = 0; d < FROM_DIRECTION_SUMMARY_DIM; ++d) {
                    gw[c * FROM_DIRECTION_SUMMARY_DIM + d] += g * x[d];
                }
            }
        }
        float inv_n = 1.0f / static_cast<float>(n);
        float norm2 = 0.0f;
        for (float g : gw) norm2 += g * g * inv_n * inv_n;
        for (float g : gb) norm2 += g * g * inv_n * inv_n;
        return std::sqrt(norm2);
    }

    float train_batch(const std::vector<Sample>& samples, size_t start, size_t n, float* accuracy) {
        std::vector<float> gw(weight.size(), 0.0f);
        std::array<float, FROM_DIRECTION_CLASSES> gb{};
        float loss = 0.0f;
        size_t correct = 0;
        for (size_t i = 0; i < n; ++i) {
            const Sample& sample = samples[start + i];
            auto x = summarize(sample);
            auto p = softmax(logits(x));
            size_t truth = truth_class(sample);
            size_t pred = 0;
            for (size_t c = 1; c < FROM_DIRECTION_CLASSES; ++c) {
                if (p[c] > p[pred]) pred = c;
            }
            if (pred == truth) ++correct;
            loss -= std::log(p[truth] + FROM_EPS_F);
            for (size_t c = 0; c < FROM_DIRECTION_CLASSES; ++c) {
                float g = p[c] - sample.y_dir[c];
                gb[c] += g;
                for (size_t d = 0; d < FROM_DIRECTION_SUMMARY_DIM; ++d) {
                    gw[c * FROM_DIRECTION_SUMMARY_DIM + d] += g * x[d];
                }
            }
        }
        float inv_n = 1.0f / static_cast<float>(n);
        float norm2 = 0.0f;
        for (float g : gw) norm2 += g * g * inv_n * inv_n;
        for (float g : gb) norm2 += g * g * inv_n * inv_n;
        float scale = 1.0f;
        float norm = std::sqrt(norm2);
        last_grad_norm = norm;
        require(norm > 1.0e-10f, "backward produced zero parameter gradients");
        if (norm > 1.0f) scale = 1.0f / (norm + FROM_EPS_F);
        for (size_t i = 0; i < weight.size(); ++i) weight[i] -= lr * scale * gw[i] * inv_n;
        for (size_t c = 0; c < FROM_DIRECTION_CLASSES; ++c) bias[c] -= lr * scale * gb[c] * inv_n;
        *accuracy = static_cast<float>(correct) * inv_n;
        return loss * inv_n;
    }

    // ULTRA-FAST: Train on PRE-COMPUTED summaries!
    // Summaries already computed by AVX2 kernel in data pipeline.
    // This function is JUST a 32→3 matmul + softmax + SGD.
    // Should take <1ms for batch=256!
    float train_batch_summarized(
        const float* summaries,  // [n, 32] PRE-COMPUTED by pipeline!
        const float* labels,     // [n, 3]
        size_t n,
        float* accuracy
    ) {
        std::vector<float> gw(weight.size(), 0.0f);
        std::array<float, FROM_DIRECTION_CLASSES> gb{};
        float loss = 0.0f;
        size_t correct = 0;

        for (size_t i = 0; i < n; ++i) {
            const float* x = summaries + i * FROM_DIRECTION_SUMMARY_DIM;
            const float* y = labels + i * 3;

            // Forward: z = W @ x + b (just 3 × 32 = 96 FMAs!)
            float z[3];
            for (size_t c = 0; c < 3; ++c) {
                z[c] = bias[c];
                const float* w_row = weight.data() + c * FROM_DIRECTION_SUMMARY_DIM;
                for (size_t d = 0; d < FROM_DIRECTION_SUMMARY_DIM; ++d) {
                    z[c] += w_row[d] * x[d];
                }
            }

            // Softmax
            float m = std::max({z[0], z[1], z[2]});
            float e0 = std::exp(z[0] - m);
            float e1 = std::exp(z[1] - m);
            float e2 = std::exp(z[2] - m);
            float s = e0 + e1 + e2 + FROM_EPS_F;
            float p[3] = {e0 / s, e1 / s, e2 / s};

            // Accuracy + loss
            size_t truth = (y[1] > y[0]) ? ((y[2] > y[1]) ? 2 : 1) : ((y[2] > y[0]) ? 2 : 0);
            size_t pred = (p[1] > p[0]) ? ((p[2] > p[1]) ? 2 : 1) : ((p[2] > p[0]) ? 2 : 0);
            if (pred == truth) ++correct;
            loss -= std::log(p[truth] + FROM_EPS_F);

            // Backward: grad = (p - y) ⊗ x
            for (size_t c = 0; c < 3; ++c) {
                float g = p[c] - y[c];
                gb[c] += g;
                for (size_t d = 0; d < FROM_DIRECTION_SUMMARY_DIM; ++d) {
                    gw[c * FROM_DIRECTION_SUMMARY_DIM + d] += g * x[d];
                }
            }
        }

        // SGD update with gradient clipping
        float inv_n = 1.0f / static_cast<float>(n);
        float norm2 = 0.0f;
        for (float g : gw) norm2 += g * g * inv_n * inv_n;
        for (float g : gb) norm2 += g * g * inv_n * inv_n;
        float norm = std::sqrt(norm2);
        last_grad_norm = norm;
        float scale = (norm > 1.0f) ? 1.0f / (norm + FROM_EPS_F) : 1.0f;
        for (size_t i = 0; i < weight.size(); ++i) weight[i] -= lr * scale * gw[i] * inv_n;
        for (size_t c = 0; c < 3; ++c) bias[c] -= lr * scale * gb[c] * inv_n;

        *accuracy = static_cast<float>(correct) * inv_n;
        return loss * inv_n;
    }

    float evaluate(const std::vector<Sample>& samples, float* accuracy) const {
        if (samples.empty()) {
            *accuracy = 0.0f;
            return 0.0f;
        }
        float loss = 0.0f;
        size_t correct = 0;
        for (const auto& sample : samples) {
            auto p = predict(sample);
            size_t truth = truth_class(sample);
            size_t pred = 0;
            for (size_t c = 1; c < FROM_DIRECTION_CLASSES; ++c) {
                if (p[c] > p[pred]) pred = c;
            }
            if (pred == truth) ++correct;
            loss -= std::log(p[truth] + FROM_EPS_F);
        }
        *accuracy = static_cast<float>(correct) / static_cast<float>(samples.size());
        return loss / static_cast<float>(samples.size());
    }
};

class DirectionModelIO {
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
        require(pos + sizeof(T) <= buf.size(), "direction model read past end");
        T v{};
        std::memcpy(&v, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

public:
    static void save(const DirectionModel& model, const Normalizer& normalizer, const std::string& path) {
        std::vector<uint8_t> buf;
        char magic[4] = {'F', 'D', 'R', '1'};
        append_bytes(buf, magic, 4);
        uint32_t version = 1;
        append_value(buf, version);
        uint64_t created = static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
        append_value(buf, created);
        uint64_t dim = FROM_DIRECTION_SUMMARY_DIM;
        uint64_t classes = FROM_DIRECTION_CLASSES;
        append_value(buf, dim);
        append_value(buf, classes);
        append_value(buf, model.lr);
        uint64_t nw = model.weight.size();
        append_value(buf, nw);
        append_bytes(buf, model.weight.data(), model.weight.size() * sizeof(float));
        append_bytes(buf, model.bias.data(), model.bias.size() * sizeof(float));
        uint64_t nd = normalizer.mean().size();
        append_value(buf, nd);
        for (double v : normalizer.mean()) append_value(buf, v);
        for (double v : normalizer.m2()) append_value(buf, v);
        for (uint64_t v : normalizer.count()) append_value(buf, v);
        uint8_t frozen = normalizer.frozen() ? 1 : 0;
        append_value(buf, frozen);
        uint64_t crc = from_crc64(buf.data(), buf.size());
        append_value(buf, crc);
        std::ofstream out(path, std::ios::binary);
        require(static_cast<bool>(out), "Cannot write direction model: " + path);
        out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }

    static void load(const std::string& path, DirectionModel* model, Normalizer* normalizer) {
        std::ifstream in(path, std::ios::binary);
        require(static_cast<bool>(in), "Cannot open direction model: " + path);
        std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        require(buf.size() > 16 + sizeof(uint64_t), "Direction model file too small");
        uint64_t stored_crc{};
        std::memcpy(&stored_crc, buf.data() + buf.size() - sizeof(uint64_t), sizeof(uint64_t));
        uint64_t calc_crc = from_crc64(buf.data(), buf.size() - sizeof(uint64_t));
        require(stored_crc == calc_crc, "Direction model checksum mismatch");
        size_t pos = 0;
        require(std::memcmp(buf.data(), "FDR1", 4) == 0, "Invalid direction model magic");
        pos += 4;
        uint32_t version = read_value<uint32_t>(buf, pos);
        require(version == 1, "Unsupported direction model version");
        (void)read_value<uint64_t>(buf, pos);
        uint64_t dim = read_value<uint64_t>(buf, pos);
        uint64_t classes = read_value<uint64_t>(buf, pos);
        require(dim == FROM_DIRECTION_SUMMARY_DIM && classes == FROM_DIRECTION_CLASSES, "Direction model shape mismatch");
        model->lr = read_value<float>(buf, pos);
        uint64_t nw = read_value<uint64_t>(buf, pos);
        require(nw == model->weight.size(), "Direction model weight count mismatch");
        std::memcpy(model->weight.data(), buf.data() + pos, static_cast<size_t>(nw) * sizeof(float));
        pos += static_cast<size_t>(nw) * sizeof(float);
        std::memcpy(model->bias.data(), buf.data() + pos, model->bias.size() * sizeof(float));
        pos += model->bias.size() * sizeof(float);
        uint64_t nd = read_value<uint64_t>(buf, pos);
        require(nd <= 1024, "Direction model normalizer dimension is invalid");
        std::vector<double> mean(static_cast<size_t>(nd)), m2(static_cast<size_t>(nd));
        std::vector<uint64_t> count(static_cast<size_t>(nd));
        for (double& v : mean) v = read_value<double>(buf, pos);
        for (double& v : m2) v = read_value<double>(buf, pos);
        for (uint64_t& v : count) v = read_value<uint64_t>(buf, pos);
        bool frozen = read_value<uint8_t>(buf, pos) != 0;
        if (normalizer) normalizer->set_state(std::move(mean), std::move(m2), std::move(count), frozen);
    }
};

}  // namespace from
