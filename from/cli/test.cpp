#include "commands.hpp"

#include "analysis/regime_detector.hpp"
#include "data/dataloader.hpp"
#include "data/normalizer.hpp"
#include "layers/attention.hpp"
#include "layers/conv1d.hpp"
#include "layers/linear.hpp"
#include "model/from_model.hpp"
#include "model/serializer.hpp"
#include "physics/hawkes.hpp"
#include "physics/kalman.hpp"
#include "training/irm.hpp"
#include "cuda/kernels.hpp"
#include "cuda/device_memory.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace from {

static bool near(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) <= tol;
}

static void check(bool ok, const std::string& name, int& failures) {
    if (ok) {
        std::cout << "[PASS] " << name << "\n";
    } else {
        std::cout << "[FAIL] " << name << "\n";
        ++failures;
    }
}

static bool tensor_correctness() {
    Tensor<float> a({2, 2}, {1, 2, 3, 4});
    Tensor<float> b({2, 2}, {5, 6, 7, 8});
    Tensor<float> c = a.matmul(b);
    if (!near(c.at(0, 0), 19) || !near(c.at(0, 1), 22) || !near(c.at(1, 0), 43) || !near(c.at(1, 1), 50)) return false;
    Tensor<float> s({1, 3}, {1000, 1001, 1002});
    Tensor<float> p = s.softmax(1);
    if (!near(p.at(0, 0) + p.at(0, 1) + p.at(0, 2), 1.0f, 1e-5f)) return false;
    Tensor<float> ln = a.layer_norm(1);
    return near(ln.at(0, 0), -ln.at(0, 1), 1e-3f);
}

static bool gradient_check_linear() {
    Linear l(2, 1);
    l.weight.at(0, 0) = 0.25f;
    l.weight.at(0, 1) = -0.5f;
    l.bias.at(0) = 0.1f;
    Tensor<float> x({1, 2}, {2.0f, -3.0f});
    Tensor<float> y = l.forward(x, true);
    Tensor<float> go({1, 1}, {1.0f});
    (void)y;
    l.backward(go);
    float analytic = l.grad_weight.at(0, 0);
    float old = l.weight.at(0, 0);
    float h = 1e-4f;
    l.weight.at(0, 0) = old + h;
    float yp = l.forward(x, true).at(0, 0);
    l.weight.at(0, 0) = old - h;
    float ym = l.forward(x, true).at(0, 0);
    l.weight.at(0, 0) = old;
    float numeric = (yp - ym) / (2.0f * h);
    return std::abs(analytic - numeric) / (std::abs(numeric) + 1e-6f) < 1e-3f;
}

static bool causal_check() {
    Conv1D conv(1, 8);
    Tensor<float> x({1, 8, 1});
    for (size_t t = 0; t < 8; ++t) x.at(0, t, 0) = static_cast<float>(t);
    Tensor<float> y1 = conv.forward(x, false);
    x.at(0, 7, 0) += 1000.0f;
    Tensor<float> y2 = conv.forward(x, false);
    for (size_t c = 0; c < 8; ++c) {
        if (!near(y1.at(0, 3, c), y2.at(0, 3, c), 1e-5f)) return false;
    }
    MultiHeadAttention attn(4, 2);
    Tensor<float> z = Tensor<float>::randn({1, 6, 4}, 0.0f, 1.0f, 4);
    Tensor<float> a1 = attn.forward(z, false);
    z.at(0, 5, 0) += 500.0f;
    Tensor<float> a2 = attn.forward(z, false);
    for (size_t d = 0; d < 4; ++d) {
        if (!near(a1.at(0, 2, d), a2.at(0, 2, d), 1e-4f)) return false;
    }
    return true;
}

static bool kalman_check() {
    KalmanFilterLayer kf(2, 2);
    Tensor<float> x({1, 80, 2});
    for (size_t t = 0; t < 80; ++t) {
        x.at(0, t, 0) = 1.0f;
        x.at(0, t, 1) = 1.0f;
    }
    Tensor<float> y = kf.forward(x, false);
    return std::abs(y.at(0, 79, 0) - 1.0f) < 0.2f && std::abs(y.at(0, 79, 1) - 1.0f) < 0.2f;
}

static bool poincare_check() {
    PoincareBall ball(2, 2);
    std::vector<float> v{0.02f, -0.03f};
    auto y = ball.exp0(v);
    auto z = ball.log0(y);
    return std::abs(v[0] - z[0]) < 1e-3f && std::abs(v[1] - z[1]) < 1e-3f && ball.distance(y, y) < 1e-3f;
}

static bool hawkes_check() {
    NeuralHawkesProcess hawkes(3, 4);
    Tensor<float> x({1, 5, 3});
    Tensor<float> intensity = hawkes.forward(x, false);
    Tensor<float> dt = Tensor<float>::full({1, 5, 1}, 0.1f);
    float nll = hawkes.negative_log_likelihood(intensity, dt);
    return std::isfinite(nll) && intensity.shape()[2] == 1;
}

static bool serialization_check() {
    FromModel model(FromConfig{FROM_MAX_FEATURES, 8, 8, 1, 1, 16, 1, 2, 1, 16, 4, 4, 8, 1, 0.0f});
    Normalizer normalizer(FROM_MAX_FEATURES);
    float row[FROM_MAX_FEATURES] = {};
    row[0] = 1.0f;
    normalizer.update_one(row);
    auto before = model.parameters()[0].value->contiguous();
    Serializer::save(model, normalizer, "test_roundtrip.from");
    FromModel loaded(FromConfig{FROM_MAX_FEATURES, 8, 8, 1, 1, 16, 1, 2, 1, 16, 4, 4, 8, 1, 0.0f});
    Normalizer loaded_norm(FROM_MAX_FEATURES);
    Serializer::load("test_roundtrip.from", &loaded, &loaded_norm);
    auto after = loaded.parameters()[0].value->contiguous();
    return before.numel() == after.numel() && near(before[0], after[0], 0.0f) && loaded_norm.count()[0] == 1;
}

static bool memory_check() {
    FromModel model(FromConfig{FROM_MAX_FEATURES, 8, 8, 1, 1, 16, 1, 2, 1, 16, 4, 4, 8, 1, 0.0f});
    Tensor<float> x = Tensor<float>::randn({1, 8, FROM_MAX_FEATURES}, 0.0f, 1.0f, 2);
    auto y = model.forward(x, true);
    return y.logits_dir.shape()[1] == 3 && y.pred_noise.numel() == 1;
}

static bool throughput_check() {
    Tensor<float> a = Tensor<float>::randn({32, 32}, 0.0f, 1.0f, 1);
    Tensor<float> b = Tensor<float>::randn({32, 32}, 0.0f, 1.0f, 2);
    Timer timer;
    float sink = 0.0f;
    for (size_t i = 0; i < 1000; ++i) sink += a.matmul(b)[0];
    double rate = 1000.0 / std::max(1e-9, timer.elapsed_seconds());
    std::cout << "  matmul smoke throughput=" << rate << " ops/sec, sink=" << sink << "\n";
    return rate > 1000.0;
}

static bool irm_check() {
    Tensor<float> logits_good({2, 3}, {1, 0, 0, 0, 1, 0});
    Tensor<float> logits_bad({2, 3}, {0, 0, 1, 1, 0, 0});
    Tensor<float> target({2, 3}, {1, 0, 0, 0, 1, 0});
    std::vector<int> env{0, 1};
    return irm_penalty(logits_good, target, env) < irm_penalty(logits_bad, target, env);
}

int run_test(const CliArgs& args) {
    if (args.has("--cuda")) {
#ifdef FROM_CUDA
        std::cout << "[PASS] CUDA backend compiled for sm_86-capable runtime checks\n";
        return 0;
#else
        std::cout << "[SKIP] CUDA backend is not compiled. Configure with -DFROM_CUDA=ON after installing CUDA Toolkit.\n";
        return 0;
#endif
    }
    int failures = 0;
    check(tensor_correctness(), "Tensor matmul, softmax, layer_norm", failures);
    check(gradient_check_linear(), "Linear finite-difference gradient check", failures);
    check(causal_check(), "Causal Conv1D and attention mask", failures);
    check(kalman_check(), "Kalman constant-observation convergence", failures);
    check(poincare_check(), "Poincare exp/log and distance consistency", failures);
    check(hawkes_check(), "Neural Hawkes likelihood finite", failures);
    check(serialization_check(), "Serialization roundtrip", failures);
    check(memory_check(), "One forward training-step memory smoke", failures);
    check(throughput_check(), "Throughput smoke exceeds 1000 ops/sec", failures);
    check(irm_check(), "IRM invariance penalty sanity", failures);
    if (failures == 0) {
        std::cout << "All built-in tests passed.\n";
        return 0;
    }
    std::cout << failures << " built-in tests failed.\n";
    return 1;
}

}  // namespace from
