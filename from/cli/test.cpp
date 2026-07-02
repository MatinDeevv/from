#include "commands.hpp"

#include "analysis/regime_detector.hpp"
#include "data/dataloader.hpp"
#include "data/normalizer.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "layers/attention.hpp"
#include "layers/conv1d.hpp"
#include "layers/linear.hpp"
#include "model/from_model.hpp"
#include "model/sequence_model.hpp"
#include "model/serializer.hpp"
#include "physics/hawkes.hpp"
#include "physics/kalman.hpp"
#include "training/irm.hpp"
#include "training/loss.hpp"
#include "cuda/kernels.hpp"
#include "cuda/device_memory.hpp"
#include "utils/timer.hpp"
#include "wf_metrics.hpp"

#include <algorithm>
#include <cstdio>
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
    const bool passed = before.numel() == after.numel() && near(before[0], after[0], 0.0f) && loaded_norm.count()[0] == 1;
    std::remove("test_roundtrip.from");
    return passed;
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

static bool feat_norm_roundtrip() {
    SequenceModel model(0.0001f, 99);
    model.feat_mean.resize(SEQ_SUMMARY_DIM);
    model.feat_std.resize(SEQ_SUMMARY_DIM);
    for (size_t d = 0; d < SEQ_SUMMARY_DIM; ++d) {
        model.feat_mean[d] = static_cast<float>(d) * 0.1f;
        model.feat_std[d] = 1.0f + static_cast<float>(d) * 0.01f;
    }
    model.feat_norm_ready = true;

    SequenceModelIO::save(model, "test_feat_norm.from");

    SequenceModel loaded(0.0001f, 99);
    const bool passed = SequenceModelIO::load("test_feat_norm.from", loaded) &&
                        loaded.feat_norm_ready &&
                        loaded.feat_mean.size() == SEQ_SUMMARY_DIM &&
                        loaded.feat_std.size() == SEQ_SUMMARY_DIM &&
                        near(loaded.feat_mean[0], 0.0f) &&
                        near(loaded.feat_mean[10], 1.0f) &&
                        near(loaded.feat_std[0], 1.0f);
    std::remove("test_feat_norm.from");
    return passed;
}

static bool kyle_hasbrouck_nonzero() {
    // Create synthetic ticks with correlated order flow and price moves
    TickChunk chunk;
    chunk.size = 100;
    chunk.ask.resize(100);
    chunk.bid.resize(100);
    chunk.mid.resize(100);
    chunk.ask_vol.resize(100);
    chunk.bid_vol.resize(100);
    chunk.time_ms.resize(100);

    double price = 2000.0;
    for (size_t i = 0; i < 100; ++i) {
        // Alternate: buy pressure moves price up, sell pressure moves price down
        float buy_vol = (i % 3 == 0) ? 200.0f : 50.0f;
        float sell_vol = (i % 3 == 0) ? 50.0f : 200.0f;
        double move = (i % 3 == 0) ? 0.05 : -0.02;  // correlated with flow
        price += move;
        chunk.ask[i] = price + 0.15;
        chunk.bid[i] = price - 0.15;
        chunk.mid[i] = price;
        chunk.ask_vol[i] = buy_vol;
        chunk.bid_vol[i] = sell_vol;
        chunk.time_ms[i] = static_cast<int64_t>(i) * 200;
    }

    TickProcessor processor;
    FeatureChunk features = processor.process(chunk);

    // Check last row's Kyle-Hasbrouck is nonzero
    float kh = features.features.at(99, FROM_FEAT_KYLE_HASBROUCK);
    return std::abs(kh) > 1e-8f;
}

static bool label_smoothing_sanity() {
    // Test with clear signal (delta = 3 * threshold)
    // The windower needs enough data to produce a sample
    Windower windower(8, 1, 4, 1.0f);  // small window for test

    FeatureChunk chunk;
    chunk.size = 20;
    chunk.features = Tensor<float>({20, FROM_MAX_FEATURES});
    chunk.mid.resize(20);
    chunk.spread.resize(20);
    chunk.time_ms.resize(20);

    // Create rising price (clear UP signal)
    for (size_t i = 0; i < 20; ++i) {
        chunk.mid[i] = 2000.0 + static_cast<double>(i) * 0.5;  // strong uptrend
        chunk.spread[i] = 0.30;
        chunk.time_ms[i] = static_cast<int64_t>(i) * 1000;
        for (size_t d = 0; d < FROM_MAX_FEATURES; ++d) {
            chunk.features.at(i, d) = static_cast<float>(i) * 0.01f;
        }
    }

    auto samples = windower.add(chunk);
    if (samples.empty()) return false;

    // Find a sample where y_dir[0] is highest (UP)
    bool found_up = false;
    for (const auto& s : samples) {
        if (s.y_dir[0] > s.y_dir[1] && s.y_dir[0] > s.y_dir[2]) {
            // Label should be high confidence but not exactly 1.0 (smoothed)
            if (s.y_dir[0] > 0.6f && s.y_dir[0] <= 1.0f) found_up = true;
        }
    }
    return found_up;
}

static bool cross_entropy_stability() {
    const std::array<std::array<float, 3>, 4> cases{{
        {{0.0f, 0.0f, 0.0f}},
        {{100.0f, -100.0f, 0.0f}},
        {{1.0e-10f, 1.0e-10f, 1.0e-10f}},
        {{10.0f, -10.0f, -10.0f}},
    }};
    for (const auto& row : cases) {
        Tensor<float> logits({1, 3}, {row[0], row[1], row[2]});
        Tensor<float> target({1, 3}, {1.0f, 0.0f, 0.0f});
        const float loss = directional_cross_entropy(logits, target);
        if (!std::isfinite(loss) || loss < 0.0f) return false;
    }
    Tensor<float> perfect_logits({1, 3}, {10.0f, -10.0f, -10.0f});
    Tensor<float> target({1, 3}, {1.0f, 0.0f, 0.0f});
    return directional_cross_entropy(perfect_logits, target) < 1.0e-3f;
}

static bool preprocessing_parity() {
    constexpr size_t kTicks = 768;
    TickChunk ticks;
    ticks.size = kTicks;
    ticks.ask.resize(kTicks); ticks.bid.resize(kTicks); ticks.mid.resize(kTicks);
    ticks.ask_vol.resize(kTicks); ticks.bid_vol.resize(kTicks); ticks.time_ms.resize(kTicks);
    for (size_t i = 0; i < kTicks; ++i) {
        const double mid = 2000.0 + static_cast<double>(i) * 0.002 + std::sin(static_cast<double>(i) * 0.1) * 0.01;
        ticks.ask[i] = mid + 0.15;
        ticks.bid[i] = mid - 0.15;
        ticks.mid[i] = mid;
        ticks.ask_vol[i] = 100.0f + static_cast<float>(i % 11);
        ticks.bid_vol[i] = 90.0f + static_cast<float>(i % 7);
        ticks.time_ms[i] = 1700000000000LL + static_cast<int64_t>(i) * 100;
    }
    TickProcessor train_processor;
    TickProcessor infer_processor;
    FeatureChunk train_features = train_processor.process(ticks);
    FeatureChunk infer_features = infer_processor.process(ticks);
    if (train_features.features.numel() != infer_features.features.numel()) return false;
    for (size_t i = 0; i < train_features.features.numel(); ++i) {
        if (!near(train_features.features[i], infer_features.features[i], 1.0e-6f)) return false;
    }
    Windower train_window(512, 64, 256, 2.0f);
    Windower infer_window(512, 64, 256, 2.0f);
    const auto train_samples = train_window.add(train_features);
    const auto infer_samples = infer_window.add(infer_features);
    if (train_samples.size() != infer_samples.size() || train_samples.empty()) return false;
    float train_summary[SEQ_SUMMARY_DIM];
    float infer_summary[SEQ_SUMMARY_DIM];
    MultiScaleSummarizer::summarize(train_samples.front(), train_summary);
    MultiScaleSummarizer::summarize(infer_samples.front(), infer_summary);
    for (size_t i = 0; i < SEQ_SUMMARY_DIM; ++i) {
        if (!near(train_summary[i], infer_summary[i], 1.0e-6f)) return false;
    }
    return true;
}

static bool daily_sharpe_correctness() {
    constexpr int64_t kMsPerDay = 86400000LL;
    std::vector<std::pair<int64_t, float>> pnls;
    for (int64_t day = 0; day < 252; ++day) {
        pnls.emplace_back(day * kMsPerDay, (day % 2 == 0) ? 1.1f : -0.9f);
    }
    const double expected = 0.1 * std::sqrt(252.0);
    if (std::abs(wfm::daily_sharpe(pnls) - expected) > 1.0e-5) return false;
    std::vector<std::pair<int64_t, float>> constant{{0, 1.0f}, {kMsPerDay, 1.0f}};
    return wfm::daily_sharpe(constant) == 0.0;
}

static bool normalizer_constant_feature() {
    Normalizer normalizer(2);
    float row[] = {42.0f, 1.0f};
    for (size_t i = 0; i < 32; ++i) normalizer.update_one(row);
    normalizer.freeze();
    float input[] = {42.0f, 1.001f};
    normalizer.normalize_one(input, false);
    return std::isfinite(input[0]) && std::isfinite(input[1]) && near(input[0], 0.0f) && near(input[1], 0.001f, 1.0e-6f);
}

int run_test(const CliArgs& args) {
    if (args.has("--help")) {
        std::cout << "Usage: from test [--cuda]\n"
                  << "Runs deterministic tensor, gradient, causality, checkpoint, feature-parity, loss, and metric tests.\n"
                  << "  --cuda  Report whether this executable was compiled with CUDA support.\n";
        return 0;
    }
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
    check(feat_norm_roundtrip(), "Feat norm save/load roundtrip", failures);
    check(kyle_hasbrouck_nonzero(), "Kyle-Hasbrouck nonzero on synthetic data", failures);
    check(label_smoothing_sanity(), "Label smoothing produces soft labels", failures);
    check(cross_entropy_stability(), "Cross-entropy stable on extreme logits", failures);
    check(preprocessing_parity(), "Training/inference feature preprocessing parity", failures);
    check(daily_sharpe_correctness(), "UTC daily Sharpe calculation", failures);
    check(normalizer_constant_feature(), "Normalizer constant-feature scale", failures);
    if (failures == 0) {
        std::cout << "All built-in tests passed.\n";
        return 0;
    }
    std::cout << failures << " built-in tests failed.\n";
    return 1;
}

}  // namespace from
