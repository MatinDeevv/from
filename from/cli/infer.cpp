#include "commands.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "io/artifact.hpp"
#include "model/direction_model.hpp"
#include "model/sequence_model.hpp"
#include "utils/config_parser.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace from {

int run_infer(const CliArgs& args) {
    Config config;
    const std::string config_path = args.get("--config", "config.toml");
    if (std::filesystem::exists(config_path)) config.load(config_path);
    std::string model_path = args.get("--model", "weights.from");
    std::string input = args.get("--input", "XAUUSD_ticks_all.parquet");
    std::string output = args.get("--output", "signals.csv");
    const float threshold = std::stof(args.get("--threshold",
                                                std::to_string(config.get_float("inference.confidence_threshold", 0.50f))));
    const size_t window = static_cast<size_t>(std::stoull(args.get("--window",
                                                std::to_string(config.get_size("data.window_size", FROM_DEFAULT_WINDOW)))));
    const size_t stride = static_cast<size_t>(std::stoull(args.get("--stride",
                                                std::to_string(config.get_size("data.stride", 64)))));
    const size_t horizon = static_cast<size_t>(std::stoull(args.get("--horizon",
                                                std::to_string(config.get_size("data.horizon", FROM_DEFAULT_HORIZON)))));
    const float direction_threshold = std::stof(args.get("--direction-threshold",
                                                std::to_string(config.get_float("data.direction_threshold", 2.0f))));
    size_t tick_limit = static_cast<size_t>(std::stoull(args.get("--ticks", "10000")));
    bool use_ensemble = args.has("--ensemble");
    bool use_linear = args.has("--linear");

    ParquetReader reader(input);
    TickProcessor processor;
    Normalizer normalizer(FROM_MAX_FEATURES);
    require(window > 0 && stride > 0 && horizon > 0, "Inference window, stride, and horizon must be positive");
    Windower windower(window, stride, horizon, direction_threshold);

    std::ofstream out(output);
    require(static_cast<bool>(out), "Cannot write output: " + output);

    if (use_linear) {
        DirectionModel model;
        DirectionModelIO::load(model_path, &model, &normalizer);
        out << "timestamp_ms,direction,confidence,long_prob,flat_prob,short_prob\n";

        size_t emitted = 0;
        while (reader.has_next_chunk() && reader.rows_read() < tick_limit) {
            TickChunk ticks = reader.read_chunk(std::min<size_t>(tick_limit - reader.rows_read(), 500000));
            FeatureChunk features = processor.process(ticks);
            features.features = normalizer.normalize_chunk(features.features, false);
            for (const auto& s : windower.add(features)) {
                auto probs = model.predict(s);
                size_t cls = 0;
                for (size_t c = 1; c < 3; ++c) if (probs[c] > probs[cls]) cls = c;
                float conf = probs[cls];
                const char* dir = conf < threshold ? "flat" : (cls == 0 ? "long" : (cls == 2 ? "short" : "flat"));
                int64_t ts = features.time_ms.empty() ? 0 : features.time_ms[std::min(features.time_ms.size() - 1, emitted * stride + window)];
                out << ts << "," << dir << "," << std::fixed << std::setprecision(4)
                    << conf << "," << probs[0] << "," << probs[1] << "," << probs[2] << "\n";
                ++emitted;
            }
        }
        std::cout << "Wrote " << emitted << " signals to " << output << "\n";
        return 0;
    }

    // Sequence model inference with ensemble + regime + confidence
    out << "timestamp_ms,direction,confidence,ensemble_agree,regime,ood_flag,size_multiplier,session\n";

    std::vector<SequenceModel> models;
    std::string loaded_model_path;  // first model file actually loaded (for norm1 sidecar resolution)
    if (use_ensemble) {
        for (uint32_t seed : Ensemble::SEEDS) {
            std::ostringstream p;
            p << "weights_seed" << seed << "_best.from";
            if (!std::filesystem::exists(p.str())) { p.str(""); p.clear(); p << "weights_seed" << seed << ".from"; }
            require(std::filesystem::exists(p.str()), "Missing ensemble model: " + p.str());
            models.emplace_back(0.00005f, seed);
            SequenceModelIO::load(p.str(), models.back());
            if (loaded_model_path.empty()) loaded_model_path = p.str();
            std::cout << "[ENSEMBLE] Loaded " << p.str() << "\n";
        }
    } else {
        models.emplace_back(0.00005f, 42);
        std::string p = std::filesystem::exists(model_path) ? model_path : "weights_best.from";
        require(std::filesystem::exists(p), "No model: " + p);
        SequenceModelIO::load(p, models.back());
        loaded_model_path = p;
        std::cout << "[INFER] Loaded " << p << "\n";
    }

    // Verify feat_norm is available
    require(models[0].feat_norm_ready, "Model was saved without feat_norm; retrain with updated code.");

    // Load FIRST-PASS Normalizer (norm1.bin) so inference re-derives the SAME
    // raw->normalized feature mapping the model trained on. The walk-forward /
    // train artifacts write it next to the weights. Without it the fresh
    // Normalizer above stays at its default (mean 0 / var 1) and feeds raw-scale
    // features into a model trained on z-scored inputs => meaningless signals.
    // --norm1 overrides the sidecar location.
    std::string norm1_path = args.get("--norm1", "");
    if (norm1_path.empty()) {
        std::filesystem::path sidecar = std::filesystem::path(loaded_model_path).parent_path() / "norm1.bin";
        norm1_path = sidecar.string();
    }
    require(std::filesystem::exists(norm1_path),
            "First-pass normalizer not found: " + norm1_path +
            " (expected norm1.bin beside the model; pass --norm1 <path> to override). "
            "Inference without it produces wrong-scale inputs.");
    require(from::io::load_norm1(norm1_path, normalizer),
            "Failed to load first-pass normalizer (bad/truncated NRM1): " + norm1_path);
    std::cout << "[INFER] Loaded first-pass normalizer: " << norm1_path << "\n";

    size_t emitted = 0;
    while (reader.has_next_chunk() && reader.rows_read() < tick_limit) {
        TickChunk ticks = reader.read_chunk(std::min<size_t>(tick_limit - reader.rows_read(), 500000));
        FeatureChunk features = processor.process(ticks);
        features.features = normalizer.normalize_chunk(features.features, false);

        for (const auto& s : windower.add(features)) {
            std::vector<Sample> batch = {s};

            size_t votes[SEQ_NUM_CLASSES] = {};
            for (auto& m : models) {
                float summary[SEQ_SUMMARY_DIM];
                MultiScaleSummarizer::summarize(s, summary);
                // Apply second-pass normalization (critical for correct inference)
                m.apply_feat_norm(summary, 1);
                float logits[SEQ_NUM_CLASSES];
                m.forward(summary, 1, logits, false);
                float probs[SEQ_NUM_CLASSES];
                SequenceModel::softmax(logits, 1, probs);
                size_t pred = 0;
                for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) if (probs[c] > probs[pred]) pred = c;
                votes[pred]++;
            }

            size_t best_dir = 0;
            for (size_t c = 1; c < SEQ_NUM_CLASSES; ++c) if (votes[c] > votes[best_dir]) best_dir = c;
            size_t agree = votes[best_dir];

            // OOD
            float hidden[SEQ_HIDDEN2];
            models[0].get_hidden(batch, 0, 1, hidden);
            auto [regime, dist, ood] = models[0].regime.classify(hidden);

            float conf = (agree == models.size()) ? 1.0f : (agree >= 2) ? 0.5f : 0.0f;
            float size_mult = (!ood && agree == models.size()) ? 1.0f : (!ood && agree >= 2) ? 0.5f : 0.0f;

            // Always emit — use "flat" for no-trade signals
            const char* dir = "flat";
            if (!ood && conf >= threshold) {
                if (best_dir == 0) dir = "long";
                else if (best_dir == 2) dir = "short";
            }

            int64_t ts = features.time_ms.empty() ? 0 : features.time_ms[std::min(features.time_ms.size() - 1, emitted * stride + window)];

            // Session from timestamp (UTC hour)
            int64_t ms_day = 24LL * 60LL * 60LL * 1000LL;
            int64_t t_day = ((ts % ms_day) + ms_day) % ms_day;
            int hour = static_cast<int>(t_day / (60LL * 60LL * 1000LL));
            const char* session = "other";
            if (hour >= 7 && hour < 12) session = "london";
            else if (hour >= 13 && hour < 17) session = "newyork";
            else if (hour >= 12 && hour < 13) session = "overlap";
            else if (hour >= 23 || hour < 4) session = "asian";

            out << ts << "," << dir << "," << std::fixed << std::setprecision(4) << conf
                << "," << agree << "," << regime << "," << (ood ? 1 : 0) << "," << size_mult
                << "," << session << "\n";
            ++emitted;
        }
    }

    std::cout << "Wrote " << emitted << " signals to " << output << "\n";
    return 0;
}

}  // namespace from
