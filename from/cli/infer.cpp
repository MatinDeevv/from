#include "commands.hpp"

#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"
#include "model/direction_model.hpp"
#include "model/sequence_model.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace from {

int run_infer(const CliArgs& args) {
    std::string model_path = args.get("--model", "weights.from");
    std::string input = args.get("--input", "XAUUSD_ticks_all.parquet");
    std::string output = args.get("--output", "signals.csv");
    float threshold = std::stof(args.get("--threshold", "0.65"));
    size_t tick_limit = static_cast<size_t>(std::stoull(args.get("--ticks", "10000")));
    bool use_ensemble = args.has("--ensemble");
    bool use_linear = args.has("--linear");

    ParquetReader reader(input);
    TickProcessor processor;
    Normalizer normalizer(FROM_MAX_FEATURES);
    Windower windower(512, 64, 128);

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
                int64_t ts = features.time_ms.empty() ? 0 : features.time_ms[std::min(features.time_ms.size() - 1, emitted * 64 + 512)];
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
    if (use_ensemble) {
        for (uint32_t seed : Ensemble::SEEDS) {
            std::ostringstream p;
            p << "weights_seed" << seed << "_best.from";
            if (!std::filesystem::exists(p.str())) { p.str(""); p.clear(); p << "weights_seed" << seed << ".from"; }
            require(std::filesystem::exists(p.str()), "Missing ensemble model: " + p.str());
            models.emplace_back(0.00005f, seed);
            SequenceModelIO::load(p.str(), models.back());
            std::cout << "[ENSEMBLE] Loaded " << p.str() << "\n";
        }
    } else {
        models.emplace_back(0.00005f, 42);
        std::string p = std::filesystem::exists(model_path) ? model_path : "weights_best.from";
        require(std::filesystem::exists(p), "No model: " + p);
        SequenceModelIO::load(p, models.back());
        std::cout << "[INFER] Loaded " << p << "\n";
    }

    // Verify feat_norm is available
    require(models[0].feat_norm_ready, "Model was saved without feat_norm; retrain with updated code.");

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

            int64_t ts = features.time_ms.empty() ? 0 : features.time_ms[std::min(features.time_ms.size() - 1, emitted * 64 + 512)];

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
