#include "data/dataloader.hpp"

#include <cstddef>
#include <iterator>
#include <utility>

namespace from {

Batch make_batch(const std::vector<Sample>& samples, size_t start, size_t batch_size) {
    size_t n = std::min(batch_size, samples.size() - start);
    size_t seq = samples[start].X.shape()[0];
    size_t dim = samples[start].X.shape()[1];
    Batch b;
    b.X = Tensor<float>({n, seq, dim});
    b.y_dir = Tensor<float>({n, 3});
    b.y_mag = Tensor<float>({n, 1});
    b.y_vol = Tensor<float>({n, 1});
    b.y_spread = Tensor<float>({n, 1});
    b.env_ids.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const Sample& s = samples[start + i];
        for (size_t t = 0; t < seq; ++t) {
            for (size_t d = 0; d < dim; ++d) {
                b.X.at(i, t, d) = s.X.at(t, d);
            }
        }
        for (size_t c = 0; c < 3; ++c) {
            b.y_dir.at(i, c) = s.y_dir[c];
        }
        b.y_mag.at(i, 0) = s.y_mag;
        b.y_vol.at(i, 0) = s.y_vol;
        b.y_spread.at(i, 0) = s.y_spread;
        b.env_ids[i] = s.env_id;
    }
    return b;
}

DataLoader::DataLoader(std::string path, size_t batch_size, size_t chunk_size, size_t prefetch_depth,
                       bool augment, size_t window_size, size_t stride, size_t horizon)
    : path_(std::move(path)),
      batch_size_(batch_size),
      chunk_size_(chunk_size),
      prefetch_depth_(prefetch_depth),
      augment_(augment),
      window_size_(window_size),
      stride_(stride),
      horizon_(horizon),
      reader_worker_([this]() { reader_loop(); }),
      feature_worker_([this]() { feature_loop(); }),
      sample_worker_([this]() { sample_loop(); }),
      batch_worker_([this]() { batch_loop(); }) {}

DataLoader::~DataLoader() {
    stop();
    if (reader_worker_.joinable()) reader_worker_.join();
    if (feature_worker_.joinable()) feature_worker_.join();
    if (sample_worker_.joinable()) sample_worker_.join();
    if (batch_worker_.joinable()) batch_worker_.join();
}

void DataLoader::stop() {
    shutdown_.store(true);
    cv_full_.notify_all();
    cv_empty_.notify_all();
    cv_tick_.notify_all();
    cv_feature_.notify_all();
    cv_sample_.notify_all();
}

void DataLoader::reader_loop() {
    try {
        ParquetReader reader(path_);
        while (!shutdown_.load() && reader.has_next_chunk()) {
            TickChunk ticks = reader.read_chunk(chunk_size_);
            std::unique_lock<std::mutex> lock(mutex_);
            cv_full_.wait(lock, [this]() { return shutdown_.load() || tick_queue_.size() < prefetch_depth_; });
            if (shutdown_.load()) break;
            tick_queue_.push(std::move(ticks));
            cv_tick_.notify_one();
        }
    } catch (...) {
    }
    reader_done_.store(true);
    cv_tick_.notify_all();
}

void DataLoader::feature_loop() {
    TickProcessor processor;
    Normalizer normalizer(FROM_MAX_FEATURES);
    while (!shutdown_.load()) {
        TickChunk ticks;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_tick_.wait(lock, [this]() { return shutdown_.load() || !tick_queue_.empty() || reader_done_.load(); });
            if (shutdown_.load()) break;
            if (tick_queue_.empty()) {
                if (reader_done_.load()) break;
                continue;
            }
            ticks = std::move(tick_queue_.front());
            tick_queue_.pop();
            cv_full_.notify_all();
        }
        FeatureChunk features = processor.process(ticks);
        features.features = normalizer.normalize_chunk(features.features, true);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_full_.wait(lock, [this]() { return shutdown_.load() || feature_queue_.size() < prefetch_depth_; });
            if (shutdown_.load()) break;
            feature_queue_.push(std::move(features));
            cv_feature_.notify_one();
        }
    }
    feature_done_.store(true);
    cv_feature_.notify_all();
}

void DataLoader::sample_loop() {
    Windower windower(window_size_, stride_, horizon_);
    Augmenter augmenter;
    while (!shutdown_.load()) {
        FeatureChunk features;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_feature_.wait(lock, [this]() { return shutdown_.load() || !feature_queue_.empty() || feature_done_.load(); });
            if (shutdown_.load()) break;
            if (feature_queue_.empty()) {
                if (feature_done_.load()) break;
                continue;
            }
            features = std::move(feature_queue_.front());
            feature_queue_.pop();
            cv_full_.notify_all();
        }
        std::vector<Sample> samples = windower.add(features);
        for (auto& s : samples) {
            if (augment_) augmenter.apply(s);
        }
        if (!samples.empty()) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_full_.wait(lock, [this]() { return shutdown_.load() || sample_queue_.size() < prefetch_depth_; });
            if (shutdown_.load()) break;
            sample_queue_.push(std::move(samples));
            cv_sample_.notify_one();
        }
    }
    sample_done_.store(true);
    cv_sample_.notify_all();
}

void DataLoader::batch_loop() {
    std::vector<Sample> pending;
    while (!shutdown_.load()) {
        std::vector<Sample> samples;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_sample_.wait(lock, [this]() { return shutdown_.load() || !sample_queue_.empty() || sample_done_.load(); });
            if (shutdown_.load()) break;
            if (sample_queue_.empty()) {
                if (sample_done_.load()) break;
                continue;
            }
            samples = std::move(sample_queue_.front());
            sample_queue_.pop();
            cv_full_.notify_all();
        }
        pending.insert(pending.end(), std::make_move_iterator(samples.begin()), std::make_move_iterator(samples.end()));
        while (pending.size() >= batch_size_ && !shutdown_.load()) {
            Batch b = make_batch(pending, 0, batch_size_);
            pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(batch_size_));
            std::unique_lock<std::mutex> lock(mutex_);
            cv_full_.wait(lock, [this]() { return shutdown_.load() || queue_.size() < prefetch_depth_; });
            if (shutdown_.load()) break;
            queue_.push(std::move(b));
            cv_empty_.notify_one();
        }
    }
    done_.store(true);
    cv_empty_.notify_all();
}

bool DataLoader::next(Batch& batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_empty_.wait(lock, [this]() { return !queue_.empty() || done_.load() || shutdown_.load(); });
    if (queue_.empty()) {
        return false;
    }
    batch = std::move(queue_.front());
    queue_.pop();
    cv_full_.notify_one();
    return true;
}

}  // namespace from
