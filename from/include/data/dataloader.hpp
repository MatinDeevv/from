#pragma once

#include "data/augmenter.hpp"
#include "data/normalizer.hpp"
#include "data/parquet_reader.hpp"
#include "data/tick_processor.hpp"
#include "data/windower.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace from {

struct Batch {
    Tensor<float> X;
    Tensor<float> y_dir;
    Tensor<float> y_mag;
    Tensor<float> y_vol;
    Tensor<float> y_spread;
    std::vector<int> env_ids;
};

class DataLoader {
    std::string path_;
    size_t batch_size_;
    size_t chunk_size_;
    size_t prefetch_depth_;
    bool augment_;
    size_t window_size_;
    size_t stride_;
    size_t horizon_;

    std::queue<TickChunk> tick_queue_;
    std::queue<FeatureChunk> feature_queue_;
    std::queue<std::vector<Sample>> sample_queue_;
    std::queue<Batch> queue_;
    std::mutex mutex_;
    std::condition_variable cv_full_;
    std::condition_variable cv_empty_;
    std::condition_variable cv_tick_;
    std::condition_variable cv_feature_;
    std::condition_variable cv_sample_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> reader_done_{false};
    std::atomic<bool> feature_done_{false};
    std::atomic<bool> sample_done_{false};
    std::atomic<bool> done_{false};
    std::thread reader_worker_;
    std::thread feature_worker_;
    std::thread sample_worker_;
    std::thread batch_worker_;

    void reader_loop();
    void feature_loop();
    void sample_loop();
    void batch_loop();

public:
    DataLoader(std::string path, size_t batch_size = 256, size_t chunk_size = 2000000, size_t prefetch_depth = 32,
               bool augment = true, size_t window_size = 512, size_t stride = 64, size_t horizon = 128);
    ~DataLoader();
    bool next(Batch& batch);
    void stop();
};

Batch make_batch(const std::vector<Sample>& samples, size_t start, size_t batch_size);

}  // namespace from
