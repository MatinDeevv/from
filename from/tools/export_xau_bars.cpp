#include "data/parquet_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace from;

namespace {

struct Args {
    std::vector<std::string> xs;
    std::string get(const std::string& k, const std::string& d = "") const {
        for (size_t i = 0; i + 1 < xs.size(); ++i) {
            if (xs[i] == k) return xs[i + 1];
        }
        return d;
    }
};

struct Bar {
    int64_t bucket = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double spread_sum = 0.0;
    double ask_vol_sum = 0.0;
    double bid_vol_sum = 0.0;
    uint32_t ticks = 0;
};

void add_tick(std::vector<Bar>& bars, int64_t bucket, double mid, double spread, float ask_vol, float bid_vol) {
    if (bars.empty() || bars.back().bucket != bucket) {
        Bar b;
        b.bucket = bucket;
        b.open = b.high = b.low = b.close = mid;
        bars.push_back(b);
    }
    Bar& b = bars.back();
    b.high = std::max(b.high, mid);
    b.low = std::min(b.low, mid);
    b.close = mid;
    b.spread_sum += spread;
    b.ask_vol_sum += ask_vol;
    b.bid_vol_sum += bid_vol;
    ++b.ticks;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) args.xs.emplace_back(argv[i]);

    std::string data = args.get("--data", "XAUUSD_ticks_all.parquet");
    std::string out = args.get("--out", "xauusd_5m.csv");
    int tf_min = std::stoi(args.get("--tf-min", "5"));
    size_t skip_ticks = static_cast<size_t>(std::stoull(args.get("--skip-ticks", "0")));
    size_t max_ticks = static_cast<size_t>(std::stoull(args.get("--ticks", "50000000")));
    size_t chunk_size = static_cast<size_t>(std::stoull(args.get("--chunk-size", "1000000")));
    int64_t tf_ms = static_cast<int64_t>(tf_min) * 60LL * 1000LL;

    ParquetReader reader(data);
    size_t total_rows = reader.total_rows();
    skip_ticks = std::min(skip_ticks, total_rows);
    size_t eval_ticks = std::min(max_ticks, total_rows - skip_ticks);

    size_t skipped = 0;
    while (reader.has_next_chunk() && skipped < skip_ticks) {
        TickChunk t = reader.read_chunk(std::min(chunk_size, skip_ticks - skipped));
        skipped += t.size;
    }

    std::vector<Bar> bars;
    size_t read = 0;
    while (reader.has_next_chunk() && read < eval_ticks) {
        TickChunk t = reader.read_chunk(std::min(chunk_size, eval_ticks - read));
        if (t.size == 0) break;
        read += t.size;
        for (size_t i = 0; i < t.size; ++i) {
            int64_t bucket = (t.time_ms[i] / tf_ms) * tf_ms;
            double mid = t.mid[i] != 0.0 ? t.mid[i] : (t.ask[i] + t.bid[i]) * 0.5;
            add_tick(bars, bucket, mid, t.ask[i] - t.bid[i], t.ask_vol[i], t.bid_vol[i]);
        }
        if ((read / 10000000) != ((read - t.size) / 10000000)) {
            std::cout << "read " << (read / 1000000) << "M ticks bars=" << bars.size() << "\n";
        }
    }

    std::filesystem::create_directories(std::filesystem::path(out).parent_path());
    std::ofstream os(out);
    os << std::fixed << std::setprecision(8);
    os << "time_ms,open,high,low,close,spread,ask_vol,bid_vol,ticks\n";
    for (const Bar& b : bars) {
        double spread = b.ticks ? b.spread_sum / static_cast<double>(b.ticks) : 0.0;
        os << b.bucket << ',' << b.open << ',' << b.high << ',' << b.low << ',' << b.close << ','
           << spread << ',' << b.ask_vol_sum << ',' << b.bid_vol_sum << ',' << b.ticks << '\n';
    }

    std::cout << "EXPORT " << out << "\n";
    std::cout << "ticks_skipped=" << skipped << " ticks_read=" << read << " bars=" << bars.size() << "\n";
    if (!bars.empty()) {
        std::cout << "first_ms=" << bars.front().bucket << " last_ms=" << bars.back().bucket << "\n";
    }
    return 0;
}
