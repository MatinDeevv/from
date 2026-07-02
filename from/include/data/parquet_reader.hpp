#pragma once

#include "common.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace from {

struct TickChunk {
    std::vector<double> ask;
    std::vector<double> bid;
    std::vector<double> mid;
    std::vector<float> ask_vol;
    std::vector<float> bid_vol;
    std::vector<int64_t> time_ms;
    size_t size = 0;
};

class ParquetReader {
public:
    struct ColumnMeta {
        int physical_type = 0;
        int codec = 0;
        uint64_t num_values = 0;
        uint64_t data_page_offset = 0;
        uint64_t dictionary_page_offset = 0;
        uint64_t total_compressed_size = 0;
    };

    struct RowGroupMeta {
        uint64_t num_rows = 0;
        std::vector<ColumnMeta> columns;
    };

private:
    std::string path_;
    std::ifstream in_;
    size_t total_rows_ = 0;
    size_t rows_read_ = 0;
    bool parquet_magic_ = false;
    std::vector<RowGroupMeta> row_groups_;
    size_t current_row_group_ = 0;
    TickChunk decoded_group_;
    size_t decoded_pos_ = 0;
    int64_t last_timestamp_ms_ = -1;

    void parse_footer();
    TickChunk decode_row_group(size_t row_group_index);
    void validate_tick_chunk(const TickChunk& chunk);

public:
    explicit ParquetReader(const std::string& path);
    bool has_next_chunk() const;
    TickChunk read_chunk(size_t chunk_size = 500000);
    size_t total_rows() const;
    size_t rows_read() const;
    void reset();
    bool has_parquet_magic() const { return parquet_magic_; }
};

}  // namespace from

