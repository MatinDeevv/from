#include "data/parquet_reader.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace from {
namespace {

enum CompactType {
    CT_STOP = 0,
    CT_BOOL_TRUE = 1,
    CT_BOOL_FALSE = 2,
    CT_BYTE = 3,
    CT_I16 = 4,
    CT_I32 = 5,
    CT_I64 = 6,
    CT_DOUBLE = 7,
    CT_BINARY = 8,
    CT_LIST = 9,
    CT_SET = 10,
    CT_MAP = 11,
    CT_STRUCT = 12
};

struct CompactReader {
    const std::vector<uint8_t>& data;
    size_t pos = 0;

    explicit CompactReader(const std::vector<uint8_t>& bytes, size_t start = 0) : data(bytes), pos(start) {}

    uint8_t byte() {
        require(pos < data.size(), "Parquet metadata read past end");
        return data[pos++];
    }

    uint64_t varint() {
        uint64_t out = 0;
        uint32_t shift = 0;
        while (shift < 64) {
            uint8_t b = byte();
            out |= static_cast<uint64_t>(b & 0x7fU) << shift;
            if ((b & 0x80U) == 0) return out;
            shift += 7;
        }
        throw std::runtime_error("Invalid Parquet varint");
    }

    int64_t zigzag() {
        uint64_t n = varint();
        return static_cast<int64_t>((n >> 1U) ^ (~(n & 1U) + 1U));
    }

    std::string binary() {
        size_t n = static_cast<size_t>(varint());
        require(pos + n <= data.size(), "Parquet binary exceeds metadata");
        std::string s(reinterpret_cast<const char*>(data.data() + pos), n);
        pos += n;
        return s;
    }
};

struct Field {
    int16_t id = 0;
    uint8_t type = 0;
};

Field next_field(CompactReader& r, int16_t& prev_id) {
    uint8_t h = r.byte();
    uint8_t type = h & 0x0fU;
    if (type == CT_STOP) return {};
    uint8_t delta = h >> 4U;
    int16_t id = delta ? static_cast<int16_t>(prev_id + delta) : static_cast<int16_t>(r.zigzag());
    prev_id = id;
    return {id, type};
}

void skip_value(CompactReader& r, uint8_t type);

void skip_struct(CompactReader& r) {
    int16_t prev = 0;
    while (true) {
        Field f = next_field(r, prev);
        if (f.type == CT_STOP) return;
        skip_value(r, f.type);
    }
}

void skip_value(CompactReader& r, uint8_t type) {
    switch (type) {
        case CT_BOOL_TRUE:
        case CT_BOOL_FALSE:
            return;
        case CT_BYTE:
            (void)r.byte();
            return;
        case CT_I16:
        case CT_I32:
        case CT_I64:
            (void)r.zigzag();
            return;
        case CT_DOUBLE:
            require(r.pos + 8 <= r.data.size(), "Parquet double exceeds metadata");
            r.pos += 8;
            return;
        case CT_BINARY:
            (void)r.binary();
            return;
        case CT_LIST:
        case CT_SET: {
            uint8_t h = r.byte();
            size_t size = h >> 4U;
            uint8_t elem = h & 0x0fU;
            if (size == 15) size = static_cast<size_t>(r.varint());
            for (size_t i = 0; i < size; ++i) skip_value(r, elem);
            return;
        }
        case CT_MAP: {
            size_t size = static_cast<size_t>(r.varint());
            if (size == 0) return;
            uint8_t h = r.byte();
            uint8_t kt = h >> 4U;
            uint8_t vt = h & 0x0fU;
            for (size_t i = 0; i < size; ++i) {
                skip_value(r, kt);
                skip_value(r, vt);
            }
            return;
        }
        case CT_STRUCT:
            skip_struct(r);
            return;
        default:
            throw std::runtime_error("Unknown Parquet compact type");
    }
}

std::vector<std::string> read_string_list(CompactReader& r) {
    uint8_t h = r.byte();
    size_t size = h >> 4U;
    uint8_t elem = h & 0x0fU;
    if (size == 15) size = static_cast<size_t>(r.varint());
    require(elem == CT_BINARY, "Expected Parquet string list");
    std::vector<std::string> out;
    for (size_t i = 0; i < size; ++i) out.push_back(r.binary());
    return out;
}

std::vector<int> read_i32_list(CompactReader& r) {
    uint8_t h = r.byte();
    size_t size = h >> 4U;
    uint8_t elem = h & 0x0fU;
    if (size == 15) size = static_cast<size_t>(r.varint());
    require(elem == CT_I32, "Expected Parquet i32 list");
    std::vector<int> out;
    for (size_t i = 0; i < size; ++i) out.push_back(static_cast<int>(r.zigzag()));
    return out;
}

struct ColumnMetaTmp {
    int type = 0;
    int codec = 0;
    uint64_t num_values = 0;
    uint64_t total_uncompressed_size = 0;
    uint64_t total_compressed_size = 0;
    uint64_t data_page_offset = 0;
    uint64_t dictionary_page_offset = 0;
    std::string path;
};

ColumnMetaTmp parse_column_meta(CompactReader& r) {
    ColumnMetaTmp m;
    int16_t prev = 0;
    while (true) {
        Field f = next_field(r, prev);
        if (f.type == CT_STOP) return m;
        switch (f.id) {
            case 1:
                m.type = static_cast<int>(r.zigzag());
                break;
            case 2:
                (void)read_i32_list(r);
                break;
            case 3: {
                auto path = read_string_list(r);
                if (!path.empty()) m.path = path[0];
                break;
            }
            case 4:
                m.codec = static_cast<int>(r.zigzag());
                break;
            case 5:
                m.num_values = static_cast<uint64_t>(r.zigzag());
                break;
            case 6:
                m.total_uncompressed_size = static_cast<uint64_t>(r.zigzag());
                break;
            case 7:
                m.total_compressed_size = static_cast<uint64_t>(r.zigzag());
                break;
            case 9:
                m.data_page_offset = static_cast<uint64_t>(r.zigzag());
                break;
            case 11:
                m.dictionary_page_offset = static_cast<uint64_t>(r.zigzag());
                break;
            default:
                skip_value(r, f.type);
                break;
        }
    }
}

ColumnMetaTmp parse_column_chunk(CompactReader& r) {
    ColumnMetaTmp m;
    int16_t prev = 0;
    while (true) {
        Field f = next_field(r, prev);
        if (f.type == CT_STOP) return m;
        if (f.id == 3) {
            m = parse_column_meta(r);
        } else {
            skip_value(r, f.type);
        }
    }
}

struct RowGroupTmp {
    uint64_t num_rows = 0;
    std::vector<ColumnMetaTmp> columns;
};

RowGroupTmp parse_row_group(CompactReader& r) {
    RowGroupTmp rg;
    int16_t prev = 0;
    while (true) {
        Field f = next_field(r, prev);
        if (f.type == CT_STOP) return rg;
        if (f.id == 1) {
            uint8_t h = r.byte();
            size_t size = h >> 4U;
            uint8_t elem = h & 0x0fU;
            if (size == 15) size = static_cast<size_t>(r.varint());
            require(elem == CT_STRUCT, "Expected Parquet column chunk list");
            for (size_t i = 0; i < size; ++i) rg.columns.push_back(parse_column_chunk(r));
        } else if (f.id == 3) {
            rg.num_rows = static_cast<uint64_t>(r.zigzag());
        } else {
            skip_value(r, f.type);
        }
    }
}

struct PageHeaderInfo {
    int type = -1;
    int uncompressed_size = 0;
    int compressed_size = 0;
    int num_values = 0;
    int encoding = 0;
    int dictionary_values = 0;
};

PageHeaderInfo parse_page_header(const std::vector<uint8_t>& bytes, size_t start, size_t* header_size) {
    CompactReader r(bytes, start);
    PageHeaderInfo ph;
    int16_t prev = 0;
    while (true) {
        Field f = next_field(r, prev);
        if (f.type == CT_STOP) break;
        if (f.id == 1) ph.type = static_cast<int>(r.zigzag());
        else if (f.id == 2) ph.uncompressed_size = static_cast<int>(r.zigzag());
        else if (f.id == 3) ph.compressed_size = static_cast<int>(r.zigzag());
        else if (f.id == 5) {
            int16_t dp = 0;
            while (true) {
                Field df = next_field(r, dp);
                if (df.type == CT_STOP) break;
                if (df.id == 1) ph.num_values = static_cast<int>(r.zigzag());
                else if (df.id == 2) ph.encoding = static_cast<int>(r.zigzag());
                else skip_value(r, df.type);
            }
        } else if (f.id == 7) {
            int16_t dp = 0;
            while (true) {
                Field df = next_field(r, dp);
                if (df.type == CT_STOP) break;
                if (df.id == 1) ph.dictionary_values = static_cast<int>(r.zigzag());
                else if (df.id == 2) ph.encoding = static_cast<int>(r.zigzag());
                else skip_value(r, df.type);
            }
        } else {
            skip_value(r, f.type);
        }
    }
    *header_size = r.pos - start;
    return ph;
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8U) |
           (static_cast<uint32_t>(p[2]) << 16U) | (static_cast<uint32_t>(p[3]) << 24U);
}

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t lo = read_u32_le(p);
    uint64_t hi = read_u32_le(p + 4);
    return lo | (hi << 32U);
}

std::vector<uint8_t> snappy_decode_or_copy(const std::vector<uint8_t>& src, size_t pos, size_t compressed, size_t uncompressed) {
    require(pos + compressed <= src.size(), "Parquet page exceeds column chunk");
    std::vector<uint8_t> out(uncompressed);
    size_t out_len = out.size();
    if (from_snappy_uncompress(src.data() + pos, compressed, out.data(), &out_len)) {
        out.resize(out_len);
        return out;
    }
    if (compressed == uncompressed) {
        std::memcpy(out.data(), src.data() + pos, compressed);
        return out;
    }
    throw std::runtime_error("SNAPPY decompression failed for Parquet page");
}

struct BitReader {
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t bit = 0;

    uint32_t read_bits(int width) {
        uint32_t out = 0;
        for (int i = 0; i < width; ++i) {
            size_t byte = bit >> 3U;
            if (byte >= size) return out;
            uint8_t b = data[byte];
            out |= static_cast<uint32_t>((b >> (bit & 7U)) & 1U) << i;
            ++bit;
        }
        return out;
    }
};

uint64_t read_unsigned_varint(const uint8_t* data, size_t size, size_t& pos) {
    uint64_t out = 0;
    uint32_t shift = 0;
    while (pos < size && shift < 64) {
        uint8_t b = data[pos++];
        out |= static_cast<uint64_t>(b & 0x7fU) << shift;
        if ((b & 0x80U) == 0) return out;
        shift += 7;
    }
    throw std::runtime_error("Invalid RLE varint");
}

std::vector<uint32_t> decode_rle_bitpacked(const uint8_t* data, size_t size, int bit_width, size_t count) {
    std::vector<uint32_t> out;
    out.reserve(count);
    size_t pos = 0;
    size_t value_bytes = static_cast<size_t>((bit_width + 7) / 8);
    while (pos < size && out.size() < count) {
        uint64_t header = read_unsigned_varint(data, size, pos);
        if ((header & 1U) == 0) {
            size_t run = static_cast<size_t>(header >> 1U);
            uint32_t value = 0;
            for (size_t i = 0; i < value_bytes && pos < size; ++i) value |= static_cast<uint32_t>(data[pos++]) << (8U * i);
            for (size_t i = 0; i < run && out.size() < count; ++i) out.push_back(value);
        } else {
            size_t groups = static_cast<size_t>(header >> 1U);
            size_t bytes = (groups * 8U * static_cast<size_t>(bit_width) + 7U) / 8U;
            if (pos + bytes > size) {
                std::ostringstream os;
                os << "Parquet bit-packed run exceeds buffer: bit_width=" << bit_width
                   << " groups=" << groups << " bytes=" << bytes
                   << " pos=" << pos << " size=" << size << " produced=" << out.size()
                   << " target=" << count;
                throw std::runtime_error(os.str());
            }
            BitReader br{data + pos, bytes, 0};
            for (size_t i = 0; i < groups * 8U && out.size() < count; ++i) out.push_back(br.read_bits(bit_width));
            pos += bytes;
        }
    }
    return out;
}

template <class T>
std::vector<T> decode_column_values(const std::vector<uint8_t>& chunk, const ParquetReader::ColumnMeta& meta, uint64_t expected_rows) {
    require(meta.codec == 1, "Only SNAPPY-compressed Parquet columns are supported");
    if constexpr (std::is_same_v<T, double>) {
        require(meta.physical_type == 5, "Expected DOUBLE Parquet column");
    } else if constexpr (std::is_same_v<T, float>) {
        require(meta.physical_type == 4, "Expected FLOAT Parquet column");
    } else if constexpr (std::is_same_v<T, int64_t>) {
        require(meta.physical_type == 2, "Expected INT64 Parquet column");
    }
    size_t offset = 0;
    std::vector<T> dictionary;
    std::vector<T> values;
    values.reserve(static_cast<size_t>(expected_rows));
    while (offset < chunk.size() && values.size() < expected_rows) {
        size_t header_size = 0;
        PageHeaderInfo ph = parse_page_header(chunk, offset, &header_size);
        offset += header_size;
        std::vector<uint8_t> page = snappy_decode_or_copy(chunk, offset, static_cast<size_t>(ph.compressed_size),
                                                          static_cast<size_t>(ph.uncompressed_size));
        offset += static_cast<size_t>(ph.compressed_size);
        if (ph.type == 2) {
            require(ph.encoding == 0, "Only PLAIN dictionary pages are supported for this Parquet file");
            size_t n = static_cast<size_t>(ph.dictionary_values);
            require(page.size() >= n * sizeof(T), "Parquet dictionary page too small");
            dictionary.resize(n);
            for (size_t i = 0; i < n; ++i) {
                if constexpr (sizeof(T) == 8) {
                    uint64_t raw = read_u64_le(page.data() + i * sizeof(T));
                    std::memcpy(&dictionary[i], &raw, sizeof(T));
                } else {
                    uint32_t raw = read_u32_le(page.data() + i * sizeof(T));
                    std::memcpy(&dictionary[i], &raw, sizeof(T));
                }
            }
        } else if (ph.type == 0) {
            size_t p = 0;
            size_t num_values = static_cast<size_t>(ph.num_values);
            require(p + 4 <= page.size(), "Parquet definition-level length missing");
            uint32_t def_len = read_u32_le(page.data() + p);
            p += 4;
            require(p + def_len <= page.size(), "Parquet definition levels exceed page");
            std::vector<uint32_t> defs = decode_rle_bitpacked(page.data() + p, def_len, 1, num_values);
            p += def_len;
            size_t present = 0;
            for (uint32_t d : defs) if (d == 1) ++present;

            if (ph.encoding == 0) {
                require(p + present * sizeof(T) <= page.size(), "Parquet PLAIN data page too small");
                size_t raw_pos = p;
                for (size_t i = 0; i < defs.size() && values.size() < expected_rows; ++i) {
                    if (defs[i] == 0) {
                        values.push_back(T{});
                    } else {
                        T v{};
                        if constexpr (sizeof(T) == 8) {
                            uint64_t raw = read_u64_le(page.data() + raw_pos);
                            std::memcpy(&v, &raw, sizeof(T));
                        } else {
                            uint32_t raw = read_u32_le(page.data() + raw_pos);
                            std::memcpy(&v, &raw, sizeof(T));
                        }
                        raw_pos += sizeof(T);
                        values.push_back(v);
                    }
                }
            } else {
                require(ph.encoding == 8, "Only PLAIN and RLE_DICTIONARY data pages are supported");
                require(!dictionary.empty(), "Parquet dictionary data page arrived before dictionary page");
                require(p < page.size(), "Parquet dictionary index bit width missing");
                int bit_width = page[p++];
                std::vector<uint32_t> ids = decode_rle_bitpacked(page.data() + p, page.size() - p, bit_width, present);
                size_t id_pos = 0;
                for (size_t i = 0; i < defs.size() && values.size() < expected_rows; ++i) {
                    if (defs[i] == 0) {
                        values.push_back(T{});
                    } else {
                        require(id_pos < ids.size(), "Parquet dictionary id underrun");
                        uint32_t id = ids[id_pos++];
                        require(id < dictionary.size(), "Parquet dictionary id out of range");
                        values.push_back(dictionary[id]);
                    }
                }
            }
        }
    }
    require(values.size() == expected_rows, "Decoded Parquet column row count mismatch");
    return values;
}

}  // namespace

ParquetReader::ParquetReader(const std::string& path) : path_(path) {
    in_.open(path_, std::ios::binary);
    if (!in_) throw std::runtime_error("Cannot open data file: " + path_);
    char magic[4] = {};
    in_.read(magic, 4);
    parquet_magic_ = in_.gcount() == 4 && std::memcmp(magic, "PAR1", 4) == 0;
    require(parquet_magic_, "Input is not a Parquet file with PAR1 magic: " + path_);
    parse_footer();
    reset();
}

void ParquetReader::parse_footer() {
    uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(path_));
    require(file_size > 12, "Parquet file too small");
    in_.seekg(static_cast<std::streamoff>(file_size - 8), std::ios::beg);
    uint8_t tail[8] = {};
    in_.read(reinterpret_cast<char*>(tail), 8);
    require(std::memcmp(tail + 4, "PAR1", 4) == 0, "Invalid Parquet footer magic");
    uint32_t footer_size = read_u32_le(tail);
    require(footer_size < file_size - 8, "Invalid Parquet footer size");
    std::vector<uint8_t> footer(footer_size);
    in_.seekg(static_cast<std::streamoff>(file_size - 8 - footer_size), std::ios::beg);
    in_.read(reinterpret_cast<char*>(footer.data()), static_cast<std::streamsize>(footer.size()));
    require(static_cast<size_t>(in_.gcount()) == footer.size(), "Could not read Parquet footer");

    CompactReader r(footer);
    int16_t prev = 0;
    while (true) {
        Field f = next_field(r, prev);
        if (f.type == CT_STOP) break;
        if (f.id == 3) {
            total_rows_ = static_cast<size_t>(r.zigzag());
        } else if (f.id == 4) {
            uint8_t h = r.byte();
            size_t size = h >> 4U;
            uint8_t elem = h & 0x0fU;
            if (size == 15) size = static_cast<size_t>(r.varint());
            require(elem == CT_STRUCT, "Expected Parquet row group list");
            for (size_t i = 0; i < size; ++i) {
                RowGroupTmp tmp = parse_row_group(r);
                RowGroupMeta rg;
                rg.num_rows = tmp.num_rows;
                for (const auto& c : tmp.columns) {
                    ColumnMeta cm;
                    cm.physical_type = c.type;
                    cm.codec = c.codec;
                    cm.num_values = c.num_values;
                    cm.data_page_offset = c.data_page_offset;
                    cm.dictionary_page_offset = c.dictionary_page_offset;
                    cm.total_compressed_size = c.total_compressed_size;
                    rg.columns.push_back(cm);
                }
                require(rg.columns.size() == 6, "Parquet row group does not have the expected six columns");
                row_groups_.push_back(std::move(rg));
            }
        } else {
            skip_value(r, f.type);
        }
    }
    require(total_rows_ > 0 && !row_groups_.empty(), "Parquet footer has no rows or row groups");
}

TickChunk ParquetReader::decode_row_group(size_t row_group_index) {
    const RowGroupMeta& rg = row_groups_.at(row_group_index);
    TickChunk out;
    out.size = static_cast<size_t>(rg.num_rows);
    auto read_chunk_bytes = [&](const ColumnMeta& c) {
        uint64_t start = c.dictionary_page_offset ? c.dictionary_page_offset : c.data_page_offset;
        std::vector<uint8_t> bytes(static_cast<size_t>(c.total_compressed_size));
        in_.seekg(static_cast<std::streamoff>(start), std::ios::beg);
        in_.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<size_t>(in_.gcount()) == bytes.size(), "Could not read Parquet column chunk");
        return bytes;
    };

    auto ask_bytes = read_chunk_bytes(rg.columns[0]);
    auto bid_bytes = read_chunk_bytes(rg.columns[1]);
    auto mid_bytes = read_chunk_bytes(rg.columns[2]);
    auto ask_vol_bytes = read_chunk_bytes(rg.columns[3]);
    auto bid_vol_bytes = read_chunk_bytes(rg.columns[4]);
    auto time_bytes = read_chunk_bytes(rg.columns[5]);

    auto ask_future = std::async(std::launch::async, [&]() {
        return decode_column_values<double>(ask_bytes, rg.columns[0], rg.num_rows);
    });
    auto bid_future = std::async(std::launch::async, [&]() {
        return decode_column_values<double>(bid_bytes, rg.columns[1], rg.num_rows);
    });
    auto mid_future = std::async(std::launch::async, [&]() {
        return decode_column_values<double>(mid_bytes, rg.columns[2], rg.num_rows);
    });
    auto ask_vol_future = std::async(std::launch::async, [&]() {
        return decode_column_values<float>(ask_vol_bytes, rg.columns[3], rg.num_rows);
    });
    auto bid_vol_future = std::async(std::launch::async, [&]() {
        return decode_column_values<float>(bid_vol_bytes, rg.columns[4], rg.num_rows);
    });
    auto time_future = std::async(std::launch::async, [&]() {
        return decode_column_values<int64_t>(time_bytes, rg.columns[5], rg.num_rows);
    });

    out.ask = ask_future.get();
    out.bid = bid_future.get();
    out.mid = mid_future.get();
    out.ask_vol = ask_vol_future.get();
    out.bid_vol = bid_vol_future.get();
    out.time_ms = time_future.get();
    return out;
}

bool ParquetReader::has_next_chunk() const {
    return rows_read_ < total_rows_;
}

TickChunk ParquetReader::read_chunk(size_t chunk_size) {
    TickChunk out;
    size_t target = std::min(chunk_size, total_rows_ - rows_read_);
    out.ask.reserve(target);
    out.bid.reserve(target);
    out.mid.reserve(target);
    out.ask_vol.reserve(target);
    out.bid_vol.reserve(target);
    out.time_ms.reserve(target);
    while (out.size < target && has_next_chunk()) {
        if (decoded_pos_ >= decoded_group_.size) {
            require(current_row_group_ < row_groups_.size(), "Parquet row group cursor exceeded metadata");
            decoded_group_ = decode_row_group(current_row_group_++);
            decoded_pos_ = 0;
        }
        size_t take = std::min(target - out.size, decoded_group_.size - decoded_pos_);
        if (out.size == 0 && decoded_pos_ == 0 && take == decoded_group_.size && take == target) {
            out = std::move(decoded_group_);
            decoded_pos_ = out.size;
            rows_read_ += out.size;
            return out;
        }
        auto append = [&](auto& dst, const auto& src) {
            dst.insert(dst.end(), src.begin() + static_cast<std::ptrdiff_t>(decoded_pos_),
                       src.begin() + static_cast<std::ptrdiff_t>(decoded_pos_ + take));
        };
        append(out.ask, decoded_group_.ask);
        append(out.bid, decoded_group_.bid);
        append(out.mid, decoded_group_.mid);
        append(out.ask_vol, decoded_group_.ask_vol);
        append(out.bid_vol, decoded_group_.bid_vol);
        append(out.time_ms, decoded_group_.time_ms);
        decoded_pos_ += take;
        out.size += take;
        rows_read_ += take;
    }
    return out;
}

size_t ParquetReader::total_rows() const { return total_rows_; }
size_t ParquetReader::rows_read() const { return rows_read_; }

void ParquetReader::reset() {
    rows_read_ = 0;
    current_row_group_ = 0;
    decoded_group_ = TickChunk{};
    decoded_pos_ = 0;
    in_.clear();
}

}  // namespace from
