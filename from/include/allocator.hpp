#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace from {

class Arena {
    std::vector<uint8_t> buf_;
    size_t top_ = 0;

public:
    explicit Arena(size_t capacity_bytes) : buf_(capacity_bytes, 0) {}

    void* alloc(size_t bytes, size_t align = 16) {
        size_t base = reinterpret_cast<size_t>(buf_.data());
        size_t current = base + top_;
        size_t aligned = (current + align - 1) & ~(align - 1);
        size_t next = aligned - base + bytes;
        require(next <= buf_.size(), "Arena capacity exceeded");
        top_ = next;
        return reinterpret_cast<void*>(aligned);
    }

    void reset() { top_ = 0; }
    size_t used() const { return top_; }
    size_t capacity() const { return buf_.size(); }
};

}  // namespace from

