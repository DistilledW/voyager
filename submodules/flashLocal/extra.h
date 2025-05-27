#pragma once

#include "types.h"
#include <cstdint>

struct SimpleNode {
    std::uint8_t depth = static_cast<std::uint8_t>(-1);
    std::uint8_t count_merged;
    std::uint16_t count_children;

    int parent = -1;
    int start;
    int start_children;

    __device__ __host__ auto is_leaf() const -> bool {
        return depth == 0;
    }

    __device__ __host__ auto is_root() const -> bool {
        return parent == -1;
    }

    __device__ __host__ operator Node() const {
        return {// clang-format off
            static_cast<int>(depth),
            static_cast<int>(parent),
            start,
            static_cast<int>(is_leaf()),
            static_cast<int>(count_merged),
            start_children,
            static_cast<int>(count_children)
        };
        // clang-format on
    }
};

inline __device__ __host__ Node::operator SimpleNode() const {
    return {
        static_cast<std::uint8_t>(depth),
        static_cast<std::uint8_t>(count_merged),
        static_cast<std::uint16_t>(count_children),
        parent,
        start,
        start_children
    };
}

namespace dark {

auto reorder_nodes(int N, int *, float *, bool) -> void;
auto reorder_result(int N, int *) -> void;

} // namespace dark
