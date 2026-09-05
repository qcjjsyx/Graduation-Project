#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "model_types.hpp"

namespace hw {

inline std::int64_t abs_i64(std::int64_t value) {
    if (value == std::numeric_limits<std::int64_t>::min()) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return value < 0 ? -value : value;
}

inline std::int64_t round_shift_signed(std::int64_t value, int shift) {
    if (shift == 0) {
        return value;
    }
    if (shift < 0) {
        const unsigned left = static_cast<unsigned>(-shift);
        if (left >= 63) {
            throw std::overflow_error("left shift exceeds int64 width");
        }
        const auto factor = std::int64_t{1} << left;
        if (value > std::numeric_limits<std::int64_t>::max() / factor ||
            value < std::numeric_limits<std::int64_t>::min() / factor) {
            throw std::overflow_error("left shift overflows int64");
        }
        // Signed left shift of a negative value is undefined in C++. Checked
        // multiplication has the same arithmetic meaning without invoking UB.
        return value * factor;
    }

    if (shift >= 65) {
        return 0;
    }
    const auto extended = static_cast<__int128>(value);
    const auto magnitude = value < 0 ? -extended : extended;
    const auto offset = __int128{1} << (shift - 1);
    const auto rounded = (magnitude + offset) >> shift;
    return static_cast<std::int64_t>(value < 0 ? -rounded : rounded);
}

// Integer equivalent of ceil(log2(max_abs / q_limit)).
inline int scale_exponent_delta(std::int64_t max_abs, std::int64_t q_limit) {
    if (max_abs <= 0) {
        return 0;
    }
    if (q_limit <= 0) {
        throw std::invalid_argument("q_limit must be positive");
    }

    int delta = 0;
    std::int64_t scaled = max_abs;
    while (scaled > q_limit) {
        scaled = (scaled >> 1) + (scaled & 1);
        ++delta;
    }
    while (scaled <= q_limit / 2 && scaled <= std::numeric_limits<std::int64_t>::max() / 2) {
        scaled <<= 1;
        --delta;
    }
    return delta;
}

inline std::pair<std::vector<std::int32_t>, QuantStats> requantize(
    const std::vector<std::int64_t>& accumulator,
    std::int16_t assembly_exp,
    std::int32_t q_limit,
    QuantStats stats) {
    std::int64_t max_abs = 0;
    for (const auto value : accumulator) {
        max_abs = std::max(max_abs, abs_i64(value));
    }
    stats.max_abs_acc = max_abs;
    const int delta = scale_exponent_delta(max_abs, q_limit);
    stats.assembly_exp = assembly_exp;
    const auto node_exp = static_cast<int>(assembly_exp) + delta;
    if (node_exp < std::numeric_limits<std::int16_t>::min() ||
        node_exp > std::numeric_limits<std::int16_t>::max()) {
        throw std::overflow_error("node exponent exceeds int16 range");
    }
    stats.node_exp = static_cast<std::int16_t>(node_exp);

    std::vector<std::int32_t> output;
    output.reserve(accumulator.size());
    for (const auto value : accumulator) {
        auto shifted = round_shift_signed(value, delta);
        if (shifted > q_limit) {
            shifted = q_limit;
        } else if (shifted < -q_limit) {
            shifted = -q_limit;
        }
        if (shifted == q_limit || shifted == -q_limit) {
            ++stats.saturation_count;
        }
        output.push_back(static_cast<std::int32_t>(shifted));
    }
    return {output, stats};
}

}  // namespace hw
