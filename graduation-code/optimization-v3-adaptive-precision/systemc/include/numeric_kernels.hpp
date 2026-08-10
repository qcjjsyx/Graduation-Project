#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atu.hpp"
#include "model_types.hpp"
#include "quantization.hpp"
#include "system_memory.hpp"

namespace hw {

class NumericFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PrecisionRescueRequired : public NumericFailure {
public:
    using NumericFailure::NumericFailure;
};

class FactorCheckFailure : public PrecisionRescueRequired {
public:
    using PrecisionRescueRequired::PrecisionRescueRequired;
};

struct PivotCandidate {
    std::uint16_t row{0};
    std::int32_t value{0};
};

struct FixedComputation {
    FixedFactor factor{};
    std::vector<std::vector<PivotCandidate>> candidates{};
    std::vector<std::uint16_t> selected_rows{};
    unsigned swap_count{0};
    std::uint64_t matrix_overflow_count{0};
    std::uint64_t workspace_renormalize_count{0};
    std::uint64_t small_pivot_count{0};
    double min_pivot_ratio{1.0};
    double max_growth_ratio{1.0};
    bool precision_rescued{false};
};

struct Fp64Computation {
    Fp64Factor factor{};
    std::vector<std::uint16_t> selected_rows{};
    unsigned swap_count{0};
};

inline std::int64_t round_div_signed(
    std::int64_t numerator,
    std::int64_t denominator) {
    if (denominator == 0) {
        throw NumericFailure("integer divider received zero denominator");
    }
    const bool negative = (numerator < 0) != (denominator < 0);
    const auto numerator_abs = static_cast<std::uint64_t>(
        numerator < 0 ? -static_cast<__int128>(numerator) : numerator);
    const auto denominator_abs = static_cast<std::uint64_t>(
        denominator < 0 ? -static_cast<__int128>(denominator) : denominator);
    const auto quotient =
        (numerator_abs + denominator_abs / 2) / denominator_abs;
    if (quotient > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max())) {
        throw NumericFailure("rounded integer division overflows int64");
    }
    const auto signed_quotient = static_cast<std::int64_t>(quotient);
    return negative ? -signed_quotient : signed_quotient;
}

inline __int128 round_div_signed_wide(
    __int128 numerator,
    std::int64_t denominator) {
    if (denominator == 0) {
        throw NumericFailure("integer divider received zero denominator");
    }
    const bool negative = (numerator < 0) != (denominator < 0);
    const auto numerator_abs = numerator < 0 ? -numerator : numerator;
    const auto denominator_abs =
        denominator < 0 ? -static_cast<__int128>(denominator) : denominator;
    const auto quotient =
        (numerator_abs + denominator_abs / 2) / denominator_abs;
    return negative ? -quotient : quotient;
}

inline std::int64_t clamp_matrix_accumulator(
    __int128 value,
    unsigned accumulator_bits,
    std::uint64_t& overflow_count);

inline std::int64_t round_shift_wide(
    __int128 value,
    unsigned shift,
    unsigned accumulator_bits,
    std::uint64_t& overflow_count) {
    __int128 rounded = value;
    if (shift != 0) {
        const auto magnitude = value < 0 ? -value : value;
        rounded =
            (magnitude + (__int128{1} << (shift - 1))) >> shift;
        if (value < 0) rounded = -rounded;
    }
    return clamp_matrix_accumulator(
        rounded, accumulator_bits, overflow_count);
}

inline std::int32_t saturate_i32(
    __int128 value,
    std::uint64_t& overflow_count) {
    if (value > std::numeric_limits<std::int32_t>::max()) {
        ++overflow_count;
        return std::numeric_limits<std::int32_t>::max();
    }
    if (value < std::numeric_limits<std::int32_t>::min()) {
        ++overflow_count;
        return std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(value);
}

inline std::int64_t clamp_matrix_accumulator(
    __int128 value,
    unsigned accumulator_bits,
    std::uint64_t& overflow_count) {
    const __int128 maximum =
        accumulator_bits == 64 ?
            std::numeric_limits<std::int64_t>::max() :
            ((__int128{1} << (accumulator_bits - 1)) - 1);
    const __int128 minimum =
        accumulator_bits == 64 ?
            std::numeric_limits<std::int64_t>::min() :
            -(__int128{1} << (accumulator_bits - 1));
    if (value > maximum) {
        ++overflow_count;
        return static_cast<std::int64_t>(maximum);
    }
    if (value < minimum) {
        ++overflow_count;
        return static_cast<std::int64_t>(minimum);
    }
    return static_cast<std::int64_t>(value);
}

inline unsigned magnitude_bits(std::int64_t value) {
    auto magnitude = static_cast<std::uint64_t>(abs_i64(value));
    unsigned bits = 0;
    while (magnitude != 0) {
        ++bits;
        magnitude >>= 1;
    }
    return bits;
}

inline unsigned magnitude_bits_i128(__int128 value) {
    auto magnitude =
        static_cast<unsigned __int128>(value < 0 ? -value : value);
    unsigned bits = 0;
    while (magnitude != 0) {
        ++bits;
        magnitude >>= 1;
    }
    return bits;
}

inline std::vector<PivotCandidate> normalize_pivot_candidates(
    const std::vector<std::pair<std::uint16_t, std::int64_t>>& wide,
    std::uint64_t& renormalize_count) {
    std::int64_t max_abs = 0;
    for (const auto& [row, value] : wide) {
        (void)row;
        max_abs = std::max(max_abs, abs_i64(value));
    }
    if (max_abs == 0) {
        return std::vector<PivotCandidate>(
            wide.size(), PivotCandidate{});
    }
    constexpr unsigned target_bits = 30;
    const int shift =
        static_cast<int>(magnitude_bits(max_abs)) -
        static_cast<int>(target_bits);
    if (shift != 0) ++renormalize_count;

    std::vector<PivotCandidate> result;
    result.reserve(wide.size());
    for (const auto& [row, value] : wide) {
        __int128 normalized = value;
        if (shift > 0) {
            const auto magnitude =
                normalized < 0 ? -normalized : normalized;
            normalized =
                (magnitude + (__int128{1} << (shift - 1))) >> shift;
            if (value < 0) normalized = -normalized;
        } else if (shift < 0) {
            normalized *= (__int128{1} << static_cast<unsigned>(-shift));
        }
        normalized = std::clamp(
            normalized,
            static_cast<__int128>(std::numeric_limits<std::int32_t>::min()),
            static_cast<__int128>(std::numeric_limits<std::int32_t>::max()));
        result.push_back({
            row,
            static_cast<std::int32_t>(normalized),
        });
    }
    return result;
}

inline std::pair<std::vector<std::int32_t>, std::int16_t>
quantize_wide_block(
    const std::vector<std::int64_t>& values,
    std::int16_t base_exponent,
    const ModelConfig& config,
    std::uint64_t& overflow_count) {
    if (values.empty()) return {{}, base_exponent};
    if (config.adaptive_factor_scaling) {
        QuantStats stats{};
        auto result =
            requantize(values, base_exponent, config.q_limit(), stats);
        overflow_count += stats.saturation_count;
        return {std::move(result.first), result.second.node_exp};
    }
    std::vector<std::int32_t> output;
    output.reserve(values.size());
    for (const auto value : values) {
        output.push_back(
            saturate_i32(value, overflow_count));
    }
    return {std::move(output), base_exponent};
}

inline std::size_t bfp_tile_index(
    std::uint32_t row,
    std::uint32_t col,
    std::uint32_t cols,
    unsigned tile_size) {
    const auto tile_cols = (cols + tile_size - 1) / tile_size;
    return static_cast<std::size_t>(row / tile_size) * tile_cols +
           col / tile_size;
}

inline std::int64_t round_shift_i128(
    __int128 value,
    unsigned shift,
    unsigned accumulator_bits,
    std::uint64_t& overflow_count) {
    if (shift >= 128) return 0;
    if (shift == 0) {
        return clamp_matrix_accumulator(
            value, accumulator_bits, overflow_count);
    }
    const auto magnitude = value < 0 ? -value : value;
    auto rounded =
        (magnitude + (__int128{1} << (shift - 1))) >> shift;
    if (value < 0) rounded = -rounded;
    return clamp_matrix_accumulator(
        rounded, accumulator_bits, overflow_count);
}

inline std::int32_t bfp_divide_qf(
    std::int64_t value_mantissa,
    std::int16_t value_exponent,
    std::int64_t pivot_mantissa,
    std::int16_t pivot_exponent,
    const ModelConfig& config,
    std::uint64_t& overflow_count) {
    const int shift =
        static_cast<int>(config.frac_bits) +
        static_cast<int>(value_exponent) -
        static_cast<int>(pivot_exponent);
    __int128 numerator = value_mantissa;
    __int128 denominator = pivot_mantissa;
    if (shift >= 0) {
        if (shift >= 126 ||
            magnitude_bits(value_mantissa) +
                    static_cast<unsigned>(shift) >= 126) {
            throw PrecisionRescueRequired(
                "tile-BFP L division exceeds int128 range");
        }
        numerator *= (__int128{1} << static_cast<unsigned>(shift));
    } else {
        const auto right = static_cast<unsigned>(-shift);
        if (right >= 126 ||
            magnitude_bits(pivot_mantissa) + right >= 126) {
            return 0;
        }
        denominator *= (__int128{1} << right);
    }
    if (denominator == 0) {
        throw NumericFailure("tile-BFP divider received zero pivot");
    }
    const bool negative = (numerator < 0) != (denominator < 0);
    const auto numerator_abs = numerator < 0 ? -numerator : numerator;
    const auto denominator_abs =
        denominator < 0 ? -denominator : denominator;
    auto quotient =
        (numerator_abs + denominator_abs / 2) / denominator_abs;
    if (negative) quotient = -quotient;
    return saturate_i32(quotient, overflow_count);
}

inline void quantize_bfp_pairs(
    const std::vector<std::int64_t>& mantissas,
    const std::vector<std::int16_t>& exponents,
    std::uint32_t rows,
    std::uint32_t cols,
    const ModelConfig& config,
    std::vector<std::int32_t>& output,
    std::vector<std::int16_t>& output_exponents,
    std::uint64_t& overflow_count) {
    if (mantissas.size() != static_cast<std::size_t>(rows) * cols ||
        exponents.size() != mantissas.size()) {
        throw NumericFailure("tile-BFP output pair dimensions mismatch");
    }
    output.assign(mantissas.size(), 0);
    if (rows == 0 || cols == 0) {
        output_exponents.clear();
        return;
    }
    const auto tile = config.bfp_tile_size;
    const auto tile_rows = (rows + tile - 1) / tile;
    const auto tile_cols = (cols + tile - 1) / tile;
    output_exponents.assign(
        static_cast<std::size_t>(tile_rows) * tile_cols, 0);
    for (std::uint32_t tr = 0; tr < tile_rows; ++tr) {
        const auto row_end = std::min((tr + 1) * tile, rows);
        for (std::uint32_t tc = 0; tc < tile_cols; ++tc) {
            const auto col_end = std::min((tc + 1) * tile, cols);
            long double max_abs = 0.0L;
            for (auto row = tr * tile; row < row_end; ++row) {
                for (auto col = tc * tile; col < col_end; ++col) {
                    const auto index =
                        static_cast<std::size_t>(row) * cols + col;
                    const auto value = std::ldexp(
                        static_cast<long double>(
                            abs_i64(mantissas[index])),
                        exponents[index]);
                    if (!std::isfinite(value)) {
                        throw PrecisionRescueRequired(
                            "tile-BFP factor magnitude is not finite");
                    }
                    max_abs = std::max(max_abs, value);
                }
            }
            int result_exponent = 0;
            if (max_abs != 0.0L) {
                result_exponent = static_cast<int>(std::ceil(std::log2(
                    max_abs /
                    static_cast<long double>(config.q_limit()))));
            }
            if (result_exponent <
                    std::numeric_limits<std::int16_t>::min() ||
                result_exponent >
                    std::numeric_limits<std::int16_t>::max()) {
                throw NumericFailure(
                    "tile-BFP factor exponent exceeds int16");
            }
            const auto tile_id =
                static_cast<std::size_t>(tr) * tile_cols + tc;
            output_exponents[tile_id] =
                static_cast<std::int16_t>(result_exponent);
            for (auto row = tr * tile; row < row_end; ++row) {
                for (auto col = tc * tile; col < col_end; ++col) {
                    const auto index =
                        static_cast<std::size_t>(row) * cols + col;
                    const int shift =
                        result_exponent -
                        static_cast<int>(exponents[index]);
                    std::int64_t value = 0;
                    try {
                        value = round_shift_signed(
                            mantissas[index], shift);
                    } catch (const std::overflow_error&) {
                        ++overflow_count;
                        throw PrecisionRescueRequired(
                            "tile-BFP factor requantization overflow");
                    }
                    if (value > config.q_limit()) {
                        value = config.q_limit();
                        ++overflow_count;
                    } else if (value < -config.q_limit()) {
                        value = -config.q_limit();
                        ++overflow_count;
                    }
                    output[index] = static_cast<std::int32_t>(value);
                }
            }
        }
    }
}

inline FixedComputation factor_fixed_front_tile_bfp(
    const std::vector<std::int32_t>& assembled,
    const std::vector<std::int16_t>& assembled_tile_exponents,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    const ModelConfig& config) {
    if (config.bfp_tile_size != 16 || pivot_dim == 0 ||
        pivot_dim > total_dim || pivot_dim > MAX_ROWS ||
        assembled.size() != static_cast<std::size_t>(total_dim) * total_dim) {
        throw NumericFailure("invalid tile-BFP fixed front dimensions");
    }
    const auto tile = config.bfp_tile_size;
    const auto tile_extent = (total_dim + tile - 1) / tile;
    if (assembled_tile_exponents.size() !=
        static_cast<std::size_t>(tile_extent) * tile_extent) {
        throw NumericFailure("assembled tile exponent count mismatch");
    }

    FixedComputation output{};
    std::vector<std::int64_t> physical(assembled.size(), 0);
    std::vector<std::int16_t> workspace_exponents =
        assembled_tile_exponents;
    for (auto& exponent : workspace_exponents) {
        const auto guarded =
            static_cast<int>(exponent) -
            static_cast<int>(config.workspace_guard_bits);
        if (guarded < std::numeric_limits<std::int16_t>::min()) {
            throw NumericFailure(
                "tile-BFP guard exponent exceeds int16");
        }
        exponent = static_cast<std::int16_t>(guarded);
    }
    long double initial_max_abs = 0.0L;
    long double workspace_max_abs = 0.0L;
    for (std::uint32_t row = 0; row < total_dim; ++row) {
        for (std::uint32_t col = 0; col < total_dim; ++col) {
            const auto index =
                static_cast<std::size_t>(row) * total_dim + col;
            physical[index] = clamp_matrix_accumulator(
                static_cast<__int128>(assembled[index]) *
                    (__int128{1} << config.workspace_guard_bits),
                config.accumulator_bits,
                output.matrix_overflow_count);
            const auto real_abs = std::ldexp(
                static_cast<long double>(abs_i64(physical[index])),
                workspace_exponents[
                    bfp_tile_index(row, col, total_dim, tile)]);
            initial_max_abs = std::max(initial_max_abs, real_abs);
            workspace_max_abs = std::max(workspace_max_abs, real_abs);
        }
    }
    std::vector<std::int32_t> l_physical(
        static_cast<std::size_t>(total_dim) * pivot_dim, 0);
    std::vector<std::uint16_t> pvec(pivot_dim);
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        pvec[row] = static_cast<std::uint16_t>(row);
    }
    const auto physical_row = [&](std::uint32_t logical) {
        return logical < pivot_dim ?
            static_cast<std::uint32_t>(pvec[logical]) : logical;
    };
    const auto exponent_at =
        [&](std::uint32_t physical_r, std::uint32_t col) {
            return workspace_exponents[
                bfp_tile_index(physical_r, col, total_dim, tile)];
        };
    const auto real_abs =
        [&](std::int64_t mantissa, std::int16_t exponent) {
            return std::ldexp(
                static_cast<long double>(abs_i64(mantissa)), exponent);
        };

    for (std::uint32_t column = 0; column < pivot_dim; ++column) {
        std::uint32_t best_row = column;
        long double best_abs = -1.0L;
        for (std::uint32_t row = column; row < pivot_dim; ++row) {
            const auto physical_r = physical_row(row);
            const auto magnitude = real_abs(
                physical[physical_r * total_dim + column],
                exponent_at(physical_r, column));
            if (magnitude > best_abs) {
                best_abs = magnitude;
                best_row = row;
            }
        }
        if (!(best_abs > 0.0L) || !std::isfinite(best_abs)) {
            throw PrecisionRescueRequired(
                "tile-BFP pivot column contains only zeros");
        }
        const auto pivot_ratio =
            initial_max_abs == 0.0L ? 0.0 :
            static_cast<double>(best_abs / initial_max_abs);
        output.min_pivot_ratio =
            std::min(output.min_pivot_ratio, pivot_ratio);
        if (pivot_ratio <= config.fixed_pivot_rel_tol) {
            ++output.small_pivot_count;
            throw PrecisionRescueRequired(
                "tile-BFP pivot below precision threshold in column " +
                std::to_string(column));
        }

        // HPU observes a normalized int32 candidate vector even though the
        // comparison above is exponent-aware.
        std::vector<PivotCandidate> candidates;
        candidates.reserve(pivot_dim - column);
        for (std::uint32_t row = column; row < pivot_dim; ++row) {
            const auto physical_r = physical_row(row);
            const auto value =
                physical[physical_r * total_dim + column];
            const auto normalized = std::nearbyint(
                std::ldexp(
                    static_cast<long double>(value),
                    exponent_at(physical_r, column)) /
                best_abs *
                static_cast<long double>(
                    std::numeric_limits<std::int32_t>::max()));
            candidates.push_back({
                static_cast<std::uint16_t>(row),
                static_cast<std::int32_t>(std::clamp(
                    normalized,
                    static_cast<long double>(
                        std::numeric_limits<std::int32_t>::min()),
                    static_cast<long double>(
                        std::numeric_limits<std::int32_t>::max()))),
            });
        }
        output.candidates.push_back(std::move(candidates));
        output.selected_rows.push_back(
            static_cast<std::uint16_t>(best_row));
        if (best_row != column) {
            std::swap(pvec[column], pvec[best_row]);
            ++output.swap_count;
        }

        const auto pivot_physical = physical_row(column);
        const auto pivot =
            physical[pivot_physical * total_dim + column];
        const auto pivot_exponent =
            exponent_at(pivot_physical, column);
        if (pivot == 0) {
            throw NumericFailure("tile-BFP pivot became zero");
        }
        for (std::uint32_t logical_row = column + 1;
             logical_row < total_dim; ++logical_row) {
            const auto row_physical = physical_row(logical_row);
            const auto multiplier = bfp_divide_qf(
                physical[row_physical * total_dim + column],
                exponent_at(row_physical, column),
                pivot,
                pivot_exponent,
                config,
                output.matrix_overflow_count);
            l_physical[row_physical * pivot_dim + column] =
                multiplier;

            for (std::uint32_t target_col = column + 1;
                 target_col < total_dim; ++target_col) {
                const auto source_value =
                    physical[pivot_physical * total_dim + target_col];
                if (multiplier == 0 || source_value == 0) {
                    continue;
                }
                const auto destination_tile = bfp_tile_index(
                    row_physical, target_col, total_dim, tile);
                const auto source_tile = bfp_tile_index(
                    pivot_physical, target_col, total_dim, tile);
                const auto product_exponent =
                    static_cast<int>(workspace_exponents[source_tile]) -
                    static_cast<int>(config.frac_bits);
                const auto product =
                    static_cast<__int128>(multiplier) *
                    source_value;
                int exponent_difference =
                    product_exponent -
                    static_cast<int>(
                        workspace_exponents[destination_tile]);
                unsigned product_left_shift = 0;
                if (exponent_difference > 0) {
                    const auto product_bits =
                        magnitude_bits_i128(product);
                    const auto safe_bits =
                        config.accumulator_bits > 2 ?
                        config.accumulator_bits - 2 : 0;
                    const auto available_left =
                        product_bits < safe_bits ?
                        safe_bits - product_bits : 0;
                    product_left_shift = std::min(
                        static_cast<unsigned>(exponent_difference),
                        available_left);
                    const auto tile_shift =
                        static_cast<unsigned>(exponent_difference) -
                        product_left_shift;
                    if (tile_shift != 0) {
                        const auto tile_row =
                            (row_physical / tile) * tile;
                        const auto tile_col =
                            (target_col / tile) * tile;
                        for (auto row = tile_row;
                             row < std::min(
                                 tile_row + tile, total_dim);
                             ++row) {
                            for (auto col = tile_col;
                                 col < std::min(
                                     tile_col + tile, total_dim);
                                 ++col) {
                                const auto index =
                                    static_cast<std::size_t>(row) *
                                        total_dim + col;
                                const auto before = physical[index];
                                physical[index] =
                                    round_shift_signed(
                                        before, tile_shift);
                                if (before != 0 &&
                                    physical[index] == 0) {
                                    ++output
                                        .workspace_renormalize_count;
                                }
                            }
                        }
                        workspace_exponents[destination_tile] =
                            static_cast<std::int16_t>(
                                static_cast<int>(
                                    workspace_exponents[
                                        destination_tile]) +
                                static_cast<int>(tile_shift));
                        ++output.workspace_renormalize_count;
                    }
                }
                const auto before_overflow =
                    output.matrix_overflow_count;
                std::int64_t delta = 0;
                exponent_difference =
                    product_exponent -
                    static_cast<int>(
                        workspace_exponents[destination_tile]);
                if (exponent_difference > 0) {
                    delta = clamp_matrix_accumulator(
                        product *
                            (__int128{1} <<
                                static_cast<unsigned>(
                                    exponent_difference)),
                        config.accumulator_bits,
                        output.matrix_overflow_count);
                } else {
                    delta = round_shift_i128(
                        product,
                        static_cast<unsigned>(-exponent_difference),
                        config.accumulator_bits,
                        output.matrix_overflow_count);
                }
                const auto index =
                    static_cast<std::size_t>(row_physical) *
                        total_dim + target_col;
                physical[index] = clamp_matrix_accumulator(
                    static_cast<__int128>(physical[index]) - delta,
                    config.accumulator_bits,
                    output.matrix_overflow_count);
                if (output.matrix_overflow_count != before_overflow) {
                    throw PrecisionRescueRequired(
                        "tile-BFP workspace overflow in column " +
                        std::to_string(column));
                }
                workspace_max_abs = std::max(
                    workspace_max_abs,
                    real_abs(
                        physical[index],
                        workspace_exponents[destination_tile]));
            }
        }
    }

    output.factor.pvec = pvec;
    output.factor.l.assign(
        static_cast<std::size_t>(total_dim) * pivot_dim, 0);
    const auto qf_one =
        static_cast<std::int32_t>(std::uint32_t{1} << config.frac_bits);
    for (std::uint32_t row = 0; row < total_dim; ++row) {
        const auto physical_r = physical_row(row);
        for (std::uint32_t col = 0; col < pivot_dim; ++col) {
            if (row == col) {
                output.factor.l[row * pivot_dim + col] = qf_one;
            } else if (row > col) {
                output.factor.l[row * pivot_dim + col] =
                    l_physical[physical_r * pivot_dim + col];
            }
        }
    }

    std::vector<std::int64_t> u_mantissas(
        static_cast<std::size_t>(pivot_dim) * total_dim, 0);
    std::vector<std::int16_t> u_exponents(
        u_mantissas.size(), 0);
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        const auto physical_r = physical_row(row);
        for (std::uint32_t col = row; col < total_dim; ++col) {
            const auto output_index =
                static_cast<std::size_t>(row) * total_dim + col;
            u_mantissas[output_index] =
                physical[physical_r * total_dim + col];
            u_exponents[output_index] =
                exponent_at(physical_r, col);
        }
    }
    quantize_bfp_pairs(
        u_mantissas,
        u_exponents,
        pivot_dim,
        total_dim,
        config,
        output.factor.u,
        output.factor.u_tile_exponents,
        output.matrix_overflow_count);
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        if (output.factor.u[row * total_dim + row] == 0) {
            throw PrecisionRescueRequired(
                "tile-BFP U diagonal lost during requantization");
        }
    }

    const auto update_dim = total_dim - pivot_dim;
    std::vector<std::int64_t> update_mantissas(
        static_cast<std::size_t>(update_dim) * update_dim, 0);
    std::vector<std::int16_t> update_exponents(
        update_mantissas.size(), 0);
    for (std::uint32_t row = 0; row < update_dim; ++row) {
        for (std::uint32_t col = 0; col < update_dim; ++col) {
            const auto source_row = pivot_dim + row;
            const auto source_col = pivot_dim + col;
            const auto index =
                static_cast<std::size_t>(row) * update_dim + col;
            update_mantissas[index] =
                physical[source_row * total_dim + source_col];
            update_exponents[index] =
                exponent_at(source_row, source_col);
        }
    }
    quantize_bfp_pairs(
        update_mantissas,
        update_exponents,
        update_dim,
        update_dim,
        config,
        output.factor.update,
        output.factor.update_tile_exponents,
        output.matrix_overflow_count);
    output.factor.tile_size = tile;
    output.factor.u_exponent =
        output.factor.u_tile_exponents.empty() ?
            0 : output.factor.u_tile_exponents.front();
    output.factor.update_exponent =
        output.factor.update_tile_exponents.empty() ?
            output.factor.u_exponent :
            output.factor.update_tile_exponents.front();
    long double check_numerator = 0.0L;
    long double check_denominator = 0.0L;
    const auto l_scale =
        std::ldexp(1.0L, -static_cast<int>(config.frac_bits));
    for (std::uint32_t row = 0; row < total_dim; ++row) {
        const auto source_row =
            row < pivot_dim ?
            static_cast<std::uint32_t>(pvec[row]) : row;
        for (std::uint32_t col = 0; col < total_dim; ++col) {
            const auto expected =
                static_cast<long double>(
                    assembled[source_row * total_dim + col]) *
                std::ldexp(
                    1.0L,
                    assembled_tile_exponents[
                        bfp_tile_index(
                            source_row,
                            col,
                            total_dim,
                            tile)]);
            long double reconstructed = 0.0L;
            for (std::uint32_t k = 0; k < pivot_dim; ++k) {
                const auto u_exponent =
                    output.factor.u_tile_exponents[
                        bfp_tile_index(
                            k,
                            col,
                            total_dim,
                            tile)];
                reconstructed +=
                    static_cast<long double>(
                        output.factor.l[row * pivot_dim + k]) *
                    l_scale *
                    static_cast<long double>(
                        output.factor.u[k * total_dim + col]) *
                    std::ldexp(1.0L, u_exponent);
            }
            if (row >= pivot_dim && col >= pivot_dim) {
                const auto update_row = row - pivot_dim;
                const auto update_col = col - pivot_dim;
                reconstructed +=
                    static_cast<long double>(
                        output.factor.update[
                            update_row * update_dim + update_col]) *
                    std::ldexp(
                        1.0L,
                        output.factor.update_tile_exponents[
                            bfp_tile_index(
                                update_row,
                                update_col,
                                update_dim,
                                tile)]);
            }
            const auto error = expected - reconstructed;
            check_numerator += error * error;
            check_denominator += expected * expected;
        }
    }
    const auto factor_error =
        std::sqrt(check_numerator) /
        std::max(std::sqrt(check_denominator), 1e-300L);
    if (!std::isfinite(factor_error) ||
        factor_error > config.fixed_factor_rel_tol) {
        throw FactorCheckFailure(
            "tile-BFP local factor check failed: relative error=" +
            std::to_string(static_cast<double>(factor_error)));
    }
    output.max_growth_ratio =
        initial_max_abs == 0.0L ? 1.0 :
        static_cast<double>(workspace_max_abs / initial_max_abs);
    output.factor.l_frac_bits = config.frac_bits;
    output.factor.valid = true;
    return output;
}

inline FixedComputation factor_fixed_front(
    const std::vector<std::int32_t>& assembled,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    std::int16_t exponent,
    const ModelConfig& config) {
    if (pivot_dim == 0 || pivot_dim > total_dim || pivot_dim > MAX_ROWS ||
        assembled.size() != static_cast<std::size_t>(total_dim) * total_dim) {
        throw NumericFailure("invalid fixed front dimensions");
    }
    FixedComputation output{};
    const auto guard_bits = config.workspace_guard_bits;
    const auto workspace_exponent_value =
        static_cast<int>(exponent) - static_cast<int>(guard_bits);
    if (workspace_exponent_value <
            std::numeric_limits<std::int16_t>::min() ||
        workspace_exponent_value >
            std::numeric_limits<std::int16_t>::max()) {
        throw NumericFailure("guard-bit workspace exponent exceeds int16");
    }
    const auto workspace_exponent =
        static_cast<std::int16_t>(workspace_exponent_value);
    std::vector<std::int64_t> physical;
    physical.reserve(assembled.size());
    std::int64_t initial_max_abs = 0;
    for (const auto value : assembled) {
        const auto lifted = clamp_matrix_accumulator(
            static_cast<__int128>(value) *
                (__int128{1} << guard_bits),
            config.accumulator_bits,
            output.matrix_overflow_count);
        physical.push_back(lifted);
        initial_max_abs = std::max(initial_max_abs, abs_i64(lifted));
    }
    std::int64_t workspace_max_abs = initial_max_abs;
    std::vector<std::uint16_t> pvec(pivot_dim);
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        pvec[row] = static_cast<std::uint16_t>(row);
    }
    const auto physical_row = [&](std::uint32_t logical) -> std::uint32_t {
        return logical < pivot_dim ? pvec[logical] : logical;
    };

    for (std::uint32_t column = 0; column < pivot_dim; ++column) {
        std::vector<std::pair<std::uint16_t, std::int64_t>> wide_candidates;
        wide_candidates.reserve(pivot_dim - column);
        for (std::uint32_t row = column; row < pivot_dim; ++row) {
            const auto value =
                physical[physical_row(row) * total_dim + column];
            wide_candidates.push_back({
                static_cast<std::uint16_t>(row),
                value,
            });
        }
        auto candidates = normalize_pivot_candidates(
            wide_candidates, output.workspace_renormalize_count);
        std::uint32_t best_row = column;
        std::int64_t best_abs = -1;
        for (const auto& candidate : candidates) {
            const auto magnitude =
                abs_i64(candidate.value);
            if (magnitude > best_abs) {
                best_abs = magnitude;
                best_row = candidate.row;
            }
        }
        std::int64_t best_wide_abs = 0;
        for (const auto& [row, value] : wide_candidates) {
            if (row == best_row) {
                best_wide_abs = abs_i64(value);
                break;
            }
        }
        output.candidates.push_back(std::move(candidates));
        output.selected_rows.push_back(static_cast<std::uint16_t>(best_row));
        if (best_abs <= 0 || best_wide_abs <= 0) {
            throw PrecisionRescueRequired(
                "wide fixed pivot column " + std::to_string(column) +
                " contains only zeros");
        }
        const auto pivot_ratio =
            initial_max_abs == 0 ? 0.0 :
            static_cast<double>(best_wide_abs) /
                static_cast<double>(initial_max_abs);
        output.min_pivot_ratio =
            std::min(output.min_pivot_ratio, pivot_ratio);
        if (pivot_ratio <= config.fixed_pivot_rel_tol) {
            ++output.small_pivot_count;
            throw PrecisionRescueRequired(
                "fixed pivot below precision threshold in column " +
                std::to_string(column));
        }
        if (best_row != column) {
            std::swap(pvec[column], pvec[best_row]);
            ++output.swap_count;
        }

        const auto pivot_physical = physical_row(column);
        const auto pivot =
            physical[pivot_physical * total_dim + column];
        if (pivot == 0) {
            throw NumericFailure("quantized pivot became zero");
        }

        for (std::uint32_t logical_row = column + 1;
             logical_row < total_dim; ++logical_row) {
            const auto row_physical = physical_row(logical_row);
            const auto value =
                physical[row_physical * total_dim + column];
            const auto multiplier_wide = round_div_signed_wide(
                static_cast<__int128>(value) *
                    (__int128{1} << config.frac_bits),
                pivot);
            if (multiplier_wide >
                    std::numeric_limits<std::int32_t>::max() ||
                multiplier_wide <
                    std::numeric_limits<std::int32_t>::min()) {
                throw PrecisionRescueRequired(
                    "fixed L multiplier exceeds int32 in column " +
                    std::to_string(column));
            }
            const auto multiplier = saturate_i32(
                multiplier_wide, output.matrix_overflow_count);
            physical[row_physical * total_dim + column] =
                multiplier;

            const auto stored_multiplier = multiplier;
            for (std::uint32_t target_col = column + 1;
                 target_col < total_dim; ++target_col) {
                const __int128 product =
                    static_cast<__int128>(stored_multiplier) *
                    physical[pivot_physical * total_dim + target_col];
                const auto overflow_before =
                    output.matrix_overflow_count;
                const auto delta = round_shift_wide(
                    product,
                    config.frac_bits,
                    config.accumulator_bits,
                    output.matrix_overflow_count);
                const __int128 updated =
                    static_cast<__int128>(
                        physical[row_physical * total_dim + target_col]) -
                    delta;
                physical[row_physical * total_dim + target_col] =
                    clamp_matrix_accumulator(
                        updated,
                        config.accumulator_bits,
                        output.matrix_overflow_count);
                if (output.matrix_overflow_count !=
                    overflow_before) {
                    throw PrecisionRescueRequired(
                        "guard-bit workspace overflow in column " +
                        std::to_string(column));
                }
                workspace_max_abs = std::max(
                    workspace_max_abs,
                    abs_i64(
                        physical[
                            row_physical * total_dim + target_col]));
            }
        }
    }

    output.factor.pvec = pvec;
    output.factor.l.assign(
        static_cast<std::size_t>(total_dim) * pivot_dim, 0);
    output.factor.u.assign(
        static_cast<std::size_t>(pivot_dim) * total_dim, 0);
    const auto qf_one = std::int32_t{1} << config.frac_bits;
    for (std::uint32_t row = 0; row < total_dim; ++row) {
        const auto row_physical = physical_row(row);
        for (std::uint32_t col = 0; col < pivot_dim; ++col) {
            if (row == col) {
                output.factor.l[row * pivot_dim + col] = qf_one;
            } else if (row > col) {
                output.factor.l[row * pivot_dim + col] =
                    saturate_i32(
                        physical[row_physical * total_dim + col],
                        output.matrix_overflow_count);
            }
        }
    }
    std::vector<std::int64_t> u_wide(
        static_cast<std::size_t>(pivot_dim) * total_dim, 0);
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        const auto row_physical = physical_row(row);
        for (std::uint32_t col = row; col < total_dim; ++col) {
            u_wide[row * total_dim + col] =
                physical[row_physical * total_dim + col];
        }
    }
    auto quantized_u = quantize_wide_block(
        u_wide,
        workspace_exponent,
        config,
        output.matrix_overflow_count);
    output.factor.u = std::move(quantized_u.first);
    output.factor.u_exponent = quantized_u.second;
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        if (output.factor.u[row * total_dim + row] == 0) {
            throw PrecisionRescueRequired(
                "U diagonal lost during factor requantization in column " +
                std::to_string(row));
        }
    }

    const auto update_dim = total_dim - pivot_dim;
    std::vector<std::int64_t> update_wide;
    update_wide.reserve(
        static_cast<std::size_t>(update_dim) * update_dim);
    for (std::uint32_t row = 0; row < update_dim; ++row) {
        for (std::uint32_t col = 0; col < update_dim; ++col) {
            update_wide.push_back(
                physical[(pivot_dim + row) * total_dim + pivot_dim + col]);
        }
    }
    auto quantized_update = quantize_wide_block(
        update_wide,
        workspace_exponent,
        config,
        output.matrix_overflow_count);
    output.factor.update = std::move(quantized_update.first);
    output.factor.update_exponent =
        update_wide.empty() ?
            output.factor.u_exponent : quantized_update.second;
    output.max_growth_ratio =
        initial_max_abs == 0 ? 1.0 :
        static_cast<double>(workspace_max_abs) /
            static_cast<double>(initial_max_abs);
    output.factor.l_frac_bits = config.frac_bits;
    output.factor.valid = true;
    return output;
}

inline Fp64Computation factor_fp64_front(
    const std::vector<double>& assembled,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    const ModelConfig& config) {
    if (pivot_dim == 0 || pivot_dim > total_dim ||
        assembled.size() != static_cast<std::size_t>(total_dim) * total_dim) {
        throw NumericFailure("invalid FP64 front dimensions");
    }
    Fp64Computation output{};
    auto physical = assembled;
    std::vector<std::uint16_t> pvec(pivot_dim);
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        pvec[row] = static_cast<std::uint16_t>(row);
    }
    const auto physical_row = [&](std::uint32_t logical) -> std::uint32_t {
        return logical < pivot_dim ? pvec[logical] : logical;
    };
    double front_norm = 0.0;
    for (const auto value : assembled) {
        front_norm = std::max(front_norm, std::abs(value));
    }
    const auto threshold = config.pivot_rel_tol * front_norm;

    for (std::uint32_t column = 0; column < pivot_dim; ++column) {
        std::uint32_t best_row = column;
        double best_abs = -1.0;
        for (std::uint32_t row = column; row < pivot_dim; ++row) {
            const auto magnitude = std::abs(
                physical[physical_row(row) * total_dim + column]);
            if (magnitude > best_abs) {
                best_abs = magnitude;
                best_row = row;
            }
        }
        output.selected_rows.push_back(static_cast<std::uint16_t>(best_row));
        if (best_abs <= threshold) {
            throw NumericFailure(
                "FP64 pivot below threshold in column " +
                std::to_string(column));
        }
        if (best_row != column) {
            std::swap(pvec[column], pvec[best_row]);
            ++output.swap_count;
        }
        const auto pivot_physical = physical_row(column);
        const auto pivot =
            physical[pivot_physical * total_dim + column];
        for (std::uint32_t logical_row = column + 1;
             logical_row < total_dim; ++logical_row) {
            const auto row_physical = physical_row(logical_row);
            const auto multiplier =
                physical[row_physical * total_dim + column] / pivot;
            physical[row_physical * total_dim + column] = multiplier;
            for (std::uint32_t target_col = column + 1;
                 target_col < total_dim; ++target_col) {
                physical[row_physical * total_dim + target_col] -=
                    multiplier *
                    physical[pivot_physical * total_dim + target_col];
            }
        }
    }

    output.factor.pvec = pvec;
    output.factor.l.assign(
        static_cast<std::size_t>(total_dim) * pivot_dim, 0.0);
    output.factor.u.assign(
        static_cast<std::size_t>(pivot_dim) * total_dim, 0.0);
    for (std::uint32_t row = 0; row < total_dim; ++row) {
        const auto row_physical = physical_row(row);
        for (std::uint32_t col = 0; col < pivot_dim; ++col) {
            if (row == col) {
                output.factor.l[row * pivot_dim + col] = 1.0;
            } else if (row > col) {
                output.factor.l[row * pivot_dim + col] =
                    physical[row_physical * total_dim + col];
            }
        }
    }
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        const auto row_physical = physical_row(row);
        for (std::uint32_t col = row; col < total_dim; ++col) {
            output.factor.u[row * total_dim + col] =
                physical[row_physical * total_dim + col];
        }
    }
    const auto update_dim = total_dim - pivot_dim;
    output.factor.update.reserve(
        static_cast<std::size_t>(update_dim) * update_dim);
    for (std::uint32_t row = 0; row < update_dim; ++row) {
        for (std::uint32_t col = 0; col < update_dim; ++col) {
            output.factor.update.push_back(
                physical[(pivot_dim + row) * total_dim + pivot_dim + col]);
        }
    }
    output.factor.valid = true;
    return output;
}

inline std::pair<std::vector<std::int32_t>, std::int16_t>
quantize_fp64_block(
    const std::vector<double>& values,
    const ModelConfig& config,
    std::uint64_t& overflow_count) {
    if (values.empty()) return {{}, 0};
    double max_abs = 0.0;
    for (const auto value : values) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    int exponent = 0;
    if (max_abs != 0.0) {
        exponent = static_cast<int>(
            std::ceil(std::log2(
                max_abs / static_cast<double>(config.q_limit()))));
    }
    if (exponent < std::numeric_limits<std::int16_t>::min() ||
        exponent > std::numeric_limits<std::int16_t>::max()) {
        throw NumericFailure("rescued factor exponent exceeds int16");
    }
    std::vector<std::int32_t> quantized;
    quantized.reserve(values.size());
    for (const auto value : values) {
        const auto scaled = std::nearbyint(std::ldexp(value, -exponent));
        if (!std::isfinite(scaled)) {
            throw NumericFailure("rescued factor contains non-finite value");
        }
        quantized.push_back(
            saturate_i32(
                static_cast<__int128>(scaled),
                overflow_count));
    }
    return {
        std::move(quantized),
        static_cast<std::int16_t>(exponent),
    };
}

inline void quantize_fp64_tile_block(
    const std::vector<double>& values,
    std::uint32_t rows,
    std::uint32_t cols,
    const ModelConfig& config,
    std::vector<std::int32_t>& output,
    std::vector<std::int16_t>& exponents,
    std::uint64_t& overflow_count) {
    if (values.size() != static_cast<std::size_t>(rows) * cols) {
        throw NumericFailure("rescued tile factor dimensions mismatch");
    }
    output.assign(values.size(), 0);
    if (rows == 0 || cols == 0) {
        exponents.clear();
        return;
    }
    const auto tile = config.bfp_tile_size;
    const auto tile_rows = (rows + tile - 1) / tile;
    const auto tile_cols = (cols + tile - 1) / tile;
    exponents.assign(
        static_cast<std::size_t>(tile_rows) * tile_cols, 0);
    for (std::uint32_t tr = 0; tr < tile_rows; ++tr) {
        for (std::uint32_t tc = 0; tc < tile_cols; ++tc) {
            const auto row_end = std::min((tr + 1) * tile, rows);
            const auto col_end = std::min((tc + 1) * tile, cols);
            double max_abs = 0.0;
            for (auto row = tr * tile; row < row_end; ++row) {
                for (auto col = tc * tile; col < col_end; ++col) {
                    max_abs = std::max(
                        max_abs,
                        std::abs(values[row * cols + col]));
                }
            }
            int exponent = 0;
            if (max_abs != 0.0) {
                exponent = static_cast<int>(std::ceil(std::log2(
                    max_abs / static_cast<double>(config.q_limit()))));
            }
            if (exponent < std::numeric_limits<std::int16_t>::min() ||
                exponent > std::numeric_limits<std::int16_t>::max()) {
                throw NumericFailure(
                    "rescued tile exponent exceeds int16");
            }
            exponents[
                static_cast<std::size_t>(tr) * tile_cols + tc] =
                static_cast<std::int16_t>(exponent);
            for (auto row = tr * tile; row < row_end; ++row) {
                for (auto col = tc * tile; col < col_end; ++col) {
                    const auto scaled = std::nearbyint(std::ldexp(
                        values[row * cols + col], -exponent));
                    if (!std::isfinite(scaled)) {
                        throw NumericFailure(
                            "rescued tile factor is non-finite");
                    }
                    output[row * cols + col] =
                        saturate_i32(
                            static_cast<__int128>(scaled),
                            overflow_count);
                }
            }
        }
    }
}

inline FixedComputation quantize_rescued_factor(
    const Fp64Computation& source,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    const ModelConfig& config) {
    FixedComputation output{};
    output.selected_rows = source.selected_rows;
    output.swap_count = source.swap_count;
    output.precision_rescued = true;
    output.min_pivot_ratio = 0.0;
    output.factor.pvec = source.factor.pvec;
    output.factor.l.reserve(source.factor.l.size());
    const auto l_scale = std::ldexp(1.0, config.frac_bits);
    for (const auto value : source.factor.l) {
        const auto scaled = std::nearbyint(value * l_scale);
        if (!std::isfinite(scaled)) {
            throw NumericFailure("rescued L factor contains non-finite value");
        }
        output.factor.l.push_back(
            saturate_i32(
                static_cast<__int128>(scaled),
                output.matrix_overflow_count));
    }
    if (config.bfp_tile_size != 0) {
        quantize_fp64_tile_block(
            source.factor.u,
            pivot_dim,
            total_dim,
            config,
            output.factor.u,
            output.factor.u_tile_exponents,
            output.matrix_overflow_count);
        const auto update_dim = total_dim - pivot_dim;
        quantize_fp64_tile_block(
            source.factor.update,
            update_dim,
            update_dim,
            config,
            output.factor.update,
            output.factor.update_tile_exponents,
            output.matrix_overflow_count);
        output.factor.tile_size = config.bfp_tile_size;
        output.factor.u_exponent =
            output.factor.u_tile_exponents.empty() ?
                0 : output.factor.u_tile_exponents.front();
        output.factor.update_exponent =
            output.factor.update_tile_exponents.empty() ?
                output.factor.u_exponent :
                output.factor.update_tile_exponents.front();
    } else {
        auto u = quantize_fp64_block(
            source.factor.u, config, output.matrix_overflow_count);
        output.factor.u = std::move(u.first);
        output.factor.u_exponent = u.second;
        auto update = quantize_fp64_block(
            source.factor.update, config, output.matrix_overflow_count);
        output.factor.update = std::move(update.first);
        output.factor.update_exponent =
            source.factor.update.empty() ? u.second : update.second;
    }
    output.factor.l_frac_bits = config.frac_bits;
    output.factor.precision_rescued = true;
    output.factor.valid = true;
    return output;
}

inline std::uint64_t ceil_div_u64(std::uint64_t value, std::uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline std::vector<OperationLog> build_operation_plan(
    std::uint16_t node_id,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    const ModelConfig& config,
    std::uint64_t queued_cycle) {
    std::vector<OperationLog> operations;
    const auto tile = config.tile_size;
    const auto pivot_tiles = (pivot_dim + tile - 1) / tile;
    const auto update_dim = total_dim - pivot_dim;
    const auto update_tiles = (update_dim + tile - 1) / tile;
    const auto extent = [&](std::uint32_t index, std::uint32_t dimension) {
        const auto begin = index * tile;
        return begin >= dimension ? 0u :
            std::min(tile, dimension - begin);
    };
    auto add = [&](OpType type, unsigned i, unsigned j, unsigned k,
                   unsigned m, unsigned n, unsigned inner) {
        operations.push_back({
            node_id, type, i, j, k, m, n, inner,
            queued_cycle, 0, 0,
        });
    };

    for (std::uint32_t k = 0; k < pivot_tiles; ++k) {
        const auto kd = extent(k, pivot_dim);
        add(OpType::Fact, k, k, k, kd, kd, kd);
        for (std::uint32_t j = k + 1; j < pivot_tiles; ++j) {
            add(OpType::TrsmU, k, j, k, kd, extent(j, pivot_dim), kd);
        }
        for (std::uint32_t i = k + 1; i < pivot_tiles; ++i) {
            add(OpType::TrsmL, i, k, k, extent(i, pivot_dim), kd, kd);
        }
        for (std::uint32_t i = k + 1; i < pivot_tiles; ++i) {
            for (std::uint32_t j = k + 1; j < pivot_tiles; ++j) {
                add(OpType::GemmPivot, i, j, k,
                    extent(i, pivot_dim), extent(j, pivot_dim), kd);
            }
        }
    }
    for (std::uint32_t k = 0; k < pivot_tiles; ++k) {
        const auto kd = extent(k, pivot_dim);
        for (std::uint32_t j = 0; j < update_tiles; ++j) {
            add(OpType::TrsmF12, k, j, k,
                kd, extent(j, update_dim), kd);
        }
        for (std::uint32_t i = 0; i < update_tiles; ++i) {
            add(OpType::TrsmF21, i, k, k,
                extent(i, update_dim), kd, kd);
        }
        for (std::uint32_t i = 0; i < update_tiles; ++i) {
            for (std::uint32_t j = 0; j < update_tiles; ++j) {
                add(OpType::GemmSchur, i, j, k,
                    extent(i, update_dim), extent(j, update_dim), kd);
            }
        }
    }
    return operations;
}

inline std::uint64_t operation_latency(
    const OperationLog& operation,
    const ModelConfig& config) {
    const auto m = std::max(operation.m_dim, 1u);
    const auto n = std::max(operation.n_dim, 1u);
    const auto k = std::max(operation.k_dim, 1u);
    switch (operation.type) {
    case OpType::Fact: {
        const std::uint64_t work =
            static_cast<std::uint64_t>(m) * m * (m + 3) / 3;
        return config.panel_startup +
               ceil_div_u64(work, config.panel_ops_per_cycle);
    }
    case OpType::TrsmU:
    case OpType::TrsmL:
    case OpType::TrsmF12:
    case OpType::TrsmF21: {
        const std::uint64_t work =
            static_cast<std::uint64_t>(m) * n * k;
        return config.trsm_startup +
               ceil_div_u64(work, config.trsm_macs_per_cycle);
    }
    case OpType::GemmPivot:
    case OpType::GemmSchur: {
        const std::uint64_t work =
            static_cast<std::uint64_t>(m) * n * k;
        return config.gemm_startup +
               ceil_div_u64(work, config.gemm_macs_per_cycle);
    }
    case OpType::PrecisionRescue: {
        const std::uint64_t work =
            static_cast<std::uint64_t>(m) * n * k;
        return config.precision_rescue_startup +
               ceil_div_u64(
                   work, config.precision_rescue_macs_per_cycle);
    }
    case OpType::SolveForward:
    case OpType::SolveBackward:
    case OpType::SolveResidual:
        return config.trsm_startup +
               ceil_div_u64(
                   static_cast<std::uint64_t>(m) * n,
                   config.trsm_macs_per_cycle);
    }
    return 0;
}

inline std::uint64_t schedule_operations(
    std::vector<OperationLog>& operations,
    const ModelConfig& config,
    std::uint64_t start_cycle) {
    if (config.scheduler_policy == "serial") {
        auto cursor = start_cycle;
        for (auto& operation : operations) {
            operation.start_cycle = cursor;
            cursor += operation_latency(operation, config);
            operation.end_cycle = cursor;
        }
        return cursor;
    }

    std::vector<std::uint64_t> panel_available(
        config.panel_units, start_cycle);
    std::vector<std::uint64_t> trsm_available(
        config.trsm_units, start_cycle);
    std::vector<std::uint64_t> gemm_available(
        config.gemm_units, start_cycle);
    std::uint64_t dependency_barrier = start_cycle;
    std::uint64_t fact_ready = start_cycle;
    std::uint64_t trsm_ready = start_cycle;
    std::uint64_t gemm_ready = start_cycle;
    std::optional<unsigned> active_pivot_tile;
    bool in_border_phase = false;
    for (auto& operation : operations) {
        std::uint64_t ready = dependency_barrier;
        switch (operation.type) {
        case OpType::Fact:
            if (!active_pivot_tile || *active_pivot_tile != operation.tile_k) {
                dependency_barrier =
                    std::max(dependency_barrier, gemm_ready);
                active_pivot_tile = operation.tile_k;
            }
            ready = dependency_barrier;
            break;
        case OpType::TrsmU:
        case OpType::TrsmL:
            ready = fact_ready;
            break;
        case OpType::GemmPivot:
            ready = trsm_ready;
            break;
        case OpType::TrsmF12:
        case OpType::TrsmF21:
            if (!in_border_phase) {
                dependency_barrier =
                    std::max(dependency_barrier, gemm_ready);
                in_border_phase = true;
            }
            ready = dependency_barrier;
            break;
        case OpType::GemmSchur:
            ready = trsm_ready;
            break;
        case OpType::SolveForward:
        case OpType::SolveBackward:
        case OpType::SolveResidual:
        case OpType::PrecisionRescue:
            ready = dependency_barrier;
            break;
        }
        auto* pool = &trsm_available;
        if (operation.type == OpType::Fact ||
            operation.type == OpType::PrecisionRescue) {
            pool = &panel_available;
        } else if (
            operation.type == OpType::GemmPivot ||
            operation.type == OpType::GemmSchur) {
            pool = &gemm_available;
        }
        auto resource =
            std::min_element(pool->begin(), pool->end());
        operation.start_cycle =
            std::max({*resource, operation.queued_cycle, ready});
        operation.end_cycle =
            operation.start_cycle + operation_latency(operation, config);
        *resource = operation.end_cycle;
        switch (operation.type) {
        case OpType::Fact:
            fact_ready = operation.end_cycle;
            trsm_ready = fact_ready;
            break;
        case OpType::TrsmU:
        case OpType::TrsmL:
        case OpType::TrsmF12:
        case OpType::TrsmF21:
            trsm_ready = std::max(trsm_ready, operation.end_cycle);
            break;
        case OpType::GemmPivot:
        case OpType::GemmSchur:
            gemm_ready = std::max(gemm_ready, operation.end_cycle);
            break;
        case OpType::SolveForward:
        case OpType::SolveBackward:
        case OpType::SolveResidual:
        case OpType::PrecisionRescue:
            dependency_barrier = operation.end_cycle;
            break;
        }
    }
    std::uint64_t finish = gemm_ready;
    for (const auto available : panel_available) {
        finish = std::max(finish, available);
    }
    for (const auto available : trsm_available) {
        finish = std::max(finish, available);
    }
    for (const auto available : gemm_available) {
        finish = std::max(finish, available);
    }
    return finish;
}

}  // namespace hw
