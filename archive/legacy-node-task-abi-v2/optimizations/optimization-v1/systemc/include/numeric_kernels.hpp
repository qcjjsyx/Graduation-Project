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

inline FixedComputation quantize_rescued_factor(
    const Fp64Computation& source,
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
    auto u = quantize_fp64_block(
        source.factor.u, config, output.matrix_overflow_count);
    output.factor.u = std::move(u.first);
    output.factor.u_exponent = u.second;
    auto update = quantize_fp64_block(
        source.factor.update, config, output.matrix_overflow_count);
    output.factor.update = std::move(update.first);
    output.factor.update_exponent =
        source.factor.update.empty() ? u.second : update.second;
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
