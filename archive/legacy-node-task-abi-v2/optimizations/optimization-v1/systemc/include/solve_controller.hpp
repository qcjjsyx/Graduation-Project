#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_types.hpp"
#include "numeric_kernels.hpp"
#include "system_memory.hpp"

namespace hw {

struct SolveMetrics {
    bool valid{false};
    std::string failure_reason{};
    std::vector<double> x_permuted{};
    std::vector<double> x_original{};
    std::vector<std::int64_t> x_mantissa{};
    std::vector<std::int16_t> exponent_by_node{};
    double relative_residual{std::numeric_limits<double>::infinity()};
    double scaled_relative_residual{std::numeric_limits<double>::infinity()};
    double componentwise_backward_error{
        std::numeric_limits<double>::infinity()};
    double relative_solution_error{std::numeric_limits<double>::infinity()};
    double initial_relative_residual{std::numeric_limits<double>::infinity()};
    std::vector<double> residual_history{};
    unsigned refinement_iterations{0};
    bool refinement_converged{false};
    bool refined_solution{false};
    bool used_precision_rescue{false};
    std::string refinement_stop_reason{};
    std::uint64_t cycles{0};
    QuantStats vector_stats{};
};

inline double l2_norm(const std::vector<double>& values) {
    long double sum = 0.0;
    for (const auto value : values) {
        sum += static_cast<long double>(value) * value;
    }
    return std::sqrt(static_cast<double>(sum));
}

inline double relative_residual(
    const std::vector<double>& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    const auto n = rhs.size();
    if (matrix.size() != n * n || x.size() != n) {
        throw std::invalid_argument("relative_residual dimension mismatch");
    }
    std::vector<double> residual(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        long double value = -rhs[row];
        for (std::size_t col = 0; col < n; ++col) {
            value += static_cast<long double>(matrix[row * n + col]) * x[col];
        }
        residual[row] = static_cast<double>(value);
    }
    return l2_norm(residual) / std::max(l2_norm(rhs), 1e-300);
}

inline std::vector<double> residual_vector(
    const std::vector<double>& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    const auto n = rhs.size();
    if (matrix.size() != n * n || x.size() != n) {
        throw std::invalid_argument("residual_vector dimension mismatch");
    }
    std::vector<double> residual(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        long double value = rhs[row];
        for (std::size_t col = 0; col < n; ++col) {
            value -= static_cast<long double>(
                matrix[row * n + col]) * x[col];
        }
        residual[row] = static_cast<double>(value);
    }
    return residual;
}

inline double componentwise_backward_error(
    const std::vector<double>& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    const auto residual = residual_vector(matrix, x, rhs);
    const auto n = rhs.size();
    double maximum = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        long double denominator = std::abs(rhs[row]);
        for (std::size_t col = 0; col < n; ++col) {
            denominator +=
                std::abs(static_cast<long double>(
                    matrix[row * n + col])) *
                std::abs(static_cast<long double>(x[col]));
        }
        const auto numerator = std::abs(residual[row]);
        if (denominator == 0.0L) {
            if (numerator != 0.0) {
                return std::numeric_limits<double>::infinity();
            }
            continue;
        }
        maximum = std::max(
            maximum,
            static_cast<double>(numerator / denominator));
    }
    return maximum;
}

inline double relative_solution_error(
    const std::vector<double>& x,
    const std::vector<double>& reference) {
    if (x.size() != reference.size()) {
        throw std::invalid_argument("solution/reference dimension mismatch");
    }
    std::vector<double> difference(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        difference[i] = x[i] - reference[i];
    }
    return l2_norm(difference) / std::max(l2_norm(reference), 1e-300);
}

inline std::vector<double> restore_ordering(
    const std::vector<double>& permuted,
    const std::vector<std::uint32_t>& permutation) {
    if (permuted.size() != permutation.size()) {
        throw std::invalid_argument("solution/permutation dimension mismatch");
    }
    std::vector<double> original(permuted.size());
    for (std::size_t index = 0; index < permuted.size(); ++index) {
        original.at(permutation.at(index)) = permuted[index];
    }
    return original;
}

inline std::vector<double> restore_original_coordinates(
    const std::vector<double>& permuted,
    const std::vector<std::uint32_t>& permutation,
    const std::vector<std::int16_t>& column_scale_exponents) {
    auto original = restore_ordering(permuted, permutation);
    if (column_scale_exponents.size() != original.size()) {
        throw std::invalid_argument(
            "column-scale/solution dimension mismatch");
    }
    for (std::size_t index = 0; index < original.size(); ++index) {
        original[index] = std::ldexp(
            original[index], column_scale_exponents[index]);
    }
    return original;
}

inline std::uint64_t solve_node_cycles(
    const NodeTask& task,
    const ModelConfig& config,
    bool backward) {
    const auto p = static_cast<std::uint64_t>(task.pivot_dim);
    const auto u = static_cast<std::uint64_t>(
        task.total_dim - task.pivot_dim);
    const auto work = p * (p + u);
    (void)backward;
    return config.trsm_startup +
           ceil_div_u64(work, config.trsm_macs_per_cycle);
}

inline SolveMetrics solve_fp64(
    SystemMemory& memory,
    const ModelConfig& config,
    SimulationStats& stats,
    std::uint64_t start_cycle) {
    SolveMetrics result{};
    try {
        const auto n = memory.matrix_dim();
        if (memory.rhs_fp64.size() != n) {
            throw NumericFailure("FP64 RHS length mismatch");
        }
        std::vector<double> work = memory.rhs_fp64;
        std::vector<double> y(n, 0.0);

        std::uint64_t cursor = start_cycle;
        for (std::uint16_t node_id = 0;
             node_id < memory.size(); ++node_id) {
            const auto& node = memory.at(node_id);
            if (!node.fp64.valid) {
                throw NumericFailure("FP64 factor is missing for forward solve");
            }
            const auto p = node.task.pivot_dim;
            const auto dim = node.task.total_dim;
            std::vector<double> pivot_rhs(p);
            for (std::uint32_t row = 0; row < p; ++row) {
                const auto source_local = node.fp64.pvec[row];
                pivot_rhs[row] =
                    work[node.front_indices[source_local]];
            }
            for (std::uint32_t row = 0; row < p; ++row) {
                long double value = pivot_rhs[row];
                for (std::uint32_t col = 0; col < row; ++col) {
                    value -= static_cast<long double>(
                        node.fp64.l[row * p + col]) *
                        y[node.front_indices[col]];
                }
                y[node.front_indices[row]] = static_cast<double>(value);
            }
            for (std::uint32_t row = p; row < dim; ++row) {
                long double value = work[node.front_indices[row]];
                for (std::uint32_t col = 0; col < p; ++col) {
                    value -= static_cast<long double>(
                        node.fp64.l[row * p + col]) *
                        y[node.front_indices[col]];
                }
                work[node.front_indices[row]] = static_cast<double>(value);
            }

            const auto begin = cursor;
            cursor += solve_node_cycles(node.task, config, false);
            stats.operations.push_back({
                node_id, OpType::SolveForward, 0, 0, 0,
                p, 1, dim, begin, begin, cursor,
            });
        }

        std::vector<double> x(n, 0.0);
        for (std::size_t reverse = memory.size(); reverse > 0; --reverse) {
            const auto node_id = static_cast<std::uint16_t>(reverse - 1);
            const auto& node = memory.at(node_id);
            const auto p = node.task.pivot_dim;
            const auto dim = node.task.total_dim;
            for (std::uint32_t reverse_row = p; reverse_row > 0; --reverse_row) {
                const auto row = reverse_row - 1;
                long double value = y[node.front_indices[row]];
                for (std::uint32_t col = row + 1; col < p; ++col) {
                    value -= static_cast<long double>(
                        node.fp64.u[row * dim + col]) *
                        x[node.front_indices[col]];
                }
                for (std::uint32_t col = p; col < dim; ++col) {
                    value -= static_cast<long double>(
                        node.fp64.u[row * dim + col]) *
                        x[node.front_indices[col]];
                }
                const auto diagonal = node.fp64.u[row * dim + row];
                if (diagonal == 0.0) {
                    throw NumericFailure("zero FP64 U diagonal in backward solve");
                }
                x[node.front_indices[row]] =
                    static_cast<double>(value / diagonal);
            }

            const auto begin = cursor;
            cursor += solve_node_cycles(node.task, config, true);
            stats.operations.push_back({
                node_id, OpType::SolveBackward, 0, 0, 0,
                p, 1, dim, begin, begin, cursor,
            });
        }

        result.valid = true;
        result.x_permuted = std::move(x);
        result.x_original = restore_original_coordinates(
            result.x_permuted,
            memory.permutation,
            memory.column_scale_exponents);
        result.cycles = cursor - start_cycle;
        result.scaled_relative_residual = relative_residual(
            memory.reconstruct_original_fp64(),
            result.x_permuted,
            memory.rhs_fp64);
        result.relative_residual = relative_residual(
            memory.original_matrix_fp64,
            result.x_original,
            memory.original_rhs_fp64);
        result.initial_relative_residual = result.relative_residual;
        result.residual_history.push_back(result.relative_residual);
        result.componentwise_backward_error =
            componentwise_backward_error(
                memory.original_matrix_fp64,
                result.x_original,
                memory.original_rhs_fp64);
        result.relative_solution_error = relative_solution_error(
            result.x_original, memory.original_solution_reference);
    } catch (const std::exception& exception) {
        result.failure_reason = exception.what();
    }
    return result;
}

inline std::int64_t clamp_vector_acc(
    __int128 value,
    unsigned accumulator_bits,
    QuantStats& stats) {
    const __int128 maximum =
        accumulator_bits == 64 ?
            std::numeric_limits<std::int64_t>::max() :
            ((__int128{1} << (accumulator_bits - 1)) - 1);
    const __int128 minimum =
        accumulator_bits == 64 ?
            std::numeric_limits<std::int64_t>::min() :
            -(__int128{1} << (accumulator_bits - 1));
    if (value > maximum) {
        ++stats.vector_overflow_count;
        return static_cast<std::int64_t>(maximum);
    }
    if (value < minimum) {
        ++stats.vector_overflow_count;
        return static_cast<std::int64_t>(minimum);
    }
    return static_cast<std::int64_t>(value);
}

inline std::int64_t round_shift_i128(
    __int128 value,
    int shift,
    unsigned accumulator_bits,
    QuantStats& stats) {
    ++stats.vector_shift_count;
    if (shift < 0) {
        const auto left = static_cast<unsigned>(-shift);
        if (left >= 127) {
            ++stats.vector_overflow_count;
            return value < 0 ?
                std::numeric_limits<std::int64_t>::min() :
                std::numeric_limits<std::int64_t>::max();
        }
        return clamp_vector_acc(
            value * (__int128{1} << left), accumulator_bits, stats);
    }
    if (shift == 0) {
        return clamp_vector_acc(value, accumulator_bits, stats);
    }
    if (shift >= 127) {
        if (value != 0) ++stats.vector_drop_count;
        return 0;
    }
    const auto magnitude = value < 0 ? -value : value;
    const auto rounded =
        (magnitude + (__int128{1} << (shift - 1))) >> shift;
    const auto signed_value = value < 0 ? -rounded : rounded;
    const auto result =
        clamp_vector_acc(signed_value, accumulator_bits, stats);
    if (value != 0 && result == 0) ++stats.vector_drop_count;
    return result;
}

inline __int128 round_shift_i128_value(
    __int128 value,
    unsigned shift) {
    if (shift == 0) return value;
    if (shift >= 127) return 0;
    const auto magnitude = value < 0 ? -value : value;
    auto rounded =
        (magnitude + (__int128{1} << (shift - 1))) >> shift;
    return value < 0 ? -rounded : rounded;
}

inline unsigned shift_to_fit_i128(
    __int128 value,
    unsigned use_bits) {
    const auto limit = (__int128{1} << use_bits) - 1;
    auto magnitude = value < 0 ? -value : value;
    unsigned shift = 0;
    while (magnitude > limit) {
        magnitude = (magnitude + 1) >> 1;
        ++shift;
    }
    return shift;
}

inline SolveMetrics solve_fixed_rhs(
    SystemMemory& memory,
    const ModelConfig& config,
    SimulationStats& stats,
    std::uint64_t start_cycle,
    const std::vector<std::int32_t>& rhs_q,
    std::int16_t rhs_exp) {
    SolveMetrics result{};
    try {
        const auto n = memory.matrix_dim();
        if (rhs_q.size() != n) {
            throw NumericFailure("fixed RHS length mismatch");
        }
        std::vector<std::int64_t> work(
            rhs_q.begin(), rhs_q.end());
        std::vector<std::int64_t> y(n, 0);
        auto& vector_stats = result.vector_stats;
        std::uint64_t cursor = start_cycle;

        for (std::uint16_t node_id = 0;
             node_id < memory.size(); ++node_id) {
            const auto& node = memory.at(node_id);
            if (!node.fixed.valid) {
                throw NumericFailure("fixed factor is missing for forward solve");
            }
            const auto p = node.task.pivot_dim;
            const auto dim = node.task.total_dim;
            std::vector<std::int64_t> pivot_rhs(p);
            for (std::uint32_t row = 0; row < p; ++row) {
                pivot_rhs[row] =
                    work[node.front_indices[node.fixed.pvec[row]]];
            }
            for (std::uint32_t row = 0; row < p; ++row) {
                __int128 value = pivot_rhs[row];
                for (std::uint32_t col = 0; col < row; ++col) {
                    const __int128 product =
                        static_cast<__int128>(
                            node.fixed.l[row * p + col]) *
                        y[node.front_indices[col]];
                    value -= round_shift_i128(
                        product, config.frac_bits,
                        config.accumulator_bits, vector_stats);
                }
                y[node.front_indices[row]] =
                    clamp_vector_acc(
                        value, config.accumulator_bits, vector_stats);
            }
            for (std::uint32_t row = p; row < dim; ++row) {
                __int128 value = work[node.front_indices[row]];
                for (std::uint32_t col = 0; col < p; ++col) {
                    const __int128 product =
                        static_cast<__int128>(
                            node.fixed.l[row * p + col]) *
                        y[node.front_indices[col]];
                    value -= round_shift_i128(
                        product, config.frac_bits,
                        config.accumulator_bits, vector_stats);
                }
                work[node.front_indices[row]] =
                    clamp_vector_acc(
                        value, config.accumulator_bits, vector_stats);
            }
            const auto begin = cursor;
            cursor += solve_node_cycles(node.task, config, false);
            stats.operations.push_back({
                node_id, OpType::SolveForward, 0, 0, 0,
                p, 1, dim, begin, begin, cursor,
            });
        }

        std::vector<std::int64_t> x_q(n, 0);
        std::vector<std::int16_t> x_exp(memory.size(), 0);
        std::vector<std::uint16_t> variable_owner(n, 0);
        for (std::uint16_t node_id = 0;
             node_id < memory.size(); ++node_id) {
            const auto& node = memory.at(node_id);
            for (std::uint32_t local = 0;
                 local < node.task.pivot_dim; ++local) {
                variable_owner[node.front_indices[local]] = node_id;
            }
        }

        for (std::size_t reverse = memory.size(); reverse > 0; --reverse) {
            const auto node_id = static_cast<std::uint16_t>(reverse - 1);
            const auto& node = memory.at(node_id);
            const auto p = node.task.pivot_dim;
            const auto dim = node.task.total_dim;
            const int base_node_x_exp =
                static_cast<int>(rhs_exp) -
                static_cast<int>(node.fixed.u_exponent) -
                static_cast<int>(config.frac_bits);
            if (base_node_x_exp <
                    std::numeric_limits<std::int16_t>::min() ||
                base_node_x_exp >
                    std::numeric_limits<std::int16_t>::max()) {
                throw NumericFailure("solution exponent exceeds int16");
            }
            x_exp[node_id] =
                static_cast<std::int16_t>(base_node_x_exp);

            for (std::uint32_t reverse_row = p; reverse_row > 0; --reverse_row) {
                const auto row = reverse_row - 1;
                __int128 residual_q = y[node.front_indices[row]];
                for (std::uint32_t col = row + 1; col < dim; ++col) {
                    const auto global_col = node.front_indices[col];
                    const auto owner = variable_owner[global_col];
                    const __int128 product =
                        static_cast<__int128>(
                            node.fixed.u[row * dim + col]) *
                        x_q[global_col];
                    const int product_exp =
                        static_cast<int>(node.fixed.u_exponent) +
                        static_cast<int>(x_exp[owner]);
                    const int align_shift =
                        static_cast<int>(rhs_exp) - product_exp;
                    residual_q -= round_shift_i128(
                        product, align_shift,
                        config.accumulator_bits, vector_stats);
                }
                const auto diagonal = node.fixed.u[row * dim + row];
                if (diagonal == 0) {
                    ++vector_stats.divide_by_zero_count;
                    throw NumericFailure("zero fixed U diagonal in backward solve");
                }
                const __int128 scaled =
                    residual_q *
                    (__int128{1} << config.frac_bits);
                auto quotient =
                    round_div_signed_wide(scaled, diagonal);
                const auto existing_shift =
                    static_cast<unsigned>(
                        static_cast<int>(x_exp[node_id]) -
                        base_node_x_exp);
                quotient =
                    round_shift_i128_value(
                        quotient, existing_shift);
                const auto extra_shift =
                    shift_to_fit_i128(
                        quotient, config.vector_use_bits);
                if (extra_shift != 0) {
                    for (std::uint32_t solved = row + 1;
                         solved < p; ++solved) {
                        const auto variable =
                            node.front_indices[solved];
                        x_q[variable] = round_shift_i128(
                            x_q[variable],
                            static_cast<int>(extra_shift),
                            config.accumulator_bits,
                            vector_stats);
                    }
                    const auto new_exp =
                        static_cast<int>(x_exp[node_id]) +
                        static_cast<int>(extra_shift);
                    if (new_exp >
                        std::numeric_limits<std::int16_t>::max()) {
                        throw NumericFailure(
                            "renormalized solution exponent exceeds int16");
                    }
                    x_exp[node_id] =
                        static_cast<std::int16_t>(new_exp);
                    quotient = round_shift_i128_value(
                        quotient, extra_shift);
                    ++vector_stats.solution_renormalize_count;
                }
                x_q[node.front_indices[row]] =
                    clamp_vector_acc(
                        quotient,
                        config.accumulator_bits, vector_stats);
            }
            const auto begin = cursor;
            cursor += solve_node_cycles(node.task, config, true);
            stats.operations.push_back({
                node_id, OpType::SolveBackward, 0, 0, 0,
                p, 1, dim, begin, begin, cursor,
            });
        }

        result.x_permuted.resize(n);
        for (std::size_t variable = 0; variable < n; ++variable) {
            result.x_permuted[variable] =
                std::ldexp(
                    static_cast<double>(x_q[variable]),
                    x_exp[variable_owner[variable]]);
        }
        std::vector<double> rhs_fixed(n);
        const auto rhs_scale = std::ldexp(1.0, rhs_exp);
        for (std::size_t i = 0; i < n; ++i) {
            rhs_fixed[i] =
                static_cast<double>(rhs_q[i]) * rhs_scale;
        }
        result.valid = true;
        result.x_mantissa = std::move(x_q);
        result.exponent_by_node = std::move(x_exp);
        result.x_original = restore_original_coordinates(
            result.x_permuted,
            memory.permutation,
            memory.column_scale_exponents);
        result.cycles = cursor - start_cycle;
        result.scaled_relative_residual = relative_residual(
            memory.reconstruct_original_fixed(),
            result.x_permuted,
            rhs_fixed);
        result.relative_residual =
            result.scaled_relative_residual;
    } catch (const std::exception& exception) {
        result.failure_reason = exception.what();
    }
    return result;
}

inline bool has_precision_rescued_factor(
    const SystemMemory& memory) {
    for (std::uint16_t node_id = 0;
         node_id < memory.size(); ++node_id) {
        if (memory.at(node_id).fixed.precision_rescued) return true;
    }
    return false;
}

inline SolveMetrics solve_hybrid_fixed_rhs(
    SystemMemory& memory,
    const ModelConfig& config,
    SimulationStats& stats,
    std::uint64_t start_cycle,
    const std::vector<std::int32_t>& rhs_q,
    std::int16_t rhs_exp) {
    SolveMetrics result{};
    try {
        const auto n = memory.matrix_dim();
        if (rhs_q.size() != n) {
            throw NumericFailure("hybrid RHS length mismatch");
        }
        const auto rhs_scale = std::ldexp(1.0, rhs_exp);
        std::vector<double> work(n, 0.0);
        for (std::size_t index = 0; index < n; ++index) {
            work[index] =
                static_cast<double>(rhs_q[index]) * rhs_scale;
        }
        std::vector<double> y(n, 0.0);
        std::uint64_t cursor = start_cycle;

        for (std::uint16_t node_id = 0;
             node_id < memory.size(); ++node_id) {
            const auto& node = memory.at(node_id);
            const auto p = node.task.pivot_dim;
            const auto dim = node.task.total_dim;
            const auto rescued = node.fixed.precision_rescued;
            if (!node.fixed.valid ||
                (rescued && !node.fixed_rescue_fp64.valid)) {
                throw NumericFailure(
                    "hybrid factor is missing for forward solve");
            }
            const auto fixed_l_scale =
                std::ldexp(
                    1.0, -static_cast<int>(config.frac_bits));
            const auto l_value =
                [&](std::uint32_t row,
                    std::uint32_t col) -> double {
                    return rescued ?
                        node.fixed_rescue_fp64.l[row * p + col] :
                        static_cast<double>(
                            node.fixed.l[row * p + col]) *
                            fixed_l_scale;
                };
            std::vector<double> pivot_rhs(p, 0.0);
            for (std::uint32_t row = 0; row < p; ++row) {
                pivot_rhs[row] =
                    work[
                        node.front_indices[
                            node.fixed.pvec[row]]];
            }
            for (std::uint32_t row = 0; row < p; ++row) {
                long double value = pivot_rhs[row];
                for (std::uint32_t col = 0; col < row; ++col) {
                    value -=
                        static_cast<long double>(
                            l_value(row, col)) *
                        y[node.front_indices[col]];
                }
                y[node.front_indices[row]] =
                    static_cast<double>(value);
            }
            for (std::uint32_t row = p; row < dim; ++row) {
                long double value =
                    work[node.front_indices[row]];
                for (std::uint32_t col = 0; col < p; ++col) {
                    value -=
                        static_cast<long double>(
                            l_value(row, col)) *
                        y[node.front_indices[col]];
                }
                work[node.front_indices[row]] =
                    static_cast<double>(value);
            }
            const auto begin = cursor;
            cursor += solve_node_cycles(
                node.task, config, false);
            stats.operations.push_back({
                node_id, OpType::SolveForward, 0, 0, 0,
                p, 1, dim, begin, begin, cursor,
            });
        }

        std::vector<double> x(n, 0.0);
        for (std::size_t reverse = memory.size();
             reverse > 0; --reverse) {
            const auto node_id =
                static_cast<std::uint16_t>(reverse - 1);
            const auto& node = memory.at(node_id);
            const auto p = node.task.pivot_dim;
            const auto dim = node.task.total_dim;
            const auto rescued =
                node.fixed.precision_rescued;
            const auto fixed_u_scale =
                std::ldexp(1.0, node.fixed.u_exponent);
            const auto u_value =
                [&](std::uint32_t row,
                    std::uint32_t col) -> double {
                    return rescued ?
                        node.fixed_rescue_fp64.u[row * dim + col] :
                        static_cast<double>(
                            node.fixed.u[row * dim + col]) *
                            fixed_u_scale;
                };
            for (std::uint32_t reverse_row = p;
                 reverse_row > 0; --reverse_row) {
                const auto row = reverse_row - 1;
                long double value =
                    y[node.front_indices[row]];
                for (std::uint32_t col = row + 1;
                     col < dim; ++col) {
                    value -=
                        static_cast<long double>(
                            u_value(row, col)) *
                        x[node.front_indices[col]];
                }
                const auto diagonal = u_value(row, row);
                if (diagonal == 0.0) {
                    throw NumericFailure(
                        "zero hybrid U diagonal in backward solve");
                }
                x[node.front_indices[row]] =
                    static_cast<double>(value / diagonal);
            }
            const auto begin = cursor;
            cursor += solve_node_cycles(
                node.task, config, true);
            stats.operations.push_back({
                node_id, OpType::SolveBackward, 0, 0, 0,
                p, 1, dim, begin, begin, cursor,
            });
        }
        result.valid = true;
        result.used_precision_rescue = true;
        result.refined_solution = true;
        result.x_permuted = std::move(x);
        result.x_original = restore_original_coordinates(
            result.x_permuted,
            memory.permutation,
            memory.column_scale_exponents);
        result.cycles = cursor - start_cycle;
    } catch (const std::exception& exception) {
        result.failure_reason = exception.what();
    }
    return result;
}

inline SolveMetrics solve_low_precision_rhs(
    SystemMemory& memory,
    const ModelConfig& config,
    SimulationStats& stats,
    std::uint64_t start_cycle,
    const std::vector<std::int32_t>& rhs_q,
    std::int16_t rhs_exp) {
    return has_precision_rescued_factor(memory) ?
        solve_hybrid_fixed_rhs(
            memory, config, stats, start_cycle, rhs_q, rhs_exp) :
        solve_fixed_rhs(
            memory, config, stats, start_cycle, rhs_q, rhs_exp);
}

inline std::pair<std::vector<std::int32_t>, std::int16_t>
quantize_rhs_vector(
    const std::vector<double>& values,
    std::int32_t q_limit) {
    double max_abs = 0.0;
    for (const auto value : values) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    int exponent = 0;
    if (max_abs != 0.0) {
        exponent = static_cast<int>(
            std::ceil(std::log2(
                max_abs / static_cast<double>(q_limit))));
    }
    if (exponent < std::numeric_limits<std::int16_t>::min() ||
        exponent > std::numeric_limits<std::int16_t>::max()) {
        throw NumericFailure("refinement RHS exponent exceeds int16");
    }
    std::vector<std::int32_t> quantized;
    quantized.reserve(values.size());
    for (const auto value : values) {
        const auto scaled =
            std::nearbyint(std::ldexp(value, -exponent));
        if (!std::isfinite(scaled) ||
            scaled > std::numeric_limits<std::int32_t>::max() ||
            scaled < std::numeric_limits<std::int32_t>::min()) {
            throw NumericFailure("refinement RHS quantization overflow");
        }
        quantized.push_back(static_cast<std::int32_t>(scaled));
    }
    return {
        std::move(quantized),
        static_cast<std::int16_t>(exponent),
    };
}

inline void accumulate_quant_stats(
    QuantStats& destination,
    const QuantStats& source) {
    destination.vector_shift_count += source.vector_shift_count;
    destination.vector_drop_count += source.vector_drop_count;
    destination.vector_overflow_count += source.vector_overflow_count;
    destination.divide_by_zero_count += source.divide_by_zero_count;
    destination.solution_renormalize_count +=
        source.solution_renormalize_count;
}

inline std::vector<double> matrix_vector_product(
    const std::vector<double>& matrix,
    const std::vector<double>& vector) {
    const auto n = vector.size();
    if (matrix.size() != n * n) {
        throw std::invalid_argument("matrix_vector_product dimension mismatch");
    }
    std::vector<double> result(n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        long double value = 0.0;
        for (std::size_t col = 0; col < n; ++col) {
            value += static_cast<long double>(
                matrix[row * n + col]) * vector[col];
        }
        result[row] = static_cast<double>(value);
    }
    return result;
}

inline SolveMetrics solve_fixed(
    SystemMemory& memory,
    const ModelConfig& config,
    SimulationStats& stats,
    std::uint64_t start_cycle) {
    auto result = solve_low_precision_rhs(
        memory,
        config,
        stats,
        start_cycle,
        memory.rhs_q,
        memory.rhs_exp);
    if (!result.valid) return result;

    const auto scaled_matrix = memory.reconstruct_original_fixed();
    std::vector<double> rhs_fixed(memory.matrix_dim());
    const auto rhs_scale = std::ldexp(1.0, memory.rhs_exp);
    for (std::size_t index = 0; index < rhs_fixed.size(); ++index) {
        rhs_fixed[index] =
            static_cast<double>(memory.rhs_q[index]) * rhs_scale;
    }
    result.scaled_relative_residual = relative_residual(
        scaled_matrix, result.x_permuted, rhs_fixed);
    result.relative_residual = relative_residual(
        memory.original_matrix_fp64,
        result.x_original,
        memory.original_rhs_fp64);
    result.initial_relative_residual = result.relative_residual;
    result.residual_history = {result.relative_residual};
    result.componentwise_backward_error =
        componentwise_backward_error(
            memory.original_matrix_fp64,
            result.x_original,
            memory.original_rhs_fp64);
    result.relative_solution_error = relative_solution_error(
        result.x_original, memory.original_solution_reference);

    if (!config.iterative_refinement) {
        result.refinement_stop_reason = "disabled";
        return result;
    }
    if (result.relative_residual <= config.ir_tolerance) {
        result.refinement_converged = true;
        result.refinement_stop_reason = "initial solution met tolerance";
        return result;
    }

    auto cursor = start_cycle + result.cycles;
    for (unsigned iteration = 0;
         iteration < config.ir_max_iters; ++iteration) {
        const auto residual = residual_vector(
            memory.original_matrix_fp64,
            result.x_original,
            memory.original_rhs_fp64);
        std::vector<double> scaled_permuted_rhs(
            memory.matrix_dim(), 0.0);
        for (std::size_t permuted = 0;
             permuted < memory.matrix_dim(); ++permuted) {
            const auto original = memory.permutation[permuted];
            scaled_permuted_rhs[permuted] =
                std::ldexp(
                    residual[original],
                    memory.row_scale_exponents[original]);
        }

        SolveMetrics correction{};
        try {
            auto correction_rhs = quantize_rhs_vector(
                scaled_permuted_rhs, config.q_limit());
            correction = solve_low_precision_rhs(
                memory,
                config,
                stats,
                cursor,
                correction_rhs.first,
                correction_rhs.second);
        } catch (const std::exception& exception) {
            result.refinement_stop_reason =
                std::string("correction setup failed: ") + exception.what();
            break;
        }
        result.cycles += correction.cycles;
        cursor += correction.cycles;
        accumulate_quant_stats(
            result.vector_stats, correction.vector_stats);
        if (!correction.valid) {
            result.refinement_stop_reason =
                "correction solve failed: " + correction.failure_reason;
            break;
        }

        const auto a_delta = matrix_vector_product(
            memory.original_matrix_fp64,
            correction.x_original);
        long double numerator = 0.0;
        long double denominator = 0.0;
        for (std::size_t index = 0; index < residual.size(); ++index) {
            numerator +=
                static_cast<long double>(residual[index]) *
                a_delta[index];
            denominator +=
                static_cast<long double>(a_delta[index]) *
                a_delta[index];
        }
        if (denominator == 0.0L || numerator <= 0.0L) {
            result.refinement_stop_reason =
                "correction is not a descent direction";
            break;
        }
        const auto alpha = std::clamp(
            static_cast<double>(numerator / denominator),
            0.0,
            1.0);
        std::vector<double> candidate_original(
            memory.matrix_dim(), 0.0);
        std::vector<double> candidate_permuted(
            memory.matrix_dim(), 0.0);
        for (std::size_t index = 0;
             index < memory.matrix_dim(); ++index) {
            candidate_original[index] =
                result.x_original[index] +
                alpha * correction.x_original[index];
            candidate_permuted[index] =
                result.x_permuted[index] +
                alpha * correction.x_permuted[index];
        }
        const auto candidate_residual = relative_residual(
            memory.original_matrix_fp64,
            candidate_original,
            memory.original_rhs_fp64);

        const auto residual_begin = cursor;
        const auto residual_cycles = ceil_div_u64(
            static_cast<std::uint64_t>(memory.matrix_dim()) *
                memory.matrix_dim(),
            config.ir_residual_macs_per_cycle);
        cursor += residual_cycles;
        result.cycles += residual_cycles;
        stats.operations.push_back({
            ROOT_PARENT_ID,
            OpType::SolveResidual,
            iteration, 0, 0,
            memory.matrix_dim(), 1, memory.matrix_dim(),
            residual_begin, residual_begin, cursor,
        });

        if (!std::isfinite(candidate_residual) ||
            candidate_residual >=
                result.relative_residual *
                    (1.0 - config.ir_min_improvement)) {
            result.refinement_stop_reason =
                "safeguard rejected non-improving correction";
            break;
        }
        result.x_original = std::move(candidate_original);
        result.x_permuted = std::move(candidate_permuted);
        result.relative_residual = candidate_residual;
        result.residual_history.push_back(candidate_residual);
        ++result.refinement_iterations;
        result.refined_solution = true;
        if (candidate_residual <= config.ir_tolerance) {
            result.refinement_converged = true;
            result.refinement_stop_reason = "tolerance reached";
            break;
        }
    }
    if (result.refinement_stop_reason.empty()) {
        result.refinement_stop_reason =
            result.refinement_converged ?
                "tolerance reached" : "iteration limit reached";
    }
    result.scaled_relative_residual = relative_residual(
        scaled_matrix, result.x_permuted, rhs_fixed);
    result.componentwise_backward_error =
        componentwise_backward_error(
            memory.original_matrix_fp64,
            result.x_original,
            memory.original_rhs_fp64);
    result.relative_solution_error = relative_solution_error(
        result.x_original, memory.original_solution_reference);
    if (result.refined_solution) {
        result.x_mantissa.clear();
        result.exponent_by_node.clear();
    }
    return result;
}

inline double factorization_relative_error_fp64(
    const SystemMemory& memory) {
    long double numerator = 0.0;
    long double denominator = 0.0;
    for (std::uint16_t node_id = 0; node_id < memory.size(); ++node_id) {
        const auto& node = memory.at(node_id);
        if (!node.fp64.valid) continue;
        const auto p = node.task.pivot_dim;
        const auto dim = node.task.total_dim;
        const auto u = dim - p;
        for (std::uint32_t row = 0; row < dim; ++row) {
            for (std::uint32_t col = 0; col < dim; ++col) {
                const auto source_row =
                    row < p ? node.fp64.pvec[row] : row;
                const auto expected =
                    node.assembled_fp64[source_row * dim + col];
                long double reconstructed = 0.0;
                for (std::uint32_t k = 0; k < p; ++k) {
                    reconstructed +=
                        static_cast<long double>(
                            node.fp64.l[row * p + k]) *
                        node.fp64.u[k * dim + col];
                }
                if (row >= p && col >= p) {
                    reconstructed +=
                        node.fp64.update[(row - p) * u + col - p];
                }
                const auto error =
                    static_cast<long double>(expected) - reconstructed;
                numerator += error * error;
                denominator +=
                    static_cast<long double>(expected) * expected;
            }
        }
    }
    return std::sqrt(static_cast<double>(numerator)) /
           std::max(std::sqrt(static_cast<double>(denominator)), 1e-300);
}

inline double factorization_relative_error_fixed(
    const SystemMemory& memory,
    const ModelConfig& config) {
    long double numerator = 0.0;
    long double denominator = 0.0;
    const auto l_scale = std::ldexp(1.0, -static_cast<int>(config.frac_bits));
    for (std::uint16_t node_id = 0; node_id < memory.size(); ++node_id) {
        const auto& node = memory.at(node_id);
        if (!node.fixed.valid) continue;
        const auto p = node.task.pivot_dim;
        const auto dim = node.task.total_dim;
        const auto u = dim - p;
        const auto matrix_scale =
            std::ldexp(1.0, node.fixed.u_exponent);
        const auto update_scale =
            std::ldexp(1.0, node.fixed.update_exponent);
        for (std::uint32_t row = 0; row < dim; ++row) {
            for (std::uint32_t col = 0; col < dim; ++col) {
                const auto source_row =
                    row < p ? node.fixed.pvec[row] : row;
                const auto expected =
                    static_cast<double>(
                        node.assembled_q[source_row * dim + col]) *
                    std::ldexp(1.0, node.assembled_exp);
                long double reconstructed = 0.0;
                for (std::uint32_t k = 0; k < p; ++k) {
                    if (node.fixed.precision_rescued &&
                        node.fixed_rescue_fp64.valid) {
                        reconstructed +=
                            static_cast<long double>(
                                node.fixed_rescue_fp64.l[
                                    row * p + k]) *
                            node.fixed_rescue_fp64.u[
                                k * dim + col];
                    } else {
                        reconstructed +=
                            static_cast<long double>(
                                node.fixed.l[row * p + k]) * l_scale *
                            static_cast<double>(
                                node.fixed.u[k * dim + col]) * matrix_scale;
                    }
                }
                if (row >= p && col >= p) {
                    reconstructed +=
                        static_cast<double>(
                            node.fixed.update[(row - p) * u + col - p]) *
                        update_scale;
                }
                const auto error =
                    static_cast<long double>(expected) - reconstructed;
                numerator += error * error;
                denominator +=
                    static_cast<long double>(expected) * expected;
            }
        }
    }
    return std::sqrt(static_cast<double>(numerator)) /
           std::max(std::sqrt(static_cast<double>(denominator)), 1e-300);
}

}  // namespace hw
