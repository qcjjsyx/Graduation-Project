#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "artifact.hpp"
#include "model_types.hpp"
#include "quantization.hpp"

namespace hw {

struct ChildMap {
    std::uint16_t child_id{0};
    std::vector<std::uint32_t> row_map{};
    std::vector<std::uint32_t> col_map{};
};

struct FixedFactor {
    std::vector<std::int32_t> l{};
    std::vector<std::int32_t> u{};
    std::vector<std::uint16_t> pvec{};
    std::vector<std::int32_t> update{};
    std::int16_t u_exponent{0};
    std::int16_t update_exponent{0};
    std::vector<std::int16_t> u_tile_exponents{};
    std::vector<std::int16_t> update_tile_exponents{};
    unsigned tile_size{0};
    bool precision_rescued{false};
    bool valid{false};
};

struct Fp64Factor {
    std::vector<double> l{};
    std::vector<double> u{};
    std::vector<std::uint16_t> pvec{};
    std::vector<double> update{};
    bool valid{false};
};

struct NodeStorage {
    NodeTask task{};
    std::vector<std::uint32_t> front_indices{};
    std::vector<ChildMap> child_maps{};

    std::vector<std::int32_t> local_q{};
    std::int16_t local_exp{0};
    std::vector<std::int16_t> local_tile_exponents{};
    std::vector<double> local_fp64{};

    std::vector<std::int64_t> assembly_acc{};
    std::vector<std::int32_t> assembled_q{};
    std::int16_t assembled_exp{0};
    std::vector<std::int16_t> assembled_tile_exponents{};
    std::vector<double> assembled_fp64{};
    QuantStats quant_stats{};

    FixedFactor fixed{};
    Fp64Factor fp64{};
    Fp64Factor fixed_rescue_fp64{};
    NodeStatus status{NodeStatus::Pending};
    std::string failure_reason{};
};

class SystemMemory {
public:
    SystemMemory(
        std::uint32_t matrix_dim,
        std::int32_t q_limit,
        unsigned accumulator_bits,
        unsigned bfp_tile_size = 0)
        : matrix_dim_(matrix_dim),
          q_limit_(q_limit),
          accumulator_bits_(accumulator_bits),
          bfp_tile_size_(bfp_tile_size) {
        if (matrix_dim_ == 0 || q_limit_ <= 0 ||
            accumulator_bits_ < 32 || accumulator_bits_ > 64 ||
            (bfp_tile_size_ != 0 && bfp_tile_size_ != 16)) {
            throw std::invalid_argument("invalid SystemMemory configuration");
        }
    }

    void add_node(NodeStorage node) {
        const auto node_id = node.task.node_id;
        const auto expected =
            static_cast<std::size_t>(node.task.total_dim) * node.task.total_dim;
        if (node.local_q.size() != expected ||
            node.local_fp64.size() != expected ||
            node.front_indices.size() != node.task.total_dim) {
            throw std::invalid_argument("node local front dimensions do not match task");
        }
        if (!nodes_.emplace(node_id, std::move(node)).second) {
            throw std::invalid_argument("duplicate node id in SystemMemory");
        }
    }

    NodeStorage& at(std::uint16_t node_id) {
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            throw std::out_of_range("unknown node id " + std::to_string(node_id));
        }
        return it->second;
    }

    const NodeStorage& at(std::uint16_t node_id) const {
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            throw std::out_of_range("unknown node id " + std::to_string(node_id));
        }
        return it->second;
    }

    void assemble_fixed(std::uint16_t node_id) {
        if (bfp_tile_size_ != 0) {
            assemble_fixed_tile_bfp(node_id);
            return;
        }
        auto& node = at(node_id);
        const auto dim = node.task.total_dim;
        std::int16_t assembly_exp = node.local_exp;
        for (const auto& mapping : node.child_maps) {
            const auto& child = at(mapping.child_id);
            if (!child.fixed.valid) {
                throw std::runtime_error(
                    "fixed parent assembly observed an uncommitted child update");
            }
            assembly_exp =
                std::max(assembly_exp, child.fixed.update_exponent);
        }

        QuantStats stats{};
        stats.assembly_exp = assembly_exp;
        std::vector<std::int64_t> accumulator(
            static_cast<std::size_t>(dim) * dim, 0);
        const auto add = [&](std::size_t index, std::int64_t value) {
            accumulator[index] =
                saturating_add(accumulator[index], value, stats);
        };

        const int local_shift = assembly_exp - node.local_exp;
        stats.align_shift_max =
            std::max(stats.align_shift_max, static_cast<unsigned>(local_shift));
        for (std::size_t index = 0; index < node.local_q.size(); ++index) {
            const auto aligned = round_shift_signed(node.local_q[index], local_shift);
            if (node.local_q[index] != 0 && aligned == 0) {
                ++stats.align_drop_count;
            }
            add(index, aligned);
        }

        for (const auto& mapping : node.child_maps) {
            const auto& child = at(mapping.child_id);
            if (mapping.row_map.size() != mapping.col_map.size()) {
                throw std::runtime_error("row/column map length mismatch");
            }
            const auto child_update_dim =
                child.task.total_dim - child.task.pivot_dim;
            const int shift =
                assembly_exp - child.fixed.update_exponent;
            stats.align_shift_max =
                std::max(stats.align_shift_max, static_cast<unsigned>(shift));
            for (std::size_t row = 0; row < mapping.row_map.size(); ++row) {
                for (std::size_t col = 0; col < mapping.row_map.size(); ++col) {
                    const auto src_row = mapping.row_map[row];
                    const auto src_col = mapping.row_map[col];
                    const auto dst_row = mapping.col_map[row];
                    const auto dst_col = mapping.col_map[col];
                    if (src_row >= child_update_dim || src_col >= child_update_dim ||
                        dst_row >= dim || dst_col >= dim) {
                        throw std::runtime_error("map table index is out of bounds");
                    }
                    const auto source =
                        child.fixed.update[src_row * child_update_dim + src_col];
                    const auto aligned = round_shift_signed(source, shift);
                    if (source != 0 && aligned == 0) {
                        ++stats.align_drop_count;
                    }
                    add(dst_row * dim + dst_col, aligned);
                }
            }
        }

        auto quantized =
            requantize(accumulator, assembly_exp, q_limit_, stats);
        node.assembly_acc = std::move(accumulator);
        node.assembled_q = std::move(quantized.first);
        node.quant_stats = quantized.second;
        node.assembled_exp = node.quant_stats.node_exp;
    }

    void assemble_fp64(std::uint16_t node_id) {
        auto& node = at(node_id);
        const auto dim = node.task.total_dim;
        node.assembled_fp64 = node.local_fp64;
        for (const auto& mapping : node.child_maps) {
            const auto& child = at(mapping.child_id);
            if (!child.fp64.valid) {
                throw std::runtime_error(
                    "FP64 parent assembly observed an uncommitted child update");
            }
            if (mapping.row_map.size() != mapping.col_map.size()) {
                throw std::runtime_error("row/column map length mismatch");
            }
            const auto child_update_dim =
                child.task.total_dim - child.task.pivot_dim;
            for (std::size_t row = 0; row < mapping.row_map.size(); ++row) {
                for (std::size_t col = 0; col < mapping.row_map.size(); ++col) {
                    const auto src_row = mapping.row_map[row];
                    const auto src_col = mapping.row_map[col];
                    const auto dst_row = mapping.col_map[row];
                    const auto dst_col = mapping.col_map[col];
                    if (src_row >= child_update_dim || src_col >= child_update_dim ||
                        dst_row >= dim || dst_col >= dim) {
                        throw std::runtime_error("map table index is out of bounds");
                    }
                    node.assembled_fp64[dst_row * dim + dst_col] +=
                        child.fp64.update[src_row * child_update_dim + src_col];
                }
            }
        }
    }

    std::vector<double> assemble_fixed_rescue_fp64(
        std::uint16_t node_id) const {
        const auto& node = at(node_id);
        const auto dim = node.task.total_dim;
        const auto expected = static_cast<std::size_t>(dim) * dim;
        if (node.assembled_q.size() != expected) {
            throw std::runtime_error(
                "precision rescue requires an assembled fixed front");
        }
        std::vector<double> front(expected, 0.0);
        for (std::uint32_t row = 0; row < dim; ++row) {
            for (std::uint32_t col = 0; col < dim; ++col) {
                const auto exponent = exponent_at(
                    node.assembled_tile_exponents,
                    node.assembled_exp,
                    row,
                    col,
                    dim,
                    dim);
                const auto index = static_cast<std::size_t>(row) * dim + col;
                front[index] =
                    static_cast<double>(node.assembled_q[index]) *
                    std::ldexp(1.0, exponent);
            }
        }
        return front;
    }

    std::vector<double> reconstruct_original_fp64() const {
        std::vector<double> matrix(
            static_cast<std::size_t>(matrix_dim_) * matrix_dim_, 0.0);
        for (const auto& [node_id, node] : nodes_) {
            (void)node_id;
            const auto dim = node.task.total_dim;
            for (std::uint32_t row = 0; row < dim; ++row) {
                for (std::uint32_t col = 0; col < dim; ++col) {
                    const auto global_row = node.front_indices[row];
                    const auto global_col = node.front_indices[col];
                    matrix[global_row * matrix_dim_ + global_col] +=
                        node.local_fp64[row * dim + col];
                }
            }
        }
        return matrix;
    }

    std::vector<double> reconstruct_original_fixed() const {
        std::vector<double> matrix(
            static_cast<std::size_t>(matrix_dim_) * matrix_dim_, 0.0);
        for (const auto& [node_id, node] : nodes_) {
            (void)node_id;
            const auto dim = node.task.total_dim;
            for (std::uint32_t row = 0; row < dim; ++row) {
                for (std::uint32_t col = 0; col < dim; ++col) {
                    const auto global_row = node.front_indices[row];
                    const auto global_col = node.front_indices[col];
                    matrix[global_row * matrix_dim_ + global_col] +=
                        static_cast<double>(node.local_q[row * dim + col]) *
                        std::ldexp(
                            1.0,
                            exponent_at(
                                node.local_tile_exponents,
                                node.local_exp,
                                row,
                                col,
                                dim,
                                dim));
                }
            }
        }
        return matrix;
    }

    std::size_t size() const { return nodes_.size(); }
    std::uint32_t matrix_dim() const { return matrix_dim_; }
    std::int32_t q_limit() const { return q_limit_; }
    unsigned bfp_tile_size() const { return bfp_tile_size_; }

    std::int16_t u_exponent_at(
        const NodeStorage& node,
        std::uint32_t row,
        std::uint32_t col) const {
        return exponent_at(
            node.fixed.u_tile_exponents,
            node.fixed.u_exponent,
            row,
            col,
            node.task.pivot_dim,
            node.task.total_dim);
    }

    std::int16_t update_exponent_at(
        const NodeStorage& node,
        std::uint32_t row,
        std::uint32_t col) const {
        const auto dim = node.task.total_dim - node.task.pivot_dim;
        return exponent_at(
            node.fixed.update_tile_exponents,
            node.fixed.update_exponent,
            row,
            col,
            dim,
            dim);
    }

    std::int16_t assembled_exponent_at(
        const NodeStorage& node,
        std::uint32_t row,
        std::uint32_t col) const {
        return exponent_at(
            node.assembled_tile_exponents,
            node.assembled_exp,
            row,
            col,
            node.task.total_dim,
            node.task.total_dim);
    }

    std::vector<std::uint32_t> permutation{};
    std::vector<double> rhs_fp64{};
    std::vector<double> original_matrix_fp64{};
    std::vector<double> original_rhs_fp64{};
    std::vector<double> original_solution_reference{};
    std::vector<double> solution_reference{};
    std::vector<std::int16_t> row_scale_exponents{};
    std::vector<std::int16_t> column_scale_exponents{};
    std::vector<std::int32_t> rhs_q{};
    std::int16_t rhs_exp{0};

private:
    std::uint32_t matrix_dim_;
    std::int32_t q_limit_;
    unsigned accumulator_bits_;
    unsigned bfp_tile_size_;
    std::map<std::uint16_t, NodeStorage> nodes_{};

    std::size_t tile_index(
        std::uint32_t row,
        std::uint32_t col,
        std::uint32_t rows,
        std::uint32_t cols) const {
        (void)rows;
        const auto tile_cols =
            (cols + bfp_tile_size_ - 1) / bfp_tile_size_;
        return static_cast<std::size_t>(row / bfp_tile_size_) * tile_cols +
               col / bfp_tile_size_;
    }

    std::int16_t exponent_at(
        const std::vector<std::int16_t>& tile_exponents,
        std::int16_t scalar_exponent,
        std::uint32_t row,
        std::uint32_t col,
        std::uint32_t rows,
        std::uint32_t cols) const {
        if (bfp_tile_size_ == 0 || tile_exponents.empty()) {
            return scalar_exponent;
        }
        return tile_exponents.at(tile_index(row, col, rows, cols));
    }

    void assemble_fixed_tile_bfp(std::uint16_t node_id) {
        auto& node = at(node_id);
        const auto dim = node.task.total_dim;
        const auto tile_rows = (dim + bfp_tile_size_ - 1) / bfp_tile_size_;
        const auto tile_cols = tile_rows;
        const auto tile_count =
            static_cast<std::size_t>(tile_rows) * tile_cols;
        if (node.local_tile_exponents.size() != tile_count) {
            throw std::runtime_error("local tile exponent count mismatch");
        }

        const auto unset = std::numeric_limits<std::int16_t>::min();
        std::vector<std::int16_t> assembly_exponents(tile_count, unset);
        for (std::uint32_t row = 0; row < dim; ++row) {
            for (std::uint32_t col = 0; col < dim; ++col) {
                const auto value = node.local_q[row * dim + col];
                if (value != 0) {
                    const auto tile = tile_index(row, col, dim, dim);
                    assembly_exponents[tile] = std::max(
                        assembly_exponents[tile],
                        node.local_tile_exponents[tile]);
                }
            }
        }
        for (const auto& mapping : node.child_maps) {
            const auto& child = at(mapping.child_id);
            if (!child.fixed.valid) {
                throw std::runtime_error(
                    "tile-BFP parent observed an uncommitted child update");
            }
            const auto child_dim =
                child.task.total_dim - child.task.pivot_dim;
            for (std::size_t row = 0; row < mapping.row_map.size(); ++row) {
                for (std::size_t col = 0; col < mapping.row_map.size(); ++col) {
                    const auto src_row = mapping.row_map[row];
                    const auto src_col = mapping.row_map[col];
                    const auto dst_row = mapping.col_map[row];
                    const auto dst_col = mapping.col_map[col];
                    if (src_row >= child_dim || src_col >= child_dim ||
                        dst_row >= dim || dst_col >= dim) {
                        throw std::runtime_error(
                            "tile-BFP map table index is out of bounds");
                    }
                    const auto source =
                        child.fixed.update[src_row * child_dim + src_col];
                    if (source != 0) {
                        const auto destination =
                            tile_index(dst_row, dst_col, dim, dim);
                        assembly_exponents[destination] = std::max(
                            assembly_exponents[destination],
                            exponent_at(
                                child.fixed.update_tile_exponents,
                                child.fixed.update_exponent,
                                src_row,
                                src_col,
                                child_dim,
                                child_dim));
                    }
                }
            }
        }
        for (auto& exponent : assembly_exponents) {
            if (exponent == unset) exponent = 0;
        }

        QuantStats stats{};
        stats.assembly_exp = *std::min_element(
            assembly_exponents.begin(), assembly_exponents.end());
        std::vector<std::int64_t> accumulator(
            static_cast<std::size_t>(dim) * dim, 0);
        const auto add_aligned =
            [&](std::uint32_t row,
                std::uint32_t col,
                std::int32_t value,
                std::int16_t source_exponent) {
                const auto tile = tile_index(row, col, dim, dim);
                const auto shift = static_cast<int>(assembly_exponents[tile]) -
                                   static_cast<int>(source_exponent);
                stats.align_shift_max = std::max(
                    stats.align_shift_max,
                    static_cast<unsigned>(std::max(shift, 0)));
                const auto aligned = round_shift_signed(value, shift);
                if (value != 0 && aligned == 0) ++stats.align_drop_count;
                const auto index = static_cast<std::size_t>(row) * dim + col;
                accumulator[index] =
                    saturating_add(accumulator[index], aligned, stats);
            };

        for (std::uint32_t row = 0; row < dim; ++row) {
            for (std::uint32_t col = 0; col < dim; ++col) {
                add_aligned(
                    row,
                    col,
                    node.local_q[row * dim + col],
                    exponent_at(
                        node.local_tile_exponents,
                        node.local_exp,
                        row,
                        col,
                        dim,
                        dim));
            }
        }
        for (const auto& mapping : node.child_maps) {
            const auto& child = at(mapping.child_id);
            const auto child_dim =
                child.task.total_dim - child.task.pivot_dim;
            for (std::size_t row = 0; row < mapping.row_map.size(); ++row) {
                for (std::size_t col = 0; col < mapping.row_map.size(); ++col) {
                    const auto src_row = mapping.row_map[row];
                    const auto src_col = mapping.row_map[col];
                    add_aligned(
                        mapping.col_map[row],
                        mapping.col_map[col],
                        child.fixed.update[src_row * child_dim + src_col],
                        exponent_at(
                            child.fixed.update_tile_exponents,
                            child.fixed.update_exponent,
                            src_row,
                            src_col,
                            child_dim,
                            child_dim));
                }
            }
        }

        std::vector<std::int32_t> quantized(accumulator.size(), 0);
        std::vector<std::int16_t> result_exponents(tile_count, 0);
        std::int64_t global_max_abs = 0;
        for (std::uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
            const auto row_begin = tile_row * bfp_tile_size_;
            const auto row_end =
                std::min(row_begin + bfp_tile_size_, dim);
            for (std::uint32_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
                const auto col_begin = tile_col * bfp_tile_size_;
                const auto col_end =
                    std::min(col_begin + bfp_tile_size_, dim);
                const auto tile =
                    static_cast<std::size_t>(tile_row) * tile_cols + tile_col;
                std::int64_t max_abs = 0;
                for (auto row = row_begin; row < row_end; ++row) {
                    for (auto col = col_begin; col < col_end; ++col) {
                        max_abs = std::max(
                            max_abs,
                            abs_i64(accumulator[row * dim + col]));
                    }
                }
                global_max_abs = std::max(global_max_abs, max_abs);
                const auto delta =
                    scale_exponent_delta(max_abs, q_limit_);
                const auto result_exp =
                    static_cast<int>(assembly_exponents[tile]) + delta;
                if (result_exp < std::numeric_limits<std::int16_t>::min() ||
                    result_exp > std::numeric_limits<std::int16_t>::max()) {
                    throw std::overflow_error(
                        "tile-BFP assembled exponent exceeds int16");
                }
                result_exponents[tile] =
                    static_cast<std::int16_t>(result_exp);
                for (auto row = row_begin; row < row_end; ++row) {
                    for (auto col = col_begin; col < col_end; ++col) {
                        auto value = round_shift_signed(
                            accumulator[row * dim + col], delta);
                        if (value > q_limit_) value = q_limit_;
                        if (value < -q_limit_) value = -q_limit_;
                        if (value == q_limit_ || value == -q_limit_) {
                            ++stats.saturation_count;
                        }
                        quantized[row * dim + col] =
                            static_cast<std::int32_t>(value);
                    }
                }
            }
        }
        stats.max_abs_acc = global_max_abs;
        stats.node_exp = *std::min_element(
            result_exponents.begin(), result_exponents.end());
        node.assembly_acc = std::move(accumulator);
        node.assembled_q = std::move(quantized);
        node.assembled_tile_exponents = std::move(result_exponents);
        node.quant_stats = stats;
        node.assembled_exp = stats.node_exp;
    }

    std::int64_t accumulator_min() const {
        if (accumulator_bits_ == 64) {
            return std::numeric_limits<std::int64_t>::min();
        }
        return -(std::int64_t{1} << (accumulator_bits_ - 1));
    }

    std::int64_t accumulator_max() const {
        if (accumulator_bits_ == 64) {
            return std::numeric_limits<std::int64_t>::max();
        }
        return (std::int64_t{1} << (accumulator_bits_ - 1)) - 1;
    }

    std::int64_t saturating_add(
        std::int64_t lhs,
        std::int64_t rhs,
        QuantStats& stats) const {
        const auto minimum = accumulator_min();
        const auto maximum = accumulator_max();
        if (rhs > 0 && lhs > maximum - rhs) {
            ++stats.assembly_overflow_count;
            return maximum;
        }
        if (rhs < 0 && lhs < minimum - rhs) {
            ++stats.assembly_overflow_count;
            return minimum;
        }
        return lhs + rhs;
    }
};

inline std::uint32_t read_u32_le(
    const std::vector<std::uint8_t>& data,
    std::size_t& cursor) {
    if (cursor + 4 > data.size()) {
        throw std::runtime_error("truncated map table");
    }
    std::uint32_t value = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(data[cursor + byte]) << (8 * byte);
    }
    cursor += 4;
    return value;
}

inline std::vector<ChildMap> decode_child_maps(
    const std::vector<std::uint8_t>& bytes) {
    std::size_t cursor = 0;
    const auto count = read_u32_le(bytes, cursor);
    std::vector<ChildMap> maps;
    maps.reserve(count);
    for (std::uint32_t entry = 0; entry < count; ++entry) {
        const auto child_id = read_u32_le(bytes, cursor);
        const auto row_count = read_u32_le(bytes, cursor);
        const auto col_count = read_u32_le(bytes, cursor);
        if (child_id > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("map table child id exceeds uint16");
        }
        ChildMap map{};
        map.child_id = static_cast<std::uint16_t>(child_id);
        map.row_map.reserve(row_count);
        map.col_map.reserve(col_count);
        for (std::uint32_t index = 0; index < row_count; ++index) {
            map.row_map.push_back(read_u32_le(bytes, cursor));
        }
        for (std::uint32_t index = 0; index < col_count; ++index) {
            map.col_map.push_back(read_u32_le(bytes, cursor));
        }
        maps.push_back(std::move(map));
    }
    if (cursor != bytes.size()) {
        throw std::runtime_error("map table contains trailing bytes");
    }
    return maps;
}

inline std::vector<double> read_f64_file(
    const std::filesystem::path& path) {
    const auto bytes = read_binary_file(path);
    if (bytes.size() % sizeof(double) != 0) {
        throw std::runtime_error("FP64 reference file size is not a multiple of 8");
    }
    std::vector<double> values(bytes.size() / sizeof(double));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint64_t raw = 0;
        for (unsigned byte = 0; byte < 8; ++byte) {
            raw |= static_cast<std::uint64_t>(bytes[index * 8 + byte])
                   << (8 * byte);
        }
        std::memcpy(&values[index], &raw, sizeof(double));
    }
    return values;
}

inline std::vector<std::int16_t> read_i16_file(
    const std::filesystem::path& path) {
    const auto bytes = read_binary_file(path);
    if (bytes.size() % sizeof(std::int16_t) != 0) {
        throw std::runtime_error("int16 file size is not a multiple of 2");
    }
    std::vector<std::int16_t> values(bytes.size() / sizeof(std::int16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto raw =
            static_cast<std::uint16_t>(bytes[index * 2]) |
            (static_cast<std::uint16_t>(bytes[index * 2 + 1]) << 8);
        values[index] = static_cast<std::int16_t>(raw);
    }
    return values;
}

inline SystemMemory load_system_memory(
    const ArtifactManifest& manifest,
    const std::vector<NodeTask>& tasks_in_file_order,
    DdrMemory& ddr,
    const ModelConfig& config) {
    SystemMemory memory(
        manifest.matrix_dim,
        config.q_limit(),
        config.accumulator_bits,
        config.bfp_tile_size);
    memory.permutation = manifest.permutation;
    memory.rhs_fp64 = read_f64_file(manifest.rhs_reference_path);
    memory.original_matrix_fp64 =
        read_f64_file(manifest.original_matrix_reference_path);
    memory.original_rhs_fp64 =
        read_f64_file(manifest.original_rhs_reference_path);
    memory.original_solution_reference =
        read_f64_file(manifest.original_solution_reference_path);
    memory.solution_reference =
        read_f64_file(manifest.solution_reference_path);
    memory.row_scale_exponents =
        read_i16_file(manifest.row_scale_exponents_path);
    memory.column_scale_exponents =
        read_i16_file(manifest.column_scale_exponents_path);
    if (memory.rhs_fp64.size() != manifest.matrix_dim ||
        memory.original_rhs_fp64.size() != manifest.matrix_dim ||
        memory.original_solution_reference.size() != manifest.matrix_dim ||
        memory.solution_reference.size() != manifest.matrix_dim ||
        memory.row_scale_exponents.size() != manifest.matrix_dim ||
        memory.column_scale_exponents.size() != manifest.matrix_dim ||
        memory.original_matrix_fp64.size() !=
            static_cast<std::size_t>(manifest.matrix_dim) *
                manifest.matrix_dim) {
        throw std::runtime_error("RHS/reference solution length mismatch");
    }

    const auto rhs_q_region = manifest.global_regions.at("rhs_q");
    const auto rhs_e_region = manifest.global_regions.at("rhs_e");
    memory.rhs_q =
        ddr.read_i32_vector(rhs_q_region.offset, manifest.matrix_dim, false);
    memory.rhs_exp = ddr.read_i16(rhs_e_region.offset, false);

    const auto reference_fronts =
        read_f64_file(manifest.reference_front_path);
    std::map<std::uint16_t, NodeTask> tasks;
    for (const auto& task : tasks_in_file_order) {
        tasks.emplace(task.node_id, task);
    }
    for (const auto& meta : manifest.nodes) {
        const auto task_it = tasks.find(meta.node_id);
        if (task_it == tasks.end()) {
            throw std::runtime_error("missing task for manifest node");
        }
        NodeStorage node{};
        node.task = task_it->second;
        node.front_indices = meta.front_indices;
        const auto elements =
            static_cast<std::size_t>(node.task.total_dim) * node.task.total_dim;
        node.local_q =
            ddr.read_i32_vector(meta.front_q.offset, elements, false);
        node.local_tile_exponents = ddr.read_i16_vector(
            meta.front_e.offset,
            meta.front_e.size / sizeof(std::int16_t),
            false);
        node.local_exp = node.local_tile_exponents.empty() ?
            0 : node.local_tile_exponents.front();
        const auto reference_index =
            meta.reference_front_file_offset / sizeof(double);
        if (meta.reference_front_file_offset % sizeof(double) != 0 ||
            reference_index + elements > reference_fronts.size()) {
            throw std::runtime_error("reference front range is invalid");
        }
        node.local_fp64.assign(
            reference_fronts.begin() + static_cast<std::ptrdiff_t>(reference_index),
            reference_fronts.begin() +
                static_cast<std::ptrdiff_t>(reference_index + elements));
        node.child_maps = decode_child_maps(
            ddr.read_bytes(meta.map_table.offset, meta.map_table.size, false));
        if (node.child_maps.size() != node.task.children_count) {
            throw std::runtime_error(
                "map table entry count does not match children_count");
        }
        std::set<std::uint16_t> mapped_children;
        for (const auto& mapping : node.child_maps) {
            if (!mapped_children.insert(mapping.child_id).second ||
                mapping.child_id >= manifest.node_count ||
                manifest.parent[mapping.child_id] != node.task.node_id) {
                throw std::runtime_error(
                    "map table contains an invalid or duplicate child");
            }
            const auto& child_meta = manifest.nodes[mapping.child_id];
            const auto child_update =
                child_meta.front_indices.size() -
                (child_meta.range_end - child_meta.range_start);
            if (mapping.row_map.size() != child_update ||
                mapping.col_map.size() != child_update) {
                throw std::runtime_error(
                    "map table does not cover the complete child update");
            }
            std::set<std::uint32_t> source_rows(
                mapping.row_map.begin(), mapping.row_map.end());
            std::set<std::uint32_t> destination_rows(
                mapping.col_map.begin(), mapping.col_map.end());
            if (source_rows.size() != child_update ||
                (!source_rows.empty() &&
                 (*source_rows.begin() != 0 ||
                  *source_rows.rbegin() != child_update - 1))) {
                throw std::runtime_error(
                    "map table child source indices are incomplete");
            }
            if (destination_rows.size() != child_update ||
                (!destination_rows.empty() &&
                 *destination_rows.rbegin() >= node.task.total_dim)) {
                throw std::runtime_error(
                    "map table parent destination indices are invalid");
            }
        }
        memory.add_node(std::move(node));
    }
    return memory;
}

}  // namespace hw
