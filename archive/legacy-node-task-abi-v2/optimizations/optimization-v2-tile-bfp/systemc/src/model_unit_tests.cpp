#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_types.hpp"
#include "node_task_codec.hpp"
#include "numeric_kernels.hpp"
#include "quantization.hpp"
#include "system_memory.hpp"

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_node_task_codec() {
    hw::NodeTask task{};
    task.node_id = 0x1234;
    task.flags = 0x0003;
    task.parent_id = hw::ROOT_PARENT_ID;
    task.children_count = 2;
    task.total_dim = 0x01020304u;
    task.pivot_dim = 0x20u;
    task.tile_count = 2;
    task.tail_dim = 16;
    task.map_table_bytes = 28;
    task.front_q_addr = 0x0102030405060708ull;
    task.front_e_addr = 0x1112131415161718ull;
    task.update_q_addr = 0x191A1B1C1D1E1F20ull;
    task.update_e_addr = 0x2021222324252627ull;
    task.map_table_addr = 0x2122232425262728ull;
    task.l_factor_addr = 0x3132333435363738ull;
    task.u_factor_addr = 0x4142434445464748ull;
    task.p_vector_addr = 0x5152535455565758ull;
    task.reserved = 0;

    const auto record = hw::NodeTaskCodec::encode(task);
    expect(record.size() == 128, "NodeTask record size must be 128 bytes");
    expect(record[0] == 0x34 && record[1] == 0x12, "node_id must be little-endian");
    expect(record[12] == 0x20 && record[13] == 0, "pivot_dim offset mismatch");
    expect(record[32] == 0x08 && record[39] == 0x01,
           "front_q_addr offset mismatch");

    const auto decoded = hw::NodeTaskCodec::decode(record);
    expect(decoded.node_id == task.node_id, "node_id round-trip mismatch");
    expect(decoded.total_dim == task.total_dim, "total_dim round-trip mismatch");
    expect(decoded.pivot_dim == task.pivot_dim, "pivot_dim round-trip mismatch");
    expect(decoded.front_q_addr == task.front_q_addr,
           "front_q_addr round-trip mismatch");
    expect(decoded.p_vector_addr == task.p_vector_addr, "p_vector_addr round-trip mismatch");
    expect(decoded.reserved == task.reserved, "reserved round-trip mismatch");
}

void test_fixed_numeric_helpers() {
    expect(hw::round_div_signed(5, 2) == 3, "positive division rounding mismatch");
    expect(hw::round_div_signed(-5, 2) == -3, "negative division rounding mismatch");
    expect(hw::round_div_signed(5, -2) == -3, "signed denominator mismatch");
    expect(
        hw::round_div_signed_wide(
            static_cast<__int128>(1) << 70,
            std::int64_t{1} << 10) ==
            (static_cast<__int128>(1) << 60),
        "wide signed divider must retain an over-64-bit numerator");

    hw::ModelConfig config{};
    config.frac_bits = 8;
    config.workspace_guard_bits = 0;
    config.adaptive_factor_scaling = false;
    config.fixed_pivot_rel_tol = 0.0;
    const std::vector<std::int32_t> front{
        4, 2, 1,
        8, 8, 2,
        2, 1, 6,
    };
    const auto result = hw::factor_fixed_front(front, 3, 2, 0, config);
    expect(result.factor.valid, "fixed factor should be valid");
    expect(result.selected_rows.front() == 1, "fixed pivot row mismatch");
    expect(result.factor.pvec == std::vector<std::uint16_t>({1, 0}),
           "fixed P-vector mismatch");
    expect(result.factor.l == std::vector<std::int32_t>({
               256, 0,
               128, 256,
               64, 128,
           }),
           "fixed L disagrees with Python bit-exact vector");
    expect(result.factor.u == std::vector<std::int32_t>({
               8, 8, 2,
               0, -2, 0,
           }),
           "fixed U disagrees with Python bit-exact vector");
    expect(result.factor.update == std::vector<std::int32_t>({5}),
           "fixed update disagrees with Python bit-exact vector");
}

void test_guard_bit_workspace_and_rescue_detection() {
    const std::vector<std::int32_t> cancellation_front{
        3, 1,
        1, 0,
    };
    hw::ModelConfig narrow{};
    narrow.frac_bits = 8;
    narrow.workspace_guard_bits = 0;
    narrow.adaptive_factor_scaling = false;
    narrow.fixed_pivot_rel_tol = 0.0;
    bool requested_rescue = false;
    try {
        (void)hw::factor_fixed_front(
            cancellation_front, 2, 2, 0, narrow);
    } catch (const hw::PrecisionRescueRequired&) {
        requested_rescue = true;
    }
    expect(
        requested_rescue,
        "narrow workspace must detect a cancellation-induced zero pivot");

    auto guarded = narrow;
    guarded.workspace_guard_bits = 8;
    const auto guarded_result = hw::factor_fixed_front(
        cancellation_front, 2, 2, 0, guarded);
    expect(
        guarded_result.factor.valid,
        "guard-bit workspace must preserve the sub-LSB Schur value");
    expect(
        guarded_result.workspace_renormalize_count > 0,
        "guard-bit pivot path must record active-column renormalization");

    const std::vector<std::int32_t> growth_front{
        1, 1,
        std::numeric_limits<std::int32_t>::max(), 1,
    };
    requested_rescue = false;
    try {
        (void)hw::factor_fixed_front(
            growth_front, 2, 1, 0, narrow);
    } catch (const hw::PrecisionRescueRequired&) {
        requested_rescue = true;
    }
    expect(
        requested_rescue,
        "out-of-range L multiplier must request precision rescue");
}

hw::NodeTask make_test_task(
    std::uint16_t node_id,
    std::uint16_t parent_id,
    std::uint16_t children) {
    hw::NodeTask task{};
    task.node_id = node_id;
    task.parent_id = parent_id;
    task.children_count = children;
    task.total_dim = 2;
    task.pivot_dim = 1;
    task.tile_count = 1;
    task.tail_dim = 1;
    return task;
}

void test_qau_multisource_assembly() {
    hw::SystemMemory memory(2, 127, 64);

    hw::NodeStorage child0{};
    child0.task = make_test_task(0, 2, 0);
    child0.front_indices = {0, 1};
    child0.local_q = {100, 20, 30, 32};
    child0.local_fp64 = {100, 20, 30, 32};
    child0.fixed.update = {32};
    child0.fixed.update_exponent = 0;
    child0.fixed.valid = true;
    memory.add_node(std::move(child0));

    hw::NodeStorage child1{};
    child1.task = make_test_task(1, 2, 0);
    child1.front_indices = {0, 1};
    child1.local_q = {40, 10, 90, 16};
    child1.local_fp64 = {40, 10, 90, 16};
    child1.fixed.update = {16};
    child1.fixed.update_exponent = 1;
    child1.fixed.valid = true;
    memory.add_node(std::move(child1));

    hw::NodeStorage root{};
    root.task = make_test_task(2, hw::ROOT_PARENT_ID, 2);
    root.front_indices = {0, 1};
    root.local_q = {64, 0, 0, 96};
    root.local_fp64 = {64, 0, 0, 96};
    root.child_maps = {
        hw::ChildMap{0, {0}, {0}},
        hw::ChildMap{1, {0}, {1}},
    };
    memory.add_node(std::move(root));

    memory.assemble_fixed(2);
    const auto& assembled = memory.at(2);
    expect(
        assembled.assembled_q ==
            std::vector<std::int32_t>({48, 0, 0, 64}),
        "QAU multisource mantissa disagrees with Python golden vector");
    expect(
        assembled.assembled_exp == 1,
        "QAU multisource exponent disagrees with Python golden vector");
    expect(
        assembled.quant_stats.align_shift_max == 1,
        "QAU exponent alignment statistic mismatch");
}

void test_tile_bfp_qau_and_factor() {
    constexpr std::uint32_t dim = 17;
    hw::SystemMemory memory(dim, 127, 64, 16);
    hw::NodeStorage node{};
    node.task.node_id = 0;
    node.task.parent_id = hw::ROOT_PARENT_ID;
    node.task.total_dim = dim;
    node.task.pivot_dim = dim;
    node.front_indices.resize(dim);
    node.local_q.assign(static_cast<std::size_t>(dim) * dim, 0);
    node.local_fp64.assign(node.local_q.size(), 0.0);
    node.local_tile_exponents = {10, 0, 0, -10};
    for (std::uint32_t index = 0; index < dim; ++index) {
        node.front_indices[index] = index;
        node.local_q[index * dim + index] = 1;
        node.local_fp64[index * dim + index] =
            std::ldexp(1.0, index < 16 ? 10 : -10);
    }
    memory.add_node(std::move(node));
    memory.assemble_fixed(0);
    const auto& assembled = memory.at(0);
    expect(
        assembled.assembled_tile_exponents.size() == 4,
        "tile-BFP QAU exponent grid size mismatch");
    expect(
        assembled.assembled_tile_exponents[0] >
            assembled.assembled_tile_exponents[3],
        "tile-BFP QAU collapsed independent tile scales");

    hw::ModelConfig config{};
    config.bfp_tile_size = 16;
    config.q_use_bits = 7;
    config.frac_bits = 6;
    config.workspace_guard_bits = 0;
    config.fixed_pivot_rel_tol = 0.0;
    const auto factor = hw::factor_fixed_front_tile_bfp(
        assembled.assembled_q,
        assembled.assembled_tile_exponents,
        dim,
        dim,
        config);
    expect(factor.factor.valid, "tile-BFP factor should be valid");
    expect(
        factor.factor.u_tile_exponents.size() == 4,
        "tile-BFP U exponent grid size mismatch");
    expect(
        factor.factor.u_tile_exponents[0] >
            factor.factor.u_tile_exponents[3],
        "tile-BFP U quantization collapsed tile dynamic ranges");
    expect(
        factor.factor.u[16 * dim + 16] != 0,
        "small-scale tile diagonal was dropped");
}

void test_quantization_helpers() {
    expect(hw::round_shift_signed(3, 1) == 2, "positive rounding mismatch");
    expect(hw::round_shift_signed(-3, 1) == -2, "negative rounding mismatch");
    expect(hw::round_shift_signed(2, 1) == 1, "exact positive shift mismatch");
    expect(hw::round_shift_signed(-2, 1) == -1, "exact negative shift mismatch");
    expect(hw::round_shift_signed(7, -2) == 28, "left shift mismatch");
    expect(hw::round_shift_signed(-7, -2) == -28, "negative left shift mismatch");
    expect(
        hw::round_shift_signed(std::numeric_limits<std::int64_t>::min(), 63) == -1,
        "minimum int64 right shift mismatch");
    expect(
        hw::round_shift_signed(std::numeric_limits<std::int64_t>::min(), 64) == -1,
        "minimum int64 tie rounding mismatch");
    expect(hw::scale_exponent_delta(64, 127) == 0, "scale delta near half range mismatch");
    expect(hw::scale_exponent_delta(63, 127) == -1, "negative scale delta mismatch");
    expect(hw::scale_exponent_delta(255, 127) == 2, "positive scale delta mismatch");

    hw::QuantStats stats{};
    const auto result = hw::requantize({48, 0, 0, 64}, 1, 127, stats);
    expect(result.first == std::vector<std::int32_t>({48, 0, 0, 64}),
           "requantized mantissa mismatch");
    expect(result.second.node_exp == 1, "requantized exponent mismatch");
    expect(result.second.saturation_count == 0, "unexpected saturation");

    hw::QuantStats boundary_stats{};
    const auto boundary = hw::requantize({127, -127}, 0, 127, boundary_stats);
    expect(boundary.first == std::vector<std::int32_t>({127, -127}),
           "boundary mantissa mismatch");
    expect(boundary.second.saturation_count == 2,
           "boundary saturation count must match Python semantics");
}

}  // namespace

int sc_main(int, char**) {
    try {
        test_node_task_codec();
        test_quantization_helpers();
        test_fixed_numeric_helpers();
        test_guard_bit_workspace_and_rescue_detection();
        test_qau_multisource_assembly();
        test_tile_bfp_qau_and_factor();
        std::cout << "[MODEL_UNIT_TESTS] ALL PASSED\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[MODEL_UNIT_TESTS] FAILED: " << exc.what() << '\n';
        return 1;
    }
}
