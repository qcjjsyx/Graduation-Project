// T12 SolveExecutionUnit tests.
//
// Coverage:
//   - exact hand-computable cases: 2x2 single node (forward z + backward x),
//     3x1 single node with an update segment (read-modify-write),
//   - a three-level chain whose L/U/P are produced by the independent FP64
//     reference (factor_front_tree) and whose solution is compared against
//     that reference,
//   - a non-identity P-vector (pivot-block row swap) case, and a 3-cycle
//     P-vector case (not an involution, so gather vs scatter is observable)
//     with a non-trivial L11, an update segment and a non-zero accumulator,
//   - negative cases: -1 sentinel P-vector, non-permutation, zero U diagonal,
//     NaN in L / RHS, missing payload, insufficient scratch,
//   - abort drain (exactly one terminal, in-flight requests drained, arbiter
//     released) and reset invalidation.
//
// Expectations are hand-computed or come from the independent FP64 reference /
// a dense solve; the device output is never its own expectation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "controlplane/solve_exec_unit.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/on_chip_arbiter.hpp"
#include "fp64_reference.hpp"

namespace {

namespace cmd = hw::command;
namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace ref = hw::reference_fp64;

int checks = 0;
int failures = 0;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct ImageBuilder {
    std::vector<std::uint8_t> image;
    std::vector<dp::RegionInfo> regions;
    std::uint64_t cursor{0};

    std::uint64_t add(const std::string& name, std::uint64_t bytes) {
        cursor = align_up(cursor, 64);
        regions.push_back(
            {name, dp::classify_region_role(name), cursor, bytes});
        const std::uint64_t offset = cursor;
        cursor += bytes;
        return offset;
    }
    void finish() { image.resize(static_cast<std::size_t>(cursor), 0); }
    void write_f32(std::uint64_t offset, const std::vector<float>& values) {
        std::memcpy(image.data() + offset, values.data(), values.size() * 4);
    }
    void write_i32(std::uint64_t offset, const std::vector<std::int32_t>& values) {
        std::memcpy(image.data() + offset, values.data(), values.size() * 4);
    }
    void write_bytes(std::uint64_t offset, const std::vector<std::uint8_t>& data) {
        std::memcpy(image.data() + offset, data.data(), data.size());
    }
    std::vector<float> read_f32(std::uint64_t offset, std::size_t count) const {
        std::vector<float> values(count, 0.0f);
        std::memcpy(values.data(), image.data() + offset, count * 4);
        return values;
    }
};

// One solve descriptor data set: a single node front (n x p), its P-vector,
// the global RHS and the initial solution accumulator.
struct SolveFixture {
    std::uint32_t total_dim{0};
    std::uint32_t pivot_dim{0};
    std::uint32_t dimension{0};
    std::vector<std::uint32_t> variables;
    std::vector<float> l;   // n x p row major
    std::vector<float> u;   // p x n row major
    std::vector<std::int32_t> pvector;
    std::vector<float> rhs;   // length dimension
    std::vector<float> w;     // initial accumulator, length dimension
};

struct SolveHarness {
    ImageBuilder builder;
    std::unique_ptr<dp::DeviceMemory> device_memory;
    std::unique_ptr<dp::MemoryService> memory;
    std::unique_ptr<cp::SolveExecutionUnit> unit;
    dp::OnChipArbiter arbiter;
    std::uint64_t rhs_offset{0};
    std::uint64_t solution_offset{0};
    std::uint64_t workspace_offset{0};
    std::vector<std::uint8_t> rhs_initial;

    explicit SolveHarness(const SolveFixture& fixture, cp::SolveConfig config = {}) {
        builder.add("rhs_data", static_cast<std::uint64_t>(fixture.dimension) * 4);
        builder.add("solution_data", static_cast<std::uint64_t>(fixture.dimension) * 4);
        builder.add(
            "node.0.solve_workspace",
            static_cast<std::uint64_t>(2) * fixture.pivot_dim * 4);
        builder.add(
            "node.0.p_vector", static_cast<std::uint64_t>(fixture.pivot_dim) * 4);
        builder.add(
            "node.0.factor_l",
            static_cast<std::uint64_t>(fixture.total_dim) * fixture.pivot_dim * 4);
        builder.add(
            "node.0.factor_u",
            static_cast<std::uint64_t>(fixture.pivot_dim) * fixture.total_dim * 4);
        builder.finish();
        rhs_offset = builder.regions[0].offset;
        solution_offset = builder.regions[1].offset;
        workspace_offset = builder.regions[2].offset;
        builder.write_f32(rhs_offset, fixture.rhs);
        builder.write_f32(solution_offset, fixture.w);
        builder.write_i32(builder.regions[3].offset, fixture.pvector);
        builder.write_f32(builder.regions[4].offset, fixture.l);
        builder.write_f32(builder.regions[5].offset, fixture.u);
        rhs_initial.assign(
            builder.image.begin() + rhs_offset,
            builder.image.begin() + rhs_offset + fixture.dimension * 4);

        dp::DeviceMemory device_memory_value(builder.image, builder.regions);
        device_memory =
            std::make_unique<dp::DeviceMemory>(std::move(device_memory_value));
        memory = std::make_unique<dp::MemoryService>(
            *device_memory, dp::MemoryServiceConfig{1, 16, 1, 8});
        unit = std::make_unique<cp::SolveExecutionUnit>(0, memory.get(), config);
        arbiter.configure(64, 64);
        unit->set_on_chip_arbiter(&arbiter);
    }

    dp::MemoryService& service() { return *memory; }

    cp::ExecRequest make_request(
        cmd::Opcode opcode,
        std::uint64_t txn_id,
        std::uint64_t required_scratch,
        cp::SolveConfig config = {}) {
        cp::ExecRequest request;
        request.txn_id = txn_id;
        request.command_id = 1;
        request.node_id = 0;
        request.opcode = opcode;
        auto& view = request.solve;
        view.factor.total_dim = fixture_total_;
        view.factor.pivot_dim = fixture_pivot_;
        view.factor.format = cmd::DataFormat::Fp32;
        view.factor.l = region_view(4);
        view.factor.u = region_view(5);
        view.rhs = region_view(0);
        view.solution = region_view(1);
        view.p = region_view(3);
        view.workspace = region_view(2);
        view.rhs_count = 1;
        view.direction = opcode == cmd::Opcode::SolveForward
                             ? cmd::SolveDirection::Forward
                             : cmd::SolveDirection::Backward;
        view.format = cmd::DataFormat::Fp32;
        view.variables = variables_;
        request.range_bytes = required_scratch;
        request.temp_lease.valid = true;
        request.temp_lease.write = true;
        request.temp_lease.length = required_scratch;
        request.has_temp_lease = true;
        (void)config;
        return request;
    }

    void set_fixture_dims(const SolveFixture& fixture) {
        fixture_total_ = fixture.total_dim;
        fixture_pivot_ = fixture.pivot_dim;
        variables_ = fixture.variables;
    }

    dp::RegionView region_view(std::size_t index) const {
        dp::RegionView view;
        view.descriptor_id = static_cast<std::uint32_t>(index + 1);
        view.base_addr = builder.regions[index].offset;
        view.byte_size = builder.regions[index].size;
        view.name = builder.regions[index].name;
        const bool int32 = builder.regions[index].role == dp::RegionRole::PVector;
        view.format = int32 ? cmd::DataFormat::Int32 : cmd::DataFormat::Fp32;
        view.element_bytes = 4;
        switch (index) {
        case 0:
        case 1:
            view.rows = 1;
            view.cols = static_cast<std::uint32_t>(builder.regions[index].size / 4);
            view.row_stride = static_cast<std::uint32_t>(builder.regions[index].size);
            break;
        case 2:
            view.rows = 2;
            view.cols = fixture_pivot_;
            view.row_stride = fixture_pivot_ * 4;
            break;
        case 3:
            view.rows = 1;
            view.cols = fixture_pivot_;
            view.row_stride = fixture_pivot_ * 4;
            break;
        case 4:
            view.rows = fixture_total_;
            view.cols = fixture_pivot_;
            view.row_stride = fixture_pivot_ * 4;
            break;
        default:
            view.rows = fixture_pivot_;
            view.cols = fixture_total_;
            view.row_stride = fixture_total_ * 4;
            break;
        }
        return view;
    }

    cp::ExecResponse run(const cp::ExecRequest& request, std::uint64_t limit = 100000) {
        if (!unit->submit(request, cycle_)) {
            throw std::runtime_error("solve unit refused the request");
        }
        cp::ExecResponse response;
        std::uint64_t cycle = cycle_ + 1;
        while (cycle < cycle_ + limit) {
            memory->tick(cycle);
            dp::MemoryResponse memory_response;
            while (memory->try_pop_response(memory_response)) {
                unit->deliver_memory_response(memory_response);
            }
            arbiter.begin_cycle(cycle);
            unit->tick(cycle);
            if (unit->poll_response(response)) {
                break;
            }
            ++cycle;
        }
        cycle_ = cycle;
        return response;
    }

    std::uint64_t cycle_{0};

    std::vector<float> solution() const {
        const auto& image = device_memory->image();
        const std::size_t count = builder.regions[1].size / 4;
        std::vector<float> values(count, 0.0f);
        std::memcpy(values.data(), image.data() + solution_offset, count * 4);
        return values;
    }

    std::vector<float> workspace() const {
        const auto& image = device_memory->image();
        const std::size_t count = builder.regions[2].size / 4;
        std::vector<float> values(count, 0.0f);
        std::memcpy(values.data(), image.data() + workspace_offset, count * 4);
        return values;
    }

    std::vector<std::uint8_t> rhs_bytes() const {
        const auto& image = device_memory->image();
        return {
            image.begin() + static_cast<std::ptrdiff_t>(rhs_offset),
            image.begin() + static_cast<std::ptrdiff_t>(
                                rhs_offset + rhs_initial.size())};
    }

    std::uint32_t fixture_total_{0};
    std::uint32_t fixture_pivot_{0};
    std::vector<std::uint32_t> variables_;
};

std::uint64_t one_node_scratch(const SolveFixture& fixture) {
    return cp::SolveExecutionUnit::scratch_bytes(
        fixture.total_dim, fixture.pivot_dim, cp::SolveConfig{}.row_buffer_depth);
}

void run_forward_backward(SolveHarness& harness, const SolveFixture& fixture) {
    const std::uint64_t scratch = one_node_scratch(fixture);
    harness.set_fixture_dims(fixture);
    const auto forward = harness.run(
        harness.make_request(cmd::Opcode::SolveForward, 1, scratch));
    expect(
        forward.status == cmd::StatusCode::Ok,
        "forward response is OK: " + forward.reason);
    const auto backward = harness.run(
        harness.make_request(cmd::Opcode::SolveBackward, 2, scratch));
    expect(
        backward.status == cmd::StatusCode::Ok,
        "backward response is OK: " + backward.reason);
}

// ---------------------------------------------------------------------------
// Hand-computable cases
// ---------------------------------------------------------------------------

void test_two_by_two_manual() {
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 2.0f, 1.0f};   // unit lower [[1,0],[2,1]]
    fixture.u = {4.0f, 1.0f, 0.0f, 2.0f};   // upper    [[4,1],[0,2]]
    fixture.pvector = {0, 1};
    fixture.rhs = {1.0f, 2.0f};
    fixture.w = {0.0f, 0.0f};
    SolveHarness harness(fixture);
    run_forward_backward(harness, fixture);
    // L z = b: z = [1, 0]; U x = z: x = [0.25, 0].
    expect(
        harness.workspace()[0] == 1.0f && harness.workspace()[1] == 0.0f,
        "2x2 manual: workspace row 0 holds z = [1, 0]");
    const auto x = harness.solution();
    expect(
        x[0] == 0.25f && x[1] == 0.0f, "2x2 manual: x = [0.25, 0]");
    // The RHS input must be byte-identical after both commands.
    expect(harness.rhs_bytes() == harness.rhs_initial, "2x2 manual: rhs_data is read-only");
    expect(
        harness.arbiter.stats(dp::OnChipDomain::Read, dp::OnChipClientKind::Solve, 0)
                .bytes > 0 &&
            harness.arbiter.stats(
                dp::OnChipDomain::Write, dp::OnChipClientKind::Solve, 0)
                    .bytes > 0,
        "2x2 manual: on-chip staging/consume charged to the unified arbiter");
}

void test_single_node_with_update_manual() {
    SolveFixture fixture;
    fixture.total_dim = 3;
    fixture.pivot_dim = 1;
    fixture.dimension = 3;
    fixture.variables = {0, 1, 2};
    fixture.l = {1.0f, 2.0f, 3.0f};       // L11 = [1], L21 = [2, 3]
    fixture.u = {5.0f, 6.0f, 7.0f};       // U11 = [5], U12 = [6, 7]
    fixture.pvector = {0};
    fixture.rhs = {10.0f, 0.0f, 0.0f};
    fixture.w = {0.0f, 0.0f, 0.0f};
    SolveHarness harness(fixture);
    run_forward_backward(harness, fixture);
    // z0 = 10; w1 = 0 - 2*10 = -20; w2 = 0 - 3*10 = -30;
    // x0 = (10 - 6*(-20) - 7*(-30)) / 5 = 340/5 = 68.
    const auto x = harness.solution();
    expect(x[0] == 68.0f, "1x3 manual: x0 = 68");
    expect(
        x[1] == -20.0f && x[2] == -30.0f,
        "1x3 manual: update segment keeps the accumulated RHS values");
    expect(harness.workspace()[0] == 10.0f, "1x3 manual: workspace z0 = 10");
}

// ---------------------------------------------------------------------------
// Chain tree vs the independent FP64 reference
// ---------------------------------------------------------------------------

// A multi-node chain needs the cross-command accumulator; drive it in one
// harness with three nodes side by side.
void test_chain_tree_multi_node_solution() {
    std::vector<ref::FrontInput> fronts(3);
    fronts[0].variables = {0, 1};
    fronts[0].pivot_columns = 1;
    fronts[0].local_front = ref::DenseMatrix(2, 2, {4.0, 1.0, 0.0, 0.0});
    fronts[1].variables = {1, 2};
    fronts[1].pivot_columns = 1;
    fronts[1].local_front = ref::DenseMatrix(2, 2, {5.0, 1.0, 1.0, 0.0});
    fronts[1].contributions.push_back(ref::ContributionMap{0, {0}, {0}});
    fronts[2].variables = {2};
    fronts[2].pivot_columns = 1;
    fronts[2].local_front = ref::DenseMatrix(1, 1, {3.0});
    fronts[2].contributions.push_back(ref::ContributionMap{1, {0}, {0}});
    const std::vector<int> parent = {1, 2, -1};
    const auto factors = ref::factor_front_tree(
        fronts, parent, ref::PivotPolicy::Static, 0.0);
    const std::vector<double> rhs = {1.0, 2.0, 3.0};
    const auto reference = ref::solve_front_tree(factors, rhs, 0.0);

    // Build a three-node harness: one shared solution/rhs region and one
    // factor/workspace/P set per node, then run the four commands in the
    // exact chain order (forward 0,1,2 then backward 2,1,0).
    ImageBuilder builder;
    const std::uint64_t rhs_offset = builder.add("rhs_data", 3 * 4);
    const std::uint64_t solution_offset = builder.add("solution_data", 3 * 4);
    std::vector<std::uint64_t> workspace_offsets;
    std::vector<std::uint64_t> p_offsets;
    std::vector<std::uint64_t> l_offsets;
    std::vector<std::uint64_t> u_offsets;
    for (std::uint32_t node = 0; node < 3; ++node) {
        const std::uint32_t total = static_cast<std::uint32_t>(fronts[node].variables.size());
        workspace_offsets.push_back(
            builder.add("node." + std::to_string(node) + ".solve_workspace", 2 * 4));
        p_offsets.push_back(
            builder.add("node." + std::to_string(node) + ".p_vector", 1 * 4));
        l_offsets.push_back(
            builder.add("node." + std::to_string(node) + ".factor_l", total * 1 * 4));
        u_offsets.push_back(
            builder.add("node." + std::to_string(node) + ".factor_u", 1 * total * 4));
    }
    builder.finish();
    builder.write_f32(rhs_offset, {1.0f, 2.0f, 3.0f});
    builder.write_f32(solution_offset, {0.0f, 0.0f, 0.0f});
    for (std::uint32_t node = 0; node < 3; ++node) {
        const auto& factorization = factors[node].factorization;
        std::vector<float> l(
            factorization.l.values.begin(), factorization.l.values.end());
        std::vector<float> u(
            factorization.u.values.begin(), factorization.u.values.end());
        builder.write_f32(l_offsets[node], l);
        builder.write_f32(u_offsets[node], u);
        builder.write_i32(
            p_offsets[node], {static_cast<std::int32_t>(factorization.permutation[0])});
    }
    std::vector<std::uint8_t> rhs_before(
        builder.image.begin() + rhs_offset, builder.image.begin() + rhs_offset + 12);

    dp::DeviceMemory device_memory(builder.image, builder.regions);
    dp::MemoryService memory(device_memory, dp::MemoryServiceConfig{1, 16, 1, 8});
    cp::SolveConfig config;
    cp::SolveExecutionUnit unit(0, &memory, config);
    dp::OnChipArbiter arbiter;
    arbiter.configure(64, 64);
    unit.set_on_chip_arbiter(&arbiter);

    const auto region = [&](std::size_t index) {
        dp::RegionView view;
        view.descriptor_id = static_cast<std::uint32_t>(index + 1);
        view.base_addr = builder.regions[index].offset;
        view.byte_size = builder.regions[index].size;
        view.name = builder.regions[index].name;
        view.format = builder.regions[index].role == dp::RegionRole::PVector
                          ? cmd::DataFormat::Int32
                          : cmd::DataFormat::Fp32;
        view.element_bytes = 4;
        return view;
    };
    const auto make_request = [&](cmd::Opcode opcode, std::uint32_t node,
                                  std::uint64_t txn) {
        const std::uint32_t total =
            static_cast<std::uint32_t>(fronts[node].variables.size());
        const std::uint64_t scratch =
            cp::SolveExecutionUnit::scratch_bytes(total, 1, config.row_buffer_depth);
        cp::ExecRequest request;
        request.txn_id = txn;
        request.command_id = txn;
        request.node_id = node;
        request.opcode = opcode;
        auto& view = request.solve;
        view.factor.total_dim = total;
        view.factor.pivot_dim = 1;
        view.factor.format = cmd::DataFormat::Fp32;
        // Region indices: 0 rhs, 1 solution, then per node workspace/p/L/U.
        view.rhs = region(0);
        view.solution = region(1);
        view.factor.l = region(4 + node * 4);
        view.factor.u = region(4 + node * 4 + 1);
        view.factor.l.rows = total;
        view.factor.l.cols = 1;
        view.factor.l.row_stride = 4;
        view.factor.u.rows = 1;
        view.factor.u.cols = total;
        view.factor.u.row_stride = total * 4;
        view.p = region(2 + node * 4 + 1);
        view.workspace = region(2 + node * 4);
        view.rhs_count = 1;
        view.direction = opcode == cmd::Opcode::SolveForward
                             ? cmd::SolveDirection::Forward
                             : cmd::SolveDirection::Backward;
        view.format = cmd::DataFormat::Fp32;
        view.variables.clear();
        for (const auto variable : fronts[node].variables) {
            view.variables.push_back(static_cast<std::uint32_t>(variable));
        }
        request.range_bytes = scratch;
        request.temp_lease.valid = true;
        request.temp_lease.write = true;
        request.temp_lease.length = scratch;
        request.has_temp_lease = true;
        return request;
    };
    const auto run = [&](const cp::ExecRequest& request, std::uint64_t* cycle_counter) {
        expect(unit.submit(request, *cycle_counter), "chain multi: request accepted");
        cp::ExecResponse response;
        std::uint64_t cycle = *cycle_counter + 1;
        while (cycle < 200000) {
            memory.tick(cycle);
            dp::MemoryResponse memory_response;
            while (memory.try_pop_response(memory_response)) {
                unit.deliver_memory_response(memory_response);
            }
            arbiter.begin_cycle(cycle);
            unit.tick(cycle);
            if (unit.poll_response(response)) {
                break;
            }
            ++cycle;
        }
        *cycle_counter = cycle;
        expect(response.status == cmd::StatusCode::Ok, "chain multi: command OK: " + response.reason);
        return response;
    };
    std::uint64_t txn = 1;
    std::uint64_t cycle_counter = 0;
    for (std::uint32_t node = 0; node < 3; ++node) {
        run(make_request(cmd::Opcode::SolveForward, node, txn++), &cycle_counter);
    }
    for (std::uint32_t node = 3; node > 0; --node) {
        run(make_request(cmd::Opcode::SolveBackward, node - 1, txn++), &cycle_counter);
    }
    std::vector<float> device_solution(3, 0.0f);
    std::memcpy(
        device_solution.data(), device_memory.image().data() + solution_offset, 12);
    for (std::uint32_t index = 0; index < 3; ++index) {
        const double expected = reference[index];
        const double actual = device_solution[index];
        expect(
            std::fabs(actual - expected) <= 1e-5 * (1.0 + std::fabs(expected)),
            "chain multi: x[" + std::to_string(index) + "] matches the FP64 reference (" +
                std::to_string(actual) + " vs " + std::to_string(expected) + ")");
    }
    const std::vector<std::uint8_t> rhs_after(
        device_memory.image().begin() + static_cast<std::ptrdiff_t>(rhs_offset),
        device_memory.image().begin() + static_cast<std::ptrdiff_t>(rhs_offset) + 12);
    expect(rhs_after == rhs_before, "chain multi: rhs_data unchanged byte for byte");
}

// ---------------------------------------------------------------------------
// P-vector non-identity (pivot-block row swap)
// ---------------------------------------------------------------------------

void test_pvector_non_identity() {
    // F = [[0,1],[2,3]] with a dynamic pivot swap: P F = [[2,3],[0,1]],
    // L = I, U = [[2,3],[0,1]], P-vector = [1,0] (gather direction).
    const ref::DenseMatrix matrix(2, 2, {0.0, 1.0, 2.0, 3.0});
    const auto factorization =
        ref::panel_lu_policy(matrix, 2, ref::PivotPolicy::DynamicPanel, 0.0);
    expect(
        factorization.permutation == std::vector<std::size_t>{1, 0},
        "swap: reference produces the non-identity permutation");
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 0.0f, 1.0f};
    fixture.u = {2.0f, 3.0f, 0.0f, 1.0f};
    fixture.pvector = {1, 0};
    fixture.rhs = {1.0f, 3.0f};
    fixture.w = {0.0f, 0.0f};
    SolveHarness harness(fixture);
    run_forward_backward(harness, fixture);
    // Gather: z = [b1, b0] = [3, 1]; x = [0, 1] (F x = b).
    expect(
        harness.workspace()[0] == 3.0f && harness.workspace()[1] == 1.0f,
        "swap: workspace row 0 holds the gathered z = [3, 1]");
    const auto x = harness.solution();
    expect(x[0] == 0.0f && x[1] == 1.0f, "swap: x = [0, 1]");
}

// ---------------------------------------------------------------------------
// P-vector 3-cycle with update segment and non-zero accumulator (T12 revision)
// ---------------------------------------------------------------------------
//
// The 2x2 swap above cannot tell gather from scatter (a transposition is its
// own inverse), uses L = I and starts from a zero accumulator.  This case
// fixes all three blind spots: a 3-cycle P (not an involution), a non-trivial
// unit lower L11, non-zero L21/U12, and a non-zero initial solution_data on
// both pivot and update variables (as if children had already accumulated).

void test_pvector_three_cycle_manual() {
    SolveFixture fixture;
    fixture.total_dim = 5;
    fixture.pivot_dim = 3;
    fixture.dimension = 8;
    // Pivot block {1,2,3} is contiguous (descriptor rule); the update
    // variables {5,7} are separated from it and from each other.
    fixture.variables = {1, 2, 3, 5, 7};
    // Gather semantics: pivot row i comes from pivot-block row p[i].
    fixture.pvector = {1, 2, 0};
    // Inverse q (q[p[i]] = i) is {2, 0, 1} != p, so p is not an involution.
    bool involution = true;
    for (std::size_t i = 0; i < fixture.pvector.size(); ++i) {
        const auto pi = static_cast<std::size_t>(fixture.pvector[i]);
        involution = involution &&
                     static_cast<std::size_t>(fixture.pvector[pi]) == i;
    }
    expect(!involution, "3-cycle: p is not an involution (p[p[i]] != i for some i)");
    // L (5x3): L11 = [[1,0,0],[2,1,0],[-1,0.5,1]], L21 = [[1,0,2],[0.5,-1,0]].
    fixture.l = {
        1.0f,  0.0f, 0.0f,
        2.0f,  1.0f, 0.0f,
        -1.0f, 0.5f, 1.0f,
        1.0f,  0.0f, 2.0f,
        0.5f, -1.0f, 0.0f,
    };
    // U (3x5): U11 = [[2,1,-1],[0,4,2],[0,0,-2]], U12 = [[1,0],[0,-1],[0.5,0]].
    fixture.u = {
        2.0f, 1.0f, -1.0f, 1.0f,  0.0f,
        0.0f, 4.0f,  2.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -2.0f, 0.5f,  0.0f,
    };
    // Global vectors indexed by variable 0..7; 0, 4, 6 are not touched.
    fixture.rhs = {100.0f, 3.0f, 5.0f, -2.0f, 200.0f, 4.0f, 300.0f, 6.0f};
    fixture.w = {7.0f, 1.0f, -2.0f, 3.0f, 7.0f, 0.5f, 7.0f, -1.5f};
    SolveHarness harness(fixture);
    run_forward_backward(harness, fixture);

    // Hand computation.
    // Local pivot rows k=0,1,2 are variables 1,2,3:
    //   c_k = b[g[k]] + w[g[k]] = [3+1, 5-2, -2+3] = [4, 3, 1].
    // Gather: r_i = c[p[i]] = [c1, c2, c0] = [3, 1, 4].
    //   (Scatter would give r = c[q[i]] = [1, 4, 3].)
    // Forward, L11 z = r:
    //   z0 = 3
    //   z1 = 1 - 2*3 = -5
    //   z2 = 4 - (-1)*3 - 0.5*(-5) = 4 + 3 + 2.5 = 9.5
    // Update segment RMW, w[g[3+t]] -= L21[t] . z:
    //   L21[0] . z = 1*3 + 0*(-5) + 2*9.5 = 22    -> w[5] = 0.5 - 22  = -21.5
    //   L21[1] . z = 0.5*3 + (-1)*(-5) + 0 = 6.5 -> w[7] = -1.5 - 6.5 = -8
    // Pivot write-back w[1..3] = z = [3, -5, 9.5].
    const auto z = harness.workspace();
    expect(
        z[0] == 3.0f && z[1] == -5.0f && z[2] == 9.5f,
        "3-cycle: workspace row 0 holds the hand-computed z = [3, -5, 9.5]");
    // Backward, c_i = z_i - U12[i] . [x5, x7] with x5 = -21.5, x7 = -8:
    //   c0 = 3    - (1*(-21.5) + 0*(-8))    = 24.5
    //   c1 = -5   - (0*(-21.5) + (-1)*(-8)) = -13
    //   c2 = 9.5  - (0.5*(-21.5) + 0*(-8))  = 20.25
    // U11 x = c:
    //   x2 = 20.25 / -2                          = -10.125
    //   x1 = (-13 - 2*(-10.125)) / 4 = 7.25 / 4  = 1.8125
    //   x0 = (24.5 - 1*1.8125 - (-1)*(-10.125)) / 2 = 12.5625 / 2 = 6.28125
    // All intermediates are short dyadic fractions, exact in FP32.
    const auto x = harness.solution();
    expect(
        x[5] == -21.5f && x[7] == -8.0f,
        "3-cycle: update segment holds the hand-computed RMW values [-21.5, -8]");
    expect(
        x[1] == 6.28125f && x[2] == 1.8125f && x[3] == -10.125f,
        "3-cycle: x = [6.28125, 1.8125, -10.125]");
    expect(
        x[0] == 7.0f && x[4] == 7.0f && x[6] == 7.0f,
        "3-cycle: variables outside the front are untouched");
    expect(harness.rhs_bytes() == harness.rhs_initial, "3-cycle: rhs_data is byte-identical");
}

void test_pvector_three_cycle_reference() {
    // Cross-check of the P-vector convention against the independent FP64
    // reference: DYNAMIC_PANEL on F picks the column-0 pivot in row 1 (|4|),
    // then the column-1 pivot in original row 2 (|7.5| vs |1.75|), giving the
    // 3-cycle permutation [1, 2, 0] (P*F row i = F row permutation[i]).
    const ref::DenseMatrix matrix(
        3, 3, {1.0, 2.0, 1.0, 4.0, 1.0, 0.0, 2.0, 8.0, 1.0});
    const auto factorization =
        ref::panel_lu_policy(matrix, 3, ref::PivotPolicy::DynamicPanel, 0.0);
    expect(
        factorization.permutation == std::vector<std::size_t>{1, 2, 0},
        "3-cycle ref: DYNAMIC_PANEL produces the 3-cycle permutation [1, 2, 0]");
    SolveFixture fixture;
    fixture.total_dim = 3;
    fixture.pivot_dim = 3;
    fixture.dimension = 3;
    fixture.variables = {0, 1, 2};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            fixture.l.push_back(static_cast<float>(factorization.l.at(row, col)));
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            fixture.u.push_back(static_cast<float>(factorization.u.at(row, col)));
        }
    }
    for (const auto index : factorization.permutation) {
        fixture.pvector.push_back(static_cast<std::int32_t>(index));
    }
    // F x = b with x = [1, -1, 2]: b = [1-2+2, 4-1+0, 2-8+2] = [1, 3, -4].
    fixture.rhs = {1.0f, 3.0f, -4.0f};
    fixture.w = {0.0f, 0.0f, 0.0f};
    SolveHarness harness(fixture);
    run_forward_backward(harness, fixture);
    const auto x = harness.solution();
    const double error = std::max(
        {std::fabs(x[0] - 1.0), std::fabs(x[1] + 1.0), std::fabs(x[2] - 2.0)});
    expect(error <= 1e-6, "3-cycle ref: device solves F x = b with the reference L/U/P");
}

// ---------------------------------------------------------------------------
// Negative cases
// ---------------------------------------------------------------------------

cp::ExecResponse run_single(SolveFixture fixture, cmd::Opcode opcode) {
    SolveHarness harness(fixture);
    harness.set_fixture_dims(fixture);
    const std::uint64_t scratch = one_node_scratch(fixture);
    return harness.run(harness.make_request(opcode, 1, scratch));
}

void test_negative_pvector() {
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 2.0f, 1.0f};
    fixture.u = {4.0f, 1.0f, 0.0f, 2.0f};
    fixture.rhs = {1.0f, 2.0f};
    fixture.w = {0.0f, 0.0f};

    auto sentinel = fixture;
    sentinel.pvector = {-1, 1};
    auto response = run_single(sentinel, cmd::Opcode::SolveForward);
    expect(
        response.status == cmd::StatusCode::BadDescriptor &&
            response.reason.find("sentinel") != std::string::npos,
        "negative: -1 P-vector -> BAD_DESCRIPTOR with the not-written reason");

    auto duplicate = fixture;
    duplicate.pvector = {0, 0};
    response = run_single(duplicate, cmd::Opcode::SolveForward);
    expect(
        response.status == cmd::StatusCode::BadDescriptor &&
            response.reason.find("permutation") != std::string::npos,
        "negative: repeated P-vector entry -> BAD_DESCRIPTOR permutation");

    auto out_of_range = fixture;
    out_of_range.pvector = {0, 2};
    response = run_single(out_of_range, cmd::Opcode::SolveForward);
    expect(
        response.status == cmd::StatusCode::BadDescriptor,
        "negative: out-of-range P-vector entry -> BAD_DESCRIPTOR");
}

void test_negative_zero_diagonal_and_nan() {
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 2.0f, 1.0f};
    fixture.u = {4.0f, 1.0f, 0.0f, 0.0f};  // zero diagonal in row 1
    fixture.pvector = {0, 1};
    fixture.rhs = {1.0f, 2.0f};
    fixture.w = {0.0f, 0.0f};
    // Run forward first so the accumulator is well defined, then backward.
    SolveHarness harness(fixture);
    harness.set_fixture_dims(fixture);
    const std::uint64_t scratch = one_node_scratch(fixture);
    auto response = harness.run(harness.make_request(cmd::Opcode::SolveForward, 1, scratch));
    expect(response.status == cmd::StatusCode::Ok, "zero diag: forward still OK");
    response = harness.run(harness.make_request(cmd::Opcode::SolveBackward, 2, scratch));
    expect(
        response.status == cmd::StatusCode::StaticPivotFailure,
        "negative: zero U diagonal -> STATIC_PIVOT_FAILURE without substitution");

    auto nan_l = fixture;
    nan_l.u = {4.0f, 1.0f, 0.0f, 2.0f};
    nan_l.l = {1.0f, 0.0f, std::nanf(""), 1.0f};
    response = run_single(nan_l, cmd::Opcode::SolveForward);
    expect(
        response.status == cmd::StatusCode::NonFiniteValue,
        "negative: NaN in L -> NON_FINITE_VALUE");

    auto nan_rhs = fixture;
    nan_rhs.u = {4.0f, 1.0f, 0.0f, 2.0f};
    nan_rhs.rhs = {1.0f, std::nanf("")};
    response = run_single(nan_rhs, cmd::Opcode::SolveForward);
    expect(
        response.status == cmd::StatusCode::NonFiniteValue,
        "negative: NaN in the RHS -> NON_FINITE_VALUE");
}

void test_negative_payload_and_scratch() {
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 2.0f, 1.0f};
    fixture.u = {4.0f, 1.0f, 0.0f, 2.0f};
    fixture.pvector = {0, 1};
    fixture.rhs = {1.0f, 2.0f};
    fixture.w = {0.0f, 0.0f};
    SolveHarness harness(fixture);
    harness.set_fixture_dims(fixture);
    const std::uint64_t scratch = one_node_scratch(fixture);

    auto request = harness.make_request(cmd::Opcode::SolveForward, 1, scratch);
    request.solve.variables.clear();
    auto response = harness.run(request);
    expect(
        response.status == cmd::StatusCode::BadDescriptor &&
            response.reason.find("payload") != std::string::npos,
        "negative: missing variable payload / wrong length -> BAD_DESCRIPTOR");

    request = harness.make_request(cmd::Opcode::SolveForward, 2, scratch);
    request.temp_lease.length = scratch - 4;
    response = harness.run(request);
    expect(
        response.status == cmd::StatusCode::BufferFull,
        "negative: insufficient scratch lease -> BUFFER_FULL");

    request = harness.make_request(cmd::Opcode::SolveForward, 3, scratch);
    request.opcode = cmd::Opcode::SolveBackward;
    request.solve.direction = cmd::SolveDirection::Forward;
    response = harness.run(request);
    expect(
        response.status == cmd::StatusCode::BadDescriptor,
        "negative: direction/opcode mismatch -> BAD_DESCRIPTOR");
}

// ---------------------------------------------------------------------------
// Abort drain and reset
// ---------------------------------------------------------------------------

void test_abort_drains_and_releases() {
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 2.0f, 1.0f};
    fixture.u = {4.0f, 1.0f, 0.0f, 2.0f};
    fixture.pvector = {0, 1};
    fixture.rhs = {1.0f, 2.0f};
    fixture.w = {0.0f, 0.0f};
    SolveHarness harness(fixture);
    harness.set_fixture_dims(fixture);
    const std::uint64_t scratch = one_node_scratch(fixture);
    auto request = harness.make_request(cmd::Opcode::SolveForward, 7, scratch);
    expect(harness.unit->submit(request, 0), "abort: request accepted");
    // Let the P-vector read land in flight, then abort.
    bool aborted = false;
    bool saw_pending = false;
    for (std::uint64_t cycle = 1; cycle < 1000; ++cycle) {
        harness.service().tick(cycle);
        dp::MemoryResponse memory_response;
        while (harness.service().try_pop_response(memory_response)) {
            harness.unit->deliver_memory_response(memory_response);
        }
        if (!aborted && harness.unit->pending_request_count() > 0) {
            saw_pending = true;
            harness.unit->abort_txn(7, cycle);
            aborted = true;
        }
        harness.arbiter.begin_cycle(cycle);
        harness.unit->tick(cycle);
        cp::ExecResponse response;
        if (harness.unit->poll_response(response)) {
            expect(
                response.txn_id == 7 && response.status == cmd::StatusCode::Aborted,
                "abort: exactly one terminal ABORTED response");
            expect(
                !harness.unit->busy(),
                "abort: the transaction is fully drained");
            expect(
                harness.unit->pending_request_count() == 0,
                "abort: no in-flight request survives the terminal");
            bool late = false;
            for (std::uint64_t id = 1; id < 1000; ++id) {
                late = late || harness.unit->handles_request(id);
            }
            expect(!late, "abort: no request id still routes to the unit");
            cp::ExecResponse second;
            expect(
                !harness.unit->poll_response(second),
                "abort: no second terminal response");
            break;
        }
    }
    expect(saw_pending && aborted, "abort: abort issued while a read was in flight");
}

void test_reset_invalidates_old_transactions() {
    SolveFixture fixture;
    fixture.total_dim = 2;
    fixture.pivot_dim = 2;
    fixture.dimension = 2;
    fixture.variables = {0, 1};
    fixture.l = {1.0f, 0.0f, 2.0f, 1.0f};
    fixture.u = {4.0f, 1.0f, 0.0f, 2.0f};
    fixture.pvector = {0, 1};
    fixture.rhs = {1.0f, 2.0f};
    fixture.w = {0.0f, 0.0f};
    SolveHarness harness(fixture);
    harness.set_fixture_dims(fixture);
    const std::uint64_t scratch = one_node_scratch(fixture);
    expect(
        harness.unit->submit(harness.make_request(cmd::Opcode::SolveForward, 5, scratch), 0),
        "reset: request accepted");
    harness.service().tick(1);
    harness.arbiter.begin_cycle(1);
    harness.unit->tick(1);
    harness.unit->reset();
    expect(!harness.unit->busy(), "reset: unit is idle");
    expect(harness.unit->pending_request_count() == 0, "reset: no pending requests");
    cp::ExecResponse response;
    expect(!harness.unit->poll_response(response), "reset: no terminal is delivered");
    // A late response of the old transaction must not resurrect it.
    dp::MemoryResponse late;
    late.request_id = 1;
    late.status = cmd::StatusCode::Ok;
    harness.unit->deliver_memory_response(late);
    harness.unit->tick(2);
    expect(!harness.unit->poll_response(response), "reset: late responses vanish");
}

}  // namespace

int main() {
    test_two_by_two_manual();
    test_single_node_with_update_manual();
    test_chain_tree_multi_node_solution();
    test_pvector_non_identity();
    test_pvector_three_cycle_manual();
    test_pvector_three_cycle_reference();
    test_negative_pvector();
    test_negative_zero_diagonal_and_nan();
    test_negative_payload_and_scratch();
    test_abort_drains_and_releases();
    test_reset_invalidates_old_transactions();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
