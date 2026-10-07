// T07 Panel-LU real FP32 engine integration tests.
//
// These tests drive the real backend (shared PanelEngine on BufferManager
// leases + FactorStoreUnit for STORE_FACTOR) through the T05 CommandDevice
// with schema-v2 artifacts built in memory. Expected values come from the
// independent FP64 reference (reference_fp64::factor_front, STATIC policy) or
// from independently written hand calculations, never from the device kernel.
//
// Numerical acceptance here is the panel subsystem (A11 packed LU and its
// persistent L/U conversion), not a full factorization/solve claim.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include "controlplane/control_plane.hpp"
#include "control_test_artifact.hpp"
#include "fp64_reference.hpp"

namespace {

namespace cmd = hw::command;
namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace testing = hw::testing;
namespace ref = hw::reference_fp64;

int failures = 0;
int checks = 0;
std::string g_output_dir;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::uint32_t f32_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Deterministic, asymmetric, strictly diagonally dominant test matrix.
std::vector<float> make_matrix(std::uint32_t n, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(n) * n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            const int pattern =
                static_cast<int>((i * 7 + j * 13 + seed * 29) % 11) - 5;
            float value = 0.25f * static_cast<float>(pattern);
            if (i == j) {
                value += 4.0f;
            }
            values[static_cast<std::size_t>(i) * n + j] = value;
        }
    }
    return values;
}

cp::ControlPlaneConfig base_config(std::uint32_t pe_count = 1) {
    cp::ControlPlaneConfig config;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 1;
    config.trsm_unit_count = 2;
    config.gemm_unit_count = 1;
    config.solve_unit_count = 1;
    config.assemble_unit_count = 1;
    config.factor_unit_count = 1;
    config.buffer_slot_count = 5;
    config.buffer_capacity_bytes = 64 * 1024;
    config.buffer_port_count = 4;
    config.command_window_depth = 16;
    config.completion_queue_depth = 16;
    config.dma_queue_depth = 8;
    config.backend = cp::ExecBackend::RequireRealKernels;
    config.enable_test_stub = false;
    config.enable_real_panel = true;
    config.memory.startup_cycles = 1;
    config.memory.beat_bytes = 16;
    config.memory.beat_cycles = 1;
    config.dma.beat_bytes = 16;
    config.dma.startup_cycles = 1;
    config.dma.beat_interval_cycles = 1;
    config.panel.compute_group_count = pe_count;
    config.panel.pes_per_group = 1;
    config.panel.pe_count = pe_count;
    config.panel.task_slots_per_pe = 4;
    config.panel.dispatcher_queue_depth = 32;
    return config;
}

testing::TestArtifact build_artifact(
    const testing::TestNode& node,
    std::uint32_t threshold_bits = 0) {
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.threshold_bits = threshold_bits;
    options.tile_size = 16;
    options.matrix_dim = node.total_dim;
    return testing::build_test_artifact({node}, options);
}

testing::TestNode make_node(
    std::uint32_t node_id,
    std::uint32_t dim,
    std::uint32_t seed) {
    testing::TestNode node;
    node.node_id = node_id;
    node.parent_id = -1;
    node.total_dim = dim;
    node.pivot_dim = dim;
    node.front_values = make_matrix(dim, seed);
    return node;
}

std::uint32_t command_index(
    const testing::TestArtifact& artifact,
    cmd::Opcode opcode) {
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode == opcode) {
            return index;
        }
    }
    return cmd::NONE;
}

std::vector<float> region_f32(
    cp::CommandDevice& device,
    const std::string& name,
    std::size_t count) {
    const auto* region = device.boot().region(name);
    if (region == nullptr) {
        return {};
    }
    std::vector<float> values(count, 0.0f);
    const auto& image = device.dataplane().memory().image();
    std::memcpy(values.data(), image.data() + region->offset, count * 4);
    return values;
}

std::vector<std::uint8_t> region_bytes(
    cp::CommandDevice& device,
    const std::string& name) {
    const auto* region = device.boot().region(name);
    if (region == nullptr) {
        return {};
    }
    const auto& image = device.dataplane().memory().image();
    return std::vector<std::uint8_t>(
        image.begin() + static_cast<std::ptrdiff_t>(region->offset),
        image.begin() + static_cast<std::ptrdiff_t>(region->offset + region->size));
}

struct RunResult {
    bool started{false};
    bool idle{false};
    cp::BatchState state{cp::BatchState::Idle};
    std::map<std::uint32_t, cmd::StatusCode> status;
    std::vector<std::uint8_t> front_snapshot;
    bool snapshot_taken{false};
    std::uint64_t cycles{0};
};

RunResult run_device(
    cp::CommandDevice& device,
    std::uint32_t node_id,
    std::uint32_t panel_command,
    std::uint64_t max_cycles = 2'000'000) {
    RunResult result;
    std::string error;
    result.started = device.start(&error);
    if (!result.started) {
        std::cerr << "start refused: " << error << '\n';
        return result;
    }
    while (!device.finished() && device.cycle() < max_cycles) {
        device.tick();
        if (!result.snapshot_taken &&
            device.completion_status().count(panel_command) != 0) {
            const dp::BufferHandle front = device.front_slot(node_id);
            if (front.slot != cmd::NONE) {
                result.front_snapshot = device.dataplane().buffers().bytes(front.slot);
                result.snapshot_taken = true;
            }
        }
    }
    result.idle = device.finished();
    result.state = device.state();
    result.status = device.completion_status();
    result.cycles = device.cycle();
    return result;
}

// Max memory-write data_cycle (MemoryService finish) of one command, read from
// the independent dataplane trace; 0 when the command had no writes.
std::uint64_t last_write_finish_cycle(
    cp::CommandDevice& device,
    std::uint32_t command_id) {
    std::uint64_t last = 0;
    for (const auto& row : device.dataplane().trace().transactions()) {
        if (row.layer != "memory" || !row.write || !row.has_command ||
            row.command_id != command_id) {
            continue;
        }
        last = std::max(last, row.data_cycle);
    }
    return last;
}

std::vector<float> packed_from_snapshot(
    const std::vector<std::uint8_t>& snapshot,
    std::uint32_t total_dim) {
    std::vector<float> values(static_cast<std::size_t>(total_dim) * total_dim, 0.0f);
    if (snapshot.size() < values.size() * 4) {
        return values;
    }
    std::memcpy(values.data(), snapshot.data(), values.size() * 4);
    return values;
}

// Independent FP64 reference under the same STATIC policy.
ref::PanelLuResult reference_factor(
    const std::vector<float>& matrix,
    std::uint32_t n,
    std::uint32_t pivot) {
    std::vector<double> values(matrix.begin(), matrix.end());
    ref::DenseMatrix dense(n, n, values);
    std::vector<std::size_t> variables(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        variables[i] = i;
    }
    const ref::FrontFactor front =
        ref::factor_front(variables, dense, pivot, ref::PivotPolicy::Static, 0.0);
    return front.factorization;
}

void check_factor_match(
    const std::vector<float>& device_l,
    const std::vector<float>& device_u,
    const ref::PanelLuResult& reference,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    const std::string& label,
    bool panel_block_only = false) {
    double max_l_diff = 0.0;
    double max_u_diff = 0.0;
    // For n>p only the p x p panel block is compared: L21/U12 belong to the
    // top-level TRSM commands, which are not implemented in T07.
    const std::uint32_t l_rows = panel_block_only ? pivot_dim : total_dim;
    const std::uint32_t u_cols = panel_block_only ? pivot_dim : total_dim;
    for (std::uint32_t row = 0; row < l_rows; ++row) {
        for (std::uint32_t col = 0; col < pivot_dim; ++col) {
            const double expected = reference.l.at(row, col);
            const double got = device_l[static_cast<std::size_t>(row) * pivot_dim + col];
            max_l_diff = std::max(max_l_diff, std::fabs(expected - got));
        }
    }
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        for (std::uint32_t col = 0; col < u_cols; ++col) {
            const double expected = reference.u.at(row, col);
            const double got = device_u[static_cast<std::size_t>(row) * total_dim + col];
            max_u_diff = std::max(max_u_diff, std::fabs(expected - got));
        }
    }
    expect(max_l_diff < 5.0e-3, label + ": L matches the FP64 STATIC reference");
    expect(max_u_diff < 5.0e-3, label + ": U matches the FP64 STATIC reference");
}

void check_masks(
    const std::vector<float>& device_l,
    const std::vector<float>& device_u,
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    const std::string& label) {
    bool l_mask = true;
    bool l_unit = true;
    bool u_mask = true;
    for (std::uint32_t row = 0; row < total_dim; ++row) {
        for (std::uint32_t col = 0; col < pivot_dim; ++col) {
            const float value = device_l[static_cast<std::size_t>(row) * pivot_dim + col];
            if (col > row && value != 0.0f) {
                l_mask = false;
            }
            if (col == row && value != 1.0f) {
                l_unit = false;
            }
        }
    }
    for (std::uint32_t row = 0; row < pivot_dim; ++row) {
        for (std::uint32_t col = 0; col < total_dim; ++col) {
            if (col < row) {
                const float value =
                    device_u[static_cast<std::size_t>(row) * total_dim + col];
                if (value != 0.0f) {
                    u_mask = false;
                }
            }
        }
    }
    expect(l_mask, label + ": L upper triangle is zero-filled");
    expect(l_unit, label + ": L diagonal is explicitly unit");
    expect(u_mask, label + ": U lower triangle is zero-filled");
}

// ---------------------------------------------------------------------------

void test_numeric_np_inprocess() {
    for (const std::uint32_t n : {4u, 15u, 16u, 17u, 23u}) {
        const std::vector<float> matrix = make_matrix(n, n + 1);
        testing::TestNode node;
        node.node_id = 0;
        node.parent_id = -1;
        node.total_dim = n;
        node.pivot_dim = n;
        node.front_values = matrix;
        testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        const std::uint32_t factor_command =
            command_index(artifact, cmd::Opcode::StoreFactor);
        cp::CommandDevice device(artifact.artifact, base_config(1));
        const RunResult run = run_device(device, 0, panel_command);
        const std::string label = "n=p=" + std::to_string(n);
        expect(run.started && run.idle, label + ": batch reaches a terminal state");
        expect(
            run.status.at(panel_command) == cmd::StatusCode::Ok,
            label + ": PANEL_LU succeeds");
        expect(
            run.status.at(factor_command) == cmd::StatusCode::Ok,
            label + ": STORE_FACTOR succeeds");
        // Top-level TRSM/GEMM/solve are not implemented in T07; the batch must
        // end in explicit refusal, not a stubbed success. Panel numerics below
        // are still device FP32 and reference-checked.
        if (command_index(artifact, cmd::Opcode::TrsmLeft) != cmd::NONE) {
            expect(
                run.status.at(command_index(artifact, cmd::Opcode::TrsmLeft)) ==
                    cmd::StatusCode::PolicyNotSupported,
                label + ": top-level TRSM_LEFT is explicitly refused");
        }
        expect(
            run.status.at(command_index(artifact, cmd::Opcode::SolveForward)) ==
                cmd::StatusCode::PolicyNotSupported,
            label + ": SOLVE is explicitly refused (no full-solve claim)");

        const std::vector<float> device_l = region_f32(device, "node.0.factor_l",
                                                       static_cast<std::size_t>(n) * n);
        const std::vector<float> device_u = region_f32(device, "node.0.factor_u",
                                                       static_cast<std::size_t>(n) * n);
        const ref::PanelLuResult reference = reference_factor(matrix, n, n);
        check_factor_match(device_l, device_u, reference, n, n, label);
        check_masks(device_l, device_u, n, n, label);

        // Packed A11 reconstructs the input matrix: A11 = L * U.
        double max_error = 0.0;
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (std::uint32_t t = 0; t < n; ++t) {
                    sum += static_cast<double>(
                               device_l[static_cast<std::size_t>(i) * n + t]) *
                           static_cast<double>(
                               device_u[static_cast<std::size_t>(t) * n + j]);
                }
                max_error = std::max(
                    max_error,
                    std::fabs(sum - static_cast<double>(
                                        matrix[static_cast<std::size_t>(i) * n + j])));
            }
        }
        expect(max_error < 5.0e-3, label + ": L*U reconstructs A11");

        // STATIC success writes the full identity P-vector.
        const std::vector<std::uint8_t> pvector =
            region_bytes(device, "node.0.p_vector");
        bool identity = pvector.size() == static_cast<std::size_t>(n) * 4;
        for (std::uint32_t index = 0; index < n && identity; ++index) {
            std::uint32_t value = 0;
            std::memcpy(
                &value, pvector.data() + static_cast<std::size_t>(index) * 4, 4);
            identity = value == index;
        }
        expect(identity, label + ": P-vector is the identity");

        if (n == 23 && !g_output_dir.empty()) {
            // One canonical run writes the T07 evidence reports into the
            // caller-provided build directory (never the repository).
            device.write_reports(g_output_dir);
            std::ifstream summary(g_output_dir + "/summary.json");
            expect(summary.good(), label + ": summary.json written");
            std::ifstream tasks(g_output_dir + "/panel_tasks.csv");
            expect(tasks.good(), label + ": panel_tasks.csv written");
        }

        const auto& records = device.panel_engine()->task_records();
        for (std::size_t i = 0; i < records.size(); ++i) {
            for (std::size_t j = i + 1; j < records.size(); ++j) {
                if (records[i].pe_id != records[j].pe_id) {
                    continue;
                }
                const bool overlap =
                    records[i].exec_start_cycle < records[j].exec_done_cycle &&
                    records[j].exec_start_cycle < records[i].exec_done_cycle;
                expect(!overlap, label + ": one PE task at a time");
            }
            for (const std::uint64_t dep : records[i].task.deps) {
                for (const auto& producer : records) {
                    if (producer.task.task_id != dep) {
                        continue;
                    }
                    expect(
                        producer.committed &&
                            records[i].dispatch_cycle > producer.writeback_done_cycle,
                        label + ": dependent dispatch waits for the producer writeback");
                }
            }
        }
    }
}

void test_n_gt_p_outer_blocks_unchanged() {
    const std::uint32_t n = 17;
    const std::uint32_t p = 16;
    const std::vector<float> matrix = make_matrix(n, 99);
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = p;
    node.front_values = matrix;
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    const std::uint32_t panel_command = command_index(artifact, cmd::Opcode::PanelLu);
    cp::CommandDevice device(artifact.artifact, base_config(2));
    const RunResult run = run_device(device, 0, panel_command);

    expect(run.started, "n>p: device starts");
    expect(
        run.status.at(panel_command) == cmd::StatusCode::Ok,
        "n>p: PANEL_LU succeeds on A11");
    expect(
        run.status.at(command_index(artifact, cmd::Opcode::TrsmLeft)) ==
            cmd::StatusCode::Ok,
        "n>p: top-level TRSM_LEFT now executes on the shared engine (T09)");
    expect(
        run.status.at(command_index(artifact, cmd::Opcode::TrsmRight)) ==
            cmd::StatusCode::Ok,
        "n>p: top-level TRSM_RIGHT now executes on the shared engine (T09)");
    expect(
        run.status.at(command_index(artifact, cmd::Opcode::GemmSchur)) ==
            cmd::StatusCode::Ok,
        "n>p: top-level GEMM_SCHUR now executes on the shared engine (T10)");
    expect(
        run.status.at(command_index(artifact, cmd::Opcode::StoreFactor)) ==
            cmd::StatusCode::Ok,
        "n>p: STORE_FACTOR saves the full L/U after TRSM (T09)");
    // SOLVE is not implemented: it must not report success (no full-solve
    // claim) even though the decomposition chain now reaches node commit.
    expect(
        run.status.at(command_index(artifact, cmd::Opcode::SolveForward)) !=
            cmd::StatusCode::Ok,
        "n>p: SOLVE cannot succeed, no full-solve claim");

    expect(run.snapshot_taken, "n>p: front snapshot taken while resident");
    const std::vector<float> packed = packed_from_snapshot(run.front_snapshot, n);
    expect(!packed.empty(), "n>p: snapshot decoded");
    const ref::PanelLuResult reference = reference_factor(matrix, n, p);
    // STORE_FACTOR is dependency-blocked in n>p (top-level TRSM is refused), so
    // the packed factors are verified on the resident front snapshot: build the
    // L/U views of the p x p block from the packed values.
    std::vector<float> device_l(static_cast<std::size_t>(n) * p, 0.0f);
    std::vector<float> device_u(static_cast<std::size_t>(p) * n, 0.0f);
    for (std::uint32_t i = 0; i < p; ++i) {
        for (std::uint32_t j = 0; j < p; ++j) {
            const float got = packed[static_cast<std::size_t>(i) * n + j];
            if (i > j) {
                device_l[static_cast<std::size_t>(i) * p + j] = got;
            } else if (i == j) {
                device_l[static_cast<std::size_t>(i) * p + j] = 1.0f;
                device_u[static_cast<std::size_t>(i) * n + j] = got;
            } else {
                device_u[static_cast<std::size_t>(i) * n + j] = got;
            }
        }
    }
    check_factor_match(device_l, device_u, reference, n, p, "n>p", true);
    check_masks(device_l, device_u, n, p, "n>p");

    // A11 occupies [0,p) x [0,p) of the scratch front; the outer blocks must be
    // exactly the assembled input (no panel task touches A12/A21/A22).
    bool outer_unchanged = true;
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            if (i < p && j < p) {
                continue;
            }
            if (packed[static_cast<std::size_t>(i) * n + j] !=
                matrix[static_cast<std::size_t>(i) * n + j]) {
                outer_unchanged = false;
            }
        }
    }
    expect(outer_unchanged, "n>p: A12/A21/A22 are byte-unchanged");
}

void test_failure_zero_pivot() {
    const std::uint32_t n = 16;
    std::vector<float> matrix = make_matrix(n, 7);
    // Exact zero after eliminating column 2: row 3 is a full copy of row 2, so
    // all earlier multipliers match and the (3,3) pivot becomes exactly zero.
    for (std::uint32_t col = 0; col < n; ++col) {
        matrix[3 * n + col] = matrix[2 * n + col];
    }
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = n;
    node.front_values = matrix;
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    const std::uint32_t panel_command = command_index(artifact, cmd::Opcode::PanelLu);
    cp::CommandDevice device(artifact.artifact, base_config(1));
    const RunResult run = run_device(device, 0, panel_command);

    expect(
        run.status.at(panel_command) == cmd::StatusCode::StaticPivotFailure,
        "zero pivot -> STATIC_PIVOT_FAILURE (not a singularity claim)");
    std::uint32_t pivot_count = 0;
    for (const auto& stats : device.command_stats()) {
        if (stats.command_id == panel_command) {
            pivot_count = stats.pivot_count;
        }
    }
    expect(pivot_count == 3, "zero pivot failure reports the failing column");
    // No P-vector, no factor conversion on failure.
    const std::vector<std::uint8_t> pvector = region_bytes(device, "node.0.p_vector");
    bool sentinel = !pvector.empty();
    for (std::size_t index = 0; index < pvector.size(); ++index) {
        if (pvector[index] != 0xFF) {
            sentinel = false;
        }
    }
    expect(sentinel, "failed command leaves the P-vector sentinel untouched");
    const std::vector<std::uint8_t> l_bytes = region_bytes(device, "node.0.factor_l");
    bool factor_untouched = !l_bytes.empty();
    for (const std::uint8_t byte : l_bytes) {
        if (byte != 0xEE) {
            factor_untouched = false;
        }
    }
    expect(factor_untouched, "failed command writes no L factors");
    expect(
        run.status.at(command_index(artifact, cmd::Opcode::StoreFactor)) !=
            cmd::StatusCode::Ok,
        "STORE_FACTOR does not run after a panel failure");
}

void test_failure_eliminated_zero_and_threshold_and_nan() {
    {
        const std::uint32_t n = 16;
        std::vector<float> matrix = make_matrix(n, 11);
        // Row 6 becomes a full copy of row 5: all earlier multipliers match,
        // so eliminating column 5 leaves an exact zero pivot at column 6 even
        // though the input diagonal is non-zero.
        for (std::uint32_t col = 0; col < n; ++col) {
            matrix[6 * n + col] = matrix[5 * n + col];
        }
        testing::TestNode node;
        node.node_id = 0;
        node.parent_id = -1;
        node.total_dim = n;
        node.pivot_dim = n;
        node.front_values = matrix;
        testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        cp::CommandDevice device(artifact.artifact, base_config(1));
        const RunResult run = run_device(device, 0, panel_command);
        expect(
            run.status.at(panel_command) == cmd::StatusCode::StaticPivotFailure,
            "eliminated zero pivot -> STATIC_PIVOT_FAILURE");
        std::uint32_t pivot_count = 0;
        for (const auto& stats : device.command_stats()) {
            if (stats.command_id == panel_command) {
                pivot_count = stats.pivot_count;
            }
        }
        expect(pivot_count == 6, "eliminated zero pivot reports the actual column");
    }
    {
        // PIVOT_UNSTABLE: every element is 2.0 except a unit diagonal; with a
        // 0.9 threshold ratio the first pivot (1.0) is below 0.9 * maxabs = 1.8.
        const std::uint32_t n = 16;
        std::vector<float> matrix(static_cast<std::size_t>(n) * n, 2.0f);
        for (std::uint32_t i = 0; i < n; ++i) {
            matrix[i * n + i] = 1.0f;
        }
        testing::TestNode node;
        node.node_id = 0;
        node.parent_id = -1;
        node.total_dim = n;
        node.pivot_dim = n;
        node.front_values = matrix;
        testing::TestArtifact artifact = build_artifact(node, f32_bits(0.9f));
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        cp::CommandDevice device(artifact.artifact, base_config(1));
        const RunResult run = run_device(device, 0, panel_command);
        expect(
            run.status.at(panel_command) == cmd::StatusCode::PivotUnstable,
            "pivot <= threshold_ratio * maxabs -> PIVOT_UNSTABLE");
        std::uint32_t pivot_count = 0;
        for (const auto& stats : device.command_stats()) {
            if (stats.command_id == panel_command) {
                pivot_count = stats.pivot_count;
            }
        }
        expect(pivot_count == 0, "threshold failure reports the failing column");
    }
    {
        const std::uint32_t n = 16;
        std::vector<float> matrix = make_matrix(n, 13);
        matrix[1 * n + 2] = std::nanf("");
        testing::TestNode node;
        node.node_id = 0;
        node.parent_id = -1;
        node.total_dim = n;
        node.pivot_dim = n;
        node.front_values = matrix;
        testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        cp::CommandDevice device(artifact.artifact, base_config(1));
        const RunResult run = run_device(device, 0, panel_command);
        expect(
            run.status.at(panel_command) == cmd::StatusCode::NonFiniteValue,
            "NaN in the front -> NON_FINITE_VALUE from the full-front scan");
        expect(
            device.panel_engine()->task_records().empty(),
            "no tile task runs when the scan rejects non-finite input");
        const std::vector<std::uint8_t> pvector = region_bytes(device, "node.0.p_vector");
        bool sentinel = !pvector.empty();
        for (const std::uint8_t byte : pvector) {
            if (byte != 0xFF) {
                sentinel = false;
            }
        }
        expect(sentinel, "scan failure leaves the P-vector sentinel untouched");
    }
}

void test_dual_front_and_directed_abort() {
    testing::TestNode node0;
    node0.node_id = 0;
    node0.parent_id = -1;
    node0.total_dim = 16;
    node0.pivot_dim = 16;
    node0.front_values = make_matrix(16, 21);
    testing::TestNode node1;
    node1.node_id = 1;
    node1.parent_id = -1;
    node1.total_dim = 16;
    node1.pivot_dim = 16;
    node1.front_values = make_matrix(16, 33);
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    testing::TestArtifact artifact =
        testing::build_test_artifact({node0, node1}, options);
    std::uint32_t panel0 = cmd::NONE;
    std::uint32_t panel1 = cmd::NONE;
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode != cmd::Opcode::PanelLu) {
            continue;
        }
        if (artifact.commands[index].node_id == 0) {
            panel0 = index;
        } else {
            panel1 = index;
        }
    }
    expect(
        panel0 != cmd::NONE && panel1 != cmd::NONE,
        "dual front: both panel commands exist");

    cp::CommandDevice device(artifact.artifact, base_config(2));
    std::string error;
    expect(device.start(&error), "dual front: device starts");
    // Abort while node 0's panel still has pending work: wait until at least
    // one tile task has completed but the panel command has not finished.
    while (!device.finished() && device.cycle() < 1000000 &&
           !(device.panel_engine()->task_records().size() >= 1 &&
             device.completion_status().count(panel0) == 0)) {
        device.tick();
    }
    expect(
        device.panel_engine()->task_records().size() >= 1 &&
            device.completion_status().count(panel0) == 0,
        "dual front: node 0 is mid-panel when the abort is issued");
    device.abort_node(0);
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    expect(device.finished(), "dual front: batch terminates after abort");
    const auto& status = device.completion_status();
    expect(
        status.at(panel0) == cmd::StatusCode::Aborted,
        "directed abort cancels node 0's panel");
    expect(
        status.at(panel1) == cmd::StatusCode::Ok,
        "directed abort does not hurt the other front");
    expect(
        device.panel_engine()->task_records().size() > 0,
        "aborted front leaves diagnostics");
    // After the abort drains, the engine context is reusable.
    expect(
        !device.panel_engine()->unit_busy(0),
        "aborted engine context is released for reuse");
}


void test_abort_during_scan() {
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = 32;
    node.pivot_dim = 32;
    node.front_values = make_matrix(32, 77);
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    const std::uint32_t panel_command = command_index(artifact, cmd::Opcode::PanelLu);
    cp::CommandDevice device(artifact.artifact, base_config(1));
    std::string error;
    expect(device.start(&error), "scan abort: device starts");
    // A few cycles into the full-front scan (32x32 = 4096B at 64B/cycle).
    for (int i = 0; i < 6 && !device.finished(); ++i) {
        device.tick();
    }
    device.abort_node(0);
    while (!device.finished() && device.cycle() < 1000000) {
        device.tick();
    }
    expect(device.finished(), "scan abort: batch terminates");
    expect(
        device.completion_status().at(panel_command) == cmd::StatusCode::Aborted,
        "abort during the front scan yields ABORTED, not a partial success");
    expect(
        !device.panel_engine()->unit_busy(0),
        "scan abort releases the engine context after draining");
}


// Issue 1/3: a bound-but-busy factor unit must be waited on (never the legacy
// rectangular DMA fallback), and factor responses must be routed by request id
// to the owning instance only.
void test_factor_busy_wait_and_precise_routing() {
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = 32;
    testing::TestArtifact artifact = testing::build_test_artifact(
        {make_node(0, 32, 61), make_node(1, 32, 73)}, options);
    cp::ControlPlaneConfig config = base_config(2);
    config.factor_unit_count = 1;  // second STORE_FACTOR must wait
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "factor busy: device starts");
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    expect(device.finished(), "factor busy: batch terminates");
    expect(
        !device.panel_engine()->unit_busy(0),
        "factor busy: shared engine drained by batch completion");

    // Both nodes' factors must be the real converted L/U (the old DMA path
    // would leave L[0,0]=2.75 and L[0,1]=-0.75 for the second node).
    for (std::uint32_t node = 0; node < 2; ++node) {
        const std::string prefix = "node." + std::to_string(node) + ".";
        const std::vector<float> l = region_f32(device, prefix + "factor_l", 32 * 32);
        const std::vector<float> u = region_f32(device, prefix + "factor_u", 32 * 32);
        expect(
            !l.empty() && l[0] == 1.0f && l[1] == 0.0f,
            "factor busy: node " + std::to_string(node) +
                " has unit-diagonal/masked L (no DMA fallback)");
        const ref::PanelLuResult reference =
            reference_factor(make_matrix(32, node == 0 ? 61 : 73), 32, 32);
        check_factor_match(l, u, reference, 32, 32,
                           "factor busy node " + std::to_string(node));
    }

    // Factor responses must not finish before their own last write confirmed
    // (independent evidence from the dataplane memory trace).
    for (const auto& stats : device.command_stats()) {
        if (stats.unit_kind != "factor") {
            continue;
        }
        const std::uint64_t confirm =
            last_write_finish_cycle(device, stats.command_id);
        if (!(confirm > 0 && stats.finish_cycle >= confirm)) {
            std::cerr << "DEBUG factor cmd " << stats.command_id
                      << " finish " << stats.finish_cycle << " confirm " << confirm
                      << " unit " << stats.unit_id << "\n";
        }
        expect(
            confirm > 0 && stats.finish_cycle >= confirm,
            "factor routing: terminal response waits for the last DDR write");
    }
}

void test_factor_multi_instance_and_reuse() {
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = 16;
    testing::TestArtifact artifact = testing::build_test_artifact(
        {make_node(0, 16, 91), make_node(1, 16, 97)}, options);
    cp::ControlPlaneConfig config = base_config(2);
    config.factor_unit_count = 2;  // two instances may overlap
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "factor multi: device starts");
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    expect(device.finished(), "factor multi: batch terminates");
    // Both instances processed writes without stealing each other's responses:
    // every factor command finishes only after its own last DDR write.
    for (const auto& stats : device.command_stats()) {
        if (stats.unit_kind != "factor") {
            continue;
        }
        const std::uint64_t confirm =
            last_write_finish_cycle(device, stats.command_id);
        if (!(confirm > 0 && stats.finish_cycle >= confirm)) {
            std::cerr << "DEBUG factor multi cmd " << stats.command_id
                      << " finish " << stats.finish_cycle << " confirm " << confirm
                      << "\n";
        }
        expect(
            confirm > 0 && stats.finish_cycle >= confirm,
            "factor multi: each instance waits for its own writes");
    }
    for (std::uint32_t node = 0; node < 2; ++node) {
        const std::string prefix = "node." + std::to_string(node) + ".";
        const std::vector<float> l = region_f32(device, prefix + "factor_l", 256);
        const std::vector<float> u = region_f32(device, prefix + "factor_u", 256);
        const ref::PanelLuResult reference =
            reference_factor(make_matrix(16, node == 0 ? 91 : 97), 16, 16);
        check_factor_match(l, u, reference, 16, 16,
                           "factor multi node " + std::to_string(node));
    }

    // Instance reuse across batches (no stale pending ids): reset and rerun.
    device.reset();
    expect(device.start(&error), "factor reuse: restart accepted");
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    expect(device.finished(), "factor reuse: second batch terminates");
    for (std::uint32_t node = 0; node < 2; ++node) {
        const std::string prefix = "node." + std::to_string(node) + ".";
        const std::vector<float> l = region_f32(device, prefix + "factor_l", 256);
        const std::vector<float> u = region_f32(device, prefix + "factor_u", 256);
        expect(
            !l.empty() && l[0] == 1.0f && u[0] == make_matrix(16, node == 0 ? 91 : 97)[0],
            "factor reuse node " + std::to_string(node) +
                ": factors rewritten after reset");
    }
}

// Issue 4: memory queue backpressure must wait, not report ADDRESS_FAULT.
void test_factor_queue_backpressure_is_not_an_error() {
    testing::TestNode node = make_node(0, 16, 101);
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    cp::ControlPlaneConfig config = base_config(1);
    config.memory.max_inflight = 1;  // legal, tight memory queue
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "factor bp: device starts");
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    expect(device.finished(), "factor bp: batch terminates");
    const std::uint32_t factor_command =
        command_index(artifact, cmd::Opcode::StoreFactor);
    expect(
        device.completion_status().at(factor_command) == cmd::StatusCode::Ok,
        "factor bp: STORE_FACTOR succeeds under legal memory backpressure");
    const std::vector<float> l = region_f32(device, "node.0.factor_l", 256);
    const std::vector<float> u = region_f32(device, "node.0.factor_u", 256);
    const ref::PanelLuResult reference = reference_factor(make_matrix(16, 101), 16, 16);
    check_factor_match(l, u, reference, 16, 16, "factor bp");
}

// Issue 5: operand bytes must be fetched through the bounded staged load; the
// access events happen during Loading, not at accept time.
void test_operand_load_is_budget_constrained() {
    testing::TestNode node = make_node(0, 16, 131);
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    cp::ControlPlaneConfig config = base_config(1);
    config.panel.load_bytes_per_cycle = 8;  // 1024B of diag operands => >=128 cycles
    cp::CommandDevice device(artifact.artifact, config);
    const std::uint32_t panel_command =
        command_index(artifact, cmd::Opcode::PanelLu);
    const RunResult run = run_device(device, 0, panel_command);
    expect(run.started && run.idle, "staged load: batch terminates");
    expect(
        run.status.at(panel_command) == cmd::StatusCode::Ok,
        "staged load: PANEL_LU succeeds");
    const auto& records = device.panel_engine()->task_records();
    expect(!records.empty(), "staged load: diag task recorded");
    if (!records.empty()) {
        const auto& diag = records.front();
        expect(
            diag.bytes_read == 1024,
            "staged load: exactly the operand bytes were fetched");
        expect(
            diag.load_done_cycle - diag.dispatch_cycle >= 128,
            "staged load: loading takes at least bytes/budget cycles");
        expect(
            diag.exec_start_cycle >= diag.load_done_cycle + 2,
            "staged load: execution starts only after the load latency");
        expect(
            diag.exec_start_cycle > diag.dispatch_cycle,
            "staged load: no accept-time precomputation shortcut");
    }
}

// Issue 2: an abort only requests cancellation; the batch must wait for the
// engine to drain before releasing resources or declaring completion.
void test_abort_waits_for_engine_drain() {
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = 16;
    testing::TestArtifact artifact = testing::build_test_artifact(
        {make_node(0, 16, 151), make_node(1, 16, 157)}, options);
    cp::ControlPlaneConfig config = base_config(2);
    config.panel.timing.div_latency_cycles = 10000;  // long-running diag
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "abort drain: device starts");
    while (!device.finished() && device.cycle() < 2000000 &&
           !(device.panel_engine()->dispatcher().dispatched_count() >= 1 &&
             device.dataplane().buffers().active_lease_count() > 0)) {
        device.tick();
    }
    expect(
        device.dataplane().buffers().active_lease_count() > 0,
        "abort drain: a front lease is held by the running panel");
    const std::uint32_t panel0 = [&] {
        for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
            if (artifact.commands[index].opcode == cmd::Opcode::PanelLu &&
                artifact.commands[index].node_id == 0) {
                return index;
            }
        }
        return cmd::NONE;
    }();
    device.abort_node(0);
    expect(
        device.dataplane().buffers().active_lease_count() > 0,
        "abort drain: cancellation does not free the lease immediately");
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    expect(device.finished(), "abort drain: batch terminates");
    expect(
        device.completion_status().at(panel0) == cmd::StatusCode::Aborted,
        "abort drain: panel terminates as ABORTED");
    expect(
        device.dataplane().buffers().active_lease_count() == 0,
        "abort drain: leases released only after the engine drained");
    expect(
        !device.panel_engine()->unit_busy(0),
        "abort drain: engine context drained before batch completion");
    for (std::size_t index = 0; index < config.panel.compute_group_count; ++index) {
        expect(
            !device.panel_engine()->group_busy(static_cast<std::uint32_t>(index)),
            "abort drain: every PE group is released at completion");
    }
}

// Scope gap: explicit FP64 module diagnostic mode (never an automatic
// fallback). The discriminator is FP32 reciprocal overflow: 1e-39f is a
// representable binary32 subnormal whose binary32 reciprocal overflows to Inf,
// while binary64 keeps 1e-39 (normal double) and the reciprocal finite.
// The input value itself is exact in both formats, so the difference comes
// from the declared numeric mode, not from lost input bits.
void test_fp64_diagnostic_mode() {
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = 1;
    node.pivot_dim = 1;
    node.front_values = {1.0e-39f};
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = 1;
    {
        // Default FP32 device mode: reciprocal overflows -> NON_FINITE_VALUE.
        testing::TestArtifact artifact =
            testing::build_test_artifact({node}, options);
        cp::CommandDevice device(artifact.artifact, base_config(1));
        std::string error;
        expect(device.start(&error), "fp64 mode: fp32 device starts");
        while (!device.finished() && device.cycle() < 200000) {
            device.tick();
        }
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        expect(
            device.completion_status().at(panel_command) ==
                cmd::StatusCode::NonFiniteValue,
            "fp64 mode: default FP32 device reports the reciprocal overflow");
    }
    {
        // Explicit FP64 diagnostic mode: binary64 reciprocal is finite, the
        // panel succeeds and stores the FP32-rounded factor. The mode must be
        // declared by configuration; nothing falls back automatically.
        testing::TestArtifact artifact =
            testing::build_test_artifact({node}, options);
        cp::ControlPlaneConfig config = base_config(1);
        config.panel.numeric_mode = hw::panel_lu::PanelNumericMode::Fp64Diagnostic;
        cp::CommandDevice device(artifact.artifact, config);
        std::string error;
        expect(device.start(&error), "fp64 mode: diagnostic device starts");
        while (!device.finished() && device.cycle() < 200000) {
            device.tick();
        }
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        expect(
            device.completion_status().at(panel_command) == cmd::StatusCode::Ok,
            "fp64 mode: explicit FP64 diagnostic keeps the subnormal pivot");
        const std::vector<float> l = region_f32(device, "node.0.factor_l", 1);
        const std::vector<float> u = region_f32(device, "node.0.factor_u", 1);
        expect(l.size() == 1 && l[0] == 1.0f, "fp64 mode: L unit diagonal");
        expect(
            u.size() == 1 && u[0] == 1.0e-39f,
            "fp64 mode: U keeps the subnormal pivot");
        if (!g_output_dir.empty()) {
            device.write_reports(g_output_dir + "/fp64_mode");
            std::ifstream summary(g_output_dir + "/fp64_mode/summary.json");
            std::string text(
                (std::istreambuf_iterator<char>(summary)),
                std::istreambuf_iterator<char>());
            expect(
                text.find("fp64_diagnostic") != std::string::npos,
                "fp64 mode: summary marks the diagnostic mode");
        }
    }
    {
        // Coverage: a 16x16 panel under the diagnostic mode still matches the
        // FP64 reference within FP32 storage tolerance.
        testing::TestNode wide = make_node(0, 16, 181);
        testing::TestArtifact artifact = build_artifact(wide, f32_bits(0.0f));
        cp::ControlPlaneConfig config = base_config(1);
        config.panel.numeric_mode = hw::panel_lu::PanelNumericMode::Fp64Diagnostic;
        cp::CommandDevice device(artifact.artifact, config);
        std::string error;
        expect(device.start(&error), "fp64 mode: 16x16 device starts");
        while (!device.finished() && device.cycle() < 2000000) {
            device.tick();
        }
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        expect(
            device.completion_status().at(panel_command) == cmd::StatusCode::Ok,
            "fp64 mode: 16x16 panel succeeds");
        const std::vector<float> l = region_f32(device, "node.0.factor_l", 256);
        const std::vector<float> u = region_f32(device, "node.0.factor_u", 256);
        const ref::PanelLuResult reference =
            reference_factor(make_matrix(16, 181), 16, 16);
        check_factor_match(l, u, reference, 16, 16, "fp64 mode 16x16");
    }
}


// Rounding-overflow failure positions for LU_DIAG use the global pivot
// coordinate (pivot_base + local column). The second diagonal tile of an
// n=p=18 front overflows while rounding the binary64 result back to FP32;
// the report must carry column 16 and pivot_count 16, not column/pivot 0.
void test_fp64_diagnostic_rounding_failed_column() {
    const std::uint32_t n = 18;
    const float max_float = std::numeric_limits<float>::max();
    std::vector<float> front(n * n, 0.0f);
    for (std::uint32_t i = 0; i < 16; ++i) {
        front[i * n + i] = 1.0f;
    }
    front[16 * n + 16] = 0.5f;
    front[16 * n + 17] = max_float;
    front[17 * n + 16] = max_float;
    front[17 * n + 17] = max_float;

    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = n;
    node.front_values = front;
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = n;
    options.threshold_bits = f32_bits(0.0f);
    testing::TestArtifact artifact =
        testing::build_test_artifact({node}, options);

    cp::ControlPlaneConfig config = base_config(1);
    config.panel.numeric_mode = hw::panel_lu::PanelNumericMode::Fp64Diagnostic;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "fp64 rounding column: device starts");
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    const std::uint32_t panel_command =
        command_index(artifact, cmd::Opcode::PanelLu);
    expect(
        device.completion_status().at(panel_command) ==
            cmd::StatusCode::NonFiniteValue,
        "fp64 rounding column: overflow -> NON_FINITE_VALUE");
    std::uint32_t failed_tasks = 0;
    std::uint32_t failed_column = cmd::NONE;
    for (const auto& record : device.panel_engine()->task_records()) {
        // The failing task itself is also flagged cancelled for accounting;
        // it is identified by the non-Ok status.
        if (record.task.command_id != panel_command ||
            record.task.mode != hw::panel_lu::TileMode::LuDiag ||
            record.status == cmd::StatusCode::Ok) {
            continue;
        }
        ++failed_tasks;
        failed_column = record.failed_column;
        expect(
            record.task.pivot_base == 16,
            "fp64 rounding column: failing tile is the second diagonal tile");
    }
    expect(failed_tasks == 1, "fp64 rounding column: exactly one failed LU_DIAG");
    expect(
        failed_column == 16,
        "fp64 rounding column: failure reported at global column 16");
    bool pivot_checked = false;
    for (const auto& stats : device.command_stats()) {
        if (stats.command_id != panel_command) {
            continue;
        }
        pivot_checked = true;
        expect(
            stats.pivot_count == 16,
            "fp64 rounding column: pivot_count equals the completed pivots");
    }
    expect(pivot_checked, "fp64 rounding column: command stats present");
}

// Factor reuse after an abort between the L and U submit of the same row: the
// transaction-local partial-row flags must not leak into the next command.
void test_factor_abort_reuse_clears_partial_row() {
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = 16;
    // Node 1's panel is larger, so node 0's STORE_FACTOR is the first factor
    // transaction to reach the unit deterministically.
    testing::TestArtifact artifact =
        testing::build_test_artifact({make_node(0, 16, 201), make_node(1, 32, 211)},
                                     options);
    std::uint32_t factor0 = cmd::NONE;
    std::uint32_t factor1 = cmd::NONE;
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode != cmd::Opcode::StoreFactor) {
            continue;
        }
        if (artifact.commands[index].node_id == 0) {
            factor0 = index;
        } else {
            factor1 = index;
        }
    }
    expect(factor0 != cmd::NONE && factor1 != cmd::NONE, "factor abort: commands exist");

    cp::ControlPlaneConfig config = base_config(2);
    config.factor_unit_count = 1;
    config.memory.max_inflight = 1;  // L accepted, then U backpressure
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "factor abort: device starts");
    // Wait for the exact window: node 0's factor row 0 has one accepted L
    // write (U is still queued behind memory backpressure).
    while (!device.finished() && device.cycle() < 2000000 &&
           !(device.unit_active_command_id(cp::UnitKind::Factor, 0) == factor0 &&
             device.unit_pending_request_count(cp::UnitKind::Factor, 0) == 1)) {
        device.tick();
    }
    expect(
        device.unit_active_command_id(cp::UnitKind::Factor, 0) == factor0 &&
            device.unit_pending_request_count(cp::UnitKind::Factor, 0) == 1,
        "factor abort: reached L-accepted / U-backpressured window");
    device.abort_node(0);
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    expect(device.finished(), "factor abort: batch terminates");
    expect(
        device.completion_status().at(factor0) == cmd::StatusCode::Aborted,
        "factor abort: aborted front's STORE_FACTOR is ABORTED");
    expect(
        device.completion_status().at(factor1) == cmd::StatusCode::Ok,
        "factor abort: other front's STORE_FACTOR succeeds");
    // Full byte accounting: 32 rows x (L row 128B + U row 128B) = 8192B. The
    // stale-flag bug drops exactly one row (L row 0), leaving 8064B.
    std::uint64_t factor1_bytes = 0;
    for (const auto& stats : device.command_stats()) {
        if (stats.command_id == factor1) {
            factor1_bytes = stats.bytes_written;
        }
    }
    expect(
        factor1_bytes == 8192,
        "factor abort: surviving command writes the complete 8192 factor bytes");
    // Full content check, not just the status code.
    const std::vector<float> l = region_f32(device, "node.1.factor_l", 32 * 32);
    const std::vector<float> u = region_f32(device, "node.1.factor_u", 32 * 32);
    expect(
        !l.empty() && l[0] == 1.0f && l[1] == 0.0f,
        "factor abort: surviving L row 0 is unit diagonal and masked");
    const ref::PanelLuResult reference = reference_factor(make_matrix(32, 211), 32, 32);
    check_factor_match(l, u, reference, 32, 32, "factor abort reuse");
    check_masks(l, u, 32, 32, "factor abort reuse");
}

void test_reset_and_restart() {
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = 16;
    node.pivot_dim = 16;
    node.front_values = make_matrix(16, 55);
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    const std::uint32_t panel_command = command_index(artifact, cmd::Opcode::PanelLu);
    cp::CommandDevice device(artifact.artifact, base_config(1));
    std::string error;
    expect(device.start(&error), "reset: device starts");
    for (int i = 0; i < 200 && !device.finished(); ++i) {
        device.tick();
    }
    const std::uint64_t before = device.cycle();
    device.reset();
    expect(device.state() == cp::BatchState::Idle, "reset: back to idle");
    expect(device.cycle() >= before, "reset: cycle base is monotone");
    expect(device.panel_engine()->task_records().empty(), "reset clears task records");
    expect(device.start(&error), "reset: restart accepted");
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    expect(device.finished(), "reset: restarted batch finishes");
    expect(
        device.completion_status().at(panel_command) == cmd::StatusCode::Ok,
        "reset: post-reset panel succeeds");
}

void test_shared_read_bandwidth_and_backpressure() {
    testing::TestNode node0;
    node0.node_id = 0;
    node0.parent_id = -1;
    node0.total_dim = 23;
    node0.pivot_dim = 23;
    node0.front_values = make_matrix(23, 3);
    testing::TestNode node1;
    node1.node_id = 1;
    node1.parent_id = -1;
    node1.total_dim = 23;
    node1.pivot_dim = 23;
    node1.front_values = make_matrix(23, 5);
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::Static;
    options.tile_size = 16;
    options.matrix_dim = 23;
    testing::TestArtifact artifact =
        testing::build_test_artifact({node0, node1}, options);

    cp::ControlPlaneConfig config = base_config(1);
    config.panel.load_bytes_per_cycle = 16;   // shared scan + load bandwidth
    config.panel.dispatcher_queue_depth = 1;  // bounded queue backpressure
    config.panel.task_slots_per_pe = 1;
    config.panel.writeback_queue_depth = 1;
    config.panel.writeback_bytes_per_cycle = 4;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "contention: device starts");
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    expect(device.finished(), "contention: batch terminates");
    std::uint32_t panels_ok = 0;
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode == cmd::Opcode::PanelLu &&
            device.completion_status().at(index) == cmd::StatusCode::Ok) {
            ++panels_ok;
        }
    }
    expect(panels_ok == 2, "contention: both fronts complete (no starvation)");
    expect(!device.panel_engine()->control_fault(), "contention: no control fault");
}

void test_engine_capability_refusals() {
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = 16;
    node.pivot_dim = 16;
    node.front_values = make_matrix(16, 4);
    testing::TestArtifact artifact = build_artifact(node, f32_bits(0.0f));
    {
        // DYNAMIC_PANEL is refused by the engine (and by T05 before dispatch).
        testing::TestArtifactOptions options;
        options.policy = cmd::PivotPolicy::DynamicPanel;
        options.tile_size = 16;
        testing::TestNode dynamic_node = node;
        testing::TestArtifact dynamic_artifact =
            testing::build_test_artifact({dynamic_node}, options);
        cp::CommandDevice device(dynamic_artifact.artifact, base_config(1));
        std::string error;
        expect(device.start(&error), "dynamic: device starts");
        while (!device.finished() && device.cycle() < 100000) {
            device.tick();
        }
        const std::uint32_t dynamic_panel =
            command_index(dynamic_artifact, cmd::Opcode::PanelLu);
        expect(
            device.completion_status().at(dynamic_panel) ==
                cmd::StatusCode::PolicyNotSupported,
            "DYNAMIC_PANEL -> POLICY_NOT_SUPPORTED, never executed");
        expect(
            dynamic_artifact.artifact.config.region("node.0.p_vector") != nullptr,
            "dynamic: artifact loaded");
    }
    {
        // Unsupported descriptor.tile_size is refused before execution.
        testing::TestNode tile_node = node;
        testing::TestArtifactOptions options;
        options.policy = cmd::PivotPolicy::Static;
        options.tile_size = 8;
        testing::TestArtifact tile_artifact =
            testing::build_test_artifact({tile_node}, options);
        cp::CommandDevice device(tile_artifact.artifact, base_config(1));
        std::string error;
        expect(device.start(&error), "tile_size: device starts");
        while (!device.finished() && device.cycle() < 100000) {
            device.tick();
        }
        const std::uint32_t panel_command =
            command_index(tile_artifact, cmd::Opcode::PanelLu);
        expect(
            device.completion_status().at(panel_command) ==
                cmd::StatusCode::BadDescriptor,
            "tile_size=8 -> BAD_DESCRIPTOR (never silently ignored)");
    }
}

}  // namespace

int main(int argc, char** argv) {
    g_output_dir = argc > 1 ? argv[1] : "";
    const std::pair<const char*, void (*)()> tests[] = {
        {"test_numeric_np_inprocess", test_numeric_np_inprocess},
        {"test_n_gt_p_outer_blocks_unchanged", test_n_gt_p_outer_blocks_unchanged},
        {"test_failure_zero_pivot", test_failure_zero_pivot},
        {"test_failure_eliminated_zero_and_threshold_and_nan",
         test_failure_eliminated_zero_and_threshold_and_nan},
        {"test_dual_front_and_directed_abort", test_dual_front_and_directed_abort},
        {"test_abort_during_scan", test_abort_during_scan},
        {"test_reset_and_restart", test_reset_and_restart},
        {"test_shared_read_bandwidth_and_backpressure",
         test_shared_read_bandwidth_and_backpressure},
        {"test_engine_capability_refusals", test_engine_capability_refusals},
        {"test_factor_busy_wait_and_precise_routing",
         test_factor_busy_wait_and_precise_routing},
        {"test_factor_multi_instance_and_reuse", test_factor_multi_instance_and_reuse},
        {"test_factor_queue_backpressure_is_not_an_error",
         test_factor_queue_backpressure_is_not_an_error},
        {"test_operand_load_is_budget_constrained",
         test_operand_load_is_budget_constrained},
        {"test_abort_waits_for_engine_drain", test_abort_waits_for_engine_drain},
        {"test_fp64_diagnostic_mode", test_fp64_diagnostic_mode},
        {"test_fp64_diagnostic_rounding_failed_column",
         test_fp64_diagnostic_rounding_failed_column},
        {"test_factor_abort_reuse_clears_partial_row",
         test_factor_abort_reuse_clears_partial_row},
    };
    for (const auto& [name, function] : tests) {
        try {
            function();
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "EXCEPTION in " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
