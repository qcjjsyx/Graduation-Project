// T09 TRSM engine tests.
//
// Coverage:
//   - the three hand-computable cases from the T08 design (left 2x2, right
//     2x2, left 4x4 bidiagonal),
//   - the full n=42 / p=23 / u=19 real data path (LOAD -> PANEL -> TRSM_LEFT/
//     TRSM_RIGHT -> STORE_FACTOR) against an independent FP64 triangular
//     oracle, including boundary invariance and full-factor byte accounting,
//   - PE grouping (2 groups x 2 PEs), intra-command tile parallelism, group
//     reuse across kinds/fronts and PE exclusivity,
//   - unified on-chip bandwidth arbitration (unit + integration),
//   - diagonal failure semantics, abort/cancel drain and reuse,
//   - illegal kernel shapes / tile / backend / lease refusals.
//
// Numeric expectations come from hand calculation or the independent FP64
// reference/oracle consuming the same FP32 input; device output is never used
// as its own expectation. No full solve / Ax=b closure is claimed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "controlplane/control_plane.hpp"
#include "controlplane/panel_exec_unit.hpp"
#include "control_test_artifact.hpp"
#include "dataplane/on_chip_arbiter.hpp"
#include "panel_lu/tile_kernels.hpp"
#include "fp64_reference.hpp"

namespace {

namespace cmd = hw::command;
namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace pl = hw::panel_lu;
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

cp::ControlPlaneConfig base_config(std::uint32_t groups, std::uint32_t per_group) {
    cp::ControlPlaneConfig config;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 2;
    config.trsm_unit_count = 1;
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
    config.panel.compute_group_count = groups;
    config.panel.pes_per_group = per_group;
    config.panel.pe_count = groups * per_group;
    return config;
}

std::uint32_t command_index(const testing::TestArtifact& artifact, cmd::Opcode opcode,
                            std::uint32_t node_id = cmd::NONE) {
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode != opcode) {
            continue;
        }
        if (node_id != cmd::NONE && artifact.commands[index].node_id != node_id) {
            continue;
        }
        return index;
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

// ---------------------------------------------------------------------------
// Hand-computable cases (kernel level)
// ---------------------------------------------------------------------------

void test_manual_cases() {
    // A: left solve L X = B, L unit lower [[1,0],[2,1]], B=[[3,1],[10,5]].
    {
        float l[4] = {1.0f, 0.0f, 2.0f, 1.0f};
        float b[4] = {3.0f, 1.0f, 10.0f, 5.0f};
        const auto outcome = pl::kernels::run_trsm_left(b, 2, 2, 2, l, 2);
        expect(outcome.status == cmd::StatusCode::Ok, "case A: left solve succeeds");
        expect(
            b[0] == 3.0f && b[1] == 1.0f && b[2] == 4.0f && b[3] == 3.0f,
            "case A: X = [[3,1],[4,3]]");
    }
    // B: right solve X U = B, U=[[2,3],[0,4]], B=[[4,14],[2,11]].
    {
        float u[4] = {2.0f, 3.0f, 0.0f, 4.0f};
        float b[4] = {4.0f, 14.0f, 2.0f, 11.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 2, 2, 2, u, 2);
        expect(outcome.status == cmd::StatusCode::Ok, "case B: right solve succeeds");
        expect(
            b[0] == 2.0f && b[1] == 2.0f && b[2] == 1.0f && b[3] == 2.0f,
            "case B: X = [[2,2],[1,2]]");
    }
    // C: 4x4 left solve, unit lower bidiagonal with subdiagonal 2,3,4,
    //    b = [1,4,9,16] -> x = [1,2,3,4].
    {
        float l[16] = {0};
        for (std::uint32_t i = 0; i < 4; ++i) {
            l[i * 4 + i] = 1.0f;
            if (i > 0) {
                l[i * 4 + (i - 1)] = static_cast<float>(i + 1);
            }
        }
        float b[16] = {1.0f, 0, 0, 0, 4.0f, 0, 0, 0, 9.0f, 0, 0, 0, 16.0f, 0, 0, 0};
        const auto outcome = pl::kernels::run_trsm_left(b, 4, 4, 1, l, 4);
        expect(outcome.status == cmd::StatusCode::Ok, "case C: 4x4 left solve succeeds");
        expect(
            b[0] == 1.0f && b[4] == 2.0f && b[8] == 3.0f && b[12] == 4.0f,
            "case C: x = [1,2,3,4]");
    }
    {
        float u[4] = {2.0f, 1.0f, 0.0f, 0.0f};
        float b[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const auto zero = pl::kernels::run_trsm_right(b, 2, 2, 2, u, 2);
        expect(
            zero.status == cmd::StatusCode::StaticPivotFailure &&
                zero.failed_column == 1 && zero.diagonal_failure,
            "kernel: zero U diagonal -> STATIC_PIVOT_FAILURE at its column");
        float u2[4] = {2.0f, 1.0f, 0.0f, 1.0e-3f};
        float b2[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const auto unstable =
            pl::kernels::run_trsm_right(b2, 2, 2, 2, u2, 2, 1.0e-2);
        expect(
            unstable.status == cmd::StatusCode::PivotUnstable &&
                unstable.failed_column == 1,
            "kernel: |U diag| <= threshold product -> PIVOT_UNSTABLE");
        float u3[4] = {2.0f, 1.0f, 0.0f, 1.0e-39f};
        float b3[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const auto overflow = pl::kernels::run_trsm_right(b3, 2, 2, 2, u3, 2);
        expect(
            overflow.status == cmd::StatusCode::NonFiniteValue &&
                overflow.failed_column == 1,
            "kernel: reciprocal overflow -> NON_FINITE_VALUE at its column");
    }
}

// ---------------------------------------------------------------------------
// Direct engine harness: crafted A11 data, diagonal failures, refusals
// ---------------------------------------------------------------------------

struct EngineHarness {
    dp::BufferManager buffers;
    pl::PanelEngine engine;
    dp::BufferHandle handle{};
    dp::BufferLease lease{};

    explicit EngineHarness(const pl::PanelLuConfig& config)
        : buffers(dp::BufferConfig{2, 64 * 1024, 4, 1}), engine(config) {}

    bool prepare(const std::vector<float>& front) { return prepare_at(0, front); }

    bool prepare_at(std::uint64_t front_offset, const std::vector<float>& front) {
        engine.set_buffers(&buffers);
        if (!engine.configure().ok) {
            return false;
        }
        const std::uint64_t total = front_offset + front.size() * 4;
        const auto allocate = buffers.allocate("trsm-test", total, 0);
        if (!allocate.ok) {
            return false;
        }
        handle = allocate.handle;
        const auto lease_result =
            buffers.begin_write(handle, "trsm-test", 0, total, 0);
        if (!lease_result.ok) {
            return false;
        }
        lease = lease_result.lease;
        std::vector<std::uint8_t> bytes(front.size() * 4);
        std::memcpy(bytes.data(), front.data(), bytes.size());
        return buffers.lease_write(lease.lease_id, front_offset, bytes, 1, nullptr) ==
               cmd::StatusCode::Ok;
    }

    std::vector<std::uint8_t> storage() const {
        return buffers.bytes(handle.slot);
    }

    pl::PanelEngineResponse run(
        const pl::PanelEngineRequest& request,
        std::uint64_t max_cycles = 200000) {
        const std::uint64_t key = pl::PanelEngine::adapter_key(2, 0);
        const pl::ConfigCheck started = engine.begin_command(request, key, 2);
        if (!started.ok) {
            pl::PanelEngineResponse response;
            response.status = started.status;
            response.reason = started.reason;
            return response;
        }
        for (std::uint64_t cycle = 2; cycle < max_cycles; ++cycle) {
            engine.tick(cycle);
            pl::PanelEngineResponse response;
            if (engine.poll_response(key, response)) {
                return response;
            }
        }
        pl::PanelEngineResponse timeout;
        timeout.status = cmd::StatusCode::Timeout;
        return timeout;
    }
};

pl::PanelEngineRequest trsm_request(
    pl::ComputeKind kind,
    std::uint32_t total,
    std::uint32_t pivot,
    const dp::BufferHandle& handle,
    const dp::BufferLease& lease,
    std::uint64_t front_offset = 0) {
    pl::PanelEngineRequest request;
    request.txn_id = 1;
    request.command_id = 1;
    request.node_id = 0;
    request.total_dim = total;
    request.pivot_dim = pivot;
    request.tile_size = 16;
    request.compute_kind = kind;
    request.front = handle;
    request.front_lease = lease;
    request.front_offset = front_offset;
    const std::uint32_t update = total - pivot;
    if (kind == pl::ComputeKind::TrsmLeft) {
        request.kernel_m = pivot;
        request.kernel_n = update;
        request.kernel_k = pivot;
    } else {
        request.kernel_m = update;
        request.kernel_n = pivot;
        request.kernel_k = pivot;
    }
    return request;
}

void test_engine_diagonal_failures_and_refusals() {
    pl::PanelLuConfig config;
    config.compute_group_count = 1;
    config.pes_per_group = 1;
    config.pe_count = 1;

    const std::uint32_t n = 32;
    const std::uint32_t p = 16;
    std::vector<float> front(n * n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            front[i * n + j] = 0.125f * static_cast<float>(i + j + 1);
        }
    }
    for (std::uint32_t i = 0; i < p; ++i) {
        for (std::uint32_t j = 0; j < p; ++j) {
            if (i == j) {
                front[i * n + j] = 2.0f;
            } else if (i > j) {
                front[i * n + j] = 0.25f * static_cast<float>(i - j);
            } else {
                front[i * n + j] = 0.5f * static_cast<float>(j - i);
            }
        }
    }
    {
        EngineHarness harness(config);
        expect(harness.prepare(front), "engine: crafted front staged");
        auto request = trsm_request(
            pl::ComputeKind::TrsmRight, n, p, harness.handle, harness.lease);
        const auto response = harness.run(request);
        expect(
            response.status == cmd::StatusCode::Ok,
            "engine: healthy crafted A11 right solve succeeds");
        expect(response.group_id == 0, "engine: command bound to group 0");
    }
    {
        std::vector<float> broken = front;
        broken[1 * n + 1] = 0.0f;
        EngineHarness harness(config);
        expect(harness.prepare(broken), "engine: zero-diagonal front staged");
        auto request = trsm_request(
            pl::ComputeKind::TrsmRight, n, p, harness.handle, harness.lease);
        const auto response = harness.run(request);
        expect(
            response.status == cmd::StatusCode::StaticPivotFailure,
            "engine: zero U diagonal -> STATIC_PIVOT_FAILURE");
        expect(
            response.failed_column == 1,
            "engine: failure mapped to the global pivot column");
    }
    {
        std::vector<float> broken = front;
        broken[1 * n + 1] = 1.0e-3f;
        EngineHarness harness(config);
        expect(harness.prepare(broken), "engine: unstable front staged");
        auto request = trsm_request(
            pl::ComputeKind::TrsmRight, n, p, harness.handle, harness.lease);
        request.threshold_product = 1.0e-2;
        const auto response = harness.run(request);
        expect(
            response.status == cmd::StatusCode::PivotUnstable &&
                response.failed_column == 1,
            "engine: |U diag| <= latched threshold -> PIVOT_UNSTABLE");
    }
    {
        std::vector<float> broken = front;
        broken[0 * n + 0] = 1.0e-39f;
        EngineHarness harness(config);
        expect(harness.prepare(broken), "engine: subnormal diagonal staged");
        auto request = trsm_request(
            pl::ComputeKind::TrsmRight, n, p, harness.handle, harness.lease);
        const auto response = harness.run(request);
        expect(
            response.status == cmd::StatusCode::NonFiniteValue &&
                response.failed_column == 0,
            "engine: reciprocal overflow -> NON_FINITE_VALUE at its column");
    }
    {
        EngineHarness harness(config);
        expect(harness.prepare(front), "engine: refusal front staged");
        auto shape = trsm_request(
            pl::ComputeKind::TrsmLeft, n, p, harness.handle, harness.lease);
        shape.kernel_n = 7;
        expect(
            harness.run(shape).status == cmd::StatusCode::BadDescriptor,
            "engine: wrong TRSM kernel shape -> BAD_DESCRIPTOR");
        auto tile = trsm_request(
            pl::ComputeKind::TrsmLeft, n, p, harness.handle, harness.lease);
        tile.kernel_tile_size = 8;
        expect(
            harness.run(tile).status == cmd::StatusCode::BadDescriptor,
            "engine: kernel tile_size != 16 -> BAD_DESCRIPTOR");
        auto backend = trsm_request(
            pl::ComputeKind::TrsmLeft, n, p, harness.handle, harness.lease);
        backend.kernel_backend = 0x02;
        expect(
            harness.run(backend).status == cmd::StatusCode::BadDescriptor,
            "engine: non-FP32 kernel backend -> BAD_DESCRIPTOR");
        auto no_lease = trsm_request(
            pl::ComputeKind::TrsmLeft, n, p, harness.handle, dp::BufferLease{});
        expect(
            harness.run(no_lease).status == cmd::StatusCode::BadDescriptor,
            "engine: missing front lease -> BAD_DESCRIPTOR");
    }
}

// ---------------------------------------------------------------------------
// n=42 / p=23 / u=19 real data path with the FP64 oracle
// ---------------------------------------------------------------------------

struct TrsmOracle {
    ref::PanelLuResult factorization;
    std::vector<double> u12;  // p x u
    std::vector<double> l21;  // u x p
};

TrsmOracle build_oracle(
    const std::vector<float>& matrix,
    std::uint32_t n,
    std::uint32_t p) {
    const std::uint32_t u = n - p;
    TrsmOracle oracle;
    ref::DenseMatrix a11(p, p);
    for (std::uint32_t i = 0; i < p; ++i) {
        for (std::uint32_t j = 0; j < p; ++j) {
            a11.at(i, j) = static_cast<double>(matrix[i * n + j]);
        }
    }
    std::vector<std::size_t> variables(p);
    for (std::uint32_t i = 0; i < p; ++i) {
        variables[i] = i;
    }
    oracle.factorization =
        ref::factor_front(variables, a11, p, ref::PivotPolicy::Static, 0.0)
            .factorization;
    oracle.u12.assign(static_cast<std::size_t>(p) * u, 0.0);
    for (std::uint32_t j = 0; j < u; ++j) {
        for (std::uint32_t i = 0; i < p; ++i) {
            double acc = static_cast<double>(matrix[i * n + p + j]);
            for (std::uint32_t t = 0; t < i; ++t) {
                acc -= oracle.factorization.l.at(i, t) * oracle.u12[t * u + j];
            }
            oracle.u12[i * u + j] = acc;
        }
    }
    oracle.l21.assign(static_cast<std::size_t>(u) * p, 0.0);
    for (std::uint32_t i = 0; i < u; ++i) {
        for (std::uint32_t j = 0; j < p; ++j) {
            double acc = static_cast<double>(matrix[(p + i) * n + j]);
            for (std::uint32_t t = 0; t < j; ++t) {
                acc -= oracle.l21[i * p + t] * oracle.factorization.u.at(t, j);
            }
            oracle.l21[i * p + j] = acc / oracle.factorization.u.at(j, j);
        }
    }
    return oracle;
}

struct IntegrationOutcome {
    bool finished{false};
    std::map<std::uint32_t, cmd::StatusCode> status;
    std::vector<std::uint8_t> front_snapshot;  // latest resident front state
    std::vector<std::uint8_t> panel_snapshot;  // front right after PANEL completion
    std::uint32_t panel_command{cmd::NONE};
    std::uint32_t left_command{cmd::NONE};
    std::uint32_t right_command{cmd::NONE};
    std::uint32_t factor_command{cmd::NONE};
    std::uint32_t gemm_command{cmd::NONE};
    std::uint32_t solve_command{cmd::NONE};
    std::uint64_t panel_bytes_written{0};
    std::uint64_t factor_bytes_written{0};
    std::vector<float> l_factors;  // n x p
    std::vector<float> u_factors;  // p x n
};

IntegrationOutcome run_integration(
    const std::vector<float>& matrix,
    std::uint32_t n,
    std::uint32_t p,
    const cp::ControlPlaneConfig& config) {
    IntegrationOutcome outcome;
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = p;
    node.front_values = matrix;
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = n;
    testing::TestArtifact artifact = testing::build_test_artifact({node}, options);
    outcome.panel_command = command_index(artifact, cmd::Opcode::PanelLu);
    outcome.left_command = command_index(artifact, cmd::Opcode::TrsmLeft);
    outcome.right_command = command_index(artifact, cmd::Opcode::TrsmRight);
    outcome.factor_command = command_index(artifact, cmd::Opcode::StoreFactor);
    outcome.gemm_command = command_index(artifact, cmd::Opcode::GemmSchur);
    outcome.solve_command = command_index(artifact, cmd::Opcode::SolveForward);

    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    if (!device.start(&error)) {
        return outcome;
    }
    while (!device.finished() && device.cycle() < 8000000) {
        device.tick();
        const dp::BufferHandle front = device.front_slot(0);
        if (front.slot != cmd::NONE) {
            outcome.front_snapshot = device.dataplane().buffers().bytes(front.slot);
            if (device.completion_status().count(outcome.panel_command) != 0 &&
                outcome.panel_snapshot.empty()) {
                outcome.panel_snapshot = outcome.front_snapshot;
            }
        }
    }
    outcome.finished = device.finished();
    outcome.status = device.completion_status();
    outcome.l_factors = region_f32(device, "node.0.factor_l",
                                   static_cast<std::size_t>(n) * p);
    outcome.u_factors = region_f32(device, "node.0.factor_u",
                                   static_cast<std::size_t>(p) * n);
    for (const auto& stats : device.command_stats()) {
        if (stats.command_id == outcome.panel_command) {
            outcome.panel_bytes_written = stats.bytes_written;
        }
        if (stats.command_id == outcome.factor_command) {
            outcome.factor_bytes_written = stats.bytes_written;
        }
    }
    return outcome;
}

void test_integration_n42_full_factor_and_oracle() {
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    const std::uint32_t u = n - p;
    const std::vector<float> matrix = make_matrix(n, 7);
    const TrsmOracle oracle = build_oracle(matrix, n, p);

    for (const auto mode : {pl::PanelNumericMode::Fp32Device,
                            pl::PanelNumericMode::Fp64Diagnostic}) {
        cp::ControlPlaneConfig config = base_config(2, 2);
        config.panel.numeric_mode = mode;
        const std::string label =
            mode == pl::PanelNumericMode::Fp32Device ? "n42 fp32" : "n42 fp64diag";
        const IntegrationOutcome outcome = run_integration(matrix, n, p, config);
        if (mode == pl::PanelNumericMode::Fp32Device && !g_output_dir.empty()) {
            // Device execution report (architecture model) for the canonical
            // n=42 run; external oracle checks are reported separately by the
            // test assertions above.
            cp::CommandDevice report_device(
                [&] {
                    testing::TestNode report_node;
                    report_node.node_id = 0;
                    report_node.parent_id = -1;
                    report_node.total_dim = n;
                    report_node.pivot_dim = p;
                    report_node.front_values = matrix;
                    testing::TestArtifactOptions report_options;
                    report_options.threshold_bits = f32_bits(0.0f);
                    report_options.tile_size = 16;
                    report_options.matrix_dim = n;
                    return testing::build_test_artifact({report_node}, report_options);
                }().artifact,
                config);
            std::string report_error;
            if (report_device.start(&report_error)) {
                while (!report_device.finished() && report_device.cycle() < 4000000) {
                    report_device.tick();
                }
                report_device.write_reports(g_output_dir + "/n42_fp32");
                std::ifstream summary(g_output_dir + "/n42_fp32/summary.json");
                std::ifstream tasks(g_output_dir + "/n42_fp32/panel_tasks.csv");
                expect(summary.good(), label + ": summary.json written");
                expect(tasks.good(), label + ": panel_tasks.csv written");
            }
        }
        expect(outcome.finished, label + ": batch terminates");
        expect(
            outcome.status.at(outcome.panel_command) == cmd::StatusCode::Ok,
            label + ": PANEL_LU succeeds");
        expect(
            outcome.status.at(outcome.left_command) == cmd::StatusCode::Ok,
            label + ": TRSM_LEFT succeeds");
        expect(
            outcome.status.at(outcome.right_command) == cmd::StatusCode::Ok,
            label + ": TRSM_RIGHT succeeds");
        expect(
            outcome.status.at(outcome.factor_command) == cmd::StatusCode::Ok,
            label + ": STORE_FACTOR succeeds");
        expect(
            outcome.status.at(outcome.gemm_command) == cmd::StatusCode::Ok,
            label + ": top-level GEMM_SCHUR now executes (T10)");
        expect(
            outcome.status.at(outcome.solve_command) != cmd::StatusCode::Ok,
            label + ": SOLVE cannot report success (no full-solve claim)");
        expect(
            outcome.factor_bytes_written ==
                static_cast<std::uint64_t>(8) * n * p,
            label + ": full L/U save writes exactly 8*n*p bytes");

        // Full factor blocks vs the independent FP64 oracle.
        double max_l = 0.0;
        double max_u = 0.0;
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t j = 0; j < p; ++j) {
                const double expected = i < p ? oracle.factorization.l.at(i, j)
                                              : oracle.l21[(i - p) * p + j];
                max_l = std::max(
                    max_l,
                    std::fabs(expected - outcome.l_factors[
                                             static_cast<std::size_t>(i) * p + j]));
            }
        }
        for (std::uint32_t i = 0; i < p; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                const double expected =
                    j < p ? oracle.factorization.u.at(i, j)
                          : oracle.u12[i * u + (j - p)];
                max_u = std::max(
                    max_u,
                    std::fabs(expected - outcome.u_factors[
                                             static_cast<std::size_t>(i) * n + j]));
            }
        }
        expect(max_l < 5.0e-3, label + ": L matches the FP64 oracle");
        expect(max_u < 5.0e-3, label + ": U matches the FP64 oracle");

        // Masks: L11 unit lower, U11 upper incl diag.
        bool l_mask = true;
        bool u_mask = true;
        for (std::uint32_t i = 0; i < p; ++i) {
            for (std::uint32_t j = 0; j < p; ++j) {
                const float l_value =
                    outcome.l_factors[static_cast<std::size_t>(i) * p + j];
                if (j > i && l_value != 0.0f) {
                    l_mask = false;
                }
                if (j == i && l_value != 1.0f) {
                    l_mask = false;
                }
                const float u_value =
                    outcome.u_factors[static_cast<std::size_t>(i) * n + j];
                if (j < i && u_value != 0.0f) {
                    u_mask = false;
                }
            }
        }
        expect(l_mask, label + ": L11 is unit lower with zero upper triangle");
        expect(u_mask, label + ": U11 is upper triangular with zero lower part");

        // Equations with device factors: L11*U12 = A12, L21*U11 = A21.
        double a12_scale = 0.0;
        double a21_scale = 0.0;
        for (std::uint32_t i = 0; i < p; ++i) {
            for (std::uint32_t j = 0; j < u; ++j) {
                a12_scale = std::max(
                    a12_scale,
                    std::fabs(static_cast<double>(matrix[i * n + p + j])));
            }
        }
        for (std::uint32_t i = 0; i < u; ++i) {
            for (std::uint32_t j = 0; j < p; ++j) {
                a21_scale = std::max(
                    a21_scale,
                    std::fabs(static_cast<double>(matrix[(p + i) * n + j])));
            }
        }
        double left_residual = 0.0;
        for (std::uint32_t i = 0; i < p; ++i) {
            for (std::uint32_t j = 0; j < u; ++j) {
                double acc = 0.0;
                for (std::uint32_t t = 0; t <= i; ++t) {
                    acc += static_cast<double>(
                               outcome.l_factors[static_cast<std::size_t>(i) * p + t]) *
                           static_cast<double>(outcome.u_factors[
                               static_cast<std::size_t>(t) * n + (p + j)]);
                }
                left_residual = std::max(
                    left_residual,
                    std::fabs(acc - static_cast<double>(matrix[i * n + p + j])));
            }
        }
        double right_residual = 0.0;
        for (std::uint32_t i = 0; i < u; ++i) {
            for (std::uint32_t j = 0; j < p; ++j) {
                double acc = 0.0;
                for (std::uint32_t t = 0; t <= j; ++t) {
                    acc += static_cast<double>(outcome.l_factors[
                               static_cast<std::size_t>(p + i) * p + t]) *
                           static_cast<double>(outcome.u_factors[
                               static_cast<std::size_t>(t) * n + j]);
                }
                right_residual = std::max(
                    right_residual,
                    std::fabs(acc - static_cast<double>(matrix[(p + i) * n + j])));
            }
        }
        expect(
            left_residual / std::max(1.0, a12_scale) < 5.0e-3,
            label + ": normalized L11*U12 - A12 residual small");
        expect(
            right_residual / std::max(1.0, a21_scale) < 5.0e-3,
            label + ": normalized L21*U11 - A21 residual small");

        // Boundary invariance on the resident front: the panel/TRSM commands
        // touched A11 (in place) and A12/A21; the T10 GEMM_SCHUR command then
        // updated A22 in place with the device's own committed L21/U12. The
        // guard beyond n*n stays untouched throughout.
        expect(
            outcome.panel_snapshot.size() == outcome.front_snapshot.size() &&
                !outcome.front_snapshot.empty(),
            label + ": front snapshots captured");
        if (outcome.panel_snapshot.size() == outcome.front_snapshot.size()) {
            bool a11_unchanged = true;
            bool a22_changed = false;
            bool a12_changed = false;
            bool a21_changed = false;
            bool guard_ok = true;
            const std::size_t front_bytes =
                static_cast<std::size_t>(n) * n * 4;
            for (std::size_t index = 0; index < outcome.front_snapshot.size();
                 index += 4) {
                const std::size_t element = index / 4;
                const std::uint32_t i =
                    static_cast<std::uint32_t>(element / n);
                const std::uint32_t j =
                    static_cast<std::uint32_t>(element % n);
                float before = 0.0f;
                float after = 0.0f;
                std::memcpy(&before, outcome.panel_snapshot.data() + index, 4);
                std::memcpy(&after, outcome.front_snapshot.data() + index, 4);
                if (i < p && j < p && after != before) {
                    a11_unchanged = false;
                }
                if (i < n && j < n && i >= p && j >= p) {
                    if (after != matrix[element]) {
                        a22_changed = true;
                    }
                }
                if (i < p && j >= p && after != before) {
                    a12_changed = true;
                }
                if (i >= p && j < p && after != before) {
                    a21_changed = true;
                }
            }
            for (std::size_t index = front_bytes;
                 index < outcome.front_snapshot.size(); ++index) {
                if (outcome.front_snapshot[index] != 0) {
                    guard_ok = false;
                }
            }
            expect(a11_unchanged, label + ": A11 unchanged by the compute commands");
            expect(a22_changed, label + ": A22 changed by GEMM_SCHUR");
            expect(a12_changed && a21_changed, label + ": A12/A21 actually updated");
            expect(guard_ok, label + ": guard/padding bytes untouched");

            // T10 numeric check: A22 := A22 - L21*U12 using the DEVICE's own
            // committed factors, evaluated independently in FP64 (not a mirror
            // of the engine's tile arithmetic).
            double max_a22 = 0.0;
            double a22_scale = 1.0;
            for (std::uint32_t i = 0; i < u; ++i) {
                for (std::uint32_t j = 0; j < u; ++j) {
                    double acc = static_cast<double>(matrix[(p + i) * n + (p + j)]);
                    for (std::uint32_t t = 0; t < p; ++t) {
                        acc -= static_cast<double>(
                                   outcome.l_factors[
                                       static_cast<std::size_t>(p + i) * p + t]) *
                               static_cast<double>(
                                   outcome.u_factors[
                                       static_cast<std::size_t>(t) * n + (p + j)]);
                    }
                    float actual = 0.0f;
                    std::memcpy(
                        &actual,
                        outcome.front_snapshot.data() +
                            static_cast<std::size_t>(p + i) * n * 4 +
                            static_cast<std::size_t>(p + j) * 4,
                        4);
                    max_a22 = std::max(
                        max_a22, std::fabs(static_cast<double>(actual) - acc));
                    a22_scale = std::max(a22_scale, std::fabs(acc));
                }
            }
            expect(
                max_a22 / a22_scale < 5.0e-3,
                label + ": A22 matches the FP64 Schur evaluation of the device factors");
        }
    }
}


// Nonzero front_offset must be applied exactly once by the access layer.
void test_nonzero_front_offset() {
    const std::uint32_t n = 4;
    const std::uint32_t p = 2;
    // Left case: A11 unit-lower [[1,0],[2,1]], A12 = [[3,1],[10,5]].
    std::vector<float> left_front(n * n, 0.0f);
    left_front[0 * n + 0] = 1.0f;
    left_front[1 * n + 0] = 2.0f;
    left_front[0 * n + 1] = 0.0f;
    left_front[1 * n + 1] = 1.0f;
    left_front[0 * n + 2] = 3.0f;
    left_front[0 * n + 3] = 1.0f;
    left_front[1 * n + 2] = 10.0f;
    left_front[1 * n + 3] = 5.0f;
    // Right case: A11 = [[2,3],[0,4]], A21 = [[4,14],[2,11]].
    std::vector<float> right_front(n * n, 0.0f);
    right_front[0 * n + 0] = 2.0f;
    right_front[0 * n + 1] = 3.0f;
    right_front[1 * n + 0] = 0.0f;
    right_front[1 * n + 1] = 4.0f;
    right_front[2 * n + 0] = 4.0f;
    right_front[2 * n + 1] = 14.0f;
    right_front[3 * n + 0] = 2.0f;
    right_front[3 * n + 1] = 11.0f;

    for (const std::uint64_t offset : {std::uint64_t{0}, std::uint64_t{4},
                                       std::uint64_t{8}}) {
        const std::string label = "offset " + std::to_string(offset);
        {
            pl::PanelLuConfig config;
            config.compute_group_count = 1;
            config.pes_per_group = 1;
            config.pe_count = 1;
            EngineHarness harness(config);
            expect(harness.prepare_at(offset, left_front), label + ": left staged");
            auto request = trsm_request(
                pl::ComputeKind::TrsmLeft, n, p, harness.handle, harness.lease, offset);
            const auto response = harness.run(request);
            expect(response.status == cmd::StatusCode::Ok, label + ": left Ok");
            const auto bytes = harness.storage();
            float value = 0.0f;
            auto read = [&](std::size_t element) {
                std::memcpy(&value, bytes.data() + offset + element * 4, 4);
                return value;
            };
            expect(
                read(0 * n + 2) == 3.0f && read(0 * n + 3) == 1.0f &&
                    read(1 * n + 2) == 4.0f && read(1 * n + 3) == 3.0f,
                label + ": left X = [[3,1],[4,3]]");
            expect(
                read(0 * n + 0) == 1.0f && read(1 * n + 0) == 2.0f,
                label + ": left A11 unchanged");
            bool guard_ok = true;
            for (std::size_t index = 0; index < offset; ++index) {
                guard_ok = guard_ok && bytes[index] == 0;
            }
            for (std::size_t index = offset + n * n * 4; index < bytes.size(); ++index) {
                guard_ok = guard_ok && bytes[index] == 0;
            }
            expect(guard_ok, label + ": guard bytes unchanged");
        }
        {
            pl::PanelLuConfig config;
            config.compute_group_count = 1;
            config.pes_per_group = 1;
            config.pe_count = 1;
            EngineHarness harness(config);
            expect(harness.prepare_at(offset, right_front), label + ": right staged");
            auto request = trsm_request(
                pl::ComputeKind::TrsmRight, n, p, harness.handle, harness.lease, offset);
            const auto response = harness.run(request);
            expect(response.status == cmd::StatusCode::Ok, label + ": right Ok");
            const auto bytes = harness.storage();
            float value = 0.0f;
            auto read = [&](std::size_t element) {
                std::memcpy(&value, bytes.data() + offset + element * 4, 4);
                return value;
            };
            expect(
                read(2 * n + 0) == 2.0f && read(2 * n + 1) == 2.0f &&
                    read(3 * n + 0) == 1.0f && read(3 * n + 1) == 2.0f,
                label + ": right X = [[2,2],[1,2]]");
            expect(
                read(0 * n + 0) == 2.0f && read(0 * n + 1) == 3.0f &&
                    read(1 * n + 0) == 0.0f && read(1 * n + 1) == 4.0f,
                label + ": right A11 unchanged");
        }
    }
}

// FP64 diagnostic: rounding the binary64 result back to FP32 storage must be
// finite-checked; the overflowing tile must not commit.
void test_fp64_diagnostic_rounding_overflow() {
    const std::uint32_t n = 32;
    const std::uint32_t p = 16;
    const float max_float = std::numeric_limits<float>::max();
    std::vector<float> front(n * n, 0.0f);
    for (std::uint32_t i = 0; i < p; ++i) {
        for (std::uint32_t j = 0; j < p; ++j) {
            front[i * n + j] = i == j ? 2.0f : 0.25f * static_cast<float>(i + j + 1);
        }
    }
    front[0 * n + 0] = 0.5f;           // reciprocal 2x
    front[p * n + 0] = max_float;      // A21(0,0) = FLT_MAX

    auto run_case = [&](pl::PanelNumericMode mode, const std::string& label) {
        pl::PanelLuConfig config;
        config.compute_group_count = 1;
        config.pes_per_group = 1;
        config.pe_count = 1;
        config.numeric_mode = mode;
        EngineHarness harness(config);
        expect(harness.prepare(front), label + ": front staged");
        auto request = trsm_request(
            pl::ComputeKind::TrsmRight, n, p, harness.handle, harness.lease);
        const auto response = harness.run(request);
        expect(
            response.status == cmd::StatusCode::NonFiniteValue,
            label + ": overflow -> NON_FINITE_VALUE");
        expect(response.failed_column == 0, label + ": failing column reported");
        const auto bytes = harness.storage();
        float committed = 0.0f;
        std::memcpy(&committed, bytes.data() + p * n * 4, 4);
        expect(
            committed == max_float,
            label + ": overflowing tile is not committed");
        return response;
    };
    const auto fp32 = run_case(pl::PanelNumericMode::Fp32Device, "fp32");
    const auto fp64 = run_case(pl::PanelNumericMode::Fp64Diagnostic, "fp64diag");
    expect(
        fp64.reason.find("fp64 diagnostic") != std::string::npos,
        "fp64diag: rounding overflow reported by the diagnostic path");
    (void)fp32;
}

// KERNEL_DESC operands must reference the resident front of the node being
// dispatched; a cross-node reference is refused before any execution.
void test_kernel_operand_reference_validation() {
    const std::uint32_t n = 32;
    const std::uint32_t p = 16;
    testing::TestNode node0;
    node0.node_id = 0;
    node0.parent_id = -1;
    node0.total_dim = n;
    node0.pivot_dim = p;
    node0.front_values = make_matrix(n, 71);
    testing::TestNode node1;
    node1.node_id = 1;
    node1.parent_id = -1;
    node1.total_dim = n;
    node1.pivot_dim = p;
    node1.front_values = make_matrix(n, 73);
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = n;
    testing::TestArtifact artifact =
        testing::build_test_artifact({node0, node1}, options);

    // Patch node 0's TRSM_LEFT KERNEL_DESC to point its three operands at
    // node 1's front region (a cross-node reference the loader accepts).
    const std::uint32_t left0 =
        command_index(artifact, cmd::Opcode::TrsmLeft, 0);
    expect(left0 != cmd::NONE, "kernel refs: left command exists");
    const std::uint32_t desc_id = artifact.commands[left0].descriptor_id;
    std::uint32_t node1_front_region = cmd::NONE;
    for (std::uint32_t index = 0; index < artifact.descriptors.size(); ++index) {
        const auto& descriptor = artifact.descriptors[index];
        if (descriptor.descriptor_type != cmd::DescriptorType::Region) {
            continue;
        }
        const std::uint64_t base =
            static_cast<std::uint64_t>(descriptor.body_words[0]) |
            (static_cast<std::uint64_t>(descriptor.body_words[1]) << 32);
        if (base == artifact.region_addr("node.1.front")) {
            node1_front_region = index;
        }
    }
    expect(node1_front_region != cmd::NONE, "kernel refs: node 1 front region found");
    cmd::Descriptor patched = artifact.descriptors[desc_id];
    patched.body_words[0] = node1_front_region;
    patched.body_words[1] = node1_front_region;
    patched.body_words[2] = node1_front_region;
    const auto record = cmd::DescriptorCodec::encode(patched);
    const std::uint64_t table = artifact.region_addr("descriptor_table");
    std::memcpy(
        artifact.artifact.image.data() + table +
            static_cast<std::uint64_t>(desc_id) * cmd::DESCRIPTOR_RECORD_BYTES,
        record.data(), record.size());

    // Snapshot node 1's front region so we can prove nothing touched it.
    const std::uint64_t node1_front_offset = artifact.region_addr("node.1.front");
    const std::uint64_t node1_front_size = artifact.region_size("node.1.front");
    const std::vector<std::uint8_t> node1_before(
        artifact.artifact.image.begin() +
            static_cast<std::ptrdiff_t>(node1_front_offset),
        artifact.artifact.image.begin() +
            static_cast<std::ptrdiff_t>(node1_front_offset + node1_front_size));

    cp::ControlPlaneConfig config = base_config(2, 2);
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "kernel refs: device starts");
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    expect(device.finished(), "kernel refs: batch terminates");
    expect(
        device.completion_status().at(left0) == cmd::StatusCode::BadDescriptor,
        "kernel refs: cross-node operands -> BAD_DESCRIPTOR");
    const auto& image = device.dataplane().memory().image();
    const std::vector<std::uint8_t> node1_after(
        image.begin() + static_cast<std::ptrdiff_t>(node1_front_offset),
        image.begin() + static_cast<std::ptrdiff_t>(node1_front_offset + node1_front_size));
    expect(node1_before == node1_after, "kernel refs: node 1 front untouched");
    bool left_executed = false;
    for (const auto& record_row : device.panel_engine()->task_records()) {
        if (record_row.task.command_id == left0) {
            left_executed = true;
        }
    }
    expect(!left_executed, "kernel refs: refused command ran no tile task");
    const std::uint32_t factor0 =
        command_index(artifact, cmd::Opcode::StoreFactor, 0);
    expect(
        device.completion_status().at(factor0) != cmd::StatusCode::Ok,
        "kernel refs: dependent STORE_FACTOR does not publish success");
}

// ---------------------------------------------------------------------------
// n=p: no TRSM commands, factor chain still complete
// ---------------------------------------------------------------------------

void test_np_factor_chain() {
    const std::uint32_t n = 16;
    const std::vector<float> matrix = make_matrix(n, 11);
    cp::ControlPlaneConfig config = base_config(1, 1);
    const IntegrationOutcome outcome = run_integration(matrix, n, n, config);
    expect(outcome.finished, "n=p: batch terminates");
    expect(
        outcome.status.at(outcome.panel_command) == cmd::StatusCode::Ok &&
            outcome.status.at(outcome.factor_command) == cmd::StatusCode::Ok,
        "n=p: PANEL + STORE_FACTOR succeed");
    expect(outcome.left_command == cmd::NONE, "n=p: no TRSM_LEFT command");
    expect(
        outcome.factor_bytes_written == 8u * n * n,
        "n=p: factor save writes 8*n*p bytes");
    const TrsmOracle oracle = build_oracle(matrix, n, n);
    double max_diff = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            max_diff = std::max(
                max_diff,
                std::fabs(oracle.factorization.l.at(i, j) -
                          outcome.l_factors[static_cast<std::size_t>(i) * n + j]));
            max_diff = std::max(
                max_diff,
                std::fabs(oracle.factorization.u.at(i, j) -
                          outcome.u_factors[static_cast<std::size_t>(i) * n + j]));
        }
    }
    expect(max_diff < 5.0e-3, "n=p: factors match the FP64 oracle");
}

// ---------------------------------------------------------------------------
// Groups, intra-command parallelism and reuse
// ---------------------------------------------------------------------------

struct GroupTrace {
    std::map<std::uint32_t, std::uint32_t> command_to_group;
    std::map<std::uint32_t, std::set<std::uint32_t>> command_to_pes;
    std::map<std::uint32_t, std::string> command_kinds;
    bool two_groups_concurrent{false};
    bool cross_kind_group_reuse{false};
    bool intra_command_parallel{false};
    bool pe_outside_group{false};
    bool pe_overlap_same_pe{false};
    bool group_conflict{false};
    std::size_t max_active_groups{0};
    std::uint32_t pe_count{0};
    std::uint32_t group_count{0};
};

GroupTrace run_group_trace(
    const std::vector<std::pair<std::vector<float>, std::pair<std::uint32_t, std::uint32_t>>>&
        nodes,
    const cp::ControlPlaneConfig& config) {
    GroupTrace trace;
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = 64;
    std::vector<testing::TestNode> test_nodes;
    std::uint32_t node_id = 0;
    for (const auto& entry : nodes) {
        testing::TestNode node;
        node.node_id = node_id++;
        node.parent_id = -1;
        node.total_dim = entry.second.first;
        node.pivot_dim = entry.second.second;
        node.front_values = entry.first;
        test_nodes.push_back(node);
    }
    testing::TestArtifact artifact =
        testing::build_test_artifact(test_nodes, options);
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    if (!device.start(&error)) {
        return trace;
    }
    std::map<std::uint32_t, std::uint32_t> seen_commands;
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
        auto* engine = device.panel_engine();
        // Concurrent group separation while two commands are active.
        std::vector<std::uint32_t> active_groups;
        for (const auto& command : artifact.commands) {
            const std::uint32_t group =
                engine->group_of_command(command.command_id);
            if (group != cmd::NONE) {
                active_groups.push_back(group);
                seen_commands[command.command_id] = group;
            }
        }
        trace.max_active_groups =
            std::max(trace.max_active_groups, active_groups.size());
        std::sort(active_groups.begin(), active_groups.end());
        if (std::adjacent_find(active_groups.begin(), active_groups.end()) !=
            active_groups.end()) {
            trace.group_conflict = true;  // two commands shared one group
        } else if (active_groups.size() >= 2) {
            trace.two_groups_concurrent = true;
        }
        for (const auto& record : engine->task_records()) {
            const std::uint32_t group = record.group_id;
            const std::uint32_t first = group * config.panel.pes_per_group;
            if (record.pe_id < first ||
                record.pe_id >= first + config.panel.pes_per_group) {
                trace.pe_outside_group = true;
            }
            trace.command_to_group[record.task.command_id] = group;
            trace.command_to_pes[record.task.command_id].insert(record.pe_id);
        }
        for (const auto& record : engine->task_records()) {
            for (const auto& other : engine->task_records()) {
                if (&record == &other || record.task.command_id == other.task.command_id ||
                    record.pe_id != other.pe_id) {
                    continue;
                }
                if (record.exec_start_cycle < other.exec_done_cycle &&
                    other.exec_start_cycle < record.exec_done_cycle) {
                    trace.pe_overlap_same_pe = true;
                }
            }
        }
    }
    auto* engine = device.panel_engine();
    trace.pe_count = static_cast<std::uint32_t>(engine->pes().size());
    trace.group_count = engine->group_count();
    // Recorded commands -> kind names and intra-command parallelism.
    for (const auto& entry : trace.command_to_pes) {
        const auto& command = artifact.commands[entry.first];
        trace.command_kinds[entry.first] = unit_kind_name(
            command.opcode == cmd::Opcode::PanelLu
                ? cp::UnitKind::Panel
                : (command.opcode == cmd::Opcode::TrsmLeft ||
                           command.opcode == cmd::Opcode::TrsmRight
                       ? cp::UnitKind::Trsm
                       : cp::UnitKind::Factor));
        if (entry.second.size() >= 2) {
            trace.intra_command_parallel = true;
        }
    }
    // Cross-kind reuse: one group serving a panel command and a trsm command.
    for (const auto& left : trace.command_to_group) {
        for (const auto& right : trace.command_to_group) {
            if (left.second != right.second) {
                continue;
            }
            if (trace.command_kinds[left.first] != trace.command_kinds[right.first] &&
                (trace.command_kinds[left.first] == "panel" ||
                 trace.command_kinds[right.first] == "panel")) {
                trace.cross_kind_group_reuse = true;
            }
        }
    }
    return trace;
}

void test_groups_and_reuse() {
    const std::uint32_t n = 48;
    const std::uint32_t p = 32;
    const std::vector<float> first = make_matrix(n, 31);
    const std::vector<float> second = make_matrix(n, 47);
    cp::ControlPlaneConfig config = base_config(2, 2);
    const GroupTrace trace = run_group_trace(
        {{first, {n, p}}, {second, {n, p}}}, config);
    expect(trace.pe_count == 4, "groups: exactly 4 physical PEs");
    expect(trace.group_count == 2, "groups: 2 compute groups");
    expect(!trace.group_conflict, "groups: two commands never share one group");
    expect(!trace.pe_outside_group, "groups: tasks only run on their group's PEs");
    expect(!trace.pe_overlap_same_pe, "groups: one array task per PE at a time");
    expect(trace.intra_command_parallel, "groups: an accepted command uses multiple PEs");
    expect(trace.cross_kind_group_reuse, "groups: a group is reused across command kinds");
}


// Engine-level group exclusivity: two accepted commands bind different groups,
// a busy pool refuses a third, and the group is released after the terminal
// response is polled.
void test_engine_group_exclusion_and_binding() {
    pl::PanelLuConfig config;
    config.compute_group_count = 2;
    config.pes_per_group = 2;
    config.pe_count = 4;
    dp::BufferManager buffers(dp::BufferConfig{4, 64 * 1024, 4, 1});
    pl::PanelEngine engine(config);
    engine.set_buffers(&buffers);
    expect(engine.configure().ok, "engine groups: configured");

    const std::uint32_t n = 32;
    const std::uint32_t p = 16;
    std::vector<float> front(n * n, 0.0f);
    for (std::uint32_t i = 0; i < p; ++i) {
        for (std::uint32_t j = 0; j < p; ++j) {
            front[i * n + j] = i == j ? 2.0f : 0.25f * static_cast<float>(i + j + 1);
        }
    }
    auto stage = [&](const std::string& owner, std::size_t slot_index) {
        const auto allocate = buffers.allocate(owner, front.size() * 4, 0);
        expect(allocate.ok && allocate.handle.slot == slot_index,
               "engine groups: slot staged");
        const auto lease = buffers.begin_write(
            allocate.handle, owner, 0, front.size() * 4, 0);
        std::vector<std::uint8_t> bytes(front.size() * 4);
        std::memcpy(bytes.data(), front.data(), bytes.size());
        buffers.lease_write(lease.lease.lease_id, 0, bytes, 1, nullptr);
        return std::make_pair(allocate.handle, lease.lease);
    };
    const auto first = stage("g0", 0);
    const auto second = stage("g1", 1);

    auto left = trsm_request(
        pl::ComputeKind::TrsmLeft, n, p, first.first, first.second);
    left.txn_id = 1;
    left.command_id = 1;
    auto right = trsm_request(
        pl::ComputeKind::TrsmLeft, n, p, second.first, second.second);
    right.txn_id = 2;
    right.command_id = 2;
    const std::uint64_t key_a = pl::PanelEngine::adapter_key(2, 0);
    const std::uint64_t key_b = pl::PanelEngine::adapter_key(2, 1);
    expect(
        engine.begin_command(left, key_a, 2).ok &&
            engine.begin_command(right, key_b, 2).ok,
        "engine groups: two commands accepted");
    const std::uint32_t group_a = engine.group_of_command(1);
    const std::uint32_t group_b = engine.group_of_command(2);
    expect(
        group_a != cmd::NONE && group_b != cmd::NONE && group_a != group_b,
        "engine groups: two accepted commands bind distinct groups");
    // A third command cannot enter a busy group pool.
    auto third = trsm_request(
        pl::ComputeKind::TrsmLeft, n, p, first.first, first.second);
    third.txn_id = 3;
    third.command_id = 3;
    const auto refused = engine.begin_command(third, pl::PanelEngine::adapter_key(2, 2), 2);
    expect(
        !refused.ok && refused.status == cmd::StatusCode::BufferFull,
        "engine groups: a busy pool refuses a third command (resource stall)");
    std::vector<bool> groups_busy_seen(2, false);
    bool responded_a = false;
    bool responded_b = false;
    for (std::uint64_t cycle = 2; cycle < 400000; ++cycle) {
        engine.tick(cycle);
        groups_busy_seen[0] = groups_busy_seen[0] || engine.group_busy(0);
        groups_busy_seen[1] = groups_busy_seen[1] || engine.group_busy(1);
        pl::PanelEngineResponse response;
        if (!responded_a && engine.poll_response(key_a, response)) {
            responded_a = true;
            expect(response.status == cmd::StatusCode::Ok, "engine groups: cmd A Ok");
        }
        if (!responded_b && engine.poll_response(key_b, response)) {
            responded_b = true;
            expect(response.status == cmd::StatusCode::Ok, "engine groups: cmd B Ok");
        }
        if (responded_a && responded_b) {
            break;
        }
    }
    expect(responded_a && responded_b, "engine groups: both commands responded");
    expect(groups_busy_seen[0] && groups_busy_seen[1], "engine groups: both groups were used");
    expect(
        !engine.group_busy(0) && !engine.group_busy(1),
        "engine groups: released after the responses were polled");
    // Every task of a command ran inside its own group's PE range.
    for (const auto& record : engine.task_records()) {
        const std::uint32_t group =
            record.task.command_id == 1 ? group_a : group_b;
        const std::uint32_t first_pe = group * 2;
        expect(
            record.pe_id >= first_pe && record.pe_id < first_pe + 2,
            "engine groups: tasks stay inside their group");
    }
}

// ---------------------------------------------------------------------------
// Unified on-chip bandwidth arbitration
// ---------------------------------------------------------------------------

void test_arbiter_unit() {
    dp::OnChipArbiter arbiter;
    arbiter.configure(64, 64);
    arbiter.begin_cycle(1);
    const auto a = arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, 0, 64);
    const auto b = arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PanelScan, 0, 64);
    expect(a == 64, "arbiter: first requester gets the full budget");
    expect(b == 0, "arbiter: the cycle remainder is not shared");
    expect(arbiter.pending(dp::OnChipDomain::Read) == 1, "arbiter: denied need kept");
    arbiter.begin_cycle(2);
    const auto a2 = arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, 0, 64);
    const auto b2 = arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PanelScan, 0, 64);
    expect(a2 == 0, "arbiter: rotation denies the previous winner");
    expect(b2 == 64, "arbiter: rotation serves the waiting requester");
    // A granted client may consume the remaining budget in the same cycle.
    arbiter.clear_pending(dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, 0);
    arbiter.begin_cycle(3);
    const auto c1 = arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, 1, 40);
    const auto c2 = arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, 1, 40);
    expect(c1 == 40 && c2 == 24, "arbiter: same requester drains the remainder");
    expect(
        arbiter.grants(dp::OnChipDomain::Read) == 4,
        "arbiter: grant accounting");
    // Domain separation.
    const auto write_grant = arbiter.acquire(
        dp::OnChipDomain::Write, dp::OnChipClientKind::PeStore, 0, 64);
    expect(write_grant == 64, "arbiter: write domain has its own grant");
    // Stale pending requests expire instead of blocking the rotation.
    arbiter.begin_cycle(4);
    (void)arbiter.acquire(
        dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, 0, 64);
    arbiter.begin_cycle(5);
    expect(
        arbiter.pending(dp::OnChipDomain::Read) == 0,
        "arbiter: stale pending needs are dropped");
}

void test_bandwidth_integration_low_budget() {
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    const std::vector<float> matrix = make_matrix(n, 17);
    cp::ControlPlaneConfig config = base_config(2, 2);
    config.panel.load_bytes_per_cycle = 16;
    config.panel.writeback_bytes_per_cycle = 16;
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = p;
    node.front_values = matrix;
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = n;
    testing::TestArtifact artifact = testing::build_test_artifact({node}, options);
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "bandwidth: device starts");
    while (!device.finished() && device.cycle() < 8000000) {
        device.tick();
    }
    expect(device.finished(), "bandwidth: batch terminates under low budget");
    expect(
        device.completion_status().at(command_index(artifact, cmd::Opcode::TrsmLeft)) ==
            cmd::StatusCode::Ok,
        "bandwidth: TRSM_LEFT succeeds under low budget");
    const auto& arbiter = device.on_chip_arbiter();
    std::uint64_t pe_load_grants = 0;
    for (std::uint32_t pe = 0; pe < device.panel_engine()->pes().size(); ++pe) {
        pe_load_grants +=
            arbiter.stats(dp::OnChipDomain::Read, dp::OnChipClientKind::PeLoad, pe)
                .grants;
    }
    expect(pe_load_grants > 0, "bandwidth: PE loads receive read grants");
    std::uint64_t scan_grants = 0;
    for (std::uint32_t id = 0; id <= 32; ++id) {
        scan_grants +=
            arbiter.stats(dp::OnChipDomain::Read, dp::OnChipClientKind::PanelScan, id)
                .grants;
    }
    expect(scan_grants > 0, "bandwidth: the scan receives read grants");
    expect(
        arbiter.grants(dp::OnChipDomain::Read) <= device.cycle(),
        "bandwidth: at most one read grant per cycle");
    std::uint64_t factor_read_grants = 0;
    for (std::uint32_t id = 0; id <= 8; ++id) {
        factor_read_grants += arbiter
                                  .stats(
                                      dp::OnChipDomain::Read,
                                      dp::OnChipClientKind::FactorRead,
                                      id)
                                  .grants;
    }
    expect(
        factor_read_grants > 0,
        "bandwidth: FactorStore staging reads receive grants");
    // The T09 chain exercises the DMA on-chip write side (LOAD_FRONT is
    // MemoryToBuffer). The DMA buffer-side read side (BufferToMemory writeback)
    // requires STORE_UPDATE, which depends on the top-level GEMM refused in
    // this scope; it is exercised by the T04/T05 DMA tests with the same
    // device-level arbiter attached.
    expect(
        arbiter.stats(dp::OnChipDomain::Write, dp::OnChipClientKind::Dma, 0).grants > 0,
        "bandwidth: DMA on-chip writes receive grants");
    expect(
        arbiter.stats(dp::OnChipDomain::Write, dp::OnChipClientKind::Dma, 0).bytes <=
            arbiter.grants(dp::OnChipDomain::Write) * 64,
        "bandwidth: DMA bytes stay within granted budgets");
    expect(
        arbiter.stats(dp::OnChipDomain::Write, dp::OnChipClientKind::PeStore, 0)
                .grants > 0,
        "bandwidth: PE writeback receives write grants");
}

// ---------------------------------------------------------------------------
// Abort during TRSM and reuse
// ---------------------------------------------------------------------------

void test_abort_trsm_and_reuse() {
    const std::uint32_t n = 48;
    const std::uint32_t p = 32;
    testing::TestNode node0;
    node0.node_id = 0;
    node0.parent_id = -1;
    node0.total_dim = n;
    node0.pivot_dim = p;
    node0.front_values = make_matrix(n, 61);
    testing::TestNode node1;
    node1.node_id = 1;
    node1.parent_id = -1;
    node1.total_dim = n;
    node1.pivot_dim = p;
    node1.front_values = make_matrix(n, 67);
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = n;
    testing::TestArtifact artifact =
        testing::build_test_artifact({node0, node1}, options);
    const std::uint32_t left0 = command_index(artifact, cmd::Opcode::TrsmLeft, 0);
    const std::uint32_t factor1 = command_index(artifact, cmd::Opcode::StoreFactor, 1);

    cp::ControlPlaneConfig config = base_config(2, 2);
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "abort trsm: device starts");
    // Wait until a TRSM task of node 0 is running, then abort that node.
    bool aborted = false;
    std::uint32_t aborted_group = cmd::NONE;
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
        if (!aborted && device.cycle() > 0) {
            for (const auto& record : device.panel_engine()->task_records()) {
                if (record.task.command_id == left0) {
                    aborted_group =
                        device.panel_engine()->group_of_command(left0);
                    device.abort_node(0);
                    aborted = true;
                    break;
                }
            }
        }
    }
    expect(aborted, "abort trsm: abort issued while TRSM is active");
    expect(device.finished(), "abort trsm: batch terminates");
    expect(
        device.completion_status().at(left0) == cmd::StatusCode::Aborted,
        "abort trsm: aborted TRSM command reports ABORTED");
    expect(
        device.completion_status().at(factor1) == cmd::StatusCode::Ok,
        "abort trsm: the other front completes normally");
    expect(
        device.dataplane().buffers().active_lease_count() == 0,
        "abort trsm: leases released after draining");
    expect(
        !device.panel_engine()->group_busy(0) && !device.panel_engine()->group_busy(1),
        "abort trsm: groups released after draining");
    // The released group is reusable: the surviving front's commands ran on it
    // after the abort.
    bool reused = false;
    for (const auto& record : device.panel_engine()->task_records()) {
        if (record.task.command_id != left0 && record.group_id == aborted_group) {
            reused = true;
        }
    }
    expect(reused, "abort trsm: the released group is reused by the other front");
}

// ---------------------------------------------------------------------------
// B01 regression (adapter level): the immediate terminal of an accepted
// request survives abort_txn
// ---------------------------------------------------------------------------

// An accepted request whose runtime shape/operands are illegal produces an
// immediate terminal response without ever entering the engine. An abort that
// lands before the consumer polls must not delete that unique terminal: it is
// converted into the same txn's ABORTED terminal (identity and statistics
// preserved) and consumed exactly once.
void test_immediate_terminal_survives_abort() {
    pl::PanelLuConfig config;
    config.compute_group_count = 1;
    config.pes_per_group = 1;
    config.pe_count = 1;
    pl::PanelEngine engine(config);
    expect(engine.configure().ok, "immediate abort: engine configured");
    cp::TrsmExecutionUnit unit(0, &engine);

    const auto make_request = [](std::uint64_t txn, std::uint32_t command) {
        cp::ExecRequest request;
        request.txn_id = txn;
        request.epoch = 4;
        request.command_id = command;
        request.node_id = 7;
        request.opcode = cmd::Opcode::TrsmLeft;
        request.unit_id = 0;
        // Two-element front with p=1: u=1.
        request.total_dim = 2;
        request.front.pivot_dim = 1;
        request.front.tile_size = 16;
        request.front.front.descriptor_id = 5;
        request.front.front.base_addr = 0x1000;
        request.front.front.byte_size = 64;
        request.kernel.a = request.front.front;
        request.kernel.b = request.front.front;
        request.kernel.c = request.front.front;
        request.kernel.m = 1;  // TRSM_LEFT M must equal p
        request.kernel.n = 1;
        request.kernel.k = 1;
        request.kernel.tile_size = 16;
        request.kernel.backend = cmd::KernelBackend::SystemCFp32DeviceModel;
        return request;
    };
    const auto check_identity = [&](const cp::ExecResponse& response,
                                    const cp::ExecRequest& request,
                                    const char* label) {
        expect(response.txn_id == request.txn_id &&
                   response.command_id == request.command_id &&
                   response.node_id == request.node_id &&
                   response.epoch == request.epoch &&
                   response.unit_id == unit.unit_id(),
               std::string(label) + ": terminal keeps txn/command/node/epoch identity");
    };

    // Case 1: abort of a DIFFERENT txn must not delete or modify the pending
    // terminal of this transaction.
    {
        cp::ExecRequest request = make_request(21, 3);
        request.kernel.m = 2;  // runtime shape error: engine refuses it
        expect(unit.submit(request, 10), "immediate abort: shape refusal accepted");
        expect(unit.busy(), "immediate abort: accepted refusal keeps the unit busy");
        unit.abort_txn(request.txn_id + 100, 11);
        cp::ExecResponse response;
        expect(unit.poll_response(response),
               "immediate abort: unrelated abort keeps the terminal");
        expect(response.status == cmd::StatusCode::BadDescriptor,
               "immediate abort: unrelated abort does not rewrite the status");
        check_identity(response, request, "immediate abort");
        expect(!unit.poll_response(response), "immediate abort: one terminal only");
    }

    // Case 2: the B01 window. The request is accepted with an immediate error
    // terminal; abort of the same txn converts (never drops) that terminal.
    {
        cp::ExecRequest request = make_request(22, 4);
        request.kernel.m = 2;  // the malformed TRSM shape of the B01 repro
        expect(unit.submit(request, 20), "immediate abort: malformed shape accepted");
        unit.abort_txn(request.txn_id, 21);
        cp::ExecResponse response;
        expect(unit.poll_response(response),
               "immediate abort: exactly one terminal survives the abort");
        expect(response.status == cmd::StatusCode::Aborted,
               "immediate abort: pending terminal becomes ABORTED");
        check_identity(response, request, "immediate abort");
        expect(!unit.poll_response(response),
               "immediate abort: terminal is consumed exactly once");
        expect(!unit.busy(), "immediate abort: unit is idle after the terminal");
        unit.abort_txn(request.txn_id, 22);
        unit.abort_txn(request.txn_id, 23);
        expect(!unit.poll_response(response),
               "immediate abort: repeated abort is idempotent");
    }

    // Case 3: the adapter's own operand refusal shares the same immediate
    // terminal path and the same lifecycle contract.
    {
        cp::ExecRequest request = make_request(23, 5);
        request.kernel.a.descriptor_id = 99;  // foreign KERNEL operand
        expect(unit.submit(request, 30), "immediate abort: operand refusal accepted");
        unit.abort_txn(request.txn_id, 31);
        cp::ExecResponse response;
        expect(unit.poll_response(response),
               "immediate abort: operand-refusal terminal survives the abort");
        expect(response.status == cmd::StatusCode::Aborted,
               "immediate abort: operand-refusal terminal becomes ABORTED");
        check_identity(response, request, "immediate abort");
        expect(!unit.poll_response(response) && !unit.busy(),
               "immediate abort: operand-refusal txn is terminal exactly once");
    }
    expect(!engine.group_busy(0),
           "immediate abort: refused requests never consumed an engine group");
}

}  // namespace

int main(int argc, char** argv) {
    g_output_dir = argc > 1 ? argv[1] : "";
    test_manual_cases();
    test_engine_diagonal_failures_and_refusals();
    test_nonzero_front_offset();
    test_fp64_diagnostic_rounding_overflow();
    test_kernel_operand_reference_validation();
    test_integration_n42_full_factor_and_oracle();
    test_np_factor_chain();
    test_groups_and_reuse();
    test_engine_group_exclusion_and_binding();
    test_arbiter_unit();
    test_bandwidth_integration_low_budget();
    test_abort_trsm_and_reuse();
    test_immediate_terminal_survives_abort();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
