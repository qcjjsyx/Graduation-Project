// T05 control-plane tests (no SystemC): fetch/decode faults, scoreboard
// rules, stub execution unit behaviour, and full-batch control flow compared
// against the independent interpreter in control_reference.hpp.

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "controlplane/control_plane.hpp"
#include "control_reference.hpp"
#include "control_test_artifact.hpp"
#include "dataplane/artifact_loader.hpp"

namespace {

namespace cmd = hw::command;
namespace dp = hw::dataplane;
namespace cp = hw::controlplane;
namespace testing = hw::testing;

int failures = 0;
int checks = 0;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

cp::ControlPlaneConfig base_config() {
    cp::ControlPlaneConfig config;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 1;
    config.trsm_unit_count = 2;
    config.gemm_unit_count = 1;
    config.solve_unit_count = 1;
    config.assemble_unit_count = 1;
    config.buffer_slot_count = 5;
    config.buffer_capacity_bytes = 256;
    config.buffer_port_count = 4;
    config.command_window_depth = 8;
    config.completion_queue_depth = 8;
    config.dma_queue_depth = 8;
    config.backend = cp::ExecBackend::TestStub;
    config.enable_test_stub = true;
    config.stub.default_latency = 3;
    config.memory.startup_cycles = 1;
    config.memory.beat_bytes = 16;
    config.memory.beat_cycles = 1;
    config.memory.max_inflight = 8;
    config.dma.beat_bytes = 16;
    config.dma.startup_cycles = 1;
    config.dma.beat_interval_cycles = 1;
    return config;
}

struct RunResult {
    cp::BatchState state{cp::BatchState::Idle};
    std::string fault;
    std::map<std::uint32_t, cmd::StatusCode> status;
    std::map<std::uint32_t, cmd::StatusCode> completion_status;
    std::map<std::uint32_t, std::uint32_t> pivot_count;
    std::map<std::uint32_t, cp::CommandStats> stats;
    std::uint64_t cycles{0};
    std::uint32_t completions{0};
    bool control_flow_pass{false};
};

RunResult collect(cp::CommandDevice& device) {
    RunResult result;
    result.state = device.state();
    result.fault = device.fault_reason();
    result.cycles = device.cycle();
    result.completions = device.completion_written();
    result.control_flow_pass = device.control_flow_pass();
    for (const auto& entry : device.completion_status()) {
        result.completion_status[entry.first] = entry.second;
    }
    for (const auto& row : device.command_stats()) {
        result.stats[row.command_id] = row;
        result.pivot_count[row.command_id] = row.pivot_count;
    }
    return result;
}

std::vector<testing::TestNode> single_node() {
    return {testing::TestNode{0, -1, 3, 2, {}}};
}

std::vector<testing::TestNode> two_roots() {
    return {testing::TestNode{0, -1, 3, 2, {}}, testing::TestNode{1, -1, 3, 2, {}}};
}

std::vector<testing::TestNode> two_chains() {
    // Two independent parent/child chains: both parents assemble while the
    // other chain still moves data, forcing buffer-port contention.
    std::vector<testing::TestNode> nodes;
    nodes.push_back(testing::TestNode{0, 1, 2, 1, {}});
    nodes.push_back(testing::TestNode{1, -1, 2, 2, {{0, {1}}}});
    nodes.push_back(testing::TestNode{2, 3, 2, 1, {}});
    nodes.push_back(testing::TestNode{3, -1, 2, 2, {{2, {1}}}});
    return nodes;
}

std::vector<testing::TestNode> parent_child() {
    // Child update (1x1) maps to parent column 1.
    std::vector<testing::TestNode> nodes;
    nodes.push_back(testing::TestNode{0, 1, 2, 1, {}});
    nodes.push_back(testing::TestNode{1, -1, 2, 2, {{0, {1}}}});
    return nodes;
}

void check_interpreter(
    const testing::TestArtifact& artifact,
    const RunResult& result,
    const std::map<std::uint32_t, testing::RefFailure>& injected,
    const std::set<std::uint32_t>& aborted,
    const std::string& label) {
    const auto expected = testing::interpret(
        artifact.commands, artifact.descriptors,
        artifact.artifact.image, static_cast<std::uint32_t>(artifact.commands.size()),
        injected, aborted);
    bool statuses_ok = true;
    for (const auto& command : artifact.commands) {
        const auto found = result.completion_status.find(command.command_id);
        if (found == result.completion_status.end()) {
            statuses_ok = false;
            std::cerr << "  missing completion for command " << command.command_id
                      << '\n';
            continue;
        }
        const auto want = expected.status.at(command.command_id);
        if (aborted.count(command.node_id) != 0 && aborted.size() == 1) {
            // Abort timing may leave already-finished commands with their own
            // status; only the cancelled set is asserted exactly.
            if (found->second != want && found->second != cmd::StatusCode::Ok) {
                statuses_ok = false;
                std::cerr << "  command " << command.command_id << " status "
                          << dp::status_name(found->second) << " expected "
                          << dp::status_name(want) << '\n';
            }
            continue;
        }
        if (found->second != want) {
            statuses_ok = false;
            std::cerr << "  command " << command.command_id << " status "
                      << dp::status_name(found->second) << " expected "
                      << dp::status_name(want) << '\n';
        }
    }
    expect(statuses_ok, label + ": command statuses match the interpreter");
    bool pivots_ok = true;
    for (const auto& command : artifact.commands) {
        const auto found = result.pivot_count.find(command.command_id);
        const auto want = expected.pivot_count.at(command.command_id);
        if (found == result.pivot_count.end() || found->second != want) {
            pivots_ok = false;
            std::cerr << "  command " << command.command_id << " pivot_count "
                      << (found == result.pivot_count.end()
                              ? 0
                              : found->second)
                      << " expected " << want << '\n';
        }
    }
    expect(pivots_ok, label + ": pivot_count matches the interpreter");
}

void check_data(
    cp::CommandDevice& device,
    const testing::TestArtifact& artifact,
    const std::string& label) {
    const auto& image = device.dataplane().memory().image();
    auto slice = [&](const std::string& region, const std::vector<std::uint8_t>& expected) {
        const auto* info = device.boot().region(region);
        if (info == nullptr) {
            expect(false, label + ": missing region " + region);
            return;
        }
        const std::vector<std::uint8_t> actual(
            image.begin() + static_cast<std::ptrdiff_t>(info->offset),
            image.begin() + static_cast<std::ptrdiff_t>(info->offset + expected.size()));
        expect(actual == expected, label + ": bytes of " + region);
    };
    for (const auto& node : artifact.nodes) {
        const std::string prefix = "node." + std::to_string(node.node_id);
        slice(prefix + ".factor_l", artifact.expected_l(node.node_id));
        slice(prefix + ".factor_u", artifact.expected_u(node.node_id));
        if (node.total_dim > node.pivot_dim) {
            slice(prefix + ".update", artifact.expected_update(node.node_id));
        }
        slice(prefix + ".p_vector", artifact.expected_pvector(node.pivot_dim));
    }
    slice("solution_data", artifact.expected_solution());
}

void test_scoreboard() {
    cp::TokenScoreboard board;
    board.init(4);
    board.register_producer(0, 0);
    board.register_producer(1, 1);
    board.register_producer(2, 2);
    expect(board.check({0}) == cp::TokenScoreboard::Deps::Pending, "deps pending");
    expect(board.check({cmd::NONE}) == cp::TokenScoreboard::Deps::Ready, "NONE ignored");
    expect(board.signal(0, cp::TokenScoreboard::State::Ready, 0), "signal ready");
    expect(!board.signal(0, cp::TokenScoreboard::State::Failed, 0),
           "single terminal transition");
    expect(!board.signal(1, cp::TokenScoreboard::State::Ready, 0),
           "producer mismatch rejected");
    expect(board.signal(1, cp::TokenScoreboard::State::Failed, 1), "signal failed");
    expect(board.check({0, 1}) == cp::TokenScoreboard::Deps::Failed, "failed dep wins");
    expect(board.check({0}) == cp::TokenScoreboard::Deps::Ready, "ready dep");
    expect(board.check({3}) == cp::TokenScoreboard::Deps::Pending, "unknown token pending");
    expect(board.ready_count() == 1 && board.failed_count() == 1, "counts");
}

void test_stub_unit() {
    dp::BufferManager buffers(dp::BufferConfig{2, 64, 2, 1});
    cp::StubConfig config;
    config.default_latency = 2;
    config.transform = cp::StubTransform::Pattern;
    config.pattern_bytes = 8;
    config.response_hold_cycles = 2;
    cp::TestStubUnit unit(cp::UnitKind::Panel, 0, config, &buffers);
    auto slot = buffers.allocate("u", 64, 0);
    auto lease = buffers.begin_write(slot.handle, "u", 0, 64, 0);
    expect(lease.ok, "stub: lease");

    cp::ExecRequest request;
    request.txn_id = 7;
    request.epoch = 0;
    request.command_id = 3;
    request.opcode = cmd::Opcode::PanelLu;
    request.front_lease = lease.lease;
    request.has_front_lease = true;
    request.range_bytes = 64;
    request.policy = cmd::PivotPolicy::Static;
    expect(unit.submit(request, 1), "stub: accepted");
    expect(!unit.submit(request, 1), "stub: busy rejects second submit");
    unit.tick(2);
    cp::ExecResponse response;
    expect(!unit.poll_response(response), "stub: not finished before latency");
    unit.tick(3);
    // Response held stable until the consumer takes it.
    expect(!unit.poll_response(response), "stub: response backpressure holds");
    unit.tick(5);
    expect(unit.poll_response(response), "stub: response ready after hold");
    expect(response.status == cmd::StatusCode::Ok, "stub: ok status");
    expect(response.needs_pvector_write && response.pvector_dim == 0,
           "stub: static panel asks for identity P-vector");
    const auto& storage = buffers.bytes(slot.handle.slot);
    expect(std::all_of(storage.begin(), storage.begin() + 8,
                       [](std::uint8_t v) { return v == 3; }),
           "stub: pattern transform uses the command id");

    unit.inject_failure(4, {cmd::StatusCode::StaticPivotFailure, 2, 0});
    request.txn_id = 8;
    request.command_id = 4;
    expect(unit.submit(request, 5), "stub: second txn");
    unit.tick(7);
    unit.tick(11);
    expect(unit.poll_response(response), "stub: injected failure response");
    expect(response.status == cmd::StatusCode::StaticPivotFailure, "stub: injected status");
    expect(response.pivot_count == 2, "stub: injected failure column");
    expect(!response.needs_pvector_write, "stub: no p-vector write on failure");

    request.txn_id = 9;
    request.command_id = 5;
    expect(unit.submit(request, 8), "stub: third txn");
    unit.abort_epoch(0, 9);
    unit.tick(11);
    expect(!unit.poll_response(response), "stub: aborted txn leaves no response");
}

void test_fetch_faults() {
    auto artifact = testing::build_test_artifact(single_node());
    // Runtime corruption: an illegal opcode in the command stream (the loader
    // accepted the artifact at build time; faults appear at fetch).
    auto corrupted = artifact.artifact;
    const auto* command_region = corrupted.config.region("command_buffer");
    corrupted.image[command_region->offset] = 0xEE;
    cp::CommandDevice device(corrupted, base_config());
    std::string error;
    expect(device.start(&error), "fetch fault: batch starts");
    device.run(2000);
    expect(device.state() == cp::BatchState::Faulted, "fetch fault: batch faulted");
    expect(device.fault_reason().find("command") != std::string::npos,
           "fetch fault: reason mentions the command stream");

    // Truncated stream: boot claims more commands than the region holds.
    auto truncated = artifact.artifact;
    truncated.config.command_count += 4;
    cp::CommandDevice device2(truncated, base_config());
    expect(device2.start(&error), "truncation: batch starts");
    device2.run(2000);
    expect(device2.state() == cp::BatchState::Faulted, "truncation: batch faulted");
}

void test_backend_gating() {
    auto artifact = testing::build_test_artifact(single_node());
    auto config = base_config();
    config.backend = cp::ExecBackend::RequireRealKernels;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(!device.start(&error), "real-kernel backend refuses to run");
    expect(error.find("refusing") != std::string::npos, "refusal is explicit");

    config.backend = cp::ExecBackend::TestStub;
    config.enable_test_stub = false;
    cp::CommandDevice device2(artifact.artifact, config);
    expect(!device2.start(&error), "test stub requires explicit enable");
}

void test_single_node_flow() {
    auto artifact = testing::build_test_artifact(single_node());
    auto config = base_config();
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "single: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "single: control flow completes");
    expect(device.completion_written() == artifact.commands.size(),
           "single: one completion per command");
    check_interpreter(artifact, result, {}, {}, "single");
    check_data(device, artifact, "single");
    // STATIC identity P-vector actually written and awaited.
    const auto* p_region = device.boot().region("node.0.p_vector");
    const auto& image = device.dataplane().memory().image();
    const std::vector<std::uint8_t> p_bytes(
        image.begin() + static_cast<std::ptrdiff_t>(p_region->offset),
        image.begin() + static_cast<std::ptrdiff_t>(p_region->offset + 8));
    expect(p_bytes == artifact.expected_pvector(2), "single: identity P-vector written");
}

void test_two_roots_dual_front() {
    auto artifact = testing::build_test_artifact(two_roots());
    cp::ControlPlaneConfig config = base_config();
    config.panel_unit_count = 2;
    config.trsm_unit_count = 2;
    config.gemm_unit_count = 2;
    // Long deterministic latencies so same-kind work of the two fronts
    // genuinely overlaps (the stub is explicitly the execution backend).
    config.stub.latency_by_opcode[cmd::Opcode::PanelLu] = 300;
    config.stub.latency_by_opcode[cmd::Opcode::TrsmLeft] = 300;
    config.stub.latency_by_opcode[cmd::Opcode::TrsmRight] = 300;
    config.stub.latency_by_opcode[cmd::Opcode::GemmSchur] = 300;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "dual: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "dual: control flow completes");
    check_interpreter(artifact, result, {}, {}, "dual");
    check_data(device, artifact, "dual");

    // Overlap evidence: two panel units busy on different nodes at the same
    // time (intervals overlap).
    bool panel_overlap = false;
    const auto& units = device.unit_records();
    for (std::size_t i = 0; i < units.size(); ++i) {
        for (std::size_t j = i + 1; j < units.size(); ++j) {
            if (units[i].kind == "panel" && units[j].kind == "panel" &&
                units[i].node_id != units[j].node_id &&
                units[i].start < units[j].finish && units[j].start < units[i].finish) {
                panel_overlap = true;
            }
        }
    }
    expect(panel_overlap, "dual: two fronts executed panel work in parallel");

    // Two GEMM instances active at the same time.
    bool gemm_overlap = false;
    for (std::size_t i = 0; i < units.size(); ++i) {
        for (std::size_t j = i + 1; j < units.size(); ++j) {
            if (units[i].kind == "gemm" && units[j].kind == "gemm" &&
                units[i].unit_id != units[j].unit_id &&
                units[i].start < units[j].finish && units[j].start < units[i].finish) {
                gemm_overlap = true;
            }
        }
    }
    expect(gemm_overlap, "dual: two GEMM instances were active together");
}

void test_single_unit_contention() {
    auto artifact = testing::build_test_artifact(two_roots());
    cp::ControlPlaneConfig config = base_config();
    config.panel_unit_count = 1;
    config.trsm_unit_count = 1;
    config.gemm_unit_count = 1;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "contention: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "contention: completes with one unit per kind");
    // Same-kind units never overlap when only one exists.
    bool same_unit_overlap = false;
    const auto& units = device.unit_records();
    for (std::size_t i = 0; i < units.size(); ++i) {
        for (std::size_t j = i + 1; j < units.size(); ++j) {
            if (units[i].kind == units[j].kind && units[i].unit_id == units[j].unit_id &&
                units[i].start < units[j].finish && units[j].start < units[i].finish) {
                same_unit_overlap = true;
            }
        }
    }
    expect(!same_unit_overlap, "contention: a unit runs one command at a time");
    check_interpreter(artifact, result, {}, {}, "contention");
}

void test_parent_child_assembly() {
    auto artifact = testing::build_test_artifact(parent_child());
    cp::ControlPlaneConfig config = base_config();
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "assembly: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "assembly: control flow completes");
    check_interpreter(artifact, result, {}, {}, "assembly");
    check_data(device, artifact, "assembly");

    // Parent assembly may only run after the child's STORE_UPDATE wrote its
    // contribution block (dependency + data visibility).
    std::uint64_t child_update_finish = 0;
    std::uint64_t parent_assemble_start = 0;
    for (const auto& row : device.command_stats()) {
        if (row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::StoreUpdate)) &&
            row.node_id == 0) {
            child_update_finish = row.finish_cycle;
        }
        if (row.opcode ==
                std::to_string(static_cast<std::uint32_t>(cmd::Opcode::AssembleExtendAdd)) &&
            row.node_id == 1) {
            parent_assemble_start = row.start_cycle;
        }
    }
    expect(
        parent_assemble_start >= child_update_finish,
        "assembly: parent assemble starts after the child contribution is visible");
}

void test_dynamic_panel_refusal() {
    testing::TestArtifactOptions options;
    options.policy = cmd::PivotPolicy::DynamicPanel;
    auto artifact = testing::build_test_artifact(two_roots(), options);
    auto config = base_config();
    config.panel_unit_count = 2;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "dynamic: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "dynamic: batch still completes");
    check_interpreter(artifact, result, {}, {}, "dynamic");
    std::uint32_t refused = 0;
    for (const auto& row : device.command_stats()) {
        if (row.status == "POLICY_NOT_SUPPORTED") {
            ++refused;
        }
    }
    expect(refused == 2, "dynamic: both PANEL_LU commands refused");
}

void test_failure_propagation() {
    auto artifact = testing::build_test_artifact(parent_child());
    const std::map<std::uint32_t, testing::RefFailure> injected = {
        {2, {cmd::StatusCode::StaticPivotFailure, 1}}};
    // Command 2 is node 0's PANEL_LU in the interleaved stream.
    bool is_panel = false;
    for (const auto& command : artifact.commands) {
        if (command.command_id == 2 && command.opcode == cmd::Opcode::PanelLu) {
            is_panel = true;
        }
    }
    expect(is_panel, "failure: injection targets a PANEL_LU command");
    cp::CommandDevice device(artifact.artifact, base_config());
    for (const auto& entry : injected) {
        device.inject_failure(
            entry.first, cp::ExecFailure{entry.second.status, entry.second.pivot_count, 0});
    }
    std::string error;
    expect(device.start(&error), "failure: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "failure: batch completes (control flow)");
    check_interpreter(artifact, result, injected, {}, "failure");
}

void test_small_window() {
    for (const std::uint32_t depth : {1u, 2u, 4u}) {
        auto artifact = testing::build_test_artifact(parent_child());
        auto config = base_config();
        config.command_window_depth = depth;
        cp::CommandDevice device(artifact.artifact, config);
        std::string error;
        expect(device.start(&error), "window: start");
        device.run(200000);
        RunResult result = collect(device);
        expect(
            device.control_flow_pass(),
            "window: completes with window depth " + std::to_string(depth));
        check_interpreter(artifact, result, {}, {}, "window" + std::to_string(depth));
    }
}

void test_single_front_config() {
    auto artifact = testing::build_test_artifact(two_roots());
    auto config = base_config();
    config.max_resident_fronts = 1;
    config.buffer_slot_count = 3;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "single-front: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "single-front: completes");
    check_interpreter(artifact, result, {}, {}, "single-front");
    // Only one front resident at a time.
    bool overlap = false;
    const auto& fronts = device.front_records();
    for (std::size_t i = 0; i < fronts.size(); ++i) {
        for (std::size_t j = i + 1; j < fronts.size(); ++j) {
            if (fronts[i].start < fronts[j].finish && fronts[j].start < fronts[i].finish) {
                overlap = true;
            }
        }
    }
    expect(!overlap, "single-front: no simultaneous residency");
}

void test_completion_backpressure() {
    auto artifact = testing::build_test_artifact(two_roots());
    auto config = base_config();
    config.completion_queue_depth = 1;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "completion: start");
    device.run(200000);
    expect(device.control_flow_pass(), "completion: completes under backpressure");
    expect(
        device.completion_written() == artifact.commands.size(),
        "completion: exactly one completion per command (none lost, none repeated)");
}

void test_buffer_capacity_failure() {
    auto artifact = testing::build_test_artifact(single_node());
    auto config = base_config();
    config.buffer_capacity_bytes = 4;  // a 3x3 front never fits
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "capacity: start");
    device.run(200000);
    bool begin_failed = false;
    bool begin_is_buffer_full = false;
    for (const auto& row : device.command_stats()) {
        if (row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::NodeBegin))) {
            begin_failed = true;
            begin_is_buffer_full = row.status == "BUFFER_FULL";
        }
    }
    expect(begin_failed, "capacity: NODE_BEGIN was tracked");
    expect(begin_is_buffer_full, "capacity: explicit BUFFER_FULL");
    expect(device.control_flow_pass(), "capacity: batch completes with failures");
}

void test_timeout() {
    auto artifact = testing::build_test_artifact(parent_child());
    auto config = base_config();
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "timeout: start");
    device.run(3);  // far too few cycles
    expect(device.state() == cp::BatchState::Faulted, "timeout: batch faulted");
    expect(device.fault_reason() == "batch timeout", "timeout: reason");
}

void test_abort_and_reset() {
    auto artifact = testing::build_test_artifact(two_roots());
    const std::set<std::uint32_t> aborted{0};
    auto config = base_config();
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "abort: start");
    // Abort node 0 after a couple of its commands finished.
    std::uint32_t finished = 0;
    while (!device.finished() && device.cycle() < 200000) {
        device.tick();
        if (finished == 0 && device.command_stats().size() >= 2) {
            finished = 1;
            device.abort_node(0);
        }
    }
    expect(device.control_flow_pass(), "abort: batch completes");
    bool node0_all_terminal = true;
    bool node1_block_ok = true;
    for (const auto& row : device.command_stats()) {
        if (row.node_id == 1 && row.opcode != "10" && row.opcode != "11" &&
            row.status != "OK") {
            // The node block of the independent front must be untouched; its
            // solve commands may fail through the global solve chain (which
            // legitimately waits on the aborted node's solve token).
            node1_block_ok = false;
        }
        if (row.node_id == 0 && row.status != "ABORTED" && row.status != "OK" &&
            row.status != "DEPENDENCY_FAILED") {
            node0_all_terminal = false;
        }
    }
    expect(node0_all_terminal, "abort: aborted node's commands are terminal");
    expect(node1_block_ok, "abort: independent front completes OK");

    // Reset clears the batch; it must be restarted explicitly.
    device.reset();
    expect(device.state() == cp::BatchState::Idle, "reset: batch back to idle");
    expect(device.command_stats().empty(), "reset: statistics cleared");
    expect(device.start(&error), "reset: batch restarts");
    device.run(200000);
    expect(device.control_flow_pass(), "reset: restarted batch completes");
}

void test_abort_with_inflight_dma() {
    auto artifact = testing::build_test_artifact(two_roots());
    cp::CommandDevice device(artifact.artifact, base_config());
    std::string error;
    expect(device.start(&error), "abort-dma: start");

    // Abort node 0 while one of its STORE_FACTOR write-backs is in flight.
    std::uint32_t store_command = cmd::NONE;
    for (const auto& command : artifact.commands) {
        if (command.node_id == 0 && command.opcode == cmd::Opcode::StoreFactor) {
            store_command = command.command_id;
        }
    }
    expect(store_command != cmd::NONE, "abort-dma: found STORE_FACTOR");
    bool aborted = false;
    while (!device.finished() && device.cycle() < 200000) {
        device.tick();
        std::uint32_t issued = 0;
        std::uint32_t done = 0;
        std::uint32_t cancelled = 0;
        if (!aborted && device.command_dma_progress(
                            store_command, &issued, &done, &cancelled) &&
            issued >= 1 && done + cancelled < issued) {
            aborted = true;
            device.abort_node(0);
        }
    }
    expect(aborted, "abort-dma: aborted while a DMA job was in flight");
    expect(device.control_flow_pass(), "abort-dma: batch completes after abort");
    expect(
        device.completion_written() == artifact.commands.size(),
        "abort-dma: every command reaches a terminal completion");
    bool node0_terminal = true;
    for (const auto& row : device.command_stats()) {
        if (row.node_id == 0 && row.status != "ABORTED" && row.status != "OK" &&
            row.status != "DEPENDENCY_FAILED") {
            node0_terminal = false;
        }
    }
    expect(node0_terminal, "abort-dma: aborted node's commands are terminal");
    bool front_released = false;
    for (const auto& record : device.front_records()) {
        if (record.node_id == 0) {
            front_released = true;
        }
    }
    expect(front_released, "abort-dma: the aborted node's slots were released");
}

void test_abort_during_solve_staging() {
    auto artifact = testing::build_test_artifact(two_roots());
    cp::CommandDevice device(artifact.artifact, base_config());
    std::string error;
    expect(device.start(&error), "abort-solve: start");
    std::uint32_t solve_command = cmd::NONE;
    for (const auto& command : artifact.commands) {
        if (command.node_id == 0 && command.opcode == cmd::Opcode::SolveForward) {
            solve_command = command.command_id;
        }
    }
    expect(solve_command != cmd::NONE, "abort-solve: found SOLVE_FORWARD");
    bool aborted = false;
    while (!device.finished() && device.cycle() < 200000) {
        device.tick();
        std::uint32_t issued = 0;
        std::uint32_t done = 0;
        std::uint32_t cancelled = 0;
        if (!aborted && device.command_dma_progress(
                            solve_command, &issued, &done, &cancelled) &&
            issued >= 1 && done + cancelled < issued) {
            aborted = true;
            device.abort_node(0);
        }
    }
    expect(aborted, "abort-solve: aborted during solve staging");
    expect(device.control_flow_pass(), "abort-solve: batch completes after abort");
    expect(
        device.completion_written() == artifact.commands.size(),
        "abort-solve: every command reaches a terminal completion");
}

void test_dma_port_contention_retry() {
    // Assembly holds two leases at once (front write + temp read); with only
    // two ports, a concurrent cross-front DMA must wait/retry instead of
    // failing or deadlocking.
    auto artifact = testing::build_test_artifact(parent_child());
    auto config = base_config();
    config.buffer_port_count = 2;
    config.stub.latency_by_opcode[cmd::Opcode::PanelLu] = 300;
    config.stub.latency_by_opcode[cmd::Opcode::TrsmLeft] = 300;
    config.stub.latency_by_opcode[cmd::Opcode::TrsmRight] = 300;
    config.stub.latency_by_opcode[cmd::Opcode::GemmSchur] = 300;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "ports: start");
    device.run(400000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "ports: completes under port contention");
    check_interpreter(artifact, result, {}, {}, "ports");
    check_data(device, artifact, "ports");
}

void test_solve_full_traversal() {
    testing::TestArtifactOptions options;
    options.matrix_dim = 128;
    auto artifact = testing::build_test_artifact(single_node(), options);
    auto config = base_config();
    config.buffer_capacity_bytes = 256;  // 64 elements per chunk
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "solve: start");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "solve: completes");
    check_interpreter(artifact, result, {}, {}, "solve");
    // The complete 128-element RHS must land in the solution region.
    const auto expected = artifact.expected_solution();
    const auto* region = device.boot().region("solution_data");
    const auto& image = device.dataplane().memory().image();
    const std::vector<std::uint8_t> actual(
        image.begin() + static_cast<std::ptrdiff_t>(region->offset),
        image.begin() + static_cast<std::ptrdiff_t>(region->offset + expected.size()));
    expect(actual == expected, "solve: full RHS is transferred (no truncation)");
    std::uint64_t solve_bytes = 0;
    for (const auto& row : device.command_stats()) {
        if (row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::SolveForward)) ||
            row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::SolveBackward))) {
            solve_bytes += row.bytes_read + row.bytes_written;
        }
    }
    expect(
        solve_bytes >= 2 * 2 * expected.size(),
        "solve: chunked traversal moved the full vector twice (read+write)");
}

void test_start_validation() {
    auto artifact = testing::build_test_artifact(single_node());
    std::string error;
    {
        auto config = base_config();
        config.dma_queue_depth = 1;
        cp::CommandDevice device(artifact.artifact, config);
        expect(!device.start(&error), "validate: dma_queue_depth=1 refused");
        expect(error.find("dma_queue_depth") != std::string::npos,
               "validate: dma queue reason");
    }
    {
        auto config = base_config();
        config.max_resident_fronts = 3;
        cp::CommandDevice device(artifact.artifact, config);
        expect(!device.start(&error), "validate: max_resident_fronts=3 refused");
    }
    {
        auto config = base_config();
        config.buffer_slot_count = 2;
        cp::CommandDevice device(artifact.artifact, config);
        expect(!device.start(&error), "validate: too few buffer slots refused");
    }
    {
        auto config = base_config();
        config.panel_unit_count = 0;
        cp::CommandDevice device(artifact.artifact, config);
        expect(!device.start(&error), "validate: missing panel unit refused");
    }
    {
        auto config = base_config();
        config.command_window_depth = 0;
        cp::CommandDevice device(artifact.artifact, config);
        expect(!device.start(&error), "validate: zero window refused");
    }
    {
        auto config = base_config();
        config.buffer_port_count = 1;
        cp::CommandDevice device(artifact.artifact, config);
        expect(!device.start(&error), "validate: single buffer port refused");
        expect(
            error.find("buffer_port_count") != std::string::npos,
            "validate: single port reason mentions assembly leases");
    }
}

void test_default_config_runs() {
    auto artifact = testing::build_test_artifact(single_node());
    cp::ControlPlaneConfig config;  // defaults must pass start()
    config.enable_test_stub = true;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "default: default config starts");
    device.run(200000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "default: default config completes");
    check_interpreter(artifact, result, {}, {}, "default");
}

void test_abort_during_pvector_write() {
    auto artifact = testing::build_test_artifact(parent_child());
    auto config = base_config();
    // Keep the identity P-vector write in flight long enough to abort during
    // the output-write phase of PANEL_LU.
    config.memory.startup_cycles = 200;
    config.memory.beat_cycles = 5;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "abort-pvector: start");

    std::uint32_t panel_command = cmd::NONE;
    for (const auto& command : artifact.commands) {
        if (command.node_id == 0 && command.opcode == cmd::Opcode::PanelLu) {
            panel_command = command.command_id;
        }
    }
    expect(panel_command != cmd::NONE, "abort-pvector: found PANEL_LU");
    bool aborted = false;
    while (!device.finished() && device.cycle() < 400000) {
        device.tick();
        if (!aborted && device.pending_pvector_command() == panel_command) {
            aborted = true;
            device.abort_node(0);
        }
    }
    expect(aborted, "abort-pvector: aborted during the P-vector write");
    expect(device.control_flow_pass(), "abort-pvector: batch completes");
    expect(
        device.completion_written() == artifact.commands.size(),
        "abort-pvector: every command is terminal");

    const auto panel_status = device.completion_status().at(panel_command);
    expect(
        panel_status == cmd::StatusCode::Aborted,
        "abort-pvector: panel reports ABORTED, not success");
    expect(
        device.scoreboard().state(panel_command) ==
            cp::TokenScoreboard::State::Failed,
        "abort-pvector: panel token is FAILED, not READY");
    bool dependent_failed = false;
    for (const auto& row : device.command_stats()) {
        if (row.node_id == 1 && row.status == "DEPENDENCY_FAILED") {
            dependent_failed = true;
        }
    }
    expect(
        dependent_failed,
        "abort-pvector: dependent front sees the FAILED token");
    bool node0_terminal = true;
    for (const auto& row : device.command_stats()) {
        if (row.node_id == 0 && row.status != "ABORTED" && row.status != "OK" &&
            row.status != "DEPENDENCY_FAILED") {
            node0_terminal = false;
        }
    }
    expect(node0_terminal, "abort-pvector: aborted node's commands are terminal");
}

void test_concurrent_assembly_dma_retry() {
    auto artifact = testing::build_test_artifact(two_chains());
    auto config = base_config();
    config.buffer_port_count = 2;
    config.assemble_unit_count = 2;
    config.stub.latency_by_opcode[cmd::Opcode::PanelLu] = 200;
    config.stub.latency_by_opcode[cmd::Opcode::TrsmLeft] = 200;
    config.stub.latency_by_opcode[cmd::Opcode::TrsmRight] = 200;
    config.stub.latency_by_opcode[cmd::Opcode::GemmSchur] = 200;
    config.memory.startup_cycles = 20;
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    expect(device.start(&error), "retry: start");
    device.run(400000);
    RunResult result = collect(device);
    expect(device.control_flow_pass(), "retry: completes with contended ports");
    expect(
        device.dma_retry_count() > 0,
        "retry: transient DMA contention was re-queued, not failed");
    check_interpreter(artifact, result, {}, {}, "retry");
    check_data(device, artifact, "retry");
}

void test_pivot_count_and_stall_accounting() {
    auto artifact = testing::build_test_artifact(single_node());
    cp::CommandDevice device(artifact.artifact, base_config());
    std::string error;
    expect(device.start(&error), "pivot: start");
    device.run(200000);
    RunResult result = collect(device);
    check_interpreter(artifact, result, {}, {}, "pivot");
    for (const auto& row : device.command_stats()) {
        if (row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::PanelLu))) {
            expect(row.pivot_count == 2, "pivot: success completion carries pivot_dim");
        }
    }

    // stall_cycles must use the mutually exclusive per-tick classes and must
    // not fold the whole window residency in as well.
    auto chain = testing::build_test_artifact(parent_child());
    cp::CommandDevice chain_device(chain.artifact, base_config());
    expect(chain_device.start(&error), "stall: start");
    chain_device.run(200000);
    bool found_dep_wait = false;
    for (const auto& row : chain_device.command_stats()) {
        if (row.node_id == 1 &&
            row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::NodeBegin))) {
            expect(row.stall_dep > 0, "stall: dependency wait is counted");
            expect(
                row.stall_cycles == row.stall_dep + row.stall_resource + row.stall_dma,
                "stall: completion stall_cycles sums only exclusive classes");
            expect(
                row.stall_cycles <= row.stall_window,
                "stall: no double counting of the window residency");
            found_dep_wait = true;
        }
    }
    expect(found_dep_wait, "stall: parent NODE_BEGIN observed");
}

}  // namespace

int main() {
    test_scoreboard();
    test_stub_unit();
    test_backend_gating();
    test_fetch_faults();
    test_single_node_flow();
    test_two_roots_dual_front();
    test_single_unit_contention();
    test_parent_child_assembly();
    test_dynamic_panel_refusal();
    test_failure_propagation();
    test_small_window();
    test_single_front_config();
    test_completion_backpressure();
    test_buffer_capacity_failure();
    test_timeout();
    test_abort_and_reset();
    test_abort_with_inflight_dma();
    test_abort_during_solve_staging();
    test_dma_port_contention_retry();
    test_solve_full_traversal();
    test_start_validation();
    test_default_config_runs();
    test_abort_during_pvector_write();
    test_concurrent_assembly_dma_retry();
    test_pivot_count_and_stall_accounting();

    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
