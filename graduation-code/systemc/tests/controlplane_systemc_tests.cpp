// SystemC timing tests for the T05 command control plane. The CommandDevice
// is a plain C++ transaction model; the SystemC testbench owns the clock and
// advances exactly one device tick per posedge.

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <systemc>

#include "controlplane/control_plane.hpp"
#include "control_reference.hpp"
#include "control_test_artifact.hpp"

namespace {

namespace cmd = hw::command;
namespace cp = hw::controlplane;
namespace testing = hw::testing;

class T05SystemCTests : public sc_core::sc_module {
public:
    sc_core::sc_in<bool> clk;

    SC_CTOR(T05SystemCTests) { SC_THREAD(run); }

    int failures() const { return failures_; }

private:
    int failures_{0};
    int checks_{0};

    void check(bool condition, const std::string& message) {
        ++checks_;
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void wait_step(cp::CommandDevice& device) {
        wait(clk.posedge_event());
        device.tick();
    }

    void run_device(cp::CommandDevice& device, std::uint64_t max_cycles) {
        const std::uint64_t start = device.cycle();
        while (!device.finished() && device.cycle() - start < max_cycles) {
            wait_step(device);
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
        config.dma.beat_bytes = 16;
        config.dma.startup_cycles = 1;
        config.dma.beat_interval_cycles = 1;
        return config;
    }

    std::map<std::uint32_t, cmd::StatusCode> status_map(cp::CommandDevice& device) {
        return device.completion_status();
    }

    void check_interpreter(
        const testing::TestArtifact& artifact,
        cp::CommandDevice& device,
        const std::string& label) {
        const auto expected = testing::interpret(
            artifact.commands, artifact.descriptors, artifact.artifact.image,
            static_cast<std::uint32_t>(artifact.commands.size()));
        bool ok = true;
        for (const auto& entry : device.completion_status()) {
            const auto want = expected.status.at(entry.first);
            if (entry.second != want) {
                ok = false;
                std::cerr << "  command " << entry.first << " status "
                          << hw::dataplane::status_name(entry.second) << " expected "
                          << hw::dataplane::status_name(want) << '\n';
            }
        }
        check(
            ok && device.completion_status().size() == artifact.commands.size(),
            label + ": statuses match the interpreter");
    }

    bool intervals_overlap(
        const std::vector<cp::UnitBusyRecord>& records,
        const std::string& kind,
        bool require_distinct_units) {
        for (std::size_t i = 0; i < records.size(); ++i) {
            for (std::size_t j = i + 1; j < records.size(); ++j) {
                const bool unit_ok = require_distinct_units
                                         ? records[i].unit_id != records[j].unit_id
                                         : records[i].unit_id == records[j].unit_id;
                if (records[i].kind == kind && records[j].kind == kind && unit_ok &&
                    records[i].node_id != records[j].node_id &&
                    records[i].start < records[j].finish &&
                    records[j].start < records[i].finish) {
                    return true;
                }
            }
        }
        return false;
    }

    void run() {
        test_clock_driven_batch();
        test_dual_front_overlap();
        test_cycle_comparison();
        test_window_and_completion_backpressure();
        test_abort_and_reset();

        std::cout << checks_ << " checks, " << failures_ << " failures\n";
        std::cout << "cycle comparison (test_stub latencies): "
                  << comparison_report_ << '\n';
        sc_core::sc_stop();
    }

    std::string comparison_report_;

    void test_clock_driven_batch() {
        auto artifact = testing::build_test_artifact(
            std::vector<testing::TestNode>{
                testing::TestNode{0, 1, 2, 1, {}},
                testing::TestNode{1, -1, 2, 2, {{0, {1}}}}});
        cp::CommandDevice device(artifact.artifact, base_config());
        std::string error;
        check(device.start(&error), "clock: batch starts");
        run_device(device, 200000);
        check(device.control_flow_pass(), "clock: control flow completes");
        check(
            device.completion_written() == artifact.commands.size(),
            "clock: exactly one completion per command");
        check_interpreter(artifact, device, "clock");
    }

    void test_dual_front_overlap() {
        auto artifact = testing::build_test_artifact(
            std::vector<testing::TestNode>{
                testing::TestNode{0, -1, 3, 2, {}},
                testing::TestNode{1, -1, 3, 2, {}}});
        auto config = base_config();
        config.panel_unit_count = 2;
        config.gemm_unit_count = 2;
        config.stub.latency_by_opcode[cmd::Opcode::PanelLu] = 300;
        config.stub.latency_by_opcode[cmd::Opcode::TrsmLeft] = 300;
        config.stub.latency_by_opcode[cmd::Opcode::TrsmRight] = 300;
        config.stub.latency_by_opcode[cmd::Opcode::GemmSchur] = 300;
        cp::CommandDevice device(artifact.artifact, config);
        std::string error;
        check(device.start(&error), "overlap: batch starts");
        run_device(device, 200000);
        check(device.control_flow_pass(), "overlap: control flow completes");
        check_interpreter(artifact, device, "overlap");

        const auto& units = device.unit_records();
        check(intervals_overlap(units, "panel", true),
              "overlap: two panel units active on different fronts");
        check(intervals_overlap(units, "gemm", true),
              "overlap: two GEMM instances active at the same time");
        check(intervals_overlap(units, "trsm", true),
              "overlap: two TRSM units active on different fronts");

        // Front residency intervals of the two nodes overlap.
        bool front_overlap = false;
        const auto& fronts = device.front_records();
        for (std::size_t i = 0; i < fronts.size(); ++i) {
            for (std::size_t j = i + 1; j < fronts.size(); ++j) {
                if (fronts[i].node_id != fronts[j].node_id &&
                    fronts[i].start < fronts[j].finish &&
                    fronts[j].start < fronts[i].finish) {
                    front_overlap = true;
                }
            }
        }
        check(front_overlap, "overlap: both fronts were resident at the same time");

        std::cout << "overlap evidence (node/command/unit intervals):\n";
        for (const auto& record : units) {
            if (record.kind == "panel" || record.kind == "gemm") {
                std::cout << "  " << record.kind << " unit=" << record.unit_id
                          << " command=" << record.command_id
                          << " node=" << record.node_id << " [" << record.start
                          << "," << record.finish << ")\n";
            }
        }
    }

    void test_cycle_comparison() {
        auto artifact = testing::build_test_artifact(
            std::vector<testing::TestNode>{
                testing::TestNode{0, -1, 3, 2, {}},
                testing::TestNode{1, -1, 3, 2, {}}});
        std::uint64_t cycles_single = 0;
        std::uint64_t cycles_dual_shared = 0;
        std::uint64_t cycles_dual_dup = 0;
        std::map<std::uint32_t, cmd::StatusCode> statuses;

        {
            auto config = base_config();
            config.max_resident_fronts = 1;
            config.buffer_slot_count = 3;
            config.panel_unit_count = 1;
            config.trsm_unit_count = 1;
            config.gemm_unit_count = 1;
            config.stub.default_latency = 20;
            cp::CommandDevice device(artifact.artifact, config);
            std::string error;
            check(device.start(&error), "cycles: single-front starts");
            run_device(device, 200000);
            check(device.control_flow_pass(), "cycles: single-front completes");
            cycles_single = device.cycle();
            statuses = status_map(device);
        }
        {
            auto config = base_config();
            config.panel_unit_count = 1;
            config.trsm_unit_count = 1;
            config.gemm_unit_count = 1;
            config.stub.default_latency = 20;
            cp::CommandDevice device(artifact.artifact, config);
            std::string error;
            check(device.start(&error), "cycles: dual-front shared units starts");
            run_device(device, 200000);
            check(device.control_flow_pass(), "cycles: dual-front shared units completes");
            cycles_dual_shared = device.cycle();
            check(
                status_map(device) == statuses,
                "cycles: control-flow statuses identical across configs");
        }
        {
            auto config = base_config();
            config.panel_unit_count = 2;
            config.trsm_unit_count = 2;
            config.gemm_unit_count = 2;
            config.stub.default_latency = 20;
            cp::CommandDevice device(artifact.artifact, config);
            std::string error;
            check(device.start(&error), "cycles: dual-front dual units starts");
            run_device(device, 200000);
            check(device.control_flow_pass(), "cycles: dual-front dual units completes");
            cycles_dual_dup = device.cycle();
            check(
                status_map(device) == statuses,
                "cycles: control-flow statuses identical with duplicated units");
        }
        check(
            cycles_dual_shared <= cycles_single,
            "cycles: two resident fronts never cost more cycles than one");
        check(
            cycles_dual_dup <= cycles_dual_shared,
            "cycles: duplicated units never cost more cycles than shared units");
        comparison_report_ =
            "1front/1unit=" + std::to_string(cycles_single) +
            " 2front/1unit=" + std::to_string(cycles_dual_shared) +
            " 2front/2unit=" + std::to_string(cycles_dual_dup);
    }

    void test_window_and_completion_backpressure() {
        auto artifact = testing::build_test_artifact(
            std::vector<testing::TestNode>{
                testing::TestNode{0, 1, 2, 1, {}},
                testing::TestNode{1, -1, 2, 2, {{0, {1}}}}});
        auto config = base_config();
        config.command_window_depth = 2;
        config.completion_queue_depth = 1;
        cp::CommandDevice device(artifact.artifact, config);
        std::string error;
        check(device.start(&error), "window: batch starts");
        run_device(device, 200000);
        check(device.control_flow_pass(), "window: completes with depth 2 window");
        check(
            device.completion_written() == artifact.commands.size(),
            "window: completion backpressure loses nothing");
        check_interpreter(artifact, device, "window");
    }

    void test_abort_and_reset() {
        auto artifact = testing::build_test_artifact(
            std::vector<testing::TestNode>{
                testing::TestNode{0, -1, 3, 2, {}},
                testing::TestNode{1, -1, 3, 2, {}}});
        cp::CommandDevice device(artifact.artifact, base_config());
        std::string error;
        check(device.start(&error), "abort: batch starts");
        bool aborted = false;
        while (!device.finished() && device.cycle() < 200000) {
            wait_step(device);
            if (!aborted && device.command_stats().size() >= 2) {
                aborted = true;
                device.abort_node(0);
            }
        }
        check(device.control_flow_pass(), "abort: batch completes");
        bool node1_block_ok = true;
        for (const auto& row : device.command_stats()) {
            if (row.node_id == 1 && row.opcode != "10" && row.opcode != "11" &&
                row.status != "OK") {
                node1_block_ok = false;
            }
        }
        check(node1_block_ok, "abort: independent front is unaffected");

        device.reset();
        check(device.state() == cp::BatchState::Idle, "reset: back to idle");
        check(device.start(&error), "reset: restart");
        run_device(device, 200000);
        check(device.control_flow_pass(), "reset: restarted batch completes");
    }
};

}  // namespace

int sc_main(int, char**) {
    sc_core::sc_clock clock("clk", sc_core::sc_time(1, sc_core::SC_NS));
    T05SystemCTests tests("t05_systemc_tests");
    tests.clk(clock);
    sc_core::sc_start();
    return tests.failures() == 0 ? 0 : 1;
}
