// T12 solve integration over the 24 T11 tree fixtures compiled with the full
// factorization_and_solve scope (both backends).
//
// Per fixture:
//   - the device drains; every command has exactly one completion;
//   - numeric_status_pass matches the fixture expectation; the expected
//     factorization failure (INT32 pivot propagation) leaves every backward
//     and every forward from the failed node on DEPENDENCY_FAILED (earlier
//     forwards may be OK; no global solve gate since 2026-10-11) and the
//     device exit status non-zero;
//   - rhs_data is byte-identical before and after the run;
//   - tree_solve_check reports solve_pass, and for good fixtures the
//     transformed-system backward error is <= 1e-5 while the solution matches
//     the independent C++ FP64 reference within 1e-4;
//   - the summary carries the solve aggregate and SolveConfig parameters;
//   - leases and scratch slots are fully released.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "controlplane/control_plane.hpp"
#include "tree_solve_check.hpp"

namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace cmd = hw::command;
namespace ref = hw::reference_fp64;
using json = nlohmann::json;

int checks = 0;
int failures = 0;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

cp::ControlPlaneConfig config() {
    cp::ControlPlaneConfig c;
    c.backend = cp::ExecBackend::RequireRealKernels;
    c.enable_real_panel = true;
    c.enable_real_solve = true;
    c.solve_unit_count = 1;
    c.panel_unit_count = 2;
    c.panel.compute_group_count = 2;
    c.panel.pes_per_group = 2;
    c.panel.pe_count = 4;
    c.memory.startup_cycles = 1;
    c.memory.beat_cycles = 1;
    c.memory.beat_bytes = 16;
    c.dma.startup_cycles = 1;
    c.dma.beat_bytes = 16;
    c.dma.beat_interval_cycles = 1;
    c.assembly.response_hold_cycles = 3;
    return c;
}

bool is_solve_opcode(const std::string& opcode) {
    return opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::SolveForward)) ||
           opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::SolveBackward));
}

void run_case(const json& spec, const std::string& output_root) {
    const std::string directory = spec.at("device_dir");
    const std::string name = spec.at("name");
    const std::string mode = spec.at("mode");
    const std::string label = name + "/" + mode;
    const bool expected_pass = spec.at("solve_numeric_pass").get<bool>();

    auto artifact = dp::load_artifact(directory);
    const std::vector<std::uint8_t> initial_image = artifact.image;
    cp::CommandDevice device(std::move(artifact), config());
    std::string error;
    expect(device.start(&error), label + ": starts: " + error);
    device.run(5000000);
    expect(device.control_flow_pass(), label + ": every command drains");
    expect(
        device.numeric_status_pass() == expected_pass,
        label + ": device status matches the fixture expectation");
    expect(
        device.completion_written() == device.boot().command_count,
        label + ": every command has exactly one completion");

    // rhs_data must be byte-identical (the solve only reads it).
    const auto* rhs = device.boot().region("rhs_data");
    expect(rhs != nullptr, label + ": rhs_data region exists");
    if (rhs != nullptr) {
        const auto& image = device.dataplane().memory().image();
        const bool unchanged = std::equal(
            initial_image.begin() + static_cast<std::ptrdiff_t>(rhs->offset),
            initial_image.begin() + static_cast<std::ptrdiff_t>(rhs->offset + rhs->size),
            image.begin() + static_cast<std::ptrdiff_t>(rhs->offset));
        expect(unchanged, label + ": rhs_data is byte-identical after the run");
    }

    // Solve command statuses.
    std::uint32_t solve_commands = 0;
    std::uint32_t solve_ok = 0;
    std::uint32_t solve_failed = 0;
    for (const auto& stats : device.command_stats()) {
        if (!is_solve_opcode(stats.opcode)) {
            continue;
        }
        ++solve_commands;
        if (stats.status == "OK") {
            ++solve_ok;
        } else {
            ++solve_failed;
            expect(
                stats.status == "DEPENDENCY_FAILED",
                label + ": a failed solve command is DEPENDENCY_FAILED, got " +
                    stats.status + " (" + stats.reason + ")");
        }
    }
    expect(solve_commands == 2 * device.boot().node_count, label + ": one solve pair per node");
    expect(
        (solve_ok == solve_commands) == expected_pass,
        label + ": solve command statuses follow the expectation");
    if (!expected_pass) {
        // Failure semantics without a global solve gate (T12 design doc §4.4,
        // revised 2026-10-11): forwards before the failed node may be OK; from
        // the first non-OK forward on, every forward and every backward is
        // DEPENDENCY_FAILED.
        std::vector<const cp::CommandStats*> forwards;
        std::vector<const cp::CommandStats*> backwards;
        const std::string forward_opcode =
            std::to_string(static_cast<std::uint32_t>(cmd::Opcode::SolveForward));
        for (const auto& stats : device.command_stats()) {
            if (stats.opcode == forward_opcode) {
                forwards.push_back(&stats);
            } else if (is_solve_opcode(stats.opcode)) {
                backwards.push_back(&stats);
            }
        }
        const auto by_id = [](const cp::CommandStats* a, const cp::CommandStats* b) {
            return a->command_id < b->command_id;
        };
        std::sort(forwards.begin(), forwards.end(), by_id);
        std::sort(backwards.begin(), backwards.end(), by_id);
        bool seen_failed_forward = false;
        for (const auto* stats : forwards) {
            if (stats->status != "OK") {
                seen_failed_forward = true;
            }
            if (seen_failed_forward) {
                expect(
                    stats->status == "DEPENDENCY_FAILED",
                    label + ": forward after the first non-OK forward is DEPENDENCY_FAILED (command " +
                        std::to_string(stats->command_id) + " got " + stats->status + ")");
            }
        }
        expect(seen_failed_forward, label + ": some forward is not OK");
        for (const auto* stats : backwards) {
            expect(
                stats->status == "DEPENDENCY_FAILED",
                label + ": every backward is DEPENDENCY_FAILED (command " +
                    std::to_string(stats->command_id) + " got " + stats->status + ")");
        }
        expect(!device.numeric_status_pass(), label + ": device numeric_status_pass is false");
    }

    const std::string report = output_root + "/" + name + "/" + mode;
    device.write_reports(report);
    const auto validation = ref::check_tree_solve_artifact(directory, report);
    expect(
        validation.pass == expected_pass,
        label + ": independent solve checker: " + validation.reason);
    if (expected_pass) {
        expect(
            validation.nodes_checked == device.boot().node_count,
            label + ": every node was checked");
        expect(
            validation.eta_forward <= 1e-4 && validation.eta_backward <= 1e-4,
            label + ": eta_f / eta_b within the triangular tolerance");
        expect(
            validation.transformed_backward_error <= 1e-5,
            label + ": transformed-system backward error <= 1e-5 (got " +
                std::to_string(validation.transformed_backward_error) + ")");
        expect(
            validation.reference_solution_relative_error <= 1e-4,
            label + ": solution matches the FP64 reference <= 1e-4 (got " +
                std::to_string(validation.reference_solution_relative_error) + ")");
    }

    // Summary carries the solve aggregate and SolveConfig parameters.
    std::ifstream summary_stream(report + "/summary.json");
    json summary;
    summary_stream >> summary;
    const auto& solve = summary.at("solve_execution");
    expect(solve.at("enable_real_solve").get<bool>(), label + ": summary marks the real solve");
    expect(
        solve.at("commands").get<std::uint32_t>() == solve_commands,
        label + ": summary counts every solve command");
    expect(
        solve.at("completed").get<std::uint32_t>() == solve_ok,
        label + ": summary solve completion count");
    expect(
        solve.at("failed").get<std::uint32_t>() == solve_failed,
        label + ": summary solve failure count");
    if (expected_pass) {
        expect(solve.at("cycles").get<std::uint64_t>() > 0, label + ": solve cycles recorded");
        expect(
            solve.at("bytes_read").get<std::uint64_t>() > 0 &&
                solve.at("bytes_written").get<std::uint64_t>() > 0,
            label + ": solve bytes recorded");
    }
    const auto& solve_config = solve.at("config");
    expect(
        solve_config.at("lanes").get<std::uint32_t>() == 16 &&
            solve_config.at("row_buffer_depth").get<std::uint32_t>() == 2,
        label + ": SolveConfig parameters are written to the summary");

    // Resources are released even after failures.
    expect(
        device.dataplane().buffers().active_lease_count() == 0,
        label + ": no leaked leases");
    for (std::uint32_t slot = 0; slot < 5; ++slot) {
        expect(
            device.dataplane().buffers().state(slot) == dp::BufferState::Free,
            label + ": scratch slots free");
    }
    for (std::uint32_t node = 0; node < device.boot().node_count; ++node) {
        expect(device.front_slot(node).slot == cmd::NONE, label + ": front released");
    }
    expect(
        device.numeric_status_pass() == expected_pass,
        label + ": final device status is stable");
    std::cout << "[t12] " << label << " cycles=" << device.cycle()
              << " solve=" << solve_ok << "/" << solve_commands
              << " eta_f=" << validation.eta_forward
              << " eta_b=" << validation.eta_backward
              << " tback=" << validation.transformed_backward_error
              << " ref_err=" << validation.reference_solution_relative_error
              << '\n';
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tree_solve_real_tests FIXTURE_DIR OUTPUT_DIR\n";
        return 2;
    }
    std::ifstream file(std::string(argv[1]) + "/cases.json");
    json cases;
    file >> cases;
    for (const auto& spec : cases) {
        run_case(spec, argv[2]);
    }
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
