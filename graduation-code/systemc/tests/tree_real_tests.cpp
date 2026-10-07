#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include "nlohmann/json.hpp"
#include "controlplane/control_plane.hpp"
#include "tree_artifact_check.hpp"

namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace cmd = hw::command;
namespace ref = hw::reference_fp64;
using json = nlohmann::json;
int checks = 0, failures = 0;
void expect(bool ok, const std::string& message) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
cp::ControlPlaneConfig config() {
    cp::ControlPlaneConfig c;
    c.backend = cp::ExecBackend::RequireRealKernels;
    c.enable_real_panel = true;
    c.panel_unit_count = 2;
    c.panel.compute_group_count = 2; c.panel.pes_per_group = 2; c.panel.pe_count = 4;
    c.memory.startup_cycles = 1; c.memory.beat_cycles = 1; c.memory.beat_bytes = 16;
    c.dma.startup_cycles = 1; c.dma.beat_bytes = 16; c.dma.beat_interval_cycles = 1;
    c.assembly.response_hold_cycles = 3;
    return c;
}
void run_case(const json& spec, const std::string& output) {
    const std::string directory = spec.at("device_dir"), name = spec.at("name"), mode = spec.at("mode");
    const std::string label = name + "/" + mode;
    std::ifstream file(directory + "/manifest.json");
    json manifest; file >> manifest;
    cp::CommandDevice device(dp::load_artifact(directory), config());
    std::string error;
    expect(device.start(&error), label + ": starts " + error);
    device.run(1000000);
    expect(device.control_flow_pass(), label + ": every command drains");
    expect(device.numeric_status_pass() == spec.at("numeric_pass").get<bool>(), label + ": numeric status distinct from control flow");
    expect(device.completion_written() == device.boot().command_count, label + ": all completions written");
    const auto report = output + "/" + name + "/" + mode;
    device.write_reports(report);
    const auto validation = ref::check_tree_artifact(directory, report);
    expect(validation.pass == spec.at("numeric_pass").get<bool>(), label + ": independent FP64 checker: " + validation.reason);
    if (validation.pass) {
        expect(validation.nodes_checked == device.boot().node_count, label + ": all L/U/P/update outputs checked");
    }
    std::map<std::uint32_t, const cp::CommandStats*> begin, commit, stored, panel;
    std::uint32_t assembly_count = 0;
    for (const auto& stats : device.command_stats()) {
        const auto opcode = static_cast<cmd::Opcode>(std::stoul(stats.opcode));
        if (opcode == cmd::Opcode::NodeBegin) begin[stats.node_id] = &stats;
        if (opcode == cmd::Opcode::NodeCommit) commit[stats.node_id] = &stats;
        if (opcode == cmd::Opcode::StoreUpdate) stored[stats.node_id] = &stats;
        if (opcode == cmd::Opcode::PanelLu) panel[stats.node_id] = &stats;
        expect(opcode != cmd::Opcode::SolveForward && opcode != cmd::Opcode::SolveBackward,
               label + ": factorization artifact never dispatches solve");
        if (opcode == cmd::Opcode::AssembleExtendAdd) {
            ++assembly_count;
            expect(stats.status == "OK" && stats.unit_kind == "assemble" && stats.bytes_written > 0,
                   label + ": real assembly performs counted writes");
            if (panel.count(stats.node_id)) {
                expect(panel.at(stats.node_id)->start_cycle >= stats.completion_cycle,
                       label + ": panel starts after assembly completion");
            }
        }
    }
    std::uint32_t edges = 0;
    for (std::uint32_t i = 0; i < device.boot().node_count; ++i) {
        const int parent = manifest.at("nodes").at(std::to_string(i)).at("parent_id");
        if (parent >= 0) {
            ++edges;
            expect(begin.at(parent)->start_cycle >= commit.at(i)->completion_cycle,
                   label + ": parent admission follows child commit acknowledgement");
        }
        expect(device.front_slot(i).slot == cmd::NONE, label + ": front released at commit/failure");
    }
    expect(assembly_count == edges, label + ": one contribution command per tree edge");
    for (const auto& stats : device.command_stats()) {
        if (static_cast<cmd::Opcode>(std::stoul(stats.opcode)) != cmd::Opcode::AssembleExtendAdd) continue;
        expect(panel.at(stats.node_id)->start_cycle >= stats.completion_cycle,
               label + ": every child addition visible before panel scan");
    }
    expect(device.dataplane().buffers().active_lease_count() == 0, label + ": no leaked leases");
    for (std::uint32_t i = 0; i < 5; ++i)
        expect(device.dataplane().buffers().state(i) == dp::BufferState::Free, label + ": scratch slots free");
    if (device.boot().node_count > 2) {
        bool reused = false;
        for (std::uint32_t i = 0; i < 5; ++i) reused = reused || device.dataplane().buffers().generation(i) > 1;
        expect(reused, label + ": physical buffer generations show reuse");
    }
    if (name == "late_child0" || name == "late_child1") {
        const std::uint32_t slow = name == "late_child0" ? 0 : 1;
        expect(commit.at(1 - slow)->completion_cycle < commit.at(slow)->completion_cycle,
               label + ": either sibling may complete first");
    }
    if (name == "pivot_propagation" && mode == "int32") {
        expect(device.completion_status().at(panel.at(1)->command_id) == cmd::StatusCode::StaticPivotFailure &&
               device.completion_status().at(commit.at(1)->command_id) != cmd::StatusCode::Ok,
               label + ": real DDR/extend-add propagation changes actual STATIC pivot outcome");
    }
    if (name == "two_level" && mode == "fp32") {
        // Mutating device output must break the independent check. Restore it
        // afterward so the saved final artifact remains reviewable.
        auto corrupted = device.dataplane().memory().image();
        const auto* l = device.boot().region("node.1.factor_l");
        corrupted[l->offset] ^= 0x40;
        corrupted[l->offset + 3] ^= 0x40;
        {
            std::ofstream out(report + "/final_memory_image.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(corrupted.data()), corrupted.size());
        }
        expect(!ref::check_tree_artifact(directory, report).pass, label + ": checker detects corrupted output");
        device.write_reports(report);
        expect(ref::check_tree_artifact(directory, report).pass, label + ": restored output revalidates");
    }
    std::cout << "[t11] " << label << " cycles=" << device.cycle()
              << " nodes=" << device.boot().node_count << " checker_error=" << validation.max_relative_error << '\n';
}

void abort_reset(const std::string& directory) {
    for (const bool reset : {false, true}) {
        cp::CommandDevice device(dp::load_artifact(directory), config());
        std::string error;
        expect(device.start(&error), "assembly abort/reset device starts");
        bool interrupted = false;
        std::uint32_t assembly_command = cmd::NONE;
        while (!device.finished() && device.cycle() < 1000000) {
            device.tick();
            if (device.unit_active_command_id(cp::UnitKind::Assemble, 0) == cmd::NONE) continue;
            interrupted = true;
            assembly_command = device.unit_active_command_id(cp::UnitKind::Assemble, 0);
            for (unsigned i = 0; i < 8; ++i) device.tick();
            if (reset) {
                device.reset();
                expect(device.start(&error), "reset restarts a clean epoch");
            } else device.abort_node(2);
            break;
        }
        expect(interrupted, "interrupted during real child assembly");
        device.run(1000000);
        expect(device.control_flow_pass() && device.numeric_status_pass() == reset,
               "abort/reset drains without late success or deadlock");
        expect(device.dataplane().buffers().active_lease_count() == 0,
               "assembly abort/reset releases both leases");
        if (!reset) {
            for (const auto& row : device.command_stats()) {
                if (row.command_id == assembly_command)
                    expect(row.status == "ABORTED" && row.bytes_read > 17 * 17 * 4,
                           "assembly abort preserves real scratchpad work in command statistics");
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream file(std::string(argv[1]) + "/cases.json");
    json cases; file >> cases;
    for (const auto& spec : cases) run_case(spec, argv[2]);
    abort_reset(std::string(argv[1]) + "/mapped_tail/fp32/device");
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
