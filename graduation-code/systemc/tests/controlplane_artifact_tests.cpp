// T05 real-artifact integration: load schema v2 artifacts produced by the
// current software compiler (device dirs contain only manifest + image), run
// the command control plane with the explicitly enabled test stub, and compare
// every command's terminal status against the independent interpreter.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

std::vector<cmd::Command> decode_commands(const dp::LoadedArtifact& artifact) {
    std::vector<cmd::Command> commands;
    const auto* region = artifact.config.region("command_buffer");
    for (std::uint32_t index = 0; index < artifact.config.command_count; ++index) {
        commands.push_back(cmd::CommandCodec::decode(
            artifact.image.data() + region->offset +
                static_cast<std::uint64_t>(index) * cmd::COMMAND_RECORD_BYTES,
            cmd::COMMAND_RECORD_BYTES));
    }
    return commands;
}

std::vector<cmd::Descriptor> decode_descriptors(const dp::LoadedArtifact& artifact) {
    std::vector<cmd::Descriptor> descriptors;
    const auto* region = artifact.config.region("descriptor_table");
    for (std::uint32_t index = 0; index < artifact.config.descriptor_count; ++index) {
        descriptors.push_back(cmd::DescriptorCodec::decode(
            artifact.image.data() + region->offset +
                static_cast<std::uint64_t>(index) * cmd::DESCRIPTOR_RECORD_BYTES,
            cmd::DESCRIPTOR_RECORD_BYTES, artifact.config.total_bytes));
    }
    return descriptors;
}

cp::ControlPlaneConfig stub_config() {
    cp::ControlPlaneConfig config;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 2;
    config.trsm_unit_count = 2;
    config.gemm_unit_count = 2;
    config.solve_unit_count = 2;
    config.assemble_unit_count = 2;
    config.buffer_slot_count = 5;  // 2 front + 2 assemble temp + 1 solve scratch
    config.buffer_capacity_bytes = 64 * 1024;
    config.buffer_port_count = 4;
    config.command_window_depth = 32;
    config.completion_queue_depth = 32;
    config.dma_queue_depth = 8;
    config.backend = cp::ExecBackend::TestStub;
    config.enable_test_stub = true;
    config.stub.default_latency = 3;
    config.memory.startup_cycles = 1;
    config.memory.beat_bytes = 32;
    config.memory.beat_cycles = 1;
    config.memory.max_inflight = 8;
    config.dma.beat_bytes = 32;
    config.dma.startup_cycles = 1;
    config.dma.beat_interval_cycles = 1;
    return config;
}

std::vector<std::uint8_t> identity_pvector(std::uint32_t dim) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(dim) * 4, 0);
    for (std::uint32_t index = 0; index < dim; ++index) {
        for (unsigned b = 0; b < 4; ++b) {
            out[static_cast<std::size_t>(index) * 4 + b] =
                static_cast<std::uint8_t>((index >> (8 * b)) & 0xFF);
        }
    }
    return out;
}

void verify_data(cp::CommandDevice& device, const dp::LoadedArtifact& artifact,
                 const std::string& label) {
    const auto& image = device.dataplane().memory().image();
    auto ends_with = [](const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    for (const auto& info : device.boot().regions) {
        const bool writable_output =
            ends_with(info.name, ".factor_l") || ends_with(info.name, ".factor_u") ||
            ends_with(info.name, ".update") || ends_with(info.name, ".p_vector") ||
            info.name == "solution_data" || info.name == "completion_status";
        if (writable_output) {
            continue;
        }
        const std::vector<std::uint8_t> before(
            artifact.image.begin() + static_cast<std::ptrdiff_t>(info.offset),
            artifact.image.begin() + static_cast<std::ptrdiff_t>(info.offset + info.size));
        const std::vector<std::uint8_t> after(
            image.begin() + static_cast<std::ptrdiff_t>(info.offset),
            image.begin() + static_cast<std::ptrdiff_t>(info.offset + info.size));
        expect(before == after, label + ": region unchanged: " + info.name);
    }
    // STATIC success contract: identity P-vectors.
    for (const auto& info : device.boot().regions) {
        if (info.name.size() < 9 ||
            info.name.compare(info.name.size() - 9, 9, ".p_vector") != 0) {
            continue;
        }
        const std::uint32_t dim = static_cast<std::uint32_t>(info.size / 4);
        const std::vector<std::uint8_t> actual(
            image.begin() + static_cast<std::ptrdiff_t>(info.offset),
            image.begin() + static_cast<std::ptrdiff_t>(info.offset + info.size));
        expect(
            actual == identity_pvector(dim),
            label + ": identity P-vector written: " + info.name);
    }
}

void run_case(const std::filesystem::path& device_dir, const std::string& label) {
    dp::LoadedArtifact artifact;
    try {
        artifact = dp::load_artifact(device_dir.string());
    } catch (const std::exception& exception) {
        ++failures;
        std::cerr << "FAIL: " << label << ": load: " << exception.what() << '\n';
        return;
    }
    const auto commands = decode_commands(artifact);
    const auto descriptors = decode_descriptors(artifact);
    const bool dynamic =
        artifact.config.pivot_policy == cmd::PivotPolicy::DynamicPanel;

    cp::CommandDevice device(artifact, stub_config());
    std::string error;
    expect(device.start(&error), label + ": batch starts");
    device.run(5000000);
    expect(device.control_flow_pass(), label + ": control flow completes");
    expect(
        device.completion_written() == commands.size(),
        label + ": one completion per command");

    const auto expected = testing::interpret(commands, descriptors, artifact.image,
                                             static_cast<std::uint32_t>(commands.size()));
    std::map<std::uint32_t, std::uint32_t> pivot_counts;
    for (const auto& row : device.command_stats()) {
        pivot_counts[row.command_id] = row.pivot_count;
    }
    bool status_ok = true;
    for (const auto& command : commands) {
        const auto found = device.completion_status().find(command.command_id);
        if (found == device.completion_status().end() ||
            found->second != expected.status.at(command.command_id)) {
            status_ok = false;
            std::cerr << "  command " << command.command_id << " node "
                      << command.node_id << " op "
                      << static_cast<unsigned>(command.opcode) << " got "
                      << dp::status_name(
                             found == device.completion_status().end()
                                 ? cmd::StatusCode::Timeout
                                 : found->second)
                      << " expected "
                      << dp::status_name(expected.status.at(command.command_id))
                      << "\n";
        }
        const auto pivot = pivot_counts.find(command.command_id);
        if (pivot == pivot_counts.end() ||
            pivot->second != expected.pivot_count.at(command.command_id)) {
            status_ok = false;
            std::cerr << "  command " << command.command_id << " pivot_count mismatch\n";
        }
    }
    expect(status_ok, label + ": statuses and pivot counts match the interpreter");

    if (dynamic) {
        std::uint32_t refused = 0;
        for (const auto& row : device.command_stats()) {
            if (row.status == "POLICY_NOT_SUPPORTED") {
                ++refused;
            }
        }
        expect(refused > 0, label + ": DYNAMIC_PANEL PANEL_LU refused");
    } else {
        verify_data(device, artifact, label);
    }
    std::cout << label << ": commands=" << commands.size()
              << " cycles=" << device.cycle()
              << " completions=" << device.completion_written() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: controlplane_artifact_tests FIXTURES_DIR\n";
        return 2;
    }
    const std::filesystem::path fixtures = argv[1];

    for (const std::string name :
         {"single_static", "multi_static", "multiroot_static", "dynamic_panel",
          "large_256"}) {
        const auto device_dir = fixtures / name / "device";
        if (!std::filesystem::exists(device_dir / "manifest.json")) {
            ++failures;
            std::cerr << "FAIL: missing device dir for " << name << '\n';
            continue;
        }
        run_case(device_dir, name);
    }

    // Failure propagation on a real artifact.
    {
        dp::LoadedArtifact artifact =
            dp::load_artifact((fixtures / "multiroot_static" / "device").string());
        const auto commands = decode_commands(artifact);
        const auto descriptors = decode_descriptors(artifact);
        std::uint32_t panel_command = cmd::NONE;
        for (const auto& command : commands) {
            if (command.opcode == cmd::Opcode::PanelLu && panel_command == cmd::NONE) {
                panel_command = command.command_id;
            }
        }
        expect(panel_command != cmd::NONE, "injection: found a PANEL_LU command");
        const std::map<std::uint32_t, testing::RefFailure> injected = {
            {panel_command, {cmd::StatusCode::StaticPivotFailure, 1}}};
        cp::CommandDevice device(artifact, stub_config());
        device.inject_failure(
            panel_command, cp::ExecFailure{cmd::StatusCode::StaticPivotFailure, 1, 0});
        std::string error;
        expect(device.start(&error), "injection: batch starts");
        device.run(5000000);
        expect(device.control_flow_pass(), "injection: control flow completes");
        const auto expected = testing::interpret(
            commands, descriptors, artifact.image,
            static_cast<std::uint32_t>(commands.size()), injected);
        bool status_ok = true;
        for (const auto& command : commands) {
            const auto found = device.completion_status().find(command.command_id);
            if (found == device.completion_status().end() ||
                found->second != expected.status.at(command.command_id)) {
                status_ok = false;
                std::cerr << "  command " << command.command_id << " mismatch\n";
            }
        }
        expect(status_ok, "injection: failure propagation matches the interpreter");
    }

    // Control-flow cycle comparison on a real artifact with two independent
    // fronts (multi-root forest). A single chain shows no difference: with one
    // shared DDR port and no independent work, residency cannot help.
    {
        dp::LoadedArtifact artifact =
            dp::load_artifact((fixtures / "multiroot_static" / "device").string());
        std::uint64_t cycles_single = 0;
        std::uint64_t cycles_dual = 0;
        {
            auto config = stub_config();
            config.max_resident_fronts = 1;
            config.buffer_slot_count = 3;
            config.panel_unit_count = 1;
            config.trsm_unit_count = 1;
            config.gemm_unit_count = 1;
            config.solve_unit_count = 1;
            config.assemble_unit_count = 1;
            config.stub.default_latency = 100;
            cp::CommandDevice device(artifact, config);
            std::string error;
            device.start(&error);
            device.run(5000000);
            expect(device.control_flow_pass(), "cycles: single-front real artifact");
            cycles_single = device.cycle();
        }
        {
            auto config = stub_config();
            config.stub.default_latency = 100;
            cp::CommandDevice device(artifact, config);
            std::string error;
            device.start(&error);
            device.run(5000000);
            expect(device.control_flow_pass(), "cycles: dual-front real artifact");
            cycles_dual = device.cycle();
        }
        expect(
            cycles_dual <= cycles_single,
            "cycles: dual front is not slower on the real artifact");
        std::cout << "multiroot_static cycle comparison "
                  << "(execution_backend=test_stub, latency=100): "
                  << "1front/1unit=" << cycles_single
                  << " 2front/2unit=" << cycles_dual << '\n';
    }

    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
