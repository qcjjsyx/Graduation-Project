// Reproducible entry point for the T05 command control plane: runs a schema v2
// device artifact through the command fetch / scheduling / completion path with
// the explicitly enabled test execution stub and writes machine-readable
// reports. The stub performs no numerical kernels: results carry
// execution_backend=test_stub and numerical_validation=not_performed.

#include <cstdlib>
#include <iostream>
#include <string>

#include "controlplane/control_plane.hpp"
#include "dataplane/artifact_loader.hpp"

namespace {

void usage(const char* program) {
    std::cerr << "usage: " << program
              << " --device-dir DIR --out DIR [--max-cycles N]\n"
              << "       [--max-resident-fronts N] [--window N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string device_dir;
    std::string out_dir;
    std::uint64_t max_cycles = 5000000;
    hw::controlplane::ControlPlaneConfig config;
    config.backend = hw::controlplane::ExecBackend::TestStub;
    config.enable_test_stub = true;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 2;
    config.trsm_unit_count = 2;
    config.gemm_unit_count = 2;
    config.solve_unit_count = 2;
    config.assemble_unit_count = 2;
    config.buffer_slot_count = 5;
    config.buffer_capacity_bytes = 64 * 1024;
    config.buffer_port_count = 4;
    config.command_window_depth = 32;
    config.completion_queue_depth = 32;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                std::exit(2);
            }
            return argv[++index];
        };
        if (argument == "--device-dir") {
            device_dir = next("--device-dir");
        } else if (argument == "--out") {
            out_dir = next("--out");
        } else if (argument == "--max-cycles") {
            max_cycles = std::strtoull(next("--max-cycles").c_str(), nullptr, 10);
        } else if (argument == "--max-resident-fronts") {
            config.max_resident_fronts = static_cast<std::uint32_t>(
                std::strtoul(next("--max-resident-fronts").c_str(), nullptr, 10));
        } else if (argument == "--window") {
            config.command_window_depth = static_cast<std::uint32_t>(
                std::strtoul(next("--window").c_str(), nullptr, 10));
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            usage(argv[0]);
            return 2;
        }
    }
    if (device_dir.empty() || out_dir.empty()) {
        usage(argv[0]);
        return 2;
    }

    hw::dataplane::LoadedArtifact artifact;
    try {
        artifact = hw::dataplane::load_artifact(device_dir);
    } catch (const std::exception& exception) {
        std::cerr << "artifact load failed: " << exception.what() << '\n';
        return 1;
    }

    hw::controlplane::CommandDevice device(std::move(artifact), config);
    std::string error;
    if (!device.start(&error)) {
        std::cerr << "refusing to start: " << error << '\n';
        return 1;
    }
    device.run(max_cycles);
    device.write_reports(out_dir);

    std::cout << "device_dir: " << device_dir << '\n'
              << "execution_backend: test_stub\n"
              << "numerical_validation: not_performed\n"
              << "batch_state: " << hw::controlplane::batch_state_name(device.state())
              << '\n'
              << "control_flow_pass: " << (device.control_flow_pass() ? "true" : "false")
              << '\n'
              << "cycles: " << device.cycle() << '\n'
              << "commands: " << device.command_stats().size() << '\n'
              << "completions_written: " << device.completion_written() << '\n'
              << "reports: " << out_dir << "/commands.csv units.csv fronts.csv "
              << "summary.json transactions.csv buffers.csv\n";
    if (!device.control_flow_pass()) {
        std::cerr << "FAILED: " << device.fault_reason() << '\n';
        return 1;
    }
    return 0;
}
