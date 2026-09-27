#include <cstdlib>
#include <iostream>
#include <string>

#include "dataplane/selftest.hpp"

namespace {

void usage(const char* program) {
    std::cerr
        << "usage: " << program << " --device-dir DIR --out DIR\n"
        << "       [--source-region NAME] [--target-region NAME]\n"
        << "       [--buffer-capacity BYTES] [--buffer-slots N]\n"
        << "       [--max-cycles N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string device_dir;
    std::string out_dir;
    hw::dataplane::DataPlaneSelftestOptions options;

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
        } else if (argument == "--source-region") {
            options.source_region = next("--source-region");
        } else if (argument == "--target-region") {
            options.target_region = next("--target-region");
        } else if (argument == "--buffer-capacity") {
            options.buffer_capacity_bytes =
                std::strtoull(next("--buffer-capacity").c_str(), nullptr, 10);
        } else if (argument == "--buffer-slots") {
            options.buffer_slots = static_cast<std::uint32_t>(
                std::strtoul(next("--buffer-slots").c_str(), nullptr, 10));
        } else if (argument == "--max-cycles") {
            options.timeout_cycles =
                std::strtoull(next("--max-cycles").c_str(), nullptr, 10);
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

    const auto report =
        hw::dataplane::run_dataplane_selftest(device_dir, options, out_dir);
    std::cout << "device_dir: " << device_dir << '\n'
              << "source_region: " << report.source_region << '\n'
              << "target_region: " << report.target_region << '\n'
              << "policy: " << report.policy << '\n'
              << "tile: " << report.tile_rows << "x" << report.tile_cols
              << " (" << report.tile_bytes << " bytes)\n"
              << "cycles: " << report.cycles << '\n'
              << "descriptor_reads: " << report.descriptor_reads << '\n'
              << "dma_read_bytes: " << report.dma_read_bytes << '\n'
              << "dma_write_bytes: " << report.dma_write_bytes << '\n'
              << "target_verified: " << (report.target_verified ? "yes" : "no")
              << '\n'
              << "non_target_unchanged: "
              << (report.non_target_unchanged ? "yes" : "no") << '\n'
              << "control_regions_unchanged: "
              << (report.control_regions_unchanged ? "yes" : "no") << '\n'
              << "note: " << report.writeback_note << '\n';
    if (!report.ok) {
        std::cerr << "FAILED: " << report.failure << '\n';
        return 1;
    }
    return 0;
}
