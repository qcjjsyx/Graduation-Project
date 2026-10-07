// T11 real command artifact executor. Independent FP64 checking is a separate program.
#include <fstream>
#include <iostream>
#include <string>
#include "nlohmann/json.hpp"
#include "controlplane/control_plane.hpp"

int main(int argc, char** argv) {
    try {
        std::string directory, output, backend = "FP32_SYSTEMC";
        std::uint64_t limit = 5000000;
        hw::controlplane::ControlPlaneConfig config;
        config.backend = hw::controlplane::ExecBackend::RequireRealKernels;
        config.enable_real_panel = true;
        config.panel_unit_count = 2;
        config.panel.compute_group_count = 2;
        config.panel.pes_per_group = 2;
        config.panel.pe_count = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "tree_factor_device --device-dir DIR --out DIR "
                    "[--backend FP32_SYSTEMC|FP32_SYSTEMC_INT32_GEMM] [--max-cycles N]\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing argument value");
            const std::string value = argv[++i];
            if (arg == "--device-dir") directory = value;
            else if (arg == "--out") output = value;
            else if (arg == "--backend") backend = value;
            else if (arg == "--max-cycles") {
                std::size_t parsed = 0;
                limit = std::stoull(value, &parsed);
                if (parsed != value.size() || value.front() == '-' || limit == 0)
                    throw std::runtime_error("max-cycles must be a positive integer");
            } else throw std::runtime_error("unknown argument " + arg);
        }
        if (directory.empty() || output.empty()) throw std::runtime_error("device-dir and out are required");
        if (backend != "FP32_SYSTEMC" && backend != "FP32_SYSTEMC_INT32_GEMM")
            throw std::runtime_error("unsupported device backend");
        std::ifstream stream(directory + "/manifest.json");
        nlohmann::json manifest;
        stream >> manifest;
        if (manifest.at("compiler").value("operation_scope", "") != "factorization_only")
            throw std::runtime_error("T11 entry requires a factorization-only artifact");
        if (manifest.at("gemm").at("backend") != backend)
            throw std::runtime_error("requested backend disagrees with artifact; compile the selected mode first");
        auto artifact = hw::dataplane::load_artifact(directory);
        for (std::uint32_t i = 0; i < artifact.config.command_count; ++i) {
            const auto command = hw::command::CommandCodec::decode(
                artifact.image.data() + artifact.config.command_table_addr + i * 32ull, 32);
            if (command.opcode != hw::command::Opcode::GemmSchur) continue;
            const auto kernel = hw::command::DescriptorCodec::decode(
                artifact.image.data() + artifact.config.descriptor_table_addr + command.descriptor_id * 64ull,
                64, artifact.image.size());
            const auto expected = backend == "FP32_SYSTEMC"
                ? hw::command::KernelBackend::SystemCFp32DeviceModel
                : hw::command::KernelBackend::SystemCInt32GemmModel;
            if (kernel.body_words[6] != static_cast<std::uint32_t>(expected))
                throw std::runtime_error("GEMM descriptor backend disagrees with selected runtime mode");
        }
        hw::controlplane::CommandDevice device(std::move(artifact), config);
        std::string error;
        if (!device.start(&error)) throw std::runtime_error(error);
        device.run(limit);
        device.write_reports(output);
        std::cout << "backend=" << backend << " control_flow_pass=" << device.control_flow_pass()
                  << " numeric_status_pass=" << device.numeric_status_pass()
                  << " numerical_validation=not_performed cycles=" << device.cycle() << '\n';
        return device.numeric_status_pass() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "tree factor device: " << e.what() << '\n';
        return 1;
    }
}
