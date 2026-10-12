// T11/T12 real command artifact executor. Independent FP64 checking is a
// separate program (tree_factor_check + tree_solve_check).
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include "nlohmann/json.hpp"
#include "controlplane/control_plane.hpp"

int main(int argc, char** argv) {
    try {
        std::string directory, output, backend = "FP32_SYSTEMC";
        // Bounded guard, not a target: the three real matrices need ~0.8M /
        // ~6.1M / ~14.5M cycles, so the default must not cut a valid run
        // short (the T11 default of 5M did). A stuck model still terminates.
        std::uint64_t limit = 40000000;
        hw::controlplane::ControlPlaneConfig config;
        config.backend = hw::controlplane::ExecBackend::RequireRealKernels;
        config.enable_real_panel = true;
        // T12: the device executes SOLVE_FORWARD/SOLVE_BACKWARD with the
        // dedicated solve unit (factorization-only artifacts simply contain
        // no solve commands).
        config.enable_real_solve = true;
        config.solve_unit_count = 1;
        config.panel_unit_count = 2;
        config.panel.compute_group_count = 2;
        config.panel.pes_per_group = 2;
        config.panel.pe_count = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "tree_factor_device --device-dir DIR --out DIR "
                    "[--backend FP32_SYSTEMC|FP32_SYSTEMC_INT32_GEMM] [--max-cycles N]\n"
                    "  accepts factorization_only and factorization_and_solve "
                    "artifacts\n";
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
        const std::string scope =
            manifest.at("compiler").value("operation_scope", "factorization_and_solve");
        if (scope != "factorization_only" && scope != "factorization_and_solve")
            throw std::runtime_error("unknown compiler operation_scope: " + scope);
        if (manifest.at("gemm").at("backend") != backend) //编译时GEMM后端和实际执行时GEMM后端要一致
            throw std::runtime_error("requested backend disagrees with artifact; compile the selected mode first");
        // 编译器按目标 scratchpad 槽容量检查过所有 front（4*n*n <= device_buffer_bytes），
        // 设备按同一容量配置槽位，保证 "编译通过" 与 "设备可容纳" 一致。
        // 旧 manifest 没有该字段时保持 ControlPlaneConfig 默认值（64 KiB）。
        if (manifest.at("compiler").contains("device_buffer_bytes")) {
            const auto bytes = manifest.at("compiler").at("device_buffer_bytes").get<std::uint64_t>();
            if (bytes == 0) throw std::runtime_error("manifest device_buffer_bytes must be positive");
            config.buffer_capacity_bytes = bytes;
        }
        // T12: the solve scratch of any node must fit the same slot (the
        // compiler already refused oversized solve scratch; re-check the
        // manifest number so a hand-edited artifact cannot pass).
        if (manifest.contains("solve")) {
            const auto& solve = manifest.at("solve");
            if (solve.contains("max_scratch_bytes")) {
                const auto scratch = solve.at("max_scratch_bytes").get<std::uint64_t>();
                if (scratch > config.buffer_capacity_bytes)
                    throw std::runtime_error(
                        "manifest solve scratch exceeds device_buffer_bytes");
            }
        }
        // 同理：Panel 引擎的 front 维度上限取编译时的 max_front_size（引擎支持 1..512），
        // 否则 --max-front-size > 256 编译出的 artifact 会在运行期报 BAD_DESCRIPTOR。
        if (manifest.at("compiler").contains("max_front_size")) {
            const auto front = manifest.at("compiler").at("max_front_size").get<std::uint64_t>();
            if (front == 0 || front > 512)
                throw std::runtime_error("manifest max_front_size must be in [1, 512] for this device");
            config.panel.max_front_dim = static_cast<std::uint32_t>(front);
        }
        auto artifact = hw::dataplane::load_artifact(directory);
        for (std::uint32_t i = 0; i < artifact.config.command_count; ++i) {
            const auto command = hw::command::CommandCodec::decode(
                artifact.image.data() + artifact.config.command_table_addr + i * 32ull, 32);
            if (command.opcode != hw::command::Opcode::GemmSchur) continue; // 检查一下每条GEMM的后端是否与设定的一致
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
        device.run(limit); //启动
        device.write_reports(output);
        std::uint64_t solve_cycles = 0;
        std::uint32_t solve_commands = 0;
        std::uint32_t solve_ok = 0;
        const std::string forward_opcode = std::to_string(
            static_cast<std::uint32_t>(hw::command::Opcode::SolveForward));
        const std::string backward_opcode = std::to_string(
            static_cast<std::uint32_t>(hw::command::Opcode::SolveBackward));
        for (const auto& stats : device.command_stats()) {
            if (stats.opcode != forward_opcode && stats.opcode != backward_opcode) {
                continue;
            }
            ++solve_commands;
            solve_ok += stats.status == "OK" ? 1 : 0;
            if (stats.finish_cycle >= stats.start_cycle) {
                solve_cycles += stats.finish_cycle - stats.start_cycle;
            }
        }
        std::cout << "backend=" << backend << " scope=" << scope
                  << " control_flow_pass=" << device.control_flow_pass()
                  << " numeric_status_pass=" << device.numeric_status_pass()
                  << " numerical_validation=not_performed cycles=" << device.cycle()
                  << " solve_commands=" << solve_commands
                  << " solve_ok=" << solve_ok
                  << " solve_cycles=" << solve_cycles << '\n';
        return device.numeric_status_pass() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "tree factor device: " << e.what() << '\n';
        return 1;
    }
}
