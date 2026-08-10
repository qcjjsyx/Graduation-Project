#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <systemc>
#include <vector>

#include "artifact.hpp"
#include "atu.hpp"
#include "full_system.hpp"
#include "hpu.hpp"
#include "report.hpp"
#include "solve_controller.hpp"
#include "system_memory.hpp"

namespace {

struct Arguments {
    std::filesystem::path artifact{};
    std::filesystem::path config{};
    std::filesystem::path output{};
    hw::NumericMode mode{hw::NumericMode::Both};
    std::string mode_name{"both"};
    bool vcd{false};
    std::optional<std::uint64_t> seed{};
};

std::string usage() {
    return
        "usage: system_sim --artifact <manifest.json> --config <sim_config.json> "
        "--mode fp64|fixed|both --out <result_dir> [--vcd] [--seed N]";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments{};
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + option);
            }
            return argv[++i];
        };
        if (option == "--artifact") {
            arguments.artifact = value();
        } else if (option == "--config") {
            arguments.config = value();
        } else if (option == "--out") {
            arguments.output = value();
        } else if (option == "--mode") {
            arguments.mode_name = value();
            if (arguments.mode_name == "fp64") {
                arguments.mode = hw::NumericMode::Fp64;
            } else if (arguments.mode_name == "fixed") {
                arguments.mode = hw::NumericMode::Fixed;
            } else if (arguments.mode_name == "both") {
                arguments.mode = hw::NumericMode::Both;
            } else {
                throw std::runtime_error("mode must be fp64, fixed, or both");
            }
        } else if (option == "--seed") {
            arguments.seed = std::stoull(value());
        } else if (option == "--vcd") {
            arguments.vcd = true;
        } else if (option == "--help" || option == "-h") {
            std::cout << usage() << '\n';
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option " + option);
        }
    }
    if (arguments.artifact.empty() ||
        arguments.config.empty() ||
        arguments.output.empty()) {
        throw std::runtime_error(usage());
    }
    return arguments;
}

}  // namespace

int sc_main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        auto config = hw::load_model_config(arguments.config);
        if (arguments.seed) config.seed = *arguments.seed;

        const auto manifest = hw::load_manifest(arguments.artifact);
        hw::validate_regions(manifest);
        auto image = hw::read_binary_file(manifest.memory_image_path);
        if (image.size() != manifest.total_bytes) {
            throw std::runtime_error(
                "memory_image.bin size does not match manifest total_bytes");
        }
        const auto tasks = hw::decode_task_queue(manifest, image);

        std::filesystem::create_directories(arguments.output);
        hw::SimulationStats stats{};
        hw::DdrMemory ddr(std::move(image), config, stats);
        auto memory =
            hw::load_system_memory(manifest, tasks, ddr, config);

        sc_core::sc_clock clk(
            "clk", sc_core::sc_time(config.clock_period_ns, sc_core::SC_NS));
        sc_core::sc_signal<bool> rst_n{"rst_n"};
        sc_core::sc_signal<bool> registration_complete{
            "registration_complete"};

        sc_core::sc_fifo<hw::NodeTask> registered_tasks(
            "registered_tasks", config.fifo_depth);
        sc_core::sc_fifo<hw::NodeTask> ready_tasks(
            "ready_tasks", config.fifo_depth);
        sc_core::sc_fifo<hw::WorkItem> loaded_work(
            "loaded_work", config.fifo_depth);
        sc_core::sc_fifo<hw::WorkItem> assembled_work(
            "assembled_work", config.fifo_depth);
        sc_core::sc_fifo<hw::ComputeDone> compute_results(
            "compute_results", config.fifo_depth);
        sc_core::sc_fifo<hw::NodeCommit> commit_events(
            "commit_events", config.fifo_depth);
        sc_core::sc_fifo<hw::BufferRelease> release_events(
            "release_events", config.fifo_depth);

        sc_core::sc_signal<bool> atu_init_identity{"atu_init_identity"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_init_rows{
            "atu_init_rows"};
        sc_core::sc_signal<bool> atu_init_done{"atu_init_done"};
        sc_core::sc_signal<bool> atu_q_req_valid{"atu_q_req_valid"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>>
            atu_q_req_row_logic{"atu_q_req_row_logic"};
        sc_core::sc_signal<bool> atu_q_req_ready{"atu_q_req_ready"};
        sc_core::sc_signal<bool> atu_q_resp_valid{"atu_q_resp_valid"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>>
            atu_q_resp_row_physical{"atu_q_resp_row_physical"};
        sc_core::sc_signal<bool> atu_pivot_req_valid{
            "atu_pivot_req_valid"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_pivot_row_i{
            "atu_pivot_row_i"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_pivot_row_j{
            "atu_pivot_row_j"};
        sc_core::sc_signal<bool> atu_pivot_req_ready{
            "atu_pivot_req_ready"};
        sc_core::sc_signal<bool> atu_pivot_done{"atu_pivot_done"};

        sc_core::sc_signal<bool> hpu_pivot_start{"hpu_pivot_start"};
        sc_core::sc_signal<bool> hpu_pivot_busy{"hpu_pivot_busy"};
        sc_core::sc_signal<bool> hpu_in_valid{"hpu_in_valid"};
        sc_core::sc_signal<bool> hpu_in_ready{"hpu_in_ready"};
        sc_core::sc_signal<sc_dt::sc_int<hw::HPU::DATA_W>> hpu_in_value{
            "hpu_in_value"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>>
            hpu_in_row_logical{"hpu_in_row_logical"};
        sc_core::sc_signal<bool> hpu_in_last{"hpu_in_last"};
        sc_core::sc_signal<bool> hpu_pivot_valid{"hpu_pivot_valid"};
        sc_core::sc_signal<bool> hpu_pivot_ready{"hpu_pivot_ready"};
        sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> hpu_pivot_row{
            "hpu_pivot_row"};
        sc_core::sc_signal<sc_dt::sc_int<hw::HPU::DATA_W>> hpu_pivot_value{
            "hpu_pivot_value"};
        sc_core::sc_signal<bool> hpu_pivot_fail{"hpu_pivot_fail"};

        hw::CycleCounter counter("cycle_counter", stats);
        counter.clk(clk);
        counter.rst_n(rst_n);

        hw::TaskFetch task_fetch("task_fetch", tasks, ddr, stats);
        task_fetch.clk(clk);
        task_fetch.rst_n(rst_n);
        task_fetch.task_out(registered_tasks);
        task_fetch.registration_complete(registration_complete);

        hw::DependencyScoreboard scoreboard(
            "scoreboard", tasks.size(), stats);
        scoreboard.clk(clk);
        scoreboard.rst_n(rst_n);
        scoreboard.registration_complete(registration_complete);
        scoreboard.task_in(registered_tasks);
        scoreboard.commit_in(commit_events);
        scoreboard.ready_out(ready_tasks);

        hw::BufferManager buffer_manager(
            "buffer_manager", manifest, config, arguments.mode, ddr, stats);
        buffer_manager.clk(clk);
        buffer_manager.rst_n(rst_n);
        buffer_manager.task_in(ready_tasks);
        buffer_manager.release_in(release_events);
        buffer_manager.work_out(loaded_work);

        hw::FrontAssembler assembler(
            "front_assembler", memory, config, arguments.mode, stats);
        assembler.clk(clk);
        assembler.rst_n(rst_n);
        assembler.work_in(loaded_work);
        assembler.work_out(assembled_work);

        hw::ATU atu("atu");
        atu.clk(clk);
        atu.rst_n(rst_n);
        atu.init_identity(atu_init_identity);
        atu.init_rows(atu_init_rows);
        atu.init_done(atu_init_done);
        atu.q_req_valid(atu_q_req_valid);
        atu.q_req_row_logic(atu_q_req_row_logic);
        atu.q_req_ready(atu_q_req_ready);
        atu.q_resp_valid(atu_q_resp_valid);
        atu.q_resp_row_physical(atu_q_resp_row_physical);
        atu.pivot_req_valid(atu_pivot_req_valid);
        atu.pivot_row_i(atu_pivot_row_i);
        atu.pivot_row_j(atu_pivot_row_j);
        atu.pivot_req_ready(atu_pivot_req_ready);
        atu.pivot_done(atu_pivot_done);

        hw::HPU hpu("hpu");
        hpu.clk(clk);
        hpu.rst_n(rst_n);
        hpu.pivot_start(hpu_pivot_start);
        hpu.pivot_busy(hpu_pivot_busy);
        hpu.in_valid(hpu_in_valid);
        hpu.in_ready(hpu_in_ready);
        hpu.in_value(hpu_in_value);
        hpu.in_row_logical(hpu_in_row_logical);
        hpu.in_last(hpu_in_last);
        hpu.pivot_valid(hpu_pivot_valid);
        hpu.pivot_ready(hpu_pivot_ready);
        hpu.pivot_row(hpu_pivot_row);
        hpu.pivot_value(hpu_pivot_value);
        hpu.pivot_fail(hpu_pivot_fail);

        hw::KernelDispatcher dispatcher(
            "kernel_dispatcher", memory, config, arguments.mode, stats);
        dispatcher.clk(clk);
        dispatcher.rst_n(rst_n);
        dispatcher.work_in(assembled_work);
        dispatcher.result_out(compute_results);
        dispatcher.atu_init_identity(atu_init_identity);
        dispatcher.atu_init_rows(atu_init_rows);
        dispatcher.atu_init_done(atu_init_done);
        dispatcher.atu_pivot_req_valid(atu_pivot_req_valid);
        dispatcher.atu_pivot_row_i(atu_pivot_row_i);
        dispatcher.atu_pivot_row_j(atu_pivot_row_j);
        dispatcher.atu_pivot_req_ready(atu_pivot_req_ready);
        dispatcher.atu_pivot_done(atu_pivot_done);
        dispatcher.hpu_pivot_start(hpu_pivot_start);
        dispatcher.hpu_in_valid(hpu_in_valid);
        dispatcher.hpu_in_ready(hpu_in_ready);
        dispatcher.hpu_in_value(hpu_in_value);
        dispatcher.hpu_in_row_logical(hpu_in_row_logical);
        dispatcher.hpu_in_last(hpu_in_last);
        dispatcher.hpu_pivot_valid(hpu_pivot_valid);
        dispatcher.hpu_pivot_ready(hpu_pivot_ready);
        dispatcher.hpu_pivot_row(hpu_pivot_row);
        dispatcher.hpu_pivot_value(hpu_pivot_value);
        dispatcher.hpu_pivot_fail(hpu_pivot_fail);

        hw::ResultWriter writer(
            "result_writer", manifest, memory, ddr, config,
            arguments.mode, stats);
        writer.clk(clk);
        writer.rst_n(rst_n);
        writer.result_in(compute_results);
        writer.commit_out(commit_events);
        writer.release_out(release_events);

        hw::CompletionMonitor monitor(
            "completion_monitor", tasks.size(), config.timeout_cycles, stats);
        monitor.clk(clk);
        monitor.rst_n(rst_n);

        atu_q_req_valid.write(false);
        atu_q_req_row_logic.write(0);
        sc_core::sc_trace_file* trace = nullptr;
        if (arguments.vcd) {
            const auto base =
                (std::filesystem::absolute(arguments.output) / "system_sim")
                    .string();
            trace = sc_core::sc_create_vcd_trace_file(base.c_str());
            trace->set_time_unit(1, sc_core::SC_NS);
            sc_core::sc_trace(trace, clk, "clk");
            sc_core::sc_trace(trace, rst_n, "rst_n");
            sc_core::sc_trace(
                trace, registration_complete, "registration_complete");
            sc_core::sc_trace(trace, atu_init_identity, "atu_init_identity");
            sc_core::sc_trace(trace, atu_init_done, "atu_init_done");
            sc_core::sc_trace(trace, atu_pivot_done, "atu_pivot_done");
            sc_core::sc_trace(trace, hpu_pivot_start, "hpu_pivot_start");
            sc_core::sc_trace(trace, hpu_pivot_busy, "hpu_pivot_busy");
            sc_core::sc_trace(trace, hpu_in_valid, "hpu_in_valid");
            sc_core::sc_trace(trace, hpu_pivot_valid, "hpu_pivot_valid");
        }

        std::cout << "SystemC multifrontal simulation: "
                  << manifest.matrix_dim << " variables, "
                  << manifest.node_count << " nodes, mode "
                  << arguments.mode_name << '\n';
        rst_n.write(false);
        sc_core::sc_start(
            sc_core::sc_time(
                config.clock_period_ns * 3, sc_core::SC_NS));
        rst_n.write(true);
        sc_core::sc_start();
        if (trace) sc_core::sc_close_vcd_trace_file(trace);

        hw::RunResults results{};
        results.mode = arguments.mode_name;
        results.factor_cycles = stats.cycle;
        if (stats.timed_out || stats.numeric_failure ||
            stats.address_failure ||
            stats.control_failure ||
            stats.completed_nodes != tasks.size()) {
            results.status = stats.timed_out ? "timeout" :
                stats.control_failure ? "control_failure" :
                stats.address_failure ? "address_failure" :
                "numeric_failure";
            results.failure_reason = stats.failure_reason;
        } else {
            if (hw::mode_has_fp64(arguments.mode)) {
                results.factor_error_fp64 =
                    hw::factorization_relative_error_fp64(memory);
                results.fp64 = hw::solve_fp64(
                    memory, config, stats,
                    results.factor_cycles + stats.solve_cycles);
                stats.solve_cycles += results.fp64.cycles;
                if (!results.fp64.valid) {
                    results.status = "numeric_failure";
                    results.failure_reason =
                        "FP64 solve: " + results.fp64.failure_reason;
                } else {
                    auto delay = ddr.write_f64_vector(
                        manifest.global_regions.at("solution_q").offset,
                        results.fp64.x_permuted);
                    delay = std::max(
                        delay,
                        ddr.write_i16_vector(
                            manifest.global_regions.at("solution_e").offset,
                            std::vector<std::int16_t>(
                                manifest.node_count, 0)));
                    stats.solve_cycles += delay;
                }
            }
            if (hw::mode_has_fixed(arguments.mode) &&
                results.status == "ok") {
                results.factor_error_fixed =
                    hw::factorization_relative_error_fixed(memory, config);
                results.fixed = hw::solve_fixed(
                    memory, config, stats,
                    results.factor_cycles + stats.solve_cycles);
                stats.solve_cycles += results.fixed.cycles;
                if (!results.fixed.valid) {
                    results.status = "numeric_failure";
                    results.failure_reason =
                        "fixed solve: " + results.fixed.failure_reason;
                } else {
                    auto delay = results.fixed.refined_solution ?
                        ddr.write_f64_vector(
                            manifest.global_regions.at("solution_q").offset,
                            results.fixed.x_permuted) :
                        ddr.write_i64_vector(
                            manifest.global_regions.at("solution_q").offset,
                            results.fixed.x_mantissa);
                    delay = std::max(
                        delay,
                        ddr.write_i16_vector(
                            manifest.global_regions.at("solution_e").offset,
                            results.fixed.refined_solution ?
                                std::vector<std::int16_t>(
                                    manifest.node_count, 0) :
                                results.fixed.exponent_by_node));
                    stats.solve_cycles += delay;
                }
            }
        }
        results.total_cycles = results.factor_cycles + stats.solve_cycles;
        hw::write_reports(
            arguments.output, manifest, config, stats, memory, results);
        hw::write_binary_file(
            arguments.output / "final_memory_image.bin", ddr.bytes());
        std::cout << "status=" << results.status
                  << " cycles=" << results.total_cycles
                  << " results=" << std::filesystem::absolute(arguments.output)
                  << '\n';
        if (!results.failure_reason.empty()) {
            std::cout << "reason=" << results.failure_reason << '\n';
        }
        return results.status == "ok" ? 0 : 2;
    } catch (const std::exception& exception) {
        std::cerr << "system_sim: " << exception.what() << '\n';
        return 1;
    }
}
