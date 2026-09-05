#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "artifact.hpp"
#include "model_types.hpp"
#include "solve_controller.hpp"
#include "system_memory.hpp"

namespace hw {

struct RunResults {
    std::string mode{};
    std::string status{"ok"};
    std::string failure_reason{};
    double factor_error_fp64{std::numeric_limits<double>::quiet_NaN()};
    double factor_error_fixed{std::numeric_limits<double>::quiet_NaN()};
    SolveMetrics fp64{};
    SolveMetrics fixed{};
    std::uint64_t factor_cycles{0};
    std::uint64_t total_cycles{0};
};

inline std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) return value;
    std::string result{"\""};
    for (const auto character : value) {
        if (character == '"') result += '"';
        result += character;
    }
    result += '"';
    return result;
}

inline nlohmann::json finite_or_null(double value) {
    return std::isfinite(value) ? nlohmann::json(value) : nlohmann::json(nullptr);
}

inline nlohmann::json quant_stats_json(const QuantStats& value) {
    return {
        {"assembly_exp", value.assembly_exp},
        {"node_exp", value.node_exp},
        {"align_shift_max", value.align_shift_max},
        {"align_drop_count", value.align_drop_count},
        {"assembly_overflow_count", value.assembly_overflow_count},
        {"saturation_count", value.saturation_count},
        {"matrix_overflow_count", value.matrix_overflow_count},
        {"vector_shift_count", value.vector_shift_count},
        {"vector_drop_count", value.vector_drop_count},
        {"vector_overflow_count", value.vector_overflow_count},
        {"divide_by_zero_count", value.divide_by_zero_count},
        {"precision_rescue_count", value.precision_rescue_count},
        {"rescue_quantization_saturation_count",
         value.rescue_quantization_saturation_count},
        {"small_pivot_count", value.small_pivot_count},
        {"workspace_renormalize_count",
         value.workspace_renormalize_count},
        {"solution_renormalize_count",
         value.solution_renormalize_count},
        {"max_abs_acc", value.max_abs_acc},
        {"min_pivot_ratio", value.min_pivot_ratio},
        {"max_growth_ratio", value.max_growth_ratio},
    };
}

inline nlohmann::json solve_json(
    const SolveMetrics& metrics,
    double accuracy_target) {
    return {
        {"valid", metrics.valid},
        {"failure_reason", metrics.failure_reason},
        {"relative_residual", finite_or_null(metrics.relative_residual)},
        {"scaled_relative_residual",
         finite_or_null(metrics.scaled_relative_residual)},
        {"componentwise_backward_error",
         finite_or_null(metrics.componentwise_backward_error)},
        {"relative_solution_error",
         finite_or_null(metrics.relative_solution_error)},
        {"initial_relative_residual",
         finite_or_null(metrics.initial_relative_residual)},
        {"residual_history", metrics.residual_history},
        {"refinement_iterations", metrics.refinement_iterations},
        {"refinement_converged", metrics.refinement_converged},
        {"refined_solution", metrics.refined_solution},
        {"used_precision_rescue", metrics.used_precision_rescue},
        {"refinement_stop_reason", metrics.refinement_stop_reason},
        {"accuracy_target", accuracy_target},
        {"accuracy_target_met",
         metrics.valid &&
             std::isfinite(metrics.relative_residual) &&
             metrics.relative_residual <= accuracy_target},
        {"cycles", metrics.cycles},
        {"vector_quantization", quant_stats_json(metrics.vector_stats)},
    };
}

inline nlohmann::json config_json(const ModelConfig& config) {
    return {
        {"clock_period_ns", config.clock_period_ns},
        {"tile_size", config.tile_size},
        {"bfp_tile_size", config.bfp_tile_size},
        {"buffer_count", config.buffer_count},
        {"buffer_capacity_bytes", config.buffer_capacity_bytes},
        {"fifo_depth", config.fifo_depth},
        {"ddr_base_latency", config.ddr_base_latency},
        {"ddr_bytes_per_cycle", config.ddr_bytes_per_cycle},
        {"ddr_burst_bytes", config.ddr_burst_bytes},
        {"ddr_outstanding", config.ddr_outstanding},
        {"ddr_jitter_cycles", config.ddr_jitter_cycles},
        {"backpressure_probability", config.backpressure_probability},
        {"panel_units", config.panel_units},
        {"panel_startup", config.panel_startup},
        {"panel_ops_per_cycle", config.panel_ops_per_cycle},
        {"trsm_units", config.trsm_units},
        {"trsm_startup", config.trsm_startup},
        {"trsm_macs_per_cycle", config.trsm_macs_per_cycle},
        {"gemm_units", config.gemm_units},
        {"gemm_startup", config.gemm_startup},
        {"gemm_macs_per_cycle", config.gemm_macs_per_cycle},
        {"writeback_latency", config.writeback_latency},
        {"timeout_cycles", config.timeout_cycles},
        {"q_use_bits", config.q_use_bits},
        {"frac_bits", config.frac_bits},
        {"accumulator_bits", config.accumulator_bits},
        {"workspace_guard_bits", config.workspace_guard_bits},
        {"vector_use_bits", config.vector_use_bits},
        {"adaptive_factor_scaling", config.adaptive_factor_scaling},
        {"fixed_pivot_rel_tol", config.fixed_pivot_rel_tol},
        {"fixed_factor_rel_tol", config.fixed_factor_rel_tol},
        {"fixed_rescue_mode", config.fixed_rescue_mode},
        {"rescue_pivot_rel_tol", config.rescue_pivot_rel_tol},
        {"precision_rescue_startup", config.precision_rescue_startup},
        {"precision_rescue_macs_per_cycle",
         config.precision_rescue_macs_per_cycle},
        {"iterative_refinement", config.iterative_refinement},
        {"ir_max_iters", config.ir_max_iters},
        {"ir_tolerance", config.ir_tolerance},
        {"ir_min_improvement", config.ir_min_improvement},
        {"ir_residual_macs_per_cycle",
         config.ir_residual_macs_per_cycle},
        {"pivot_rel_tol", config.pivot_rel_tol},
        {"scheduler_policy", config.scheduler_policy},
        {"seed", config.seed},
    };
}

inline void write_solution_csv(
    const std::filesystem::path& path,
    const SolveMetrics& fp64,
    const SolveMetrics& fixed,
    const SystemMemory& memory) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write " + path.string());
    output << "original_index,permuted_index,x_reference,x_fp64,x_fixed\n";
    std::vector<std::size_t> inverse(memory.matrix_dim());
    for (std::size_t permuted = 0; permuted < memory.matrix_dim(); ++permuted) {
        const auto original = memory.permutation.at(permuted);
        inverse.at(original) = permuted;
    }
    for (std::size_t original = 0;
         original < memory.matrix_dim(); ++original) {
        output << original << ',' << inverse.at(original) << ','
               << std::setprecision(17)
               << memory.original_solution_reference.at(original)
               << ',';
        if (fp64.valid) output << fp64.x_original.at(original);
        output << ',';
        if (fixed.valid) output << fixed.x_original.at(original);
        output << '\n';
    }
}

inline void write_reports(
    const std::filesystem::path& output_directory,
    const ArtifactManifest& manifest,
    const ModelConfig& config,
    const SimulationStats& stats,
    const SystemMemory& memory,
    const RunResults& results) {
    std::filesystem::create_directories(output_directory);
    std::uint64_t precision_rescue_nodes = 0;
    std::uint64_t matrix_overflows = 0;
    std::uint64_t rescue_quantization_saturations = 0;
    std::uint64_t assembly_drops = 0;
    std::uint64_t assembled_tile_count = 0;
    std::uint64_t factor_tile_count = 0;
    unsigned max_tile_exponent_span = 0;
    for (const auto& [node_id, node] : stats.nodes) {
        (void)node_id;
        precision_rescue_nodes +=
            node.quant.precision_rescue_count;
        matrix_overflows += node.quant.matrix_overflow_count;
        rescue_quantization_saturations +=
            node.quant.rescue_quantization_saturation_count;
        assembly_drops += node.quant.align_drop_count;
        assembled_tile_count += node.assembled_tile_count;
        factor_tile_count +=
            node.u_tile_count + node.update_tile_count;
        const auto update_span =
            [&](std::int16_t minimum, std::int16_t maximum) {
                max_tile_exponent_span = std::max(
                    max_tile_exponent_span,
                    static_cast<unsigned>(
                        static_cast<int>(maximum) -
                        static_cast<int>(minimum)));
            };
        update_span(
            node.assembled_exp_min, node.assembled_exp_max);
        update_span(node.u_exp_min, node.u_exp_max);
        update_span(node.update_exp_min, node.update_exp_max);
    }

    nlohmann::json summary{
        {"status", results.status},
        {"failure_reason", results.failure_reason},
        {"mode", results.mode},
        {"artifact", manifest.manifest_path.string()},
        {"matrix_dim", manifest.matrix_dim},
        {"node_count", manifest.node_count},
        {"equilibration_mode", manifest.equilibration_mode},
        {"root_count", stats.root_count},
        {"config", config_json(config)},
        {"cycles", {
            {"factorization", results.factor_cycles},
            {"solve", stats.solve_cycles},
            {"total", results.total_cycles},
        }},
        {"factorization", {
            {"fp64_relative_error", finite_or_null(results.factor_error_fp64)},
            {"fixed_relative_error", finite_or_null(results.factor_error_fixed)},
        }},
        {"solve", {
            {"ordering_restored_in_solution_csv", true},
            {"fp64", solve_json(results.fp64, 1e-10)},
            {"fixed", solve_json(results.fixed, config.ir_tolerance)},
        }},
        {"memory", {
            {"read_bytes", stats.memory.read_bytes},
            {"write_bytes", stats.memory.write_bytes},
            {"read_bursts", stats.memory.read_transactions},
            {"write_bursts", stats.memory.write_transactions},
            {"read_cycles", stats.memory.read_cycles},
            {"write_cycles", stats.memory.write_cycles},
        }},
        {"buffer", {
            {"busy_cycles", stats.buffer_busy_cycles},
            {"wait_cycles", stats.buffer_wait_cycles},
        }},
        {"stability", {
            {"precision_rescue_nodes", precision_rescue_nodes},
            {"matrix_overflow_count", matrix_overflows},
            {"rescue_quantization_saturation_count",
             rescue_quantization_saturations},
            {"assembly_drop_count", assembly_drops},
            {"assembled_tile_count", assembled_tile_count},
            {"factor_tile_count", factor_tile_count},
            {"max_tile_exponent_span", max_tile_exponent_span},
        }},
        {"completed_nodes", stats.completed_nodes},
        {"timed_out", stats.timed_out},
        {"address_failure", stats.address_failure},
        {"control_failure", stats.control_failure},
    };
    {
        std::ofstream output(output_directory / "summary.json");
        if (!output) throw std::runtime_error("cannot write summary.json");
        output << std::setw(2) << summary << '\n';
    }

    {
        std::ofstream output(output_directory / "nodes.csv");
        if (!output) throw std::runtime_error("cannot write nodes.csv");
        output
            << "node_id,total_dim,pivot_dim,status,failure_reason,"
               "ready_cycle,start_cycle,assembly_end_cycle,compute_end_cycle,"
               "commit_cycle,load_wait_cycles,assembly_cycles,compute_cycles,"
               "writeback_cycles,assembly_exp,node_exp,fixed_exponent,"
               "fixed_update_exponent,assembled_tile_count,"
               "assembled_exp_min,assembled_exp_max,u_tile_count,"
               "u_exp_min,u_exp_max,update_tile_count,"
               "update_exp_min,update_exp_max,"
               "pivot_swaps_fp64,pivot_swaps_fixed,align_shift_max,"
               "align_drop_count,assembly_overflow_count,saturation_count,"
               "matrix_overflow_count,precision_rescue_count,"
               "rescue_quantization_saturation_count,"
               "small_pivot_count,workspace_renormalize_count,"
               "min_pivot_ratio,max_growth_ratio\n";
        for (std::uint16_t node_id = 0; node_id < manifest.node_count; ++node_id) {
            const auto it = stats.nodes.find(node_id);
            const NodePerformance empty{};
            const auto& node = it == stats.nodes.end() ? empty : it->second;
            output << node_id << ',' << node.total_dim << ',' << node.pivot_dim
                   << ',' << to_string(node.status) << ','
                   << csv_escape(node.failure_reason) << ','
                   << node.ready_cycle << ',' << node.start_cycle << ','
                   << node.assembly_end_cycle << ',' << node.compute_end_cycle
                   << ',' << node.commit_cycle << ',' << node.load_wait_cycles
                   << ',' << node.assembly_cycles << ',' << node.compute_cycles
                   << ',' << node.writeback_cycles << ','
                   << node.quant.assembly_exp << ',' << node.quant.node_exp
                   << ',' << node.fixed_exponent << ','
                   << node.fixed_update_exponent << ','
                   << node.assembled_tile_count << ','
                   << node.assembled_exp_min << ','
                   << node.assembled_exp_max << ','
                   << node.u_tile_count << ','
                   << node.u_exp_min << ','
                   << node.u_exp_max << ','
                   << node.update_tile_count << ','
                   << node.update_exp_min << ','
                   << node.update_exp_max << ','
                   << node.pivot_swaps_fp64 << ',' << node.pivot_swaps_fixed
                   << ',' << node.quant.align_shift_max << ','
                   << node.quant.align_drop_count << ','
                   << node.quant.assembly_overflow_count << ','
                   << node.quant.saturation_count << ','
                   << node.quant.matrix_overflow_count << ','
                   << node.quant.precision_rescue_count << ','
                   << node.quant.rescue_quantization_saturation_count << ','
                   << node.quant.small_pivot_count << ','
                   << node.quant.workspace_renormalize_count << ','
                   << node.quant.min_pivot_ratio << ','
                   << node.quant.max_growth_ratio << '\n';
        }
    }

    {
        std::ofstream output(output_directory / "operations.csv");
        if (!output) throw std::runtime_error("cannot write operations.csv");
        output
            << "node_id,operation,tile_i,tile_j,tile_k,m,n,k,"
               "queued_cycle,start_cycle,end_cycle\n";
        for (const auto& operation : stats.operations) {
            output << operation.node_id << ',' << to_string(operation.type)
                   << ',' << operation.tile_i << ',' << operation.tile_j << ','
                   << operation.tile_k << ',' << operation.m_dim << ','
                   << operation.n_dim << ',' << operation.k_dim << ','
                   << operation.queued_cycle << ',' << operation.start_cycle
                   << ',' << operation.end_cycle << '\n';
        }
    }

    {
        std::ofstream output(output_directory / "memory.csv");
        if (!output) throw std::runtime_error("cannot write memory.csv");
        const auto total_cycles = std::max<std::uint64_t>(results.total_cycles, 1);
        const auto utilization =
            static_cast<double>(
                stats.memory.read_bytes + stats.memory.write_bytes) /
            (static_cast<double>(total_cycles) * config.ddr_bytes_per_cycle);
        output
            << "read_bytes,write_bytes,read_bursts,write_bursts,"
               "read_wait_cycles,write_wait_cycles,bandwidth_utilization\n"
            << stats.memory.read_bytes << ',' << stats.memory.write_bytes << ','
            << stats.memory.read_transactions << ','
            << stats.memory.write_transactions << ','
            << stats.memory.read_cycles << ',' << stats.memory.write_cycles
            << ',' << utilization << '\n';
    }

    {
        std::ofstream output(output_directory / "timeline.csv");
        if (!output) throw std::runtime_error("cannot write timeline.csv");
        output << "cycle,event\n";
        for (std::size_t index = 0; index < stats.events.size(); ++index) {
            const auto separator = stats.events[index].find('|');
            const auto cycle = separator == std::string::npos ?
                std::to_string(index) : stats.events[index].substr(0, separator);
            const auto event = separator == std::string::npos ?
                stats.events[index] : stats.events[index].substr(separator + 1);
            output << cycle << ',' << csv_escape(event) << '\n';
        }
    }

    write_solution_csv(
        output_directory / "solution.csv",
        results.fp64,
        results.fixed,
        memory);
}

}  // namespace hw
