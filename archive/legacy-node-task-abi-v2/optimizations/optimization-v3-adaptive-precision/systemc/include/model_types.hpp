#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace hw {

static constexpr std::uint16_t ROOT_PARENT_ID = 0xFFFFu;

struct ModelConfig {
    unsigned clock_period_ns{10};
    unsigned tile_size{16};
    unsigned bfp_tile_size{16};
    unsigned buffer_count{2};
    std::uint64_t buffer_capacity_bytes{1u << 20};
    unsigned fifo_depth{8};
    unsigned ddr_base_latency{20};
    unsigned ddr_bytes_per_cycle{32};
    unsigned ddr_burst_bytes{64};
    unsigned ddr_outstanding{4};
    unsigned ddr_jitter_cycles{0};
    double backpressure_probability{0.0};
    unsigned panel_units{1};
    unsigned panel_startup{8};
    unsigned panel_ops_per_cycle{8};
    unsigned trsm_units{1};
    unsigned trsm_startup{6};
    unsigned trsm_macs_per_cycle{64};
    unsigned gemm_units{1};
    unsigned gemm_startup{10};
    unsigned gemm_macs_per_cycle{256};
    unsigned writeback_latency{2};
    std::uint64_t timeout_cycles{100000000};
    unsigned q_use_bits{30};
    unsigned frac_bits{26};
    bool adaptive_frac_retry{false};
    unsigned retry_frac_bits{26};
    unsigned accumulator_bits{64};
    unsigned workspace_guard_bits{20};
    unsigned vector_use_bits{55};
    bool adaptive_factor_scaling{true};
    double fixed_pivot_rel_tol{1e-5};
    double fixed_factor_rel_tol{2e-7};
    std::string fixed_rescue_mode{"fp64"};
    double rescue_pivot_rel_tol{1e-16};
    unsigned precision_rescue_startup{32};
    unsigned precision_rescue_macs_per_cycle{32};
    bool iterative_refinement{true};
    unsigned ir_max_iters{50};
    double ir_tolerance{1e-3};
    double ir_min_improvement{1e-3};
    unsigned ir_residual_macs_per_cycle{256};
    double pivot_rel_tol{1e-12};
    std::string scheduler_policy{"serial"};
    std::uint64_t seed{1};

    std::int32_t q_limit() const {
        return static_cast<std::int32_t>((std::uint32_t{1} << q_use_bits) - 1);
    }
};

struct NodeTask {
    std::uint16_t node_id{0};
    std::uint16_t flags{0};
    std::uint16_t parent_id{ROOT_PARENT_ID};
    std::uint16_t children_count{0};
    std::uint32_t total_dim{0};
    std::uint32_t pivot_dim{0};
    std::uint32_t tile_count{0};
    std::uint32_t tail_dim{0};
    std::uint32_t map_table_bytes{0};
    std::uint32_t reserved{0};
    std::uint64_t front_q_addr{0};
    std::uint64_t front_e_addr{0};
    std::uint64_t update_q_addr{0};
    std::uint64_t update_e_addr{0};
    std::uint64_t map_table_addr{0};
    std::uint64_t l_factor_addr{0};
    std::uint64_t u_factor_addr{0};
    std::uint64_t p_vector_addr{0};
    std::uint64_t node_meta_addr{0};
    std::uint64_t solve_workspace_addr{0};
    std::uint64_t reserved_addr0{0};
    std::uint64_t reserved_addr1{0};
};

inline std::ostream& operator<<(std::ostream& os, const NodeTask& task) {
    return os << "NodeTask{id=" << task.node_id << ", parent=" << task.parent_id
              << ", children=" << task.children_count << ", total=" << task.total_dim
              << ", pivot=" << task.pivot_dim << "}";
}

struct WorkItem {
    unsigned buffer_id{0};
    NodeTask task{};
};

inline std::ostream& operator<<(std::ostream& os, const WorkItem& item) {
    return os << "WorkItem{buffer=" << item.buffer_id << ", " << item.task << "}";
}

struct PivotedWork {
    WorkItem work{};
    std::uint32_t pivot_row{0};
    std::int32_t pivot_value{0};
};

inline std::ostream& operator<<(std::ostream& os, const PivotedWork& item) {
    return os << "PivotedWork{" << item.work << ", row=" << item.pivot_row
              << ", value=" << item.pivot_value << "}";
}

struct ComputeResult {
    PivotedWork work{};
    std::vector<std::int32_t> update_q{};
    std::int16_t update_exp{0};
    std::uint32_t update_dim{0};
};

inline std::ostream& operator<<(std::ostream& os, const ComputeResult& item) {
    return os << "ComputeResult{" << item.work << ", update_dim=" << item.update_dim
              << ", update_exp=" << item.update_exp << "}";
}

struct ComputeDone {
    WorkItem work{};
    bool success{true};
    std::string failure_reason{};
};

inline std::ostream& operator<<(std::ostream& os, const ComputeDone& item) {
    return os << "ComputeDone{" << item.work << ", success=" << item.success
              << "}";
}

struct NodeCommit {
    std::uint16_t node_id{0};
    std::uint16_t parent_id{ROOT_PARENT_ID};
};

inline std::ostream& operator<<(std::ostream& os, const NodeCommit& event) {
    return os << "NodeCommit{node=" << event.node_id << ", parent=" << event.parent_id << "}";
}

struct BufferRelease {
    unsigned buffer_id{0};
    std::uint16_t node_id{0};
};

inline std::ostream& operator<<(std::ostream& os, const BufferRelease& event) {
    return os << "BufferRelease{buffer=" << event.buffer_id << ", node=" << event.node_id << "}";
}

struct QuantStats {
    std::int16_t assembly_exp{0};
    std::int16_t node_exp{0};
    unsigned align_shift_max{0};
    std::uint64_t align_drop_count{0};
    std::uint64_t assembly_overflow_count{0};
    std::uint64_t saturation_count{0};
    std::uint64_t matrix_overflow_count{0};
    std::uint64_t vector_shift_count{0};
    std::uint64_t vector_drop_count{0};
    std::uint64_t vector_overflow_count{0};
    std::uint64_t divide_by_zero_count{0};
    std::uint64_t fraction_retry_count{0};
    std::uint64_t precision_assist_count{0};
    std::uint64_t precision_rescue_count{0};
    std::uint64_t rescue_quantization_saturation_count{0};
    std::uint64_t small_pivot_count{0};
    std::uint64_t workspace_renormalize_count{0};
    std::uint64_t solution_renormalize_count{0};
    std::int64_t max_abs_acc{0};
    double min_pivot_ratio{1.0};
    double max_growth_ratio{1.0};
};

enum class NumericMode {
    Fp64,
    Fixed,
    Both,
};

enum class NodeStatus {
    Pending,
    Running,
    Complete,
    NumericFailure,
    AddressFailure,
};

inline const char* to_string(NodeStatus status) {
    switch (status) {
    case NodeStatus::Pending: return "pending";
    case NodeStatus::Running: return "running";
    case NodeStatus::Complete: return "complete";
    case NodeStatus::NumericFailure: return "numeric_failure";
    case NodeStatus::AddressFailure: return "address_failure";
    }
    return "unknown";
}

enum class OpType {
    Fact,
    TrsmU,
    TrsmL,
    GemmPivot,
    TrsmF12,
    TrsmF21,
    GemmSchur,
    PrecisionRescue,
    SolveForward,
    SolveBackward,
    SolveResidual,
};

inline const char* to_string(OpType type) {
    switch (type) {
    case OpType::Fact: return "FACT";
    case OpType::TrsmU: return "TRSM_U";
    case OpType::TrsmL: return "TRSM_L";
    case OpType::GemmPivot: return "GEMM_PIVOT";
    case OpType::TrsmF12: return "TRSM_F12";
    case OpType::TrsmF21: return "TRSM_F21";
    case OpType::GemmSchur: return "GEMM_SCHUR";
    case OpType::PrecisionRescue: return "PRECISION_RESCUE";
    case OpType::SolveForward: return "SOLVE_FORWARD";
    case OpType::SolveBackward: return "SOLVE_BACKWARD";
    case OpType::SolveResidual: return "SOLVE_RESIDUAL";
    }
    return "UNKNOWN";
}

struct OperationLog {
    std::uint16_t node_id{0};
    OpType type{OpType::Fact};
    unsigned tile_i{0};
    unsigned tile_j{0};
    unsigned tile_k{0};
    unsigned m_dim{0};
    unsigned n_dim{0};
    unsigned k_dim{0};
    std::uint64_t queued_cycle{0};
    std::uint64_t start_cycle{0};
    std::uint64_t end_cycle{0};
};

struct NodePerformance {
    std::uint16_t node_id{0};
    std::uint32_t total_dim{0};
    std::uint32_t pivot_dim{0};
    NodeStatus status{NodeStatus::Pending};
    std::string failure_reason{};
    std::uint64_t ready_cycle{0};
    std::uint64_t start_cycle{0};
    std::uint64_t assembly_end_cycle{0};
    std::uint64_t compute_end_cycle{0};
    std::uint64_t commit_cycle{0};
    std::uint64_t load_wait_cycles{0};
    std::uint64_t assembly_cycles{0};
    std::uint64_t compute_cycles{0};
    std::uint64_t writeback_cycles{0};
    std::int16_t fixed_exponent{0};
    std::int16_t fixed_update_exponent{0};
    unsigned l_frac_bits{0};
    bool fraction_retry_attempted{false};
    bool precision_assisted{false};
    bool precision_rescued{false};
    std::uint32_t assembled_tile_count{0};
    std::int16_t assembled_exp_min{0};
    std::int16_t assembled_exp_max{0};
    std::uint32_t u_tile_count{0};
    std::int16_t u_exp_min{0};
    std::int16_t u_exp_max{0};
    std::uint32_t update_tile_count{0};
    std::int16_t update_exp_min{0};
    std::int16_t update_exp_max{0};
    unsigned pivot_swaps_fixed{0};
    unsigned pivot_swaps_fp64{0};
    QuantStats quant{};
};

struct MemoryStats {
    std::uint64_t read_bytes{0};
    std::uint64_t write_bytes{0};
    std::uint64_t read_transactions{0};
    std::uint64_t write_transactions{0};
    std::uint64_t read_cycles{0};
    std::uint64_t write_cycles{0};
};

struct SimulationStats {
    std::uint64_t cycle{0};
    std::vector<std::uint16_t> start_order{};
    std::vector<std::uint16_t> commit_order{};
    std::map<std::uint16_t, std::uint64_t> start_cycle{};
    std::map<std::uint16_t, std::uint64_t> commit_cycle{};
    std::map<std::uint16_t, std::uint32_t> pivot_rows{};
    std::map<std::uint16_t, unsigned> execute_count{};
    std::map<std::uint16_t, unsigned> commit_count{};
    std::map<std::uint16_t, NodePerformance> nodes{};
    std::vector<OperationLog> operations{};
    MemoryStats memory{};
    std::uint64_t buffer_busy_cycles{0};
    std::uint64_t buffer_wait_cycles{0};
    std::uint64_t solve_cycles{0};
    std::uint64_t completed_nodes{0};
    std::uint64_t root_count{0};
    bool timed_out{false};
    bool numeric_failure{false};
    bool address_failure{false};
    bool control_failure{false};
    std::string failure_reason{};
    std::vector<std::string> events{};

    void record(const std::string& message) {
        timeline(message);
        std::cout << "[cycle " << cycle << "] " << message << '\n';
    }

    void timeline(const std::string& message) {
        events.push_back(std::to_string(cycle) + "|" + message);
    }
};

}  // namespace hw
