#pragma once

// T12 dedicated tree solve unit (design doc §4).
//
// Executes SOLVE_FORWARD / SOLVE_BACKWARD over the DDR-resident factors,
// P-vectors and RHS written by the factorization phase:
//
//   forward (topological order):
//     r_i = b_hat[g[p[i]]] + w[g[p[i]]]      (P-vector gather in the block)
//     L11 z = r (unit lower), w[g[p+t]] -= L21 z, w[g[i]] = z
//   backward (reverse topological order):
//     c_i = w[g[i]] - U12 w[g[p+t]], U11 x = c (zero diagonal refused),
//     w[g[i]] = x_i
//
// w is `solution_data` (a zero-initialized accumulator; the RHS is read-only).
// The unit owns its DDR traffic: bounded reads/writes through MemoryService
// with precise response routing (`deliver_memory_response` / `handles_request`
// / `pending_request_count`), never the legacy whole-RHS DMA copy path. On-chip
// row staging and compute consumption are charged to the shared OnChipArbiter.
//
// Scratch model (design doc §4.3): the unit holds a scratch capacity
// reservation (write lease on the solve scratch slot, acquired by the control
// plane) for the row buffer depth*n, z/x p each, the update vector u and the
// P-vector p. The live vectors are the unit's register-level storage; because
// MemoryService already delivers DDR bytes to the unit, staging them again
// through BufferManager would model the same bytes twice. The unified arbiter
// charges the staging (write domain) and the compute reads (read domain).
//
// Failure semantics (design doc §4.4): malformed view/payload or an illegal
// P-vector -> BAD_DESCRIPTOR, zero U diagonal -> STATIC_PIVOT_FAILURE,
// NaN/Inf input or result -> NON_FINITE_VALUE, insufficient scratch ->
// BUFFER_FULL, abort -> ABORTED after the in-flight traffic drains. Exactly
// one terminal response per accepted transaction; a failed command publishes
// no READY token (the control plane owns the token side).

#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "controlplane/exec_types.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/on_chip_arbiter.hpp"

namespace hw::controlplane {

struct SolveConfig {
    // Multiply-accumulate lanes: a length-m dot product costs
    // ceil(m / lanes) + reduce_latency cycles.
    std::uint32_t lanes{16};
    std::uint32_t reduce_latency{2};
    // Division latency of the backward diagonal (fixed behavior model).
    std::uint32_t div_latency{4};
    // Row buffer depth in rows of total_dim FP32 words (design doc §4.3).
    std::uint32_t row_buffer_depth{2};
    // Upper bound on one scatter/gather request (elements).
    std::uint32_t max_elements_per_request{64};
};

class SolveExecutionUnit : public IExecutionUnit {
public:
    SolveExecutionUnit(
        std::uint32_t unit_id,
        hw::dataplane::MemoryService* memory,
        SolveConfig config);

    // Unified on-chip read/write budget for row staging and compute reads.
    void set_on_chip_arbiter(hw::dataplane::OnChipArbiter* arbiter) {
        arbiter_ = arbiter;
    }

    // Scratch requirement of one front (design doc §4.3 upper bound):
    // 4 * (depth * total_dim + 2 * total_dim + pivot_dim) bytes.
    static std::uint64_t scratch_bytes(
        std::uint32_t total_dim,
        std::uint32_t pivot_dim,
        std::uint32_t depth);

    UnitKind kind() const override { return UnitKind::Solve; }
    std::uint32_t unit_id() const override { return unit_id_; }

    bool submit(const ExecRequest& request, std::uint64_t cycle) override;
    void tick(std::uint64_t cycle) override;
    bool poll_response(ExecResponse& response) override;
    bool busy() const override;
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) override;
    void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) override;
    void reset() override;

    void deliver_memory_response(const hw::dataplane::MemoryResponse& response) override;
    bool handles_request(std::uint64_t request_id) const override;
    std::size_t pending_request_count() const override;

    std::uint32_t active_command_id() const override {
        return active_.has_value() ? active_->command_id : hw::command::NONE;
    }

    // Behavior-model statistics of the last transaction (diagnostics; also
    // mirrored into the device summary by the control plane).
    std::uint64_t compute_cycles() const { return compute_cycles_; }
    std::uint64_t staged_bytes() const { return staged_bytes_; }
    std::uint64_t consumed_bytes() const { return consumed_bytes_; }

private:
    enum class Phase {
        PVector,      // read + validate the pivot-block permutation
        VectorReads,  // pivot-block b_hat (forward) / w
        UpdateReads,  // scatter/gather the update segment of w
        Rows,         // stream L (forward) or U (backward) rows
        Writes,       // update segment / workspace / pivot-block results
    };
    enum class XferKind {
        PVector,
        RhsPivot,
        WPivot,
        WUpdateRead,
        Row,
        WUpdateWrite,
        Workspace,
        WPivotWrite,
    };
    struct Xfer {
        XferKind kind{XferKind::PVector};
        std::uint32_t row{0};          // Row: slot index; others: unused
        std::uint32_t local_start{0};  // update local index / Row: row number
        std::uint32_t count{0};        // elements
    };
    struct RowSlot {
        std::uint32_t row{0};
        std::vector<std::uint8_t> incoming;
        std::uint32_t staged{0};    // bytes granted from the write domain
        std::uint32_t consumed{0};  // bytes granted from the read domain
        bool ready{false};
        bool processed{false};
    };

    // -- request plumbing --------------------------------------------------
    bool issue_read(
        XferKind kind,
        std::uint64_t address,
        std::uint64_t length,
        hw::dataplane::AccessKind access,
        const std::string& label,
        std::uint64_t cycle,
        std::uint32_t row = 0,
        std::uint32_t local_start = 0);
    bool issue_write(
        XferKind kind,
        std::uint64_t address,
        const std::vector<std::uint8_t>& data,
        hw::dataplane::AccessKind access,
        const std::string& label,
        std::uint64_t cycle,
        std::uint32_t local_start = 0);
    void drain_inbox();
    void finish_terminal(
        hw::command::StatusCode status, const std::string& reason, std::uint64_t cycle);

    // -- stages ------------------------------------------------------------
    void step_pvector(std::uint64_t cycle);
    void step_vector_reads(std::uint64_t cycle);
    void step_update_reads(std::uint64_t cycle);
    void step_rows(std::uint64_t cycle);
    void step_writes(std::uint64_t cycle);
    void process_row(RowSlot& slot);

    bool validate_pvector(std::string* reason) const;
    bool check_finite(
        const std::vector<float>& values, const char* what, std::string* reason) const;
    // (local index, count) runs of consecutive update variables.
    void build_update_runs();
    // Locate the run chunk containing the given element offset.
    bool locate_update_chunk(
        std::uint32_t offset,
        std::uint32_t* local_start,
        std::uint32_t* count) const;

    std::uint32_t unit_id_{0};
    hw::dataplane::MemoryService* memory_{nullptr};
    SolveConfig config_;
    hw::dataplane::OnChipArbiter* arbiter_{nullptr};

    std::optional<ExecRequest> active_;
    std::optional<ExecResponse> response_;
    std::deque<hw::dataplane::MemoryResponse> inbox_;
    std::map<std::uint64_t, Xfer> xfers_;
    bool aborting_{false};

    // Transaction state.
    Phase phase_{Phase::PVector};
    bool forward_{true};
    std::uint32_t total_dim_{0};
    std::uint32_t pivot_dim_{0};
    std::uint32_t update_dim_{0};
    std::uint32_t first_variable_{0};
    std::vector<std::uint32_t> variables_;
    std::vector<std::int32_t> pvector_;
    std::vector<float> rhs_pivot_;     // b_hat over the node range (forward)
    std::vector<float> pivot_values_;  // w over the node range
    std::vector<float> z_;             // forward: r then z; backward: x
    std::vector<float> update_values_; // w over the update segment
    std::vector<float> update_acc_;    // sum_j L21[t][j] z_j (forward)
    std::vector<std::pair<std::uint32_t, std::uint32_t>> update_runs_;
    std::vector<std::uint32_t> row_sequence_;
    std::vector<RowSlot> row_slots_;
    std::uint32_t row_issue_{0};
    std::uint32_t row_consume_{0};
    std::uint32_t update_issue_offset_{0};
    std::uint32_t update_write_offset_{0};
    bool pvector_requested_{false};
    bool rhs_issued_{false};
    bool wpivot_issued_{false};
    bool writes_issued_{false};
    bool workspace_issued_{false};
    bool pivot_written_{false};
    std::size_t expected_vector_reads_{0};
    std::size_t completed_vector_reads_{0};

    // Statistics.
    std::uint64_t bytes_read_{0};
    std::uint64_t bytes_written_{0};
    std::uint64_t compute_cycles_{0};
    std::uint64_t staged_bytes_{0};
    std::uint64_t consumed_bytes_{0};
    std::uint64_t start_cycle_{0};
    std::uint64_t last_confirmed_cycle_{0};
    std::string pending_reason_;
    hw::command::StatusCode pending_status_{hw::command::StatusCode::Ok};
};

}  // namespace hw::controlplane
