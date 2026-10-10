#pragma once

// T07 execution-unit adapters for the real backend.
//
// PanelExecutionUnit binds one T05 unit slot to the shared PanelEngine: several
// adapters may exist (per panel_unit_count) but they all share one PE pool,
// one dispatcher and one engine tick (driven once per cycle by CommandDevice).
//
// FactorStoreUnit performs the real packed-LU -> persistent L/U conversion for
// STORE_FACTOR: L gets the strict lower triangle plus an explicit unit
// diagonal and zeros above; U gets the upper triangle including the diagonal
// and zeros below within U11. The complete L (n x p) and U (p x n) are
// written, including L21/U12 computed by the top-level TRSM commands.
// Conversion stages one front row through the front read lease and submits
// L/U row writes via MemoryService (no full-front copy).
//
// These units never fake success: unimplemented data or missing resources end
// in an explicit terminal status.

#include <cstdint>
#include <deque>
#include <set>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "controlplane/exec_types.hpp"
#include "dataplane/buffer_manager.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/on_chip_arbiter.hpp"
#include "panel_lu/panel_engine.hpp"

namespace hw::controlplane {

class PanelExecutionUnit : public IExecutionUnit {
public:
    PanelExecutionUnit(std::uint32_t unit_id, hw::panel_lu::PanelEngine* engine);

    UnitKind kind() const override { return UnitKind::Panel; }
    std::uint32_t unit_id() const override { return unit_id_; }

    bool submit(const ExecRequest& request, std::uint64_t cycle) override;
    void tick(std::uint64_t cycle) override;  // no-op: the device ticks the engine once
    bool poll_response(ExecResponse& response) override;
    bool busy() const override;
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) override;
    void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) override;
    void reset() override;

    // Command id of the accepted-but-not-yet-consumed transaction (NONE when
    // idle), whether it runs in the engine or holds an immediate terminal.
    std::uint32_t active_command_id() const override { return active_command_id_; }

    std::uint32_t accepted_count() const { return accepted_count_; }

private:
    std::uint32_t unit_id_{0};
    std::uint64_t adapter_key_{0};
    hw::panel_lu::PanelEngine* engine_{nullptr};
    std::optional<ExecResponse> immediate_;
    std::uint32_t accepted_count_{0};
    std::uint32_t active_command_id_{hw::command::NONE};
    std::uint64_t active_txn_id_{0};
};

// Top-level TRSM adapter over the same shared engine (one group per command).
class TrsmExecutionUnit : public IExecutionUnit {
public:
    TrsmExecutionUnit(std::uint32_t unit_id, hw::panel_lu::PanelEngine* engine);

    UnitKind kind() const override { return UnitKind::Trsm; }
    std::uint32_t unit_id() const override { return unit_id_; }

    bool submit(const ExecRequest& request, std::uint64_t cycle) override;
    void tick(std::uint64_t cycle) override;  // no-op: device ticks the engine
    bool poll_response(ExecResponse& response) override;
    bool busy() const override;
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) override;
    void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) override;
    void reset() override;

    // Command id of the accepted-but-not-yet-consumed transaction (NONE when
    // idle), whether it runs in the engine or holds an immediate terminal.
    std::uint32_t active_command_id() const override { return active_command_id_; }

    std::uint32_t accepted_count() const { return accepted_count_; }

private:
    std::uint32_t unit_id_{0};
    std::uint64_t adapter_key_{0};
    hw::panel_lu::PanelEngine* engine_{nullptr};
    std::optional<ExecResponse> immediate_;
    std::uint32_t accepted_count_{0};
    std::uint32_t active_command_id_{hw::command::NONE};
    std::uint64_t active_txn_id_{0};
};

// T10 top-level GEMM_SCHUR adapter over the same shared engine. It validates
// the KERNEL_DESC shape (M/N/K = u/u/p), the fixed tile, the operand identity
// and the backend/SCALE_DESC combination before a context is established:
//   FP32 backend  -> scale_desc_id MUST be NONE,
//   INT32 backend -> a valid SCALE_DESC policy is REQUIRED (already decoded
//                    and validated by the runtime DescriptorReader).
class GemmExecutionUnit : public IExecutionUnit {
public:
    GemmExecutionUnit(std::uint32_t unit_id, hw::panel_lu::PanelEngine* engine);

    UnitKind kind() const override { return UnitKind::Gemm; }
    std::uint32_t unit_id() const override { return unit_id_; }

    bool submit(const ExecRequest& request, std::uint64_t cycle) override;
    void tick(std::uint64_t cycle) override;  // no-op: device ticks the engine
    bool poll_response(ExecResponse& response) override;
    bool busy() const override;
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) override;
    void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) override;
    void reset() override;

    // Command id of the accepted-but-not-yet-consumed transaction (NONE when
    // idle), whether it runs in the engine or holds an immediate terminal.
    std::uint32_t active_command_id() const override { return active_command_id_; }

    std::uint32_t accepted_count() const { return accepted_count_; }

private:
    std::uint32_t unit_id_{0};
    std::uint64_t adapter_key_{0};
    hw::panel_lu::PanelEngine* engine_{nullptr};
    std::optional<ExecResponse> immediate_;
    std::uint32_t accepted_count_{0};
    std::uint32_t active_command_id_{hw::command::NONE};
    std::uint64_t active_txn_id_{0};
};

class FactorStoreUnit : public IExecutionUnit {
public:
    FactorStoreUnit(
        std::uint32_t unit_id,
        hw::dataplane::BufferManager* buffers,
        hw::dataplane::MemoryService* memory);

    // Unified on-chip read budget for packed-front staging (required for the
    // real integrated path; nullptr keeps the legacy standalone behaviour).
    void set_on_chip_arbiter(hw::dataplane::OnChipArbiter* arbiter) {
        arbiter_ = arbiter;
    }

    UnitKind kind() const override { return UnitKind::Factor; }
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

private:
    enum class SubmitOutcome {
        Progress,
        Backpressure,
        Error,
    };
    // Full L/U save: stage one packed front row under the unified read budget,
    // then emit the L row (n x p, L11 unit-lower + L21 raw) and, for r < p,
    // the U row (p x n, U11 upper + U12 raw) as bounded MemoryService writes.
    SubmitOutcome stage_row(std::uint64_t cycle, std::string* reason);
    void finish_terminal(hw::command::StatusCode status, const std::string& reason,
                         std::uint64_t cycle);

    std::uint32_t unit_id_{0};
    hw::dataplane::BufferManager* buffers_{nullptr};
    hw::dataplane::MemoryService* memory_{nullptr};

    std::optional<ExecRequest> active_;
    std::optional<ExecResponse> response_;
    std::uint32_t row_{0};          // current front row 0..total_dim-1
    std::uint32_t row_read_bytes_{0};
    bool l_submitted_{false};
    bool u_submitted_{false};
    std::vector<std::uint8_t> front_row_;  // staged packed front row
    hw::dataplane::OnChipArbiter* arbiter_{nullptr};
    // Precise response routing: only ids in this set belong to the active
    // transaction; unknown/late responses are ignored.
    std::set<std::uint64_t> pending_ids_;
    bool aborting_{false};
    hw::command::StatusCode pending_status_{hw::command::StatusCode::Ok};
    std::string pending_reason_;
    std::deque<hw::dataplane::MemoryResponse> inbox_;
    std::uint64_t bytes_read_{0};
    std::uint64_t bytes_written_{0};
    std::uint64_t start_cycle_{0};
    std::uint64_t last_confirmed_cycle_{0};
};

}  // namespace hw::controlplane
