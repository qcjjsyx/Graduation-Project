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
// and zeros below. Only the p x p panel block is written; L21/U12 belong to
// the top-level TRSM commands and are left untouched. Conversion is chunked
// through the front read lease and MemoryService writes (no full-front copy).
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

    std::uint32_t accepted_count() const { return accepted_count_; }

private:
    std::uint32_t unit_id_{0};
    hw::panel_lu::PanelEngine* engine_{nullptr};
    std::optional<ExecResponse> immediate_;
    std::uint32_t accepted_count_{0};
};

class FactorStoreUnit : public IExecutionUnit {
public:
    FactorStoreUnit(
        std::uint32_t unit_id,
        hw::dataplane::BufferManager* buffers,
        hw::dataplane::MemoryService* memory);

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
    SubmitOutcome submit_next_chunk(std::uint64_t cycle, std::string* reason);
    void finish_terminal(hw::command::StatusCode status, const std::string& reason,
                         std::uint64_t cycle);

    std::uint32_t unit_id_{0};
    hw::dataplane::BufferManager* buffers_{nullptr};
    hw::dataplane::MemoryService* memory_{nullptr};

    std::optional<ExecRequest> active_;
    std::optional<ExecResponse> response_;
    std::uint32_t row_{0};
    bool l_submitted_{false};
    bool u_submitted_{false};
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
