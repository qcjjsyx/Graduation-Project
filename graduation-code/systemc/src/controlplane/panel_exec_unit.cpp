#include "controlplane/panel_exec_unit.hpp"

#include <algorithm>
#include <cstring>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

namespace {

constexpr std::uint32_t kFactorChunkRows = 4;
constexpr std::uint32_t kFactorMaxInflight = 8;

}  // namespace

// ---------------------------------------------------------------------------
// PanelExecutionUnit
// ---------------------------------------------------------------------------

PanelExecutionUnit::PanelExecutionUnit(
    std::uint32_t unit_id,
    hw::panel_lu::PanelEngine* engine)
    : unit_id_(unit_id), engine_(engine) {}

bool PanelExecutionUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) {
        return false;
    }
    if (engine_ == nullptr) {
        return false;
    }
    hw::panel_lu::PanelEngineRequest engine_request;
    engine_request.txn_id = request.txn_id;
    engine_request.epoch = request.epoch;
    engine_request.command_id = request.command_id;
    engine_request.node_id = request.node_id;
    engine_request.total_dim = request.total_dim;
    engine_request.pivot_dim = request.front.pivot_dim;
    engine_request.tile_size = request.front.tile_size;
    engine_request.policy = request.policy;
    engine_request.threshold_bits = request.threshold_bits;
    engine_request.front = request.front_slot;
    engine_request.front_lease = request.front_lease;
    engine_request.front_offset = request.front_offset;
    engine_request.label = request.front.front.name;

    const hw::panel_lu::ConfigCheck check =
        engine_->begin_command(engine_request, unit_id_, cycle);
    if (check.ok) {
        ++accepted_count_;
        return true;
    }
    if (check.status == cmd::StatusCode::BufferFull) {
        // No free engine context: a resource stall, retried by the device.
        return false;
    }
    // Unsupported/invalid request: accept and terminate with an explicit
    // failure instead of silently stalling or substituting another backend.
    ExecResponse failure;
    failure.txn_id = request.txn_id;
    failure.epoch = request.epoch;
    failure.command_id = request.command_id;
    failure.node_id = request.node_id;
    failure.unit_id = unit_id_;
    failure.status = check.status;
    failure.reason = check.reason;
    failure.start_cycle = cycle;
    failure.finish_cycle = cycle;
    immediate_ = failure;
    ++accepted_count_;
    return true;
}

void PanelExecutionUnit::tick(std::uint64_t cycle) {
    // Shared engine tick is driven exactly once per cycle by CommandDevice.
    (void)cycle;
}

bool PanelExecutionUnit::poll_response(ExecResponse& response) {
    if (immediate_.has_value()) {
        response = *immediate_;
        immediate_.reset();
        return true;
    }
    if (engine_ == nullptr) {
        return false;
    }
    hw::panel_lu::PanelEngineResponse engine_response;
    if (!engine_->poll_response(unit_id_, engine_response)) {
        return false;
    }
    response.txn_id = engine_response.txn_id;
    response.epoch = engine_response.epoch;
    response.command_id = engine_response.command_id;
    response.node_id = engine_response.node_id;
    response.unit_id = unit_id_;
    response.status = engine_response.status;
    response.reason = engine_response.reason;
    response.pivot_count = engine_response.pivot_count;
    response.overflow_count = 0;
    response.bytes_read = engine_response.bytes_read;
    response.bytes_written = engine_response.bytes_written;
    response.start_cycle = engine_response.start_cycle;
    response.finish_cycle = engine_response.finish_cycle;
    response.needs_pvector_write = engine_response.needs_pvector_write;
    response.pvector_dim = engine_response.pvector_dim;
    return true;
}

bool PanelExecutionUnit::busy() const {
    if (immediate_.has_value()) {
        return true;
    }
    return engine_ != nullptr && engine_->unit_busy(unit_id_);
}

void PanelExecutionUnit::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    if (immediate_.has_value() && immediate_->txn_id == txn_id) {
        immediate_.reset();
    }
    if (engine_ != nullptr) {
        engine_->abort_txn(txn_id, cycle);
    }
}

void PanelExecutionUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)epoch;
    (void)cycle;
    // Global reset semantics are owned by CommandDevice::reset(), which resets
    // the shared engine directly; a directed node abort must use abort_txn.
}

void PanelExecutionUnit::reset() {
    immediate_.reset();
}

// ---------------------------------------------------------------------------
// FactorStoreUnit
// ---------------------------------------------------------------------------

FactorStoreUnit::FactorStoreUnit(
    std::uint32_t unit_id,
    dp::BufferManager* buffers,
    dp::MemoryService* memory)
    : unit_id_(unit_id), buffers_(buffers), memory_(memory) {}

bool FactorStoreUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) {
        return false;
    }
    if (buffers_ == nullptr || memory_ == nullptr || !request.has_front_lease) {
        // Binding/authorization problem: explicit terminal failure instead of
        // an endless stall or a substituted path.
        ExecResponse failure;
        failure.txn_id = request.txn_id;
        failure.epoch = request.epoch;
        failure.command_id = request.command_id;
        failure.node_id = request.node_id;
        failure.unit_id = unit_id_;
        failure.status = cmd::StatusCode::BadDescriptor;
        failure.reason = "factor unit requires a buffer manager and a front read lease";
        failure.start_cycle = cycle;
        failure.finish_cycle = cycle;
        response_ = failure;
        return true;
    }
    if (request.factor.l.rows < request.factor.pivot_dim ||
        request.factor.u.cols < request.factor.pivot_dim ||
        request.factor.pivot_dim == 0) {
        // Malformed factor view: accept and fail explicitly.
        ExecResponse failure;
        failure.txn_id = request.txn_id;
        failure.epoch = request.epoch;
        failure.command_id = request.command_id;
        failure.node_id = request.node_id;
        failure.unit_id = unit_id_;
        failure.status = cmd::StatusCode::BadDescriptor;
        failure.reason = "factor view is too small for the pivot block";
        failure.start_cycle = cycle;
        failure.finish_cycle = cycle;
        response_ = failure;
        return true;
    }
    active_ = request;
    row_ = 0;
    // Transaction-local partial-row flags must never leak across commands: an
    // abort between the L and U submit of the same row leaves them set.
    l_submitted_ = false;
    u_submitted_ = false;
    pending_ids_.clear();
    aborting_ = false;
    pending_status_ = cmd::StatusCode::Ok;
    pending_reason_.clear();
    bytes_read_ = 0;
    bytes_written_ = 0;
    start_cycle_ = cycle;
    last_confirmed_cycle_ = 0;
    return true;
}

void FactorStoreUnit::tick(std::uint64_t cycle) {
    if (!active_.has_value()) {
        inbox_.clear();
        return;
    }
    // Settle finished memory writes first: only ids that belong to this
    // transaction count (precise routing, no cross-instance decay).
    while (!inbox_.empty()) {
        const dp::MemoryResponse response = inbox_.front();
        inbox_.pop_front();
        if (pending_ids_.erase(response.request_id) == 0) {
            continue;  // not ours (late/foreign response)
        }
        last_confirmed_cycle_ = std::max(last_confirmed_cycle_, response.finish_cycle);
        if (response.status != cmd::StatusCode::Ok &&
            pending_status_ == cmd::StatusCode::Ok) {
            pending_status_ = response.status;
            pending_reason_ = response.reason;
        }
    }
    if (pending_status_ != cmd::StatusCode::Ok) {
        // A permanent memory error: wait for the remaining accepted writes to
        // settle, then terminate as failed (no silent partial success).
        if (pending_ids_.empty()) {
            finish_terminal(pending_status_, pending_reason_, cycle);
        }
        return;
    }
    if (aborting_) {
        // Stop issuing new writes; only drain the accepted ones, then report
        // the single terminal ABORTED response.
        if (pending_ids_.empty()) {
            finish_terminal(cmd::StatusCode::Aborted, "node was aborted", cycle);
        }
        return;
    }
    std::string reason;
    while (active_.has_value() && row_ < active_->factor.pivot_dim &&
           pending_ids_.size() < kFactorMaxInflight) {
        const SubmitOutcome outcome = submit_next_chunk(cycle, &reason);
        if (outcome == SubmitOutcome::Error) {
            // Permanent error (lease/range); drain accepted writes first.
            pending_status_ = cmd::StatusCode::AddressFault;
            pending_reason_ = reason;
            if (pending_ids_.empty()) {
                finish_terminal(pending_status_, pending_reason_, cycle);
            }
            return;
        }
        if (outcome == SubmitOutcome::Backpressure) {
            return;  // normal queue backpressure: retry next cycle
        }
    }
    if (active_.has_value() && row_ >= active_->factor.pivot_dim &&
        pending_ids_.empty()) {
        finish_terminal(cmd::StatusCode::Ok, "", cycle);
    }
}

FactorStoreUnit::SubmitOutcome FactorStoreUnit::submit_next_chunk(
    std::uint64_t cycle,
    std::string* reason) {
    const ExecRequest& request = *active_;
    const std::uint32_t pivot = request.factor.pivot_dim;
    const std::uint32_t rows = std::min(kFactorChunkRows, pivot - row_);
    const std::uint64_t front_row_stride = static_cast<std::uint64_t>(request.total_dim) * 4;
    for (std::uint32_t done = 0; done < rows; ++done) {
        const std::uint32_t row = row_;
        std::vector<std::uint8_t> packed;
        const std::uint64_t src =
            request.front_offset + static_cast<std::uint64_t>(row) * front_row_stride;
        if (buffers_->lease_read(
                request.front_lease.lease_id,
                src,
                static_cast<std::uint64_t>(pivot) * 4,
                packed,
                cycle,
                reason) != cmd::StatusCode::Ok) {
            return SubmitOutcome::Error;
        }
        bytes_read_ += packed.size();
        std::vector<std::uint8_t> l_row(static_cast<std::size_t>(pivot) * 4, 0);
        std::vector<std::uint8_t> u_row(static_cast<std::size_t>(pivot) * 4, 0);
        for (std::uint32_t c = 0; c < pivot; ++c) {
            float value = 0.0f;
            std::memcpy(&value, packed.data() + static_cast<std::size_t>(c) * 4, 4);
            float l_value = 0.0f;
            float u_value = 0.0f;
            if (c < row) {
                l_value = value;  // strict lower triangle
            } else if (c == row) {
                l_value = 1.0f;   // explicit unit diagonal
                u_value = value;   // U keeps the diagonal
            } else {
                u_value = value;  // strict upper triangle
            }
            std::memcpy(&l_row[static_cast<std::size_t>(c) * 4], &l_value, 4);
            std::memcpy(&u_row[static_cast<std::size_t>(c) * 4], &u_value, 4);
        }

        // Per-row partial progress: a queue-full retry must not resubmit a
        // write that was already accepted (duplicate bytes/stat drift).
        if (!l_submitted_) {
            dp::MemoryRequest l_request;
            l_request.request_id = memory_->next_request_id();
            l_request.client = dp::MemoryClient::ExecUnit;
            l_request.write = true;
            l_request.address = request.factor.l.base_addr +
                                static_cast<std::uint64_t>(row) * request.factor.l.row_stride;
            l_request.length = static_cast<std::uint64_t>(pivot) * 4;
            l_request.kind = dp::AccessKind::Fp32Data;
            l_request.write_data = l_row;
            l_request.has_command = true;
            l_request.command_id = request.command_id;
            l_request.node_id = request.node_id;
            l_request.label = request.factor.l.name;
            if (!memory_->submit(l_request, cycle)) {
                return SubmitOutcome::Backpressure;
            }
            l_submitted_ = true;
            pending_ids_.insert(l_request.request_id);
            bytes_written_ += l_row.size();
        }

        if (!u_submitted_) {
            dp::MemoryRequest u_request;
            u_request.request_id = memory_->next_request_id();
            u_request.client = dp::MemoryClient::ExecUnit;
            u_request.write = true;
            u_request.address = request.factor.u.base_addr +
                                static_cast<std::uint64_t>(row) * request.factor.u.row_stride;
            u_request.length = static_cast<std::uint64_t>(pivot) * 4;
            u_request.kind = dp::AccessKind::Fp32Data;
            u_request.write_data = u_row;
            u_request.has_command = true;
            u_request.command_id = request.command_id;
            u_request.node_id = request.node_id;
            u_request.label = request.factor.u.name;
            if (!memory_->submit(u_request, cycle)) {
                return SubmitOutcome::Backpressure;
            }
            u_submitted_ = true;
            pending_ids_.insert(u_request.request_id);
            bytes_written_ += u_row.size();
        }
        ++row_;
        l_submitted_ = false;
        u_submitted_ = false;
    }
    return SubmitOutcome::Progress;
}

void FactorStoreUnit::finish_terminal(
    cmd::StatusCode status,
    const std::string& reason,
    std::uint64_t cycle) {
    if (!active_.has_value()) {
        return;
    }
    ExecResponse response;
    response.txn_id = active_->txn_id;
    response.epoch = active_->epoch;
    response.command_id = active_->command_id;
    response.node_id = active_->node_id;
    response.unit_id = unit_id_;
    response.status = status;
    response.reason = reason;
    response.bytes_read = bytes_read_;
    response.bytes_written = bytes_written_;
    response.start_cycle = start_cycle_;
    response.finish_cycle = std::max(cycle, last_confirmed_cycle_);
    response_ = response;
    active_.reset();
    pending_ids_.clear();
    aborting_ = false;
    // Terminal cleanup keeps the partial-row state consistent with the next
    // transaction's fresh start.
    l_submitted_ = false;
    u_submitted_ = false;
}

bool FactorStoreUnit::poll_response(ExecResponse& response) {
    if (!response_.has_value()) {
        return false;
    }
    response = *response_;
    response_.reset();
    return true;
}

bool FactorStoreUnit::busy() const {
    return active_.has_value() || response_.has_value();
}

void FactorStoreUnit::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    if (response_.has_value() && response_->txn_id == txn_id) {
        response_.reset();
    }
    if (!active_.has_value() || active_->txn_id != txn_id) {
        return;
    }
    // In-flight DDR writes cannot be cancelled and are not rolled back; stop
    // issuing new ones and produce the terminal ABORTED only after every
    // accepted write has settled (busy() stays true until then).
    aborting_ = true;
    if (pending_ids_.empty()) {
        finish_terminal(cmd::StatusCode::Aborted, "node was aborted", cycle);
    }
}

void FactorStoreUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)epoch;
    (void)cycle;
    // Global reset semantics are handled by CommandDevice::reset().
}

void FactorStoreUnit::reset() {
    active_.reset();
    response_.reset();
    inbox_.clear();
    pending_ids_.clear();
    aborting_ = false;
    row_ = 0;
    l_submitted_ = false;
    u_submitted_ = false;
    last_confirmed_cycle_ = 0;
}

void FactorStoreUnit::deliver_memory_response(const dp::MemoryResponse& response) {
    inbox_.push_back(response);
}

bool FactorStoreUnit::handles_request(std::uint64_t request_id) const {
    return pending_ids_.count(request_id) != 0;
}

std::size_t FactorStoreUnit::pending_request_count() const {
    return pending_ids_.size();
}

}  // namespace hw::controlplane
