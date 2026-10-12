#include "controlplane/solve_exec_unit.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

namespace {

constexpr std::uint32_t kSolveMaxInflight = 8;

std::uint64_t dot_cycles(std::uint32_t terms, const SolveConfig& config) {
    if (terms == 0) {
        return 0;
    }
    return (terms + config.lanes - 1) / config.lanes + config.reduce_latency;
}

float read_f32(const std::uint8_t* bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

void append_f32(std::vector<std::uint8_t>& out, float value) {
    std::uint8_t bytes[sizeof(float)];
    std::memcpy(bytes, &value, sizeof(value));
    out.insert(out.end(), bytes, bytes + sizeof(value));
}

std::vector<float> decode_f32(const std::vector<std::uint8_t>& bytes) {
    std::vector<float> values(bytes.size() / 4, 0.0f);
    if (!bytes.empty()) {
        std::memcpy(values.data(), bytes.data(), bytes.size());
    }
    return values;
}

}  // namespace

SolveExecutionUnit::SolveExecutionUnit(
    std::uint32_t unit_id,
    dp::MemoryService* memory,
    SolveConfig config)
    : unit_id_(unit_id), memory_(memory), config_(config) {}

std::uint64_t SolveExecutionUnit::scratch_bytes(
    std::uint32_t total_dim,
    std::uint32_t pivot_dim,
    std::uint32_t depth) {
    return 4ull * (depth * total_dim + 2ull * total_dim + pivot_dim);
}

bool SolveExecutionUnit::busy() const {
    return active_.has_value() || response_.has_value();
}

void SolveExecutionUnit::deliver_memory_response(const dp::MemoryResponse& response) {
    if (xfers_.count(response.request_id) == 0) {
        return;  // late/foreign response: precise routing only
    }
    inbox_.push_back(response);
}

bool SolveExecutionUnit::handles_request(std::uint64_t request_id) const {
    return xfers_.count(request_id) != 0;
}

std::size_t SolveExecutionUnit::pending_request_count() const {
    return xfers_.size();
}

bool SolveExecutionUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) {
        return false;
    }
    // Explicit terminal refusal: no memory access, no partial state.
    const auto refuse = [&](cmd::StatusCode status, const std::string& reason) {
        ExecResponse failure;
        failure.txn_id = request.txn_id;
        failure.epoch = request.epoch;
        failure.command_id = request.command_id;
        failure.node_id = request.node_id;
        failure.unit_id = unit_id_;
        failure.status = status;
        failure.reason = reason;
        failure.start_cycle = cycle;
        failure.finish_cycle = cycle;
        response_ = failure;
        return true;
    };
    const bool is_forward = request.opcode == cmd::Opcode::SolveForward;
    const bool is_backward = request.opcode == cmd::Opcode::SolveBackward;
    if (memory_ == nullptr || (!is_forward && !is_backward)) {
        return refuse(
            cmd::StatusCode::BadCommand,
            "solve unit requires a memory service and a "
            "SOLVE_FORWARD/SOLVE_BACKWARD opcode");
    }
    const auto& view = request.solve;
    if (view.factor.total_dim == 0 || view.factor.pivot_dim == 0 ||
        view.factor.pivot_dim > view.factor.total_dim) {
        return refuse(
            cmd::StatusCode::BadDescriptor, "SOLVE_DESC factor dimensions are invalid");
    }
    if (view.variables.size() != view.factor.total_dim) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "SOLVE_DESC front variable payload is missing or has the wrong length");
    }
    if (view.format != cmd::DataFormat::Fp32 ||
        view.factor.format != cmd::DataFormat::Fp32) {
        return refuse(cmd::StatusCode::BadDescriptor, "solve data path must be FP32");
    }
    if (view.rhs_count != 1) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "this solve version supports exactly one RHS");
    }
    if ((view.direction == cmd::SolveDirection::Forward) != is_forward) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "SOLVE_DESC direction disagrees with the command opcode");
    }
    if (!request.has_temp_lease) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "solve unit requires the solve scratch write lease");
    }
    const std::uint64_t required =
        scratch_bytes(view.factor.total_dim, view.factor.pivot_dim,
                      config_.row_buffer_depth);
    if (request.temp_lease.length < required ||
        (request.range_bytes != 0 && request.range_bytes < required)) {
        return refuse(
            cmd::StatusCode::BufferFull,
            "solve scratch is smaller than the row buffer depth requires");
    }

    active_ = request;
    forward_ = is_forward;
    total_dim_ = view.factor.total_dim;
    pivot_dim_ = view.factor.pivot_dim;
    update_dim_ = total_dim_ - pivot_dim_;
    variables_ = view.variables;
    first_variable_ = variables_.front();
    pvector_.assign(pivot_dim_, 0);
    rhs_pivot_.assign(pivot_dim_, 0.0f);
    pivot_values_.assign(pivot_dim_, 0.0f);
    z_.assign(pivot_dim_, 0.0f);
    update_values_.assign(update_dim_, 0.0f);
    update_acc_.assign(update_dim_, 0.0f);
    build_update_runs();
    row_sequence_.clear();
    if (forward_) {
        for (std::uint32_t row = 0; row < total_dim_; ++row) {
            row_sequence_.push_back(row);
        }
    } else {
        for (std::uint32_t row = pivot_dim_; row > 0; --row) {
            row_sequence_.push_back(row - 1);
        }
    }
    row_slots_.assign(config_.row_buffer_depth, RowSlot{});
    row_issue_ = 0;
    row_consume_ = 0;
    update_issue_offset_ = 0;
    update_write_offset_ = 0;
    pvector_requested_ = false;
    rhs_issued_ = false;
    wpivot_issued_ = false;
    writes_issued_ = false;
    workspace_issued_ = false;
    pivot_written_ = false;
    expected_vector_reads_ = 0;
    completed_vector_reads_ = 0;
    phase_ = Phase::PVector;
    aborting_ = false;
    pending_status_ = cmd::StatusCode::Ok;
    pending_reason_.clear();
    bytes_read_ = 0;
    bytes_written_ = 0;
    compute_cycles_ = 0;
    staged_bytes_ = 0;
    consumed_bytes_ = 0;
    start_cycle_ = cycle;
    last_confirmed_cycle_ = 0;
    return true;
}

void SolveExecutionUnit::build_update_runs() {
    update_runs_.clear();
    std::uint32_t local = 0;
    while (local < update_dim_) {
        std::uint32_t count = 1;
        while (local + count < update_dim_ &&
               variables_[pivot_dim_ + local + count] ==
                   variables_[pivot_dim_ + local + count - 1] + 1) {
            ++count;
        }
        update_runs_.emplace_back(local, count);
        local += count;
    }
}

bool SolveExecutionUnit::locate_update_chunk(
    std::uint32_t offset,
    std::uint32_t* local_start,
    std::uint32_t* count) const {
    std::uint32_t cursor = 0;
    for (const auto& run : update_runs_) {
        if (offset < cursor + run.second) {
            *local_start = run.first + (offset - cursor);
            *count = std::min(
                cursor + run.second - offset, config_.max_elements_per_request);
            return true;
        }
        cursor += run.second;
    }
    return false;
}

bool SolveExecutionUnit::issue_read(
    XferKind kind,
    std::uint64_t address,
    std::uint64_t length,
    dp::AccessKind access,
    const std::string& label,
    std::uint64_t cycle,
    std::uint32_t row,
    std::uint32_t local_start) {
    dp::MemoryRequest request;
    request.request_id = memory_->next_request_id();
    request.client = dp::MemoryClient::ExecUnit;
    request.write = false;
    request.address = address;
    request.length = length;
    request.kind = access;
    request.label = label;
    if (!memory_->submit(request, cycle)) {
        return false;  // queue backpressure: retried next tick
    }
    Xfer xfer;
    xfer.kind = kind;
    xfer.row = row;
    xfer.local_start = local_start;
    xfer.count = static_cast<std::uint32_t>(length / 4);
    xfers_[request.request_id] = xfer;
    bytes_read_ += length;
    return true;
}

bool SolveExecutionUnit::issue_write(
    XferKind kind,
    std::uint64_t address,
    const std::vector<std::uint8_t>& data,
    dp::AccessKind access,
    const std::string& label,
    std::uint64_t cycle,
    std::uint32_t local_start) {
    dp::MemoryRequest request;
    request.request_id = memory_->next_request_id();
    request.client = dp::MemoryClient::ExecUnit;
    request.write = true;
    request.address = address;
    request.length = data.size();
    request.kind = access;
    request.write_data = data;
    request.label = label;
    if (!memory_->submit(request, cycle)) {
        return false;
    }
    Xfer xfer;
    xfer.kind = kind;
    xfer.local_start = local_start;
    xfer.count = static_cast<std::uint32_t>(data.size() / 4);
    xfers_[request.request_id] = xfer;
    bytes_written_ += data.size();
    return true;
}

void SolveExecutionUnit::finish_terminal(
    cmd::StatusCode status, const std::string& reason, std::uint64_t cycle) {
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
    response.finish_cycle =
        std::max({cycle, last_confirmed_cycle_, start_cycle_ + compute_cycles_});
    response_ = response;
    active_.reset();
    xfers_.clear();
    row_slots_.clear();
    inbox_.clear();
    aborting_ = false;
    writes_issued_ = false;
}

bool SolveExecutionUnit::validate_pvector(std::string* reason) const {
    bool sentinel = false;
    std::vector<bool> seen(pivot_dim_, false);
    for (const auto entry : pvector_) {
        if (entry == hw::command::PVECTOR_INIT_VALUE) {
            sentinel = true;
            continue;
        }
        if (entry < 0 || static_cast<std::uint32_t>(entry) >= pivot_dim_ ||
            seen[static_cast<std::size_t>(entry)]) {
            *reason =
                "P-vector is not a permutation of 0..p-1 inside the pivot block";
            return false;
        }
        seen[static_cast<std::size_t>(entry)] = true;
    }
    if (sentinel) {
        *reason = "P-vector entry is still the -1 sentinel (factor not written)";
        return false;
    }
    return true;
}

bool SolveExecutionUnit::check_finite(
    const std::vector<float>& values, const char* what, std::string* reason) const {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            *reason = std::string("non-finite value in ") + what;
            return false;
        }
    }
    return true;
}

void SolveExecutionUnit::drain_inbox() {
    while (!inbox_.empty() && active_.has_value()) {
        dp::MemoryResponse response = std::move(inbox_.front());
        inbox_.pop_front();
        const auto mapping = xfers_.find(response.request_id);
        if (mapping == xfers_.end()) {
            continue;  // stale/foreign
        }
        const Xfer xfer = mapping->second;
        xfers_.erase(mapping);
        last_confirmed_cycle_ =
            std::max(last_confirmed_cycle_, response.finish_cycle);
        if (response.status != cmd::StatusCode::Ok) {
            if (pending_status_ == cmd::StatusCode::Ok) {
                pending_status_ = response.status;
                pending_reason_ = response.reason.empty()
                                      ? "solve memory access failed"
                                      : response.reason;
            }
            continue;
        }
        switch (xfer.kind) {
        case XferKind::PVector: {
            if (response.read_data.size() != static_cast<std::size_t>(pivot_dim_) * 4) {
                pending_status_ = cmd::StatusCode::BadDescriptor;
                pending_reason_ = "P-vector read returned the wrong length";
                break;
            }
            pvector_.resize(pivot_dim_);
            std::memcpy(
                pvector_.data(), response.read_data.data(),
                static_cast<std::size_t>(pivot_dim_) * 4);
            std::string reason;
            if (!validate_pvector(&reason)) {
                pending_status_ = cmd::StatusCode::BadDescriptor;
                pending_reason_ = reason;
                break;
            }
            expected_vector_reads_ = forward_ ? 2 : 1;
            completed_vector_reads_ = 0;
            phase_ = Phase::VectorReads;
            break;
        }
        case XferKind::RhsPivot: {
            rhs_pivot_ = decode_f32(response.read_data);
            std::string reason;
            if (!check_finite(rhs_pivot_, "the RHS block", &reason)) {
                pending_status_ = cmd::StatusCode::NonFiniteValue;
                pending_reason_ = reason;
                break;
            }
            ++completed_vector_reads_;
            break;
        }
        case XferKind::WPivot: {
            pivot_values_ = decode_f32(response.read_data);
            std::string reason;
            if (!check_finite(pivot_values_, "the solution accumulator", &reason)) {
                pending_status_ = cmd::StatusCode::NonFiniteValue;
                pending_reason_ = reason;
                break;
            }
            ++completed_vector_reads_;
            break;
        }
        case XferKind::WUpdateRead: {
            const auto values = decode_f32(response.read_data);
            if (values.size() != xfer.count ||
                static_cast<std::size_t>(xfer.local_start) + values.size() >
                    update_values_.size()) {
                pending_status_ = cmd::StatusCode::AddressFault;
                pending_reason_ = "update-segment read returned the wrong length";
                break;
            }
            std::copy(
                values.begin(), values.end(),
                update_values_.begin() + xfer.local_start);
            break;
        }
        case XferKind::Row: {
            if (xfer.row < row_slots_.size() && row_slots_[xfer.row].incoming.empty()) {
                RowSlot& slot = row_slots_[xfer.row];
                slot.row = xfer.local_start;
                slot.incoming = std::move(response.read_data);
                slot.staged = 0;
                slot.consumed = 0;
                slot.ready = false;
                slot.processed = false;
            }
            break;
        }
        case XferKind::WUpdateWrite:
        case XferKind::Workspace:
        case XferKind::WPivotWrite:
            break;  // write confirmations only gate the terminal
        }
        if (pending_status_ != cmd::StatusCode::Ok) {
            break;
        }
    }
}

void SolveExecutionUnit::step_pvector(std::uint64_t cycle) {
    if (pvector_requested_) {
        return;
    }
    const auto& view = active_->solve;
    if (issue_read(
            XferKind::PVector,
            view.p.base_addr,
            static_cast<std::uint64_t>(pivot_dim_) * 4,
            dp::AccessKind::Int32Data,
            view.p.name,
            cycle)) {
        pvector_requested_ = true;
    }
}

void SolveExecutionUnit::step_vector_reads(std::uint64_t cycle) {
    const auto& view = active_->solve;
    if (forward_ && !rhs_issued_) {
        if (issue_read(
                XferKind::RhsPivot,
                view.rhs.base_addr + static_cast<std::uint64_t>(first_variable_) * 4,
                static_cast<std::uint64_t>(pivot_dim_) * 4,
                dp::AccessKind::Fp32Data,
                view.rhs.name,
                cycle)) {
            rhs_issued_ = true;
        }
    }
    if (!wpivot_issued_) {
        if (issue_read(
                XferKind::WPivot,
                view.solution.base_addr + static_cast<std::uint64_t>(first_variable_) * 4,
                static_cast<std::uint64_t>(pivot_dim_) * 4,
                dp::AccessKind::Fp32Data,
                view.solution.name,
                cycle)) {
            wpivot_issued_ = true;
        }
    }
    if (forward_ && !rhs_issued_) {
        return;
    }
    if (!wpivot_issued_ || completed_vector_reads_ < expected_vector_reads_) {
        return;
    }
    // Gather the forward RHS inside the pivot block (P-vector direction).
    if (forward_) {
        std::vector<float> gathered(pivot_dim_, 0.0f);
        for (std::uint32_t row = 0; row < pivot_dim_; ++row) {
            const auto source = static_cast<std::size_t>(pvector_[row]);
            gathered[row] = rhs_pivot_[source] + pivot_values_[source];
        }
        if (!check_finite(gathered, "the gathered RHS block", &pending_reason_)) {
            pending_status_ = cmd::StatusCode::NonFiniteValue;
            return;
        }
        z_ = std::move(gathered);
    }
    phase_ = update_dim_ == 0 ? Phase::Rows : Phase::UpdateReads;
}

void SolveExecutionUnit::step_update_reads(std::uint64_t cycle) {
    const auto& view = active_->solve;
    while (update_issue_offset_ < update_dim_ && xfers_.size() < kSolveMaxInflight) {
        std::uint32_t local_start = 0;
        std::uint32_t count = 0;
        if (!locate_update_chunk(update_issue_offset_, &local_start, &count)) {
            pending_status_ = cmd::StatusCode::AddressFault;
            pending_reason_ = "update-segment layout is inconsistent";
            return;
        }
        const std::uint32_t variable = variables_[pivot_dim_ + local_start];
        if (!issue_read(
                XferKind::WUpdateRead,
                view.solution.base_addr + static_cast<std::uint64_t>(variable) * 4,
                static_cast<std::uint64_t>(count) * 4,
                dp::AccessKind::Fp32Data,
                view.solution.name,
                cycle,
                0,
                local_start)) {
            return;  // queue backpressure
        }
        update_issue_offset_ += count;
    }
    if (update_issue_offset_ < update_dim_ || !xfers_.empty()) {
        return;
    }
    std::string reason;
    if (update_dim_ > 0 &&
        !check_finite(update_values_, "the update accumulator", &reason)) {
        pending_status_ = cmd::StatusCode::NonFiniteValue;
        pending_reason_ = reason;
        return;
    }
    phase_ = Phase::Rows;
}

void SolveExecutionUnit::step_rows(std::uint64_t cycle) {
    const auto& view = active_->solve;
    // 1) Issue row reads up to the configured depth.
    while (row_issue_ < row_sequence_.size() &&
           row_issue_ - row_consume_ < config_.row_buffer_depth &&
           xfers_.size() < kSolveMaxInflight) {
        const std::uint32_t slot = row_issue_ % config_.row_buffer_depth;
        const std::uint32_t row = row_sequence_[row_issue_];
        const auto& factor = view.factor;
        const std::uint64_t address = forward_
            ? factor.l.base_addr +
                  static_cast<std::uint64_t>(row) * factor.l.row_stride
            : factor.u.base_addr +
                  static_cast<std::uint64_t>(row) * factor.u.row_stride;
        const std::uint32_t length = forward_ ? pivot_dim_ : total_dim_;
        if (!issue_read(
                XferKind::Row,
                address,
                static_cast<std::uint64_t>(length) * 4,
                dp::AccessKind::Fp32Data,
                forward_ ? factor.l.name : factor.u.name,
                cycle,
                slot,
                row)) {
            break;  // queue backpressure
        }
        ++row_issue_;
    }
    // 2) Stage received rows into the scratch model under the write budget.
    for (auto& slot : row_slots_) {
        if (slot.incoming.empty() || slot.ready) {
            continue;
        }
        const auto remaining = static_cast<std::uint32_t>(slot.incoming.size()) -
                               slot.staged;
        std::uint32_t budget = remaining;
        if (arbiter_ != nullptr) {
            budget = arbiter_->acquire(
                dp::OnChipDomain::Write, dp::OnChipClientKind::Solve, unit_id_,
                remaining);
            if (budget == 0) {
                break;  // not selected this cycle
            }
        }
        budget = std::min(budget, remaining);
        slot.staged += budget;
        staged_bytes_ += budget;
        if (slot.staged == slot.incoming.size()) {
            slot.ready = true;
        }
    }
    // 3) Consume ready rows in order and run the substitutions.
    for (auto& slot : row_slots_) {
        if (slot.incoming.empty() || !slot.ready || slot.processed ||
            row_consume_ >= row_sequence_.size()) {
            continue;
        }
        if (slot.row != row_sequence_[row_consume_]) {
            continue;  // never consume a later row first
        }
        const auto remaining = static_cast<std::uint32_t>(slot.incoming.size()) -
                               slot.consumed;
        std::uint32_t budget = remaining;
        if (arbiter_ != nullptr) {
            budget = arbiter_->acquire(
                dp::OnChipDomain::Read, dp::OnChipClientKind::Solve, unit_id_,
                remaining);
            if (budget == 0) {
                break;  // not selected this cycle
            }
        }
        budget = std::min(budget, remaining);
        slot.consumed += budget;
        consumed_bytes_ += budget;
        if (slot.consumed < slot.incoming.size()) {
            continue;
        }
        process_row(slot);
        slot.processed = true;
        slot.incoming.clear();
        ++row_consume_;
        if (pending_status_ != cmd::StatusCode::Ok) {
            return;
        }
    }
    if (row_consume_ < row_sequence_.size() || !xfers_.empty()) {
        return;
    }
    phase_ = Phase::Writes;
}

void SolveExecutionUnit::process_row(RowSlot& slot) {
    const std::uint32_t row = slot.row;
    const auto* bytes = slot.incoming.data();
    const auto row_value = [&](std::uint32_t col) {
        return read_f32(bytes + static_cast<std::size_t>(col) * 4);
    };
    if (forward_) {
        if (row < pivot_dim_) {
            float value = z_[row];
            compute_cycles_ += dot_cycles(row, config_);
            for (std::uint32_t col = 0; col < row; ++col) {
                const float multiplier = row_value(col);
                if (!std::isfinite(multiplier)) {
                    pending_status_ = cmd::StatusCode::NonFiniteValue;
                    pending_reason_ = "non-finite L multiplier in the forward solve";
                    return;
                }
                value -= multiplier * z_[col];
            }
            if (!std::isfinite(value)) {
                pending_status_ = cmd::StatusCode::NonFiniteValue;
                pending_reason_ = "forward solve produced a non-finite z";
                return;
            }
            z_[row] = value;
        } else {
            const std::uint32_t update = row - pivot_dim_;
            float accumulated = 0.0f;
            compute_cycles_ += dot_cycles(pivot_dim_, config_);
            for (std::uint32_t col = 0; col < pivot_dim_; ++col) {
                const float multiplier = row_value(col);
                if (!std::isfinite(multiplier)) {
                    pending_status_ = cmd::StatusCode::NonFiniteValue;
                    pending_reason_ = "non-finite L21 multiplier in the forward solve";
                    return;
                }
                accumulated += multiplier * z_[col];
            }
            if (!std::isfinite(accumulated)) {
                pending_status_ = cmd::StatusCode::NonFiniteValue;
                pending_reason_ = "forward update accumulation overflowed";
                return;
            }
            update_acc_[update] += accumulated;
        }
        return;
    }
    // Backward: solve U11 x = c row by row, from the bottom up.
    float value = pivot_values_[row];
    compute_cycles_ +=
        dot_cycles(total_dim_ - 1 - row, config_) + config_.div_latency;
    for (std::uint32_t col = row + 1; col < pivot_dim_; ++col) {
        const float coefficient = row_value(col);
        if (!std::isfinite(coefficient)) {
            pending_status_ = cmd::StatusCode::NonFiniteValue;
            pending_reason_ = "non-finite U11 coefficient in the backward solve";
            return;
        }
        value -= coefficient * z_[col];
    }
    for (std::uint32_t col = pivot_dim_; col < total_dim_; ++col) {
        const float coefficient = row_value(col);
        if (!std::isfinite(coefficient)) {
            pending_status_ = cmd::StatusCode::NonFiniteValue;
            pending_reason_ = "non-finite U12 coefficient in the backward solve";
            return;
        }
        value -= coefficient * update_values_[col - pivot_dim_];
    }
    const float diagonal = row_value(row);
    if (diagonal == 0.0f) {
        // No replacement, no continuation (design doc §4.4).
        pending_status_ = cmd::StatusCode::StaticPivotFailure;
        pending_reason_ = "backward solve met a zero U diagonal";
        return;
    }
    if (!std::isfinite(diagonal) || !std::isfinite(value)) {
        pending_status_ = cmd::StatusCode::NonFiniteValue;
        pending_reason_ = "backward solve produced a non-finite value";
        return;
    }
    const float solution = value / diagonal;
    if (!std::isfinite(solution)) {
        pending_status_ = cmd::StatusCode::NonFiniteValue;
        pending_reason_ = "backward solve division overflowed";
        return;
    }
    z_[row] = solution;
}

void SolveExecutionUnit::step_writes(std::uint64_t cycle) {
    if (writes_issued_) {
        return;
    }
    const auto& view = active_->solve;
    // 1) Update-segment write-back (forward only): w = read - accumulated.
    if (forward_) {
        while (update_write_offset_ < update_dim_ &&
               xfers_.size() < kSolveMaxInflight) {
            std::uint32_t local_start = 0;
            std::uint32_t count = 0;
            if (!locate_update_chunk(update_write_offset_, &local_start, &count)) {
                pending_status_ = cmd::StatusCode::AddressFault;
                pending_reason_ = "update-segment layout is inconsistent";
                return;
            }
            std::vector<std::uint8_t> data;
            data.reserve(static_cast<std::size_t>(count) * 4);
            for (std::uint32_t index = 0; index < count; ++index) {
                const std::uint32_t local = local_start + index;
                const float value = update_values_[local] - update_acc_[local];
                if (!std::isfinite(value)) {
                    pending_status_ = cmd::StatusCode::NonFiniteValue;
                    pending_reason_ =
                        "forward update write produced a non-finite value";
                    return;
                }
                append_f32(data, value);
            }
            const std::uint32_t variable = variables_[pivot_dim_ + local_start];
            if (!issue_write(
                    XferKind::WUpdateWrite,
                    view.solution.base_addr +
                        static_cast<std::uint64_t>(variable) * 4,
                    data,
                    dp::AccessKind::Fp32Data,
                    view.solution.name,
                    cycle,
                    local_start)) {
                return;  // retry next tick
            }
            update_write_offset_ += count;
        }
        if (update_write_offset_ < update_dim_) {
            return;
        }
    }
    // 2) Workspace row 0 = forward z (checker input for the forward backward
    //    error). The backward command must NOT overwrite it with x.
    if (forward_ && !workspace_issued_) {
        std::vector<std::uint8_t> data;
        data.reserve(static_cast<std::size_t>(pivot_dim_) * 4);
        for (const auto value : z_) {
            append_f32(data, value);
        }
        if (!issue_write(
                XferKind::Workspace,
                view.workspace.base_addr,
                data,
                dp::AccessKind::Fp32Data,
                view.workspace.name,
                cycle)) {
            return;
        }
        workspace_issued_ = true;
    }
    // 3) Pivot-block result write (z for forward, x for backward).
    if (!pivot_written_) {
        std::vector<std::uint8_t> data;
        data.reserve(static_cast<std::size_t>(pivot_dim_) * 4);
        for (const auto value : z_) {
            if (!std::isfinite(value)) {
                pending_status_ = cmd::StatusCode::NonFiniteValue;
                pending_reason_ = "solve result is not finite";
                return;
            }
            append_f32(data, value);
        }
        if (!issue_write(
                XferKind::WPivotWrite,
                view.solution.base_addr +
                    static_cast<std::uint64_t>(first_variable_) * 4,
                data,
                dp::AccessKind::Fp32Data,
                view.solution.name,
                cycle)) {
            return;
        }
        pivot_written_ = true;
    }
    writes_issued_ = true;
}

void SolveExecutionUnit::tick(std::uint64_t cycle) {
    if (!active_.has_value()) {
        inbox_.clear();
        return;
    }
    drain_inbox();
    if (!active_.has_value() || response_.has_value()) {
        return;
    }
    if (aborting_) {
        // Stop issuing new traffic; drain the accepted requests, then the
        // single terminal ABORTED (in-flight writes are not rolled back).
        if (arbiter_ != nullptr) {
            arbiter_->clear_pending(
                dp::OnChipDomain::Read, dp::OnChipClientKind::Solve, unit_id_);
            arbiter_->clear_pending(
                dp::OnChipDomain::Write, dp::OnChipClientKind::Solve, unit_id_);
        }
        if (xfers_.empty()) {
            finish_terminal(cmd::StatusCode::Aborted, "node was aborted", cycle);
        }
        return;
    }
    if (pending_status_ != cmd::StatusCode::Ok) {
        if (xfers_.empty()) {
            finish_terminal(pending_status_, pending_reason_, cycle);
        }
        return;
    }
    // One bounded pass through the state machine (all steps are idempotent
    // and retried on the next cycle).
    for (int guard = 0; guard < 16; ++guard) {
        const Phase before = phase_;
        switch (phase_) {
        case Phase::PVector: step_pvector(cycle); break;
        case Phase::VectorReads: step_vector_reads(cycle); break;
        case Phase::UpdateReads: step_update_reads(cycle); break;
        case Phase::Rows: step_rows(cycle); break;
        case Phase::Writes: step_writes(cycle); break;
        }
        if (pending_status_ != cmd::StatusCode::Ok) {
            break;
        }
        if (phase_ == before || phase_ == Phase::Rows) {
            break;  // response-driven or no progress this cycle
        }
    }
    if (pending_status_ != cmd::StatusCode::Ok) {
        if (xfers_.empty()) {
            finish_terminal(pending_status_, pending_reason_, cycle);
        }
        return;
    }
    if (phase_ == Phase::Writes && writes_issued_ && xfers_.empty()) {
        finish_terminal(cmd::StatusCode::Ok, "", cycle);
    }
}

bool SolveExecutionUnit::poll_response(ExecResponse& response) {
    if (!response_.has_value()) {
        return false;
    }
    response = *response_;
    response_.reset();
    return true;
}

void SolveExecutionUnit::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    if (response_.has_value() && response_->txn_id == txn_id) {
        // Never delete the unconsumed terminal: it becomes the ABORTED
        // terminal of the same transaction and is still consumed exactly once.
        response_->status = cmd::StatusCode::Aborted;
        response_->reason = "transaction aborted before response consumption";
        response_->needs_pvector_write = false;
        return;
    }
    if (!active_.has_value() || active_->txn_id != txn_id) {
        return;
    }
    aborting_ = true;
    if (arbiter_ != nullptr) {
        arbiter_->clear_pending(
            dp::OnChipDomain::Read, dp::OnChipClientKind::Solve, unit_id_);
        arbiter_->clear_pending(
            dp::OnChipDomain::Write, dp::OnChipClientKind::Solve, unit_id_);
    }
    if (xfers_.empty()) {
        finish_terminal(cmd::StatusCode::Aborted, "node was aborted", cycle);
    }
}

void SolveExecutionUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)epoch;
    (void)cycle;
    // Global reset semantics are owned by CommandDevice::reset().
}

void SolveExecutionUnit::reset() {
    active_.reset();
    response_.reset();
    inbox_.clear();
    xfers_.clear();
    row_slots_.clear();
    aborting_ = false;
    phase_ = Phase::PVector;
    pending_status_ = cmd::StatusCode::Ok;
    pending_reason_.clear();
    if (arbiter_ != nullptr) {
        arbiter_->clear_pending(
            dp::OnChipDomain::Read, dp::OnChipClientKind::Solve, unit_id_);
        arbiter_->clear_pending(
            dp::OnChipDomain::Write, dp::OnChipClientKind::Solve, unit_id_);
    }
}

}  // namespace hw::controlplane
