#include "controlplane/assembly_exec_unit.hpp"

#include <cmath>
#include <cstring>
#include <set>

namespace hw::controlplane {
namespace dp = hw::dataplane;
namespace cmd = hw::command;

std::string validate_contribution(const dp::ContributionView& c,
                                  const dp::FrontView& front,
                                  std::uint32_t node_id) {
    const auto n = front.total_dim;
    const auto d = c.row_count;
    if (c.parent_id != node_id || c.child_id == c.parent_id ||
        d == 0 || d != c.col_count || d > n ||
        c.row_map.size() != d || c.col_map.size() != d) {
        return "contribution identity, square dimensions or map lengths are invalid";
    }
    if (c.format != cmd::DataFormat::Fp32 || c.source.format != cmd::DataFormat::Fp32 ||
        c.target.format != cmd::DataFormat::Fp32 ||
        c.source.layout != cmd::DataLayout::RowMajor || c.target.layout != cmd::DataLayout::RowMajor ||
        c.source.rows != d || c.source.cols != d ||
        c.source_stride != c.source.row_stride ||
        c.target_stride != c.target.row_stride) {
        return "contribution FP32 source dimensions or declared strides are inconsistent";
    }
    if (c.target.descriptor_id != front.front.descriptor_id ||
        c.target.base_addr != front.front.base_addr || c.target.byte_size != front.front.byte_size ||
        c.target.row_stride != front.front.row_stride ||
        c.target.rows < n || c.target.cols < n ||
        c.source.descriptor_id == c.target.descriptor_id) {
        return "contribution target is not the resident front region";
    }
    std::set<std::uint32_t> source_indices, target_indices;
    for (std::uint32_t i = 0; i < d; ++i) {
        if (c.row_map[i] >= d || c.col_map[i] >= n ||
            !source_indices.insert(c.row_map[i]).second ||
            !target_indices.insert(c.col_map[i]).second) {
            return "contribution maps are out of range or contain duplicate indices";
        }
    }
    return {};
}

AssemblyExecutionUnit::AssemblyExecutionUnit(std::uint32_t id, dp::BufferManager* buffers,
                                           dp::OnChipArbiter* arbiter, AssemblyConfig config)
    : unit_id_(id), buffers_(buffers), arbiter_(arbiter), config_(config) {}

bool AssemblyExecutionUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) return false;
    cycle_ = cycle;
    active_ = request;
    progress_ = ExecResponse{};
    progress_.txn_id = request.txn_id;
    progress_.epoch = request.epoch;
    progress_.command_id = request.command_id;
    progress_.node_id = request.node_id;
    progress_.unit_id = unit_id_;
    progress_.start_cycle = cycle;
    row_ = col_ = 0;
    phase_ = Phase::ReadChild;
    const auto reason = validate_contribution(request.contribution, request.front, request.node_id);
    if (!reason.empty() || request.opcode != cmd::Opcode::AssembleExtendAdd ||
        request.total_dim != request.front.total_dim ||
        request.update_dim != request.contribution.col_count ||
        !request.has_front_lease || !request.has_temp_lease ||
        !request.front_lease.valid || !request.front_lease.write ||
        !request.temp_lease.valid || request.temp_lease.write ||
        request.front_slot == request.temp_slot ||
        request.front_slot.slot != request.front_lease.slot ||
        request.front_slot.generation != request.front_lease.generation ||
        request.temp_slot.slot != request.temp_lease.slot ||
        request.temp_slot.generation != request.temp_lease.generation ||
        buffers_ == nullptr || arbiter_ == nullptr || config_.add_latency_cycles == 0) {
        finish(cmd::StatusCode::BadDescriptor,
               reason.empty() ? "assembly request/leases/configuration are invalid" : reason, cycle);
    }
    return true;
}

std::uint64_t AssemblyExecutionUnit::source_offset() const {
    const auto& r = *active_;
    return r.temp_offset +
        (static_cast<std::uint64_t>(r.contribution.row_map[row_]) * r.update_dim +
         r.contribution.row_map[col_]) * 4;
}

std::uint64_t AssemblyExecutionUnit::target_offset() const {
    const auto& r = *active_;
    return r.front_offset +
        (static_cast<std::uint64_t>(r.contribution.col_map[row_]) * r.total_dim +
         r.contribution.col_map[col_]) * 4;
}

void AssemblyExecutionUnit::tick(std::uint64_t cycle) {
    cycle_ = cycle;
    if (!active_) return;
    const auto& r = *active_;
    // Leases authorize asynchronous scratchpad access only after ready_cycle.
    if (cycle < r.front_lease.ready_cycle || cycle < r.temp_lease.ready_cycle) return;
    std::string reason;
    if (phase_ == Phase::ReadChild || phase_ == Phase::ReadParent) {
        if (arbiter_->acquire(dp::OnChipDomain::Read, dp::OnChipClientKind::Assemble,
                              unit_id_, 4) < 4) return;
        const bool source = phase_ == Phase::ReadChild;
        std::vector<std::uint8_t> bytes;
        const auto status = buffers_->lease_read(
            source ? r.temp_lease.lease_id : r.front_lease.lease_id,
            source ? source_offset() : target_offset(), 4, bytes, cycle, &reason);
        if (status != cmd::StatusCode::Ok) {
            finish(status, reason, cycle);
            return;
        }
        progress_.bytes_read += 4;
        float value;
        std::memcpy(&value, bytes.data(), 4);
        if (!std::isfinite(value)) {
            finish(cmd::StatusCode::NonFiniteValue, "non-finite assembly operand", cycle);
            return;
        }
        if (source) {
            child_ = value;
            phase_ = Phase::ReadParent;
        } else {
            parent_ = value;
            phase_ = Phase::Add;
            add_ready_ = cycle + config_.add_latency_cycles;
        }
    } else if (phase_ == Phase::Add) {
        if (cycle < add_ready_) return;
        result_ = parent_ + child_;
        if (!std::isfinite(result_)) {
            finish(cmd::StatusCode::NonFiniteValue, "FP32 extend-add overflow", cycle);
            return;
        }
        phase_ = Phase::WriteParent;
    } else {
        if (arbiter_->acquire(dp::OnChipDomain::Write, dp::OnChipClientKind::Assemble,
                              unit_id_, 4) < 4) return;
        std::vector<std::uint8_t> bytes(4);
        std::memcpy(bytes.data(), &result_, 4);
        const auto status = buffers_->lease_write(
            r.front_lease.lease_id, target_offset(), bytes, cycle, &reason);
        if (status != cmd::StatusCode::Ok) {
            finish(status, reason, cycle);
            return;
        }
        progress_.bytes_written += 4;
        if (++col_ == r.update_dim) {
            col_ = 0;
            ++row_;
        }
        if (row_ == r.update_dim) finish(cmd::StatusCode::Ok, {}, cycle);
        else phase_ = Phase::ReadChild;
    }
}

void AssemblyExecutionUnit::clear_pending() {
    if (arbiter_ == nullptr) return;
    arbiter_->clear_pending(dp::OnChipDomain::Read, dp::OnChipClientKind::Assemble, unit_id_);
    arbiter_->clear_pending(dp::OnChipDomain::Write, dp::OnChipClientKind::Assemble, unit_id_);
}

void AssemblyExecutionUnit::finish(cmd::StatusCode status, const std::string& reason,
                                  std::uint64_t cycle) {
    progress_.status = status;
    progress_.reason = reason;
    progress_.finish_cycle = cycle;
    response_ = progress_;
    response_ready_ = cycle + config_.response_hold_cycles;
    active_.reset();
    clear_pending();
}

bool AssemblyExecutionUnit::poll_response(ExecResponse& response) {
    if (!response_ || cycle_ < response_ready_) return false;
    response = *response_;
    response_.reset();
    return true;
}

void AssemblyExecutionUnit::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    cycle_ = cycle;
    if (active_ && active_->txn_id == txn_id) {
        finish(cmd::StatusCode::Aborted, "assembly transaction aborted", cycle);
    } else if (response_ && response_->txn_id == txn_id) {
        response_->status = cmd::StatusCode::Aborted;
        response_->reason = "assembly transaction aborted before response consumption";
    }
}

void AssemblyExecutionUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    cycle_ = cycle;
    if ((active_ && active_->epoch == epoch) || (response_ && response_->epoch == epoch)) reset();
}

std::uint32_t AssemblyExecutionUnit::active_command_id() const {
    return active_ ? active_->command_id : (response_ ? response_->command_id : cmd::NONE);
}

void AssemblyExecutionUnit::reset() {
    active_.reset();
    response_.reset();
    clear_pending();
}

}  // namespace hw::controlplane
