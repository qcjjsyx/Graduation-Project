#include "controlplane/test_stub_unit.hpp"

#include <algorithm>
#include <vector>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

const char* exec_backend_name(ExecBackend backend) {
    switch (backend) {
    case ExecBackend::TestStub: return "test_stub";
    case ExecBackend::RequireRealKernels: return "require_real_kernels";
    }
    return "unknown";
}

const char* unit_kind_name(UnitKind kind) {
    switch (kind) {
    case UnitKind::Assemble: return "assemble";
    case UnitKind::Panel: return "panel";
    case UnitKind::Trsm: return "trsm";
    case UnitKind::Gemm: return "gemm";
    case UnitKind::Solve: return "solve";
    }
    return "unknown";
}

TestStubUnit::TestStubUnit(
    UnitKind kind,
    std::uint32_t unit_id,
    StubConfig config,
    dp::BufferManager* buffers)
    : kind_(kind), unit_id_(unit_id), config_(std::move(config)), buffers_(buffers) {}

bool TestStubUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) {
        return false;
    }
    const auto found = config_.latency_by_opcode.find(request.opcode);
    const std::uint32_t latency =
        found == config_.latency_by_opcode.end()
            ? std::max<std::uint32_t>(1, config_.default_latency)
            : std::max<std::uint32_t>(1, found->second);
    active_ = request;
    busy_start_ = cycle;
    ready_cycle_ = cycle + latency;
    ++accepted_count_;
    return true;
}

bool TestStubUnit::busy() const {
    return active_.has_value() || response_.has_value();
}

void TestStubUnit::tick(std::uint64_t cycle) {
    if (active_.has_value() && !response_.has_value() && cycle >= ready_cycle_) {
        finish_execution(cycle);
    }
    if (response_.has_value() && !response_visible_ &&
        cycle >= response_ready_cycle_) {
        response_visible_ = true;
    }
}

void TestStubUnit::finish_execution(std::uint64_t cycle) {
    ExecResponse response;
    response.txn_id = active_->txn_id;
    response.epoch = active_->epoch;
    response.command_id = active_->command_id;
    response.node_id = active_->node_id;
    response.unit_id = unit_id_;
    response.start_cycle = busy_start_;
    response.finish_cycle = cycle;

    const auto injected = failures_.find(active_->command_id);
    if (injected != failures_.end()) {
        // Injected failure: deterministic latency honoured, no data written.
        response.status = injected->second.status;
        response.pivot_count = injected->second.pivot_count;
        response.overflow_count = injected->second.overflow_count;
        response.reason = "test stub error injection";
    } else {
        response.status = cmd::StatusCode::Ok;
        std::string reason;
        if (active_->opcode == cmd::Opcode::AssembleExtendAdd) {
            if (!assemble_copy(cycle, &reason)) {
                response.status = cmd::StatusCode::AddressFault;
                response.reason = reason;
            }
        } else if (config_.transform == StubTransform::Pattern &&
                   buffers_ != nullptr && active_->has_front_lease) {
            if (!apply_transform(cycle, &reason)) {
                response.status = cmd::StatusCode::AddressFault;
                response.reason = reason;
            }
        }
        if (response.status == cmd::StatusCode::Ok &&
            active_->opcode == cmd::Opcode::PanelLu &&
            active_->policy == cmd::PivotPolicy::Static) {
            // STATIC success contract: the P-vector is written as explicit
            // identity by the device after this response (and awaited). A
            // completed panel reports its full pivot count (success =
            // pivot_dim pivots processed); failures report the first failed
            // column instead.
            response.needs_pvector_write = true;
            response.pvector_dim = active_->front.pivot_dim;
            response.pivot_count = active_->front.pivot_dim;
        }
    }
    response_ = response;
    response_ready_cycle_ = cycle + config_.response_hold_cycles;
    response_visible_ = config_.response_hold_cycles == 0;
    active_.reset();
}

bool TestStubUnit::apply_transform(std::uint64_t cycle, std::string* reason) {
    const auto& request = *active_;
    const std::uint64_t length = std::min<std::uint64_t>(
        request.range_bytes, config_.pattern_bytes);
    std::vector<std::uint8_t> chunk(
        static_cast<std::size_t>(length),
        static_cast<std::uint8_t>(request.command_id & 0xFF));
    const auto status = buffers_->lease_write(
        request.front_lease.lease_id, request.front_offset, chunk, cycle, reason);
    return status == cmd::StatusCode::Ok;
}

bool TestStubUnit::assemble_copy(std::uint64_t cycle, std::string* reason) {
    // Checkable test transform: scatter-copy of the staged child update into
    // the resident parent front following the contribution maps
    // (parent[col_map[i], col_map[j]] = child[row_map[i], row_map[j]]). This
    // is NOT numerical extend-add; results must not be reported as numerics.
    const auto& request = *active_;
    const auto& contribution = request.contribution;
    if (!request.has_front_lease || !request.has_temp_lease) {
        *reason = "assemble requires front and temp leases";
        return false;
    }
    const std::uint64_t front_row_stride =
        static_cast<std::uint64_t>(request.total_dim) * 4;
    const std::uint64_t temp_row_stride =
        static_cast<std::uint64_t>(request.update_dim) * 4;
    for (std::uint32_t i = 0; i < contribution.row_count; ++i) {
        std::vector<std::uint8_t> row;
        const std::uint64_t source_row = contribution.row_map[i];
        if (buffers_->lease_read(
                request.temp_lease.lease_id,
                request.temp_offset + source_row * temp_row_stride,
                static_cast<std::uint64_t>(contribution.col_count) * 4,
                row, cycle, reason) != cmd::StatusCode::Ok) {
            return false;
        }
        for (std::uint32_t j = 0; j < contribution.col_count; ++j) {
            const std::uint64_t target_offset =
                request.front_offset +
                static_cast<std::uint64_t>(contribution.col_map[i]) * front_row_stride +
                static_cast<std::uint64_t>(contribution.col_map[j]) * 4;
            const std::vector<std::uint8_t> element(
                row.begin() + static_cast<std::ptrdiff_t>(j) * 4,
                row.begin() + static_cast<std::ptrdiff_t>(j + 1) * 4);
            if (buffers_->lease_write(
                    request.front_lease.lease_id, target_offset, element, cycle,
                    reason) != cmd::StatusCode::Ok) {
                return false;
            }
        }
    }
    return true;
}

bool TestStubUnit::poll_response(ExecResponse& response) {
    if (!response_.has_value() || !response_visible_) {
        return false;  // response stays stable while the consumer is not ready
    }
    response = *response_;
    response_.reset();
    response_visible_ = false;
    return true;
}

void TestStubUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)cycle;
    if (active_.has_value() && active_->epoch == epoch) {
        active_.reset();
    }
    if (response_.has_value() && response_->epoch == epoch) {
        response_.reset();
        response_visible_ = false;
    }
}

void TestStubUnit::reset() {
    active_.reset();
    response_.reset();
    response_visible_ = false;
    busy_start_ = 0;
    ready_cycle_ = 0;
}

void TestStubUnit::inject_failure(std::uint32_t command_id, ExecFailure failure) {
    failures_[command_id] = failure;
}

void TestStubUnit::clear_injections() {
    failures_.clear();
}

std::uint32_t TestStubUnit::active_command_id() const {
    return active_.has_value() ? active_->command_id : hw::command::NONE;
}

}  // namespace hw::controlplane
