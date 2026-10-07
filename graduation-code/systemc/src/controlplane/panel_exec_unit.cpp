#include "controlplane/panel_exec_unit.hpp"

#include <algorithm>
#include <cstring>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

namespace {

constexpr std::uint32_t kFactorMaxInflight = 8;

}  // namespace

// ---------------------------------------------------------------------------
// PanelExecutionUnit
// ---------------------------------------------------------------------------

namespace {
// Adapter kind codes for the shared engine's identity (kind, unit_id). Unit
// ids repeat across kinds, so routing must include the kind.
constexpr std::uint32_t kEngineKindPanel = 1;
constexpr std::uint32_t kEngineKindTrsm = 2;
constexpr std::uint32_t kEngineKindGemm = 3;
}  // namespace

PanelExecutionUnit::PanelExecutionUnit(
    std::uint32_t unit_id,
    hw::panel_lu::PanelEngine* engine)
    : unit_id_(unit_id),
      adapter_key_(hw::panel_lu::PanelEngine::adapter_key(kEngineKindPanel, unit_id)),
      engine_(engine) {}

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
        engine_->begin_command(engine_request, adapter_key_, cycle);
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
    if (!engine_->poll_response(adapter_key_, engine_response)) {
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
    response.front_scale = engine_response.front_scale;
    response.threshold_product = engine_response.threshold_product;
    return true;
}

bool PanelExecutionUnit::busy() const {
    if (immediate_.has_value()) {
        return true;
    }
    return engine_ != nullptr && engine_->unit_busy(adapter_key_);
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
// TrsmExecutionUnit
// ---------------------------------------------------------------------------

TrsmExecutionUnit::TrsmExecutionUnit(
    std::uint32_t unit_id,
    hw::panel_lu::PanelEngine* engine)
    : unit_id_(unit_id),
      adapter_key_(hw::panel_lu::PanelEngine::adapter_key(kEngineKindTrsm, unit_id)),
      engine_(engine) {}

namespace {
// The three KERNEL_DESC operands must resolve to the RESIDENT front region of
// the node whose command is being dispatched. A cross-node/other-region
// reference is a contract violation and must be refused before dispatch.
bool kernel_operands_match_front(
    const ExecRequest& request,
    std::string* reason) {
    const hw::dataplane::RegionView& front = request.front.front;
    const auto matches = [&](const hw::dataplane::RegionView& operand) {
        return operand.descriptor_id == front.descriptor_id &&
               operand.base_addr == front.base_addr &&
               operand.byte_size == front.byte_size;
    };
    if (!matches(request.kernel.a) || !matches(request.kernel.b) ||
        !matches(request.kernel.c)) {
        if (reason != nullptr) {
            *reason =
                "TRSM KERNEL operands do not reference the resident front";
        }
        return false;
    }
    return true;
}
}  // namespace

bool TrsmExecutionUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) {
        return false;
    }
    if (engine_ == nullptr) {
        return false;
    }
    std::string operand_reason;
    if (!kernel_operands_match_front(request, &operand_reason)) {
        // Explicit terminal refusal: no engine context, no data access, no
        // READY token; the front lease is released by the device.
        ExecResponse failure;
        failure.txn_id = request.txn_id;
        failure.epoch = request.epoch;
        failure.command_id = request.command_id;
        failure.node_id = request.node_id;
        failure.unit_id = unit_id_;
        failure.status = cmd::StatusCode::BadDescriptor;
        failure.reason = operand_reason;
        failure.start_cycle = cycle;
        failure.finish_cycle = cycle;
        immediate_ = failure;
        ++accepted_count_;
        return true;
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
    engine_request.compute_kind = request.opcode == cmd::Opcode::TrsmLeft
                                      ? hw::panel_lu::ComputeKind::TrsmLeft
                                      : hw::panel_lu::ComputeKind::TrsmRight;
    engine_request.kernel_m = request.kernel.m;
    engine_request.kernel_n = request.kernel.n;
    engine_request.kernel_k = request.kernel.k;
    engine_request.kernel_tile_size = request.kernel.tile_size;
    engine_request.kernel_backend = static_cast<std::uint32_t>(request.kernel.backend);
    engine_request.front_scale = request.front_scale;
    engine_request.threshold_product = request.threshold_product;

    const hw::panel_lu::ConfigCheck check =
        engine_->begin_command(engine_request, adapter_key_, cycle);
    if (check.ok) {
        ++accepted_count_;
        return true;
    }
    if (check.status == cmd::StatusCode::BufferFull) {
        return false;  // no free context/group: resource stall
    }
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

void TrsmExecutionUnit::tick(std::uint64_t cycle) {
    (void)cycle;  // the shared engine is ticked once by CommandDevice
}

bool TrsmExecutionUnit::poll_response(ExecResponse& response) {
    if (immediate_.has_value()) {
        response = *immediate_;
        immediate_.reset();
        return true;
    }
    if (engine_ == nullptr) {
        return false;
    }
    hw::panel_lu::PanelEngineResponse engine_response;
    if (!engine_->poll_response(adapter_key_, engine_response)) {
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
    response.needs_pvector_write = false;
    response.front_scale = engine_response.front_scale;
    response.threshold_product = engine_response.threshold_product;
    return true;
}

bool TrsmExecutionUnit::busy() const {
    if (immediate_.has_value()) {
        return true;
    }
    return engine_ != nullptr && engine_->unit_busy(adapter_key_);
}

void TrsmExecutionUnit::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    if (immediate_.has_value() && immediate_->txn_id == txn_id) {
        immediate_.reset();
    }
    if (engine_ != nullptr) {
        engine_->abort_txn(txn_id, cycle);
    }
}

void TrsmExecutionUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)epoch;
    (void)cycle;  // global reset is owned by CommandDevice::reset()
}

void TrsmExecutionUnit::reset() {
    immediate_.reset();
}

// ---------------------------------------------------------------------------
// GemmExecutionUnit
// ---------------------------------------------------------------------------

GemmExecutionUnit::GemmExecutionUnit(
    std::uint32_t unit_id,
    hw::panel_lu::PanelEngine* engine)
    : unit_id_(unit_id),
      adapter_key_(hw::panel_lu::PanelEngine::adapter_key(kEngineKindGemm, unit_id)),
      engine_(engine) {}

bool GemmExecutionUnit::submit(const ExecRequest& request, std::uint64_t cycle) {
    if (busy()) {
        return false;
    }
    // Explicit terminal refusal helpers: no engine context, no data access,
    // no READY token; the front lease is released by the device.
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
        immediate_ = failure;
        ++accepted_count_;
        return true;
    };
    if (engine_ == nullptr) {
        return false;
    }
    if (request.opcode != cmd::Opcode::GemmSchur) {
        return refuse(cmd::StatusCode::BadCommand, "GEMM unit received a non-GEMM opcode");
    }
    // KERNEL_DESC operands must resolve to the resident front region.
    const hw::dataplane::RegionView& front = request.front.front;
    const auto matches_front = [&](const hw::dataplane::RegionView& operand) {
        return operand.descriptor_id == front.descriptor_id &&
               operand.base_addr == front.base_addr &&
               operand.byte_size == front.byte_size;
    };
    if (!matches_front(request.kernel.a) || !matches_front(request.kernel.b) ||
        !matches_front(request.kernel.c)) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "GEMM KERNEL operands do not reference the resident front");
    }
    const std::uint32_t update = request.total_dim - request.front.pivot_dim;
    if (request.kernel.m != update || request.kernel.n != update ||
        request.kernel.k != request.front.pivot_dim) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "GEMM KERNEL M/N/K must equal u/u/p of the resident front");
    }
    if (request.kernel.tile_size != 16) {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "GEMM KERNEL tile_size is not the fixed physical 16 tile");
    }
    const std::uint32_t wire = static_cast<std::uint32_t>(request.kernel.backend);
    const std::uint32_t fp32_wire = static_cast<std::uint32_t>(
        cmd::KernelBackend::SystemCFp32DeviceModel);
    const std::uint32_t int32_wire = static_cast<std::uint32_t>(
        cmd::KernelBackend::SystemCInt32GemmModel);
    if (wire == fp32_wire) {
        if (request.kernel.has_scale) {
            return refuse(
                cmd::StatusCode::BadDescriptor,
                "FP32 GEMM must not reference a SCALE_DESC (scale_desc_id must be NONE)");
        }
    } else if (wire == int32_wire) {
        if (!request.kernel.has_scale) {
            return refuse(
                cmd::StatusCode::BadDescriptor,
                "INT32 GEMM requires a valid SCALE_DESC reference");
        }
        if (request.front.policy != cmd::PivotPolicy::Static) {
            return refuse(
                cmd::StatusCode::PolicyNotSupported,
                "INT32 GEMM requires the STATIC pivot policy path");
        }
    } else {
        return refuse(
            cmd::StatusCode::BadDescriptor,
            "GEMM KERNEL backend code is unknown");
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
    engine_request.compute_kind = hw::panel_lu::ComputeKind::GemmSchur;
    engine_request.kernel_m = request.kernel.m;
    engine_request.kernel_n = request.kernel.n;
    engine_request.kernel_k = request.kernel.k;
    engine_request.kernel_tile_size = request.kernel.tile_size;
    engine_request.kernel_backend = wire;
    engine_request.gemm_scale = request.kernel.scale.policy;
    engine_request.kernel_has_scale = request.kernel.has_scale;
    engine_request.kernel_descriptor_id = request.kernel.descriptor_id;
    engine_request.front_scale = request.front_scale;
    engine_request.threshold_product = request.threshold_product;

    const hw::panel_lu::ConfigCheck check =
        engine_->begin_command(engine_request, adapter_key_, cycle);
    if (check.ok) {
        ++accepted_count_;
        return true;
    }
    if (check.status == cmd::StatusCode::BufferFull) {
        return false;  // no free context/group: resource stall
    }
    return refuse(check.status, check.reason);
}

void GemmExecutionUnit::tick(std::uint64_t cycle) {
    (void)cycle;  // the shared engine is ticked once by CommandDevice
}

bool GemmExecutionUnit::poll_response(ExecResponse& response) {
    if (immediate_.has_value()) {
        response = *immediate_;
        immediate_.reset();
        return true;
    }
    if (engine_ == nullptr) {
        return false;
    }
    hw::panel_lu::PanelEngineResponse engine_response;
    if (!engine_->poll_response(adapter_key_, engine_response)) {
        return false;
    }
    response.txn_id = engine_response.txn_id;
    response.epoch = engine_response.epoch;
    response.command_id = engine_response.command_id;
    response.node_id = engine_response.node_id;
    response.unit_id = unit_id_;
    response.status = engine_response.status;
    response.reason = engine_response.reason;
    // GEMM produces no pivots: the failure coordinate is not a pivot count.
    response.pivot_count = 0;
    response.overflow_count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(engine_response.overflow_count, 0xFFFFFFFFu));
    response.bytes_read = engine_response.bytes_read;
    response.bytes_written = engine_response.bytes_written;
    response.start_cycle = engine_response.start_cycle;
    response.finish_cycle = engine_response.finish_cycle;
    response.needs_pvector_write = false;
    response.front_scale = engine_response.front_scale;
    response.threshold_product = engine_response.threshold_product;
    response.saturation_count = engine_response.saturation_count;
    response.gemm_backend = engine_response.gemm_backend;
    response.local_conversion_bytes = engine_response.local_conversion_bytes;
    response.scale_reduction_cycles = engine_response.scale_reduction_cycles;
    response.quantize_cycles = engine_response.quantize_cycles;
    response.integer_compute_cycles = engine_response.integer_compute_cycles;
    response.dequantize_cycles = engine_response.dequantize_cycles;
    response.exponent_a = engine_response.exponent_a;
    response.exponent_b = engine_response.exponent_b;
    return true;
}

bool GemmExecutionUnit::busy() const {
    if (immediate_.has_value()) {
        return true;
    }
    return engine_ != nullptr && engine_->unit_busy(adapter_key_);
}

void GemmExecutionUnit::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    if (immediate_.has_value() && immediate_->txn_id == txn_id) {
        immediate_.reset();
    }
    if (engine_ != nullptr) {
        engine_->abort_txn(txn_id, cycle);
    }
}

void GemmExecutionUnit::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)epoch;
    (void)cycle;  // global reset is owned by CommandDevice::reset()
}

void GemmExecutionUnit::reset() {
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
    const std::uint32_t total = request.total_dim;
    const std::uint32_t pivot = request.factor.pivot_dim;
    if (total == 0 || pivot == 0 || pivot > total ||
        request.factor.l.rows < total || request.factor.l.cols < pivot ||
        request.factor.u.rows < pivot || request.factor.u.cols < total) {
        ExecResponse failure;
        failure.txn_id = request.txn_id;
        failure.epoch = request.epoch;
        failure.command_id = request.command_id;
        failure.node_id = request.node_id;
        failure.unit_id = unit_id_;
        failure.status = cmd::StatusCode::BadDescriptor;
        failure.reason = "factor view regions are too small for n x p / p x n";
        failure.start_cycle = cycle;
        failure.finish_cycle = cycle;
        response_ = failure;
        return true;
    }
    active_ = request;
    row_ = 0;
    row_read_bytes_ = 0;
    l_submitted_ = false;
    u_submitted_ = false;
    front_row_.assign(static_cast<std::size_t>(total) * 4, 0);
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
        if (pending_ids_.empty()) {
            finish_terminal(pending_status_, pending_reason_, cycle);
        }
        return;
    }
    if (aborting_) {
        // Stop issuing new reads/writes; drain the accepted writes, then the
        // single terminal ABORTED response.
        if (arbiter_ != nullptr) {
            arbiter_->clear_pending(
                dp::OnChipDomain::Read, dp::OnChipClientKind::FactorRead, unit_id_);
        }
        if (pending_ids_.empty()) {
            finish_terminal(cmd::StatusCode::Aborted, "node was aborted", cycle);
        }
        return;
    }
    const std::uint32_t total = active_->total_dim;
    while (active_.has_value() && row_ < total &&
           pending_ids_.size() < kFactorMaxInflight) {
        const SubmitOutcome outcome = stage_row(cycle, &pending_reason_);
        if (outcome == SubmitOutcome::Error) {
            pending_status_ = cmd::StatusCode::AddressFault;
            if (pending_ids_.empty()) {
                finish_terminal(pending_status_, pending_reason_, cycle);
            }
            return;
        }
        if (outcome == SubmitOutcome::Backpressure) {
            return;  // normal queue/credit backpressure: retry next cycle
        }
    }
    if (active_.has_value() && row_ >= total && pending_ids_.empty()) {
        finish_terminal(cmd::StatusCode::Ok, "", cycle);
    }
}

FactorStoreUnit::SubmitOutcome FactorStoreUnit::stage_row(
    std::uint64_t cycle,
    std::string* reason) {
    const ExecRequest& request = *active_;
    const std::uint32_t total = request.total_dim;
    const std::uint32_t pivot = request.factor.pivot_dim;
    const std::uint64_t front_row_stride =
        static_cast<std::uint64_t>(total) * 4;

    // 1) Stage the packed front row under the unified read budget. The grant
    //    is acquired BEFORE any byte moves; denied credit keeps the row
    //    progress for the next cycle.
    if (row_read_bytes_ < front_row_stride) {
        const std::uint32_t remaining =
            static_cast<std::uint32_t>(front_row_stride - row_read_bytes_);
        std::uint32_t budget = remaining;
        if (arbiter_ != nullptr) {
            budget = arbiter_->acquire(
                dp::OnChipDomain::Read,
                dp::OnChipClientKind::FactorRead,
                unit_id_,
                remaining);
            if (budget == 0) {
                return SubmitOutcome::Backpressure;
            }
        }
        budget = std::min(budget, remaining);
        budget -= budget % 4;
        if (budget == 0) {
            return SubmitOutcome::Backpressure;
        }
        const std::uint64_t src = request.front_offset +
                                  static_cast<std::uint64_t>(row_) * front_row_stride +
                                  row_read_bytes_;
        std::vector<std::uint8_t> bytes;
        if (buffers_->lease_read(
                request.front_lease.lease_id,
                src,
                budget,
                bytes,
                cycle,
                reason) != cmd::StatusCode::Ok) {
            return SubmitOutcome::Error;
        }
        std::memcpy(front_row_.data() + row_read_bytes_, bytes.data(), bytes.size());
        row_read_bytes_ += static_cast<std::uint32_t>(bytes.size());
        bytes_read_ += bytes.size();
        if (row_read_bytes_ < front_row_stride) {
            return SubmitOutcome::Backpressure;  // more credit needed this row
        }
    }

    // 2) Build the L row (n x p) and, for r < p, the U row (p x n) from the
    //    staged packed front row. L11 is unit-lower with an explicit diagonal;
    //    U11 is upper including the diagonal; L21/U12 are the raw solved
    //    values written back by the top-level TRSM commands.
    const auto packed_value = [&](std::uint32_t col) {
        float value = 0.0f;
        std::memcpy(&value, front_row_.data() + static_cast<std::size_t>(col) * 4, 4);
        return value;
    };
    std::vector<std::uint8_t> l_row(static_cast<std::size_t>(pivot) * 4, 0);
    for (std::uint32_t col = 0; col < pivot; ++col) {
        float value = 0.0f;
        if (row_ < pivot) {
            if (col < row_) {
                value = packed_value(col);
            } else if (col == row_) {
                value = 1.0f;
            }
        } else {
            value = packed_value(col);  // L21 raw solved values
        }
        std::memcpy(&l_row[static_cast<std::size_t>(col) * 4], &value, 4);
    }
    const bool needs_u = row_ < pivot;
    std::vector<std::uint8_t> u_row;
    if (needs_u) {
        u_row.assign(static_cast<std::size_t>(total) * 4, 0);
        for (std::uint32_t col = 0; col < total; ++col) {
            float value = 0.0f;
            if (col < pivot) {
                if (col >= row_) {
                    value = packed_value(col);  // U11 upper incl diagonal
                }
            } else {
                value = packed_value(col);  // U12 raw solved values
            }
            std::memcpy(&u_row[static_cast<std::size_t>(col) * 4], &value, 4);
        }
    }

    // 3) Submit the row writes with per-row partial progress: a queue-full
    //    retry must not resubmit an accepted write.
    if (!l_submitted_) {
        dp::MemoryRequest l_request;
        l_request.request_id = memory_->next_request_id();
        l_request.client = dp::MemoryClient::ExecUnit;
        l_request.write = true;
        l_request.address = request.factor.l.base_addr +
                            static_cast<std::uint64_t>(row_) *
                                request.factor.l.row_stride;
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

    if (needs_u && !u_submitted_) {
        dp::MemoryRequest u_request;
        u_request.request_id = memory_->next_request_id();
        u_request.client = dp::MemoryClient::ExecUnit;
        u_request.write = true;
        u_request.address = request.factor.u.base_addr +
                            static_cast<std::uint64_t>(row_) *
                                request.factor.u.row_stride;
        u_request.length = static_cast<std::uint64_t>(total) * 4;
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
    row_read_bytes_ = 0;
    l_submitted_ = false;
    u_submitted_ = false;
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
    l_submitted_ = false;
    u_submitted_ = false;
    row_ = 0;
    row_read_bytes_ = 0;
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
    // issuing new reads/writes and produce the terminal ABORTED only after
    // every accepted write has settled (busy() stays true until then).
    aborting_ = true;
    if (arbiter_ != nullptr) {
        arbiter_->clear_pending(
            dp::OnChipDomain::Read, dp::OnChipClientKind::FactorRead, unit_id_);
    }
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
    row_read_bytes_ = 0;
    front_row_.clear();
    l_submitted_ = false;
    u_submitted_ = false;
    last_confirmed_cycle_ = 0;
    if (arbiter_ != nullptr) {
        arbiter_->clear_pending(
            dp::OnChipDomain::Read, dp::OnChipClientKind::FactorRead, unit_id_);
    }
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
