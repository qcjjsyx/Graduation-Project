#include "panel_lu/gemm_controller.hpp"

#include <algorithm>

#include "panel_lu/precision_adapter.hpp"

namespace hw::panel_lu {

namespace {

std::uint32_t ceil_div16(std::uint32_t value) {
    return (value + 15) / 16;
}

}  // namespace

GemmController::GemmController(
    const PanelLuConfig& config,
    TileDispatcher* dispatcher,
    TileVersionStore* versions)
    : config_(config), dispatcher_(dispatcher), versions_(versions) {}

std::uint32_t GemmController::axis_extent(
    std::uint32_t dimension,
    std::uint32_t tile) const {
    return std::min<std::uint32_t>(16, dimension - tile * 16);
}

std::uint64_t GemmController::a21_key(
    std::uint32_t row_tile,
    std::uint32_t step) const {
    return tile_version_key(
        layout_.handle.slot,
        TileViewTag::A21,
        row_tile * pivot_tiles_ + step);
}

std::uint64_t GemmController::a12_key(
    std::uint32_t step,
    std::uint32_t col_tile) const {
    return tile_version_key(
        layout_.handle.slot,
        TileViewTag::A12,
        step * output_tiles_ + col_tile);
}

std::uint64_t GemmController::a22_key(
    std::uint32_t row_tile,
    std::uint32_t col_tile) const {
    return tile_version_key(
        layout_.handle.slot,
        TileViewTag::A22,
        row_tile * output_tiles_ + col_tile);
}

std::uint32_t GemmController::version_of(std::uint64_t key) const {
    return versions_->version(key);
}

TileOperand GemmController::a21_operand(
    std::uint32_t row_tile,
    std::uint32_t step,
    std::uint32_t version) const {
    // A21 is the u x p block anchored at row pivot_dim, column 0.
    TileOperand operand;
    operand.handle = layout_.handle;
    operand.offset =
        static_cast<std::uint64_t>(request_.pivot_dim + row_tile * 16) *
            layout_.row_stride_bytes +
        static_cast<std::uint64_t>(step) * k_block_size_ * 4;
    operand.stride_bytes = layout_.row_stride_bytes;
    operand.version = version;
    operand.version_key = a21_key(row_tile, step);
    operand.valid = true;
    return operand;
}

TileOperand GemmController::a12_operand(
    std::uint32_t step,
    std::uint32_t col_tile,
    std::uint32_t version) const {
    // A12 is the p x u block anchored at row 0, column pivot_dim.
    TileOperand operand;
    operand.handle = layout_.handle;
    operand.offset =
        static_cast<std::uint64_t>(step) * k_block_size_ * layout_.row_stride_bytes +
        static_cast<std::uint64_t>(request_.pivot_dim + col_tile * 16) * 4;
    operand.stride_bytes = layout_.row_stride_bytes;
    operand.version = version;
    operand.version_key = a12_key(step, col_tile);
    operand.valid = true;
    return operand;
}

TileOperand GemmController::a22_operand(
    std::uint32_t row_tile,
    std::uint32_t col_tile,
    std::uint32_t version) const {
    // A22 is the u x u block anchored at (pivot_dim, pivot_dim). The front
    // base offset is applied exactly once by the lease access layer.
    TileOperand operand;
    operand.handle = layout_.handle;
    operand.offset =
        static_cast<std::uint64_t>(request_.pivot_dim + row_tile * 16) *
            layout_.row_stride_bytes +
        static_cast<std::uint64_t>(request_.pivot_dim + col_tile * 16) * 4;
    operand.stride_bytes = layout_.row_stride_bytes;
    operand.version = version;
    operand.version_key = a22_key(row_tile, col_tile);
    operand.valid = true;
    return operand;
}

ConfigCheck GemmController::start(
    const PanelCommandRequest& request,
    std::uint64_t cycle,
    std::uint64_t epoch,
    const PanelFrontLayout& layout,
    std::uint64_t first_task_id) {
    ConfigCheck check;
    const std::uint32_t total = request.front_dim;
    const std::uint32_t pivot = request.pivot_dim;
    if (total == 0 || pivot == 0 || pivot >= total) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "GEMM_SCHUR requires 0 < pivot_dim < front_dim";
        return check;
    }
    if (request.tile_size != config_.tile_dim) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "GEMM_SCHUR requires the fixed 16x16 compute tile";
        return check;
    }
    if (request.compute_kind != ComputeKind::GemmSchur) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "GEMM controller received a non-GEMM compute kind";
        return check;
    }
    if (request.gemm_backend == GemmBackend::Int32Gemm) {
        // INT32 staging must not silently ride on the FP64 diagnostic path.
        if (config_.numeric_mode != PanelNumericMode::Fp32Device) {
            check.ok = false;
            check.status = StatusCode::PolicyNotSupported;
            check.reason =
                "INT32 GEMM cannot be combined with the FP64 diagnostic mode";
            return check;
        }
        const ConfigCheck precision = validate_precision_config(config_.precision);
        if (!precision.ok) {
            check = precision;
            return check;
        }
        if (!request.gemm_scale.auto_output_scale) {
            check.ok = false;
            check.status = StatusCode::PolicyNotSupported;
            check.reason = "fixed output scale is not supported by this version";
            return check;
        }
    }
    request_ = request;
    kind_ = request.compute_kind;
    epoch_ = epoch;
    layout_ = layout;
    update_dim_ = total - pivot;
    k_block_size_ = request.gemm_backend == GemmBackend::Int32Gemm
                        ? config_.precision.k_block_max
                        : 16;
    pivot_tiles_ = (pivot + k_block_size_ - 1) / k_block_size_;
    output_tiles_ = ceil_div16(update_dim_);
    num_steps_ = pivot_tiles_;
    step_ = 0;
    phase_ = Phase::StepGenerate;
    start_cycle_ = cycle;
    next_task_id_ = first_task_id;
    sequence_ = 0;
    task_graph_.clear();
    cancelled_tasks_.clear();
    outstanding_.clear();
    task_meta_.clear();
    gemm_requests_.clear();
    gemm_responses_.clear();
    abort_requested_ = false;
    cancel_requested_ = false;
    cancel_taken_ = false;
    failure_status_ = StatusCode::Ok;
    failure_reason_.clear();
    failed_coordinate_ = hw::command::NONE;
    response_.reset();
    saturation_count_ = 0;
    overflow_count_ = 0;
    local_conversion_bytes_ = 0;
    scale_reduction_cycles_ = 0;
    quantize_cycles_ = 0;
    integer_compute_cycles_ = 0;
    dequantize_cycles_ = 0;
    exponent_a_ = 0;
    exponent_b_ = 0;
    have_exponents_ = false;
    expected_per_step_ = output_tiles_ * output_tiles_;
    generated_count_ = 0;
    committed_count_ = 0;
    step_ready_ = false;
    generated_.assign(output_tiles_, std::vector<bool>(output_tiles_, false));
    committed_.assign(output_tiles_, std::vector<bool>(output_tiles_, false));
    check.ok = true;
    return check;
}

void GemmController::tick(
    std::uint64_t cycle,
    hw::dataplane::OnChipArbiter& arbiter) {
    (void)arbiter;  // this controller issues no scratch bytes of its own
    if (abort_requested_) {
        // Abort is terminal for generation: no new task and no new lease
        // access may be issued after the engine performed the directed
        // cancellation. Only the drain of already accepted tasks is awaited.
        if (outstanding_.empty() &&
            (phase_ == Phase::StepGenerate || phase_ == Phase::StepWait)) {
            phase_ = Phase::Aborted;
            publish(cycle);
        }
        return;
    }
    switch (phase_) {
        case Phase::StepGenerate:
            generate_step(cycle);
            break;
        case Phase::StepWait:
            if (committed_count_ == expected_per_step_) {
                ++step_;
                if (step_ >= num_steps_) {
                    phase_ = Phase::Done;
                    publish(cycle);
                } else {
                    // K-step barrier: the next step may only be generated
                    // after every task of this step truly committed, so the
                    // same C tile is never read-modify-written concurrently.
                    generated_.assign(output_tiles_, std::vector<bool>(output_tiles_, false));
                    committed_.assign(output_tiles_, std::vector<bool>(output_tiles_, false));
                    generated_count_ = 0;
                    committed_count_ = 0;
                    phase_ = Phase::StepGenerate;
                }
            }
            break;
        default:
            break;
    }
}

void GemmController::generate_step(std::uint64_t cycle) {
    const std::uint32_t k = std::min(
        k_block_size_, request_.pivot_dim - step_ * k_block_size_);
    for (std::uint32_t row = 0; row < output_tiles_; ++row) {
        for (std::uint32_t col = 0; col < output_tiles_; ++col) {
            if (generated_[row][col]) {
                continue;
            }
            TileTask task;
            task.task_id = next_task_id_++;
            task.epoch = epoch_;
            task.command_id = request_.command_id;
            task.node_id = request_.node_id;
            task.step = step_;
            task.tile_row = row;
            task.tile_col = col;
            task.mode = TileMode::GemmSchur;
            task.group_id = layout_.group_id;
            task.sequence = sequence_++;
            task.gemm_backend = request_.gemm_backend;
            task.gemm_scale = request_.gemm_scale;
            const std::uint32_t m = axis_extent(update_dim_, row);
            const std::uint32_t n = axis_extent(update_dim_, col);
            task.dims = {m, n, k};
            // C(i,j) -= A(i,s) * B(s,j): lhs = L21 block, rhs = U12 block,
            // dst = A22 tile (in place). Versions are read from the version
            // table: the barrier guarantees the previous step's writeback is
            // committed before the next step is generated.
            task.lhs = a21_operand(row, step_, version_of(a21_key(row, step_)));
            task.rhs = a12_operand(step_, col, version_of(a12_key(step_, col)));
            task.dst = a22_operand(row, col, version_of(a22_key(row, col)));
            task.output_version = task.dst.version + 1;
            task.mask = {task.dims.m, task.dims.n};
            if (!submit_task(task, cycle, nullptr)) {
                return;  // queue full / retry next cycle, progress preserved
            }
            generated_[row][col] = true;
            ++generated_count_;
            task_meta_[task.task_id] = TaskMeta{step_, row, col};
        }
    }
    if (generated_count_ == expected_per_step_) {
        phase_ = Phase::StepWait;
    }
}

bool GemmController::submit_task(
    const TileTask& task,
    std::uint64_t cycle,
    std::string* reason) {
    (void)reason;
    std::string local_reason;
    if (!dispatcher_->submit(task, cycle, &local_reason)) {
        if (local_reason == "dispatcher queue full") {
            return false;  // normal backpressure: retry later
        }
        fail_command(
            StatusCode::BadDescriptor,
            "internal task rejected: " + local_reason,
            hw::command::NONE,
            cycle);
        return false;
    }
    task_graph_.push_back(task);
    outstanding_.insert(task.task_id);
    gemm_requests_.push_back(make_gemm_req(
        task, request_.descriptor_id, request_.front_dim, request_.pivot_dim));
    return true;
}

std::uint32_t GemmController::map_failure(
    const TaskMeta& meta,
    const TileTaskResponse& response) const {
    // The single front coordinate carried by the command response is the A22
    // tile column plus the local failing column, so it always lies inside the
    // affected output tile. FP32 failures report their local column through
    // response.failed_column; INT32 failures through the phase fields. The full
    // localization (K step, tile row/column, local element, failing phase)
    // stays in the internal response and the task trace; the Completion ABI is
    // not extended.
    const std::uint32_t local =
        response.gemm_failed_phase == GemmPhase::None
            ? response.failed_column
            : response.gemm_failed_local_col;
    return request_.pivot_dim + meta.col_tile * 16 +
           std::min(local, std::uint32_t{15});
}

void GemmController::on_task_response(
    const TileTaskResponse& response,
    std::uint64_t cycle) {
    if (outstanding_.count(response.task_id) == 0) {
        return;  // late response after cancellation
    }
    outstanding_.erase(response.task_id);
    if (const TileTask* task = find_task(response.task_id); task != nullptr) {
        gemm_responses_.push_back(make_gemm_rsp(
            *task, response, request_.front_dim, request_.pivot_dim));
    }
    const TaskMeta meta = task_meta_.count(response.task_id) != 0
                              ? task_meta_[response.task_id]
                              : TaskMeta{};
    // Aggregate the real backend statistics of this tile BEFORE any terminal
    // handling: a failing tile still reports the saturation/overflow counters
    // and the partial phase cycles actually consumed.
    saturation_count_ += response.saturation_count;
    overflow_count_ += response.overflow_count;
    local_conversion_bytes_ += response.local_conversion_bytes;
    scale_reduction_cycles_ += response.scale_reduction_cycles;
    quantize_cycles_ += response.quantize_cycles;
    integer_compute_cycles_ += response.integer_compute_cycles;
    dequantize_cycles_ += response.dequantize_cycles;
    if (!have_exponents_) {
        exponent_a_ = response.gemm_exponent_a;
        exponent_b_ = response.gemm_exponent_b;
        have_exponents_ = true;
    }
    if (phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted) {
        return;
    }
    if (abort_requested_) {
        if (outstanding_.empty()) {
            phase_ = Phase::Aborted;
            publish(cycle);
        }
        return;
    }
    if (!response.result_visible) {
        fail_command(
            response.status == StatusCode::Ok ? StatusCode::Aborted : response.status,
            response.reason.empty() ? "task did not produce a visible result"
                                    : response.reason,
            map_failure(meta, response),
            cycle);
        return;
    }
    if (response.status != StatusCode::Ok) {
        // Preserve the failing tile / K step / local element for the trace.
        failure_reason_ = response.reason;
        fail_command(
            response.status,
            response.reason,
            map_failure(meta, response),
            cycle);
        return;
    }
    if (meta.row_tile < committed_.size() && meta.col_tile < committed_.size() &&
        !committed_[meta.row_tile][meta.col_tile]) {
        committed_[meta.row_tile][meta.col_tile] = true;
        ++committed_count_;
    }
}

void GemmController::on_tasks_cancelled(const std::vector<TileTask>& tasks) {
    for (const TileTask& task : tasks) {
        outstanding_.erase(task.task_id);
        cancelled_tasks_.push_back(task);
    }
}

void GemmController::request_abort() {
    if (phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted) {
        return;
    }
    abort_requested_ = true;
    cancel_requested_ = true;
}

bool GemmController::take_cancel_request() {
    if (cancel_requested_ && !cancel_taken_) {
        cancel_taken_ = true;
        return true;
    }
    return false;
}

bool GemmController::finished() const {
    return phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted;
}

void GemmController::fail_command(
    StatusCode status,
    const std::string& reason,
    std::uint32_t front_coordinate,
    std::uint64_t cycle) {
    if (phase_ == Phase::Failed || phase_ == Phase::Aborted || phase_ == Phase::Done) {
        return;
    }
    phase_ = Phase::Failed;
    failure_status_ = status;
    failure_reason_ = reason;
    failed_coordinate_ = front_coordinate;
    cancel_requested_ = true;
    publish(cycle);
}

void GemmController::publish(std::uint64_t cycle) {
    if (response_.has_value()) {
        return;
    }
    PanelCommandResponse response;
    response.command_id = request_.command_id;
    response.node_id = request_.node_id;
    response.start_cycle = start_cycle_;
    response.finish_cycle = cycle;
    response.tasks_generated = static_cast<std::uint32_t>(task_graph_.size());
    response.tasks_cancelled = static_cast<std::uint32_t>(cancelled_tasks_.size());
    response.pvector_written = false;
    response.pvector_bytes = 0;
    response.completed_pivots = 0;
    response.gemm_backend = request_.gemm_backend;
    response.saturation_count = saturation_count_;
    response.overflow_count = overflow_count_;
    response.local_conversion_bytes = local_conversion_bytes_;
    response.scale_reduction_cycles = scale_reduction_cycles_;
    response.quantize_cycles = quantize_cycles_;
    response.integer_compute_cycles = integer_compute_cycles_;
    response.dequantize_cycles = dequantize_cycles_;
    response.exponent_a = exponent_a_;
    response.exponent_b = exponent_b_;
    if (phase_ == Phase::Done) {
        response.status = StatusCode::Ok;
        // GEMM produces no pivots: pivot_count must stay zero (a failure
        // coordinate is never smuggled into it either).
        response.pivot_count = 0;
    } else if (phase_ == Phase::Failed) {
        response.status = failure_status_;
        response.reason = failure_reason_;
        response.failed_column = failed_coordinate_;
        response.pivot_count = 0;
    } else {
        response.status = StatusCode::Aborted;
        response.reason = "command aborted";
        response.failed_column = hw::command::NONE;
        response.pivot_count = 0;
    }
    response_ = response;
}

bool GemmController::poll_response(PanelCommandResponse& out) {
    if (!response_.has_value()) {
        return false;
    }
    // Failure publication may precede the drain of other accepted tasks.
    // Include every drained task's actual counters in the terminal response.
    response_->saturation_count = saturation_count_;
    response_->overflow_count = overflow_count_;
    response_->local_conversion_bytes = local_conversion_bytes_;
    response_->scale_reduction_cycles = scale_reduction_cycles_;
    response_->quantize_cycles = quantize_cycles_;
    response_->integer_compute_cycles = integer_compute_cycles_;
    response_->dequantize_cycles = dequantize_cycles_;
    response_->tasks_cancelled = static_cast<std::uint32_t>(cancelled_tasks_.size());
    out = *response_;
    response_.reset();
    return true;
}

const TileTask* GemmController::find_task(std::uint64_t task_id) const {
    for (const TileTask& task : task_graph_) {
        if (task.task_id == task_id) {
            return &task;
        }
    }
    return nullptr;
}

void GemmController::reset() {
    request_ = PanelCommandRequest{};
    kind_ = ComputeKind::GemmSchur;
    phase_ = Phase::Idle;
    epoch_ = 0;
    update_dim_ = 0;
    k_block_size_ = 16;
    pivot_tiles_ = 0;
    output_tiles_ = 0;
    num_steps_ = 0;
    step_ = 0;
    step_ready_ = false;
    generated_.clear();
    committed_.clear();
    expected_per_step_ = 0;
    generated_count_ = 0;
    committed_count_ = 0;
    start_cycle_ = 0;
    next_task_id_ = 1;
    sequence_ = 0;
    task_graph_.clear();
    cancelled_tasks_.clear();
    outstanding_.clear();
    task_meta_.clear();
    gemm_requests_.clear();
    gemm_responses_.clear();
    abort_requested_ = false;
    cancel_requested_ = false;
    cancel_taken_ = false;
    failure_status_ = StatusCode::Ok;
    failure_reason_.clear();
    failed_coordinate_ = hw::command::NONE;
    response_.reset();
    saturation_count_ = 0;
    overflow_count_ = 0;
    local_conversion_bytes_ = 0;
    scale_reduction_cycles_ = 0;
    quantize_cycles_ = 0;
    integer_compute_cycles_ = 0;
    dequantize_cycles_ = 0;
    exponent_a_ = 0;
    exponent_b_ = 0;
    have_exponents_ = false;
}

}  // namespace hw::panel_lu
