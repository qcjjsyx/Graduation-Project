#include "panel_lu/trsm_controller.hpp"

#include <algorithm>

namespace hw::panel_lu {

namespace {

std::uint32_t ceil_div16(std::uint32_t value) {
    return (value + 15) / 16;
}

}  // namespace

TrsmController::TrsmController(
    const PanelLuConfig& config,
    TileDispatcher* dispatcher,
    TileVersionStore* versions)
    : config_(config), dispatcher_(dispatcher), versions_(versions) {}

std::uint32_t TrsmController::axis_extent(
    std::uint32_t tiles,
    std::uint32_t tile) const {
    return std::min<std::uint32_t>(16, tiles - tile * 16);
}

std::uint64_t TrsmController::front_grid_key(
    std::uint32_t row_tile,
    std::uint32_t col_tile) const {
    return tile_version_key(
        layout_.handle.slot,
        TileViewTag::FrontGrid,
        row_tile * layout_.global_grid + col_tile);
}

std::uint64_t TrsmController::a12_key(
    std::uint32_t row_tile,
    std::uint32_t col_tile) const {
    return tile_version_key(
        layout_.handle.slot,
        TileViewTag::A12,
        row_tile * update_tiles_ + col_tile);
}

std::uint64_t TrsmController::a21_key(
    std::uint32_t row_tile,
    std::uint32_t col_tile) const {
    return tile_version_key(
        layout_.handle.slot,
        TileViewTag::A21,
        row_tile * pivot_tiles_ + col_tile);
}

std::uint32_t TrsmController::version_of(std::uint64_t key) const {
    return versions_->version(key);
}

TileOperand TrsmController::front_grid_operand(
    std::uint32_t row_tile,
    std::uint32_t col_tile,
    std::uint32_t version) const {
    TileOperand operand;
    operand.handle = layout_.handle;
    // Offsets are relative to the front base; LeaseTileAccess applies
    // binding.base_offset exactly once.
    operand.offset =
        static_cast<std::uint64_t>(row_tile) * 16 * layout_.row_stride_bytes +
        static_cast<std::uint64_t>(col_tile) * 16 * 4;
    operand.stride_bytes = layout_.row_stride_bytes;
    operand.version = version;
    operand.version_key = front_grid_key(row_tile, col_tile);
    operand.valid = true;
    return operand;
}

TileOperand TrsmController::a12_operand(
    std::uint32_t row_tile,
    std::uint32_t col_tile,
    std::uint32_t version) const {
    // A12 starts at column pivot_dim (not at a 16-aligned full-front grid
    // boundary): row pivot_tile*16, column pivot_dim + col_tile*16.
    TileOperand operand;
    operand.handle = layout_.handle;
    operand.offset =
        static_cast<std::uint64_t>(row_tile) * 16 * layout_.row_stride_bytes +
        static_cast<std::uint64_t>(request_.pivot_dim + col_tile * 16) * 4;
    operand.stride_bytes = layout_.row_stride_bytes;
    operand.version = version;
    operand.version_key = a12_key(row_tile, col_tile);
    operand.valid = true;
    return operand;
}

TileOperand TrsmController::a21_operand(
    std::uint32_t row_tile,
    std::uint32_t col_tile,
    std::uint32_t version) const {
    // A21 starts at row pivot_dim.
    TileOperand operand;
    operand.handle = layout_.handle;
    operand.offset =
        static_cast<std::uint64_t>(request_.pivot_dim + row_tile * 16) *
            layout_.row_stride_bytes +
        static_cast<std::uint64_t>(col_tile) * 16 * 4;
    operand.stride_bytes = layout_.row_stride_bytes;
    operand.version = version;
    operand.version_key = a21_key(row_tile, col_tile);
    operand.valid = true;
    return operand;
}

ConfigCheck TrsmController::start(
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
        check.reason = "TRSM requires 0 < pivot_dim < front_dim";
        return check;
    }
    if (request.tile_size != config_.tile_dim) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "TRSM requires the fixed 16x16 compute tile";
        return check;
    }
    if (request.compute_kind != ComputeKind::TrsmLeft &&
        request.compute_kind != ComputeKind::TrsmRight) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "TRSM controller received a non-TRSM compute kind";
        return check;
    }
    request_ = request;
    kind_ = request.compute_kind;
    epoch_ = epoch;
    layout_ = layout;
    update_dim_ = total - pivot;
    pivot_tiles_ = ceil_div16(pivot);
    update_tiles_ = ceil_div16(update_dim_);
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
    abort_requested_ = false;
    cancel_requested_ = false;
    cancel_taken_ = false;
    failure_status_ = StatusCode::Ok;
    failure_reason_.clear();
    failed_coordinate_ = hw::command::NONE;
    response_.reset();
    solve_committed_count_ = 0;
    update_generated_count_ = 0;
    update_committed_count_ = 0;
    solve_targets_ = 0;
    update_expected_count_ = 0;
    step_ready_ = false;
    check.ok = true;
    return check;
}

void TrsmController::tick(
    std::uint64_t cycle,
    hw::dataplane::OnChipArbiter& arbiter) {
    (void)arbiter;  // This controller issues no pixel/scratch bytes of its own.
    if (abort_requested_) {
        // Abort is terminal for generation: after the engine performed the
        // directed cancellation this controller must never submit another
        // task or touch the front lease; only the drain is awaited.
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
            if (solve_committed_count_ == solve_targets_ &&
                update_generated_count_ == update_expected_count_ &&
                update_committed_count_ == update_expected_count_) {
                ++step_;
                if (step_ >= num_steps_) {
                    phase_ = Phase::Done;
                    publish(cycle);
                } else {
                    reset_step_bookkeeping();
                    phase_ = Phase::StepGenerate;
                }
            }
            break;
        default:
            break;
    }
}

void TrsmController::reset_step_bookkeeping() {
    const std::uint32_t target_count = update_tiles_;
    solve_targets_ = target_count;
    solve_ids_.assign(target_count, 0);
    solve_generated_.assign(target_count, false);
    solve_committed_.assign(target_count, false);
    update_ids_.assign(target_count, {});
    update_generated_.assign(target_count, {});
    update_expected_count_ = 0;
    const std::uint32_t remaining_blocks = pivot_tiles_ - step_ - 1;
    for (std::uint32_t target = 0; target < target_count; ++target) {
        update_expected_count_ += remaining_blocks;
        update_ids_[target].assign(remaining_blocks, 0);
        update_generated_[target].assign(remaining_blocks, false);
    }
    solve_committed_count_ = 0;
    update_generated_count_ = 0;
    update_committed_count_ = 0;
    step_ready_ = true;
}

void TrsmController::generate_step(std::uint64_t cycle) {
    // Per step: one solve per update-space tile (columns for LEFT, rows for
    // RIGHT), then one update per remaining pivot block (q > s) per target.
    const std::uint32_t target_count = update_tiles_;
    if (!step_ready_) {
        reset_step_bookkeeping();
    }

    // 1) Emit all solve tasks for this step (Lss^-1 Bsj / Bis Uss^-1).
    for (std::uint32_t target = 0; target < target_count; ++target) {
        if (solve_generated_[target]) {
            continue;
        }
        const std::uint32_t diag = axis_extent(request_.pivot_dim, step_);
        const std::uint32_t other = axis_extent(update_dim_, target);
        TileTask task;
        task.task_id = next_task_id_++;
        task.epoch = epoch_;
        task.command_id = request_.command_id;
        task.node_id = request_.node_id;
        task.step = step_;
        task.group_id = layout_.group_id;
        task.sequence = sequence_++;
        task.threshold_product = request_.threshold_product;
        if (kind_ == ComputeKind::TrsmLeft) {
            task.mode = TileMode::TrsmLeft;
            task.tile_row = step_;
            task.tile_col = target;
            task.dims = {diag, other, diag};
            task.lhs = front_grid_operand(
                step_, step_, version_of(front_grid_key(step_, step_)));
            task.rhs = a12_operand(step_, target, version_of(a12_key(step_, target)));
            task.dst = task.rhs;
        } else {
            task.mode = TileMode::TrsmRight;
            task.tile_row = target;
            task.tile_col = step_;
            task.dims = {other, diag, diag};
            task.lhs = a21_operand(target, step_, version_of(a21_key(target, step_)));
            task.rhs = front_grid_operand(
                step_, step_, version_of(front_grid_key(step_, step_)));
            task.dst = task.lhs;
        }
        task.output_version = task.dst.version + 1;
        task.mask = {task.dims.m, task.dims.n};
        if (!submit_task(task, cycle, nullptr)) {
            return;  // queue full: keep progress, retry next cycle
        }
        solve_generated_[target] = true;
        solve_ids_[target] = task.task_id;
        task_meta_[task.task_id] =
            TaskMeta{task.mode, step_, task.tile_row, task.tile_col};
    }

    // 2) Emit updates for every committed solve (incremental per solve).
    const std::uint32_t remaining_blocks = pivot_tiles_ - step_ - 1;
    for (std::uint32_t target = 0; target < target_count; ++target) {
        if (!solve_committed_[target] || solve_ids_[target] == 0) {
            continue;
        }
        const std::uint32_t diag = axis_extent(request_.pivot_dim, step_);
        for (std::uint32_t q = 0; q < remaining_blocks; ++q) {
            if (update_generated_[target][q]) {
                continue;
            }
            const std::uint32_t block = step_ + 1 + q;
            TileTask task;
            task.task_id = next_task_id_++;
            task.epoch = epoch_;
            task.command_id = request_.command_id;
            task.node_id = request_.node_id;
            task.step = step_;
            task.group_id = layout_.group_id;
            task.sequence = sequence_++;
            task.deps = {solve_ids_[target]};
            if (kind_ == ComputeKind::TrsmLeft) {
                // B(q,j) -= L(q,s) * X(s,j)
                task.mode = TileMode::GemmUpdate;
                task.tile_row = block;
                task.tile_col = target;
                const std::uint32_t m = axis_extent(request_.pivot_dim, block);
                const std::uint32_t n = axis_extent(update_dim_, target);
                task.dims = {m, n, diag};
                task.lhs = front_grid_operand(
                    block, step_, version_of(front_grid_key(block, step_)));
                task.rhs = a12_operand(
                    step_, target, version_of(a12_key(step_, target)));
                task.dst = a12_operand(
                    block, target, version_of(a12_key(block, target)));
            } else {
                // B(i,q) -= X(i,s) * U(s,q)
                task.mode = TileMode::GemmUpdate;
                task.tile_row = target;
                task.tile_col = block;
                const std::uint32_t m = axis_extent(update_dim_, target);
                const std::uint32_t n = axis_extent(request_.pivot_dim, block);
                task.dims = {m, n, diag};
                task.lhs = a21_operand(
                    target, step_, version_of(a21_key(target, step_)));
                task.rhs = front_grid_operand(
                    step_, block, version_of(front_grid_key(step_, block)));
                task.dst = a21_operand(
                    target, block, version_of(a21_key(target, block)));
            }
            task.output_version = task.dst.version + 1;
            task.mask = {task.dims.m, task.dims.n};
            if (!submit_task(task, cycle, nullptr)) {
                return;
            }
            update_generated_[target][q] = true;
            update_ids_[target][q] = task.task_id;
            ++update_generated_count_;
            task_meta_[task.task_id] =
                TaskMeta{task.mode, step_, task.tile_row, task.tile_col};
        }
    }
    // Generation for this step is complete only when every solve and every
    // expected update has been queued; otherwise stay in the generate phase
    // and retry as solves commit.
    bool all_solves = true;
    for (const bool generated : solve_generated_) {
        all_solves = all_solves && generated;
    }
    if (all_solves && update_generated_count_ == update_expected_count_) {
        phase_ = Phase::StepWait;
    }
}

bool TrsmController::submit_task(
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
    return true;
}

std::uint32_t TrsmController::map_failure(
    const TaskMeta& meta,
    std::uint32_t local_column) const {
    switch (meta.mode) {
        case TileMode::TrsmLeft:
            // Unit-lower solve failure is reported per RHS row (pivot row).
            return meta.step * 16 + local_column;
        case TileMode::TrsmRight:
            // Non-unit U diagonal failure: pivot column of the diag block.
            return meta.step * 16 + local_column;
        case TileMode::GemmUpdate:
            if (kind_ == ComputeKind::TrsmLeft) {
                return request_.pivot_dim + meta.col_tile * 16 + local_column;
            }
            return meta.col_tile * 16 + local_column;
        case TileMode::GemmSchur:
            // Never generated by TrsmController; mapped like the panel update
            // for exhaustiveness.
            return request_.pivot_dim + meta.col_tile * 16 + local_column;
        case TileMode::LuDiag:
            return local_column;
    }
    return local_column;
}

void TrsmController::on_task_response(
    const TileTaskResponse& response,
    std::uint64_t cycle) {
    if (outstanding_.count(response.task_id) == 0) {
        return;  // late response after cancellation
    }
    outstanding_.erase(response.task_id);
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
    const TaskMeta meta = task_meta_.count(response.task_id) != 0
                              ? task_meta_[response.task_id]
                              : TaskMeta{};
    if (!response.result_visible) {
        fail_command(
            response.status == StatusCode::Ok ? StatusCode::Aborted : response.status,
            response.reason.empty() ? "task did not produce a visible result"
                                    : response.reason,
            map_failure(meta, response.failed_column),
            cycle);
        return;
    }
    if (response.status != StatusCode::Ok) {
        fail_command(
            response.status,
            response.reason,
            map_failure(meta, response.failed_column),
            cycle);
        return;
    }
    if (meta.mode == TileMode::TrsmLeft || meta.mode == TileMode::TrsmRight) {
        // Solve target index equals the tile coordinate that is not the step.
        const std::uint32_t target =
            kind_ == ComputeKind::TrsmLeft ? meta.col_tile : meta.row_tile;
        if (target >= solve_committed_.size() || solve_committed_[target]) {
            return;
        }
        solve_committed_[target] = true;
        ++solve_committed_count_;
    } else if (meta.mode == TileMode::GemmUpdate) {
        ++update_committed_count_;
    }
}

void TrsmController::on_tasks_cancelled(const std::vector<TileTask>& tasks) {
    for (const TileTask& task : tasks) {
        outstanding_.erase(task.task_id);
        cancelled_tasks_.push_back(task);
    }
}

void TrsmController::request_abort() {
    if (phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted) {
        return;
    }
    abort_requested_ = true;
    cancel_requested_ = true;
}

bool TrsmController::take_cancel_request() {
    if (cancel_requested_ && !cancel_taken_) {
        cancel_taken_ = true;
        return true;
    }
    return false;
}

bool TrsmController::finished() const {
    return phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted;
}

void TrsmController::fail_command(
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

void TrsmController::publish(std::uint64_t cycle) {
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
    if (phase_ == Phase::Done) {
        response.status = StatusCode::Ok;
        response.pivot_count = 0;
        response.completed_pivots = 0;
    } else if (phase_ == Phase::Failed) {
        response.status = failure_status_;
        response.reason = failure_reason_;
        response.failed_column = failed_coordinate_;
        response.pivot_count =
            failed_coordinate_ == hw::command::NONE ? 0 : failed_coordinate_;
    } else {
        response.status = StatusCode::Aborted;
        response.reason = "command aborted";
        response.failed_column = hw::command::NONE;
        response.pivot_count = 0;
    }
    response_ = response;
}

bool TrsmController::poll_response(PanelCommandResponse& out) {
    if (!response_.has_value()) {
        return false;
    }
    out = *response_;
    response_.reset();
    return true;
}

const TileTask* TrsmController::find_task(std::uint64_t task_id) const {
    for (const TileTask& task : task_graph_) {
        if (task.task_id == task_id) {
            return &task;
        }
    }
    return nullptr;
}

void TrsmController::reset() {
    request_ = PanelCommandRequest{};
    kind_ = ComputeKind::TrsmLeft;
    phase_ = Phase::Idle;
    epoch_ = 0;
    update_dim_ = 0;
    pivot_tiles_ = 0;
    update_tiles_ = 0;
    num_steps_ = 0;
    step_ = 0;
    solve_ids_.clear();
    solve_generated_.clear();
    solve_committed_.clear();
    update_ids_.clear();
    update_generated_.clear();
    solve_committed_count_ = 0;
    update_generated_count_ = 0;
    update_committed_count_ = 0;
    update_expected_count_ = 0;
    solve_targets_ = 0;
    step_ready_ = false;
    start_cycle_ = 0;
    next_task_id_ = 1;
    sequence_ = 0;
    task_graph_.clear();
    cancelled_tasks_.clear();
    outstanding_.clear();
    task_meta_.clear();
    abort_requested_ = false;
    cancel_requested_ = false;
    cancel_taken_ = false;
    failure_status_ = StatusCode::Ok;
    failure_reason_.clear();
    failed_coordinate_ = hw::command::NONE;
    response_.reset();
}

}  // namespace hw::panel_lu
