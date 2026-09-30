#include "panel_lu/panel_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace hw::panel_lu {

namespace {

std::uint32_t ceil_div16(std::uint32_t value) {
    return (value + 15) / 16;
}

}  // namespace

PanelController::PanelController(
    const PanelLuConfig& config,
    TileDispatcher* dispatcher,
    TileVersionStore* versions,
    FrontReader* reader)
    : config_(config),
      dispatcher_(dispatcher),
      versions_(versions),
      reader_(reader) {}

StatusCode PanelController::decode_threshold_bits(std::uint32_t bits, float* out) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    if (!(value >= 0.0f && value < 1.0f)) {
        return StatusCode::BadDescriptor;
    }
    if (out != nullptr) {
        *out = value;
    }
    return StatusCode::Ok;
}

ConfigCheck PanelController::start(
    const PanelCommandRequest& request,
    std::uint64_t cycle,
    std::uint64_t epoch,
    const PanelFrontLayout& layout,
    std::uint64_t first_task_id) {
    const ConfigCheck capability = check_front_capability(
        request.front_dim,
        request.pivot_dim,
        request.tile_size,
        request.policy,
        config_);
    if (!capability.ok) {
        return capability;
    }
    // Validate the threshold of THIS request before establishing any command
    // state; a rejected threshold must never fall back to a default value.
    float threshold_ratio = 0.0f;
    const StatusCode threshold_status =
        decode_threshold_bits(request.threshold_bits, &threshold_ratio);
    if (threshold_status != StatusCode::Ok) {
        ConfigCheck check;
        check.ok = false;
        check.status = threshold_status;
        check.reason =
            "threshold bits 0x" + std::to_string(request.threshold_bits) +
            " do not decode to a binary32 value in [0,1)";
        return check;
    }
    (void)threshold_ratio;
    if (request.command_id == hw::command::NONE) {
        ConfigCheck check;
        check.ok = false;
        check.status = StatusCode::BadCommand;
        check.reason = "command_id must be assigned";
        return check;
    }

    request_ = request;
    epoch_ = epoch;
    layout_ = layout;
    global_grid_ = layout.global_grid != 0 ? layout.global_grid : 1;
    local_grid_ = ceil_div16(request.pivot_dim);
    num_steps_ = local_grid_;
    step_ = 0;
    phase_ = Phase::Scan;
    start_cycle_ = cycle;
    scan_bytes_total_ =
        static_cast<std::uint64_t>(request.front_dim) * request.front_dim * 4;
    scan_bytes_remaining_ = scan_bytes_total_;
    scan_elements_done_ = 0;
    scan_done_cycle_ = 0;
    scale_ready_at_ = 0;
    scan_non_finite_ = false;
    front_max_abs_ = 0.0;
    threshold_product_ = 0.0;
    pvector_bytes_remaining_ = 0;
    pvector_done_at_ = 0;
    pvector_write_started_ = false;
    pvector_written_ = false;
    next_task_id_ = first_task_id;
    sequence_ = 0;
    task_graph_.clear();
    cancelled_tasks_.clear();
    outstanding_.clear();
    step_wait_.clear();
    abort_requested_ = false;
    cancel_requested_ = false;
    cancel_taken_ = false;
    failure_status_ = StatusCode::Ok;
    failure_reason_.clear();
    failed_column_ = hw::command::NONE;
    completed_pivots_ = 0;
    stall_dispatcher_full_ = 0;
    stall_scan_port_ = 0;
    response_.reset();

    diag_task_ids_.assign(local_grid_, 0);
    left_task_ids_.assign(local_grid_, std::vector<std::uint64_t>(local_grid_, 0));
    right_task_ids_.assign(local_grid_, std::vector<std::uint64_t>(local_grid_, 0));
    gemm_done_.clear();

    ConfigCheck check;
    check.ok = true;
    return check;
}

std::uint32_t PanelController::grid_extent(std::uint32_t dimension) const {
    return ceil_div16(dimension);
}

std::uint32_t PanelController::tile_width(std::uint32_t tile) const {
    const std::uint32_t base = tile * 16;
    if (base >= request_.pivot_dim) {
        return 0;
    }
    return std::min<std::uint32_t>(16, request_.pivot_dim - base);
}

std::uint64_t PanelController::version_key(TileCoord coord) const {
    if (layout_.kind == PanelFrontLayout::Kind::PerTileSlots) {
        return layout_.handle.slot + coord.row * global_grid_ + coord.col;
    }
    const std::uint64_t tile_index =
        static_cast<std::uint64_t>(coord.row) * global_grid_ + coord.col;
    return (static_cast<std::uint64_t>(layout_.handle.slot) << 32) | (tile_index + 1);
}

std::uint32_t PanelController::operand_version(TileCoord coord) const {
    return versions_->version(version_key(coord));
}

TileOperand PanelController::operand_for(TileCoord coord, std::uint32_t version) const {
    TileOperand result;
    result.version = version;
    result.version_key = version_key(coord);
    result.valid = true;
    if (layout_.kind == PanelFrontLayout::Kind::PerTileSlots) {
        result.handle.slot = layout_.handle.slot + coord.row * global_grid_ + coord.col;
        result.handle.generation = layout_.generation;
        result.offset = 0;
        result.stride_bytes = 64;
    } else {
        result.handle = layout_.handle;
        result.handle.generation = layout_.generation;
        result.offset = layout_.base_offset +
                        static_cast<std::uint64_t>(coord.row) * 16 * layout_.row_stride_bytes +
                        static_cast<std::uint64_t>(coord.col) * 16 * 4;
        result.stride_bytes = layout_.row_stride_bytes;
    }
    return result;
}

TileTask PanelController::make_task(
    TileMode mode,
    TileCoord coord,
    TileDims dims,
    std::uint32_t step,
    const std::vector<std::uint64_t>& deps,
    std::uint64_t cycle) {
    (void)cycle;
    TileTask task;
    task.task_id = next_task_id_++;
    task.epoch = epoch_;
    task.command_id = request_.command_id;
    task.node_id = request_.node_id;
    task.step = step;
    task.tile_row = coord.row;
    task.tile_col = coord.col;
    task.mode = mode;
    task.dims = dims;
    task.mask.rows = dims.m;
    task.mask.cols = dims.n;
    task.deps = deps;
    task.sequence = sequence_++;
    task.front_pre_elimination_max_abs = front_max_abs_;
    task.threshold_product = threshold_product_;

    const TileCoord diag_coord{step, step};
    const std::uint32_t target_version = operand_version(coord);
    const std::uint32_t diag_version = operand_version(diag_coord);

    switch (mode) {
        case TileMode::LuDiag:
            task.dst = operand_for(diag_coord, diag_version);
            task.output_version = diag_version + 1;
            task.pivot_base = step * 16;
            break;
        case TileMode::TrsmLeft:
            // lhs = packed L diag tile, rhs/dst = A(s,j) which becomes U(s,j).
            task.lhs = operand_for(diag_coord, diag_version);
            task.rhs = operand_for(coord, target_version);
            task.dst = operand_for(coord, target_version);
            task.output_version = target_version + 1;
            break;
        case TileMode::TrsmRight:
            // lhs/dst = A(i,s) which becomes L(i,s), rhs = packed U diag tile.
            task.lhs = operand_for(coord, target_version);
            task.rhs = operand_for(diag_coord, diag_version);
            task.dst = operand_for(coord, target_version);
            task.output_version = target_version + 1;
            break;
        case TileMode::GemmUpdate: {
            const TileCoord lhs_coord{coord.row, step};
            const TileCoord rhs_coord{step, coord.col};
            task.lhs = operand_for(lhs_coord, operand_version(lhs_coord));
            task.rhs = operand_for(rhs_coord, operand_version(rhs_coord));
            task.dst = operand_for(coord, target_version);
            task.output_version = target_version + 1;
            break;
        }
    }
    return task;
}

bool PanelController::submit_task(
    const TileTask& task,
    std::uint64_t cycle,
    std::string* reason) {
    std::string local_reason;
    if (!dispatcher_->submit(task, cycle, &local_reason)) {
        if (local_reason == "dispatcher queue full") {
            ++stall_dispatcher_full_;
        } else {
            fail_command(
                StatusCode::BadDescriptor,
                "internal task rejected: " + local_reason,
                hw::command::NONE,
                cycle);
        }
        if (reason != nullptr) {
            *reason = local_reason;
        }
        return false;
    }
    task_graph_.push_back(task);
    outstanding_.insert(task.task_id);
    return true;
}

void PanelController::produce_diag(std::uint64_t cycle) {
    diag_task_ids_[step_] = 0;
    step_wait_.clear();
    gemm_done_.clear();
    const std::uint32_t size = tile_width(step_);
    TileDims dims{size, size, size};
    const TileTask task =
        make_task(TileMode::LuDiag, TileCoord{step_, step_}, dims, step_, {}, cycle);
    if (!submit_task(task, cycle, nullptr)) {
        return;  // retry next cycle (queue full) or command already failed
    }
    diag_task_ids_[step_] = task.task_id;
    step_wait_.insert(task.task_id);
    phase_ = Phase::DiagWait;
}

void PanelController::produce_trsm(std::uint64_t cycle) {
    const std::uint64_t diag_id = diag_task_ids_[step_];
    const std::uint32_t k = tile_width(step_);
    // Row panel: U(s,j) = L(s,s)^-1 * A(s,j) for j > s.
    for (std::uint32_t j = step_ + 1; j < local_grid_; ++j) {
        if (left_task_ids_[step_][j] != 0) {
            continue;
        }
        const std::uint32_t n = tile_width(j);
        TileDims dims{k, n, k};
        const TileTask task = make_task(
            TileMode::TrsmLeft, TileCoord{step_, j}, dims, step_, {diag_id}, cycle);
        if (!submit_task(task, cycle, nullptr)) {
            return;
        }
        left_task_ids_[step_][j] = task.task_id;
        step_wait_.insert(task.task_id);
    }
    // Column panel: L(i,s) = A(i,s) * U(s,s)^-1 for i > s.
    for (std::uint32_t i = step_ + 1; i < local_grid_; ++i) {
        if (right_task_ids_[i][step_] != 0) {
            continue;
        }
        const std::uint32_t m = tile_width(i);
        TileDims dims{m, k, k};
        const TileTask task = make_task(
            TileMode::TrsmRight, TileCoord{i, step_}, dims, step_, {diag_id}, cycle);
        if (!submit_task(task, cycle, nullptr)) {
            return;
        }
        right_task_ids_[i][step_] = task.task_id;
        step_wait_.insert(task.task_id);
    }
    phase_ = Phase::TrsmWait;
}

void PanelController::produce_gemm(std::uint64_t cycle) {
    const std::uint32_t k = tile_width(step_);
    for (std::uint32_t i = step_ + 1; i < local_grid_; ++i) {
        for (std::uint32_t j = step_ + 1; j < local_grid_; ++j) {
            if (gemm_done_.count({i, j}) != 0) {
                continue;
            }
            const std::uint64_t lhs_id = right_task_ids_[i][step_];
            const std::uint64_t rhs_id = left_task_ids_[step_][j];
            if (lhs_id == 0 || rhs_id == 0) {
                continue;
            }
            const std::uint32_t m = tile_width(i);
            const std::uint32_t n = tile_width(j);
            TileDims dims{m, n, k};
            const TileTask task = make_task(
                TileMode::GemmUpdate,
                TileCoord{i, j},
                dims,
                step_,
                {lhs_id, rhs_id},
                cycle);
            if (!submit_task(task, cycle, nullptr)) {
                return;
            }
            gemm_done_.insert({i, j});
            step_wait_.insert(task.task_id);
        }
    }
    phase_ = Phase::GemmWait;
}

void PanelController::finish_step(std::uint64_t cycle) {
    (void)cycle;
    ++step_;
    if (step_ < num_steps_) {
        phase_ = Phase::Diag;
        return;
    }
    pvector_bytes_remaining_ = static_cast<std::uint64_t>(request_.pivot_dim) * 4;
    pvector_write_started_ = false;
    phase_ = Phase::PVector;
}

void PanelController::fail_command(
    StatusCode status,
    const std::string& reason,
    std::uint32_t failed_column,
    std::uint64_t cycle) {
    if (phase_ == Phase::Failed || phase_ == Phase::Aborted || phase_ == Phase::Done) {
        return;
    }
    phase_ = Phase::Failed;
    failure_status_ = status;
    failure_reason_ = reason;
    failed_column_ = failed_column;
    cancel_requested_ = true;
    publish(cycle);
}

void PanelController::finalize_aborted(std::uint64_t cycle) {
    phase_ = Phase::Aborted;
    publish(cycle);
}

void PanelController::finalize_success(std::uint64_t cycle) {
    pvector_written_ = true;
    phase_ = Phase::Done;
    publish(cycle);
}

void PanelController::publish(std::uint64_t cycle) {
    if (response_.has_value()) {
        return;
    }
    PanelCommandResponse response;
    response.command_id = request_.command_id;
    response.node_id = request_.node_id;
    response.start_cycle = start_cycle_;
    response.finish_cycle = cycle;
    response.scan_bytes = scan_bytes_total_;
    response.scan_done_cycle = scan_done_cycle_;
    response.front_pre_elimination_max_abs = front_max_abs_;
    response.threshold_product = threshold_product_;
    response.tasks_generated = static_cast<std::uint32_t>(task_graph_.size());
    response.tasks_cancelled = static_cast<std::uint32_t>(cancelled_tasks_.size());
    response.stall_dispatcher_full = stall_dispatcher_full_;
    response.stall_scan_port = stall_scan_port_;
    response.pvector_written = pvector_written_;
    response.pvector_bytes = pvector_written_ ? request_.pivot_dim * 4 : 0;
    switch (phase_) {
        case Phase::Done:
            response.status = StatusCode::Ok;
            response.pivot_count = request_.pivot_dim;
            response.completed_pivots = request_.pivot_dim;
            break;
        case Phase::Failed:
            response.status = failure_status_;
            response.reason = failure_reason_;
            response.failed_column = failed_column_;
            response.pivot_count = failed_column_ == hw::command::NONE ? 0 : failed_column_;
            response.completed_pivots = completed_pivots_;
            break;
        case Phase::Aborted:
            response.status = StatusCode::Aborted;
            response.reason = "command aborted";
            response.failed_column = hw::command::NONE;
            response.pivot_count = completed_pivots_;
            response.completed_pivots = completed_pivots_;
            break;
        default:
            response.status = StatusCode::BadCommand;
            response.reason = "published before terminal phase";
            break;
    }
    response_ = response;
}

void PanelController::tick(
    std::uint64_t cycle,
    PortBudget& scan_budget,
    PortBudget& write_budget) {
    if (abort_requested_) {
        switch (phase_) {
            case Phase::Scan:
            case Phase::ScanReduce:
            case Phase::Diag:
            case Phase::Trsm:
            case Phase::Gemm:
                if (outstanding_.empty()) {
                    finalize_aborted(cycle);
                }
                return;
            default:
                break;
        }
    }
    switch (phase_) {
        case Phase::Scan: {
            if (scan_bytes_remaining_ > 0) {
                std::uint64_t chunk = std::min(scan_bytes_remaining_, scan_budget.remaining());
                chunk -= chunk % 4;
                if (chunk == 0) {
                    ++stall_scan_port_;
                    break;
                }
                const std::uint64_t byte_offset = scan_bytes_total_ - scan_bytes_remaining_;
                std::vector<std::uint8_t> bytes(static_cast<std::size_t>(chunk));
                if (!reader_->read_front(byte_offset, static_cast<std::uint32_t>(chunk),
                                         bytes.data(), cycle)) {
                    // Front bytes are not readable yet (lease latency/backpressure);
                    // retry next cycle without consuming read credit.
                    ++stall_scan_port_;
                    break;
                }
                if (!scan_budget.request(chunk)) {
                    ++stall_scan_port_;
                    break;
                }
                const std::uint32_t elements = static_cast<std::uint32_t>(chunk / 4);
                scan_bytes_remaining_ -= chunk;
                for (std::uint32_t e = 0; e < elements; ++e) {
                    float value = 0.0f;
                    std::memcpy(&value, bytes.data() + static_cast<std::size_t>(e) * 4, 4);
                    if (!std::isfinite(value)) {
                        scan_non_finite_ = true;
                    } else {
                        const double magnitude = std::fabs(static_cast<double>(value));
                        if (magnitude > front_max_abs_) {
                            front_max_abs_ = magnitude;
                        }
                    }
                }
                scan_elements_done_ += elements;
            }
            if (scan_bytes_remaining_ == 0) {
                scan_done_cycle_ = cycle;
                scale_ready_at_ = cycle + config_.scan_reduce_latency_cycles;
                phase_ = Phase::ScanReduce;
            }
            break;
        }
        case Phase::ScanReduce:
            if (cycle >= scale_ready_at_) {
                float threshold_ratio = 0.0f;
                const StatusCode threshold_status =
                    decode_threshold_bits(request_.threshold_bits, &threshold_ratio);
                if (threshold_status != StatusCode::Ok) {
                    // start() already rejected invalid bits; this branch exists
                    // so a corrupted state can never silently use a default.
                    fail_command(
                        threshold_status,
                        "threshold bits became invalid after admission",
                        hw::command::NONE,
                        cycle);
                    break;
                }
                if (scan_non_finite_) {
                    fail_command(
                        StatusCode::NonFiniteValue,
                        "non-finite value found in the assembled front scan",
                        hw::command::NONE,
                        cycle);
                    break;
                }
                threshold_product_ =
                    static_cast<double>(threshold_ratio) * front_max_abs_;
                step_ = 0;
                phase_ = Phase::Diag;
            }
            break;
        case Phase::Diag:
            if (abort_requested_) {
                finalize_aborted(cycle);
                break;
            }
            produce_diag(cycle);
            break;
        case Phase::DiagWait:
            if (abort_requested_ && outstanding_.empty()) {
                finalize_aborted(cycle);
            }
            break;
        case Phase::Trsm:
            produce_trsm(cycle);
            break;
        case Phase::TrsmWait:
            if (step_wait_.empty()) {
                phase_ = Phase::Gemm;
            } else if (abort_requested_ && outstanding_.empty()) {
                finalize_aborted(cycle);
            }
            break;
        case Phase::Gemm:
            produce_gemm(cycle);
            break;
        case Phase::GemmWait:
            if (step_wait_.empty()) {
                finish_step(cycle);
            } else if (abort_requested_ && outstanding_.empty()) {
                finalize_aborted(cycle);
            }
            break;
        case Phase::PVector: {
            if (pvector_bytes_remaining_ > 0) {
                std::uint64_t chunk =
                    std::min(pvector_bytes_remaining_, write_budget.remaining());
                chunk -= chunk % 4;
                if (chunk == 0 || !write_budget.request(chunk)) {
                    break;
                }
                pvector_bytes_remaining_ -= chunk;
                pvector_write_started_ = true;
                if (pvector_bytes_remaining_ == 0) {
                    pvector_done_at_ = cycle + config_.writeback_latency_cycles;
                }
            }
            if (pvector_bytes_remaining_ == 0 && pvector_done_at_ != 0 &&
                cycle >= pvector_done_at_) {
                finalize_success(cycle);
            }
            break;
        }
        case Phase::Done:
        case Phase::Failed:
        case Phase::Aborted:
        case Phase::Idle:
            break;
    }
}

void PanelController::on_task_response(
    const TileTaskResponse& response,
    std::uint64_t cycle) {
    if (outstanding_.count(response.task_id) == 0) {
        return;  // late response after cancellation
    }
    outstanding_.erase(response.task_id);
    step_wait_.erase(response.task_id);

    if (phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted) {
        return;
    }
    if (abort_requested_) {
        if (outstanding_.empty()) {
            finalize_aborted(cycle);
        }
        return;
    }
    if (!response.result_visible) {
        fail_command(
            response.status == StatusCode::Ok ? StatusCode::Aborted : response.status,
            response.reason.empty() ? "task did not produce a visible result"
                                    : response.reason,
            response.failed_column,
            cycle);
        return;
    }
    if (response.status != StatusCode::Ok) {
        fail_command(response.status, response.reason, response.failed_column, cycle);
        return;
    }
    switch (phase_) {
        case Phase::DiagWait:
            completed_pivots_ =
                std::min(request_.pivot_dim, (step_ + 1) * config_.tile_dim);
            phase_ = Phase::Trsm;
            break;
        case Phase::TrsmWait:
            if (step_wait_.empty()) {
                phase_ = Phase::Gemm;
            }
            break;
        case Phase::GemmWait:
            if (step_wait_.empty()) {
                finish_step(cycle);
            }
            break;
        default:
            break;
    }
}

void PanelController::on_tasks_cancelled(const std::vector<TileTask>& tasks) {
    for (const TileTask& task : tasks) {
        outstanding_.erase(task.task_id);
        step_wait_.erase(task.task_id);
        cancelled_tasks_.push_back(task);
        if (task.step < diag_task_ids_.size() &&
            diag_task_ids_[task.step] == task.task_id) {
            diag_task_ids_[task.step] = 0;
        }
        if (task.step < local_grid_ && task.tile_row < local_grid_ &&
            task.tile_col < local_grid_ && !left_task_ids_.empty()) {
            if (left_task_ids_[task.step][task.tile_col] == task.task_id) {
                left_task_ids_[task.step][task.tile_col] = 0;
            }
            if (right_task_ids_[task.tile_row][task.step] == task.task_id) {
                right_task_ids_[task.tile_row][task.step] = 0;
            }
            gemm_done_.erase({task.tile_row, task.tile_col});
        }
    }
}

void PanelController::request_abort() {
    if (phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted) {
        return;
    }
    abort_requested_ = true;
    cancel_requested_ = true;
}

bool PanelController::take_cancel_request() {
    if (cancel_requested_ && !cancel_taken_) {
        cancel_taken_ = true;
        return true;
    }
    return false;
}

bool PanelController::finished() const {
    return phase_ == Phase::Done || phase_ == Phase::Failed || phase_ == Phase::Aborted;
}

const TileTask* PanelController::find_task(std::uint64_t task_id) const {
    for (const TileTask& task : task_graph_) {
        if (task.task_id == task_id) {
            return &task;
        }
    }
    return nullptr;
}

bool PanelController::poll_response(PanelCommandResponse& out) {
    if (!response_.has_value()) {
        return false;
    }
    out = *response_;
    response_.reset();
    return true;
}

void PanelController::reset() {
    request_ = PanelCommandRequest{};
    phase_ = Phase::Idle;
    epoch_ = 0;
    layout_ = PanelFrontLayout{};
    local_grid_ = 0;
    num_steps_ = 0;
    step_ = 0;
    start_cycle_ = 0;
    scan_bytes_total_ = 0;
    scan_bytes_remaining_ = 0;
    scan_elements_done_ = 0;
    scan_done_cycle_ = 0;
    scale_ready_at_ = 0;
    scan_non_finite_ = false;
    front_max_abs_ = 0.0;
    threshold_product_ = 0.0;
    pvector_bytes_remaining_ = 0;
    pvector_done_at_ = 0;
    pvector_write_started_ = false;
    pvector_written_ = false;
    next_task_id_ = 1;
    sequence_ = 0;
    task_graph_.clear();
    cancelled_tasks_.clear();
    outstanding_.clear();
    step_wait_.clear();
    diag_task_ids_.clear();
    left_task_ids_.clear();
    right_task_ids_.clear();
    gemm_done_.clear();
    abort_requested_ = false;
    cancel_requested_ = false;
    cancel_taken_ = false;
    failure_status_ = StatusCode::Ok;
    failure_reason_.clear();
    failed_column_ = hw::command::NONE;
    completed_pivots_ = 0;
    stall_dispatcher_full_ = 0;
    stall_scan_port_ = 0;
    response_.reset();
}

}  // namespace hw::panel_lu
