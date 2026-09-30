#include "panel_lu/pe_model.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "panel_lu/tile_kernels.hpp"

namespace hw::panel_lu {

const char* pe_event_name(PeEventKind kind) {
    switch (kind) {
        case PeEventKind::TaskAccepted:
            return "TASK_ACCEPTED";
        case PeEventKind::LoadStart:
            return "LOAD_START";
        case PeEventKind::LoadDone:
            return "LOAD_DONE";
        case PeEventKind::ExecStart:
            return "EXEC_START";
        case PeEventKind::ExecDone:
            return "EXEC_DONE";
        case PeEventKind::WritebackStart:
            return "WRITEBACK_START";
        case PeEventKind::WritebackDone:
            return "WRITEBACK_DONE";
        case PeEventKind::TaskDone:
            return "TASK_DONE";
        case PeEventKind::TaskFailed:
            return "TASK_FAILED";
        case PeEventKind::TaskAborted:
            return "TASK_ABORTED";
        case PeEventKind::TaskCancelled:
            return "TASK_CANCELLED";
    }
    return "UNKNOWN";
}

namespace {

std::uint64_t failure_exec_cycles(
    const TileTask& task,
    const PeTimingConfig& timing,
    StatusCode status,
    std::uint32_t failed_column,
    bool diagonal_failure) {
    if (status == StatusCode::Ok) {
        return task_exec_cycles(task, timing);
    }
    const std::uint64_t local = failed_column >= task.pivot_base
                                    ? failed_column - task.pivot_base
                                    : 0;
    switch (task.mode) {
        case TileMode::LuDiag:
            return lu_diag_failure_cycles(task.pivot_base, failed_column, timing);
        case TileMode::TrsmLeft:
            return timing.tsolve_fill_cycles + (local + 1) + task.dims.n;
        case TileMode::TrsmRight:
            if (diagonal_failure) {
                // Detected while the pipelined reciprocal of column `local` is
                // produced: issue at local, result after div_latency.
                return timing.div_latency_cycles + local;
            }
            return (timing.div_latency_cycles + (task.dims.k - 1)) +
                   timing.tsolve_fill_cycles + (local + 1) + task.dims.m;
        case TileMode::GemmUpdate:
            return timing.fill_cycles + task.dims.k * timing.gemm_k_ii +
                   (local + 1) * timing.drain_row_cycles;
    }
    return 0;
}

// Operand staging plan: up to three (operand, rows, cols) entries per mode.
// The byte sum equals task_load_bytes(), so the staged load performs exactly
// the modeled amount of port traffic.
struct OperandSpec {
    const TileOperand* operand{nullptr};
    std::uint32_t rows{0};
    std::uint32_t cols{0};
};

std::uint32_t operand_specs(const TileTask& task, std::array<OperandSpec, 3>& specs) {
    switch (task.mode) {
        case TileMode::LuDiag:
            specs[0] = {&task.dst, task.dims.k, task.dims.k};
            return 1;
        case TileMode::TrsmLeft:
            specs[0] = {&task.lhs, task.dims.k, task.dims.k};
            specs[1] = {&task.rhs, task.dims.k, task.dims.n};
            return 2;
        case TileMode::TrsmRight:
            specs[0] = {&task.lhs, task.dims.m, task.dims.k};
            specs[1] = {&task.rhs, task.dims.k, task.dims.k};
            return 2;
        case TileMode::GemmUpdate:
            specs[0] = {&task.dst, task.dims.m, task.dims.n};
            specs[1] = {&task.lhs, task.dims.m, task.dims.k};
            specs[2] = {&task.rhs, task.dims.k, task.dims.n};
            return 3;
    }
    return 0;
}

// Explicit FP64 diagnostic execution: same algorithms on a binary64 copy of
// the staged snapshot; the result is rounded back to binary32 storage. Only
// reachable when the configuration declares PanelNumericMode::Fp64Diagnostic.
kernels::KernelOutcome run_staged_fp64(
    const TileTask& task,
    const std::vector<std::vector<float>>& operands,
    std::vector<float>& result) {
    constexpr std::uint32_t stride = 16;
    std::vector<std::vector<double>> wide(operands.size());
    for (std::size_t index = 0; index < operands.size(); ++index) {
        wide[index].assign(TileStore::kTileFloats, 0.0);
        for (std::size_t i = 0; i < operands[index].size(); ++i) {
            wide[index][i] = static_cast<double>(operands[index][i]);
        }
    }
    std::vector<double> computed(TileStore::kTileFloats, 0.0);
    auto copy_region = [&](std::size_t index, std::uint32_t rows,
                           std::uint32_t cols) {
        std::fill(computed.begin(), computed.end(), 0.0);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t c = 0; c < cols; ++c) {
                computed[r * 16 + c] = wide[index][r * 16 + c];
            }
        }
    };
    kernels::KernelOutcome outcome;
    switch (task.mode) {
        case TileMode::LuDiag:
            copy_region(0, task.dims.k, task.dims.k);
            outcome = kernels::fp64_diagnostic::run_lu_diag(
                computed.data(), stride, task.dims.k, task.threshold_product,
                task.pivot_base);
            break;
        case TileMode::TrsmLeft:
            copy_region(1, task.dims.k, task.dims.n);
            outcome = kernels::fp64_diagnostic::run_trsm_left(
                computed.data(), stride, task.dims.k, task.dims.n,
                wide[0].data(), stride);
            break;
        case TileMode::TrsmRight:
            copy_region(0, task.dims.m, task.dims.k);
            outcome = kernels::fp64_diagnostic::run_trsm_right(
                computed.data(), stride, task.dims.m, task.dims.k,
                wide[1].data(), stride);
            break;
        case TileMode::GemmUpdate:
            copy_region(0, task.dims.m, task.dims.n);
            outcome = kernels::fp64_diagnostic::run_gemm_update(
                computed.data(), stride, wide[1].data(), stride, wide[2].data(),
                stride, task.dims.m, task.dims.n, task.dims.k);
            break;
    }
    result.assign(TileStore::kTileFloats, 0.0f);
    for (std::size_t i = 0; i < computed.size(); ++i) {
        result[i] = static_cast<float>(computed[i]);
    }
    return outcome;
}

// Functional computation on the fully staged operand snapshot.
kernels::KernelOutcome run_staged(
    const TileTask& task,
    const std::vector<std::vector<float>>& operands,
    std::vector<float>& result) {
    constexpr std::uint32_t stride = 16;
    result.assign(TileStore::kTileFloats, 0.0f);
    kernels::KernelOutcome outcome;
    auto copy = [&](std::size_t index, std::uint32_t rows, std::uint32_t cols) {
        result.assign(TileStore::kTileFloats, 0.0f);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t c = 0; c < cols; ++c) {
                result[r * 16 + c] = operands[index][r * 16 + c];
            }
        }
    };
    switch (task.mode) {
        case TileMode::LuDiag:
            copy(0, task.dims.k, task.dims.k);
            return kernels::run_lu_diag(
                result.data(), stride, task.dims.k, task.threshold_product,
                task.pivot_base);
        case TileMode::TrsmLeft:
            copy(1, task.dims.k, task.dims.n);
            return kernels::run_trsm_left(
                result.data(), stride, task.dims.k, task.dims.n, operands[0].data(),
                stride);
        case TileMode::TrsmRight:
            copy(0, task.dims.m, task.dims.k);
            return kernels::run_trsm_right(
                result.data(), stride, task.dims.m, task.dims.k, operands[1].data(),
                stride);
        case TileMode::GemmUpdate:
            copy(0, task.dims.m, task.dims.n);
            return kernels::run_gemm_update(
                result.data(),
                stride,
                operands[1].data(),
                stride,
                operands[2].data(),
                stride,
                task.dims.m,
                task.dims.n,
                task.dims.k);
    }
    outcome.status = StatusCode::BadDescriptor;
    outcome.reason = "unknown tile mode";
    return outcome;
}

}  // namespace

MultiModePe::MultiModePe(std::uint32_t pe_id, const PanelLuConfig& config)
    : pe_id_(pe_id), config_(config) {
    slots_.resize(config.task_slots_per_pe);
}

bool MultiModePe::has_free_slot() const {
    return free_slot_count() > 0;
}

std::uint32_t MultiModePe::free_slot_count() const {
    std::uint32_t count = 0;
    for (const Slot& slot : slots_) {
        if (!slot.active) {
            ++count;
        }
    }
    return count;
}

std::uint32_t MultiModePe::active_task_count() const {
    return static_cast<std::uint32_t>(slots_.size()) - free_slot_count();
}

bool MultiModePe::try_accept(
    const TileTask& task,
    std::uint64_t cycle,
    std::string* reason) {
    Slot* target = nullptr;
    for (Slot& slot : slots_) {
        if (!slot.active) {
            target = &slot;
            break;
        }
    }
    if (target == nullptr) {
        if (reason != nullptr) {
            *reason = "no free PE task slot";
        }
        return false;
    }
    const std::uint64_t fifo_bytes = task_operand_fifo_bytes(task);
    const std::uint64_t load_bytes = task_load_bytes(task);
    if (fifo_bytes > config_.pe_input_fifo_bytes - fifo_used_bytes_) {
        if (reason != nullptr) {
            *reason = "PE operand FIFO full";
        }
        return false;
    }

    const ConfigCheck check = validate_tile_task(task, config_);
    if (!check.ok) {
        // The dispatcher validates before dispatch; reaching this branch is a
        // control bug, reported instead of silently accepted.
        if (reason != nullptr) {
            *reason = "invalid task: " + check.reason;
        }
        return false;
    }

    // Operand staging plan only; no bytes are read at accept time. The
    // bounded Loading phase fetches the operands under the shared read budget
    // and then computes on the completed snapshot.
    std::array<OperandSpec, 3> specs{};
    const std::uint32_t spec_count = operand_specs(task, specs);
    if (spec_count == 0) {
        if (reason != nullptr) {
            *reason = "unknown tile mode";
        }
        return false;
    }
    target->active = true;
    target->task = task;
    target->phase = TaskPhase::Dispatched;
    target->fifo_bytes = fifo_bytes;
    target->load_remaining = load_bytes;
    target->wb_remaining = task_writeback_bytes(task);
    target->load_stall = 0;
    target->wb_stall = 0;
    target->wb_reserved = false;
    target->abort_requested = false;
    target->result_visible = false;
    target->status = StatusCode::Ok;
    target->reason.clear();
    target->failed_column = 0;
    target->result.assign(TileStore::kTileFloats, 0.0f);
    target->operand_tiles.assign(
        spec_count, std::vector<float>(TileStore::kTileFloats, 0.0f));
    target->operand_refs.fill(TileOperand{});
    target->operand_present.fill(false);
    for (std::uint32_t index = 0; index < spec_count; ++index) {
        target->operand_refs[index] = *specs[index].operand;
    }
    target->load_operand = 0;
    target->load_row = 0;
    target->load_col = 0;
    target->exec_cycles = 0;
    target->accepted_cycle = cycle;
    target->load_done_cycle = 0;
    target->exec_start_cycle = 0;
    target->exec_done_cycle = 0;
    target->writeback_done_cycle = 0;
    target->bytes_read = 0;
    target->bytes_written = 0;
    target->response_ready_cycle = 0;
    fifo_used_bytes_ += static_cast<std::uint32_t>(fifo_bytes);
    emit(cycle, static_cast<std::uint32_t>(target - slots_.data()), task, PeEventKind::TaskAccepted);
    return true;
}

void MultiModePe::tick(
    std::uint64_t cycle,
    PortBudget& load_budget,
    PortBudget& store_write_budget,
    WritebackReservation& writeback_reservation,
    TileAccess& access) {
    // Phase 1: per-slot progression (load, ready, execute, drain).
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        Slot& slot = slots_[index];
        if (!slot.active) {
            continue;
        }
        switch (slot.phase) {
            case TaskPhase::Dispatched:
                slot.phase = TaskPhase::Loading;
                emit(cycle, index, slot.task, PeEventKind::LoadStart);
                [[fallthrough]];
            case TaskPhase::Loading: {
                if (slot.abort_requested) {
                    finalize(slot, cycle, StatusCode::Aborted, "aborted before execution", index);
                    continue;
                }
                std::array<OperandSpec, 3> specs{};
                const std::uint32_t spec_count = operand_specs(slot.task, specs);
                bool read_failed = false;
                // Budget-constrained staged fetch: segments of the current
                // operands are read row by row, only while load credit is
                // available. Row/segment progress persists across cycles so an
                // accepted write/read is never replayed.
                while (slot.load_remaining > 0 && slot.load_operand < spec_count) {
                    const OperandSpec& spec = specs[slot.load_operand];
                    const std::uint64_t row_bytes =
                        static_cast<std::uint64_t>(spec.cols - slot.load_col) * 4;
                    std::uint64_t chunk =
                        std::min(row_bytes, load_budget.remaining());
                    chunk -= chunk % 4;
                    if (chunk == 0) {
                        break;  // budget exhausted this cycle
                    }
                    const std::uint32_t count = static_cast<std::uint32_t>(chunk / 4);
                    if (!access.read_operand_segment(
                            slot.operand_refs[slot.load_operand],
                            slot.load_row,
                            slot.load_col,
                            count,
                            slot.operand_tiles[slot.load_operand].data() +
                                static_cast<std::size_t>(slot.load_row) * 16,
                            cycle)) {
                        read_failed = true;
                        break;
                    }
                    load_budget.request(chunk);
                    slot.bytes_read += chunk;
                    slot.load_remaining -= chunk;
                    slot.load_col += count;
                    if (slot.load_col == spec.cols) {
                        slot.load_col = 0;
                        ++slot.load_row;
                        if (slot.load_row == spec.rows) {
                            slot.load_row = 0;
                            ++slot.load_operand;
                        }
                    }
                }
                if (read_failed) {
                    finalize(
                        slot,
                        cycle,
                        StatusCode::AddressFault,
                        "operand load failed on the real access path",
                        index);
                    continue;
                }
                if (slot.load_remaining > 0) {
                    ++slot.load_stall;
                    ++fifo_stall_cycles_;
                    break;
                }
                // Operand bytes complete: compute on the staged snapshot; the
                // result stays isolated until the modeled writeback completes.
                const kernels::KernelOutcome outcome =
                    config_.numeric_mode == PanelNumericMode::Fp64Diagnostic
                        ? run_staged_fp64(slot.task, slot.operand_tiles, slot.result)
                        : run_staged(slot.task, slot.operand_tiles, slot.result);
                slot.status = outcome.status;
                slot.reason = outcome.reason;
                slot.failed_column = outcome.failed_column;
                slot.exec_cycles = failure_exec_cycles(
                    slot.task,
                    config_.timing,
                    outcome.status,
                    outcome.failed_column,
                    outcome.diagonal_failure);
                slot.load_done_cycle = cycle;
                slot.ready_at = cycle + config_.load_latency_cycles;
                slot.phase = TaskPhase::ReadyWait;
                emit(cycle, index, slot.task, PeEventKind::LoadDone);
                break;
            }
            case TaskPhase::ReadyWait:
                if (cycle >= slot.ready_at) {
                    slot.phase = TaskPhase::Ready;
                }
                break;
            case TaskPhase::Ready:
                break;
            case TaskPhase::Running: {
                ++array_busy_cycles_;
                if (slot.exec_remaining > 0) {
                    --slot.exec_remaining;
                }
                if (slot.exec_remaining == 0) {
                    slot.exec_done_cycle = cycle;
                    emit(cycle, index, slot.task, PeEventKind::ExecDone);
                    if (slot.status != StatusCode::Ok) {
                        finalize(slot, cycle, slot.status, slot.reason, index);
                        continue;
                    }
                    if (slot.abort_requested) {
                        finalize(slot, cycle, StatusCode::Aborted, "aborted after execution", index);
                        continue;
                    }
                    slot.phase = TaskPhase::ExecDone;
                }
                break;
            }
            case TaskPhase::ExecDone:
                if (slot.abort_requested) {
                    finalize(slot, cycle, StatusCode::Aborted, "aborted before writeback", index);
                    continue;
                }
                if (slot.wb_remaining == 0) {
                    slot.writeback_done_cycle = cycle;
                    if (!slot.result.empty() &&
                        !access.write_result(slot.task, slot.result.data(), cycle)) {
                        finalize(
                            slot,
                            cycle,
                            StatusCode::AddressFault,
                            "tile writeback failed",
                            index);
                        continue;
                    }
                    slot.result_visible = true;
                    emit(cycle, index, slot.task, PeEventKind::WritebackDone);
                    finalize(slot, cycle, StatusCode::Ok, "", index);
                    continue;
                }
                if (writeback_reservation.acquire()) {
                    slot.wb_reserved = true;
                    slot.phase = TaskPhase::Writeback;
                    emit(cycle, index, slot.task, PeEventKind::WritebackStart);
                } else {
                    ++slot.wb_stall;
                    ++writeback_stall_cycles_;
                }
                break;
            case TaskPhase::Writeback: {
                if (slot.wb_remaining > 0) {
                    const std::uint64_t chunk =
                        std::min<std::uint64_t>(slot.wb_remaining, store_write_budget.remaining());
                    if (chunk > 0 && store_write_budget.request(chunk)) {
                        slot.wb_remaining -= chunk;
                        slot.bytes_written += chunk;
                    } else {
                        ++slot.wb_stall;
                        ++writeback_stall_cycles_;
                    }
                }
                if (slot.wb_remaining == 0 && slot.writeback_done_cycle == 0) {
                    // Fixed writeback latency after the last byte is accepted.
                    slot.writeback_done_cycle = cycle + config_.writeback_latency_cycles;
                }
                if (slot.wb_remaining == 0 && cycle >= slot.writeback_done_cycle) {
                    bool wrote = true;
                    if (!slot.abort_requested && !slot.result.empty()) {
                        wrote = access.write_result(slot.task, slot.result.data(), cycle);
                    }
                    slot.result_visible = !slot.abort_requested && wrote;
                    if (!wrote) {
                        slot.status = StatusCode::AddressFault;
                        slot.reason = "tile writeback failed";
                    }
                    if (slot.wb_reserved) {
                        writeback_reservation.release();
                        slot.wb_reserved = false;
                    }
                    emit(cycle, index, slot.task, PeEventKind::WritebackDone);
                    finalize(
                        slot,
                        cycle,
                        slot.abort_requested ? StatusCode::Aborted : slot.status,
                        slot.abort_requested ? "aborted during writeback" : slot.reason,
                        index);
                }
                break;
            }
            case TaskPhase::Queued:
            case TaskPhase::Blocked:
            case TaskPhase::Done:
            case TaskPhase::Failed:
            case TaskPhase::Cancelled:
                break;
        }
    }

    // Phase 2: launch the next ready task on the shared array (double-buffered
    // accumulator: a task that became ready this cycle may start now).
    bool running = false;
    for (const Slot& slot : slots_) {
        if (slot.active && slot.phase == TaskPhase::Running) {
            running = true;
            break;
        }
    }
    if (!running) {
        Slot* chosen = nullptr;
        for (Slot& slot : slots_) {
            if (!slot.active || slot.phase != TaskPhase::Ready) {
                continue;
            }
            if (chosen == nullptr || slot.task.task_id < chosen->task.task_id) {
                chosen = &slot;
            }
        }
        if (chosen != nullptr) {
            chosen->phase = TaskPhase::Running;
            chosen->exec_start_cycle = cycle;
            chosen->exec_remaining = chosen->exec_cycles;
            emit(
                cycle,
                static_cast<std::uint32_t>(chosen - slots_.data()),
                chosen->task,
                PeEventKind::ExecStart);
        }
    }
}

bool MultiModePe::poll_response(TileTaskResponse& out, std::uint64_t cycle) {
    if (responses_.empty()) {
        return false;
    }
    if (responses_.front().first > cycle) {
        return false;
    }
    out = responses_.front().second;
    responses_.pop_front();
    return true;
}

MultiModePe::Slot* MultiModePe::find_slot(std::uint64_t task_id) {
    for (Slot& slot : slots_) {
        if (slot.active && slot.task.task_id == task_id) {
            return &slot;
        }
    }
    return nullptr;
}

void MultiModePe::emit(
    std::uint64_t cycle,
    std::uint32_t slot,
    const TileTask& task,
    PeEventKind kind) {
    PeEvent event;
    event.cycle = cycle;
    event.pe_id = pe_id_;
    event.slot = slot;
    event.task_id = task.task_id;
    event.command_id = task.command_id;
    event.kind = kind;
    events_.push_back(event);
}

void MultiModePe::finalize(
    Slot& slot,
    std::uint64_t cycle,
    StatusCode status,
    const std::string& reason,
    std::uint32_t slot_index) {
    slot.status = status;
    slot.reason = reason;
    if (status != StatusCode::Ok) {
        slot.result_visible = false;
    }
    switch (status) {
        case StatusCode::Ok:
            emit(cycle, slot_index, slot.task, PeEventKind::TaskDone);
            break;
        case StatusCode::Aborted:
            emit(cycle, slot_index, slot.task, PeEventKind::TaskCancelled);
            break;
        default:
            emit(cycle, slot_index, slot.task, PeEventKind::TaskFailed);
            break;
    }
    queue_response(slot, cycle);
}

void MultiModePe::queue_response(Slot& slot, std::uint64_t cycle) {
    TileTaskResponse response;
    response.task_id = slot.task.task_id;
    response.epoch = slot.task.epoch;
    response.command_id = slot.task.command_id;
    response.node_id = slot.task.node_id;
    response.step = slot.task.step;
    response.tile_row = slot.task.tile_row;
    response.tile_col = slot.task.tile_col;
    response.mode = slot.task.mode;
    response.pe_id = pe_id_;
    response.slot = static_cast<std::uint32_t>(&slot - slots_.data());
    response.status = slot.status;
    response.reason = slot.reason;
    response.failed_column = slot.failed_column;
    response.result_visible = slot.result_visible;
    response.enqueue_cycle = 0;  // filled in by the dispatcher
    response.dispatch_cycle = slot.accepted_cycle;
    response.load_done_cycle = slot.load_done_cycle;
    response.exec_start_cycle = slot.exec_start_cycle;
    response.exec_done_cycle = slot.exec_done_cycle;
    response.writeback_done_cycle = slot.writeback_done_cycle;
    response.bytes_read = slot.bytes_read;
    response.bytes_written = slot.bytes_written;
    response.stall_port_cycles = slot.load_stall;
    response.stall_writeback_cycles = slot.wb_stall;
    responses_.emplace_back(cycle + config_.response_hold_cycles, response);

    fifo_used_bytes_ -= static_cast<std::uint32_t>(slot.fifo_bytes);
    const bool was_reserved = slot.wb_reserved;
    slot = Slot{};
    slot.wb_reserved = false;
    (void)was_reserved;
}

bool MultiModePe::abort_task(std::uint64_t task_id, std::uint64_t cycle) {
    Slot* slot = find_slot(task_id);
    if (slot == nullptr) {
        return false;
    }
    switch (slot->phase) {
        case TaskPhase::Dispatched:
        case TaskPhase::Loading:
        case TaskPhase::ReadyWait:
        case TaskPhase::Ready:
            finalize(*slot, cycle, StatusCode::Aborted, "aborted before execution",
                     static_cast<std::uint32_t>(slot - slots_.data()));
            return true;
        case TaskPhase::Running:
        case TaskPhase::ExecDone:
        case TaskPhase::Writeback:
            slot->abort_requested = true;
            return true;
        default:
            return false;
    }
}

std::uint32_t MultiModePe::abort_command(std::uint32_t command_id, std::uint64_t cycle) {
    std::uint32_t count = 0;
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        Slot& slot = slots_[index];
        if (!slot.active || slot.task.command_id != command_id) {
            continue;
        }
        switch (slot.phase) {
            case TaskPhase::Dispatched:
            case TaskPhase::Loading:
            case TaskPhase::ReadyWait:
            case TaskPhase::Ready:
                finalize(slot, cycle, StatusCode::Aborted, "command aborted", index);
                ++count;
                break;
            case TaskPhase::Running:
            case TaskPhase::ExecDone:
            case TaskPhase::Writeback:
                slot.abort_requested = true;
                ++count;
                break;
            default:
                break;
        }
    }
    return count;
}

std::uint32_t MultiModePe::abort_epoch(std::uint64_t epoch, std::uint64_t cycle) {
    (void)cycle;
    std::uint32_t count = 0;
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        Slot& slot = slots_[index];
        if (!slot.active || slot.task.epoch == epoch) {
            continue;
        }
        slot.abort_requested = true;
        ++count;
    }
    return count;
}

void MultiModePe::reset() {
    slots_.assign(config_.task_slots_per_pe, Slot{});
    responses_.clear();
    events_.clear();
    fifo_used_bytes_ = 0;
    array_busy_cycles_ = 0;
    fifo_stall_cycles_ = 0;
    writeback_stall_cycles_ = 0;
}

std::vector<PeEvent> MultiModePe::drain_events() {
    std::vector<PeEvent> out;
    out.swap(events_);
    return out;
}

std::vector<MultiModePe::SlotView> MultiModePe::slot_views() const {
    std::vector<SlotView> views;
    for (const Slot& slot : slots_) {
        SlotView view;
        view.active = slot.active;
        view.task_id = slot.task.task_id;
        view.command_id = slot.task.command_id;
        view.mode = slot.task.mode;
        view.phase = slot.phase;
        views.push_back(view);
    }
    return views;
}

}  // namespace hw::panel_lu
