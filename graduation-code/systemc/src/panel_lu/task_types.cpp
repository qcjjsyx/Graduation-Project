#include "panel_lu/task_types.hpp"

#include <algorithm>

namespace hw::panel_lu {

const char* tile_mode_name(TileMode mode) {
    switch (mode) {
        case TileMode::LuDiag:
            return "LU_DIAG";
        case TileMode::TrsmLeft:
            return "TRSM_LEFT";
        case TileMode::TrsmRight:
            return "TRSM_RIGHT";
        case TileMode::GemmUpdate:
            return "GEMM_UPDATE";
    }
    return "UNKNOWN";
}

const char* task_phase_name(TaskPhase phase) {
    switch (phase) {
        case TaskPhase::Queued:
            return "QUEUED";
        case TaskPhase::Blocked:
            return "BLOCKED";
        case TaskPhase::Dispatched:
            return "DISPATCHED";
        case TaskPhase::Loading:
            return "LOADING";
        case TaskPhase::ReadyWait:
            return "READY_WAIT";
        case TaskPhase::Ready:
            return "READY";
        case TaskPhase::Running:
            return "RUNNING";
        case TaskPhase::ExecDone:
            return "EXEC_DONE";
        case TaskPhase::Writeback:
            return "WRITEBACK";
        case TaskPhase::Done:
            return "DONE";
        case TaskPhase::Failed:
            return "FAILED";
        case TaskPhase::Cancelled:
            return "CANCELLED";
    }
    return "UNKNOWN";
}

namespace {

ConfigCheck fail(StatusCode status, std::string reason) {
    ConfigCheck check;
    check.ok = false;
    check.status = status;
    check.reason = std::move(reason);
    return check;
}

ConfigCheck pass() {
    ConfigCheck check;
    check.ok = true;
    return check;
}

bool check_operand(const TileOperand& operand, std::string* reason) {
    if (!operand.valid) {
        *reason = "operand handle not set";
        return false;
    }
    if (operand.offset % 4 != 0) {
        *reason = "operand offset is not 4B aligned";
        return false;
    }
    if (operand.stride_bytes % 4 != 0 || operand.stride_bytes < 64) {
        *reason = "operand stride must be a 4B multiple and >= 64B (16 FP32)";
        return false;
    }
    return true;
}

}  // namespace

ConfigCheck validate_panel_config(const PanelLuConfig& config) {
    if (config.tile_dim != 16) {
        return fail(
            StatusCode::BadDescriptor,
            "physical compute tile is fixed at 16; descriptor.tile_size=" +
                std::to_string(config.tile_dim) + " is not supported");
    }
    if (config.timing.array_dim != 16 || config.timing.mac_lanes != 256) {
        return fail(StatusCode::BadDescriptor, "PE array must be a 16x16 FMAC array");
    }
    if (config.pe_count == 0) {
        return fail(StatusCode::BadDescriptor, "pe_count must be >= 1");
    }
    if (config.task_slots_per_pe == 0) {
        return fail(StatusCode::BadDescriptor, "task_slots_per_pe must be >= 1");
    }
    if (config.dispatcher_queue_depth == 0) {
        return fail(StatusCode::BadDescriptor, "dispatcher_queue_depth must be >= 1");
    }
    if (config.max_resident_commands == 0 || config.max_resident_commands > 2) {
        return fail(
            StatusCode::BadDescriptor,
            "max_resident_commands must be 1 or 2 (dual-front first version)");
    }
    if (config.max_front_dim == 0 || config.max_front_dim > 512) {
        return fail(StatusCode::BadDescriptor, "max_front_dim must be in [1, 512]");
    }
    if (config.load_bytes_per_cycle < 4 || config.writeback_bytes_per_cycle < 4 ||
        config.scan_bytes_per_cycle < 4) {
        return fail(
            StatusCode::BadDescriptor,
            "port budgets must move at least one FP32 element per cycle");
    }
    if (config.pe_input_fifo_bytes < 2048) {
        return fail(
            StatusCode::BadDescriptor,
            "pe_input_fifo_bytes must hold the largest operand set (A+B of a full "
            "16x16 GEMM = 2048B); the accumulator is a dedicated per-slot register "
            "file and must not be counted as FIFO");
    }
    if (config.load_queue_depth == 0 || config.writeback_queue_depth == 0) {
        return fail(StatusCode::BadDescriptor, "queue depths must be >= 1");
    }
    if (config.timing.gemm_k_ii == 0 || config.timing.div_latency_cycles == 0 ||
        config.timing.fill_cycles == 0 || config.timing.tsolve_fill_cycles == 0 ||
        config.timing.drain_row_cycles == 0 || config.timing.lu_update_cycles == 0 ||
        config.timing.lu_pivot_check_cycles == 0) {
        return fail(StatusCode::BadDescriptor, "timing parameters must be >= 1");
    }
    return pass();
}

ConfigCheck validate_tile_task(const TileTask& task, const PanelLuConfig& config) {
    const std::uint32_t td = config.tile_dim;
    std::string reason;
    if (task.dims.m == 0 || task.dims.m > td || task.dims.n == 0 || task.dims.n > td ||
        task.dims.k == 0 || task.dims.k > td) {
        return fail(
            StatusCode::BadDescriptor,
            "effective M/N/K must be in [1,16], got " + std::to_string(task.dims.m) +
                "x" + std::to_string(task.dims.n) + "x" + std::to_string(task.dims.k));
    }
    if (task.mask.rows != task.dims.m || task.mask.cols != task.dims.n) {
        return fail(
            StatusCode::BadDescriptor,
            "tile mask must equal the effective result dimensions");
    }
    switch (task.mode) {
        case TileMode::LuDiag:
            if (task.dims.m != task.dims.k || task.dims.n != task.dims.k) {
                return fail(StatusCode::BadDescriptor, "LU_DIAG must be a square k x k task");
            }
            if (task.dims.k < 1) {
                return fail(StatusCode::BadDescriptor, "LU_DIAG k must be >= 1");
            }
            break;
        case TileMode::TrsmLeft:
            if (task.dims.m != task.dims.k) {
                return fail(
                    StatusCode::BadDescriptor,
                    "TRSM_LEFT solves L11^-1 * A12 so result rows must equal k");
            }
            break;
        case TileMode::TrsmRight:
            if (task.dims.n != task.dims.k) {
                return fail(
                    StatusCode::BadDescriptor,
                    "TRSM_RIGHT solves A21 * U11^-1 so result cols must equal k");
            }
            break;
        case TileMode::GemmUpdate:
            break;
    }
    if (!check_operand(task.dst, &reason)) {
        return fail(StatusCode::BadDescriptor, "dst: " + reason);
    }
    if (task.mode != TileMode::LuDiag) {
        if (!check_operand(task.lhs, &reason)) {
            return fail(StatusCode::BadDescriptor, "lhs: " + reason);
        }
        if (!check_operand(task.rhs, &reason)) {
            return fail(StatusCode::BadDescriptor, "rhs: " + reason);
        }
    }
    if (task.deps.size() > 8) {
        return fail(StatusCode::BadDescriptor, "too many dependencies (max 8)");
    }
    return pass();
}

ConfigCheck check_front_capability(
    std::uint32_t front_dim,
    std::uint32_t pivot_dim,
    std::uint32_t tile_size,
    hw::command::PivotPolicy policy,
    const PanelLuConfig& config) {
    if (policy == hw::command::PivotPolicy::DynamicPanel) {
        return fail(
            StatusCode::PolicyNotSupported,
            "DYNAMIC_PANEL requires HPU/ATU; not implemented in T06 prototype");
    }
    if (!hw::command::detail::known_pivot_policy(policy)) {
        return fail(StatusCode::BadDescriptor, "unknown pivot policy");
    }
    if (front_dim == 0 || front_dim > config.max_front_dim) {
        return fail(
            StatusCode::BadDescriptor,
            "front_dim outside [1, max_front_dim=" + std::to_string(config.max_front_dim) + "]");
    }
    if (pivot_dim == 0 || pivot_dim > front_dim) {
        return fail(StatusCode::BadDescriptor, "pivot_dim must be in [1, front_dim]");
    }
    if (tile_size != config.tile_dim) {
        return fail(
            StatusCode::BadDescriptor,
            "descriptor.tile_size=" + std::to_string(tile_size) +
                " rejected; physical tile is fixed at 16 (not silently ignored)");
    }
    return pass();
}

std::uint64_t task_load_bytes(const TileTask& task) {
    const std::uint64_t m = task.dims.m;
    const std::uint64_t n = task.dims.n;
    const std::uint64_t k = task.dims.k;
    switch (task.mode) {
        case TileMode::LuDiag:
            return k * k * 4;
        case TileMode::TrsmLeft:
            return (k * k + k * n) * 4;  // diagonal L tile + A12 tile
        case TileMode::TrsmRight:
            return (m * k + k * k) * 4;  // A21 tile + diagonal U tile
        case TileMode::GemmUpdate:
            // L21 + U12 + accumulator read of the target tile.
            return (m * k + k * n + m * n) * 4;
    }
    return 0;
}

std::uint64_t task_operand_fifo_bytes(const TileTask& task) {
    const std::uint64_t m = task.dims.m;
    const std::uint64_t n = task.dims.n;
    const std::uint64_t k = task.dims.k;
    switch (task.mode) {
        case TileMode::LuDiag:
            // The diagonal tile is read into the per-slot accumulator register
            // file, so it does not consume the operand FIFO.
            return 0;
        case TileMode::TrsmLeft:
            return (k * k + k * n) * 4;
        case TileMode::TrsmRight:
            return (m * k + k * k) * 4;
        case TileMode::GemmUpdate:
            return (m * k + k * n) * 4;  // A + B; D lives in the accumulator
    }
    return 0;
}

std::uint64_t task_writeback_bytes(const TileTask& task) {
    const std::uint64_t m = task.dims.m;
    const std::uint64_t n = task.dims.n;
    const std::uint64_t k = task.dims.k;
    switch (task.mode) {
        case TileMode::LuDiag:
            return k * k * 4;
        case TileMode::TrsmLeft:
            return k * n * 4;
        case TileMode::TrsmRight:
            return m * k * 4;
        case TileMode::GemmUpdate:
            return m * n * 4;
    }
    return 0;
}

std::uint64_t task_exec_cycles(const TileTask& task, const PeTimingConfig& timing) {
    const std::uint64_t m = task.dims.m;
    const std::uint64_t n = task.dims.n;
    const std::uint64_t k = task.dims.k;
    switch (task.mode) {
        case TileMode::LuDiag:
            // Data-dependency chain: pivot j -> corner reciprocal/divide
            // (div_latency) -> rank-1 update beat that produces pivot j+1
            // (lu_update). Stage interval = div + update; k-1 intervals for a
            // k x k tile, plus one final pivot check for the last pivot.
            return (k - 1) * (timing.div_latency_cycles + timing.lu_update_cycles) +
                   timing.lu_pivot_check_cycles;
        case TileMode::TrsmLeft:
            return timing.tsolve_fill_cycles + k + n;
        case TileMode::TrsmRight:
            // Non-unit diagonal: precompute k reciprocals through the corner
            // divide path (independent values, pipelined II=1, so
            // div_latency + (k-1) cycles), then solve k columns with
            // multiply-only feedback and drain m rows.
            return timing.tsolve_fill_cycles +
                   (timing.div_latency_cycles + (k - 1)) + k + m;
        case TileMode::GemmUpdate:
            return timing.fill_cycles + k * timing.gemm_k_ii +
                   m * timing.drain_row_cycles;
    }
    return 0;
}

std::uint64_t lu_diag_failure_cycles(
    std::uint32_t pivot_base,
    std::uint32_t failed_column,
    const PeTimingConfig& timing) {
    const std::uint64_t f = failed_column >= pivot_base ? failed_column - pivot_base : 0;
    // Stages 0..f-1 complete fully; the failing pivot is available at the end
    // of the last completed stage, and the pivot check that detects the
    // failure costs lu_pivot_check_cycles.
    return f * (timing.div_latency_cycles + timing.lu_update_cycles) +
           timing.lu_pivot_check_cycles;
}

}  // namespace hw::panel_lu
