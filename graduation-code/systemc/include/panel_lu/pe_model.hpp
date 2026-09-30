#pragma once

// T06 multi-mode PE model. One PE = one 16x16 output-stationary FMAC array,
// a finite number of task slots, a bounded operand FIFO, a corner
// divide/reciprocal path and a result writeback path. Panel/TRSM/GEMM tasks all
// execute on this same engine; a PE runs at most one task on the array at any
// time, while a different slot may load operands or drain a result.
//
// Timing is task-granular and derived from the documented dataflow
// (hw::panel_lu::task_exec_cycles); the module never advances time itself and
// has no dependence on SystemC. The prototype is marked
// architecture_prototype / numerical_validation=not_performed.

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "panel_lu/task_types.hpp"
#include "panel_lu/tile_store.hpp"

namespace hw::panel_lu {

enum class PeEventKind : std::uint8_t {
    TaskAccepted = 0,
    LoadStart = 1,
    LoadDone = 2,
    ExecStart = 3,
    ExecDone = 4,
    WritebackStart = 5,
    WritebackDone = 6,
    TaskDone = 7,
    TaskFailed = 8,
    TaskAborted = 9,
    TaskCancelled = 10,
};

const char* pe_event_name(PeEventKind kind);

struct PeEvent {
    std::uint64_t cycle{0};
    std::uint32_t pe_id{0};
    std::uint32_t slot{0};
    std::uint64_t task_id{0};
    std::uint32_t command_id{hw::command::NONE};
    PeEventKind kind{PeEventKind::TaskAccepted};
};

class MultiModePe {
public:
    MultiModePe(std::uint32_t pe_id, const PanelLuConfig& config);

    std::uint32_t pe_id() const { return pe_id_; }
    const PanelLuConfig& config() const { return config_; }

    bool has_free_slot() const;
    std::uint32_t free_slot_count() const;
    std::uint32_t active_task_count() const;

    // Accepts a task into a free slot when the PE-local operand FIFO has room.
    // Operands are NOT read here: the bounded Loading phase fetches them
    // through TileAccess under the shared read budget, then computes on the
    // staged snapshot. Returns false (stall, not an error) otherwise.
    bool try_accept(
        const TileTask& task,
        std::uint64_t cycle,
        std::string* reason);

    // Advances one cycle. Shared per-cycle budgets and the bounded writeback
    // reservation are owned by the prototype.
    void tick(
        std::uint64_t cycle,
        PortBudget& load_budget,
        PortBudget& store_write_budget,
        WritebackReservation& writeback_reservation,
        TileAccess& access);

    bool poll_response(TileTaskResponse& out, std::uint64_t cycle);

    bool abort_task(std::uint64_t task_id, std::uint64_t cycle);
    std::uint32_t abort_command(std::uint32_t command_id, std::uint64_t cycle);
    // Aborts every task whose epoch differs (reset/old-epoch invalidation).
    std::uint32_t abort_epoch(std::uint64_t epoch, std::uint64_t cycle);

    void reset();

    std::vector<PeEvent> drain_events();
    std::uint64_t array_busy_cycles() const { return array_busy_cycles_; }
    std::uint64_t fifo_stall_cycles() const { return fifo_stall_cycles_; }
    std::uint64_t writeback_stall_cycles() const { return writeback_stall_cycles_; }
    std::uint32_t fifo_used_bytes() const { return fifo_used_bytes_; }

    struct SlotView {
        bool active{false};
        std::uint64_t task_id{0};
        std::uint32_t command_id{hw::command::NONE};
        TileMode mode{TileMode::GemmUpdate};
        TaskPhase phase{TaskPhase::Queued};
    };
    std::vector<SlotView> slot_views() const;

private:
    struct Slot {
        bool active{false};
        TileTask task;
        TaskPhase phase{TaskPhase::Queued};
        std::uint64_t fifo_bytes{0};
        std::uint64_t load_remaining{0};
        std::uint32_t load_stall{0};
        std::uint64_t ready_at{0};
        std::uint64_t exec_remaining{0};
        std::uint64_t wb_remaining{0};
        std::uint32_t wb_stall{0};
        bool wb_reserved{false};
        bool abort_requested{false};
        bool result_visible{false};
        StatusCode status{StatusCode::Ok};
        std::string reason;
        std::uint32_t failed_column{0};
        // Staged operand snapshots (index 0=dst, 1=lhs, 2=rhs per mode) and
        // the functional result. Nothing is read at accept time; the bounded
        // Loading phase fills these under the shared read budget, then runs
        // the kernel on the completed snapshot. The result becomes visible in
        // the tile store only when the modeled writeback completes.
        std::vector<std::vector<float>> operand_tiles;
        std::array<TileOperand, 3> operand_refs{};
        std::array<bool, 3> operand_present{};
        std::uint32_t load_operand{0};
        std::uint32_t load_row{0};
        std::uint32_t load_col{0};
        std::vector<float> result;
        std::uint64_t exec_cycles{0};
        std::uint64_t accepted_cycle{0};
        std::uint64_t load_done_cycle{0};
        std::uint64_t exec_start_cycle{0};
        std::uint64_t exec_done_cycle{0};
        std::uint64_t writeback_done_cycle{0};
        std::uint64_t bytes_read{0};
        std::uint64_t bytes_written{0};
        std::uint64_t response_ready_cycle{0};
    };

    Slot* find_slot(std::uint64_t task_id);
    void emit(std::uint64_t cycle, std::uint32_t slot, const TileTask& task, PeEventKind kind);
    void finalize(
        Slot& slot,
        std::uint64_t cycle,
        StatusCode status,
        const std::string& reason,
        std::uint32_t slot_index);
    void queue_response(Slot& slot, std::uint64_t cycle);

    std::uint32_t pe_id_{0};
    PanelLuConfig config_;
    std::vector<Slot> slots_;
    std::uint32_t fifo_used_bytes_{0};
    std::deque<std::pair<std::uint64_t, TileTaskResponse>> responses_;
    std::vector<PeEvent> events_;
    std::uint64_t array_busy_cycles_{0};
    std::uint64_t fifo_stall_cycles_{0};
    std::uint64_t writeback_stall_cycles_{0};
};

}  // namespace hw::panel_lu
