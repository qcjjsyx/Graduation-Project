#pragma once

// T06 Panel-LU architecture prototype (executable timing/resource model).
//
// This is NOT a numerical execution backend. It consumes explicitly synthetic
// workloads, models the shared multi-mode PE pool, bounded queues, ports,
// writeback and directed abort/reset, and emits reproducible traces marked
// architecture_prototype / numerical_validation=not_performed. It is not wired
// into CommandDevice and never reports numerical success for a real command.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "panel_lu/panel_controller.hpp"
#include "panel_lu/task_types.hpp"
#include "panel_lu/tile_dispatcher.hpp"
#include "panel_lu/tile_store.hpp"

namespace hw::panel_lu {

// Consolidated record of one internal tile task as observed by the prototype.
struct TaskRecord {
    TileTask task;
    TaskPhase final_phase{TaskPhase::Queued};
    StatusCode status{StatusCode::Ok};
    std::string reason;
    std::uint32_t failed_column{0};
    std::uint32_t pe_id{hw::command::NONE};
    bool result_visible{false};  // committed (externally visible) result
    bool committed{false};
    bool cancelled{false};
    std::uint64_t enqueue_cycle{0};
    std::uint64_t dispatch_cycle{0};
    std::uint64_t load_done_cycle{0};
    std::uint64_t exec_start_cycle{0};
    std::uint64_t exec_done_cycle{0};
    std::uint64_t writeback_done_cycle{0};
    std::uint64_t bytes_read{0};
    std::uint64_t bytes_written{0};
    std::uint64_t stall_dependency_cycles{0};
    std::uint64_t stall_resource_cycles{0};
    std::uint64_t stall_port_cycles{0};
    std::uint64_t stall_writeback_cycles{0};
};

struct PrototypeEvent {
    std::uint64_t cycle{0};
    std::string event;
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::uint64_t task_id{0};
    std::uint32_t pe_id{0};
    std::string detail;
};

class PanelLuPrototype {
public:
    explicit PanelLuPrototype(PanelLuConfig config);

    // Validates the resource configuration and allocates the synthetic tile
    // store and PE pool. Invalid configurations are rejected explicitly.
    ConfigCheck configure();

    // Diagnostic task-id base override for tests (0 = monotonic allocation).
    // Internal prototype hook only; not part of the device contract.
    ConfigCheck submit_command(
        const PanelCommandRequest& request,
        std::uint64_t first_task_id_override = 0);
    void tick();
    bool poll_response(PanelCommandResponse& out);
    void abort_command(std::uint32_t command_id);
    void reset();

    bool run_until_idle(std::uint64_t max_cycles);  // bounded driver, tests only

    std::uint64_t cycle() const { return cycle_; }
    std::uint64_t epoch() const { return epoch_; }
    bool idle() const;
    bool control_fault() const { return control_fault_; }
    const std::string& control_fault_reason() const { return control_fault_reason_; }

    const PanelLuConfig& config() const { return config_; }
    TileStore& store() { return store_; }
    const TileStore& store() const { return store_; }
    TileDispatcher& dispatcher() { return dispatcher_; }
    const TileDispatcher& dispatcher() const { return dispatcher_; }
    std::vector<MultiModePe>& pes() { return pes_; }
    const std::vector<MultiModePe>& pes() const { return pes_; }

    const std::vector<TaskRecord>& task_records() const { return task_records_; }
    const std::vector<PrototypeEvent>& events() const { return events_; }
    const PanelController* controller(std::uint32_t command_id) const;
    const std::vector<TileTask>* command_task_graph(std::uint32_t command_id) const;

    // Trace/report writers. summary.json carries architecture_prototype=true
    // and numerical_validation=not_performed.
    void write_reports(const std::string& output_dir) const;
    std::string summary_json() const;

private:
    struct ControllerSlot {
        std::unique_ptr<PanelController> controller;
        std::uint32_t command_id{hw::command::NONE};
        bool active{false};
    };

    PanelController* find_controller(std::uint32_t command_id);
    void route_pe_responses();
    void handle_cancel_requests();
    void drain_command_responses();
    void record_task(const TileTask& task, const TileTaskResponse* response, bool committed);
    void emit(std::uint64_t cycle, const std::string& event, std::uint32_t command_id,
              std::uint32_t node_id, std::uint64_t task_id, std::uint32_t pe_id,
              const std::string& detail);

    PanelLuConfig config_;
    bool configured_{false};
    std::uint64_t cycle_{0};
    std::uint64_t epoch_{0};
    // Monotonic per-command task-id base: a released resident slot must never
    // reuse task ids, because dispatcher dependency/record tables may still
    // reference recently finished tasks.
    std::uint64_t next_task_id_base_{1};
    std::uint32_t global_grid_{16};
    std::uint32_t max_tiles_{256};
    TileStore store_;
    std::vector<MultiModePe> pes_;
    WritebackReservation writeback_reservation_;
    TileDispatcher dispatcher_;
    std::vector<ControllerSlot> controller_slots_;
    PortBudget load_budget_;
    PortBudget store_write_budget_;
    PortBudget scan_budget_;

    std::vector<TaskRecord> task_records_;
    std::vector<PrototypeEvent> events_;
    std::vector<PanelCommandResponse> command_responses_;
    std::map<std::uint32_t, std::vector<TileTask>> command_graphs_;
    std::map<std::uint32_t, PanelCommandResponse> command_history_;
    std::map<std::uint32_t, PanelCommandRequest> command_requests_;

    bool control_fault_{false};
    std::string control_fault_reason_;
    std::size_t controller_rotation_{0};

    // Aggregate statistics for the summary report.
    std::map<std::string, std::uint64_t> mode_counts_;
    std::map<std::string, std::uint64_t> mode_cycles_;
    std::uint64_t scan_bytes_{0};
    std::uint64_t load_bytes_{0};
    std::uint64_t writeback_bytes_{0};
    std::uint64_t pvector_bytes_{0};
    std::uint64_t tasks_committed_{0};
    std::uint64_t tasks_cancelled_{0};
    std::uint64_t tasks_failed_{0};
};

}  // namespace hw::panel_lu
