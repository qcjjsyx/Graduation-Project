#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "controlplane/command_fetch.hpp"
#include "controlplane/completion_writer.hpp"
#include "controlplane/dma_arbiter.hpp"
#include "controlplane/exec_types.hpp"
#include "controlplane/test_stub_unit.hpp"
#include "controlplane/token_scoreboard.hpp"
#include "dataplane/dataplane.hpp"
#include "panel_lu/panel_engine.hpp"

namespace hw::controlplane {

struct ControlPlaneConfig {
    // Residency: at most this many fronts may hold scratch slots at once.
    std::uint32_t max_resident_fronts{2};
    // Shared execution unit pools (both fronts draw from the same pools).
    std::uint32_t panel_unit_count{1};
    std::uint32_t trsm_unit_count{2};
    std::uint32_t gemm_unit_count{1};
    std::uint32_t solve_unit_count{1};
    std::uint32_t assemble_unit_count{1};
    std::uint32_t factor_unit_count{1};
    // Scratchpad: slot count / per-slot capacity / ports (T04 checks apply).
    // Default slot count matches the residency requirement
    // (2 * max_resident_fronts + 1 = 5).
    std::uint32_t buffer_slot_count{5};
    std::uint64_t buffer_capacity_bytes{64 * 1024};
    std::uint32_t buffer_port_count{4};
    // Bounded scheduling window and completion queue.
    std::uint32_t command_window_depth{16};
    std::uint32_t completion_queue_depth{16};
    std::uint32_t dma_queue_depth{8};
    ExecBackend backend{ExecBackend::TestStub};
    bool enable_test_stub{false};
    StubConfig stub;
    // T07 real backend: binds the shared PanelEngine to Panel slots and the
    // FactorStoreUnit to STORE_FACTOR. Explicitly selected, never implicit.
    bool enable_real_panel{false};
    hw::panel_lu::PanelLuConfig panel;
    hw::dataplane::MemoryServiceConfig memory;
    hw::dataplane::DmaConfig dma;
};

enum class BatchState { Idle, Running, Completed, Faulted };

const char* batch_state_name(BatchState state);

struct CommandStats {
    std::uint32_t command_id{0};
    std::uint32_t node_id{0};
    std::string opcode;
    std::string unit_kind;
    std::int32_t unit_id{-1};
    std::uint64_t fetch_cycle{0};
    std::uint64_t admit_cycle{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint64_t completion_cycle{0};
    std::string status;
    std::string reason;
    std::uint64_t bytes_read{0};
    std::uint64_t bytes_written{0};
    std::uint64_t stall_window{0};
    std::uint64_t stall_dep{0};
    std::uint64_t stall_resource{0};
    std::uint64_t stall_dma{0};
    std::uint64_t stall_cycles{0};
    std::uint32_t pivot_count{0};
    std::uint32_t overflow_count{0};
    std::uint32_t dma_jobs{0};
};

struct UnitBusyRecord {
    std::string kind;
    std::uint32_t unit_id{0};
    std::uint32_t command_id{0};
    std::uint32_t node_id{0};
    std::uint64_t start{0};
    std::uint64_t finish{0};
};

struct FrontBusyRecord {
    std::uint32_t slot{0};
    std::uint32_t node_id{0};
    std::uint64_t start{0};
    std::uint64_t finish{0};
};

// Unified SystemC-command control top: fixed per-cycle phases, non-blocking
// modules, no internal time-advancing loops.
class CommandDevice : public hw::dataplane::MemoryResponseSink {
public:
    CommandDevice(hw::dataplane::LoadedArtifact artifact, ControlPlaneConfig config);
    ~CommandDevice();

    // Shared panel compute subsystem (nullptr when the real backend is off).
    hw::panel_lu::PanelEngine* panel_engine() { return panel_engine_.get(); }
    const hw::panel_lu::PanelEngine* panel_engine() const { return panel_engine_.get(); }
    bool real_panel_enabled() const { return panel_engine_ != nullptr; }

    // Test/diagnostic: active scratch front buffer of a node (slot == NONE
    // after the node's context is released).
    hw::dataplane::BufferHandle front_slot(std::uint32_t node_id) const;

    // Test/diagnostic: in-flight memory request count and active command of a
    // unit.
    std::size_t unit_pending_request_count(UnitKind kind, std::uint32_t unit_id) const;
    std::uint32_t unit_active_command_id(UnitKind kind, std::uint32_t unit_id) const;


    // Explicit backend gating: without real kernels the device refuses to run
    // instead of silently succeeding on the test stub.
    bool start(std::string* error);
    void tick();  // exactly one cycle, non-blocking
    // Test-driver convenience only (bounded); the runtime never calls this.
    void run(std::uint64_t max_cycles);

    bool finished() const;
    BatchState state() const { return state_; }
    const std::string& fault_reason() const { return fault_reason_; }
    bool control_flow_pass() const { return state_ == BatchState::Completed; }

    void abort_node(std::uint32_t node_id);
    void reset();
    void inject_failure(std::uint32_t command_id, ExecFailure failure);

    // MemoryResponseSink for ExecUnit client responses (P-vector writes).
    void deliver_memory_response(
        const hw::dataplane::MemoryResponse& response) override;

    std::uint64_t cycle() const { return plane_.cycle(); }
    hw::dataplane::DataPlane& dataplane() { return plane_; }
    const hw::dataplane::DeviceBootConfig& boot() const { return plane_.boot(); }
    TokenScoreboard& scoreboard() { return scoreboard_; }
    const TokenScoreboard& scoreboard() const { return scoreboard_; }

    const std::vector<CommandStats>& command_stats() const { return command_stats_; }
    const std::vector<UnitBusyRecord>& unit_records() const { return unit_records_; }
    const std::vector<FrontBusyRecord>& front_records() const { return front_records_; }
    const std::map<std::uint32_t, hw::command::StatusCode>& completion_status() const {
        return completion_status_;
    }
    std::uint32_t completion_written() const {
        return completion_writer_.written_count();
    }
    bool batch_faulted() const { return state_ == BatchState::Faulted; }

    void write_reports(const std::string& output_dir) const;

    // Compact state dump for tests/diagnostics.
    std::string debug_dump() const;

    // DMA progress of one command (for abort/reset timing tests).
    bool command_dma_progress(
        std::uint32_t command_id,
        std::uint32_t* issued,
        std::uint32_t* done,
        std::uint32_t* cancelled) const;

    // Command whose output writes are still in flight (P-vector write), or
    // NONE. Used by abort timing tests.
    std::uint32_t pending_pvector_command() const;

    // Number of DMA jobs re-queued after transient buffer contention.
    std::uint64_t dma_retry_count() const { return dma_retry_count_; }

private:
    struct Tracker {
        hw::command::Command command;
        std::vector<std::uint32_t> waits;
        // The scheduler window is the set of trackers that are not yet Running:
        // InWindow (raw) -> Resolving (descriptor reads) -> Waiting (deps or
        // resources) -> Running -> PostExec -> Completing -> Done.
        enum class Phase {
            InWindow,
            Resolving,
            Waiting,
            Running,
            PostExec,
            Completing,
            Done
        };
        Phase phase{Phase::InWindow};

        bool resolve_pending{false};
        std::uint64_t resolve_request_id{0};
        std::uint64_t wait_request_id{0};
        bool waits_ready{false};
        bool stage2_pending{false};
        bool solve_load_pending{false};
        bool solve_store_pending{false};
        std::uint32_t tile_width{0};

        // Filled by descriptor resolution.
        bool view_ready{false};
        hw::dataplane::FrontView front_view;
        hw::dataplane::FactorView factor_view;
        hw::dataplane::ContributionView contribution_view;
        hw::dataplane::KernelView kernel_view;
        hw::dataplane::SolveView solve_view;
        hw::dataplane::RegionView region_view;

        // Unit / DMA bookkeeping.
        UnitKind unit_kind{UnitKind::Panel};
        std::uint32_t unit_id{hw::command::NONE};
        bool unit_active{false};
        std::uint64_t txn_id{0};
        std::uint32_t dma_jobs_done{0};
        std::uint32_t dma_jobs_cancelled{0};
        std::uint32_t dma_jobs_total{0};
        bool aborted{false};

        // Chunked SOLVE traversal (the whole RHS is moved, never a prefix).
        std::uint32_t solve_rows{0};
        std::uint32_t solve_cols{0};
        std::uint32_t solve_row{0};
        std::uint32_t solve_elem{0};
        bool solve_store_issued{false};
        // Logical DMA jobs rejected by transient contention, awaiting a safe
        // re-submit (no beat has run, so no side effects exist).
        std::vector<hw::dataplane::DmaRequest> job_retries;

        // Buffer leases (acquired atomically at execution start).
        hw::dataplane::BufferLease front_lease{};
        hw::dataplane::BufferLease temp_lease{};
        bool has_front_lease{false};
        bool has_temp_lease{false};
        bool front_write_lease{false};

        // Slots owned by this command (NODE_BEGIN context / solve temp).
        bool owns_context{false};
        bool owns_temp{false};
        std::uint32_t context_slot{hw::command::NONE};
        std::uint32_t front_slot_index{hw::command::NONE};

        // P-vector identity write after successful PANEL_LU.
        bool pvector_pending{false};
        std::uint64_t pvector_request_id{0};

        hw::command::StatusCode status{hw::command::StatusCode::Ok};
        std::string reason;
        std::uint32_t pivot_count{0};
        std::uint32_t overflow_count{0};
        std::uint64_t bytes_read{0};
        std::uint64_t bytes_written{0};
        std::uint64_t stall_window{0};
        std::uint64_t stall_dep{0};
        std::uint64_t stall_resource{0};
        std::uint64_t stall_dma{0};
        std::uint64_t fetch_cycle{0};
        std::uint64_t admit_cycle{0};
        std::uint64_t start_cycle{0};
        std::uint64_t finish_cycle{0};
    };

    struct FrontSlot {
        bool active{false};
        bool busy{false};  // one front-touching command at a time (T05 v1)
        std::uint32_t node_id{hw::command::NONE};
        std::string owner;
        hw::dataplane::BufferHandle front{};
        hw::dataplane::BufferHandle temp{};
        std::uint64_t occupy_start{0};
        hw::dataplane::FrontView front_view;
    };

    void update_phase(std::uint64_t cycle);
    void dispatch_phase(std::uint64_t cycle);
    void issue_phase(std::uint64_t cycle);

    Tracker* find_tracker(std::uint32_t command_id);
    Tracker* find_tracker_by_txn(std::uint64_t txn_id);
    Tracker* find_tracker_by_job(std::uint64_t job_id);
    Tracker* find_tracker_by_pvector(std::uint64_t request_id);

    void try_execute(Tracker& tracker, std::uint64_t cycle);
    IExecutionUnit* find_free_unit(UnitKind kind);
    bool unit_bound(UnitKind kind) const;
    bool issue_solve_store(Tracker& tracker, std::uint64_t cycle);
    bool submit_pvector_write(Tracker& tracker, std::uint64_t cycle);
    std::uint64_t enqueue_dma_job(
        Tracker& tracker,
        const hw::dataplane::DmaRequest& request,
        std::uint64_t cycle);
    void handle_job_done(
        Tracker& tracker,
        const DmaArbiter::Completion& completion,
        std::uint64_t cycle);
    void handle_exec_response(Tracker& tracker, const ExecResponse& response, std::uint64_t cycle);
    void handle_pvector_done(Tracker& tracker, const hw::dataplane::MemoryResponse& response, std::uint64_t cycle);
    void release_command_resources(Tracker& tracker, std::uint64_t cycle);
    void finish_command(Tracker& tracker, hw::command::StatusCode status, const std::string& reason, std::uint64_t cycle);
    void enqueue_completion(Tracker& tracker, std::uint64_t cycle);
    bool issue_solve_load(Tracker& tracker, std::uint64_t cycle);
    void advance_solve(Tracker& tracker, std::uint64_t cycle);
    std::uint64_t requeue_dma_job(
        Tracker& tracker,
        const hw::dataplane::DmaRequest& request,
        std::uint64_t cycle);
    void drain_completion_results(std::uint64_t cycle);
    void maybe_cleanup_node(std::uint32_t node_id, std::uint64_t cycle);
    void fault(const std::string& reason, std::uint64_t cycle);

    bool acquire_exec_resources(
        Tracker& tracker,
        UnitKind kind,
        bool need_front_write,
        bool need_temp_read,
        std::uint64_t cycle);
    std::optional<std::uint32_t> context_for_node(std::uint32_t node_id) const;

    hw::dataplane::LoadedArtifact artifact_;
    ControlPlaneConfig config_;
    hw::dataplane::DataPlane plane_;
    hw::dataplane::TraceRecorder& trace_;

    CommandFetch fetch_;
    CompletionWriter completion_writer_;
    DmaArbiter dma_arbiter_;
    TokenScoreboard scoreboard_;
    std::unique_ptr<hw::panel_lu::PanelEngine> panel_engine_;
    std::vector<std::unique_ptr<IExecutionUnit>> units_;
    std::map<std::uint64_t, hw::dataplane::MemoryResponse> exec_inbox_;

    std::vector<FrontSlot> front_slots_;
    std::deque<std::unique_ptr<Tracker>> trackers_;
    std::set<std::uint32_t> aborted_nodes_;
    std::set<std::uint32_t> commit_seen_;
    std::map<std::uint32_t, hw::command::StatusCode> completion_status_;

    // Bookkeeping maps.
    std::map<std::uint64_t, std::uint32_t> resolve_map_;   // reader request -> command
    std::map<std::uint64_t, std::uint32_t> job_map_;       // dma job -> command
    std::map<std::uint64_t, std::uint32_t> txn_map_;       // exec txn -> command
    std::map<std::uint64_t, std::uint32_t> pvector_map_;   // memory request -> command

    std::uint64_t next_txn_id_{1};
    std::uint64_t epoch_{0};
    std::uint64_t dma_retry_count_{0};
    BatchState state_{BatchState::Idle};
    std::string fault_reason_;
    std::uint32_t completed_commands_{0};
    std::uint32_t total_commands_{0};

    std::vector<CommandStats> command_stats_;
    std::vector<UnitBusyRecord> unit_records_;
    std::vector<FrontBusyRecord> front_records_;
    std::map<std::uint64_t, UnitBusyRecord> open_unit_records_;  // txn -> record
};

}  // namespace hw::controlplane
