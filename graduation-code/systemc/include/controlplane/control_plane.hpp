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
#include "controlplane/assembly_exec_unit.hpp"
#include "controlplane/test_stub_unit.hpp"
#include "controlplane/token_scoreboard.hpp"
#include "dataplane/dataplane.hpp"
#include "panel_lu/panel_engine.hpp"

namespace hw::controlplane {

struct ControlPlaneConfig {
    // Residency: at most this many fronts may hold scratch slots at once.
    std::uint32_t max_resident_fronts{2};
    // Shared execution unit pools (both fronts draw from the same pools). 每个表明一个执行单元
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
    StubConfig stub; //测试桩执行单元，不执行真实数值算法，只模拟行为
    // T07 real backend: binds the shared PanelEngine to Panel slots and the
    // FactorStoreUnit to STORE_FACTOR. Explicitly selected, never implicit.
    bool enable_real_panel{false};
    hw::panel_lu::PanelLuConfig panel;
    AssemblyConfig assembly;
    hw::dataplane::MemoryServiceConfig memory;
    hw::dataplane::DmaConfig dma;
};

enum class BatchState { Idle, Running, Completed, Faulted };

const char* batch_state_name(BatchState state);

struct CommandStats {
    std::uint32_t command_id{0};       // 命令在当前 batch 中的唯一编号
    std::uint32_t node_id{0};          // 命令所属的计算节点编号
    std::string opcode;                // 命令操作码名称
    std::string unit_kind;             // 执行该命令的执行单元类型
    std::int32_t unit_id{-1};          // 执行单元编号，-1 表示尚未分配或不适用
    std::uint64_t fetch_cycle{0};      // 从 command buffer 取出命令的周期
    std::uint64_t admit_cycle{0};      // 命令通过调度准入的周期
    std::uint64_t start_cycle{0};      // 执行单元开始执行命令的周期
    std::uint64_t finish_cycle{0};     // 执行单元完成命令执行的周期
    std::uint64_t completion_cycle{0}; // 完成记录写回并完成收尾的周期
    std::string status;                // 命令最终状态
    std::string reason;                // 失败或特殊状态的原因
    std::uint64_t bytes_read{0};       // 命令从设备内存读取的字节数
    std::uint64_t bytes_written{0};    // 命令向设备内存写入的字节数
    std::uint64_t stall_window{0};     // 因调度窗口已满而等待的周期数
    std::uint64_t stall_dep{0};        // 因依赖 token 未满足而等待的周期数
    std::uint64_t stall_resource{0};   // 因执行单元或 buffer 资源不足而等待的周期数
    std::uint64_t stall_dma{0};        // 因 DMA 队列或 DMA 资源而等待的周期数
    std::uint64_t stall_cycles{0};     // 命令累计等待周期数
    std::uint32_t pivot_count{0};      // 已完成的主元数，失败时表示首个失败列号
    std::uint32_t overflow_count{0};   // 检测到的数值溢出次数
    std::uint32_t dma_jobs{0};         // 命令关联或提交的 DMA 作业数量
    // T10 GEMM_SCHUR statistics (empty/zero for other opcodes).
    std::string gemm_backend;           // GEMM 使用的数值计算后端
    std::uint64_t saturation_count{0};  // GEMM 局部量化发生饱和的次数
    std::uint64_t local_conversion_bytes{0}; // FP/INT 局部转换涉及的字节数
    std::uint64_t scale_reduction_cycles{0}; // 计算输入 scale 所消耗的周期数
    std::uint64_t quantize_cycles{0};        // 输入量化所消耗的周期数
    std::uint64_t integer_compute_cycles{0}; // INT GEMM 核心计算所消耗的周期数
    std::uint64_t dequantize_cycles{0};      // 输出反量化所消耗的周期数
    std::int32_t exponent_a{0};              // 矩阵 A 使用的量化指数
    std::int32_t exponent_b{0};              // 矩阵 B 使用的量化指数
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
    // Device-wide unified on-chip bandwidth arbiter.
    hw::dataplane::OnChipArbiter& on_chip_arbiter() { return on_chip_; }
    const hw::dataplane::OnChipArbiter& on_chip_arbiter() const { return on_chip_; }
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
    // Device status is distinct from independent numerical validation.
    bool numeric_status_pass() const;

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
            InWindow,   // 已进入调度窗口，尚未开始解析
            Resolving,  // 正在读取并解析 descriptor
            Waiting,    // 等待依赖 token 或执行资源
            Running,    // 已获得资源，正在执行命令
            PostExec,   // 执行结束，正在处理后续结果或资源释放
            Completing, // 正在写入 completion 并发布结果 token
            Done        // 命令已完成，不再参与调度
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
        // T10 GEMM_SCHUR backend statistics of the last executed GEMM command
        // (kept per command so completion/token statistics are truthful).
        hw::panel_lu::GemmBackend gemm_backend{hw::panel_lu::GemmBackend::Fp32Device};
        std::uint64_t gemm_saturation_count{0};
        std::uint64_t gemm_local_conversion_bytes{0};
        std::uint64_t gemm_scale_reduction_cycles{0};
        std::uint64_t gemm_quantize_cycles{0};
        std::uint64_t gemm_integer_compute_cycles{0};
        std::uint64_t gemm_dequantize_cycles{0};
        std::int32_t gemm_exponent_a{0};
        std::int32_t gemm_exponent_b{0};
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
        // Latched PANEL_LU numeric context (front_pre_elimination_max_abs and
        // its threshold product); consumed by top-level TRSM diagonal checks
        // and never re-derived from the eliminated front.
        double front_scale{0.0};
        double threshold_product{0.0};
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
    std::string model_config_error_;
    hw::dataplane::DataPlane plane_;
    hw::dataplane::TraceRecorder& trace_;
    hw::dataplane::OnChipArbiter on_chip_;

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
