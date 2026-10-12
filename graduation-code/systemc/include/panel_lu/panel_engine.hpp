#pragma once

// T07 Panel compute engine: the shared, top-level-owned compute subsystem for
// PANEL_LU (and, later, T09/T10 TRSM/GEMM) command execution.
//
// One engine owns:
//   - pe_count homogeneous multi-mode PEs,
//   - one bounded TileDispatcher + tile version table,
//   - one shared scratchpad read budget (PE loads AND full-front scan; bounded,
//     fairly rotated) and one writeback budget,
//   - up to max_resident_commands PanelController contexts working on real
//     front buffers through BufferManager leases.
//
// The engine never advances time itself; its owner calls tick() exactly once
// per cycle. It is data-agnostic with respect to the front storage: the only
// byte path is BufferManager lease_read/lease_write, so the T05 adapter can
// hand it the command's front write lease (readable in place). Status codes
// are reused from hw::command::StatusCode.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dataplane/on_chip_arbiter.hpp"
#include "panel_lu/compute_controller.hpp"
#include "panel_lu/gemm_controller.hpp"
#include "panel_lu/panel_controller.hpp"
#include "panel_lu/trsm_controller.hpp"
#include "panel_lu/task_types.hpp"
#include "panel_lu/tile_dispatcher.hpp"
#include "panel_lu/tile_store.hpp"

namespace hw::panel_lu {

// Engine-level request/response between a command adapter and the shared
// engine. No controlplane types are used here (the adapter converts).
struct PanelEngineRequest {
    std::uint64_t txn_id{0};
    std::uint64_t epoch{0};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::uint32_t total_dim{0};
    std::uint32_t pivot_dim{0};
    std::uint32_t tile_size{16};
    hw::command::PivotPolicy policy{hw::command::PivotPolicy::Static};
    std::uint32_t threshold_bits{0};
    hw::dataplane::BufferHandle front{};
    hw::dataplane::BufferLease front_lease{};
    std::uint64_t front_offset{0};
    std::string label;
    // Compute command routed through the shared engine.
    ComputeKind compute_kind{ComputeKind::PanelLu};
    // KERNEL_DESC block shape for TRSM validation (ignored for PANEL_LU).
    std::uint32_t kernel_m{0};
    std::uint32_t kernel_n{0};
    std::uint32_t kernel_k{0};
    std::uint32_t kernel_tile_size{16};
    // KERNEL_DESC backend code (SystemCFp32DeviceModel = 0x01; other/INT32
    // backends are refused for the FP32 device path).
    std::uint32_t kernel_backend{0x01};
    // Decoded and validated SCALE_DESC policy for the INT32 GEMM backend.
    // Ignored for the FP32 backend (which must not reference a SCALE_DESC).
    GemmScalePolicy gemm_scale;
    bool kernel_has_scale{false};
    // KERNEL_DESC id (kept for tracing/unified GEMM records).
    std::uint32_t kernel_descriptor_id{hw::command::NONE};
    // Latched front numeric context (PANEL response -> TRSM U checks).
    double front_scale{0.0};
    double threshold_product{0.0};
};

struct PanelEngineResponse {
    std::uint64_t txn_id{0};
    std::uint64_t epoch{0};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    StatusCode status{StatusCode::Ok};
    std::string reason;
    std::uint32_t pivot_count{0};
    std::uint32_t failed_column{hw::command::NONE};
    std::uint32_t group_id{0};
    double front_scale{0.0};
    double threshold_product{0.0};
    bool needs_pvector_write{false};
    std::uint32_t pvector_dim{0};
    std::uint64_t bytes_read{0};
    std::uint64_t bytes_written{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint32_t tasks_total{0};
    std::uint32_t tasks_cancelled{0};
    // T10 GEMM_SCHUR backend statistics (zero for other compute kinds).
    GemmBackend gemm_backend{GemmBackend::Fp32Device};
    std::uint64_t saturation_count{0};
    std::uint64_t overflow_count{0};
    std::uint64_t local_conversion_bytes{0};
    std::uint64_t scale_reduction_cycles{0};
    std::uint64_t quantize_cycles{0};
    std::uint64_t integer_compute_cycles{0};
    std::uint64_t dequantize_cycles{0};
    std::int32_t exponent_a{0};
    std::int32_t exponent_b{0};
};

// Unified GEMM boundary logs: when a GEMM context terminates, its GemmReq /
// GemmRsp records are appended here so reports/tests can inspect the same
// contract the upper layer consumed.
struct EngineGemmLog {
    std::vector<GemmReq> requests;
    std::vector<GemmRsp> responses;
};

// Diagnostic record of one internal tile task (tests/trace only).
struct EngineTaskRecord {
    TileTask task;
    StatusCode status{StatusCode::Ok};
    std::string reason;
    std::uint32_t failed_column{0};
    std::uint32_t pe_id{hw::command::NONE};
    std::uint32_t group_id{0};
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
    // T10 INT32 GEMM per-task bookkeeping (zero for other modes).
    GemmBackend gemm_backend{GemmBackend::Fp32Device};
    GemmPhase gemm_failed_phase{GemmPhase::None};
    std::uint32_t gemm_failed_local_row{0};
    std::uint32_t gemm_failed_local_col{0};
    std::int32_t gemm_exponent_a{0};
    std::int32_t gemm_exponent_b{0};
    std::uint64_t saturation_count{0};
    std::uint64_t overflow_count{0};
    std::uint64_t local_conversion_bytes{0};
    std::uint64_t scale_reduction_cycles{0};
    std::uint64_t quantize_cycles{0};
    std::uint64_t integer_compute_cycles{0};
    std::uint64_t dequantize_cycles{0};
};

class PanelEngine : public TileVersionStore {
public:
    explicit PanelEngine(PanelLuConfig config);

    // TileVersionStore (keys are tile_version_key(slot, view, tile_index)).
    std::uint32_t version(std::uint64_t key) const override;
    bool commit(std::uint64_t key, std::uint32_t expected_version) override;

    // Validates the resource configuration and allocates the PE pool. Real
    // front data is never materialized here.
    ConfigCheck configure();

    // Binds the scratchpad manager used by the real lease data path.
    void set_buffers(hw::dataplane::BufferManager* buffers) { buffers_ = buffers; }

    // Binds the device-wide on-chip bandwidth arbiter. The owner must call
    // arbiter->begin_cycle() exactly once per tick before ticking the engine;
    // without an external arbiter the engine uses its own fallback (standalone
    // tests) and initializes it in tick().
    void set_arbiter(hw::dataplane::OnChipArbiter* arbiter) { arbiter_ = arbiter; }

    // Adapter identity: kind + unit_id, because unit ids repeat across kinds.
    static std::uint64_t adapter_key(std::uint32_t kind, std::uint32_t unit_id) {
        return (static_cast<std::uint64_t>(kind) << 32) | unit_id;
    }

    // Starts one command context bound to one free PE group. Returns a failing
    // status when the request is unsupported or all groups/contexts are busy.
    ConfigCheck begin_command(
        const PanelEngineRequest& request,
        std::uint64_t adapter_key,
        std::uint64_t cycle);

    void tick(std::uint64_t cycle);

    bool poll_response(std::uint64_t adapter_key, PanelEngineResponse& out);
    bool unit_busy(std::uint64_t adapter_key) const;

    // Directed cancellation: the context stops generating, cancels queued
    // tasks and marks running tasks to skip writeback. After this call the
    // engine performs no further lease access for that transaction.
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle);

    void reset();

    // Diagnostics.
    const std::vector<EngineTaskRecord>& task_records() const { return task_records_; }
    std::uint32_t group_count() const { return config_.compute_group_count; }
    bool group_busy(std::uint32_t group_id) const;
    std::uint32_t group_of_command(std::uint32_t command_id) const;
    const std::vector<std::uint32_t>& command_ids_by_group() const {
        return group_commands_;
    }
    const std::vector<MultiModePe>& pes() const { return pes_; }
    const TileDispatcher& dispatcher() const { return dispatcher_; }
    TileDispatcher& dispatcher() { return dispatcher_; }
    std::uint64_t cycle() const { return cycle_; }
    const std::vector<EngineGemmLog>& gemm_log() const { return gemm_log_; }
    bool control_fault() const { return control_fault_; }
    const std::string& control_fault_reason() const { return control_fault_reason_; }

private:
    // Per-front-buffer-slot lease binding for the tile data path.
    struct SlotBinding {
        hw::dataplane::BufferManager* buffers{nullptr};
        std::uint64_t lease_id{0};
        std::uint64_t ready_cycle{0};
        std::uint64_t base_offset{0};
        std::uint32_t row_stride_bytes{0};
        bool valid{false};
    };

    class LeaseTileAccess : public TileAccess {
    public:
        void bind(std::uint32_t slot, const SlotBinding& binding);
        void unbind(std::uint32_t slot);
        bool binding_for(std::uint32_t slot, SlotBinding* out) const;
        bool read_operand_segment(
            const TileOperand& operand,
            std::uint32_t row,
            std::uint32_t col_begin,
            std::uint32_t col_count,
            float* out_row,
            std::uint64_t cycle) override;
        bool write_result(
            const TileTask& task,
            const float* data,
            std::uint64_t cycle) override;

    private:
        std::map<std::uint32_t, SlotBinding> bindings_;
    };

    class LeaseFrontReader : public FrontReader {
    public:
        explicit LeaseFrontReader(LeaseTileAccess& access) : access_(access) {}
        ~LeaseFrontReader() override = default;
        void configure(
            std::uint32_t slot,
            std::uint64_t base_offset,
            std::uint32_t row_stride_bytes,
            std::uint32_t total_dim);
        bool read_front(
            std::uint64_t byte_offset,
            std::uint32_t length_bytes,
            std::uint8_t* out,
            std::uint64_t cycle) override;

    private:
        LeaseTileAccess& access_;
        std::uint32_t slot_{0};
        std::uint64_t base_offset_{0};
        std::uint32_t row_stride_bytes_{0};
        std::uint32_t total_dim_{0};
        bool configured_{false};
    };

    struct Context {
        bool active{false};
        bool aborting{false};
        bool response_ready{false};
        PanelEngineRequest request;
        std::unique_ptr<IComputeController> controller;
        std::unique_ptr<LeaseFrontReader> reader;
        PanelEngineResponse response;
        std::uint64_t adapter_key{0};
        std::uint32_t group_id{0};
        std::uint64_t bytes_read{0};
        std::uint64_t bytes_written{0};
    };

    ConfigCheck build_context(Context& context, std::uint64_t cycle);
    void erase_slot_versions(std::uint32_t slot);
    void route_pe_responses(std::uint64_t cycle);
    void handle_cancel_requests(std::uint64_t cycle);
    void service_contexts(std::uint64_t cycle, hw::dataplane::OnChipArbiter& arbiter);
    void finalize_contexts(std::uint64_t cycle);
    void release_context(Context& context);
    Context* find_context(std::uint32_t command_id);
    Context* find_context_by_txn(std::uint64_t txn_id);
    Context* find_context_by_adapter(std::uint64_t adapter_key);

    PanelLuConfig config_;
    bool configured_{false};
    std::uint64_t cycle_{0};
    std::uint64_t epoch_{0};

    std::map<std::uint64_t, std::uint32_t> versions_;
    std::uint64_t next_task_id_{1};
    std::vector<MultiModePe> pes_;
    LeaseTileAccess access_;
    WritebackReservation writeback_reservation_;
    TileDispatcher dispatcher_;
    hw::dataplane::OnChipArbiter fallback_arbiter_;
    hw::dataplane::OnChipArbiter* arbiter_{nullptr};
    std::vector<Context> contexts_;
    std::vector<bool> group_busy_;
    std::vector<std::uint32_t> group_commands_;
    std::size_t rotation_{0};
    hw::dataplane::BufferManager* buffers_{nullptr};

    std::vector<EngineTaskRecord> task_records_;
    std::vector<EngineGemmLog> gemm_log_;
    std::uint64_t tasks_committed_{0};
    std::uint64_t tasks_cancelled_{0};
    bool control_fault_{false};
    std::string control_fault_reason_;
};

}  // namespace hw::panel_lu
