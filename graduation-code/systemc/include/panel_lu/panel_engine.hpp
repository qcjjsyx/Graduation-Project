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

#include "panel_lu/panel_controller.hpp"
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
    bool needs_pvector_write{false};
    std::uint32_t pvector_dim{0};
    std::uint64_t bytes_read{0};
    std::uint64_t bytes_written{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint32_t tasks_total{0};
    std::uint32_t tasks_cancelled{0};
};

// Diagnostic record of one internal tile task (tests/trace only).
struct EngineTaskRecord {
    TileTask task;
    StatusCode status{StatusCode::Ok};
    std::string reason;
    std::uint32_t failed_column{0};
    std::uint32_t pe_id{hw::command::NONE};
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
};

class PanelEngine : public TileVersionStore {
public:
    explicit PanelEngine(PanelLuConfig config);

    // TileVersionStore (keys are (slot << 32) | (tile_index + 1)).
    std::uint32_t version(std::uint64_t key) const override;
    bool commit(std::uint64_t key, std::uint32_t expected_version) override;

    // Validates the resource configuration and allocates the PE pool. Real
    // front data is never materialized here.
    ConfigCheck configure();

    // Binds the scratchpad manager used by the real lease data path.
    void set_buffers(hw::dataplane::BufferManager* buffers) { buffers_ = buffers; }

    // Starts one command context. Returns a failing status when the request is
    // unsupported (policy/tile_size/dimensions) or all contexts are busy.
    ConfigCheck begin_command(
        const PanelEngineRequest& request,
        std::uint32_t adapter_unit_id,
        std::uint64_t cycle);

    void tick(std::uint64_t cycle);

    bool poll_response(std::uint32_t adapter_unit_id, PanelEngineResponse& out);
    bool unit_busy(std::uint32_t adapter_unit_id) const;

    // Directed cancellation: the context stops generating, cancels queued
    // tasks and marks running tasks to skip writeback. After this call the
    // engine performs no further lease access for that transaction.
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle);

    void reset();

    // Diagnostics.
    const std::vector<EngineTaskRecord>& task_records() const { return task_records_; }
    const std::vector<MultiModePe>& pes() const { return pes_; }
    const TileDispatcher& dispatcher() const { return dispatcher_; }
    TileDispatcher& dispatcher() { return dispatcher_; }
    std::uint64_t cycle() const { return cycle_; }
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
        std::unique_ptr<PanelController> controller;
        std::unique_ptr<LeaseFrontReader> reader;
        PanelEngineResponse response;
        std::uint32_t adapter_unit_id{0};
        std::uint64_t bytes_read{0};
        std::uint64_t bytes_written{0};
    };

    ConfigCheck build_context(Context& context, std::uint64_t cycle);
    void erase_slot_versions(std::uint32_t slot);
    void route_pe_responses(std::uint64_t cycle);
    void handle_cancel_requests(std::uint64_t cycle);
    void service_contexts(std::uint64_t cycle);
    void finalize_contexts(std::uint64_t cycle);
    void release_context(Context& context);
    Context* find_context(std::uint32_t command_id);
    Context* find_context_by_txn(std::uint64_t txn_id);

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
    PortBudget read_budget_;
    PortBudget write_budget_;
    std::vector<Context> contexts_;
    std::size_t rotation_{0};
    hw::dataplane::BufferManager* buffers_{nullptr};

    std::vector<EngineTaskRecord> task_records_;
    std::uint64_t tasks_committed_{0};
    std::uint64_t tasks_cancelled_{0};
    bool control_fault_{false};
    std::string control_fault_reason_;
};

}  // namespace hw::panel_lu
