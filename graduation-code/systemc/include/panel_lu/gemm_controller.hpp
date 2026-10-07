#pragma once

// T10 top-level GEMM_SCHUR controller (shared compute engine, one command
// context).
//
//   A22 := A22 - L21 * U12        (u = n - p rows/cols of the update block)
//
// K is split into 16-element blocks s = 0 .. ceil(p/16)-1 and the controller
// uses a K-step barrier: within one s every output tile C(i,j) may run in
// parallel, and step s+1 is only generated after every task of step s truly
// wrote back and committed its tile version. Same-target read-modify-write of
// a C tile never overlaps another step; generation is bounded by the finite
// dispatcher queue (a full queue preserves progress and retries).
//
// Address blocks are anchored at their block origins with the front base
// applied exactly once:
//   A21 origin (row p, col 0), A12 origin (row 0, col p), A22 origin (p, p).
// Effective tile dimensions are m = min(16, u-16i), n = min(16, u-16j),
// k = min(16, p-16s), so p = 23 and u = 19 are handled correctly (8 tasks,
// 2 K steps) without a full-front tile grid.
//
// The A22 view has its own version-key space so its per-step versions can
// never alias A11/A12/A21 tiles. The backend is either the formal FP32 update
// primitive or the local INT32 PrecisionAdapter path selected by the command;
// both go through the shared PE/dispatcher/timing path (no side channel).

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "dataplane/on_chip_arbiter.hpp"
#include "panel_lu/compute_controller.hpp"
#include "panel_lu/panel_controller.hpp"
#include "panel_lu/task_types.hpp"
#include "panel_lu/tile_dispatcher.hpp"
#include "panel_lu/tile_store.hpp"

namespace hw::panel_lu {

class GemmController : public IComputeController {
public:
    GemmController(
        const PanelLuConfig& config,
        TileDispatcher* dispatcher,
        TileVersionStore* versions);

    ConfigCheck start(
        const PanelCommandRequest& request,
        std::uint64_t cycle,
        std::uint64_t epoch,
        const PanelFrontLayout& layout,
        std::uint64_t first_task_id) override;

    void tick(
        std::uint64_t cycle,
        hw::dataplane::OnChipArbiter& arbiter) override;

    void on_task_response(const TileTaskResponse& response, std::uint64_t cycle) override;
    void on_tasks_cancelled(const std::vector<TileTask>& tasks) override;
    void request_abort() override;
    bool take_cancel_request() override;
    bool finished() const override;
    bool poll_response(PanelCommandResponse& out) override;
    void reset() override;

    const TileTask* find_task(std::uint64_t task_id) const override;
    std::size_t outstanding_count() const override { return outstanding_.size(); }
    std::uint64_t next_task_id() const override { return next_task_id_; }
    const PanelCommandRequest& request() const override { return request_; }
    std::vector<TileTask> task_graph_snapshot() const override { return task_graph_; }

    // Unified GemmReq/GemmRsp records produced by this command (the upper
    // layer boundary; the shared PE pool still moves TileTask internally).
    const std::vector<GemmReq>& gemm_requests() const { return gemm_requests_; }
    const std::vector<GemmRsp>& gemm_responses() const { return gemm_responses_; }

    // Diagnostics for tests/reports.
    ComputeKind compute_kind() const { return kind_; }
    std::uint32_t step() const { return step_; }
    std::uint32_t num_steps() const { return num_steps_; }
    std::uint32_t tasks_per_step() const { return output_tiles_ * output_tiles_; }
    std::uint32_t failed_front_coordinate() const { return failed_coordinate_; }

private:
    enum class Phase {
        Idle = 0,
        StepGenerate = 1,
        StepWait = 2,
        Done = 3,
        Failed = 4,
        Aborted = 5,
    };

    struct TaskMeta {
        std::uint32_t step{0};
        std::uint32_t row_tile{0};
        std::uint32_t col_tile{0};
    };

    std::uint32_t axis_extent(std::uint32_t dimension, std::uint32_t tile) const;
    std::uint64_t a21_key(std::uint32_t row_tile, std::uint32_t step) const;
    std::uint64_t a12_key(std::uint32_t step, std::uint32_t col_tile) const;
    std::uint64_t a22_key(std::uint32_t row_tile, std::uint32_t col_tile) const;
    TileOperand a21_operand(
        std::uint32_t row_tile,
        std::uint32_t step,
        std::uint32_t version) const;
    TileOperand a12_operand(
        std::uint32_t step,
        std::uint32_t col_tile,
        std::uint32_t version) const;
    TileOperand a22_operand(
        std::uint32_t row_tile,
        std::uint32_t col_tile,
        std::uint32_t version) const;
    std::uint32_t version_of(std::uint64_t key) const;

    bool submit_task(const TileTask& task, std::uint64_t cycle, std::string* reason);
    void generate_step(std::uint64_t cycle);
    void fail_command(
        StatusCode status,
        const std::string& reason,
        std::uint32_t front_coordinate,
        std::uint64_t cycle);
    void publish(std::uint64_t cycle);
    std::uint32_t map_failure(const TaskMeta& meta, const TileTaskResponse& response) const;

    PanelLuConfig config_;
    TileDispatcher* dispatcher_{nullptr};
    TileVersionStore* versions_{nullptr};
    PanelFrontLayout layout_{};

    PanelCommandRequest request_;
    ComputeKind kind_{ComputeKind::GemmSchur};
    Phase phase_{Phase::Idle};
    std::uint64_t epoch_{0};
    std::uint32_t update_dim_{0};    // u = n - p
    std::uint32_t k_block_size_{16}; // INT32 uses precision.k_block_max
    std::uint32_t pivot_tiles_{0};   // ceil(p/k_block_size)
    std::uint32_t output_tiles_{0};  // ceil(u/16)
    std::uint32_t num_steps_{0};     // == pivot_tiles_
    std::uint32_t step_{0};

    // Step barrier bookkeeping.
    bool step_ready_{false};
    std::vector<std::vector<bool>> generated_;   // [row_tile][col_tile]
    std::vector<std::vector<bool>> committed_;   // [row_tile][col_tile]
    std::uint32_t expected_per_step_{0};
    std::uint32_t generated_count_{0};
    std::uint32_t committed_count_{0};

    std::uint64_t start_cycle_{0};
    std::uint64_t next_task_id_{1};
    std::uint32_t sequence_{0};
    std::vector<TileTask> task_graph_;
    std::vector<TileTask> cancelled_tasks_;
    std::set<std::uint64_t> outstanding_;
    std::map<std::uint64_t, TaskMeta> task_meta_;

    bool abort_requested_{false};
    bool cancel_requested_{false};
    bool cancel_taken_{false};
    StatusCode failure_status_{StatusCode::Ok};
    std::string failure_reason_;
    std::uint32_t failed_coordinate_{hw::command::NONE};
    std::optional<PanelCommandResponse> response_;
    std::vector<GemmReq> gemm_requests_;
    std::vector<GemmRsp> gemm_responses_;

    // Aggregated INT32 backend statistics (per-tile AUTO scales may differ, so
    // the representative exponents are those of the first completed task).
    std::uint64_t saturation_count_{0};
    std::uint64_t overflow_count_{0};
    std::uint64_t local_conversion_bytes_{0};
    std::uint64_t scale_reduction_cycles_{0};
    std::uint64_t quantize_cycles_{0};
    std::uint64_t integer_compute_cycles_{0};
    std::uint64_t dequantize_cycles_{0};
    std::int32_t exponent_a_{0};
    std::int32_t exponent_b_{0};
    bool have_exponents_{false};
};

}  // namespace hw::panel_lu
