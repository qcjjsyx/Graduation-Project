#pragma once

// T09 top-level TRSM controller (shared compute engine, one command context).
//
// TRSM_LEFT  : U12 = L11^-1 * A12   (A12 is pivot rows x update columns)
// TRSM_RIGHT : L21 = A21 * U11^-1   (A21 is update rows x pivot columns)
//
// Both are in-place on the resident front buffer and only touch their own
// block (A12 or A21). Blocks are 16x16 tiles anchored at the block origin
// (column p for A12, row p for A21), NOT at the full-front grid; tail tiles
// use the effective m/n/k and masks.
//
// Step barrier (s over pivot blocks, increasing):
//   LEFT : solve X(s,j) = Lss^-1 * B(s,j) for every update column j, then
//          update B(q,j) -= L(q,s) * X(s,j) for q > s; next step only after
//          every task of the step committed.
//   RIGHT: solve X(i,s) = B(i,s) * Uss^-1 for every update row i, then
//          update B(i,q) -= X(i,s) * U(s,q) for q > s.
//
// Numeric failures keep the frozen semantics: U diagonal zero ->
// STATIC_PIVOT_FAILURE, |diag| <= latched threshold product -> PIVOT_UNSTABLE,
// non-finite / reciprocal overflow / non-finite result -> NON_FINITE_VALUE.
// Failure positions are mapped to front coordinates (pivot column for
// right/diagonal failures, pivot row for left) and reported in the command
// response.

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

class TrsmController : public IComputeController {
public:
    TrsmController(
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

    // Diagnostics for tests/reports.
    ComputeKind compute_kind() const { return kind_; }
    std::uint32_t step() const { return step_; }
    std::uint32_t num_steps() const { return num_steps_; }
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
        TileMode mode{TileMode::GemmUpdate};
        std::uint32_t step{0};
        std::uint32_t row_tile{0};
        std::uint32_t col_tile{0};
    };

    std::uint32_t axis_extent(std::uint32_t tiles, std::uint32_t tile) const;
    TileOperand front_grid_operand(
        std::uint32_t row_tile,
        std::uint32_t col_tile,
        std::uint32_t version) const;
    TileOperand a12_operand(
        std::uint32_t row_tile,
        std::uint32_t col_tile,
        std::uint32_t version) const;
    TileOperand a21_operand(
        std::uint32_t row_tile,
        std::uint32_t col_tile,
        std::uint32_t version) const;
    std::uint64_t front_grid_key(std::uint32_t row_tile, std::uint32_t col_tile) const;
    std::uint64_t a12_key(std::uint32_t row_tile, std::uint32_t col_tile) const;
    std::uint64_t a21_key(std::uint32_t row_tile, std::uint32_t col_tile) const;
    std::uint32_t version_of(std::uint64_t key) const;

    bool submit_task(const TileTask& task, std::uint64_t cycle, std::string* reason);
    void reset_step_bookkeeping();
    void generate_step(std::uint64_t cycle);
    void fail_command(
        StatusCode status,
        const std::string& reason,
        std::uint32_t front_coordinate,
        std::uint64_t cycle);
    void publish(std::uint64_t cycle);
    std::uint32_t map_failure(const TaskMeta& meta, std::uint32_t local_column) const;

    PanelLuConfig config_;
    TileDispatcher* dispatcher_{nullptr};
    TileVersionStore* versions_{nullptr};
    PanelFrontLayout layout_{};

    PanelCommandRequest request_;
    ComputeKind kind_{ComputeKind::TrsmLeft};
    Phase phase_{Phase::Idle};
    std::uint64_t epoch_{0};
    std::uint32_t update_dim_{0};   // n - p
    std::uint32_t pivot_tiles_{0};  // ceil(p/16)
    std::uint32_t update_tiles_{0}; // ceil((n-p)/16)
    std::uint32_t num_steps_{0};    // pivot tiles
    std::uint32_t step_{0};

    // Step bookkeeping.
    std::uint32_t solve_targets_{0};                  // solve tiles this step
    bool step_ready_{false};
    std::vector<std::uint64_t> solve_ids_;            // per target of the step
    std::vector<bool> solve_generated_;
    std::vector<bool> solve_committed_;
    std::vector<std::vector<std::uint64_t>> update_ids_;  // [target][q]
    std::vector<std::vector<bool>> update_generated_;
    std::uint32_t solve_committed_count_{0};
    std::uint32_t update_generated_count_{0};
    std::uint32_t update_committed_count_{0};
    std::uint32_t update_expected_count_{0};

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
};

}  // namespace hw::panel_lu
