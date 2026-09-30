#pragma once

// Panel controller (per command). One controller owns exactly one PANEL_LU
// command: it runs the post-assembly full-front scan (finite check + exact
// maxabs reduction), generates the internal LU_DIAG / TRSM_LEFT / TRSM_RIGHT /
// GEMM_UPDATE tile tasks for A11 with a per-step barrier, tracks dependencies
// and outstanding tasks, drives the fixed-scale actual-pivot checks reported
// by LU_DIAG, and publishes one command response.
//
// The controller is data-path agnostic: the front buffer is described by a
// PanelFrontLayout (per-tile synthetic slots or one strided real buffer), tile
// bytes are read through a FrontReader (scan) and TileAccess (PE operands),
// and committed versions live in a TileVersionStore shared with the
// dispatcher. The controller never advances time itself; its owner calls
// tick() once per cycle. It does not repeat top-level TRSM/GEMM work
// (A12/A21/A22 stay with the top-level commands) and it never reads a golden
// object.

#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "panel_lu/task_types.hpp"
#include "panel_lu/tile_dispatcher.hpp"
#include "panel_lu/tile_store.hpp"

namespace hw::panel_lu {

// Synthetic workload injection for prototype runs only.
struct SyntheticInject {
    enum class Kind : std::uint8_t {
        None = 0,
        ZeroDiagonal = 1,
        TinyDiagonal = 2,
        SingularPair = 3,
        NanEntry = 4,
        InfEntry = 5,
    };
    Kind kind{Kind::None};
    std::uint32_t row{0};
    std::uint32_t col{0};
    float value{0.0f};
};

struct SyntheticFrontSpec {
    float diagonal{3.0f};
    float off_diagonal{0.2f};
    std::uint32_t seed{7};
    std::vector<SyntheticInject> injects;
};

struct PanelCommandRequest {
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::uint32_t front_dim{0};
    std::uint32_t pivot_dim{0};
    hw::command::PivotPolicy policy{hw::command::PivotPolicy::Static};
    std::uint32_t threshold_bits{0};
    std::uint32_t tile_size{16};
    SyntheticFrontSpec synthetic;
    std::string label;
};

// Physical placement of one command's front buffer.
struct PanelFrontLayout {
    enum class Kind : std::uint8_t {
        // Synthetic T06 layout: one TileStore slot per 16x16 tile, row stride
        // 64B, version key == slot index.
        PerTileSlots,
        // Real T07 layout: one strided buffer slot holds the whole n x n front.
        SingleBufferStrided,
    };
    Kind kind{Kind::SingleBufferStrided};
    hw::dataplane::BufferHandle handle{};
    std::uint32_t generation{0};
    std::uint64_t base_offset{0};
    std::uint32_t row_stride_bytes{0};
    std::uint32_t global_grid{0};
    std::uint32_t total_dim{0};  // front n (scan/reader extent)
};

// Byte source for the full-front scan. Implementations copy an assembled span
// of the front; returning false means "not this cycle" (backpressure), never a
// data error. Reads must not add timing of their own: the scan budget stays
// with the controller.
class FrontReader {
public:
    virtual ~FrontReader() = default;
    virtual bool read_front(
        std::uint64_t byte_offset,
        std::uint32_t length_bytes,
        std::uint8_t* out,
        std::uint64_t cycle) = 0;
};

struct PanelCommandResponse {
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    StatusCode status{StatusCode::Ok};
    std::string reason;
    std::uint32_t pivot_count{0};
    std::uint32_t failed_column{hw::command::NONE};
    std::uint32_t completed_pivots{0};
    bool pvector_written{false};
    std::uint32_t pvector_bytes{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint64_t scan_bytes{0};
    std::uint64_t scan_done_cycle{0};
    double front_pre_elimination_max_abs{0.0};
    double threshold_product{0.0};
    std::uint32_t tasks_generated{0};
    std::uint32_t tasks_cancelled{0};
    std::uint64_t stall_dispatcher_full{0};
    std::uint64_t stall_scan_port{0};
};

class PanelController {
public:
    enum class Phase {
        Idle = 0,
        Scan = 1,
        ScanReduce = 2,
        Diag = 3,
        DiagWait = 4,
        Trsm = 5,
        TrsmWait = 6,
        Gemm = 7,
        GemmWait = 8,
        PVector = 9,
        Done = 10,
        Failed = 11,
        Aborted = 12,
    };

    PanelController(
        const PanelLuConfig& config,
        TileDispatcher* dispatcher,
        TileVersionStore* versions,
        FrontReader* reader);

    ConfigCheck start(
        const PanelCommandRequest& request,
        std::uint64_t cycle,
        std::uint64_t epoch,
        const PanelFrontLayout& layout,
        std::uint64_t first_task_id);

    void tick(std::uint64_t cycle, PortBudget& scan_budget, PortBudget& write_budget);
    void on_task_response(const TileTaskResponse& response, std::uint64_t cycle);
    void on_tasks_cancelled(const std::vector<TileTask>& tasks);

    // Sets the abort flag and stops generating; outstanding tasks must still
    // terminate before the ABORTED response is published.
    void request_abort();

    // Returns true exactly once after a failure/abort requires the prototype to
    // cancel queued tasks and abort PE-owned tasks of this command.
    bool take_cancel_request();

    bool finished() const;
    bool poll_response(PanelCommandResponse& out);
    void reset();

    Phase phase() const { return phase_; }
    std::uint32_t step() const { return step_; }
    const PanelCommandRequest& request() const { return request_; }
    const std::vector<TileTask>& task_graph() const { return task_graph_; }
    const TileTask* find_task(std::uint64_t task_id) const;
    const std::vector<TileTask>& cancelled_tasks() const { return cancelled_tasks_; }
    double front_max_abs() const { return front_max_abs_; }
    double threshold_product() const { return threshold_product_; }
    std::uint64_t scan_done_cycle() const { return scan_done_cycle_; }
    std::uint64_t outstanding_count() const { return outstanding_.size(); }
    // Next task id after the last allocation (global, prototype-assigned base).
    std::uint64_t next_task_id() const { return next_task_id_; }

private:
    struct TileCoord {
        std::uint32_t row{0};
        std::uint32_t col{0};
    };

    std::uint32_t grid_extent(std::uint32_t dimension) const;
    std::uint32_t tile_width(std::uint32_t tile) const;
    std::uint64_t version_key(TileCoord coord) const;
    std::uint32_t operand_version(TileCoord coord) const;
    TileOperand operand_for(TileCoord coord, std::uint32_t version) const;
    bool submit_task(const TileTask& task, std::uint64_t cycle, std::string* reason);
    TileTask make_task(
        TileMode mode,
        TileCoord coord,
        TileDims dims,
        std::uint32_t step,
        const std::vector<std::uint64_t>& deps,
        std::uint64_t cycle);
    void produce_diag(std::uint64_t cycle);
    void produce_trsm(std::uint64_t cycle);
    void produce_gemm(std::uint64_t cycle);
    void finish_step(std::uint64_t cycle);
    void fail_command(
        StatusCode status,
        const std::string& reason,
        std::uint32_t failed_column,
        std::uint64_t cycle);
    void finalize_aborted(std::uint64_t cycle);
    void finalize_success(std::uint64_t cycle);
    void publish(std::uint64_t cycle);
    // Validates the threshold bits of the *given* request; must be called
    // before any command state is established.
    static StatusCode decode_threshold_bits(std::uint32_t bits, float* out);

    PanelLuConfig config_;
    TileDispatcher* dispatcher_{nullptr};
    TileVersionStore* versions_{nullptr};
    FrontReader* reader_{nullptr};
    PanelFrontLayout layout_{};
    std::uint32_t global_grid_{16};

    PanelCommandRequest request_;
    Phase phase_{Phase::Idle};
    std::uint64_t epoch_{0};
    std::uint32_t local_grid_{0};
    std::uint32_t num_steps_{0};
    std::uint32_t step_{0};

    std::uint64_t start_cycle_{0};
    std::uint64_t scan_bytes_total_{0};
    std::uint64_t scan_bytes_remaining_{0};
    std::uint64_t scan_elements_done_{0};
    std::uint64_t scan_done_cycle_{0};
    std::uint64_t scale_ready_at_{0};
    bool scan_non_finite_{false};
    double front_max_abs_{0.0};
    double threshold_product_{0.0};

    std::uint64_t pvector_bytes_remaining_{0};
    std::uint64_t pvector_done_at_{0};
    bool pvector_write_started_{false};
    bool pvector_written_{false};

    std::uint64_t next_task_id_{1};
    std::uint32_t sequence_{0};
    std::vector<TileTask> task_graph_;
    std::vector<TileTask> cancelled_tasks_;
    std::set<std::uint64_t> outstanding_;
    std::set<std::uint64_t> step_wait_;
    std::vector<std::uint64_t> diag_task_ids_;
    std::vector<std::vector<std::uint64_t>> left_task_ids_;
    std::vector<std::vector<std::uint64_t>> right_task_ids_;
    // GEMM tasks produced for the current step (retry-safe sentinel); cleared
    // at every step boundary because the same tile is updated once per step.
    std::set<std::pair<std::uint32_t, std::uint32_t>> gemm_done_;
    std::vector<std::uint64_t> last_diag_id_;

    bool abort_requested_{false};
    bool cancel_requested_{false};
    bool cancel_taken_{false};
    StatusCode failure_status_{StatusCode::Ok};
    std::string failure_reason_;
    std::uint32_t failed_column_{hw::command::NONE};
    std::uint32_t completed_pivots_{0};

    std::uint64_t stall_dispatcher_full_{0};
    std::uint64_t stall_scan_port_{0};
    std::optional<PanelCommandResponse> response_;
};

}  // namespace hw::panel_lu
