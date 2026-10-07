#pragma once

// Shared compute-command controller interface for the device compute engine.
// A controller owns exactly one accepted compute command (PANEL_LU, TRSM_LEFT,
// TRSM_RIGHT, later GEMM/solve) and generates its internal tile tasks under a
// step barrier. The engine drives it once per cycle with the device-wide
// on-chip arbiter and routes task responses / cancellations back to it.

#include <cstdint>
#include <string>
#include <vector>

#include "dataplane/on_chip_arbiter.hpp"
#include "panel_lu/task_types.hpp"

namespace hw::panel_lu {

struct ConfigCheck;
struct PanelCommandRequest;
struct PanelCommandResponse;
struct PanelFrontLayout;
struct TileTask;
struct TileTaskResponse;

class IComputeController {
public:
    virtual ~IComputeController() = default;

    virtual ConfigCheck start(
        const PanelCommandRequest& request,
        std::uint64_t cycle,
        std::uint64_t epoch,
        const PanelFrontLayout& layout,
        std::uint64_t first_task_id) = 0;

    virtual void tick(
        std::uint64_t cycle,
        hw::dataplane::OnChipArbiter& arbiter) = 0;

    virtual void on_task_response(const TileTaskResponse& response, std::uint64_t cycle) = 0;
    virtual void on_tasks_cancelled(const std::vector<TileTask>& tasks) = 0;

    virtual void request_abort() = 0;
    virtual bool take_cancel_request() = 0;

    virtual bool finished() const = 0;
    virtual bool poll_response(PanelCommandResponse& out) = 0;
    virtual void reset() = 0;

    virtual const TileTask* find_task(std::uint64_t task_id) const = 0;
    virtual std::size_t outstanding_count() const = 0;
    virtual std::uint64_t next_task_id() const = 0;
    virtual const PanelCommandRequest& request() const = 0;
    virtual std::vector<TileTask> task_graph_snapshot() const = 0;
};

}  // namespace hw::panel_lu
