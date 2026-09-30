#pragma once

// T06 tile dispatcher + local scoreboard. The dispatcher is the single point
// where Panel/TRSM/GEMM tile tasks from all resident commands compete for the
// shared PE pool. It provides:
//   - a bounded FIFO (backpressure instead of unbounded queues),
//   - dependency/version readiness (tasks become dispatchable only after all
//     dependencies committed and their operand versions are visible),
//   - per-target-tile ordering (no later writer may pass an earlier writer),
//   - round-robin fairness between commands/fronts,
//   - directed cancellation for one command without touching another.

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "panel_lu/pe_model.hpp"
#include "panel_lu/task_types.hpp"
#include "panel_lu/tile_store.hpp"

namespace hw::panel_lu {

struct DispatchRecord {
    std::uint64_t enqueue_cycle{0};
    std::uint64_t dispatch_cycle{0};
    std::uint64_t stall_dependency_cycles{0};
    std::uint64_t stall_resource_cycles{0};
};

class TileDispatcher {
public:
    TileDispatcher(
        const PanelLuConfig& config,
        std::vector<MultiModePe>* pes,
        TileVersionStore* versions);

    // Returns false when the bounded queue is full (stall) or the task is
    // invalid/duplicated (permanent error, explicit reason).
    bool submit(const TileTask& task, std::uint64_t cycle, std::string* reason);

    void tick(std::uint64_t cycle);

    // Commit callback for a task whose result became visible. Commits the tile
    // version and keeps the commit record until release_command(). Returns
    // false on a stale version (only possible if a control invariant was
    // violated).
    bool on_task_committed(
        std::uint64_t task_id,
        std::uint32_t command_id,
        std::uint64_t version_key,
        std::uint32_t expected_version);

    // Terminal callback for failed/aborted tasks: releases writers, no commit.
    void on_task_terminal(std::uint64_t task_id);

    // Cancels all queued tasks of one command (directed abort). Returns the
    // cancelled tasks for tracing.
    std::vector<TileTask> cancel_command(std::uint32_t command_id);

    // Releases all dependency/record bookkeeping of a finished command. Must be
    // called when the command reaches its terminal response so that resident
    // slots can be reused without leaking or aliasing task ids.
    void release_command(std::uint32_t command_id);

    bool meta(std::uint64_t task_id, DispatchRecord* out) const;

    void reset();

    std::uint32_t queue_depth() const { return static_cast<std::uint32_t>(queue_.size()); }
    std::uint32_t high_watermark() const { return high_watermark_; }
    std::uint64_t queue_full_stall_cycles() const { return queue_full_stall_cycles_; }
    std::uint64_t dispatched_count() const { return dispatched_; }
    bool empty() const { return queue_.empty(); }
    bool deps_ready(const TileTask& task) const;

    // Bookkeeping introspection: after all commands of a run have terminated,
    // a correctly recycled dispatcher has no tracked tasks left.
    std::size_t known_count() const { return known_.size(); }
    std::size_t committed_count() const { return committed_.size(); }
    std::size_t record_count() const { return records_.size(); }

private:
    struct Queued {
        TileTask task;
        std::uint64_t enqueue_cycle{0};
        std::uint64_t stall_dependency_cycles{0};
        std::uint64_t stall_resource_cycles{0};
    };

    bool dispatchable(const TileTask& task) const;
    bool dispatch_one(Queued& item, std::uint64_t cycle);
    void forget_task(std::uint64_t task_id);

    PanelLuConfig config_;
    std::vector<MultiModePe>* pes_{nullptr};
    TileVersionStore* versions_{nullptr};

    std::deque<Queued> queue_;
    std::set<std::uint64_t> committed_;
    std::set<std::uint64_t> known_;
    std::map<std::uint64_t, std::set<std::uint64_t>> uncommitted_writers_;
    std::unordered_map<std::uint64_t, DispatchRecord> records_;
    // Committed task ids per command; released when the command terminates.
    std::map<std::uint32_t, std::set<std::uint64_t>> committed_by_command_;
    std::vector<std::uint32_t> command_order_;
    std::size_t rr_cursor_{0};
    std::size_t pe_cursor_{0};

    std::uint32_t high_watermark_{0};
    std::uint64_t queue_full_stall_cycles_{0};
    std::uint64_t dispatched_{0};
};

}  // namespace hw::panel_lu
