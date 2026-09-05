#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <systemc>

#include "artifact.hpp"
#include "atu.hpp"
#include "hpu.hpp"
#include "model_types.hpp"
#include "numeric_kernels.hpp"
#include "system_memory.hpp"

namespace hw {

inline bool mode_has_fp64(NumericMode mode) {
    return mode == NumericMode::Fp64 || mode == NumericMode::Both;
}

inline bool mode_has_fixed(NumericMode mode) {
    return mode == NumericMode::Fixed || mode == NumericMode::Both;
}

inline std::string event_text(
    const char* event,
    std::uint16_t node_id,
    unsigned buffer_id = 0) {
    std::ostringstream stream;
    stream << event << ",node=" << node_id << ",buffer=" << buffer_id;
    return stream.str();
}

struct CycleCounter : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};

    CycleCounter(sc_core::sc_module_name name, SimulationStats& stats)
        : sc_core::sc_module(name), stats_(stats) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

private:
    SimulationStats& stats_;

    void tick() {
        if (!rst_n.read()) {
            stats_.cycle = 0;
        } else {
            ++stats_.cycle;
        }
    }
};

struct TaskFetch : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};
    sc_core::sc_fifo_out<NodeTask> task_out{"task_out"};
    sc_core::sc_out<bool> registration_complete{"registration_complete"};

    TaskFetch(
        sc_core::sc_module_name name,
        const std::vector<NodeTask>& tasks,
        DdrMemory& ddr,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          tasks_(tasks),
          ddr_(ddr),
          stats_(stats) {
        SC_THREAD(run);
        sensitive << clk.pos();
    }

private:
    const std::vector<NodeTask>& tasks_;
    DdrMemory& ddr_;
    SimulationStats& stats_;

    void wait_cycles(std::uint64_t cycles) {
        for (std::uint64_t i = 0; i < cycles; ++i) wait();
    }

    void run() {
        registration_complete.write(false);
        wait();
        while (!rst_n.read()) wait();
        for (const auto& task : tasks_) {
            const auto delay = ddr_.account_read(NodeTaskCodec::RECORD_SIZE);
            wait_cycles(delay);
            task_out.write(task);
            stats_.timeline(event_text("task_fetched", task.node_id));
        }
        registration_complete.write(true);
    }
};

struct DependencyScoreboard : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};
    sc_core::sc_in<bool> registration_complete{"registration_complete"};
    sc_core::sc_fifo_in<NodeTask> task_in{"task_in"};
    sc_core::sc_fifo_in<NodeCommit> commit_in{"commit_in"};
    sc_core::sc_fifo_out<NodeTask> ready_out{"ready_out"};

    DependencyScoreboard(
        sc_core::sc_module_name name,
        std::size_t expected_nodes,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          expected_nodes_(expected_nodes),
          stats_(stats) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

private:
    struct Entry {
        NodeTask task{};
        std::uint32_t remaining_children{0};
        bool issued{false};
        bool committed{false};
    };

    std::size_t expected_nodes_;
    SimulationStats& stats_;
    std::map<std::uint16_t, Entry> entries_{};
    std::deque<NodeTask> ready_queue_{};
    bool initial_ready_enqueued_{false};

    void fail(const std::string& reason) {
        stats_.control_failure = true;
        stats_.failure_reason = reason;
    }

    void tick() {
        if (!rst_n.read()) {
            entries_.clear();
            ready_queue_.clear();
            initial_ready_enqueued_ = false;
            return;
        }

        NodeTask task{};
        while (task_in.nb_read(task)) {
            if (entries_.count(task.node_id) != 0) {
                fail("scoreboard observed duplicate task descriptor");
                return;
            }
            entries_.emplace(task.node_id, Entry{
                task, task.children_count, false, false,
            });
            if (task.parent_id == ROOT_PARENT_ID) ++stats_.root_count;
        }

        if (registration_complete.read() && !initial_ready_enqueued_) {
            if (entries_.size() != expected_nodes_) {
                fail("task fetch completed before all descriptors were registered");
                return;
            }
            for (auto& [node_id, entry] : entries_) {
                (void)node_id;
                if (entry.remaining_children == 0) {
                    entry.issued = true;
                    ready_queue_.push_back(entry.task);
                }
            }
            initial_ready_enqueued_ = true;
        }

        NodeCommit commit{};
        while (commit_in.nb_read(commit)) {
            const auto it = entries_.find(commit.node_id);
            if (it == entries_.end() || it->second.committed) {
                fail("scoreboard observed unknown or duplicate completion");
                return;
            }
            it->second.committed = true;
            ++stats_.completed_nodes;
            ++stats_.commit_count[commit.node_id];
            stats_.commit_order.push_back(commit.node_id);
            stats_.commit_cycle[commit.node_id] = stats_.cycle;
            if (commit.parent_id != ROOT_PARENT_ID) {
                const auto parent = entries_.find(commit.parent_id);
                if (parent == entries_.end() ||
                    parent->second.remaining_children == 0) {
                    fail("scoreboard dependency counter underflow");
                    return;
                }
                --parent->second.remaining_children;
                if (parent->second.remaining_children == 0 &&
                    !parent->second.issued) {
                    parent->second.issued = true;
                    ready_queue_.push_back(parent->second.task);
                }
            }
        }

        if (!ready_queue_.empty() &&
            ready_out.nb_write(ready_queue_.front())) {
            const auto node_id = ready_queue_.front().node_id;
            stats_.nodes[node_id].ready_cycle = stats_.cycle;
            stats_.timeline(event_text("dependency_ready", node_id));
            ready_queue_.pop_front();
        }
    }
};

struct BufferManager : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};
    sc_core::sc_fifo_in<NodeTask> task_in{"task_in"};
    sc_core::sc_fifo_in<BufferRelease> release_in{"release_in"};
    sc_core::sc_fifo_out<WorkItem> work_out{"work_out"};

    BufferManager(
        sc_core::sc_module_name name,
        const ArtifactManifest& manifest,
        const ModelConfig& config,
        NumericMode mode,
        DdrMemory& ddr,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          manifest_(manifest),
          config_(config),
          mode_(mode),
          ddr_(ddr),
          stats_(stats),
          slots_(config.buffer_count) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

private:
    struct Slot {
        bool used{false};
        bool loading{false};
        WorkItem work{};
        std::uint64_t ready_cycle{0};
    };

    const ArtifactManifest& manifest_;
    const ModelConfig& config_;
    NumericMode mode_;
    DdrMemory& ddr_;
    SimulationStats& stats_;
    std::vector<Slot> slots_;
    std::deque<NodeTask> pending_{};

    std::optional<unsigned> free_slot() const {
        for (unsigned i = 0; i < slots_.size(); ++i) {
            if (!slots_[i].used) return i;
        }
        return std::nullopt;
    }

    std::uint64_t load_bytes(const NodeTask& task) const {
        const auto& node = manifest_.nodes.at(task.node_id);
        std::uint64_t bytes =
            node.front_q.size + node.front_e.size + node.map_table.size;
        for (const auto child : memory_children(task.node_id)) {
            const auto& child_node = manifest_.nodes.at(child);
            bytes += child_node.update_q.size + child_node.update_e.size;
        }
        return bytes;
    }

    std::vector<std::uint16_t> memory_children(
        std::uint16_t parent) const {
        std::vector<std::uint16_t> children;
        for (std::uint16_t node = 0; node < manifest_.parent.size(); ++node) {
            if (manifest_.parent[node] == parent) children.push_back(node);
        }
        return children;
    }

    void tick() {
        if (!rst_n.read()) {
            pending_.clear();
            for (auto& slot : slots_) slot = Slot{};
            return;
        }

        BufferRelease release{};
        while (release_in.nb_read(release)) {
            if (release.buffer_id >= slots_.size() ||
                !slots_[release.buffer_id].used ||
                slots_[release.buffer_id].work.task.node_id != release.node_id) {
                stats_.control_failure = true;
                stats_.failure_reason = "invalid or duplicate buffer release";
                return;
            }
            slots_[release.buffer_id] = Slot{};
            stats_.timeline(
                event_text("buffer_released", release.node_id, release.buffer_id));
        }

        NodeTask task{};
        while (task_in.nb_read(task)) pending_.push_back(task);

        for (auto& slot : slots_) {
            if (slot.used) ++stats_.buffer_busy_cycles;
            if (slot.loading && stats_.cycle >= slot.ready_cycle &&
                work_out.nb_write(slot.work)) {
                slot.loading = false;
                const auto node_id = slot.work.task.node_id;
                auto& node_stats = stats_.nodes[node_id];
                node_stats.load_wait_cycles =
                    stats_.cycle - node_stats.ready_cycle;
                stats_.timeline(
                    event_text("front_loaded", node_id, slot.work.buffer_id));
            }
        }

        if (!pending_.empty()) {
            const auto available = free_slot();
            if (!available) {
                ++stats_.buffer_wait_cycles;
                return;
            }
            const auto pending_task = pending_.front();
            const auto bytes_required =
                static_cast<std::uint64_t>(pending_task.total_dim) *
                pending_task.total_dim *
                (mode_ == NumericMode::Fixed ? 4u :
                 mode_ == NumericMode::Fp64 ? 8u : 12u);
            if (bytes_required > config_.buffer_capacity_bytes) {
                stats_.address_failure = true;
                stats_.failure_reason =
                    "front exceeds configured buffer capacity";
                return;
            }
            pending_.pop_front();
            auto& slot = slots_[*available];
            slot.used = true;
            slot.loading = true;
            slot.work = WorkItem{*available, pending_task};
            const auto delay = ddr_.account_read(load_bytes(pending_task));
            slot.ready_cycle = stats_.cycle + std::max<std::uint64_t>(delay, 1);
            stats_.timeline(
                event_text("buffer_allocated", pending_task.node_id, *available));
        }
    }
};

struct FrontAssembler : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};
    sc_core::sc_fifo_in<WorkItem> work_in{"work_in"};
    sc_core::sc_fifo_out<WorkItem> work_out{"work_out"};

    FrontAssembler(
        sc_core::sc_module_name name,
        SystemMemory& memory,
        const ModelConfig& config,
        NumericMode mode,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          memory_(memory),
          config_(config),
          mode_(mode),
          stats_(stats) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

private:
    SystemMemory& memory_;
    const ModelConfig& config_;
    NumericMode mode_;
    SimulationStats& stats_;
    std::optional<WorkItem> active_{};
    std::uint64_t finish_cycle_{0};
    bool assembled_{false};

    void fail(std::uint16_t node_id, const std::string& reason) {
        auto& node = memory_.at(node_id);
        node.status = NodeStatus::NumericFailure;
        node.failure_reason = reason;
        auto& perf = stats_.nodes[node_id];
        perf.status = NodeStatus::NumericFailure;
        perf.failure_reason = reason;
        stats_.numeric_failure = true;
        stats_.failure_reason = "node " + std::to_string(node_id) +
                                " assembly: " + reason;
    }

    void tick() {
        if (!rst_n.read()) {
            active_.reset();
            assembled_ = false;
            return;
        }

        if (!active_) {
            WorkItem item{};
            if (!work_in.nb_read(item)) return;
            active_ = item;
            assembled_ = false;
            const auto dim = static_cast<std::uint64_t>(item.task.total_dim);
            const auto child_work =
                static_cast<std::uint64_t>(item.task.children_count) * dim * dim;
            const auto cycles = 1 + ceil_div_u64(
                dim * dim + child_work,
                std::max<unsigned>(config_.panel_ops_per_cycle, 1));
            finish_cycle_ = stats_.cycle + cycles;
            auto& perf = stats_.nodes[item.task.node_id];
            perf.node_id = item.task.node_id;
            perf.total_dim = item.task.total_dim;
            perf.pivot_dim = item.task.pivot_dim;
            perf.status = NodeStatus::Running;
            perf.start_cycle = stats_.cycle;
            memory_.at(item.task.node_id).status = NodeStatus::Running;
            stats_.start_order.push_back(item.task.node_id);
            stats_.start_cycle[item.task.node_id] = stats_.cycle;
            ++stats_.execute_count[item.task.node_id];
            stats_.timeline(
                event_text("assembly_started", item.task.node_id, item.buffer_id));
        }

        if (!active_ || stats_.cycle < finish_cycle_) return;
        if (!assembled_) {
            try {
                if (mode_has_fp64(mode_)) {
                    memory_.assemble_fp64(active_->task.node_id);
                }
                if (mode_has_fixed(mode_)) {
                    memory_.assemble_fixed(active_->task.node_id);
                }
                auto& perf = stats_.nodes[active_->task.node_id];
                perf.assembly_end_cycle = stats_.cycle;
                perf.assembly_cycles = stats_.cycle - perf.start_cycle;
                perf.quant = memory_.at(active_->task.node_id).quant_stats;
                assembled_ = true;
            } catch (const std::exception& exception) {
                fail(active_->task.node_id, exception.what());
                active_.reset();
                return;
            }
        }
        if (work_out.nb_write(*active_)) {
            stats_.timeline(
                event_text("assembly_finished",
                           active_->task.node_id, active_->buffer_id));
            active_.reset();
            assembled_ = false;
        }
    }
};

struct KernelDispatcher : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};
    sc_core::sc_fifo_in<WorkItem> work_in{"work_in"};
    sc_core::sc_fifo_out<ComputeDone> result_out{"result_out"};

    sc_core::sc_out<bool> atu_init_identity{"atu_init_identity"};
    sc_core::sc_out<sc_dt::sc_uint<ROW_IDX_W>> atu_init_rows{"atu_init_rows"};
    sc_core::sc_in<bool> atu_init_done{"atu_init_done"};
    sc_core::sc_out<bool> atu_pivot_req_valid{"atu_pivot_req_valid"};
    sc_core::sc_out<sc_dt::sc_uint<ROW_IDX_W>> atu_pivot_row_i{"atu_pivot_row_i"};
    sc_core::sc_out<sc_dt::sc_uint<ROW_IDX_W>> atu_pivot_row_j{"atu_pivot_row_j"};
    sc_core::sc_in<bool> atu_pivot_req_ready{"atu_pivot_req_ready"};
    sc_core::sc_in<bool> atu_pivot_done{"atu_pivot_done"};

    sc_core::sc_out<bool> hpu_pivot_start{"hpu_pivot_start"};
    sc_core::sc_out<bool> hpu_in_valid{"hpu_in_valid"};
    sc_core::sc_in<bool> hpu_in_ready{"hpu_in_ready"};
    sc_core::sc_out<sc_dt::sc_int<HPU::DATA_W>> hpu_in_value{"hpu_in_value"};
    sc_core::sc_out<sc_dt::sc_uint<ROW_IDX_W>> hpu_in_row_logical{
        "hpu_in_row_logical"};
    sc_core::sc_out<bool> hpu_in_last{"hpu_in_last"};
    sc_core::sc_in<bool> hpu_pivot_valid{"hpu_pivot_valid"};
    sc_core::sc_out<bool> hpu_pivot_ready{"hpu_pivot_ready"};
    sc_core::sc_in<sc_dt::sc_uint<ROW_IDX_W>> hpu_pivot_row{"hpu_pivot_row"};
    sc_core::sc_in<sc_dt::sc_int<HPU::DATA_W>> hpu_pivot_value{
        "hpu_pivot_value"};
    sc_core::sc_in<bool> hpu_pivot_fail{"hpu_pivot_fail"};

    KernelDispatcher(
        sc_core::sc_module_name name,
        SystemMemory& memory,
        const ModelConfig& config,
        NumericMode mode,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          memory_(memory),
          config_(config),
          mode_(mode),
          stats_(stats) {
        SC_THREAD(run);
        sensitive << clk.pos();
    }

private:
    SystemMemory& memory_;
    const ModelConfig& config_;
    NumericMode mode_;
    SimulationStats& stats_;

    void defaults() {
        atu_init_identity.write(false);
        atu_init_rows.write(0);
        atu_pivot_req_valid.write(false);
        atu_pivot_row_i.write(0);
        atu_pivot_row_j.write(0);
        hpu_pivot_start.write(false);
        hpu_in_valid.write(false);
        hpu_in_value.write(0);
        hpu_in_row_logical.write(0);
        hpu_in_last.write(false);
        hpu_pivot_ready.write(true);
    }

    void tick_wait() {
        wait();
        wait(sc_core::SC_ZERO_TIME);
    }

    void initialize_atu(unsigned rows) {
        atu_init_rows.write(rows);
        atu_init_identity.write(true);
        tick_wait();
        atu_init_identity.write(false);
        do {
            tick_wait();
        } while (!atu_init_done.read());
    }

    void replay_pivot(
        const std::vector<PivotCandidate>& candidates,
        std::uint16_t expected_row,
        unsigned column) {
        hpu_pivot_start.write(true);
        tick_wait();
        hpu_pivot_start.write(false);
        tick_wait();
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            hpu_in_value.write(candidate.value);
            hpu_in_row_logical.write(candidate.row);
            hpu_in_last.write(index + 1 == candidates.size());
            hpu_in_valid.write(true);
            do {
                tick_wait();
            } while (!hpu_in_ready.read());
            hpu_in_valid.write(false);
            hpu_in_last.write(false);
        }
        do {
            tick_wait();
        } while (!hpu_pivot_valid.read());
        if (hpu_pivot_fail.read()) {
            throw NumericFailure("HPU reported an all-zero pivot column");
        }
        if (hpu_pivot_row.read().to_uint() != expected_row) {
            throw NumericFailure("HPU selection disagrees with fixed golden kernel");
        }
        if (expected_row != column) {
            atu_pivot_row_i.write(column);
            atu_pivot_row_j.write(expected_row);
            atu_pivot_req_valid.write(true);
            do {
                tick_wait();
            } while (!atu_pivot_req_ready.read());
            atu_pivot_req_valid.write(false);
            if (!atu_pivot_done.read()) tick_wait();
        }
    }

    void replay_rescue_swaps(
        const std::vector<std::uint16_t>& selected_rows) {
        for (unsigned column = 0;
             column < selected_rows.size(); ++column) {
            const auto selected = selected_rows[column];
            if (selected == column) continue;
            atu_pivot_row_i.write(column);
            atu_pivot_row_j.write(selected);
            atu_pivot_req_valid.write(true);
            do {
                tick_wait();
            } while (!atu_pivot_req_ready.read());
            atu_pivot_req_valid.write(false);
            if (!atu_pivot_done.read()) tick_wait();
        }
    }

    void record_failure(
        const WorkItem& work,
        const std::string& reason) {
        auto& node = memory_.at(work.task.node_id);
        node.status = NodeStatus::NumericFailure;
        node.failure_reason = reason;
        auto& perf = stats_.nodes[work.task.node_id];
        perf.status = NodeStatus::NumericFailure;
        perf.failure_reason = reason;
        perf.compute_end_cycle = stats_.cycle;
        perf.compute_cycles = stats_.cycle - perf.assembly_end_cycle;
        stats_.numeric_failure = true;
        stats_.failure_reason =
            "node " + std::to_string(work.task.node_id) +
            " compute: " + reason;
        result_out.write(ComputeDone{work, false, reason});
    }

    void run() {
        defaults();
        wait();
        while (!rst_n.read()) wait();

        while (true) {
            const auto work = work_in.read();
            try {
                auto& node = memory_.at(work.task.node_id);
                std::optional<Fp64Computation> fp_result;
                std::optional<FixedComputation> fixed_result;
                if (mode_has_fp64(mode_)) {
                    fp_result = factor_fp64_front(
                        node.assembled_fp64,
                        work.task.total_dim,
                        work.task.pivot_dim,
                        config_);
                }
                if (mode_has_fixed(mode_)) {
                    try {
                        fixed_result =
                            config_.bfp_tile_size == 0 ?
                            factor_fixed_front(
                                node.assembled_q,
                                work.task.total_dim,
                                work.task.pivot_dim,
                                node.assembled_exp,
                                config_) :
                            factor_fixed_front_tile_bfp(
                                node.assembled_q,
                                node.assembled_tile_exponents,
                                work.task.total_dim,
                                work.task.pivot_dim,
                                config_);
                    } catch (const PrecisionRescueRequired&) {
                        if (config_.fixed_rescue_mode != "fp64") throw;
                        auto rescue_config = config_;
                        rescue_config.pivot_rel_tol =
                            config_.rescue_pivot_rel_tol;
                        const auto rescue_front =
                            memory_.assemble_fixed_rescue_fp64(
                                work.task.node_id);
                        const auto rescue = factor_fp64_front(
                            rescue_front,
                            work.task.total_dim,
                            work.task.pivot_dim,
                            rescue_config);
                        fixed_result =
                            quantize_rescued_factor(
                                rescue,
                                node.task.total_dim,
                                node.task.pivot_dim,
                                config_);
                        node.fixed_rescue_fp64 = rescue.factor;
                    }
                    initialize_atu(work.task.pivot_dim);
                    if (fixed_result->precision_rescued) {
                        replay_rescue_swaps(
                            fixed_result->selected_rows);
                    } else {
                        for (unsigned column = 0;
                             column < fixed_result->candidates.size();
                             ++column) {
                            replay_pivot(
                                fixed_result->candidates[column],
                                fixed_result->selected_rows[column],
                                column);
                        }
                    }
                }

                if (fp_result) node.fp64 = std::move(fp_result->factor);
                if (fixed_result) {
                    node.fixed = std::move(fixed_result->factor);
                    if (fixed_result->precision_rescued) {
                        node.quant_stats
                            .rescue_quantization_saturation_count +=
                            fixed_result->matrix_overflow_count;
                    } else {
                        node.quant_stats.matrix_overflow_count +=
                            fixed_result->matrix_overflow_count;
                    }
                    node.quant_stats.workspace_renormalize_count +=
                        fixed_result->workspace_renormalize_count;
                    node.quant_stats.small_pivot_count +=
                        fixed_result->small_pivot_count;
                    node.quant_stats.min_pivot_ratio =
                        fixed_result->min_pivot_ratio;
                    node.quant_stats.max_growth_ratio =
                        fixed_result->max_growth_ratio;
                    if (fixed_result->precision_rescued) {
                        ++node.quant_stats.precision_rescue_count;
                    }
                }

                auto operations = build_operation_plan(
                    work.task.node_id,
                    work.task.total_dim,
                    work.task.pivot_dim,
                    config_,
                    stats_.cycle);
                const auto end_cycle =
                    schedule_operations(operations, config_, stats_.cycle);
                while (stats_.cycle < end_cycle) tick_wait();
                stats_.operations.insert(
                    stats_.operations.end(),
                    operations.begin(), operations.end());
                if (fixed_result &&
                    fixed_result->precision_rescued) {
                    const auto rescue_begin = stats_.cycle;
                    const auto work_count =
                        static_cast<std::uint64_t>(
                            work.task.total_dim) *
                        work.task.total_dim *
                        work.task.pivot_dim;
                    const auto rescue_cycles =
                        config_.precision_rescue_startup +
                        ceil_div_u64(
                            work_count,
                            config_.precision_rescue_macs_per_cycle);
                    while (stats_.cycle <
                           rescue_begin + rescue_cycles) {
                        tick_wait();
                    }
                    stats_.operations.push_back({
                        work.task.node_id,
                        OpType::PrecisionRescue,
                        0, 0, 0,
                        work.task.total_dim,
                        work.task.total_dim,
                        work.task.pivot_dim,
                        rescue_begin,
                        rescue_begin,
                        stats_.cycle,
                    });
                    stats_.timeline(
                        event_text(
                            "precision_rescue",
                            work.task.node_id,
                            work.buffer_id));
                }

                auto& perf = stats_.nodes[work.task.node_id];
                perf.compute_end_cycle = stats_.cycle;
                perf.compute_cycles =
                    stats_.cycle - perf.assembly_end_cycle;
                perf.quant = node.quant_stats;
                perf.fixed_exponent = node.fixed.u_exponent;
                perf.fixed_update_exponent =
                    node.fixed.update_exponent;
                const auto record_exponent_range =
                    [](const std::vector<std::int16_t>& values,
                       std::int16_t fallback,
                       std::uint32_t& count,
                       std::int16_t& minimum,
                       std::int16_t& maximum) {
                        if (values.empty()) {
                            count = 1;
                            minimum = fallback;
                            maximum = fallback;
                            return;
                        }
                        count = static_cast<std::uint32_t>(values.size());
                        const auto range = std::minmax_element(
                            values.begin(), values.end());
                        minimum = *range.first;
                        maximum = *range.second;
                    };
                record_exponent_range(
                    node.assembled_tile_exponents,
                    node.assembled_exp,
                    perf.assembled_tile_count,
                    perf.assembled_exp_min,
                    perf.assembled_exp_max);
                record_exponent_range(
                    node.fixed.u_tile_exponents,
                    node.fixed.u_exponent,
                    perf.u_tile_count,
                    perf.u_exp_min,
                    perf.u_exp_max);
                if (node.fixed.update.empty()) {
                    perf.update_tile_count = 0;
                    perf.update_exp_min =
                        node.fixed.update_exponent;
                    perf.update_exp_max =
                        node.fixed.update_exponent;
                } else {
                    record_exponent_range(
                        node.fixed.update_tile_exponents,
                        node.fixed.update_exponent,
                        perf.update_tile_count,
                        perf.update_exp_min,
                        perf.update_exp_max);
                }
                perf.pivot_swaps_fp64 =
                    fp_result ? fp_result->swap_count : 0;
                perf.pivot_swaps_fixed =
                    fixed_result ? fixed_result->swap_count : 0;
                stats_.timeline(
                    event_text("compute_finished",
                               work.task.node_id, work.buffer_id));
                result_out.write(ComputeDone{work, true, {}});
            } catch (const std::exception& exception) {
                record_failure(work, exception.what());
            }
        }
    }
};

struct ResultWriter : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};
    sc_core::sc_fifo_in<ComputeDone> result_in{"result_in"};
    sc_core::sc_fifo_out<NodeCommit> commit_out{"commit_out"};
    sc_core::sc_fifo_out<BufferRelease> release_out{"release_out"};

    ResultWriter(
        sc_core::sc_module_name name,
        const ArtifactManifest& manifest,
        SystemMemory& memory,
        DdrMemory& ddr,
        const ModelConfig& config,
        NumericMode mode,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          manifest_(manifest),
          memory_(memory),
          ddr_(ddr),
          config_(config),
          mode_(mode),
          stats_(stats) {
        SC_THREAD(run);
        sensitive << clk.pos();
    }

private:
    const ArtifactManifest& manifest_;
    SystemMemory& memory_;
    DdrMemory& ddr_;
    const ModelConfig& config_;
    NumericMode mode_;
    SimulationStats& stats_;

    void wait_cycles(std::uint64_t cycles) {
        for (std::uint64_t i = 0; i < cycles; ++i) wait();
    }

    std::uint64_t write_fixed(const ManifestNode& meta, NodeStorage& node) {
        std::uint64_t delay = 0;
        delay = std::max(
            delay, ddr_.write_i32_vector(meta.l_factor.offset, node.fixed.l));
        delay = std::max(
            delay, ddr_.write_i32_vector(meta.u_factor.offset, node.fixed.u));
        delay = std::max(
            delay, ddr_.write_u16_vector(meta.p_vector.offset, node.fixed.pvec));
        if (!node.fixed.update.empty()) {
            delay = std::max(
                delay,
                ddr_.write_i32_vector(meta.update_q.offset, node.fixed.update));
            if (config_.bfp_tile_size == 0) {
                delay = std::max(
                    delay,
                    ddr_.write_i16(
                        meta.update_e.offset,
                        node.fixed.update_exponent));
            } else {
                if (meta.update_e.size !=
                    node.fixed.update_tile_exponents.size() *
                        sizeof(std::int16_t)) {
                    throw std::runtime_error(
                        "update tile exponent count disagrees with DDR region");
                }
                delay = std::max(
                    delay,
                    ddr_.write_i16_vector(
                        meta.update_e.offset,
                        node.fixed.update_tile_exponents));
            }
        }
        std::vector<std::uint8_t> node_meta(meta.node_meta.size, 0);
        if (node_meta.size() >= 4) {
            node_meta[0] = 1;
            node_meta[1] =
                node.fixed.precision_rescued ? 1 : 0;
            node_meta[2] =
                static_cast<std::uint8_t>(
                    static_cast<std::uint16_t>(
                        node.fixed.u_exponent) & 0xffu);
            node_meta[3] =
                static_cast<std::uint8_t>(
                    (static_cast<std::uint16_t>(
                        node.fixed.u_exponent) >> 8) &
                    0xffu);
            if (node_meta.size() >= 6) {
                node_meta[4] =
                    static_cast<std::uint8_t>(
                        static_cast<std::uint16_t>(
                            node.fixed.update_exponent) & 0xffu);
                node_meta[5] =
                    static_cast<std::uint8_t>(
                        (static_cast<std::uint16_t>(
                            node.fixed.update_exponent) >> 8) & 0xffu);
            }
            if (node_meta.size() >= 16) {
                const auto put_u16 =
                    [&](std::size_t offset, std::uint16_t value) {
                        node_meta[offset] =
                            static_cast<std::uint8_t>(value & 0xffu);
                        node_meta[offset + 1] =
                            static_cast<std::uint8_t>((value >> 8) & 0xffu);
                    };
                const auto put_u32 =
                    [&](std::size_t offset, std::uint32_t value) {
                        for (unsigned byte = 0; byte < 4; ++byte) {
                            node_meta[offset + byte] =
                                static_cast<std::uint8_t>(
                                    (value >> (8 * byte)) & 0xffu);
                        }
                    };
                put_u16(
                    6,
                    static_cast<std::uint16_t>(
                        node.fixed.tile_size));
                put_u32(
                    8,
                    static_cast<std::uint32_t>(
                        node.fixed.u_tile_exponents.size()));
                put_u32(
                    12,
                    static_cast<std::uint32_t>(
                        node.fixed.update_tile_exponents.size()));
                const auto required =
                    16 + node.fixed.u_tile_exponents.size() * 2;
                if (required > node_meta.size()) {
                    throw std::runtime_error(
                        "node_meta cannot hold U tile exponents");
                }
                for (std::size_t index = 0;
                     index < node.fixed.u_tile_exponents.size();
                     ++index) {
                    put_u16(
                        16 + index * 2,
                        static_cast<std::uint16_t>(
                            node.fixed.u_tile_exponents[index]));
                }
            }
        }
        delay = std::max(
            delay, ddr_.write_bytes(meta.node_meta.offset, node_meta));
        return delay;
    }

    std::uint64_t write_fp64_shadow(const ManifestNode& meta) {
        std::uint64_t delay = 0;
        for (const auto* region : {
                 &meta.l_factor, &meta.u_factor, &meta.p_vector,
                 &meta.update_q, &meta.update_e, &meta.node_meta}) {
            if (region->size != 0) {
                delay = std::max(
                    delay,
                    ddr_.write_bytes(
                        region->offset,
                        std::vector<std::uint8_t>(region->size, 0)));
            }
        }
        return delay;
    }

    void run() {
        wait();
        while (!rst_n.read()) wait();
        while (true) {
            const auto result = result_in.read();
            const auto node_id = result.work.task.node_id;
            if (!result.success) {
                release_out.write({
                    result.work.buffer_id, node_id,
                });
                continue;
            }
            auto& node = memory_.at(node_id);
            const auto& meta = manifest_.nodes.at(node_id);
            std::uint64_t delay = 0;
            try {
                delay = mode_has_fixed(mode_) ?
                    write_fixed(meta, node) : write_fp64_shadow(meta);
            } catch (const std::exception& exception) {
                node.status = NodeStatus::AddressFailure;
                node.failure_reason = exception.what();
                auto& perf = stats_.nodes[node_id];
                perf.status = NodeStatus::AddressFailure;
                perf.failure_reason = exception.what();
                stats_.address_failure = true;
                stats_.failure_reason =
                    "node " + std::to_string(node_id) +
                    " writeback: " + exception.what();
                release_out.write({result.work.buffer_id, node_id});
                continue;
            }
            delay += config_.writeback_latency;
            wait_cycles(delay);
            node.status = NodeStatus::Complete;
            auto& perf = stats_.nodes[node_id];
            perf.status = NodeStatus::Complete;
            perf.commit_cycle = stats_.cycle;
            perf.writeback_cycles = delay;
            commit_out.write({node_id, result.work.task.parent_id});
            release_out.write({result.work.buffer_id, node_id});
            stats_.timeline(
                event_text("writeback_committed",
                           node_id, result.work.buffer_id));
        }
    }
};

struct CompletionMonitor : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};

    CompletionMonitor(
        sc_core::sc_module_name name,
        std::size_t expected_nodes,
        std::uint64_t timeout_cycles,
        SimulationStats& stats)
        : sc_core::sc_module(name),
          expected_nodes_(expected_nodes),
          timeout_cycles_(timeout_cycles),
          stats_(stats) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

private:
    std::size_t expected_nodes_;
    std::uint64_t timeout_cycles_;
    SimulationStats& stats_;

    void tick() {
        if (!rst_n.read()) return;
        if (stats_.numeric_failure || stats_.address_failure ||
            stats_.control_failure) {
            sc_core::sc_stop();
            return;
        }
        if (stats_.completed_nodes == expected_nodes_) {
            sc_core::sc_stop();
            return;
        }
        if (stats_.cycle >= timeout_cycles_) {
            stats_.timed_out = true;
            stats_.failure_reason = "simulation timeout";
            sc_core::sc_stop();
        }
    }
};

}  // namespace hw
