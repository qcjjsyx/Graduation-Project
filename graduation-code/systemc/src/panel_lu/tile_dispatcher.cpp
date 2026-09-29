#include "panel_lu/tile_dispatcher.hpp"

#include <algorithm>

namespace hw::panel_lu {

TileDispatcher::TileDispatcher(
    const PanelLuConfig& config,
    std::vector<MultiModePe>* pes,
    TileStore* store)
    : config_(config), pes_(pes), store_(store) {}

bool TileDispatcher::submit(const TileTask& task, std::uint64_t cycle, std::string* reason) {
    if (known_.count(task.task_id) != 0) {
        if (reason != nullptr) {
            *reason = "duplicate task_id";
        }
        return false;
    }
    const ConfigCheck check = validate_tile_task(task, config_);
    if (!check.ok) {
        if (reason != nullptr) {
            *reason = "invalid task: " + check.reason;
        }
        return false;
    }
    if (queue_.size() >= config_.dispatcher_queue_depth) {
        ++queue_full_stall_cycles_;
        if (reason != nullptr) {
            *reason = "dispatcher queue full";
        }
        return false;
    }
    Queued item;
    item.task = task;
    item.enqueue_cycle = cycle;
    queue_.push_back(std::move(item));
    known_.insert(task.task_id);
    DispatchRecord& record = records_[task.task_id];
    record.enqueue_cycle = cycle;
    if (queue_.size() > high_watermark_) {
        high_watermark_ = static_cast<std::uint32_t>(queue_.size());
    }
    return true;
}

bool TileDispatcher::deps_ready(const TileTask& task) const {
    for (const std::uint64_t dep : task.deps) {
        if (committed_.count(dep) == 0) {
            return false;
        }
    }
    return true;
}

bool TileDispatcher::dispatchable(const TileTask& task) const {
    if (!deps_ready(task)) {
        return false;
    }
    const auto& operands = {task.dst, task.lhs, task.rhs};
    const std::uint32_t dst_slot = task.dst.handle.slot;
    if (store_->version(dst_slot) != task.dst.version) {
        return false;
    }
    for (const TileOperand& operand : operands) {
        if (!operand.valid || operand.handle.slot == dst_slot) {
            continue;
        }
        if (store_->version(operand.handle.slot) < operand.version) {
            return false;
        }
    }
    const auto writer = uncommitted_writers_.find(dst_slot);
    if (writer != uncommitted_writers_.end() && !writer->second.empty()) {
        return false;
    }
    return true;
}

void TileDispatcher::tick(std::uint64_t cycle) {
    if (queue_.empty()) {
        rr_cursor_ = 0;
        return;
    }

    // Per-tick stall classification: ordering wait (dependency/version/target
    // writer) versus resource wait (no PE slot / FIFO credit).
    for (Queued& item : queue_) {
        if (!dispatchable(item.task)) {
            ++item.stall_dependency_cycles;
        }
    }

    // Commands present in the queue, in first-appearance order.
    command_order_.clear();
    for (const Queued& item : queue_) {
        const std::uint32_t command = item.task.command_id;
        if (std::find(command_order_.begin(), command_order_.end(), command) ==
            command_order_.end()) {
            command_order_.push_back(command);
        }
    }
    if (!command_order_.empty() && rr_cursor_ >= command_order_.size()) {
        rr_cursor_ = 0;
    }

    std::deque<Queued> remaining;
    std::vector<bool> dispatched(queue_.size(), false);
    std::size_t served = 0;
    const std::size_t command_count = command_order_.size();
    for (std::size_t round = 0; round < command_count; ++round) {
        const std::size_t command_index = (rr_cursor_ + round) % command_count;
        const std::uint32_t command = command_order_[command_index];
        bool served_command = false;
        for (std::size_t index = 0; index < queue_.size(); ++index) {
            if (dispatched[index] || queue_[index].task.command_id != command) {
                continue;
            }
            Queued& item = queue_[index];
            if (!dispatchable(item.task)) {
                continue;
            }
            if (dispatch_one(item, cycle)) {
                dispatched[index] = true;
                served_command = true;
            } else {
                ++item.stall_resource_cycles;
            }
        }
        if (served_command) {
            ++served;
        }
    }
    if (served > 0) {
        rr_cursor_ = (rr_cursor_ + 1) % command_count;
    }

    for (std::size_t index = 0; index < queue_.size(); ++index) {
        if (!dispatched[index]) {
            // Eligible tasks that could not get a PE this tick were already
            // accounted as resource stalls inside the rounds; do not double
            // count them here.
            remaining.push_back(std::move(queue_[index]));
        }
    }
    queue_.swap(remaining);
}

bool TileDispatcher::dispatch_one(Queued& item, std::uint64_t cycle) {
    if (pes_ == nullptr || pes_->empty()) {
        return false;
    }
    // Round-robin PE selection: deterministic and no PE preference by mode.
    for (std::size_t attempt = 0; attempt < pes_->size(); ++attempt) {
        const std::size_t pe_index = (pe_cursor_ + attempt) % pes_->size();
        MultiModePe& pe = (*pes_)[pe_index];
        if (!pe.has_free_slot()) {
            continue;
        }
        std::string reason;
        if (!pe.try_accept(item.task, cycle, *store_, &reason)) {
            continue;
        }
        DispatchRecord& record = records_[item.task.task_id];
        record.dispatch_cycle = cycle;
        record.stall_dependency_cycles = item.stall_dependency_cycles;
        record.stall_resource_cycles = item.stall_resource_cycles;
        uncommitted_writers_[item.task.dst.handle.slot].insert(item.task.task_id);
        known_.erase(item.task.task_id);
        ++dispatched_;
        pe_cursor_ = (pe_index + 1) % pes_->size();
        return true;
    }
    return false;
}

void TileDispatcher::forget_task(std::uint64_t task_id) {
    known_.erase(task_id);
    records_.erase(task_id);
    committed_.erase(task_id);
    for (auto& entry : uncommitted_writers_) {
        entry.second.erase(task_id);
    }
}

bool TileDispatcher::on_task_committed(
    std::uint64_t task_id,
    std::uint32_t command_id,
    std::uint32_t slot,
    std::uint32_t expected_version) {
    auto writer = uncommitted_writers_.find(slot);
    if (writer != uncommitted_writers_.end()) {
        writer->second.erase(task_id);
    }
    const bool ok = store_->commit(slot, expected_version);
    committed_.insert(task_id);
    committed_by_command_[command_id].insert(task_id);
    known_.erase(task_id);
    records_.erase(task_id);
    return ok;
}

void TileDispatcher::on_task_terminal(std::uint64_t task_id) {
    forget_task(task_id);
}

std::vector<TileTask> TileDispatcher::cancel_command(std::uint32_t command_id) {
    std::vector<TileTask> cancelled;
    std::deque<Queued> remaining;
    for (Queued& item : queue_) {
        if (item.task.command_id == command_id) {
            cancelled.push_back(item.task);
            forget_task(item.task.task_id);
        } else {
            remaining.push_back(std::move(item));
        }
    }
    queue_.swap(remaining);
    return cancelled;
}

void TileDispatcher::release_command(std::uint32_t command_id) {
    const auto it = committed_by_command_.find(command_id);
    if (it == committed_by_command_.end()) {
        return;
    }
    for (const std::uint64_t task_id : it->second) {
        committed_.erase(task_id);
    }
    committed_by_command_.erase(it);
}

bool TileDispatcher::meta(std::uint64_t task_id, DispatchRecord* out) const {
    const auto it = records_.find(task_id);
    if (it == records_.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

void TileDispatcher::reset() {
    queue_.clear();
    committed_.clear();
    known_.clear();
    uncommitted_writers_.clear();
    records_.clear();
    committed_by_command_.clear();
    command_order_.clear();
    rr_cursor_ = 0;
    pe_cursor_ = 0;
    high_watermark_ = 0;
    queue_full_stall_cycles_ = 0;
    dispatched_ = 0;
}

}  // namespace hw::panel_lu
