#include "panel_lu/panel_lu_prototype.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace hw::panel_lu {

namespace {

std::uint32_t ceil_div16(std::uint32_t value) {
    return (value + 15) / 16;
}

// Disjoint task-id space per resident command: ids are [slot * stride + 1,
// (slot+1) * stride]. Worst case for max_front_dim = 512 is 32x32 tiles:
// 32 steps x (1 + 2*32 + 32*32) = 34848 tasks, below the stride.
constexpr std::uint64_t kTaskIdStride = 65536;

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

}  // namespace

PanelLuPrototype::PanelLuPrototype(PanelLuConfig config)
    : config_(config),
      global_grid_(ceil_div16(config.max_front_dim)),
      max_tiles_(global_grid_ * global_grid_),
      synthetic_access_(store_),
      dispatcher_(config_, &pes_, &store_) {}

bool SyntheticFrontReader::read_front(
    std::uint64_t byte_offset,
    std::uint32_t length_bytes,
    std::uint8_t* out,
    std::uint64_t cycle) {
    (void)cycle;
    if (layout_.kind != PanelFrontLayout::Kind::PerTileSlots ||
        layout_.total_dim == 0 || byte_offset % 4 != 0 || length_bytes % 4 != 0) {
        return false;
    }
    const std::uint64_t first_element = byte_offset / 4;
    const std::uint32_t elements = length_bytes / 4;
    for (std::uint32_t e = 0; e < elements; ++e) {
        const std::uint64_t index = first_element + e;
        const std::uint32_t row = static_cast<std::uint32_t>(index / layout_.total_dim);
        const std::uint32_t col = static_cast<std::uint32_t>(index % layout_.total_dim);
        const std::uint32_t slot = layout_.handle.slot + row / 16 * layout_.global_grid +
                                   col / 16;
        if (!store_.assigned(slot)) {
            return false;
        }
        const float value = store_.data(slot)[(row % 16) * 16 + (col % 16)];
        std::memcpy(out + static_cast<std::size_t>(e) * 4, &value, 4);
    }
    return true;
}

ConfigCheck PanelLuPrototype::configure() {
    const ConfigCheck check = validate_panel_config(config_);
    if (!check.ok) {
        configured_ = false;
        return check;
    }
    global_grid_ = ceil_div16(config_.max_front_dim);
    max_tiles_ = global_grid_ * global_grid_;
    store_.resize(config_.max_resident_commands * max_tiles_);
    pes_.clear();
    pes_.reserve(config_.pe_count);
    for (std::uint32_t id = 0; id < config_.pe_count; ++id) {
        pes_.emplace_back(id, config_);
    }
    writeback_reservation_.configure(config_.writeback_queue_depth);
    load_budget_.reset(config_.load_bytes_per_cycle);
    store_write_budget_.reset(config_.writeback_bytes_per_cycle);
    scan_budget_.reset(config_.scan_bytes_per_cycle);
    controller_slots_.clear();
    controller_slots_.reserve(config_.max_resident_commands);
    for (std::uint32_t i = 0; i < config_.max_resident_commands; ++i) {
        ControllerSlot slot;
        slot.reader = std::make_unique<SyntheticFrontReader>(store_);
        slot.controller = std::make_unique<PanelController>(
            config_, &dispatcher_, &store_, slot.reader.get());
        controller_slots_.push_back(std::move(slot));
    }
    configured_ = true;
    ConfigCheck result;
    result.ok = true;
    return result;
}

PanelController* PanelLuPrototype::find_controller(std::uint32_t command_id) {
    for (ControllerSlot& slot : controller_slots_) {
        if (slot.active && slot.command_id == command_id) {
            return slot.controller.get();
        }
    }
    return nullptr;
}

const PanelController* PanelLuPrototype::controller(std::uint32_t command_id) const {
    for (const ControllerSlot& slot : controller_slots_) {
        if (slot.active && slot.command_id == command_id) {
            return slot.controller.get();
        }
    }
    return nullptr;
}

const std::vector<TileTask>* PanelLuPrototype::command_task_graph(
    std::uint32_t command_id) const {
    const auto it = command_graphs_.find(command_id);
    return it == command_graphs_.end() ? nullptr : &it->second;
}

ConfigCheck PanelLuPrototype::submit_command(
    const PanelCommandRequest& request,
    std::uint64_t first_task_id_override) {
    ConfigCheck check;
    if (!configured_) {
        check.ok = false;
        check.status = StatusCode::BadCommand;
        check.reason = "prototype is not configured";
        return check;
    }
    if (command_history_.count(request.command_id) != 0 ||
        find_controller(request.command_id) != nullptr) {
        check.ok = false;
        check.status = StatusCode::BadCommand;
        check.reason = "duplicate command_id";
        return check;
    }
    ControllerSlot* free_slot = nullptr;
    std::uint32_t slot_index = 0;
    for (std::uint32_t i = 0; i < controller_slots_.size(); ++i) {
        if (!controller_slots_[i].active) {
            free_slot = &controller_slots_[i];
            slot_index = i;
            break;
        }
    }
    if (free_slot == nullptr) {
        check.ok = false;
        check.status = StatusCode::BufferFull;
        check.reason = "no free resident command slot";
        return check;
    }
    const std::uint32_t slot_base = slot_index * max_tiles_;
    const std::uint64_t first_task_id =
        first_task_id_override != 0 ? first_task_id_override : next_task_id_base_;

    // Validate the request before touching the synthetic front: an oversized
    // front must be rejected without materializing (or indexing) any tile.
    const ConfigCheck capability =
        check_front_capability(
            request.front_dim,
            request.pivot_dim,
            request.tile_size,
            request.policy,
            config_);
    if (!capability.ok) {
        return capability;
    }

    // Materialize the synthetic front (n x n) into the per-tile store slots.
    // The scan reads the whole front; the panel tasks only touch A11 (p x p).
    const std::uint32_t front_grid = ceil_div16(request.front_dim);
    for (std::uint32_t tr = 0; tr < front_grid; ++tr) {
        for (std::uint32_t tc = 0; tc < front_grid; ++tc) {
            std::vector<float> values(TileStore::kTileFloats, 0.0f);
            for (std::uint32_t i = 0; i < 16; ++i) {
                for (std::uint32_t j = 0; j < 16; ++j) {
                    const std::uint32_t gi = tr * 16 + i;
                    const std::uint32_t gj = tc * 16 + j;
                    if (gi >= request.front_dim || gj >= request.front_dim) {
                        continue;
                    }
                    float value = 0.0f;
                    if (gi == gj) {
                        value = request.synthetic.diagonal +
                                0.001f * static_cast<float>(
                                             (gi * 7 + request.synthetic.seed) % 5);
                    } else {
                        const int pattern = static_cast<int>(
                                                (gi * 3 + gj * 5 + request.synthetic.seed) % 7) -
                                            3;
                        value = request.synthetic.off_diagonal * static_cast<float>(pattern);
                    }
                    values[i * 16 + j] = value;
                }
            }
            store_.assign(
                slot_base + tr * global_grid_ + tc,
                "cmd" + std::to_string(request.command_id),
                static_cast<std::uint32_t>(epoch_),
                values);
        }
    }
    for (const SyntheticInject& inject : request.synthetic.injects) {
        if (inject.row >= request.front_dim || inject.col >= request.front_dim) {
            continue;
        }
        auto set_value = [&](std::uint32_t row, std::uint32_t col, float value) {
            const std::uint32_t slot = slot_base + row / 16 * global_grid_ + col / 16;
            float* data = store_.data(slot);
            data[(row % 16) * 16 + (col % 16)] = value;
        };
        switch (inject.kind) {
            case SyntheticInject::Kind::ZeroDiagonal:
                set_value(inject.row, inject.col, 0.0f);
                break;
            case SyntheticInject::Kind::TinyDiagonal:
                set_value(
                    inject.row,
                    inject.col,
                    inject.value != 0.0f ? inject.value : 1.0e-12f);
                break;
            case SyntheticInject::Kind::SingularPair: {
                if (inject.row + 1 >= request.front_dim) {
                    break;
                }
                const std::uint32_t row0 = inject.row;
                const std::uint32_t row1 = inject.row + 1;
                for (std::uint32_t col = inject.col; col < request.front_dim; ++col) {
                    float* src =
                        store_.data(slot_base + row0 / 16 * global_grid_ + col / 16);
                    float* dst =
                        store_.data(slot_base + row1 / 16 * global_grid_ + col / 16);
                    dst[(row1 % 16) * 16 + (col % 16)] =
                        src[(row0 % 16) * 16 + (col % 16)];
                }
                break;
            }
            case SyntheticInject::Kind::NanEntry:
                set_value(inject.row, inject.col, std::nanf(""));
                break;
            case SyntheticInject::Kind::InfEntry:
                set_value(
                    inject.row, inject.col, std::numeric_limits<float>::infinity());
                break;
            case SyntheticInject::Kind::None:
                break;
        }
    }

    PanelFrontLayout layout;
    layout.kind = PanelFrontLayout::Kind::PerTileSlots;
    layout.handle.slot = slot_base;
    layout.handle.generation = static_cast<std::uint32_t>(epoch_);
    layout.generation = static_cast<std::uint32_t>(epoch_);
    layout.base_offset = 0;
    layout.row_stride_bytes = 64;
    layout.global_grid = global_grid_;
    layout.total_dim = request.front_dim;
    free_slot->reader->configure(layout);
    const ConfigCheck started = free_slot->controller->start(
        request, cycle_, epoch_, layout, first_task_id);
    if (!started.ok) {
        return started;
    }
    const std::uint64_t base_after = first_task_id + kTaskIdStride;
    if (base_after > next_task_id_base_) {
        next_task_id_base_ = base_after;
    }
    free_slot->active = true;
    free_slot->command_id = request.command_id;
    command_requests_[request.command_id] = request;
    emit(cycle_,
         "COMMAND_START",
         request.command_id,
         request.node_id,
         0,
         0,
         request.label);
    return started;
}

void PanelLuPrototype::emit(
    std::uint64_t cycle,
    const std::string& event,
    std::uint32_t command_id,
    std::uint32_t node_id,
    std::uint64_t task_id,
    std::uint32_t pe_id,
    const std::string& detail) {
    PrototypeEvent record;
    record.cycle = cycle;
    record.event = event;
    record.command_id = command_id;
    record.node_id = node_id;
    record.task_id = task_id;
    record.pe_id = pe_id;
    record.detail = detail;
    events_.push_back(std::move(record));
}

void PanelLuPrototype::tick() {
    if (!configured_) {
        return;
    }
    ++cycle_;
    load_budget_.reset(config_.load_bytes_per_cycle);
    store_write_budget_.reset(config_.writeback_bytes_per_cycle);
    scan_budget_.reset(config_.scan_bytes_per_cycle);

    // Phase 1: PE progress. Shared load/writeback budgets are consumed here.
    for (MultiModePe& pe : pes_) {
        pe.tick(
            cycle_,
            load_budget_,
            store_write_budget_,
            writeback_reservation_,
            synthetic_access_);
    }
    // Phase 2: route terminal task responses (commit visible results).
    route_pe_responses();
    // Phase 3: dispatch ready tasks to the shared PE pool.
    dispatcher_.tick(cycle_);
    // Phase 4: controllers observe failures/aborts, cancel if required, then
    // advance scan/generation/p-vector progress.
    handle_cancel_requests();
    // Rotate the controller service order so the shared scan port is fairly
    // shared between resident commands instead of starving later slots.
    const std::size_t slot_count = controller_slots_.size();
    if (slot_count > 0) {
        for (std::size_t i = 0; i < slot_count; ++i) {
            ControllerSlot& slot =
                controller_slots_[(controller_rotation_ + i) % slot_count];
            if (slot.active) {
                slot.controller->tick(cycle_, scan_budget_, store_write_budget_);
            }
        }
        controller_rotation_ = (controller_rotation_ + 1) % slot_count;
    }
    // Phase 5: collect finished command responses.
    drain_command_responses();

    // Phase 6: copy per-PE lifecycle events into the prototype trace.
    for (MultiModePe& pe : pes_) {
        for (const PeEvent& event : pe.drain_events()) {
            emit(
                event.cycle,
                pe_event_name(event.kind),
                event.command_id,
                hw::command::NONE,
                event.task_id,
                event.pe_id,
                "");
        }
    }
}

void PanelLuPrototype::route_pe_responses() {
    for (MultiModePe& pe : pes_) {
        TileTaskResponse response;
        while (pe.poll_response(response, cycle_)) {
            PanelController* ctrl = find_controller(response.command_id);
            DispatchRecord meta;
            const bool has_meta = dispatcher_.meta(response.task_id, &meta);
            if (has_meta) {
                response.enqueue_cycle = meta.enqueue_cycle;
                response.dispatch_cycle = meta.dispatch_cycle;
                response.stall_dependency_cycles = meta.stall_dependency_cycles;
                response.stall_resource_cycles = meta.stall_resource_cycles;
            }
            const TileTask* task_view =
                ctrl != nullptr ? ctrl->find_task(response.task_id) : nullptr;
            bool committed = false;
            if (ctrl != nullptr && !ctrl->finished() && response.result_visible &&
                response.status == StatusCode::Ok && task_view != nullptr) {
                committed = dispatcher_.on_task_committed(
                    response.task_id,
                    response.command_id,
                    task_view->dst.version_key,
                    task_view->dst.version);
                if (!committed) {
                    control_fault_ = true;
                    control_fault_reason_ =
                        "stale output version at task commit (task " +
                        std::to_string(response.task_id) + ")";
                }
            } else {
                dispatcher_.on_task_terminal(response.task_id);
            }
            response.result_visible = committed;
            if (task_view != nullptr) {
                record_task(*task_view, &response, committed);
            }
            emit(
                response.writeback_done_cycle != 0 ? response.writeback_done_cycle
                                                   : response.exec_done_cycle,
                response.status == StatusCode::Ok
                    ? (committed ? "TASK_COMMIT" : "TASK_DISCARD")
                    : (response.status == StatusCode::Aborted ? "TASK_ABORTED"
                                                              : "TASK_FAILED"),
                response.command_id,
                response.node_id,
                response.task_id,
                response.pe_id,
                response.reason);
            if (ctrl != nullptr) {
                ctrl->on_task_response(response, cycle_);
            }
        }
    }
}

void PanelLuPrototype::record_task(
    const TileTask& task,
    const TileTaskResponse* response,
    bool committed) {
    TaskRecord record;
    record.task = task;
    record.committed = committed;
    record.result_visible = committed;
    if (response == nullptr) {
        record.cancelled = true;
        record.final_phase = TaskPhase::Cancelled;
        record.status = StatusCode::Aborted;
        record.reason = "cancelled in dispatcher queue";
        record.pe_id = hw::command::NONE;
        task_records_.push_back(std::move(record));
        ++tasks_cancelled_;
        return;
    }
    record.pe_id = response->pe_id;
    record.status = response->status;
    record.reason = response->reason;
    record.failed_column = response->failed_column;
    record.enqueue_cycle = response->enqueue_cycle;
    record.dispatch_cycle = response->dispatch_cycle;
    record.load_done_cycle = response->load_done_cycle;
    record.exec_start_cycle = response->exec_start_cycle;
    record.exec_done_cycle = response->exec_done_cycle;
    record.writeback_done_cycle = response->writeback_done_cycle;
    record.bytes_read = response->bytes_read;
    record.bytes_written = response->bytes_written;
    record.stall_dependency_cycles = response->stall_dependency_cycles;
    record.stall_resource_cycles = response->stall_resource_cycles;
    record.stall_port_cycles = response->stall_port_cycles;
    record.stall_writeback_cycles = response->stall_writeback_cycles;
    if (response->status == StatusCode::Ok) {
        record.final_phase = TaskPhase::Done;
        record.cancelled = !committed;
        if (committed) {
            ++tasks_committed_;
        } else {
            ++tasks_cancelled_;
        }
    } else if (response->status == StatusCode::Aborted) {
        record.final_phase = TaskPhase::Cancelled;
        record.cancelled = true;
        ++tasks_cancelled_;
    } else {
        record.final_phase = TaskPhase::Failed;
        ++tasks_failed_;
    }
    load_bytes_ += response->bytes_read;
    writeback_bytes_ += response->bytes_written;
    const std::string mode = tile_mode_name(task.mode);
    ++mode_counts_[mode];
    if (response->exec_done_cycle > response->exec_start_cycle) {
        mode_cycles_[mode] += response->exec_done_cycle - response->exec_start_cycle;
    }
    task_records_.push_back(std::move(record));
}

void PanelLuPrototype::handle_cancel_requests() {
    for (ControllerSlot& slot : controller_slots_) {
        if (!slot.active) {
            continue;
        }
        if (!slot.controller->take_cancel_request()) {
            continue;
        }
        const std::vector<TileTask> cancelled =
            dispatcher_.cancel_command(slot.command_id);
        slot.controller->on_tasks_cancelled(cancelled);
        for (const TileTask& task : cancelled) {
            record_task(task, nullptr, false);
            emit(
                cycle_,
                "TASK_CANCELLED",
                task.command_id,
                task.node_id,
                task.task_id,
                hw::command::NONE,
                "cancelled before dispatch");
        }
        for (MultiModePe& pe : pes_) {
            pe.abort_command(slot.command_id, cycle_);
        }
        emit(
            cycle_,
            "COMMAND_CANCEL",
            slot.command_id,
            slot.controller->request().node_id,
            0,
            0,
            "directed cancellation");
    }
}

void PanelLuPrototype::drain_command_responses() {
    for (ControllerSlot& slot : controller_slots_) {
        if (!slot.active) {
            continue;
        }
        PanelCommandResponse response;
        if (!slot.controller->poll_response(response)) {
            continue;
        }
        command_responses_.push_back(response);
        command_history_[response.command_id] = response;
        command_graphs_[response.command_id] = slot.controller->task_graph();
        dispatcher_.release_command(response.command_id);
        if (slot.controller->task_graph().size() >= kTaskIdStride) {
            control_fault_ = true;
            control_fault_reason_ = "task-id space exhausted";
        }
        scan_bytes_ += response.scan_bytes;
        pvector_bytes_ += response.pvector_bytes;
        emit(
            response.finish_cycle,
            response.status == StatusCode::Ok
                ? "COMMAND_DONE"
                : (response.status == StatusCode::Aborted ? "COMMAND_ABORTED"
                                                          : "COMMAND_FAILED"),
            response.command_id,
            response.node_id,
            0,
            0,
            response.reason);
        slot.controller->reset();
        slot.active = false;
        slot.command_id = hw::command::NONE;
    }
}

bool PanelLuPrototype::poll_response(PanelCommandResponse& out) {
    if (command_responses_.empty()) {
        return false;
    }
    out = command_responses_.front();
    command_responses_.erase(command_responses_.begin());
    return true;
}

void PanelLuPrototype::abort_command(std::uint32_t command_id) {
    PanelController* ctrl = find_controller(command_id);
    if (ctrl == nullptr) {
        return;
    }
    ctrl->request_abort();
}

void PanelLuPrototype::reset() {
    ++epoch_;
    next_task_id_base_ = 1;
    store_.bump_generations();
    dispatcher_.reset();
    for (MultiModePe& pe : pes_) {
        pe.reset();
    }
    for (ControllerSlot& slot : controller_slots_) {
        slot.controller->reset();
        slot.active = false;
        slot.command_id = hw::command::NONE;
    }
    writeback_reservation_.reset();
    load_budget_.reset(config_.load_bytes_per_cycle);
    store_write_budget_.reset(config_.writeback_bytes_per_cycle);
    scan_budget_.reset(config_.scan_bytes_per_cycle);
    task_records_.clear();
    events_.clear();
    command_responses_.clear();
    command_graphs_.clear();
    command_history_.clear();
    command_requests_.clear();
    mode_counts_.clear();
    mode_cycles_.clear();
    scan_bytes_ = 0;
    load_bytes_ = 0;
    writeback_bytes_ = 0;
    pvector_bytes_ = 0;
    tasks_committed_ = 0;
    tasks_cancelled_ = 0;
    tasks_failed_ = 0;
    control_fault_ = false;
    control_fault_reason_.clear();
}

bool PanelLuPrototype::idle() const {
    for (const ControllerSlot& slot : controller_slots_) {
        if (slot.active) {
            return false;
        }
    }
    if (!dispatcher_.empty()) {
        return false;
    }
    for (const MultiModePe& pe : pes_) {
        if (pe.active_task_count() != 0) {
            return false;
        }
    }
    return true;
}

bool PanelLuPrototype::run_until_idle(std::uint64_t max_cycles) {
    const std::uint64_t start = cycle_;
    while (!idle() && cycle_ - start < max_cycles) {
        tick();
    }
    return idle();
}

std::string PanelLuPrototype::summary_json() const {
    std::ostringstream out;
    out << "{\n";
    out << "  \"tool\": \"panel_lu_architecture_prototype\",\n";
    out << "  \"execution_backend\": \"architecture_prototype\",\n";
    out << "  \"architecture_prototype\": true,\n";
    out << "  \"numerical_validation\": \"not_performed\",\n";
    out << "  \"fp64_golden_used\": false,\n";
    out << "  \"description\": \"task-granularity timing/resource model of the shared multi-mode PE pool; synthetic workloads only\",\n";
    out << "  \"cycle\": " << cycle_ << ",\n";
    out << "  \"epoch\": " << epoch_ << ",\n";
    out << "  \"control_fault\": " << (control_fault_ ? "true" : "false") << ",\n";
    out << "  \"control_fault_reason\": \"" << json_escape(control_fault_reason_) << "\",\n";
    out << "  \"pe_count\": " << config_.pe_count << ",\n";
    out << "  \"task_slots_per_pe\": " << config_.task_slots_per_pe << ",\n";
    out << "  \"tile_dim\": " << config_.tile_dim << ",\n";
    out << "  \"dispatcher_queue_depth\": " << config_.dispatcher_queue_depth << ",\n";
    out << "  \"dispatcher_high_watermark\": " << dispatcher_.high_watermark() << ",\n";
    out << "  \"dispatcher_queue_full_stall_cycles\": "
        << dispatcher_.queue_full_stall_cycles() << ",\n";
    out << "  \"tasks_generated\": " << task_records_.size() << ",\n";
    out << "  \"tasks_committed\": " << tasks_committed_ << ",\n";
    out << "  \"tasks_cancelled\": " << tasks_cancelled_ << ",\n";
    out << "  \"tasks_failed\": " << tasks_failed_ << ",\n";
    out << "  \"scan_bytes\": " << scan_bytes_ << ",\n";
    out << "  \"load_bytes\": " << load_bytes_ << ",\n";
    out << "  \"writeback_bytes\": " << writeback_bytes_ << ",\n";
    out << "  \"pvector_bytes\": " << pvector_bytes_ << ",\n";
    out << "  \"commands\": [\n";
    bool first = true;
    for (const auto& entry : command_history_) {
        const PanelCommandResponse& response = entry.second;
        const auto request_it = command_requests_.find(entry.first);
        out << (first ? "" : ",\n");
        first = false;
        out << "    {\"command_id\": " << response.command_id
            << ", \"node_id\": " << response.node_id
            << ", \"front_dim\": "
            << (request_it != command_requests_.end() ? request_it->second.front_dim : 0)
            << ", \"pivot_dim\": "
            << (request_it != command_requests_.end() ? request_it->second.pivot_dim : 0)
            << ", \"policy\": \""
            << (request_it != command_requests_.end()
                    ? (request_it->second.policy == hw::command::PivotPolicy::Static
                           ? "STATIC"
                           : "DYNAMIC_PANEL")
                    : "UNKNOWN")
            << "\""
            << ", \"status\": \"" << hw::dataplane::status_name(response.status) << "\""
            << ", \"reason\": \"" << json_escape(response.reason) << "\""
            << ", \"pivot_count\": " << response.pivot_count
            << ", \"failed_column\": "
            << (response.failed_column == hw::command::NONE ? -1
                                                            : static_cast<int>(response.failed_column))
            << ", \"pvector_written\": " << (response.pvector_written ? "true" : "false")
            << ", \"start_cycle\": " << response.start_cycle
            << ", \"finish_cycle\": " << response.finish_cycle
            << ", \"tasks_generated\": " << response.tasks_generated
            << ", \"tasks_cancelled\": " << response.tasks_cancelled
            << ", \"stall_dispatcher_full\": " << response.stall_dispatcher_full
            << ", \"stall_scan_port\": " << response.stall_scan_port
            << "}";
    }
    out << "\n  ],\n";
    out << "  \"mode_counts\": {";
    first = true;
    for (const auto& entry : mode_counts_) {
        out << (first ? "" : ", ") << "\"" << entry.first << "\": " << entry.second;
        first = false;
    }
    out << "},\n";
    out << "  \"mode_exec_cycles\": {";
    first = true;
    for (const auto& entry : mode_cycles_) {
        out << (first ? "" : ", ") << "\"" << entry.first << "\": " << entry.second;
        first = false;
    }
    out << "},\n";
    out << "  \"pe_busy_cycles\": [";
    for (std::size_t i = 0; i < pes_.size(); ++i) {
        out << (i == 0 ? "" : ", ") << pes_[i].array_busy_cycles();
    }
    out << "],\n";
    out << "  \"pending_calibration\": [\n";
    out << "    \"all PE latency/bandwidth parameters are design assumptions\",\n";
    out << "    \"no RTL synthesis or silicon calibration performed\",\n";
    out << "    \"cycle counts are architecture-model cycles, not chip measurements\"\n";
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

void PanelLuPrototype::write_reports(const std::string& output_dir) const {
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);

    {
        std::ofstream tasks(output_dir + "/panel_tasks.csv");
        tasks << "task_id,command_id,node_id,step,tile_row,tile_col,mode,eff_m,eff_n,eff_k,"
                 "pe_id,status,result_visible,committed,cancelled,failed_column,"
                 "enqueue_cycle,dispatch_cycle,load_done_cycle,exec_start_cycle,"
                 "exec_done_cycle,writeback_done_cycle,bytes_read,bytes_written,"
                 "stall_dependency_cycles,stall_resource_cycles,stall_port_cycles,"
                 "stall_writeback_cycles,reason\n";
        for (const TaskRecord& record : task_records_) {
            tasks << record.task.task_id << ',' << record.task.command_id << ','
                  << record.task.node_id << ',' << record.task.step << ','
                  << record.task.tile_row << ',' << record.task.tile_col << ','
                  << tile_mode_name(record.task.mode) << ',' << record.task.dims.m << ','
                  << record.task.dims.n << ',' << record.task.dims.k << ','
                  << (record.pe_id == hw::command::NONE
                          ? -1
                          : static_cast<int>(record.pe_id))
                  << ',' << hw::dataplane::status_name(record.status) << ','
                  << (record.result_visible ? 1 : 0) << ',' << (record.committed ? 1 : 0)
                  << ',' << (record.cancelled ? 1 : 0) << ',' << record.failed_column << ','
                  << record.enqueue_cycle << ',' << record.dispatch_cycle << ','
                  << record.load_done_cycle << ',' << record.exec_start_cycle << ','
                  << record.exec_done_cycle << ',' << record.writeback_done_cycle << ','
                  << record.bytes_read << ',' << record.bytes_written << ','
                  << record.stall_dependency_cycles << ',' << record.stall_resource_cycles
                  << ',' << record.stall_port_cycles << ','
                  << record.stall_writeback_cycles << ',' << json_escape(record.reason)
                  << '\n';
        }
    }
    {
        std::ofstream commands(output_dir + "/panel_commands.csv");
        commands << "command_id,node_id,status,reason,pivot_count,failed_column,"
                    "completed_pivots,pvector_written,pvector_bytes,start_cycle,finish_cycle,"
                    "scan_bytes,scan_done_cycle,front_max_abs,threshold_product,"
                    "tasks_generated,tasks_cancelled,stall_dispatcher_full,stall_scan_port\n";
        for (const auto& entry : command_history_) {
            const PanelCommandResponse& r = entry.second;
            commands << r.command_id << ',' << r.node_id << ','
                     << hw::dataplane::status_name(r.status) << ','
                     << json_escape(r.reason) << ',' << r.pivot_count << ','
                     << (r.failed_column == hw::command::NONE
                             ? -1
                             : static_cast<int>(r.failed_column))
                     << ',' << r.completed_pivots << ','
                     << (r.pvector_written ? 1 : 0) << ',' << r.pvector_bytes << ','
                     << r.start_cycle << ',' << r.finish_cycle << ',' << r.scan_bytes << ','
                     << r.scan_done_cycle << ',' << r.front_pre_elimination_max_abs << ','
                     << r.threshold_product << ',' << r.tasks_generated << ','
                     << r.tasks_cancelled << ',' << r.stall_dispatcher_full << ','
                     << r.stall_scan_port << '\n';
        }
    }
    {
        std::ofstream events(output_dir + "/panel_events.csv");
        events << "cycle,event,command_id,node_id,task_id,pe_id,detail\n";
        for (const PrototypeEvent& record : events_) {
            events << record.cycle << ',' << record.event << ',' << record.command_id << ','
                   << record.node_id << ',' << record.task_id << ',' << record.pe_id << ','
                   << json_escape(record.detail) << '\n';
        }
    }
    {
        std::ofstream summary(output_dir + "/panel_summary.json");
        summary << summary_json();
    }
}

}  // namespace hw::panel_lu
