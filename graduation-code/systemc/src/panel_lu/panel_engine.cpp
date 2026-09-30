#include "panel_lu/panel_engine.hpp"

#include <algorithm>
#include <cstring>

namespace hw::panel_lu {

namespace {

std::uint32_t ceil_div16(std::uint32_t value) {
    return (value + 15) / 16;
}

// Disjoint task-id range per command context (ids are uint64 end-to-end).
constexpr std::uint64_t kTaskIdStride = 65536;

}  // namespace

PanelEngine::PanelEngine(PanelLuConfig config)
    : config_(config), dispatcher_(config_, &pes_, this) {}

std::uint32_t PanelEngine::version(std::uint64_t key) const {
    const auto found = versions_.find(key);
    return found == versions_.end() ? 0 : found->second;
}

bool PanelEngine::commit(std::uint64_t key, std::uint32_t expected_version) {
    std::uint32_t& current = versions_[key];
    if (current != expected_version) {
        return false;
    }
    ++current;
    return true;
}

void PanelEngine::erase_slot_versions(std::uint32_t slot) {
    const std::uint64_t prefix = static_cast<std::uint64_t>(slot) << 32;
    auto iterator = versions_.lower_bound(prefix);
    while (iterator != versions_.end() && (iterator->first >> 32) == slot) {
        iterator = versions_.erase(iterator);
    }
}

ConfigCheck PanelEngine::configure() {
    const ConfigCheck check = validate_panel_config(config_);
    if (!check.ok) {
        configured_ = false;
        return check;
    }
    pes_.clear();
    pes_.reserve(config_.pe_count);
    for (std::uint32_t id = 0; id < config_.pe_count; ++id) {
        pes_.emplace_back(id, config_);
    }
    writeback_reservation_.configure(config_.writeback_queue_depth);
    read_budget_.reset(config_.load_bytes_per_cycle);
    write_budget_.reset(config_.writeback_bytes_per_cycle);
    contexts_.clear();
    contexts_.resize(config_.max_resident_commands);
    configured_ = true;
    ConfigCheck result;
    result.ok = true;
    return result;
}

ConfigCheck PanelEngine::begin_command(
    const PanelEngineRequest& request,
    std::uint32_t adapter_unit_id,
    std::uint64_t cycle) {
    ConfigCheck check;
    if (!configured_) {
        check.ok = false;
        check.status = StatusCode::BadCommand;
        check.reason = "panel engine is not configured";
        return check;
    }
    Context* free_context = nullptr;
    for (Context& context : contexts_) {
        if (!context.active) {
            free_context = &context;
            break;
        }
    }
    if (free_context == nullptr) {
        check.ok = false;
        check.status = StatusCode::BufferFull;
        check.reason = "no free panel context";
        return check;
    }
    free_context->request = request;
    free_context->adapter_unit_id = adapter_unit_id;
    free_context->bytes_read = 0;
    free_context->bytes_written = 0;
    free_context->aborting = false;
    free_context->response_ready = false;
    check = build_context(*free_context, cycle);
    if (!check.ok) {
        *free_context = Context{};
        return check;
    }
    return check;
}

ConfigCheck PanelEngine::build_context(Context& context, std::uint64_t cycle) {
    const PanelEngineRequest& request = context.request;
    ConfigCheck check;
    if (request.policy == hw::command::PivotPolicy::DynamicPanel) {
        check.ok = false;
        check.status = StatusCode::PolicyNotSupported;
        check.reason = "DYNAMIC_PANEL is not supported by the T07 STATIC engine";
        return check;
    }
    check = check_front_capability(
        request.total_dim,
        request.pivot_dim,
        request.tile_size,
        request.policy,
        config_);
    if (!check.ok) {
        return check;
    }
    if (request.front_lease.lease_id == 0) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "panel engine requires an authorized front lease";
        return check;
    }

    // A reused front slot starts a fresh write-version sequence.
    erase_slot_versions(request.front.slot);

    const std::uint32_t row_stride = request.total_dim * 4;
    const std::uint32_t grid = ceil_div16(request.total_dim);

    if (buffers_ == nullptr) {
        check.ok = false;
        check.status = StatusCode::BadCommand;
        check.reason = "panel engine has no buffer manager binding";
        return check;
    }
    SlotBinding binding;
    binding.buffers = buffers_;
    binding.lease_id = request.front_lease.lease_id;
    binding.ready_cycle = request.front_lease.ready_cycle;
    binding.base_offset = request.front_offset;
    binding.row_stride_bytes = row_stride;
    binding.valid = true;

    PanelFrontLayout layout;
    layout.kind = PanelFrontLayout::Kind::SingleBufferStrided;
    layout.handle = request.front;
    layout.generation = request.front.generation;
    layout.base_offset = request.front_offset;
    layout.row_stride_bytes = row_stride;
    layout.global_grid = grid;
    layout.total_dim = request.total_dim;

    PanelCommandRequest controller_request;
    controller_request.command_id = request.command_id;
    controller_request.node_id = request.node_id;
    controller_request.front_dim = request.total_dim;
    controller_request.pivot_dim = request.pivot_dim;
    controller_request.policy = request.policy;
    controller_request.threshold_bits = request.threshold_bits;
    controller_request.tile_size = request.tile_size;
    controller_request.label = request.label;

    context.reader = std::make_unique<LeaseFrontReader>(access_);
    context.reader->configure(
        request.front.slot, request.front_offset, row_stride, request.total_dim);
    context.controller = std::make_unique<PanelController>(
        config_, &dispatcher_, this, context.reader.get());
    access_.bind(request.front.slot, binding);
    const ConfigCheck started = context.controller->start(
        controller_request, cycle, epoch_, layout, next_task_id_);
    if (!started.ok) {
        access_.unbind(request.front.slot);
        context.reader.reset();
        context.controller.reset();
        return started;
    }
    const std::uint64_t used = context.controller->next_task_id();
    next_task_id_ = std::max(next_task_id_, used + kTaskIdStride);
    context.active = true;
    context.response = PanelEngineResponse{};
    context.response.txn_id = request.txn_id;
    context.response.epoch = request.epoch;
    context.response.command_id = request.command_id;
    context.response.node_id = request.node_id;
    context.response.start_cycle = cycle;
    check.ok = true;
    return check;
}

void PanelEngine::tick(std::uint64_t cycle) {
    if (!configured_) {
        return;
    }
    cycle_ = cycle;
    read_budget_.reset(config_.load_bytes_per_cycle);
    write_budget_.reset(config_.writeback_bytes_per_cycle);

    // Phase 1: PE progress (shared read/write budgets).
    for (MultiModePe& pe : pes_) {
        pe.tick(cycle, read_budget_, write_budget_, writeback_reservation_, access_);
    }
    // Phase 2: route terminal task responses (commits + controller updates).
    route_pe_responses(cycle);
    // Phase 3: dispatch ready tasks to the shared PE pool.
    dispatcher_.tick(cycle);
    // Phase 4: directed cancellation requested by controllers.
    handle_cancel_requests(cycle);
    // Phase 5: advance contexts (scan/generation), fair rotation.
    service_contexts(cycle);
    // Phase 6: contexts whose controller and tasks have drained publish their
    // single terminal response.
    finalize_contexts(cycle);
}

void PanelEngine::route_pe_responses(std::uint64_t cycle) {
    for (MultiModePe& pe : pes_) {
        TileTaskResponse response;
        while (pe.poll_response(response, cycle)) {
            Context* context = find_context(response.command_id);
            DispatchRecord meta;
            if (dispatcher_.meta(response.task_id, &meta)) {
                response.enqueue_cycle = meta.enqueue_cycle;
                response.dispatch_cycle = meta.dispatch_cycle;
                response.stall_dependency_cycles = meta.stall_dependency_cycles;
                response.stall_resource_cycles = meta.stall_resource_cycles;
            }
            const TileTask* task_view =
                context != nullptr && context->controller
                    ? context->controller->find_task(response.task_id)
                    : nullptr;
            bool committed = false;
            if (context != nullptr && !context->aborting && response.result_visible &&
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
                EngineTaskRecord record;
                record.task = *task_view;
                record.status = response.status;
                record.reason = response.reason;
                record.failed_column = response.failed_column;
                record.pe_id = response.pe_id;
                record.committed = committed;
                record.cancelled = response.status != StatusCode::Ok && !committed;
                record.enqueue_cycle = response.enqueue_cycle;
                record.dispatch_cycle = response.dispatch_cycle;
                record.load_done_cycle = response.load_done_cycle;
                record.exec_start_cycle = response.exec_start_cycle;
                record.exec_done_cycle = response.exec_done_cycle;
                record.writeback_done_cycle = response.writeback_done_cycle;
                record.bytes_read = response.bytes_read;
                record.bytes_written = response.bytes_written;
                if (committed) {
                    ++tasks_committed_;
                } else if (response.status == StatusCode::Aborted) {
                    ++tasks_cancelled_;
                }
                context->bytes_read += response.bytes_read;
                context->bytes_written += response.bytes_written;
                task_records_.push_back(std::move(record));
            }
            if (context != nullptr && context->controller) {
                context->controller->on_task_response(response, cycle);
            }
        }
    }
}

void PanelEngine::handle_cancel_requests(std::uint64_t cycle) {
    for (Context& context : contexts_) {
        if (!context.active || !context.controller) {
            continue;
        }
        if (!context.controller->take_cancel_request()) {
            continue;
        }
        const std::vector<TileTask> cancelled =
            dispatcher_.cancel_command(context.request.command_id);
        context.controller->on_tasks_cancelled(cancelled);
        for (const TileTask& task : cancelled) {
            EngineTaskRecord record;
            record.task = task;
            record.status = StatusCode::Aborted;
            record.reason = "cancelled before dispatch";
            record.cancelled = true;
            task_records_.push_back(std::move(record));
            ++tasks_cancelled_;
        }
        for (MultiModePe& pe : pes_) {
            pe.abort_command(context.request.command_id, cycle);
        }
    }
}

void PanelEngine::service_contexts(std::uint64_t cycle) {
    const std::size_t count = contexts_.size();
    if (count == 0) {
        return;
    }
    for (std::size_t index = 0; index < count; ++index) {
        Context& context = contexts_[(rotation_ + index) % count];
        if (!context.active || !context.controller) {
            continue;
        }
        context.controller->tick(cycle, read_budget_, write_budget_);
    }
    rotation_ = (rotation_ + 1) % count;
}

void PanelEngine::finalize_contexts(std::uint64_t cycle) {
    for (Context& context : contexts_) {
        if (!context.active || !context.controller || context.response_ready) {
            continue;
        }
        if (!context.controller->finished() ||
            context.controller->outstanding_count() != 0) {
            continue;
        }
        PanelCommandResponse controller_response;
        if (!context.controller->poll_response(controller_response)) {
            continue;
        }
        PanelEngineResponse& response = context.response;
        response.status = controller_response.status;
        response.reason = controller_response.reason;
        response.failed_column = controller_response.failed_column;
        response.pivot_count = controller_response.pivot_count;
        response.finish_cycle = cycle;
        response.bytes_read = context.bytes_read;
        response.bytes_written = context.bytes_written;
        response.tasks_total = controller_response.tasks_generated;
        response.tasks_cancelled = controller_response.tasks_cancelled;
        if (controller_response.status == StatusCode::Ok) {
            response.pivot_count = context.request.pivot_dim;
            response.needs_pvector_write = true;
            response.pvector_dim = context.request.pivot_dim;
        }
        context.response_ready = true;
    }
}

bool PanelEngine::poll_response(std::uint32_t adapter_unit_id, PanelEngineResponse& out) {
    for (Context& context : contexts_) {
        if (!context.active || context.adapter_unit_id != adapter_unit_id ||
            !context.response_ready) {
            continue;
        }
        out = context.response;
        release_context(context);
        return true;
    }
    return false;
}

bool PanelEngine::unit_busy(std::uint32_t adapter_unit_id) const {
    for (const Context& context : contexts_) {
        if (context.active && context.adapter_unit_id == adapter_unit_id) {
            return true;
        }
    }
    return false;
}

void PanelEngine::abort_txn(std::uint64_t txn_id, std::uint64_t cycle) {
    Context* context = find_context_by_txn(txn_id);
    if (context == nullptr) {
        return;
    }
    context->aborting = true;
    context->controller->request_abort();
    const std::vector<TileTask> cancelled =
        dispatcher_.cancel_command(context->request.command_id);
    context->controller->on_tasks_cancelled(cancelled);
    for (const TileTask& task : cancelled) {
        EngineTaskRecord record;
        record.task = task;
        record.status = StatusCode::Aborted;
        record.reason = "cancelled before dispatch";
        record.cancelled = true;
        task_records_.push_back(std::move(record));
        ++tasks_cancelled_;
    }
    for (MultiModePe& pe : pes_) {
        pe.abort_command(context->request.command_id, cycle);
    }
}

void PanelEngine::release_context(Context& context) {
    if (context.controller) {
        context.controller->reset();
    }
    if (context.reader) {
        context.reader.reset();
    }
    access_.unbind(context.request.front.slot);
    erase_slot_versions(context.request.front.slot);
    context = Context{};
}

void PanelEngine::reset() {
    ++epoch_;
    versions_.clear();
    next_task_id_ = 1;
    for (MultiModePe& pe : pes_) {
        pe.reset();
    }
    dispatcher_.reset();
    for (Context& context : contexts_) {
        context = Context{};
    }
    writeback_reservation_.reset();
    read_budget_.reset(config_.load_bytes_per_cycle);
    write_budget_.reset(config_.writeback_bytes_per_cycle);
    task_records_.clear();
    tasks_committed_ = 0;
    tasks_cancelled_ = 0;
    control_fault_ = false;
    control_fault_reason_.clear();
    rotation_ = 0;
}

PanelEngine::Context* PanelEngine::find_context(std::uint32_t command_id) {
    for (Context& context : contexts_) {
        if (context.active && context.request.command_id == command_id) {
            return &context;
        }
    }
    return nullptr;
}

PanelEngine::Context* PanelEngine::find_context_by_txn(std::uint64_t txn_id) {
    for (Context& context : contexts_) {
        if (context.active && context.request.txn_id == txn_id) {
            return &context;
        }
    }
    return nullptr;
}

void PanelEngine::LeaseTileAccess::bind(std::uint32_t slot, const SlotBinding& binding) {
    bindings_[slot] = binding;
}

void PanelEngine::LeaseTileAccess::unbind(std::uint32_t slot) {
    bindings_.erase(slot);
}

bool PanelEngine::LeaseTileAccess::binding_for(
    std::uint32_t slot,
    SlotBinding* out) const {
    const auto found = bindings_.find(slot);
    if (found == bindings_.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = found->second;
    }
    return true;
}

bool PanelEngine::LeaseTileAccess::read_operand_segment(
    const TileOperand& operand,
    std::uint32_t row,
    std::uint32_t col_begin,
    std::uint32_t col_count,
    float* out_row,
    std::uint64_t cycle) {
    const auto found = bindings_.find(operand.handle.slot);
    if (found == bindings_.end() || !found->second.valid ||
        found->second.buffers == nullptr) {
        return false;
    }
    const SlotBinding& binding = found->second;
    if (cycle < binding.ready_cycle) {
        return false;
    }
    const std::uint64_t row_offset =
        binding.base_offset + operand.offset +
        static_cast<std::uint64_t>(row) * operand.stride_bytes +
        static_cast<std::uint64_t>(col_begin) * 4;
    std::vector<std::uint8_t> bytes;
    std::string reason;
    if (binding.buffers->lease_read(
            binding.lease_id,
            row_offset,
            static_cast<std::uint64_t>(col_count) * 4,
            bytes,
            cycle,
            &reason) != StatusCode::Ok) {
        return false;
    }
    for (std::uint32_t c = 0; c < col_count; ++c) {
        std::memcpy(
            out_row + col_begin + c,
            bytes.data() + static_cast<std::size_t>(c) * 4,
            4);
    }
    return true;
}

bool PanelEngine::LeaseTileAccess::write_result(
    const TileTask& task,
    const float* data,
    std::uint64_t cycle) {
    const auto found = bindings_.find(task.dst.handle.slot);
    if (found == bindings_.end() || !found->second.valid ||
        found->second.buffers == nullptr) {
        return false;
    }
    const SlotBinding& binding = found->second;
    if (cycle < binding.ready_cycle) {
        return false;
    }
    std::vector<std::uint8_t> row(static_cast<std::size_t>(task.dims.n) * 4);
    for (std::uint32_t r = 0; r < task.dims.m; ++r) {
        for (std::uint32_t c = 0; c < task.dims.n; ++c) {
            std::memcpy(&row[static_cast<std::size_t>(c) * 4], data + r * 16 + c, 4);
        }
        const std::uint64_t row_offset =
            binding.base_offset + task.dst.offset +
            static_cast<std::uint64_t>(r) * task.dst.stride_bytes;
        std::string reason;
        if (binding.buffers->lease_write(
                binding.lease_id, row_offset, row, cycle, &reason) != StatusCode::Ok) {
            return false;
        }
    }
    return true;
}

void PanelEngine::LeaseFrontReader::configure(
    std::uint32_t slot,
    std::uint64_t base_offset,
    std::uint32_t row_stride_bytes,
    std::uint32_t total_dim) {
    slot_ = slot;
    base_offset_ = base_offset;
    row_stride_bytes_ = row_stride_bytes;
    total_dim_ = total_dim;
    configured_ = true;
}

bool PanelEngine::LeaseFrontReader::read_front(
    std::uint64_t byte_offset,
    std::uint32_t length_bytes,
    std::uint8_t* out,
    std::uint64_t cycle) {
    if (!configured_ || total_dim_ == 0 || length_bytes % 4 != 0 ||
        byte_offset % 4 != 0) {
        return false;
    }
    SlotBinding binding;
    if (!access_.binding_for(slot_, &binding) || binding.buffers == nullptr) {
        return false;
    }
    if (cycle < binding.ready_cycle) {
        return false;
    }
    std::vector<std::uint8_t> bytes;
    std::string reason;
    if (binding.buffers->lease_read(
            binding.lease_id,
            base_offset_ + byte_offset,
            length_bytes,
            bytes,
            cycle,
            &reason) != StatusCode::Ok) {
        return false;
    }
    std::memcpy(out, bytes.data(), length_bytes);
    return true;
}

}  // namespace hw::panel_lu
