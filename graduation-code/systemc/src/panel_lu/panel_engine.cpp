#include "panel_lu/panel_engine.hpp"

#include <algorithm>
#include <cstring>

#include "panel_lu/precision_adapter.hpp"

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
    const std::uint64_t prefix = static_cast<std::uint64_t>(slot) << 48;
    auto iterator = versions_.lower_bound(prefix);
    while (iterator != versions_.end() && (iterator->first >> 48) == slot) {
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
    pes_.reserve(config_.total_pe_count());
    for (std::uint32_t id = 0; id < config_.total_pe_count(); ++id) {
        pes_.emplace_back(id, config_);
    }
    writeback_reservation_.configure(config_.writeback_queue_depth);
    fallback_arbiter_.reset();
    fallback_arbiter_.configure(
        config_.load_bytes_per_cycle, config_.writeback_bytes_per_cycle);
    contexts_.clear();
    contexts_.resize(std::max(config_.max_resident_commands,
                              config_.compute_group_count));
    group_busy_.assign(config_.compute_group_count, false);
    group_commands_.assign(config_.compute_group_count, hw::command::NONE);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    for (std::uint32_t group = 0; group < config_.compute_group_count; ++group) {
        ranges.emplace_back(
            group * config_.pes_per_group, config_.pes_per_group);
    }
    dispatcher_.set_group_ranges(ranges);
    configured_ = true;
    ConfigCheck result;
    result.ok = true;
    return result;
}

ConfigCheck PanelEngine::begin_command(
    const PanelEngineRequest& request,
    std::uint64_t adapter_key,
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
        check.reason = "no free compute context";
        return check;
    }
    // One accepted compute command owns one PE group for its whole execution
    // context (including e.g. the Panel scan): another command cannot share
    // the group even while it has idle PEs.
    std::uint32_t group_id = hw::command::NONE;
    for (std::uint32_t group = 0; group < group_busy_.size(); ++group) {
        if (!group_busy_[group]) {
            group_id = group;
            break;
        }
    }
    if (group_id == hw::command::NONE) {
        check.ok = false;
        check.status = StatusCode::BufferFull;
        check.reason = "no free PE group";
        return check;
    }
    group_busy_[group_id] = true;
    group_commands_[group_id] = request.command_id;
    free_context->group_id = group_id;
    free_context->request = request;
    free_context->adapter_key = adapter_key;
    free_context->bytes_read = 0;
    free_context->bytes_written = 0;
    free_context->aborting = false;
    free_context->response_ready = false;
    check = build_context(*free_context, cycle);
    if (!check.ok) {
        group_busy_[group_id] = false;
        group_commands_[group_id] = hw::command::NONE;
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
        check.reason = "DYNAMIC_PANEL is not supported by the STATIC engine";
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
    if (request.compute_kind == ComputeKind::TrsmLeft ||
        request.compute_kind == ComputeKind::TrsmRight) {
        // Strict KERNEL_DESC shape validation against the resident front:
        //   TRSM_LEFT  M/N/K = p/u/p
        //   TRSM_RIGHT M/N/K = u/p/p
        const std::uint32_t update = request.total_dim - request.pivot_dim;
        const bool shape_ok =
            request.compute_kind == ComputeKind::TrsmLeft
                ? (request.kernel_m == request.pivot_dim &&
                   request.kernel_n == update && request.kernel_k == request.pivot_dim)
                : (request.kernel_m == update &&
                   request.kernel_n == request.pivot_dim &&
                   request.kernel_k == request.pivot_dim);
        if (!shape_ok) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason = "TRSM kernel shape disagrees with the resident front";
            return check;
        }
        if (request.kernel_tile_size != config_.tile_dim) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason = "TRSM kernel tile_size must be the fixed 16x16 tile";
            return check;
        }
        if (request.kernel_backend != 0x01) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason =
                "TRSM kernel backend is not the FP32 device model; other "
                "backends are not implemented";
            return check;
        }
    }
    if (request.compute_kind == ComputeKind::GemmSchur) {
        // Strict KERNEL_DESC shape validation against the resident front:
        //   GEMM_SCHUR M/N/K = u/u/p
        const std::uint32_t update = request.total_dim - request.pivot_dim;
        if (request.kernel_m != update || request.kernel_n != update ||
            request.kernel_k != request.pivot_dim) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason = "GEMM_SCHUR kernel shape disagrees with the resident front";
            return check;
        }
        if (request.kernel_tile_size != config_.tile_dim) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason = "GEMM_SCHUR kernel tile_size must be the fixed 16x16 tile";
            return check;
        }
        const GemmBackend backend = gemm_backend_from_wire(request.kernel_backend);
        if (request.kernel_backend != gemm_backend_to_wire(GemmBackend::Fp32Device) &&
            request.kernel_backend != gemm_backend_to_wire(GemmBackend::Int32Gemm)) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason = "GEMM_SCHUR kernel backend code is unknown";
            return check;
        }
        if (backend == GemmBackend::Int32Gemm) {
            if (!request.kernel_has_scale) {
                check.ok = false;
                check.status = StatusCode::BadDescriptor;
                check.reason = "INT32 GEMM requires a valid SCALE_DESC reference";
                return check;
            }
            if (config_.numeric_mode != PanelNumericMode::Fp32Device) {
                check.ok = false;
                check.status = StatusCode::PolicyNotSupported;
                check.reason =
                    "INT32 GEMM cannot be combined with the FP64 diagnostic mode";
                return check;
            }
            const ConfigCheck precision = validate_precision_config(config_.precision);
            if (!precision.ok) {
                return precision;
            }
        } else if (request.kernel_has_scale) {
            check.ok = false;
            check.status = StatusCode::BadDescriptor;
            check.reason =
                "FP32 GEMM must not reference a SCALE_DESC (scale_desc_id must "
                "be NONE)";
            return check;
        }
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
    layout.group_id = context.group_id;

    PanelCommandRequest controller_request;
    controller_request.command_id = request.command_id;
    controller_request.node_id = request.node_id;
    controller_request.front_dim = request.total_dim;
    controller_request.pivot_dim = request.pivot_dim;
    controller_request.policy = request.policy;
    controller_request.threshold_bits = request.threshold_bits;
    controller_request.tile_size = request.tile_size;
    controller_request.label = request.label;
    controller_request.compute_kind = request.compute_kind;
    controller_request.front_pre_elimination_max_abs = request.front_scale;
    controller_request.threshold_product = request.threshold_product;
    controller_request.gemm_backend = gemm_backend_from_wire(request.kernel_backend);
    controller_request.gemm_scale = request.gemm_scale;
    controller_request.descriptor_id = request.kernel_descriptor_id;

    if (request.compute_kind == ComputeKind::PanelLu) {
        context.reader = std::make_unique<LeaseFrontReader>(access_);
        context.reader->configure(
            request.front.slot, request.front_offset, row_stride, request.total_dim);
        context.controller = std::make_unique<PanelController>(
            config_, &dispatcher_, this, context.reader.get());
    } else if (request.compute_kind == ComputeKind::GemmSchur) {
        context.controller = std::make_unique<GemmController>(
            config_, &dispatcher_, this);
    } else {
        context.controller = std::make_unique<TrsmController>(
            config_, &dispatcher_, this);
    }
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
    hw::dataplane::OnChipArbiter& arbiter =
        arbiter_ != nullptr ? *arbiter_ : fallback_arbiter_;
    if (arbiter_ == nullptr) {
        // Standalone engine (tests without a device owner): own the once-per-
        // cycle budget initialization here.
        arbiter.begin_cycle(cycle);
    }

    // Phase 1: PE progress (shared unified budgets).
    for (MultiModePe& pe : pes_) {
        pe.tick(cycle, arbiter, writeback_reservation_, access_);
    }
    // Phase 2: route terminal task responses (commits + controller updates).
    route_pe_responses(cycle);
    // Phase 3: dispatch ready tasks to the shared PE pool.
    dispatcher_.tick(cycle);
    // Phase 4: directed cancellation requested by controllers.
    handle_cancel_requests(cycle);
    // Phase 5: advance contexts (scan/generation), fair rotation.
    service_contexts(cycle, arbiter);
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
                record.group_id = context != nullptr ? context->group_id : 0;
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
                record.gemm_backend = response.gemm_backend;
                record.gemm_failed_phase = response.gemm_failed_phase;
                record.gemm_failed_local_row = response.gemm_failed_local_row;
                record.gemm_failed_local_col = response.gemm_failed_local_col;
                record.gemm_exponent_a = response.gemm_exponent_a;
                record.gemm_exponent_b = response.gemm_exponent_b;
                record.saturation_count = response.saturation_count;
                record.overflow_count = response.overflow_count;
                record.local_conversion_bytes = response.local_conversion_bytes;
                record.scale_reduction_cycles = response.scale_reduction_cycles;
                record.quantize_cycles = response.quantize_cycles;
                record.integer_compute_cycles = response.integer_compute_cycles;
                record.dequantize_cycles = response.dequantize_cycles;
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

void PanelEngine::service_contexts(
    std::uint64_t cycle,
    hw::dataplane::OnChipArbiter& arbiter) {
    const std::size_t count = contexts_.size();
    if (count == 0) {
        return;
    }
    for (std::size_t index = 0; index < count; ++index) {
        Context& context = contexts_[(rotation_ + index) % count];
        if (!context.active || !context.controller) {
            continue;
        }
        context.controller->tick(cycle, arbiter);
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
        response.group_id = context.group_id;
        response.front_scale = context.request.front_scale;
        response.threshold_product = context.request.threshold_product;
        if (context.request.compute_kind == ComputeKind::PanelLu) {
            if (auto* panel =
                    dynamic_cast<PanelController*>(context.controller.get());
                panel != nullptr) {
                response.front_scale = panel->front_max_abs();
                response.threshold_product = panel->threshold_product();
            }
            if (controller_response.status == StatusCode::Ok) {
                response.pivot_count = context.request.pivot_dim;
                response.needs_pvector_write = true;
                response.pvector_dim = context.request.pivot_dim;
            }
        } else if (context.request.compute_kind == ComputeKind::GemmSchur) {
            // GEMM_SCHUR: no P-vector and no pivots; pivot_count stays 0 even
            // on failure (the failing front coordinate is not a pivot).
            response.pivot_count = 0;
            response.gemm_backend = controller_response.gemm_backend;
            response.saturation_count = controller_response.saturation_count;
            response.overflow_count = controller_response.overflow_count;
            response.local_conversion_bytes =
                controller_response.local_conversion_bytes;
            response.scale_reduction_cycles =
                controller_response.scale_reduction_cycles;
            response.quantize_cycles = controller_response.quantize_cycles;
            response.integer_compute_cycles =
                controller_response.integer_compute_cycles;
            response.dequantize_cycles = controller_response.dequantize_cycles;
            response.exponent_a = controller_response.exponent_a;
            response.exponent_b = controller_response.exponent_b;
        } else {
            // TRSM: no P-vector; failure coordinates are front coordinates
            // mapped by the controller; pivot_count stays 0 on success.
            response.pivot_count = controller_response.status == StatusCode::Ok
                                       ? 0
                                       : controller_response.pivot_count;
        }
        context.response_ready = true;
    }
}

bool PanelEngine::poll_response(std::uint64_t adapter_key, PanelEngineResponse& out) {
    for (Context& context : contexts_) {
        if (!context.active || context.adapter_key != adapter_key ||
            !context.response_ready) {
            continue;
        }
        out = context.response;
        release_context(context);
        return true;
    }
    return false;
}

bool PanelEngine::unit_busy(std::uint64_t adapter_key) const {
    for (const Context& context : contexts_) {
        if (context.active && context.adapter_key == adapter_key) {
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

bool PanelEngine::group_busy(std::uint32_t group_id) const {
    return group_id < group_busy_.size() && group_busy_[group_id];
}

std::uint32_t PanelEngine::group_of_command(std::uint32_t command_id) const {
    for (const Context& context : contexts_) {
        if (context.active && context.request.command_id == command_id) {
            return context.group_id;
        }
    }
    return hw::command::NONE;
}

void PanelEngine::release_context(Context& context) {
    if (context.group_id < group_busy_.size()) {
        group_busy_[context.group_id] = false;
        group_commands_[context.group_id] = hw::command::NONE;
    }
    if (context.controller) {
        // Terminal release: drop this command's committed-task bookkeeping so
        // long runs do not accumulate task ids forever.
        dispatcher_.release_command(context.request.command_id);
        if (auto* gemm = dynamic_cast<GemmController*>(context.controller.get());
            gemm != nullptr) {
            EngineGemmLog log;
            log.requests = gemm->gemm_requests();
            log.responses = gemm->gemm_responses();
            gemm_log_.push_back(std::move(log));
        }
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
    for (std::size_t group = 0; group < group_busy_.size(); ++group) {
        group_busy_[group] = false;
        group_commands_[group] = hw::command::NONE;
    }
    writeback_reservation_.reset();
    fallback_arbiter_.reset();
    fallback_arbiter_.configure(
        config_.load_bytes_per_cycle, config_.writeback_bytes_per_cycle);
    task_records_.clear();
    gemm_log_.clear();
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

PanelEngine::Context* PanelEngine::find_context_by_adapter(std::uint64_t adapter_key) {
    for (Context& context : contexts_) {
        if (context.active && context.adapter_key == adapter_key) {
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
