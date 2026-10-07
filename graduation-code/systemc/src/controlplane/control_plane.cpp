#include "controlplane/control_plane.hpp"

#include "controlplane/panel_exec_unit.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

const char* batch_state_name(BatchState state) {
    switch (state) {
    case BatchState::Idle: return "idle";
    case BatchState::Running: return "running";
    case BatchState::Completed: return "completed";
    case BatchState::Faulted: return "faulted";
    }
    return "unknown";
}

namespace {

std::string csv_field(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (const char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    return escaped + '"';
}

bool opcode_touches_front(cmd::Opcode opcode) {
    switch (opcode) {
    case cmd::Opcode::NodeBegin:
    case cmd::Opcode::LoadFront:
    case cmd::Opcode::AssembleExtendAdd:
    case cmd::Opcode::PanelLu:
    case cmd::Opcode::TrsmLeft:
    case cmd::Opcode::TrsmRight:
    case cmd::Opcode::GemmSchur:
    case cmd::Opcode::StoreFactor:
    case cmd::Opcode::StoreUpdate:
        return true;
    default:
        return false;
    }
}

bool opcode_needs_view(cmd::Opcode opcode) {
    switch (opcode) {
    case cmd::Opcode::NodeBegin:
    case cmd::Opcode::LoadFront:
    case cmd::Opcode::PanelLu:
    case cmd::Opcode::AssembleExtendAdd:
    case cmd::Opcode::TrsmLeft:
    case cmd::Opcode::TrsmRight:
    case cmd::Opcode::GemmSchur:
    case cmd::Opcode::StoreFactor:
    case cmd::Opcode::StoreUpdate:
    case cmd::Opcode::SolveForward:
    case cmd::Opcode::SolveBackward:
        return true;
    case cmd::Opcode::NodeCommit:
    case cmd::Opcode::AbortNode:
        return false;
    }
    return false;
}

dp::ReadKind view_kind_for(cmd::Opcode opcode) {
    switch (opcode) {
    case cmd::Opcode::NodeBegin:
    case cmd::Opcode::LoadFront:
    case cmd::Opcode::PanelLu:
        return dp::ReadKind::Front;
    case cmd::Opcode::AssembleExtendAdd:
        return dp::ReadKind::Contribution;
    case cmd::Opcode::TrsmLeft:
    case cmd::Opcode::TrsmRight:
    case cmd::Opcode::GemmSchur:
        return dp::ReadKind::Kernel;
    case cmd::Opcode::StoreFactor:
        return dp::ReadKind::Factor;
    case cmd::Opcode::StoreUpdate:
        return dp::ReadKind::Region;
    case cmd::Opcode::SolveForward:
    case cmd::Opcode::SolveBackward:
        return dp::ReadKind::Solve;
    case cmd::Opcode::NodeCommit:
    case cmd::Opcode::AbortNode:
        break;
    }
    return dp::ReadKind::Region;
}

// Unit kind required by an opcode, if it needs an execution unit at all.
std::optional<UnitKind> required_kind_for(hw::command::Opcode opcode) {
    switch (opcode) {
    case hw::command::Opcode::AssembleExtendAdd:
        return UnitKind::Assemble;
    case hw::command::Opcode::PanelLu:
        return UnitKind::Panel;
    case hw::command::Opcode::TrsmLeft:
    case hw::command::Opcode::TrsmRight:
        return UnitKind::Trsm;
    case hw::command::Opcode::GemmSchur:
        return UnitKind::Gemm;
    case hw::command::Opcode::SolveForward:
    case hw::command::Opcode::SolveBackward:
        return UnitKind::Solve;
    default:
        return std::nullopt;
    }
}

std::vector<std::uint8_t> identity_pvector(std::uint32_t dim) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(dim) * 4);
    for (std::uint32_t index = 0; index < dim; ++index) {
        for (unsigned b = 0; b < 4; ++b) {
            bytes[static_cast<std::size_t>(index) * 4 + b] =
                static_cast<std::uint8_t>((index >> (8 * b)) & 0xFF);
        }
    }
    return bytes;
}

}  // namespace

CommandDevice::CommandDevice(dp::LoadedArtifact artifact, ControlPlaneConfig config)
    : artifact_(std::move(artifact)),
      config_(config),
      plane_(
          artifact_,
          [&config]() {
              dp::DataPlaneConfig data_config;
              data_config.memory = config.memory;
              data_config.buffers.slot_count = config.buffer_slot_count;
              data_config.buffers.capacity_bytes = config.buffer_capacity_bytes;
              data_config.buffers.port_count = config.buffer_port_count;
              data_config.buffers.access_cycles = 1;
              data_config.dma = config.dma;
              return data_config;
          }()),
      trace_(plane_.trace()),
      fetch_(
          plane_.memory_service(),
          CommandFetch::Config{
              plane_.boot().command_table_addr, plane_.boot().command_count,
              config.command_window_depth + 2},
          &trace_),
      completion_writer_(
          plane_.memory_service(),
          plane_.boot().completion_table_addr,
          config.completion_queue_depth,
          &trace_),
      dma_arbiter_(plane_.dma(), config.dma_queue_depth),
      total_commands_(plane_.boot().command_count) {
    plane_.set_memory_sink(dp::MemoryClient::CommandFetch, &fetch_);
    plane_.set_memory_sink(dp::MemoryClient::CompletionWriter, &completion_writer_);
    plane_.set_memory_sink(dp::MemoryClient::ExecUnit, this);

    if (artifact_.config.gemm_model.has_value()) {
        const auto& model = *artifact_.config.gemm_model;
        hw::panel_lu::PrecisionConfig declared;
        declared.effective_bits = model.effective_bits;
        declared.accumulator_bits = model.accumulator_bits;
        declared.k_block_max = model.k_block_max;
        declared.maxabs_elements_per_cycle = model.conversion_rate_per_cycle;
        declared.quantize_elements_per_cycle = model.conversion_rate_per_cycle;
        declared.dequantize_elements_per_cycle = model.conversion_rate_per_cycle;
        const auto equal = [](const hw::panel_lu::PrecisionConfig& a,
                              const hw::panel_lu::PrecisionConfig& b) {
            return a.effective_bits == b.effective_bits &&
                   a.accumulator_bits == b.accumulator_bits &&
                   a.k_block_max == b.k_block_max &&
                   a.maxabs_elements_per_cycle == b.maxabs_elements_per_cycle &&
                   a.quantize_elements_per_cycle == b.quantize_elements_per_cycle &&
                   a.dequantize_elements_per_cycle == b.dequantize_elements_per_cycle;
        };
        // Default device settings inherit the artifact declaration. Explicit
        // non-default settings must agree; they never silently override it.
        if (!equal(config_.panel.precision, hw::panel_lu::PrecisionConfig{}) &&
            !equal(config_.panel.precision, declared)) {
            model_config_error_ =
                "device precision configuration conflicts with manifest gemm model parameters";
        } else {
            config_.panel.precision = declared;
        }
    }

    // Device-wide unified on-chip bandwidth: one arbiter, one initialization
    // per cycle, shared by the panel engine, factor units and DMA.
    on_chip_.configure(
        config_.panel.load_bytes_per_cycle, config_.panel.writeback_bytes_per_cycle);
    plane_.set_on_chip_arbiter(&on_chip_);

    if (config_.enable_real_panel) {
        for (std::uint32_t index = 0; index < config_.assemble_unit_count; ++index) {
            units_.push_back(std::make_unique<AssemblyExecutionUnit>(
                index, &plane_.buffers(), &on_chip_, config_.assembly));
        }
        panel_engine_ = std::make_unique<hw::panel_lu::PanelEngine>(config_.panel);
        panel_engine_->set_buffers(&plane_.buffers());
        panel_engine_->set_arbiter(&on_chip_);
        for (std::uint32_t index = 0; index < config_.panel_unit_count; ++index) {
            units_.push_back(
                std::make_unique<PanelExecutionUnit>(index, panel_engine_.get()));
        }
        for (std::uint32_t index = 0; index < config_.trsm_unit_count; ++index) {
            units_.push_back(
                std::make_unique<TrsmExecutionUnit>(index, panel_engine_.get()));
        }
        for (std::uint32_t index = 0; index < config_.gemm_unit_count; ++index) {
            units_.push_back(
                std::make_unique<GemmExecutionUnit>(index, panel_engine_.get()));
        }
        for (std::uint32_t index = 0; index < config_.factor_unit_count; ++index) {
            auto factor = std::make_unique<FactorStoreUnit>(
                index, &plane_.buffers(), &plane_.memory_service());
            factor->set_on_chip_arbiter(&on_chip_);
            units_.push_back(std::move(factor));
        }
    }
    if (config_.enable_test_stub) {
        auto make_units = [this](UnitKind kind, std::uint32_t count) {
            for (std::uint32_t index = 0; index < count; ++index) {
                units_.push_back(std::make_unique<TestStubUnit>(
                    kind, index, config_.stub, &plane_.buffers()));
            }
        };
        if (!config_.enable_real_panel) {
            make_units(UnitKind::Assemble, config_.assemble_unit_count);
        }
        make_units(UnitKind::Panel, config_.panel_unit_count);
        make_units(UnitKind::Trsm, config_.trsm_unit_count);
        make_units(UnitKind::Gemm, config_.gemm_unit_count);
        make_units(UnitKind::Solve, config_.solve_unit_count);
    }

    front_slots_.resize(config_.max_resident_fronts);
    scoreboard_.init(plane_.boot().token_count);
}

CommandDevice::~CommandDevice() = default;

bool CommandDevice::start(std::string* error) {
    if (!model_config_error_.empty()) {
        if (error != nullptr) {
            *error = model_config_error_;
        }
        return false;
    }
    if (config_.backend == ExecBackend::RequireRealKernels &&
        !config_.enable_real_panel) {
        if (error != nullptr) {
            *error =
                "RequestRealKernels was requested but no real kernel backend is "
                "enabled; refusing to run instead of silently using a test stub";
        }
        return false;
    }
    if (!config_.enable_test_stub && !config_.enable_real_panel) {
        if (error != nullptr) {
            *error =
                "an execution backend must be explicitly enabled "
                "(enable_test_stub and/or enable_real_panel)";
        }
        return false;
    }
    if (config_.enable_real_panel) {
        const hw::panel_lu::ConfigCheck panel_check = panel_engine_->configure();
        if (!panel_check.ok) {
            if (error != nullptr) {
                *error = "panel engine configuration rejected: " + panel_check.reason;
            }
            return false;
        }
    }
    // Resource configurations that can never make progress are refused up
    // front instead of deadlocking at runtime.
    auto reject = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (config_.max_resident_fronts < 1 || config_.max_resident_fronts > 2) {
        return reject("max_resident_fronts must be 1 or 2 in this version");
    }
    if (config_.command_window_depth < 1) {
        return reject("command_window_depth must be at least 1");
    }
    if (config_.completion_queue_depth < 1) {
        return reject("completion_queue_depth must be at least 1");
    }
    if (config_.dma_queue_depth < 2) {
        // STORE_FACTOR submits its L and U write-backs as one atomic pair.
        return reject("dma_queue_depth must be at least 2 (atomic job pairs)");
    }
    if (config_.buffer_capacity_bytes == 0) {
        return reject("buffer_capacity_bytes must be positive");
    }
    if (config_.buffer_port_count < 2) {
        // ASSEMBLE holds a front write lease and a temp read lease at the same
        // time; with a single port that need can never be satisfied, so the
        // configuration is refused instead of stalling forever.
        return reject(
            "buffer_port_count must be at least 2 (assembly holds two buffer "
            "leases at once)");
    }
    const std::uint32_t required_slots =
        2 * config_.max_resident_fronts + 1;  // front+temp per front + solve scratch
    if (config_.buffer_slot_count < required_slots) {
        return reject(
            "buffer_slot_count must be at least 2*max_resident_fronts+1 "
            "(front+temp per resident front and one solve scratch slot)");
    }
    if (config_.enable_real_panel && config_.panel_unit_count < 1) {
        return reject("the real panel backend needs at least one panel unit slot");
    }
    if (config_.enable_real_panel && config_.factor_unit_count < 1) {
        return reject("the real panel backend needs at least one factor unit");
    }
    if (config_.enable_real_panel &&
        (config_.assemble_unit_count < 1 || config_.assembly.add_latency_cycles < 1)) {
        return reject("real assembly needs a unit and positive FP32 add latency");
    }
    if (config_.enable_test_stub &&
        (config_.panel_unit_count < 1 || config_.trsm_unit_count < 1 ||
         config_.gemm_unit_count < 1 || config_.solve_unit_count < 1 ||
         config_.assemble_unit_count < 1)) {
        return reject("every test stub execution unit pool needs at least one unit");
    }
    state_ = BatchState::Running;
    return true;
}

void CommandDevice::fault(const std::string& reason, std::uint64_t cycle) {
    (void)cycle;
    if (state_ == BatchState::Faulted) {
        return;
    }
    state_ = BatchState::Faulted;
    fault_reason_ = reason;
}

bool CommandDevice::finished() const {
    return state_ != BatchState::Running;
}

void CommandDevice::tick() {
    if (state_ != BatchState::Running) {
        return;
    }
    const std::uint64_t cycle = plane_.begin_tick();
    // Exactly one global budget initialization per cycle.
    on_chip_.begin_cycle(cycle);
    plane_.tick_memory_and_route();
    update_phase(cycle);
    if (state_ == BatchState::Running) {
        dispatch_phase(cycle);
        issue_phase(cycle);
    }
    plane_.tick_clients();

    if (state_ == BatchState::Running && fetch_.finished() &&
        completed_commands_ == total_commands_) {
        bool quiescent = true;
        for (const auto& tracker : trackers_) {
            quiescent = quiescent && tracker->phase == Tracker::Phase::Done;
        }
        // Shared engine / unit-owned memory transactions must be quiescent:
        // an aborted panel can still be draining its PE tasks, and a factor
        // unit can still have accepted DDR writes in flight.
        bool units_quiescent = true;
        for (const auto& unit : units_) {
            if (unit->busy()) {
                units_quiescent = false;
                break;
            }
        }
        if (quiescent && units_quiescent && dma_arbiter_.idle() &&
            exec_inbox_.empty()) {
            state_ = BatchState::Completed;
        }
    }
}

void CommandDevice::run(std::uint64_t max_cycles) {
    const std::uint64_t start = plane_.cycle();
    while (state_ == BatchState::Running && plane_.cycle() - start < max_cycles) {
        tick();
    }
    if (state_ == BatchState::Running) {
        fault("batch timeout", plane_.cycle());
    }
}

CommandDevice::Tracker* CommandDevice::find_tracker(std::uint32_t command_id) {
    for (auto& tracker : trackers_) {
        if (tracker->command.command_id == command_id) {
            return tracker.get();
        }
    }
    return nullptr;
}

CommandDevice::Tracker* CommandDevice::find_tracker_by_txn(std::uint64_t txn_id) {
    const auto found = txn_map_.find(txn_id);
    return found == txn_map_.end() ? nullptr : find_tracker(found->second);
}

CommandDevice::Tracker* CommandDevice::find_tracker_by_job(std::uint64_t job_id) {
    const auto found = job_map_.find(job_id);
    return found == job_map_.end() ? nullptr : find_tracker(found->second);
}

CommandDevice::Tracker* CommandDevice::find_tracker_by_pvector(std::uint64_t request_id) {
    const auto found = pvector_map_.find(request_id);
    return found == pvector_map_.end() ? nullptr : find_tracker(found->second);
}

dp::BufferHandle CommandDevice::front_slot(std::uint32_t node_id) const {
    const auto index = context_for_node(node_id);
    if (!index.has_value() || !front_slots_[*index].active) {
        return dp::BufferHandle{cmd::NONE, 0};
    }
    return front_slots_[*index].front;
}

std::size_t CommandDevice::unit_pending_request_count(
    UnitKind kind,
    std::uint32_t unit_id) const {
    for (const auto& unit : units_) {
        if (unit->kind() == kind && unit->unit_id() == unit_id) {
            return unit->pending_request_count();
        }
    }
    return 0;
}

std::uint32_t CommandDevice::unit_active_command_id(
    UnitKind kind,
    std::uint32_t unit_id) const {
    for (const auto& unit : units_) {
        if (unit->kind() == kind && unit->unit_id() == unit_id) {
            return unit->active_command_id();
        }
    }
    return cmd::NONE;
}

std::optional<std::uint32_t> CommandDevice::context_for_node(std::uint32_t node_id) const {
    for (std::uint32_t index = 0; index < front_slots_.size(); ++index) {
        if (front_slots_[index].active && front_slots_[index].node_id == node_id) {
            return index;
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Phase 3: module updates and command state transitions.
// ---------------------------------------------------------------------------
void CommandDevice::update_phase(std::uint64_t cycle) {
    fetch_.update(cycle);
    if (fetch_.faulted()) {
        fault("command fetch fault: " + fetch_.fault_reason(), cycle);
        return;
    }

    completion_writer_.update(cycle);
    if (completion_writer_.faulted()) {
        fault("completion fault: " + completion_writer_.fault_reason(), cycle);
    }
    drain_completion_results(cycle);

    dma_arbiter_.update(cycle);
    DmaArbiter::Completion job;
    while (dma_arbiter_.take_completion(job)) {
        const auto mapping = job_map_.find(job.job_id);
        if (mapping == job_map_.end()) {
            continue;
        }
        Tracker* tracker = find_tracker(mapping->second);
        job_map_.erase(mapping);
        if (tracker != nullptr) {
            handle_job_done(*tracker, job, cycle);
        }
    }

    if (panel_engine_ != nullptr) {
        // The shared compute subsystem is ticked exactly once per cycle, no
        // matter how many panel adapter slots exist.
        panel_engine_->tick(cycle);
    }
    for (auto& unit : units_) {
        unit->tick(cycle);
        ExecResponse response;
        if (unit->poll_response(response)) {
            Tracker* tracker = find_tracker_by_txn(response.txn_id);
            txn_map_.erase(response.txn_id);
            const auto record = open_unit_records_.find(response.txn_id);
            if (record != open_unit_records_.end()) {
                UnitBusyRecord busy = record->second;
                busy.finish = response.finish_cycle;
                unit_records_.push_back(busy);
                open_unit_records_.erase(record);
            }
            if (tracker != nullptr) {
                handle_exec_response(*tracker, response, cycle);
            }
        }
    }

    // Descriptor and dependency-payload resolution completions.
    std::vector<std::uint64_t> resolve_ids;
    for (auto& tracker : trackers_) {
        if (tracker->phase == Tracker::Phase::Resolving) {
            if (tracker->resolve_pending) {
                resolve_ids.push_back(tracker->resolve_request_id);
            }
            if (tracker->wait_request_id != 0) {
                resolve_ids.push_back(tracker->wait_request_id);
            }
        }
    }
    for (const auto request_id : resolve_ids) {
        if (plane_.reader().pending(request_id)) {
            continue;
        }
        dp::ReadOutcome outcome;
        if (!plane_.reader().poll(request_id, outcome)) {
            continue;
        }
        for (auto& tracker : trackers_) {
            if (tracker->phase != Tracker::Phase::Resolving) {
                continue;
            }
            if (tracker->resolve_pending && tracker->resolve_request_id == request_id) {
                tracker->resolve_pending = false;
                if (outcome.status != cmd::StatusCode::Ok) {
                    finish_command(
                        *tracker, outcome.status,
                        "descriptor resolution failed: " + outcome.reason, cycle);
                    break;
                }
                switch (outcome.kind) {
                case dp::ReadKind::Front: tracker->front_view = outcome.front; break;
                case dp::ReadKind::Contribution:
                    tracker->contribution_view = outcome.contribution;
                    break;
                case dp::ReadKind::Kernel: tracker->kernel_view = outcome.kernel; break;
                case dp::ReadKind::Factor: tracker->factor_view = outcome.factor; break;
                case dp::ReadKind::Solve: tracker->solve_view = outcome.solve; break;
                case dp::ReadKind::Region: tracker->region_view = outcome.region; break;
                case dp::ReadKind::Dependency: break;
                }
                tracker->view_ready = true;
            } else if (tracker->wait_request_id == request_id) {
                tracker->wait_request_id = 0;
                if (outcome.status != cmd::StatusCode::Ok) {
                    finish_command(
                        *tracker, outcome.status,
                        "dependency payload read failed: " + outcome.reason, cycle);
                    break;
                }
                tracker->waits = outcome.dependency.tokens;
                tracker->waits_ready = true;
            }
            if (!tracker->resolve_pending && tracker->waits_ready &&
                tracker->phase == Tracker::Phase::Resolving) {
                tracker->phase = Tracker::Phase::Waiting;
            }
        }
    }

    // P-vector identity writes (submit retries and completions).
    for (auto& tracker : trackers_) {
        if (tracker->pvector_pending && tracker->pvector_request_id == 0 &&
            tracker->phase == Tracker::Phase::PostExec) {
            submit_pvector_write(*tracker, cycle);
        }
    }
    for (auto iterator = exec_inbox_.begin(); iterator != exec_inbox_.end();) {
        Tracker* tracker = find_tracker_by_pvector(iterator->first);
        pvector_map_.erase(iterator->first);
        const auto response = std::move(iterator->second);
        iterator = exec_inbox_.erase(iterator);
        if (tracker != nullptr) {
            handle_pvector_done(*tracker, response, cycle);
        }
    }

    // Waiting trackers: dependency verdicts and resource retries. Commands of
    // an aborted node are cancelled first so abort semantics stay
    // deterministic: their status is always ABORTED and their token FAILED.
    for (auto& tracker : trackers_) {
        if (tracker->phase == Tracker::Phase::Waiting) {
            if (aborted_nodes_.count(tracker->command.node_id) != 0 &&
                tracker->command.opcode != cmd::Opcode::AbortNode) {
                finish_command(
                    *tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
                continue;
            }
            switch (scoreboard_.check(tracker->waits)) {
            case TokenScoreboard::Deps::Failed:
                finish_command(
                    *tracker, cmd::StatusCode::DependencyFailed,
                    "a dependency command failed", cycle);
                break;
            case TokenScoreboard::Deps::Pending:
                ++tracker->stall_dep;  // exclusive per-tick wait class
                break;
            case TokenScoreboard::Deps::Ready:
                try_execute(*tracker, cycle);
                break;
            }
        } else if (tracker->phase == Tracker::Phase::Running) {
            while (!tracker->job_retries.empty()) {
                dp::DmaRequest retry = tracker->job_retries.front();
                retry.request_id = plane_.memory_service().next_request_id();
                if (requeue_dma_job(*tracker, retry, cycle) == 0) {
                    break;  // queue still busy; retry next cycle
                }
                tracker->job_retries.erase(tracker->job_retries.begin());
            }
            if (tracker->solve_load_pending) {
                issue_solve_load(*tracker, cycle);
            } else if (tracker->stage2_pending) {
                try_execute(*tracker, cycle);
            } else if (tracker->solve_store_pending) {
                issue_solve_store(*tracker, cycle);
            }
        } else if (tracker->phase == Tracker::Phase::PostExec &&
                   !tracker->pvector_pending) {
            // Completion enqueue retries while the output writes are done.
            enqueue_completion(*tracker, cycle);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: bounded window management and dispatch attempts.
// ---------------------------------------------------------------------------
void CommandDevice::dispatch_phase(std::uint64_t cycle) {
    // Pull new commands into the bounded scheduler window. Rules (documented
    // in the T05 design note):
    //   * front-touching commands of a node that is not yet admitted stay in
    //     the fetch FIFO (admission gate) and do NOT consume window space;
    //   * a NODE_BEGIN of an un-admitted node is "parked": tracked, not
    //     counted against the window, so an un-admitted node can never block
    //     the window of an active front;
    //   * everything else counts against command_window_depth (pre-dispatch
    //     only; running commands are not part of the scheduling window).
    // The fetch FIFO is scanned for the first pullable entry, so a gated entry
    // cannot jam the commands behind it.
    while (!fetch_.window_empty()) {
        std::size_t counted = 0;
        for (const auto& tracker : trackers_) {
            const bool pre_dispatch =
                tracker->phase == Tracker::Phase::InWindow ||
                tracker->phase == Tracker::Phase::Resolving ||
                tracker->phase == Tracker::Phase::Waiting;
            const bool parked =
                tracker->command.opcode == cmd::Opcode::NodeBegin &&
                context_for_node(tracker->command.node_id) == std::nullopt &&
                aborted_nodes_.count(tracker->command.node_id) == 0;
            if (pre_dispatch && !parked) {
                ++counted;
            }
        }

        // A node whose NODE_BEGIN terminated without creating a context will
        // never be admitted; its remaining commands must still stream through
        // (they end as DEPENDENCY_FAILED / ABORTED).
        auto admission_closed = [this](std::uint32_t node_id) {
            for (const auto& tracker : trackers_) {
                if (tracker->command.node_id == node_id &&
                    tracker->command.opcode == cmd::Opcode::NodeBegin &&
                    tracker->finish_cycle != 0 &&
                    !context_for_node(node_id).has_value()) {
                    return true;
                }
            }
            return false;
        };

        std::optional<std::size_t> pick;
        for (std::size_t index = 0; index < fetch_.window_size(); ++index) {
            const auto& command = fetch_.at(index);
            const bool aborted = aborted_nodes_.count(command.node_id) != 0;
            const bool admitted =
                context_for_node(command.node_id).has_value() || aborted ||
                admission_closed(command.node_id);
            const bool touches = opcode_touches_front(command.opcode);
            if (touches && command.opcode != cmd::Opcode::NodeBegin && !admitted) {
                continue;  // admission gate
            }
            const bool parked_entry =
                command.opcode == cmd::Opcode::NodeBegin && !admitted;
            if (!parked_entry && counted >= config_.command_window_depth) {
                continue;  // window full for counted entries
            }
            pick = index;
            break;
        }
        if (!pick.has_value()) {
            break;
        }
        auto tracker = std::make_unique<Tracker>();
        tracker->command = fetch_.at(*pick);
        fetch_.remove_at(*pick);
        tracker->fetch_cycle = cycle;
        tracker->admit_cycle = cycle;
        tracker->phase = Tracker::Phase::InWindow;
        if (tracker->command.signal_token != cmd::NONE) {
            scoreboard_.register_producer(
                tracker->command.signal_token, tracker->command.command_id);
        }
        if (tracker->command.opcode == cmd::Opcode::NodeCommit) {
            commit_seen_.insert(tracker->command.node_id);
        }
        trackers_.push_back(std::move(tracker));
    }

    // Start descriptor/dependency reads for fresh window entries in stream
    // order (earliest first).
    for (auto& tracker : trackers_) {
        if (tracker->phase != Tracker::Phase::InWindow) {
            continue;
        }
        tracker->phase = Tracker::Phase::Resolving;
        tracker->waits_ready = tracker->command.wait_list_id == cmd::NONE;
        if (!tracker->waits_ready) {
            tracker->wait_request_id = plane_.reader().request(
                dp::ReadKind::Dependency, tracker->command.wait_list_id, cycle);
        }
        if (opcode_needs_view(tracker->command.opcode)) {
            tracker->resolve_pending = true;
            tracker->resolve_request_id = plane_.reader().request(
                view_kind_for(tracker->command.opcode),
                tracker->command.descriptor_id, cycle);
        } else {
            tracker->view_ready = true;
        }
    }
}

void CommandDevice::issue_phase(std::uint64_t cycle) {
    fetch_.tick(cycle);
    completion_writer_.tick(cycle);
    dma_arbiter_.tick(cycle);
}

// ---------------------------------------------------------------------------
// Execution attempts (atomic resource acquisition with rollback).
// ---------------------------------------------------------------------------
bool CommandDevice::unit_bound(UnitKind kind) const {
    for (const auto& candidate : units_) {
        if (candidate->kind() == kind) {
            return true;
        }
    }
    return false;
}

IExecutionUnit* CommandDevice::find_free_unit(UnitKind kind) {
    for (auto& candidate : units_) {
        if (candidate->kind() == kind && !candidate->busy()) {
            return candidate.get();
        }
    }
    return nullptr;
}

bool CommandDevice::acquire_exec_resources(
    Tracker& tracker,
    UnitKind kind,
    bool need_front_write,
    bool need_temp_read,
    std::uint64_t cycle) {
    const auto context_index = context_for_node(tracker.command.node_id);
    if ((need_front_write || need_temp_read) && !context_index.has_value()) {
        finish_command(
            tracker, cmd::StatusCode::BadCommand,
            "no resident front for this command", cycle);
        return false;
    }

    if (need_front_write) {
        const auto& context = front_slots_[*context_index];
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(context.front_view.total_dim) *
            context.front_view.total_dim * 4;
        const auto result =
            plane_.buffers().begin_write(context.front, context.owner, 0, bytes, cycle);
        if (!result.ok) {
            ++tracker.stall_resource;
            return false;
        }
        tracker.front_lease = result.lease;
        tracker.has_front_lease = true;
        tracker.front_write_lease = true;
    }
    if (need_temp_read) {
        const auto& context = front_slots_[*context_index];
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(tracker.contribution_view.col_count) *
            tracker.contribution_view.col_count * 4;
        const auto result = plane_.buffers().begin_read(
            context.temp, context.owner, 0, bytes, cycle);
        if (!result.ok) {
            if (tracker.has_front_lease) {
                plane_.buffers().end_write(
                    tracker.front_lease.lease_id, tracker.front_lease.length, cycle);
                tracker.has_front_lease = false;
                tracker.front_write_lease = false;
            }
            ++tracker.stall_resource;
            return false;
        }
        tracker.temp_lease = result.lease;
        tracker.has_temp_lease = true;
    }

    IExecutionUnit* unit = find_free_unit(kind);
    if (unit == nullptr) {
        bool kind_present = false;
        for (const auto& candidate : units_) {
            if (candidate->kind() == kind) {
                kind_present = true;
                break;
            }
        }
        release_command_resources(tracker, cycle);
        if (!kind_present) {
            // No unit of this kind is bound: the backend must refuse instead
            // of stalling forever or silently substituting the test stub.
            finish_command(
                tracker, cmd::StatusCode::PolicyNotSupported,
                std::string(unit_kind_name(kind)) +
                    " kernel is not implemented/bound in this backend",
                cycle);
            return false;
        }
        ++tracker.stall_resource;
        return false;
    }

    ExecRequest request;
    request.txn_id = next_txn_id_++;
    request.epoch = epoch_;
    request.command_id = tracker.command.command_id;
    request.node_id = tracker.command.node_id;
    request.opcode = tracker.command.opcode;
    request.unit_id = unit->unit_id();
    // Front-derived fields always come from the RESIDENT front context: a
    // KERNEL command's own descriptor does not carry policy/pivot/layout, and
    // its defaults must never be mistaken for real request parameters.
    request.front = front_slots_[*context_index].front_view;
    request.factor = tracker.factor_view;
    request.contribution = tracker.contribution_view;
    request.kernel = tracker.kernel_view;
    request.solve = tracker.solve_view;
    request.front_slot = front_slots_[*context_index].front;
    request.temp_slot = front_slots_[*context_index].temp;
    request.total_dim = front_slots_[*context_index].front_view.total_dim;
    // update_dim keeps the legacy meaning for assemble/solve staging (child
    // update block width); TRSM block shapes come from KERNEL_DESC + the
    // resident front in the engine adapter.
    request.update_dim = tracker.contribution_view.col_count;
    request.front_lease = tracker.front_lease;
    request.temp_lease = tracker.temp_lease;
    request.has_front_lease = tracker.has_front_lease;
    request.has_temp_lease = tracker.has_temp_lease;
    request.range_bytes =
        static_cast<std::uint64_t>(request.total_dim) * request.total_dim * 4;
    request.policy = front_slots_[*context_index].front_view.policy;
    request.threshold_bits = front_slots_[*context_index].front_view.threshold_bits;
    // Latched numeric context from the front slot (set by PANEL_LU); TRSM
    // diagonal checks must not re-derive the scale from the eliminated front.
    request.front_scale = front_slots_[*context_index].front_scale;
    request.threshold_product = front_slots_[*context_index].threshold_product;

    if (!unit->submit(request, cycle)) {
        release_command_resources(tracker, cycle);
        ++tracker.stall_resource;
        return false;
    }
    tracker.unit_kind = kind;
    tracker.unit_id = unit->unit_id();
    tracker.unit_active = true;
    tracker.txn_id = request.txn_id;
    txn_map_[request.txn_id] = tracker.command.command_id;
    UnitBusyRecord busy;
    busy.kind = unit_kind_name(kind);
    busy.unit_id = unit->unit_id();
    busy.command_id = tracker.command.command_id;
    busy.node_id = tracker.command.node_id;
    busy.start = cycle;
    open_unit_records_[request.txn_id] = busy;
    if (tracker.start_cycle == 0) {
        tracker.start_cycle = cycle;
    }
    tracker.phase = Tracker::Phase::Running;
    tracker.stage2_pending = false;
    return true;
}

std::uint64_t CommandDevice::enqueue_dma_job(
    Tracker& tracker,
    const dp::DmaRequest& request,
    std::uint64_t cycle) {
    DmaArbiter::Job job;
    job.front_tag = tracker.command.node_id;
    job.command_id = tracker.command.command_id;
    job.node_id = tracker.command.node_id;
    job.request = request;
    const std::uint64_t job_id = dma_arbiter_.enqueue(std::move(job), cycle);
    if (job_id == 0) {
        ++tracker.stall_dma;
        return 0;
    }
    job_map_[job_id] = tracker.command.command_id;
    ++tracker.dma_jobs_total;  // issued logical jobs (cancelled ones tracked apart)
    return job_id;
}

bool CommandDevice::submit_pvector_write(Tracker& tracker, std::uint64_t cycle) {
    const auto context = context_for_node(tracker.command.node_id);
    if (!context.has_value()) {
        release_command_resources(tracker, cycle);
        finish_command(
            tracker, cmd::StatusCode::BadCommand, "missing front context", cycle);
        return false;
    }
    const auto& slot = front_slots_[*context];
    dp::MemoryRequest request;
    request.request_id = plane_.memory_service().next_request_id();
    request.client = dp::MemoryClient::ExecUnit;
    request.write = true;
    request.address = slot.front_view.p.base_addr;
    request.length = static_cast<std::uint64_t>(slot.front_view.pivot_dim) * 4;
    request.kind = dp::AccessKind::Int32Data;
    request.write_data = identity_pvector(slot.front_view.pivot_dim);
    request.command_id = tracker.command.command_id;
    request.node_id = tracker.command.node_id;
    request.label = slot.front_view.p.name;
    if (!plane_.memory_service().submit(request, cycle)) {
        return false;  // retry next tick; resources stay held
    }
    tracker.pvector_pending = true;
    tracker.pvector_request_id = request.request_id;
    pvector_map_[request.request_id] = tracker.command.command_id;
    return true;
}

bool CommandDevice::issue_solve_load(Tracker& tracker, std::uint64_t cycle) {
    const auto& view = tracker.solve_view;
    const std::uint64_t chunk = std::min<std::uint64_t>(
        tracker.tile_width, tracker.solve_cols - tracker.solve_elem);
    const std::string owner =
        "solve:" + std::to_string(tracker.command.command_id);
    const dp::BufferHandle handle{
        tracker.context_slot, plane_.buffers().generation(tracker.context_slot)};
    if (plane_.buffers().state(tracker.context_slot) != dp::BufferState::Reserved) {
        // Refill path: previous chunks leave the scratch READY/DIRTY.
        const auto rearm = plane_.buffers().rearm(handle, owner, cycle);
        if (!rearm.ok) {
            ++tracker.stall_resource;
            return false;
        }
    }
    dp::DmaRequest load;
    load.request_id = plane_.memory_service().next_request_id();
    load.has_command = true;
    load.command_id = tracker.command.command_id;
    load.node_id = tracker.command.node_id;
    load.direction = dp::DmaDirection::MemoryToBuffer;
    load.ddr_region = view.rhs;
    load.owner = owner;
    load.buffer = handle;
    load.ddr_offset =
        static_cast<std::uint64_t>(tracker.solve_row) * view.rhs.row_stride +
        static_cast<std::uint64_t>(tracker.solve_elem) * 4;
    load.rows = 1;
    load.cols = static_cast<std::uint32_t>(chunk);
    load.buffer_row_stride = chunk * 4;
    load.element_bytes = 4;
    load.label = view.rhs.name;
    if (enqueue_dma_job(tracker, load, cycle) == 0) {
        return false;
    }
    tracker.solve_load_pending = false;
    tracker.solve_store_issued = false;
    return true;
}

bool CommandDevice::issue_solve_store(Tracker& tracker, std::uint64_t cycle) {
    const auto& view = tracker.solve_view;
    const std::uint64_t chunk = std::min<std::uint64_t>(
        tracker.tile_width, tracker.solve_cols - tracker.solve_elem);
    dp::DmaRequest store;
    store.request_id = plane_.memory_service().next_request_id();
    store.has_command = true;
    store.command_id = tracker.command.command_id;
    store.node_id = tracker.command.node_id;
    store.direction = dp::DmaDirection::BufferToMemory;
    store.ddr_region = view.solution;
    store.owner = "solve:" + std::to_string(tracker.command.command_id);
    store.buffer = dp::BufferHandle{
        tracker.context_slot, plane_.buffers().generation(tracker.context_slot)};
    store.ddr_offset =
        static_cast<std::uint64_t>(tracker.solve_row) * view.solution.row_stride +
        static_cast<std::uint64_t>(tracker.solve_elem) * 4;
    store.rows = 1;
    store.cols = static_cast<std::uint32_t>(chunk);
    store.buffer_row_stride = chunk * 4;
    store.element_bytes = 4;
    store.label = view.solution.name;
    if (!enqueue_dma_job(tracker, store, cycle)) {
        return false;
    }
    tracker.solve_store_pending = false;
    tracker.solve_store_issued = true;
    return true;
}

void CommandDevice::advance_solve(Tracker& tracker, std::uint64_t cycle) {
    const std::uint64_t chunk = std::min<std::uint64_t>(
        tracker.tile_width, tracker.solve_cols - tracker.solve_elem);
    tracker.solve_elem += static_cast<std::uint32_t>(chunk);
    if (tracker.solve_elem >= tracker.solve_cols) {
        tracker.solve_elem = 0;
        ++tracker.solve_row;
    }
    if (tracker.solve_row >= tracker.solve_rows) {
        if (tracker.owns_temp) {
            const std::string owner =
                "solve:" + std::to_string(tracker.command.command_id);
            plane_.buffers().release(
                dp::BufferHandle{
                    tracker.context_slot,
                    plane_.buffers().generation(tracker.context_slot)},
                owner,
                cycle);
            tracker.owns_temp = false;
        }
        finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
        return;
    }
    tracker.solve_load_pending = true;
    issue_solve_load(tracker, cycle);
}

void CommandDevice::try_execute(Tracker& tracker, std::uint64_t cycle) {
    const auto opcode = tracker.command.opcode;
    if (tracker.stage2_pending) {
        const bool is_assemble = opcode == cmd::Opcode::AssembleExtendAdd;
        if (is_assemble) {
            acquire_exec_resources(tracker, UnitKind::Assemble, true, true, cycle);
            return;
        }
        // Solve stage 2: solve unit + temp write lease on the solve scratch.
        const std::string owner =
            "solve:" + std::to_string(tracker.command.command_id);
        const auto lease = plane_.buffers().begin_write(
            dp::BufferHandle{
                tracker.context_slot,
                plane_.buffers().generation(tracker.context_slot)},
            owner,
            0,
            static_cast<std::uint64_t>(tracker.tile_width) * 4,
            cycle);
        if (!lease.ok) {
            ++tracker.stall_resource;
            return;
        }
        tracker.temp_lease = lease.lease;
        tracker.has_temp_lease = true;
        IExecutionUnit* unit = find_free_unit(UnitKind::Solve);
        if (unit == nullptr) {
            release_command_resources(tracker, cycle);
            ++tracker.stall_resource;
            return;
        }
        ExecRequest request;
        request.txn_id = next_txn_id_++;
        request.epoch = epoch_;
        request.command_id = tracker.command.command_id;
        request.node_id = tracker.command.node_id;
        request.opcode = opcode;
        request.unit_id = unit->unit_id();
        request.solve = tracker.solve_view;
        request.temp_slot = dp::BufferHandle{
            tracker.context_slot, plane_.buffers().generation(tracker.context_slot)};
        request.temp_lease = tracker.temp_lease;
        request.has_temp_lease = true;
        request.range_bytes = static_cast<std::uint64_t>(tracker.tile_width) * 4;
        if (!unit->submit(request, cycle)) {
            release_command_resources(tracker, cycle);
            ++tracker.stall_resource;
            return;
        }
        tracker.unit_kind = UnitKind::Solve;
        tracker.unit_id = unit->unit_id();
        tracker.unit_active = true;
        tracker.txn_id = request.txn_id;
        txn_map_[request.txn_id] = tracker.command.command_id;
        UnitBusyRecord busy;
        busy.kind = unit_kind_name(UnitKind::Solve);
        busy.unit_id = unit->unit_id();
        busy.command_id = tracker.command.command_id;
        busy.node_id = tracker.command.node_id;
        busy.start = cycle;
        open_unit_records_[request.txn_id] = busy;
        tracker.stage2_pending = false;
        return;
    }
    if (!tracker.view_ready) {
        return;
    }
    if (aborted_nodes_.count(tracker.command.node_id) != 0 &&
        opcode != cmd::Opcode::AbortNode) {
        finish_command(tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
        return;
    }
    if (const auto required = required_kind_for(opcode); required.has_value() &&
        !unit_bound(*required)) {
        // Capability refusal: a real backend must never silently fall back to
        // a stub or stall forever for an unimplemented unit kind.
        finish_command(
            tracker, cmd::StatusCode::PolicyNotSupported,
            std::string(unit_kind_name(*required)) +
                " kernel is not implemented/bound in this backend",
            cycle);
        return;
    }
    if (opcode_touches_front(opcode) && opcode != cmd::Opcode::NodeBegin) {
        const auto context = context_for_node(tracker.command.node_id);
        if (!context.has_value()) {
            finish_command(
                tracker, cmd::StatusCode::BadCommand, "missing front context", cycle);
            return;
        }
        // Conservative v1 rule: one front-touching command per resident front
        // at a time; different fronts still run in parallel.
        if (front_slots_[*context].busy) {
            ++tracker.stall_resource;
            return;
        }
    }

    switch (opcode) {
    case cmd::Opcode::NodeBegin: {
        const auto& view = tracker.front_view;
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(view.total_dim) * view.total_dim * 4;
        if (bytes > config_.buffer_capacity_bytes) {
            finish_command(
                tracker, cmd::StatusCode::BufferFull,
                "front tile exceeds the scratchpad capacity", cycle);
            return;
        }
        std::uint32_t resident = 0;
        for (const auto& slot : front_slots_) {
            resident += slot.active ? 1 : 0;
        }
        if (resident >= config_.max_resident_fronts) {
            ++tracker.stall_resource;
            return;
        }
        const std::string owner =
            "front:" + std::to_string(tracker.command.node_id);
        auto front = plane_.buffers().allocate(owner, bytes, cycle);
        if (!front.ok) {
            ++tracker.stall_resource;
            return;
        }
        auto temp = plane_.buffers().allocate(owner, bytes, cycle);
        if (!temp.ok) {
            plane_.buffers().release(front.handle, owner, cycle);
            ++tracker.stall_resource;
            return;
        }
        for (auto& slot : front_slots_) {
            if (!slot.active) {
                slot.active = true;
                slot.node_id = tracker.command.node_id;
                slot.owner = owner;
                slot.front = front.handle;
                slot.temp = temp.handle;
                slot.occupy_start = cycle;
                slot.front_view = view;
                break;
            }
        }
        tracker.owns_context = true;
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
        return;
    }
    case cmd::Opcode::LoadFront: {
        const auto context = context_for_node(tracker.command.node_id);
        if (!context.has_value()) {
            finish_command(
                tracker, cmd::StatusCode::BadCommand, "missing front context", cycle);
            return;
        }
        const auto& slot = front_slots_[*context];
        dp::DmaRequest request;
        request.request_id = plane_.memory_service().next_request_id();
        request.has_command = true;
        request.command_id = tracker.command.command_id;
        request.node_id = tracker.command.node_id;
        request.direction = dp::DmaDirection::MemoryToBuffer;
        request.ddr_region = tracker.front_view.front;
        request.ddr_row_stride = tracker.front_view.front.row_stride;
        request.owner = slot.owner;
        request.buffer = slot.front;
        request.rows = tracker.front_view.total_dim;
        request.cols = tracker.front_view.total_dim;
        request.buffer_row_stride =
            static_cast<std::uint64_t>(tracker.front_view.total_dim) * 4;
        request.element_bytes = 4;
        request.label = tracker.front_view.front.name;
        if (enqueue_dma_job(tracker, request, cycle) == 0) {
            return;
        }
        tracker.front_slot_index = *context;
        front_slots_[*context].busy = true;
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        tracker.phase = Tracker::Phase::Running;
        return;
    }
    case cmd::Opcode::AssembleExtendAdd: {
        const auto context = context_for_node(tracker.command.node_id);
        if (!context.has_value()) {
            finish_command(
                tracker, cmd::StatusCode::BadCommand, "missing front context", cycle);
            return;
        }
        const auto& slot = front_slots_[*context];
        if (config_.enable_real_panel) {
            const auto reason = validate_contribution(
                tracker.contribution_view, slot.front_view, tracker.command.node_id);
            if (!reason.empty()) {
                finish_command(tracker, cmd::StatusCode::BadDescriptor, reason, cycle);
                return;
            }
        }
        if (plane_.buffers().state(slot.temp.slot) != dp::BufferState::Reserved) {
            // Refill path: a previously used temp slot must be re-armed before
            // the DMA load; a freshly allocated one is already RESERVED.
            const auto rearm = plane_.buffers().rearm(slot.temp, slot.owner, cycle);
            if (!rearm.ok) {
                ++tracker.stall_resource;
                return;
            }
        }
        dp::DmaRequest request;
        request.request_id = plane_.memory_service().next_request_id();
        request.has_command = true;
        request.command_id = tracker.command.command_id;
        request.node_id = tracker.command.node_id;
        request.direction = dp::DmaDirection::MemoryToBuffer;
        request.ddr_region = tracker.contribution_view.source;
        request.ddr_row_stride = tracker.contribution_view.source.row_stride;
        request.owner = slot.owner;
        request.buffer = slot.temp;
        request.rows = tracker.contribution_view.col_count;
        request.cols = tracker.contribution_view.col_count;
        request.buffer_row_stride =
            static_cast<std::uint64_t>(tracker.contribution_view.col_count) * 4;
        request.element_bytes = 4;
        request.label = tracker.contribution_view.source.name;
        if (enqueue_dma_job(tracker, request, cycle) == 0) {
            return;
        }
        tracker.front_slot_index = *context;
        front_slots_[*context].busy = true;
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        tracker.phase = Tracker::Phase::Running;
        return;
    }
    case cmd::Opcode::PanelLu: {
        if (tracker.front_view.policy == cmd::PivotPolicy::DynamicPanel) {
            finish_command(
                tracker, cmd::StatusCode::PolicyNotSupported,
                "DYNAMIC_PANEL device execution is not implemented", cycle);
            return;
        }
        if (acquire_exec_resources(tracker, UnitKind::Panel, true, false, cycle)) {
            const auto slot_index = context_for_node(tracker.command.node_id);
            tracker.front_slot_index = *slot_index;
            front_slots_[*slot_index].busy = true;
        }
        return;
    }
    case cmd::Opcode::TrsmLeft:
    case cmd::Opcode::TrsmRight:
    case cmd::Opcode::GemmSchur: {
        const UnitKind kind = opcode == cmd::Opcode::GemmSchur ? UnitKind::Gemm
                                                              : UnitKind::Trsm;
        if (acquire_exec_resources(tracker, kind, true, false, cycle)) {
            const auto slot_index = context_for_node(tracker.command.node_id);
            tracker.front_slot_index = *slot_index;
            front_slots_[*slot_index].busy = true;
        }
        return;
    }
    case cmd::Opcode::StoreFactor: {
        const auto context = context_for_node(tracker.command.node_id);
        if (!context.has_value()) {
            finish_command(
                tracker, cmd::StatusCode::BadCommand, "missing front context", cycle);
            return;
        }
        const auto& slot = front_slots_[*context];
        const auto& factor = tracker.factor_view;
        const std::uint32_t total = factor.total_dim;
        const std::uint32_t pivot = factor.pivot_dim;

        // Real T07 path: chunked packed-LU -> L/U conversion through the front
        // read lease and MemoryService writes. Used whenever a Factor unit is
        // explicitly bound; when it is bound but busy the command WAITS -- the
        // legacy rectangular DMA copy below is only for the explicit test-stub
        // backend (no factor unit bound) and must never run as a busy fallback.
        if (unit_bound(UnitKind::Factor)) {
            IExecutionUnit* factor_unit = find_free_unit(UnitKind::Factor);
            if (factor_unit == nullptr) {
                ++tracker.stall_resource;
                return;
            }
            const std::uint64_t front_bytes =
                static_cast<std::uint64_t>(total) * total * 4;
            const auto lease =
                plane_.buffers().begin_read(slot.front, slot.owner, 0, front_bytes, cycle);
            if (!lease.ok) {
                ++tracker.stall_resource;
                return;
            }
            ExecRequest request;
            request.txn_id = next_txn_id_++;
            request.epoch = epoch_;
            request.command_id = tracker.command.command_id;
            request.node_id = tracker.command.node_id;
            request.opcode = opcode;
            request.unit_id = factor_unit->unit_id();
            request.front = tracker.front_view;
            request.factor = factor;
            request.front_slot = slot.front;
            request.front_lease = lease.lease;
            request.has_front_lease = true;
            request.total_dim = total;
            request.range_bytes = front_bytes;
            if (!factor_unit->submit(request, cycle)) {
                plane_.buffers().release_lease(lease.lease.lease_id, cycle);
                ++tracker.stall_resource;
                return;
            }
            tracker.front_lease = lease.lease;
            tracker.has_front_lease = true;
            tracker.front_write_lease = false;
            tracker.unit_kind = UnitKind::Factor;
            tracker.unit_id = factor_unit->unit_id();
            tracker.unit_active = true;
            tracker.txn_id = request.txn_id;
            txn_map_[request.txn_id] = tracker.command.command_id;
            UnitBusyRecord busy;
            busy.kind = unit_kind_name(UnitKind::Factor);
            busy.unit_id = factor_unit->unit_id();
            busy.command_id = tracker.command.command_id;
            busy.node_id = tracker.command.node_id;
            busy.start = cycle;
            open_unit_records_[request.txn_id] = busy;
            tracker.front_slot_index = *context;
            front_slots_[*context].busy = true;
            if (tracker.start_cycle == 0) {
                tracker.start_cycle = cycle;
            }
            tracker.phase = Tracker::Phase::Running;
            return;
        }

        dp::DmaRequest l_request;
        l_request.request_id = plane_.memory_service().next_request_id();
        l_request.has_command = true;
        l_request.command_id = tracker.command.command_id;
        l_request.node_id = tracker.command.node_id;
        l_request.direction = dp::DmaDirection::BufferToMemory;
        l_request.ddr_region = factor.l;
        l_request.ddr_row_stride = factor.l.row_stride;
        l_request.owner = slot.owner;
        l_request.buffer = slot.front;
        l_request.rows = total;
        l_request.cols = pivot;
        l_request.buffer_row_stride = static_cast<std::uint64_t>(total) * 4;
        l_request.element_bytes = 4;
        l_request.label = factor.l.name;

        dp::DmaRequest u_request = l_request;
        u_request.request_id = plane_.memory_service().next_request_id();
        u_request.ddr_region = factor.u;
        u_request.ddr_row_stride = factor.u.row_stride;
        u_request.rows = pivot;
        u_request.cols = total;
        u_request.label = factor.u.name;

        // Atomic: both jobs must fit into the shared DMA queue.
        if (!dma_arbiter_.can_enqueue(2)) {
            ++tracker.stall_dma;
            return;
        }
        if (enqueue_dma_job(tracker, l_request, cycle) == 0 ||
            enqueue_dma_job(tracker, u_request, cycle) == 0) {
            for (const auto job_id :
                 dma_arbiter_.drop_queued_command(tracker.command.command_id)) {
                job_map_.erase(job_id);
            }
            return;
        }
        tracker.front_slot_index = *context;
        front_slots_[*context].busy = true;
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        tracker.phase = Tracker::Phase::Running;
        return;
    }
    case cmd::Opcode::StoreUpdate: {
        const auto context = context_for_node(tracker.command.node_id);
        if (!context.has_value()) {
            finish_command(
                tracker, cmd::StatusCode::BadCommand, "missing front context", cycle);
            return;
        }
        const auto& slot = front_slots_[*context];
        const std::uint32_t total = slot.front_view.total_dim;
        const std::uint32_t pivot = slot.front_view.pivot_dim;
        const std::uint32_t update = total - pivot;
        dp::DmaRequest request;
        request.request_id = plane_.memory_service().next_request_id();
        request.has_command = true;
        request.command_id = tracker.command.command_id;
        request.node_id = tracker.command.node_id;
        request.direction = dp::DmaDirection::BufferToMemory;
        request.ddr_region = tracker.region_view;
        request.ddr_row_stride = tracker.region_view.row_stride;
        request.owner = slot.owner;
        request.buffer = slot.front;
        request.buffer_offset =
            static_cast<std::uint64_t>(pivot) * total * 4 +
            static_cast<std::uint64_t>(pivot) * 4;
        request.rows = update;
        request.cols = update;
        request.buffer_row_stride = static_cast<std::uint64_t>(total) * 4;
        request.element_bytes = 4;
        request.label = tracker.region_view.name;
        if (enqueue_dma_job(tracker, request, cycle) == 0) {
            return;
        }
        tracker.front_slot_index = *context;
        front_slots_[*context].busy = true;
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        tracker.phase = Tracker::Phase::Running;
        return;
    }
    case cmd::Opcode::NodeCommit: {
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
        return;
    }
    case cmd::Opcode::SolveForward:
    case cmd::Opcode::SolveBackward: {
        // The full RHS is traversed in scratchpad-sized chunks; a prefix-only
        // transfer would silently truncate data.
        const auto& view = tracker.solve_view;
        const std::uint64_t capacity = config_.buffer_capacity_bytes;
        const std::uint32_t rows =
            std::min(view.rhs.rows, view.solution.rows);
        const std::uint32_t cols =
            std::min(view.rhs.cols, view.solution.cols);
        const std::uint32_t chunk = static_cast<std::uint32_t>(
            std::min<std::uint64_t>({cols, capacity / 4}));
        if (rows == 0 || cols == 0) {
            finish_command(
                tracker, cmd::StatusCode::BufferFull,
                "solve regions have no movable elements", cycle);
            return;
        }
        if (chunk == 0) {
            finish_command(
                tracker, cmd::StatusCode::BufferFull,
                "solve tile exceeds the scratchpad capacity", cycle);
            return;
        }
        const std::string owner =
            "solve:" + std::to_string(tracker.command.command_id);
        auto temp = plane_.buffers().allocate(
            owner, static_cast<std::uint64_t>(chunk) * 4, cycle);
        if (!temp.ok) {
            ++tracker.stall_resource;
            return;
        }
        tracker.owns_temp = true;
        tracker.context_slot = temp.handle.slot;
        tracker.tile_width = chunk;
        tracker.solve_rows = rows;
        tracker.solve_cols = cols;
        tracker.solve_row = 0;
        tracker.solve_elem = 0;
        tracker.solve_load_pending = true;
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        tracker.phase = Tracker::Phase::Running;
        issue_solve_load(tracker, cycle);
        return;
    }
    case cmd::Opcode::AbortNode: {
        abort_node(tracker.command.node_id);
        if (tracker.start_cycle == 0) {
            tracker.start_cycle = cycle;
        }
        finish_command(tracker, cmd::StatusCode::Ok, "node aborted", cycle);
        return;
    }
    }
}

void CommandDevice::handle_job_done(
    Tracker& tracker,
    const DmaArbiter::Completion& completion,
    std::uint64_t cycle) {
    const auto& response = completion.response;
    tracker.bytes_read += response.read_bytes;
    tracker.bytes_written += response.write_bytes;
    const auto all_jobs_settled = [&tracker]() {
        return tracker.dma_jobs_done + tracker.dma_jobs_cancelled >=
               tracker.dma_jobs_total;
    };

    if (tracker.aborted) {
        // Any settled job (finished or rejected) counts; only actual in-flight
        // transactions are awaited after an abort.
        if (response.status == cmd::StatusCode::Ok) {
            tracker.dma_jobs_done += 1;
        } else {
            tracker.dma_jobs_cancelled += 1;
        }
        if (all_jobs_settled()) {
            finish_command(
                tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
        }
        return;
    }
    if (response.status != cmd::StatusCode::Ok) {
        if (response.retryable) {
            // Transient buffer/port contention raised before the first beat:
            // nothing has been written, so the same logical job is re-queued
            // later instead of failing the command.
            tracker.job_retries.push_back(completion.request);
            ++dma_retry_count_;
            return;
        }
        // Permanent failure: drop jobs of this command that are still queued
        // (in-flight ones finish naturally; writes are not rolled back).
        for (const auto job_id :
             dma_arbiter_.drop_queued_command(tracker.command.command_id)) {
            const auto mapping = job_map_.find(job_id);
            if (mapping != job_map_.end()) {
                job_map_.erase(mapping);
                tracker.dma_jobs_cancelled += 1;
            }
        }
        finish_command(tracker, response.status, response.reason, cycle);
        return;
    }
    tracker.dma_jobs_done += 1;
    switch (tracker.command.opcode) {
    case cmd::Opcode::AssembleExtendAdd:
        tracker.stage2_pending = true;
        return;
    case cmd::Opcode::SolveForward:
    case cmd::Opcode::SolveBackward:
        if (!tracker.solve_store_issued) {
            tracker.stage2_pending = true;  // chunk loaded: run the solve unit
            return;
        }
        tracker.solve_store_issued = false;
        advance_solve(tracker, cycle);
        return;
    case cmd::Opcode::StoreFactor:
        if (all_jobs_settled()) {
            finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
        }
        return;
    case cmd::Opcode::LoadFront:
    case cmd::Opcode::StoreUpdate:
        finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
        return;
    default:
        return;
    }
}

std::uint64_t CommandDevice::requeue_dma_job(
    Tracker& tracker,
    const dp::DmaRequest& request,
    std::uint64_t cycle) {
    DmaArbiter::Job job;
    job.front_tag = tracker.command.node_id;
    job.command_id = tracker.command.command_id;
    job.node_id = tracker.command.node_id;
    job.request = request;
    const std::uint64_t job_id = dma_arbiter_.enqueue(std::move(job), cycle);
    if (job_id == 0) {
        ++tracker.stall_dma;
        return 0;
    }
    job_map_[job_id] = tracker.command.command_id;
    return job_id;  // logical job was already counted when first submitted
}

void CommandDevice::handle_exec_response(
    Tracker& tracker,
    const ExecResponse& response,
    std::uint64_t cycle) {
    if (tracker.phase == Tracker::Phase::Done ||
        tracker.phase == Tracker::Phase::Completing || tracker.finish_cycle != 0) {
        // Late response after a directed abort or an earlier terminal state:
        // consume it without touching stats or outputs.
        tracker.unit_active = false;
        return;
    }
    tracker.unit_active = false;
    // An aborted transaction may already have performed scratchpad work.
    // Account its one terminal response before suppressing further outputs.
    tracker.bytes_read += response.bytes_read;
    tracker.bytes_written += response.bytes_written;
    tracker.gemm_backend = response.gemm_backend;
    tracker.gemm_saturation_count = response.saturation_count;
    tracker.gemm_local_conversion_bytes = response.local_conversion_bytes;
    tracker.gemm_scale_reduction_cycles = response.scale_reduction_cycles;
    tracker.gemm_quantize_cycles = response.quantize_cycles;
    tracker.gemm_integer_compute_cycles = response.integer_compute_cycles;
    tracker.gemm_dequantize_cycles = response.dequantize_cycles;
    tracker.gemm_exponent_a = response.exponent_a;
    tracker.gemm_exponent_b = response.exponent_b;
    if (tracker.aborted) {
        if (tracker.dma_jobs_done + tracker.dma_jobs_cancelled <
            tracker.dma_jobs_total) {
            // In-flight DMA writes still settle first; the job-completion path
            // releases resources and finishes the command.
            return;
        }
        release_command_resources(tracker, cycle);
        finish_command(tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
        return;
    }
    tracker.pivot_count = response.pivot_count;
    tracker.overflow_count = response.overflow_count;
    if (response.front_scale > 0.0 && tracker.front_slot_index != cmd::NONE &&
        tracker.front_slot_index < front_slots_.size()) {
        front_slots_[tracker.front_slot_index].front_scale = response.front_scale;
        front_slots_[tracker.front_slot_index].threshold_product =
            response.threshold_product;
    }
    if (response.status != cmd::StatusCode::Ok) {
        release_command_resources(tracker, cycle);
        finish_command(tracker, response.status, response.reason, cycle);
        return;
    }
    const auto opcode = tracker.command.opcode;
    if (response.needs_pvector_write) {
        // Phase 3 records the need; the p-vector write is issued in the same
        // update phase's retry step and awaited before completion.
        tracker.pvector_pending = true;
        tracker.pvector_request_id = 0;
        tracker.phase = Tracker::Phase::PostExec;
        return;
    }
    release_command_resources(tracker, cycle);
    if (opcode == cmd::Opcode::SolveForward ||
        opcode == cmd::Opcode::SolveBackward) {
        tracker.solve_store_pending = true;
        return;
    }
    finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
}

void CommandDevice::handle_pvector_done(
    Tracker& tracker,
    const dp::MemoryResponse& response,
    std::uint64_t cycle) {
    tracker.pvector_pending = false;
    tracker.bytes_written += response.bytes;
    release_command_resources(tracker, cycle);
    if (tracker.aborted) {
        // Aborted while the output write was in flight: the write may have
        // landed, but the command must not report success.
        finish_command(
            tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
        return;
    }
    if (response.status != cmd::StatusCode::Ok) {
        finish_command(tracker, response.status, response.reason, cycle);
        return;
    }
    finish_command(tracker, cmd::StatusCode::Ok, "", cycle);
}

void CommandDevice::release_command_resources(Tracker& tracker, std::uint64_t cycle) {
    if (tracker.has_front_lease) {
        if (tracker.front_write_lease) {
            plane_.buffers().end_write(
                tracker.front_lease.lease_id, tracker.front_lease.length, cycle);
        } else {
            plane_.buffers().release_lease(tracker.front_lease.lease_id, cycle);
        }
        tracker.has_front_lease = false;
        tracker.front_write_lease = false;
    }
    if (tracker.has_temp_lease) {
        if (tracker.temp_lease.write) {
            plane_.buffers().end_write(
                tracker.temp_lease.lease_id, tracker.temp_lease.length, cycle);
        } else {
            plane_.buffers().release_lease(tracker.temp_lease.lease_id, cycle);
        }
        tracker.has_temp_lease = false;
    }
}

void CommandDevice::finish_command(
    Tracker& tracker,
    cmd::StatusCode status,
    const std::string& reason,
    std::uint64_t cycle) {
    if (tracker.finish_cycle != 0 || tracker.phase == Tracker::Phase::Completing ||
        tracker.phase == Tracker::Phase::Done) {
        return;
    }
    release_command_resources(tracker, cycle);
    tracker.status = status;
    tracker.reason = reason;
    tracker.finish_cycle = cycle;
    if (tracker.front_slot_index != cmd::NONE &&
        tracker.front_slot_index < front_slots_.size()) {
        front_slots_[tracker.front_slot_index].busy = false;
    }
    const std::uint64_t end = tracker.start_cycle != 0 ? tracker.start_cycle : cycle;
    tracker.stall_window = end > tracker.fetch_cycle ? end - tracker.fetch_cycle : 0;
    tracker.phase = Tracker::Phase::PostExec;
    enqueue_completion(tracker, cycle);
}

void CommandDevice::enqueue_completion(Tracker& tracker, std::uint64_t cycle) {
    CompletionWriter::Item item;
    item.slot = tracker.command.command_id;
    item.record.command_id = tracker.command.command_id;
    item.record.node_id = tracker.command.node_id;
    item.record.status_code = tracker.status;
    item.record.pivot_count = tracker.pivot_count;
    item.record.start_cycle = tracker.start_cycle;
    item.record.finish_cycle = tracker.finish_cycle;
    item.record.read_bytes = tracker.bytes_read;
    item.record.write_bytes = tracker.bytes_written;
    // Mutually exclusive per-tick wait classes only; stall_window measures the
    // whole pre-start residency and must not be added on top.
    item.record.stall_cycles =
        tracker.stall_dep + tracker.stall_resource + tracker.stall_dma;
    item.record.overflow_count = tracker.overflow_count;
    item.record.retry_count = 0;
    if (!completion_writer_.enqueue(item, cycle)) {
        return;  // queue backpressure: retried from update_phase
    }
    tracker.phase = Tracker::Phase::Completing;
}

void CommandDevice::drain_completion_results(std::uint64_t cycle) {
    std::uint32_t slot = 0;
    bool confirmed = false;
    while (completion_writer_.take_result(slot, confirmed)) {
        Tracker* tracker = find_tracker(slot);
        if (tracker == nullptr || tracker->phase == Tracker::Phase::Done) {
            continue;
        }
        tracker->phase = Tracker::Phase::Done;
        ++completed_commands_;
        completion_status_[slot] = tracker->status;
        const bool ready = confirmed && tracker->status == cmd::StatusCode::Ok;
        scoreboard_.signal(
            tracker->command.signal_token,
            ready ? TokenScoreboard::State::Ready : TokenScoreboard::State::Failed,
            tracker->command.command_id);

        CommandStats stats;
        stats.command_id = tracker->command.command_id;
        stats.node_id = tracker->command.node_id;
        stats.opcode = std::to_string(
            static_cast<std::uint32_t>(tracker->command.opcode));
        stats.unit_kind =
            tracker->unit_id == cmd::NONE ? "" : unit_kind_name(tracker->unit_kind);
        stats.unit_id = tracker->unit_id == cmd::NONE
                            ? -1
                            : static_cast<std::int32_t>(tracker->unit_id);
        stats.fetch_cycle = tracker->fetch_cycle;
        stats.admit_cycle = tracker->admit_cycle;
        stats.start_cycle = tracker->start_cycle;
        stats.finish_cycle = tracker->finish_cycle;
        stats.completion_cycle = cycle;
        stats.status = dp::status_name(tracker->status);
        stats.reason = tracker->reason;
        stats.bytes_read = tracker->bytes_read;
        stats.bytes_written = tracker->bytes_written;
        stats.stall_window = tracker->stall_window;
        stats.stall_dep = tracker->stall_dep;
        stats.stall_resource = tracker->stall_resource;
        stats.stall_dma = tracker->stall_dma;
        stats.stall_cycles =
            tracker->stall_dep + tracker->stall_resource + tracker->stall_dma;
        stats.pivot_count = tracker->pivot_count;
        stats.overflow_count = tracker->overflow_count;
        stats.dma_jobs = tracker->dma_jobs_total;
        if (tracker->command.opcode == cmd::Opcode::GemmSchur) {
            stats.gemm_backend =
                hw::panel_lu::gemm_backend_name(tracker->gemm_backend);
            stats.saturation_count = tracker->gemm_saturation_count;
            stats.local_conversion_bytes = tracker->gemm_local_conversion_bytes;
            stats.scale_reduction_cycles = tracker->gemm_scale_reduction_cycles;
            stats.quantize_cycles = tracker->gemm_quantize_cycles;
            stats.integer_compute_cycles = tracker->gemm_integer_compute_cycles;
            stats.dequantize_cycles = tracker->gemm_dequantize_cycles;
            stats.exponent_a = tracker->gemm_exponent_a;
            stats.exponent_b = tracker->gemm_exponent_b;
        }
        command_stats_.push_back(stats);

        if (tracker->owns_temp) {
            const std::string owner =
                "solve:" + std::to_string(tracker->command.command_id);
            plane_.buffers().release(
                dp::BufferHandle{
                    tracker->context_slot,
                    plane_.buffers().generation(tracker->context_slot)},
                owner,
                cycle);
            tracker->owns_temp = false;
        }
        maybe_cleanup_node(tracker->command.node_id, cycle);
    }
}

void CommandDevice::maybe_cleanup_node(std::uint32_t node_id, std::uint64_t cycle) {
    if (commit_seen_.count(node_id) == 0 && aborted_nodes_.count(node_id) == 0) {
        return;  // the node's command stream is not finished yet
    }
    const auto context_index = context_for_node(node_id);
    if (!context_index.has_value()) {
        return;
    }
    for (const auto& tracker : trackers_) {
        if (tracker->command.node_id != node_id ||
            tracker->phase == Tracker::Phase::Done) {
            continue;
        }
        // Slot-referencing commands of the node block cleanup until they are
        // terminal; SOLVE commands of the same node use their own scratch and
        // must not block the release.
        if (opcode_touches_front(tracker->command.opcode)) {
            return;
        }
    }
    // The scheduler pulls commands out of order, so later commands of the node
    // may still sit in the fetch FIFO; releasing the slots while they are
    // untracked would break them.
    for (std::size_t index = 0; index < fetch_.window_size(); ++index) {
        const auto& command = fetch_.at(index);
        if (command.node_id == node_id && opcode_touches_front(command.opcode)) {
            return;
        }
    }
    auto& slot = front_slots_[*context_index];
    const std::string owner = slot.owner;
    plane_.buffers().release(slot.front, owner, cycle);
    plane_.buffers().release(slot.temp, owner, cycle);
    FrontBusyRecord record;
    record.slot = slot.front.slot;
    record.node_id = slot.node_id;
    record.start = slot.occupy_start;
    record.finish = cycle;
    front_records_.push_back(record);
    slot.active = false;
}

void CommandDevice::abort_node(std::uint32_t node_id) {
    const std::uint64_t cycle = plane_.cycle();
    aborted_nodes_.insert(node_id);
    // Queued DMA jobs never ran a beat: drop them and count them as cancelled
    // so the tracker only waits for actual in-flight transactions.
    for (const auto job_id : dma_arbiter_.drop_queued(node_id)) {
        const auto mapping = job_map_.find(job_id);
        if (mapping != job_map_.end()) {
            Tracker* tracker = find_tracker(mapping->second);
            if (tracker != nullptr) {
                tracker->dma_jobs_cancelled += 1;
            }
            job_map_.erase(mapping);
        }
    }
    for (auto& tracker : trackers_) {
        if (tracker->command.node_id != node_id ||
            tracker->phase == Tracker::Phase::Done ||
            tracker->phase == Tracker::Phase::Completing) {
            continue;
        }
        if (tracker->phase == Tracker::Phase::PostExec) {
            if (tracker->finish_cycle != 0 || !tracker->pvector_pending) {
                continue;  // outputs are done; the command already commits
            }
            // Output writes are still in flight: the command must not publish
            // a success token after the abort. It terminates as ABORTED once
            // the in-flight write settles.
            tracker->aborted = true;
            if (tracker->pvector_request_id == 0) {
                tracker->pvector_pending = false;
                release_command_resources(*tracker, cycle);
                finish_command(
                    *tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
            }
            continue;
        }
        tracker->aborted = true;
        // Staged work that was never submitted is not awaited.
        tracker->solve_load_pending = false;
        tracker->solve_store_pending = false;
        tracker->stage2_pending = false;
        tracker->job_retries.clear();
        if (tracker->unit_active) {
            bool abort_delivered = false;
            for (auto& unit : units_) {
                if (unit->unit_id() == tracker->unit_id &&
                    unit->kind() == tracker->unit_kind) {
                    // Directed cancellation: the unit stops issuing new work
                    // and still produces exactly one terminal ABORTED response
                    // after its in-flight accesses have settled. The tracker
                    // (and its leases) must survive until then: "cancel
                    // requested" is not "drained".
                    unit->abort_txn(tracker->txn_id, cycle);
                    abort_delivered = true;
                }
            }
            if (abort_delivered) {
                // No release/finish here; the terminal response path
                // (handle_exec_response) releases resources and finishes.
                continue;
            }
            // The bound unit disappeared (reset race): finish defensively.
            tracker->unit_active = false;
            release_command_resources(*tracker, cycle);
        }
        if (tracker->dma_jobs_done + tracker->dma_jobs_cancelled <
            tracker->dma_jobs_total) {
            // The in-flight DMA job cannot be cancelled; it finishes normally
            // and the tracker then terminates as Aborted. Its writes are not
            // rolled back.
            continue;
        }
        finish_command(*tracker, cmd::StatusCode::Aborted, "node was aborted", cycle);
    }
}

void CommandDevice::reset() {
    plane_.reset();
    fetch_.reset();
    completion_writer_.reset();
    dma_arbiter_.reset();
    plane_.set_memory_sink(dp::MemoryClient::CommandFetch, &fetch_);
    plane_.set_memory_sink(dp::MemoryClient::CompletionWriter, &completion_writer_);
    plane_.set_memory_sink(dp::MemoryClient::ExecUnit, this);
    for (auto& unit : units_) {
        unit->reset();
    }
    if (panel_engine_ != nullptr) {
        panel_engine_->reset();
    }
    on_chip_.reset();
    on_chip_.configure(
        config_.panel.load_bytes_per_cycle, config_.panel.writeback_bytes_per_cycle);
    scoreboard_ = TokenScoreboard();
    scoreboard_.init(plane_.boot().token_count);
    trackers_.clear();
    front_slots_.clear();
    front_slots_.resize(config_.max_resident_fronts);
    aborted_nodes_.clear();
    commit_seen_.clear();
    completion_status_.clear();
    job_map_.clear();
    txn_map_.clear();
    pvector_map_.clear();
    exec_inbox_.clear();
    open_unit_records_.clear();
    ++epoch_;
    completed_commands_ = 0;
    command_stats_.clear();
    unit_records_.clear();
    front_records_.clear();
    fault_reason_.clear();
    state_ = BatchState::Idle;
}

void CommandDevice::inject_failure(std::uint32_t command_id, ExecFailure failure) {
    for (auto& unit : units_) {
        if (unit->kind() == UnitKind::Panel || unit->kind() == UnitKind::Trsm ||
            unit->kind() == UnitKind::Gemm || unit->kind() == UnitKind::Solve ||
            unit->kind() == UnitKind::Assemble) {
            // Error injection is a test-stub capability; real units ignore it.
            TestStubUnit* stub = dynamic_cast<TestStubUnit*>(unit.get());
            if (stub != nullptr) {
                stub->inject_failure(command_id, failure);
            }
        }
    }
}

void CommandDevice::deliver_memory_response(const dp::MemoryResponse& response) {
    if (pvector_map_.count(response.request_id) == 0) {
        // Not a P-vector write: route by request id to exactly one unit that
        // owns the in-flight write. Broadcasting would let multiple factor
        // instances decrement each other's pending counters.
        for (auto& unit : units_) {
            if (unit->handles_request(response.request_id)) {
                unit->deliver_memory_response(response);
                return;
            }
        }
        return;  // no owner: late/foreign response, dropped
    }
    exec_inbox_[response.request_id] = response;
}

std::uint32_t CommandDevice::pending_pvector_command() const {
    for (const auto& tracker : trackers_) {
        if (tracker->pvector_pending && tracker->finish_cycle == 0 &&
            tracker->pvector_request_id != 0) {
            return tracker->command.command_id;
        }
    }
    return cmd::NONE;
}

bool CommandDevice::command_dma_progress(
    std::uint32_t command_id,
    std::uint32_t* issued,
    std::uint32_t* done,
    std::uint32_t* cancelled) const {
    for (const auto& tracker : trackers_) {
        if (tracker->command.command_id != command_id) {
            continue;
        }
        if (issued != nullptr) {
            *issued = tracker->dma_jobs_total;
        }
        if (done != nullptr) {
            *done = tracker->dma_jobs_done;
        }
        if (cancelled != nullptr) {
            *cancelled = tracker->dma_jobs_cancelled;
        }
        return true;
    }
    return false;
}

std::string CommandDevice::debug_dump() const {
    std::string out;
    for (std::uint32_t index = 0; index < front_slots_.size(); ++index) {
        const auto& slot = front_slots_[index];
        out += "front[" + std::to_string(index) + "] active=" +
               std::to_string(slot.active) + " busy=" + std::to_string(slot.busy) +
               " node=" + std::to_string(slot.node_id) + "\n";
    }
    for (std::uint32_t index = 0; index < config_.buffer_slot_count; ++index) {
        out += "buf[" + std::to_string(index) + "] state=" +
               std::to_string(static_cast<int>(plane_.buffers().state(index))) +
               "\n";
    }
    out += "fetch: window=" + std::to_string(fetch_.window_size()) +
           " fetched=" + std::to_string(fetch_.fetched_count()) +
           " total=" + std::to_string(total_commands_) +
           " window_full_cycles=" + std::to_string(fetch_.window_full_cycles()) +
           "\n";
    for (const auto& tracker : trackers_) {
        if (tracker->phase == Tracker::Phase::Done) {
            continue;
        }
        out += "cmd " + std::to_string(tracker->command.command_id) + " node " +
               std::to_string(tracker->command.node_id) + " op " +
               std::to_string(static_cast<std::uint32_t>(tracker->command.opcode));
        switch (tracker->phase) {
        case Tracker::Phase::InWindow: out += " phase=InWindow"; break;
        case Tracker::Phase::Resolving:
            out += " phase=Resolving resolve_pending=" +
                   std::to_string(tracker->resolve_pending) + " wait_id=" +
                   std::to_string(tracker->wait_request_id) +
                   " waits_ready=" + std::to_string(tracker->waits_ready);
            break;
        case Tracker::Phase::Waiting:
            out += " phase=Waiting waits_ready=" + std::to_string(tracker->waits_ready);
            break;
        case Tracker::Phase::Running:
            out += " phase=Running dma=" + std::to_string(tracker->dma_jobs_done) +
                   "+c" + std::to_string(tracker->dma_jobs_cancelled) +
                   "/" + std::to_string(tracker->dma_jobs_total) +
                   " retries=" + std::to_string(tracker->job_retries.size()) +
                   " unit=" + std::to_string(tracker->unit_active) +
                   " stage2=" + std::to_string(tracker->stage2_pending) +
                   " store=" + std::to_string(tracker->solve_store_pending);
            break;
        case Tracker::Phase::PostExec:
            out += " phase=PostExec pvector=" + std::to_string(tracker->pvector_pending);
            break;
        case Tracker::Phase::Completing: out += " phase=Completing"; break;
        case Tracker::Phase::Done: break;
        }
        out += " stalls(res=" + std::to_string(tracker->stall_resource) +
               ",dma=" + std::to_string(tracker->stall_dma) +
               ") owns_temp=" + std::to_string(tracker->owns_temp) +
               " view=" + std::to_string(tracker->view_ready) + " waits=[";
        for (const auto token : tracker->waits) {
            out += std::to_string(token) + ":" +
                   std::to_string(static_cast<int>(scoreboard_.state(token))) + " ";
        }
        out += "]\n";
    }
    return out;
}

bool CommandDevice::numeric_status_pass() const {
    if (!config_.enable_real_panel || config_.enable_test_stub || !control_flow_pass() ||
        completion_status_.size() != total_commands_) return false;
    return std::all_of(completion_status_.begin(), completion_status_.end(),
        [](const auto& entry) { return entry.second == cmd::StatusCode::Ok; });
}

void CommandDevice::write_reports(const std::string& output_dir) const {
    std::filesystem::create_directories(output_dir);
    plane_.trace().write_csv(output_dir);
    {
        const auto& image = plane_.memory().image();
        std::ofstream out(output_dir + "/final_memory_image.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
    }
    {
        std::ofstream out(output_dir + "/nodes.csv");
        out << "node_id,start_cycle,finish_cycle,completion_cycle,commands,failed_commands,status\n";
        for (std::uint32_t node = 0; node < plane_.boot().node_count; ++node) {
            std::uint64_t start = 0, finish = 0, completion = 0;
            std::uint32_t count = 0, failed = 0;
            bool committed = false;
            for (const auto& row : command_stats_) {
                if (row.node_id != node) continue;
                ++count;
                committed = committed || (row.opcode == std::to_string(static_cast<std::uint32_t>(cmd::Opcode::NodeCommit)) && row.status == "OK");
                failed += row.status != "OK";
                if (row.start_cycle != 0 && (start == 0 || row.start_cycle < start)) start = row.start_cycle;
                finish = std::max(finish, row.finish_cycle);
                completion = std::max(completion, row.completion_cycle);
            }
            out << node << ',' << start << ',' << finish << ',' << completion << ','
                << count << ',' << failed << ',' << (failed ? "FAILED" : (committed ? "OK" : (count ? "INCOMPLETE" : "NOT_EXECUTED"))) << '\n';
        }
    }

    {
        std::ofstream out(output_dir + "/commands.csv");
        out << "command_id,node_id,opcode,unit_kind,unit_id,fetch_cycle,admit_cycle,"
               "start_cycle,finish_cycle,completion_cycle,status,reason,bytes_read,"
               "bytes_written,stall_window,stall_cycles,stall_dep,stall_resource,"
               "stall_dma,pivot_count,overflow_count,dma_jobs,gemm_backend,"
               "saturation_count,local_conversion_bytes,scale_reduction_cycles,"
               "quantize_cycles,integer_compute_cycles,dequantize_cycles,"
               "exponent_a,exponent_b\n";
        for (const auto& row : command_stats_) {
            out << row.command_id << ',' << row.node_id << ',' << row.opcode << ','
                << row.unit_kind << ',' << row.unit_id << ',' << row.fetch_cycle << ','
                << row.admit_cycle << ',' << row.start_cycle << ',' << row.finish_cycle
                << ',' << row.completion_cycle << ',' << row.status << ','
                << csv_field(row.reason) << ',' << row.bytes_read << ',' << row.bytes_written
                << ',' << row.stall_window << ',' << row.stall_cycles << ','
                << row.stall_dep << ',' << row.stall_resource << ','
                << row.stall_dma << ',' << row.pivot_count << ','
                << row.overflow_count << ',' << row.dma_jobs << ','
                << row.gemm_backend << ',' << row.saturation_count << ','
                << row.local_conversion_bytes << ',' << row.scale_reduction_cycles
                << ',' << row.quantize_cycles << ',' << row.integer_compute_cycles
                << ',' << row.dequantize_cycles << ',' << row.exponent_a << ','
                << row.exponent_b << '\n';
        }
    }
    {
        std::ofstream out(output_dir + "/units.csv");
        out << "kind,unit_id,command_id,node_id,start_cycle,finish_cycle\n";
        for (const auto& row : unit_records_) {
            out << row.kind << ',' << row.unit_id << ',' << row.command_id << ','
                << row.node_id << ',' << row.start << ',' << row.finish << '\n';
        }
    }
    {
        std::ofstream out(output_dir + "/fronts.csv");
        out << "slot,node_id,occupy_start,occupy_end\n";
        for (const auto& row : front_records_) {
            out << row.slot << ',' << row.node_id << ',' << row.start << ','
                << row.finish << '\n';
        }
    }
    {
        const bool real_panel = panel_engine_ != nullptr;
        const bool mixed = real_panel && config_.enable_test_stub;
        std::string execution_backend =
            real_panel ? "real_panel_fp32" : "test_stub";
        if (mixed) {
            execution_backend = "real_panel_fp32_plus_test_stub";
        }
        // Actual GEMM execution mode of this run (never hard-coded to FP32).
        std::uint32_t gemm_commands = 0;
        std::uint32_t gemm_int32_commands = 0;
        std::uint64_t gemm_saturation = 0;
        std::uint64_t gemm_overflow = 0;
        std::int32_t gemm_exponent_a = 0;
        std::int32_t gemm_exponent_b = 0;
        bool gemm_exponents_set = false;
        std::uint64_t gemm_local_bytes = 0;
        std::uint64_t gemm_scale_cycles = 0;
        std::uint64_t gemm_quantize_cycles = 0;
        std::uint64_t gemm_integer_cycles = 0;
        std::uint64_t gemm_dequantize_cycles = 0;
        const std::string gemm_opcode_name = std::to_string(
            static_cast<std::uint32_t>(cmd::Opcode::GemmSchur));
        for (const auto& entry : command_stats_) {
            if (entry.opcode != gemm_opcode_name) {
                continue;
            }
            ++gemm_commands;
            if (entry.gemm_backend == "int32_gemm") {
                ++gemm_int32_commands;
            }
            gemm_saturation += entry.saturation_count;
            gemm_overflow += entry.overflow_count;
            if (!gemm_exponents_set && entry.gemm_backend == "int32_gemm") {
                gemm_exponent_a = entry.exponent_a;
                gemm_exponent_b = entry.exponent_b;
                gemm_exponents_set = true;
            }
            gemm_local_bytes += entry.local_conversion_bytes;
            gemm_scale_cycles += entry.scale_reduction_cycles;
            gemm_quantize_cycles += entry.quantize_cycles;
            gemm_integer_cycles += entry.integer_compute_cycles;
            gemm_dequantize_cycles += entry.dequantize_cycles;
        }
        const bool gemm_int32_active = gemm_int32_commands > 0;
        std::string numerical_scope = real_panel ? "panel_fp32_external_reference_only"
                                                 : "control_flow_only";
        if (real_panel && gemm_commands > 0) {
            numerical_scope = gemm_int32_active
                                  ? "panel_trsm_fp32_gemm_int32_external_reference_only"
                                  : "panel_trsm_gemm_fp32_external_reference_only";
        }
        std::ofstream out(output_dir + "/summary.json");
        out << "{\n"
            << "  \"tool\": \"t05_command_control_plane\",\n"
            << "  \"execution_backend\": \"" << execution_backend << "\",\n"
            << "  \"panel_backend_real_fp32\": "
            << (real_panel ? "true" : "false") << ",\n"
            << "  \"mixed_backend_explicit\": " << (mixed ? "true" : "false") << ",\n"
            << "  \"control_flow_pass\": "
            << (state_ == BatchState::Completed ? "true" : "false") << ",\n"
            << "  \"numeric_status_pass\": " << (numeric_status_pass() ? "true" : "false") << ",\n"
            << "  \"assembly_backend\": \"" << (real_panel ? "fp32_extend_add" : "test_stub_copy") << "\",\n"
            << "  \"assembly_staging_bytes_per_unit\": 12,\n"
            << "  \"assembly_add_latency_cycles\": " << config_.assembly.add_latency_cycles << ",\n"
            << "  \"numerical_validation\": \"not_performed\",\n"
            << "  \"numerical_scope\": \"" << numerical_scope << "\",\n"
            << "  \"batch_state\": \"" << batch_state_name(state_) << "\",\n"
            << "  \"fault_reason\": \"" << fault_reason_ << "\",\n"
            << "  \"cycles\": " << plane_.cycle() << ",\n"
            << "  \"commands\": " << command_stats_.size() << ",\n"
            << "  \"completions_written\": " << completion_writer_.written_count()
            << ",\n"
            << "  \"panel_numeric_mode\": \""
            << hw::panel_lu::panel_numeric_mode_name(config_.panel.numeric_mode)
            << "\",\n"
            << "  \"gemm_execution\": {\n"
            << "    \"commands\": " << gemm_commands << ",\n"
            << "    \"int32_commands\": " << gemm_int32_commands << ",\n"
            << "    \"int32_active\": " << (gemm_int32_active ? "true" : "false")
            << ",\n"
            << "    \"effective_bits\": " << config_.panel.precision.effective_bits
            << ",\n"
            << "    \"accumulator_bits\": "
            << config_.panel.precision.accumulator_bits << ",\n"
            << "    \"k_block_max\": " << config_.panel.precision.k_block_max
            << ",\n"
            << "    \"precision_config_source\": \""
            << (artifact_.config.gemm_model.has_value() ? "artifact_manifest" : "device_config")
            << "\",\n"
            << "    \"maxabs_elements_per_cycle\": "
            << config_.panel.precision.maxabs_elements_per_cycle << ",\n"
            << "    \"quantize_elements_per_cycle\": "
            << config_.panel.precision.quantize_elements_per_cycle << ",\n"
            << "    \"dequantize_elements_per_cycle\": "
            << config_.panel.precision.dequantize_elements_per_cycle << ",\n"
            << "    \"saturation_policy_default\": \"FAIL\",\n"
            << "    \"conversion_staging_bytes_per_pe_slot\": "
            << (16 * 16 * 4 * 2) << ",\n"
            << "    \"int64_accumulator_bytes_per_pe_slot\": "
            << (16 * 16 * 8) << ",\n"
            << "    \"saturation_count\": " << gemm_saturation << ",\n"
            << "    \"overflow_count\": " << gemm_overflow << ",\n"
            << "    \"representative_exponent_a\": " << gemm_exponent_a << ",\n"
            << "    \"representative_exponent_b\": " << gemm_exponent_b << ",\n"
            << "    \"local_conversion_bytes\": " << gemm_local_bytes << ",\n"
            << "    \"scale_reduction_cycles\": " << gemm_scale_cycles << ",\n"
            << "    \"quantize_cycles\": " << gemm_quantize_cycles << ",\n"
            << "    \"integer_compute_cycles\": " << gemm_integer_cycles << ",\n"
            << "    \"dequantize_cycles\": " << gemm_dequantize_cycles << "\n"
            << "  },\n"
            << "  \"panel_engine\": {\n"
            << "    \"compute_group_count\": " << config_.panel.compute_group_count
            << ",\n"
            << "    \"pes_per_group\": " << config_.panel.pes_per_group
            << ",\n"
            << "    \"pe_count\": " << config_.panel.total_pe_count() << ",\n"
            << "    \"task_slots_per_pe\": " << config_.panel.task_slots_per_pe
            << ",\n"
            << "    \"tile_dim\": " << config_.panel.tile_dim << ",\n"
            << "    \"task_records\": "
            << (real_panel ? std::to_string(panel_engine_->task_records().size())
                           : std::string("0"))
            << ",\n"
            << "    \"control_fault\": "
            << (real_panel && panel_engine_->control_fault() ? "true" : "false")
            << "\n"
            << "  },\n"
            << "  \"config\": {\n"
            << "    \"max_resident_fronts\": " << config_.max_resident_fronts
            << ",\n"
            << "    \"panel_unit_count\": " << config_.panel_unit_count << ",\n"
            << "    \"trsm_unit_count\": " << config_.trsm_unit_count << ",\n"
            << "    \"gemm_unit_count\": " << config_.gemm_unit_count << ",\n"
            << "    \"solve_unit_count\": " << config_.solve_unit_count << ",\n"
            << "    \"assemble_unit_count\": " << config_.assemble_unit_count
            << ",\n"
            << "    \"factor_unit_count\": " << config_.factor_unit_count << ",\n"
            << "    \"buffer_slot_count\": " << config_.buffer_slot_count << ",\n"
            << "    \"buffer_capacity_bytes\": " << config_.buffer_capacity_bytes
            << ",\n"
            << "    \"command_window_depth\": " << config_.command_window_depth
            << "\n"
            << "  }\n"
            << "}\n";
    }
    if (panel_engine_ != nullptr) {
        std::ofstream out(output_dir + "/panel_tasks.csv");
        out << "task_id,command_id,node_id,step,tile_row,tile_col,mode,eff_m,eff_n,"
               "eff_k,pivot_base,group_id,pe_id,status,committed,cancelled,"
               "failed_column,enqueue_cycle,dispatch_cycle,exec_start_cycle,"
               "exec_done_cycle,writeback_done_cycle,bytes_read,bytes_written,"
               "gemm_backend,gemm_phase,gemm_exp_a,gemm_exp_b,saturation_count,"
               "overflow_count,local_conversion_bytes,scale_reduction_cycles,"
               "quantize_cycles,integer_compute_cycles,dequantize_cycles,reason\n";
        for (const auto& record : panel_engine_->task_records()) {
            out << record.task.task_id << ',' << record.task.command_id << ','
                << record.task.node_id << ',' << record.task.step << ','
                << record.task.tile_row << ',' << record.task.tile_col << ','
                << hw::panel_lu::tile_mode_name(record.task.mode) << ','
                << record.task.dims.m << ',' << record.task.dims.n << ','
                << record.task.dims.k << ',' << record.task.pivot_base << ','
                << record.group_id << ',' << record.pe_id << ','
                << hw::dataplane::status_name(record.status)
                << ',' << (record.committed ? 1 : 0) << ','
                << (record.cancelled ? 1 : 0) << ',' << record.failed_column << ','
                << record.enqueue_cycle << ',' << record.dispatch_cycle << ','
                << record.exec_start_cycle << ',' << record.exec_done_cycle << ','
                << record.writeback_done_cycle << ',' << record.bytes_read << ','
                << record.bytes_written << ','
                << hw::panel_lu::gemm_backend_name(record.gemm_backend) << ','
                << hw::panel_lu::gemm_phase_name(record.gemm_failed_phase) << ','
                << record.gemm_exponent_a << ',' << record.gemm_exponent_b << ','
                << record.saturation_count << ',' << record.overflow_count << ','
                << record.local_conversion_bytes << ','
                << record.scale_reduction_cycles << ',' << record.quantize_cycles
                << ',' << record.integer_compute_cycles << ','
                << record.dequantize_cycles << ',' << csv_field(record.reason) << '\n';
        }
    }
}

}  // namespace hw::controlplane
