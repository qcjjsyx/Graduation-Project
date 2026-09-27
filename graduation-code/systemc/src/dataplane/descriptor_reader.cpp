#include "dataplane/descriptor_reader.hpp"

#include <algorithm>
#include <cstring>

namespace hw::dataplane {

namespace {

namespace cmd = hw::command;

std::uint32_t element_bytes_of(cmd::DataFormat format) {
    return format == cmd::DataFormat::Fp64 ? 8 : 4;
}

bool is_device_format(cmd::DataFormat format) {
    return format == cmd::DataFormat::Fp32 || format == cmd::DataFormat::Int32;
}

std::uint64_t region_capacity_elements(const RegionView& view) {
    return static_cast<std::uint64_t>(view.rows) * view.cols;
}

bool require_fp32(const RegionView& view, const std::string& context, std::string* reason) {
    if (view.format != cmd::DataFormat::Fp32) {
        *reason = context + " must use FP32";
        return false;
    }
    return true;
}

bool require_int32(const RegionView& view, const std::string& context, std::string* reason) {
    if (view.format != cmd::DataFormat::Int32) {
        *reason = context + " must use INT32";
        return false;
    }
    return true;
}

bool known_kernel_backend(cmd::KernelBackend backend) {
    return backend == cmd::KernelBackend::SystemCFp32DeviceModel ||
           backend == cmd::KernelBackend::SystemCInt32GemmModel;
}

bool known_solve_direction(cmd::SolveDirection direction) {
    return direction == cmd::SolveDirection::Forward ||
           direction == cmd::SolveDirection::Backward;
}

std::vector<std::uint32_t> decode_u32_payload(const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint32_t> values;
    values.reserve(payload.size() / 4);
    for (std::size_t offset = 0; offset + 4 <= payload.size(); offset += 4) {
        values.push_back(
            static_cast<std::uint32_t>(payload[offset]) |
            (static_cast<std::uint32_t>(payload[offset + 1]) << 8) |
            (static_cast<std::uint32_t>(payload[offset + 2]) << 16) |
            (static_cast<std::uint32_t>(payload[offset + 3]) << 24));
    }
    return values;
}

float decode_threshold_bits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

const char* read_kind_name(ReadKind kind) {
    switch (kind) {
    case ReadKind::Region: return "region";
    case ReadKind::Front: return "front";
    case ReadKind::Factor: return "factor";
    case ReadKind::Contribution: return "contribution";
    case ReadKind::Kernel: return "kernel";
    case ReadKind::Solve: return "solve";
    case ReadKind::Dependency: return "dependency";
    }
    return "unknown";
}

DescriptorReader::DescriptorReader(
    MemoryService& memory,
    const DeviceBootConfig& boot,
    TraceRecorder* trace)
    : memory_(memory), boot_(boot), trace_(trace) {}

std::uint64_t DescriptorReader::request(
    ReadKind kind,
    std::uint32_t descriptor_id,
    std::uint64_t cycle) {
    Task task;
    task.request_id = next_request_id_++;
    task.kind = kind;
    task.descriptor_id = descriptor_id;
    task.issue_cycle = cycle;
    const auto request_id = task.request_id;
    tasks_.emplace(request_id, std::move(task));
    submit_descriptor_read(tasks_.at(request_id), descriptor_id, cycle);
    return request_id;
}

bool DescriptorReader::pending(std::uint64_t request_id) const {
    const auto iterator = tasks_.find(request_id);
    return iterator != tasks_.end() && iterator->second.stage != Stage::Done;
}

bool DescriptorReader::poll(std::uint64_t request_id, ReadOutcome& outcome) {
    for (auto iterator = outcomes_.begin(); iterator != outcomes_.end(); ++iterator) {
        if (iterator->request_id == request_id) {
            outcome = *iterator;
            outcomes_.erase(iterator);
            tasks_.erase(request_id);
            return true;
        }
    }
    return false;
}

DescriptorReader::Task* DescriptorReader::find_task(std::uint64_t request_id) {
    const auto iterator = tasks_.find(request_id);
    return iterator == tasks_.end() ? nullptr : &iterator->second;
}

void DescriptorReader::submit_descriptor_read(
    Task& task,
    std::uint32_t descriptor_id,
    std::uint64_t cycle) {
    if (descriptor_id >= boot_.descriptor_count) {
        finish(
            task, cmd::StatusCode::BadDescriptor,
            "descriptor id out of range", cycle);
        return;
    }
    MemoryRequest request;
    request.request_id = memory_.next_request_id();
    request.client = MemoryClient::DescriptorReader;
    request.write = false;
    request.address =
        boot_.descriptor_table_addr +
        static_cast<std::uint64_t>(descriptor_id) * cmd::DESCRIPTOR_RECORD_BYTES;
    request.length = cmd::DESCRIPTOR_RECORD_BYTES;
    request.kind = AccessKind::Descriptor;
    request.label = "descriptor_table";
    if (!memory_.submit(request, cycle)) {
        task.waiting_submit = true;
        return;
    }
    task.waiting_submit = false;
    task.waiting_memory = true;
    task.pending_memory_id = request.request_id;
    memory_to_task_[request.request_id] = task.request_id;
}

void DescriptorReader::submit_payload_read(Task& task, std::uint64_t cycle) {
    MemoryRequest request;
    request.request_id = memory_.next_request_id();
    request.client = MemoryClient::DescriptorReader;
    request.write = false;
    request.address = task.payload_address;
    request.length = task.payload_length;
    request.kind = AccessKind::Payload;
    request.label = "descriptor_payload";
    if (!memory_.submit(request, cycle)) {
        task.waiting_submit = true;
        return;
    }
    task.waiting_submit = false;
    task.waiting_memory = true;
    task.pending_memory_id = request.request_id;
    memory_to_task_[request.request_id] = task.request_id;
}

void DescriptorReader::tick(std::uint64_t cycle) {
    std::vector<std::uint64_t> waiting;
    waiting.reserve(tasks_.size());
    for (const auto& entry : tasks_) {
        if (entry.second.waiting_submit && entry.second.stage != Stage::Done) {
            waiting.push_back(entry.first);
        }
    }
    for (const auto request_id : waiting) {
        Task* task = find_task(request_id);
        if (task == nullptr || !task->waiting_submit) {
            continue;
        }
        if (task->stage == Stage::Payload) {
            submit_payload_read(*task, cycle);
        } else if (task->stage == Stage::Record) {
            submit_descriptor_read(*task, task->descriptor_id, cycle);
        } else if (task->stage == Stage::References) {
            if (task->next_ref < task->refs.size()) {
                submit_descriptor_read(*task, task->refs[task->next_ref].id, cycle);
            } else if (task->payload_pending) {
                task->stage = Stage::Payload;
                submit_payload_read(*task, cycle);
            } else {
                finish(*task, cmd::StatusCode::Ok, "", cycle);
            }
        }
    }
}

void DescriptorReader::plan_references(Task& task) {
    const auto body = task.record.body_words;
    switch (task.kind) {
    case ReadKind::Region:
        break;
    case ReadKind::Front:
        task.refs.push_back(Ref{body[0]});
        task.refs.push_back(Ref{body[6]});
        task.refs.push_back(Ref{body[7]});
        task.refs.push_back(Ref{body[5]});
        if (body[4] != cmd::NONE) {
            task.refs.push_back(Ref{body[4]});
        }
        break;
    case ReadKind::Factor:
        task.refs.push_back(Ref{body[0]});
        task.refs.push_back(Ref{body[1]});
        task.refs.push_back(Ref{body[2]});
        if (body[6] != cmd::NONE) {
            task.refs.push_back(Ref{body[6]});
        }
        break;
    case ReadKind::Contribution:
        task.refs.push_back(Ref{body[0]});
        task.refs.push_back(Ref{body[1]});
        if (task.record.payload_bytes != 0) {
            task.payload_pending = true;
            task.payload_address = task.record.payload_offset;
            task.payload_length = task.record.payload_bytes;
        }
        break;
    case ReadKind::Kernel:
        task.refs.push_back(Ref{body[0]});
        task.refs.push_back(Ref{body[1]});
        task.refs.push_back(Ref{body[2]});
        if (body[8] != cmd::NONE) {
            task.refs.push_back(Ref{body[8]});
        }
        break;
    case ReadKind::Solve:
        for (std::size_t index = 0; index < 5; ++index) {
            task.refs.push_back(Ref{body[index]});
        }
        break;
    case ReadKind::Dependency:
        if (task.record.payload_bytes != 0) {
            task.payload_pending = true;
            task.payload_address = task.record.payload_offset;
            task.payload_length = task.record.payload_bytes;
        }
        break;
    }
}

void DescriptorReader::on_record(
    Task& task,
    const MemoryResponse& response,
    std::uint64_t cycle) {
    try {
        task.record = cmd::DescriptorCodec::decode(
            response.read_data.data(), response.read_data.size(), boot_.total_bytes);
    } catch (const std::exception& exception) {
        finish(
            task, cmd::StatusCode::BadDescriptor,
            std::string("descriptor decode failed: ") + exception.what(), cycle);
        return;
    }
    const cmd::DescriptorType* expected = nullptr;
    static const cmd::DescriptorType kFront = cmd::DescriptorType::Front;
    static const cmd::DescriptorType kFactor = cmd::DescriptorType::Factor;
    static const cmd::DescriptorType kContribution = cmd::DescriptorType::Contribution;
    static const cmd::DescriptorType kKernel = cmd::DescriptorType::Kernel;
    static const cmd::DescriptorType kSolve = cmd::DescriptorType::Solve;
    static const cmd::DescriptorType kDependency = cmd::DescriptorType::Dependency;
    switch (task.kind) {
    case ReadKind::Region: break;
    case ReadKind::Front: expected = &kFront; break;
    case ReadKind::Factor: expected = &kFactor; break;
    case ReadKind::Contribution: expected = &kContribution; break;
    case ReadKind::Kernel: expected = &kKernel; break;
    case ReadKind::Solve: expected = &kSolve; break;
    case ReadKind::Dependency: expected = &kDependency; break;
    }
    if (expected != nullptr && task.record.descriptor_type != *expected) {
        finish(
            task, cmd::StatusCode::BadDescriptor,
            "descriptor type does not match the requested view", cycle);
        return;
    }
    if (task.kind == ReadKind::Region) {
        finish(task, cmd::StatusCode::Ok, "", cycle);
        return;
    }
    plan_references(task);
    task.stage = Stage::References;
    if (!task.refs.empty()) {
        submit_descriptor_read(task, task.refs[0].id, cycle);
        return;
    }
    if (task.payload_pending) {
        task.stage = Stage::Payload;
        submit_payload_read(task, cycle);
        return;
    }
    finish(task, cmd::StatusCode::Ok, "", cycle);
}

void DescriptorReader::on_reference(
    Task& task,
    const MemoryResponse& response,
    std::uint64_t cycle) {
    Ref ref;
    ref.id = task.refs[task.next_ref].id;
    try {
        ref.descriptor = cmd::DescriptorCodec::decode(
            response.read_data.data(), response.read_data.size(), boot_.total_bytes);
    } catch (const std::exception& exception) {
        finish(
            task, cmd::StatusCode::BadDescriptor,
            std::string("referenced descriptor decode failed: ") + exception.what(),
            cycle);
        return;
    }
    if (ref.descriptor.descriptor_type == cmd::DescriptorType::Region) {
        std::string reason;
        if (!build_region_view(ref.id, ref.descriptor, ref.region, &reason)) {
            finish(task, cmd::StatusCode::BadDescriptor, reason, cycle);
            return;
        }
        ref.is_region = true;
    }
    if (task.kind == ReadKind::Kernel &&
        ref.descriptor.descriptor_type == cmd::DescriptorType::Front) {
        // Kernel operands are FRONT_DESC; resolve each front region as well.
        task.refs.push_back(Ref{ref.descriptor.body_words[0]});
    }
    task.refs[task.next_ref] = std::move(ref);
    ++task.next_ref;
    if (task.next_ref < task.refs.size()) {
        submit_descriptor_read(task, task.refs[task.next_ref].id, cycle);
        return;
    }
    if (task.payload_pending) {
        task.stage = Stage::Payload;
        submit_payload_read(task, cycle);
        return;
    }
    finish(task, cmd::StatusCode::Ok, "", cycle);
}

void DescriptorReader::on_payload(
    Task& task,
    const MemoryResponse& response,
    std::uint64_t cycle) {
    task.payload = response.read_data;
    finish(task, cmd::StatusCode::Ok, "", cycle);
}

bool DescriptorReader::build_region_view(
    std::uint32_t descriptor_id,
    const cmd::Descriptor& descriptor,
    RegionView& out,
    std::string* reason) const {
    if (descriptor.descriptor_type != cmd::DescriptorType::Region) {
        *reason = "reference is not a REGION_DESC";
        return false;
    }
    const auto body = descriptor.body_words;
    out = RegionView{};
    out.descriptor_id = descriptor_id;
    out.base_addr = static_cast<std::uint64_t>(body[0]) |
                    (static_cast<std::uint64_t>(body[1]) << 32);
    out.byte_size = static_cast<std::uint64_t>(body[2]) |
                    (static_cast<std::uint64_t>(body[3]) << 32);
    out.row_stride = body[4];
    out.rows = body[5];
    out.cols = body[6];
    out.format = static_cast<cmd::DataFormat>(body[7]);
    out.layout = static_cast<cmd::DataLayout>(body[8]);
    if (!is_device_format(out.format)) {
        *reason = "region data format is not FP32/INT32";
        return false;
    }
    if (out.layout != cmd::DataLayout::RowMajor) {
        *reason = "region layout is not row-major";
        return false;
    }
    out.element_bytes = element_bytes_of(out.format);
    const auto kind = out.format == cmd::DataFormat::Int32 ? AccessKind::Int32Data
                                                          : AccessKind::Fp32Data;
    const auto checked = memory_.check(out.base_addr, out.byte_size, kind, false);
    if (!checked.ok) {
        *reason = checked.reason;
        return false;
    }
    for (const auto& region : boot_.regions) {
        if (region.offset == out.base_addr && region.size == out.byte_size) {
            if (region.role == RegionRole::Command ||
                region.role == RegionRole::Descriptor ||
                region.role == RegionRole::Payload) {
                *reason = "region references control-only memory";
                return false;
            }
            out.name = region.name;
            return true;
        }
    }
    *reason = "region is not registered in the manifest region table";
    return false;
}

void DescriptorReader::finish(
    Task& task,
    cmd::StatusCode status,
    std::string reason,
    std::uint64_t cycle) {
    ReadOutcome outcome;
    outcome.request_id = task.request_id;
    outcome.kind = task.kind;
    outcome.status = status;
    outcome.reason = std::move(reason);
    outcome.issue_cycle = task.issue_cycle;
    outcome.finish_cycle = cycle;
    outcome.read_bytes = task.read_bytes;
    outcome.memory_reads = task.memory_reads;
    outcome.descriptor = task.record;
    if (status == cmd::StatusCode::Ok) {
        std::string reason_text;
        if (!finalize(task, outcome, &reason_text)) {
            outcome.status = cmd::StatusCode::BadDescriptor;
            outcome.reason = std::move(reason_text);
        }
    }
    if (outcome.status == cmd::StatusCode::Ok) {
        ++completed_count_;
    } else {
        ++error_count_;
    }
    task.waiting_submit = false;
    task.waiting_memory = false;
    task.stage = Stage::Done;
    outcomes_.push_back(std::move(outcome));
}

bool DescriptorReader::finalize(
    Task& task,
    ReadOutcome& outcome,
    std::string* reason) {
    const auto body = task.record.body_words;
    switch (task.kind) {
    case ReadKind::Region:
        return build_region_view(task.descriptor_id, task.record, outcome.region, reason);
    case ReadKind::Front: {
        FrontView view;
        view.descriptor_id = task.descriptor_id;
        view.front = task.refs[0].region;
        view.p = task.refs[1].region;
        view.workspace = task.refs[2].region;
        view.total_dim = body[1];
        view.pivot_dim = body[2];
        view.tile_size = body[3];
        view.contribution_descriptor_id = body[4];
        view.has_contribution = body[4] != cmd::NONE;
        view.factor_descriptor_id = body[5];
        view.policy = static_cast<cmd::PivotPolicy>(body[8]);
        view.threshold_bits = body[9];
        view.threshold = decode_threshold_bits(body[9]);
        if (view.total_dim == 0 || view.pivot_dim == 0 ||
            view.pivot_dim > view.total_dim || view.tile_size == 0) {
            *reason = "FRONT_DESC dimensions are invalid";
            return false;
        }
        if (!require_fp32(view.front, "front region", reason)) return false;
        if (view.front.rows < view.total_dim || view.front.cols < view.total_dim) {
            *reason = "front region is smaller than total_dim";
            return false;
        }
        if (!require_int32(view.p, "front P-vector", reason)) return false;
        if (region_capacity_elements(view.p) != view.pivot_dim ||
            view.p.byte_size != static_cast<std::uint64_t>(view.pivot_dim) * 4) {
            *reason = "front P-vector capacity must equal pivot_dim";
            return false;
        }
        if (!require_fp32(view.workspace, "front workspace", reason)) return false;
        if (region_capacity_elements(view.workspace) <
            static_cast<std::uint64_t>(2) * view.pivot_dim) {
            *reason = "front workspace is too small";
            return false;
        }
        if (task.refs.size() < 4 ||
            task.refs[3].descriptor.descriptor_type != cmd::DescriptorType::Factor) {
            *reason = "front factor reference is not a FACTOR_DESC";
            return false;
        }
        if (task.refs[3].descriptor.body_words[2] != task.refs[1].id) {
            *reason = "FRONT_DESC and FACTOR_DESC must reference one P-vector";
            return false;
        }
        if (task.refs[3].descriptor.body_words[3] != view.total_dim ||
            task.refs[3].descriptor.body_words[4] != view.pivot_dim) {
            *reason = "factor dimensions disagree with the front";
            return false;
        }
        if (view.has_contribution &&
            (task.refs.size() < 5 ||
             task.refs[4].descriptor.descriptor_type !=
                 cmd::DescriptorType::Contribution)) {
            *reason = "front contribution reference is not a CONTRIBUTION_DESC";
            return false;
        }
        outcome.front = view;
        return true;
    }
    case ReadKind::Factor: {
        FactorView view;
        view.descriptor_id = task.descriptor_id;
        view.l = task.refs[0].region;
        view.u = task.refs[1].region;
        view.p = task.refs[2].region;
        view.has_workspace = body[6] != cmd::NONE;
        if (view.has_workspace) {
            view.workspace = task.refs[3].region;
        }
        view.total_dim = body[3];
        view.pivot_dim = body[4];
        view.format = static_cast<cmd::DataFormat>(body[5]);
        if (view.total_dim == 0 || view.pivot_dim == 0 ||
            view.pivot_dim > view.total_dim) {
            *reason = "FACTOR_DESC dimensions are invalid";
            return false;
        }
        if (!require_fp32(view.l, "factor L region", reason)) return false;
        if (!require_fp32(view.u, "factor U region", reason)) return false;
        if (view.l.rows < view.total_dim || view.l.cols < view.pivot_dim) {
            *reason = "factor L region is too small";
            return false;
        }
        if (view.u.rows < view.pivot_dim || view.u.cols < view.total_dim) {
            *reason = "factor U region is too small";
            return false;
        }
        if (!require_int32(view.p, "factor P-vector", reason)) return false;
        if (region_capacity_elements(view.p) != view.pivot_dim ||
            view.p.byte_size != static_cast<std::uint64_t>(view.pivot_dim) * 4) {
            *reason = "factor P-vector capacity must equal pivot_dim";
            return false;
        }
        if (view.has_workspace) {
            if (!require_fp32(view.workspace, "factor workspace", reason)) return false;
            if (region_capacity_elements(view.workspace) <
                static_cast<std::uint64_t>(2) * view.pivot_dim) {
                *reason = "factor workspace is too small";
                return false;
            }
        }
        outcome.factor = view;
        return true;
    }
    case ReadKind::Contribution: {
        ContributionView view;
        view.descriptor_id = task.descriptor_id;
        view.source = task.refs[0].region;
        view.target = task.refs[1].region;
        view.child_id = body[2];
        view.parent_id = body[3];
        view.row_count = body[4];
        view.col_count = body[5];
        view.source_stride = body[6];
        view.target_stride = body[7];
        view.format = static_cast<cmd::DataFormat>(body[8]);
        if (task.payload.size() !=
            static_cast<std::size_t>(4) * (view.row_count + view.col_count)) {
            *reason = "contribution payload size does not match the map counts";
            return false;
        }
        const auto maps = decode_u32_payload(task.payload);
        view.row_map.assign(maps.begin(), maps.begin() + view.row_count);
        view.col_map.assign(
            maps.begin() + view.row_count, maps.begin() + view.row_count + view.col_count);
        if (!require_fp32(view.source, "contribution source", reason)) return false;
        if (!require_fp32(view.target, "contribution target", reason)) return false;
        if (view.row_count > view.source.rows || view.col_count > view.target.cols) {
            *reason = "contribution map counts exceed the region dimensions";
            return false;
        }
        if (view.source_stride < static_cast<std::uint64_t>(view.col_count) * 4 ||
            view.source_stride > view.source.row_stride ||
            view.target_stride < static_cast<std::uint64_t>(view.col_count) * 4 ||
            view.target_stride > view.target.row_stride) {
            *reason = "contribution stride is inconsistent with the regions";
            return false;
        }
        for (const auto value : view.row_map) {
            if (value >= view.source.rows) {
                *reason = "contribution row map is out of range";
                return false;
            }
        }
        for (const auto value : view.col_map) {
            if (value >= view.target.cols) {
                *reason = "contribution column map is out of range";
                return false;
            }
        }
        outcome.contribution = view;
        return true;
    }
    case ReadKind::Kernel: {
        KernelView view;
        view.descriptor_id = task.descriptor_id;
        std::vector<const Ref*> front_regions;
        for (const auto& ref : task.refs) {
            if (ref.is_region) {
                front_regions.push_back(&ref);
            }
        }
        if (front_regions.size() != 3) {
            *reason = "kernel operands must resolve to three front regions";
            return false;
        }
        view.a = front_regions[0]->region;
        view.b = front_regions[1]->region;
        view.c = front_regions[2]->region;
        view.m = body[3];
        view.n = body[4];
        view.k = body[5];
        view.backend = static_cast<cmd::KernelBackend>(body[6]);
        view.format = static_cast<cmd::DataFormat>(body[7]);
        view.scale_descriptor_id = body[8];
        view.has_scale = body[8] != cmd::NONE;
        view.tile_size = body[9];
        if (view.m == 0 || view.n == 0 || view.k == 0 || view.tile_size == 0) {
            *reason = "KERNEL_DESC dimensions are invalid";
            return false;
        }
        if (!known_kernel_backend(view.backend)) {
            *reason = "KERNEL_DESC backend is unknown";
            return false;
        }
        if (view.format != cmd::DataFormat::Fp32) {
            *reason = "KERNEL_DESC format must be FP32";
            return false;
        }
        if (!require_fp32(view.a, "kernel A", reason) ||
            !require_fp32(view.b, "kernel B", reason) ||
            !require_fp32(view.c, "kernel C", reason)) {
            return false;
        }
        if (view.a.rows < view.m || view.a.cols < view.k ||
            view.b.rows < view.k || view.b.cols < view.n ||
            view.c.rows < view.m || view.c.cols < view.n) {
            *reason = "kernel operand regions are smaller than M/N/K";
            return false;
        }
        if (view.has_scale) {
            const auto found = std::find_if(
                task.refs.begin(), task.refs.end(),
                [&](const Ref& ref) { return ref.id == view.scale_descriptor_id; });
            if (found == task.refs.end() ||
                found->descriptor.descriptor_type != cmd::DescriptorType::Scale) {
                *reason = "kernel scale reference is not a SCALE_DESC";
                return false;
            }
        }
        outcome.kernel = view;
        return true;
    }
    case ReadKind::Solve: {
        SolveView view;
        view.descriptor_id = task.descriptor_id;
        if (task.refs[0].descriptor.descriptor_type != cmd::DescriptorType::Factor) {
            *reason = "solve factor reference is not a FACTOR_DESC";
            return false;
        }
        const auto factor_body = task.refs[0].descriptor.body_words;
        view.factor.descriptor_id = task.refs[0].id;
        view.factor.total_dim = factor_body[3];
        view.factor.pivot_dim = factor_body[4];
        view.factor.format = static_cast<cmd::DataFormat>(factor_body[5]);
        view.rhs = task.refs[1].region;
        view.solution = task.refs[2].region;
        view.p = task.refs[3].region;
        view.workspace = task.refs[4].region;
        view.rhs_count = body[5];
        view.direction = static_cast<cmd::SolveDirection>(body[6]);
        view.format = static_cast<cmd::DataFormat>(body[7]);
        if (view.rhs_count == 0 || view.factor.total_dim == 0 ||
            view.factor.pivot_dim == 0) {
            *reason = "SOLVE_DESC dimensions are invalid";
            return false;
        }
        if (!known_solve_direction(view.direction)) {
            *reason = "SOLVE_DESC direction is unknown";
            return false;
        }
        if (view.format != cmd::DataFormat::Fp32) {
            *reason = "SOLVE_DESC format must be FP32";
            return false;
        }
        if (body[3] != factor_body[2]) {
            *reason = "SOLVE_DESC and FACTOR_DESC must reference one P-vector";
            return false;
        }
        if (!require_fp32(view.rhs, "solve RHS region", reason)) return false;
        if (!require_fp32(view.solution, "solve solution region", reason)) return false;
        if (view.rhs.rows < view.rhs_count ||
            view.rhs.cols < view.factor.total_dim ||
            view.solution.rows < view.rhs_count ||
            view.solution.cols < view.factor.total_dim) {
            *reason = "solve RHS/solution regions are too small";
            return false;
        }
        if (!require_int32(view.p, "solve P-vector", reason)) return false;
        if (region_capacity_elements(view.p) != view.factor.pivot_dim ||
            view.p.byte_size !=
                static_cast<std::uint64_t>(view.factor.pivot_dim) * 4) {
            *reason = "solve P-vector capacity must equal pivot_dim";
            return false;
        }
        if (!require_fp32(view.workspace, "solve workspace", reason)) return false;
        if (region_capacity_elements(view.workspace) <
            static_cast<std::uint64_t>(2) * view.factor.pivot_dim) {
            *reason = "solve workspace is too small";
            return false;
        }
        outcome.solve = view;
        return true;
    }
    case ReadKind::Dependency: {
        DependencyView view;
        view.descriptor_id = task.descriptor_id;
        const auto count = body[0];
        if (count > boot_.max_wait_tokens) {
            *reason = "dependency token count exceeds max_wait_tokens";
            return false;
        }
        if (task.payload.size() != static_cast<std::size_t>(4) * count) {
            *reason = "dependency payload size does not match the token count";
            return false;
        }
        view.tokens = decode_u32_payload(task.payload);
        std::vector<std::uint32_t> sorted = view.tokens;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            *reason = "dependency token list contains duplicates";
            return false;
        }
        for (const auto token : view.tokens) {
            if (boot_.token_count != 0 && token >= boot_.token_count) {
                *reason = "dependency token list contains an unknown token";
                return false;
            }
        }
        outcome.dependency = view;
        return true;
    }
    }
    *reason = "unsupported descriptor kind";
    return false;
}

void DescriptorReader::deliver_memory_response(const MemoryResponse& response) {
    const auto mapping = memory_to_task_.find(response.request_id);
    if (mapping == memory_to_task_.end()) {
        return;  // stale response (e.g. after reset) or not ours
    }
    const std::uint64_t request_id = mapping->second;
    memory_to_task_.erase(mapping);
    Task* task = find_task(request_id);
    if (task == nullptr) {
        return;
    }
    task->waiting_memory = false;
    task->read_bytes += response.bytes;
    task->memory_reads += 1;
    read_bytes_ += response.bytes;
    memory_reads_ += 1;
    trace_read(response);
    if (response.status != cmd::StatusCode::Ok) {
        finish(*task, response.status, response.reason, response.finish_cycle);
        return;
    }
    switch (task->stage) {
    case Stage::Record:
        on_record(*task, response, response.finish_cycle);
        break;
    case Stage::References:
        on_reference(*task, response, response.finish_cycle);
        break;
    case Stage::Payload:
        on_payload(*task, response, response.finish_cycle);
        break;
    case Stage::Done:
        break;
    }
}

void DescriptorReader::trace_read(const MemoryResponse& response) const {
    if (trace_ == nullptr) {
        return;
    }
    TransactionRow row;
    row.layer = "descriptor";
    row.client = memory_client_name(MemoryClient::DescriptorReader);
    row.request_id = response.request_id;
    row.region = response.request.label;
    row.address = response.request.address;
    row.length = response.request.length;
    row.write = false;
    row.status = response.status;
    row.reason = response.reason;
    row.issue_cycle = response.issue_cycle;
    row.accept_cycle = response.accept_cycle;
    row.data_cycle = response.finish_cycle;
    row.bytes = response.bytes;
    row.request_wait_cycles = response.request_wait_cycles;
    row.memory_service_cycles = response.service_cycles;
    trace_->add_transaction(row);
}

void DescriptorReader::reset() {
    tasks_.clear();
    memory_to_task_.clear();
    outcomes_.clear();
}

}  // namespace hw::dataplane
