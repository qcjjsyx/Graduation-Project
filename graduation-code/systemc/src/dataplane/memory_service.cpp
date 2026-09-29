#include "dataplane/memory_service.hpp"

#include <algorithm>

namespace hw::dataplane {

const char* memory_client_name(MemoryClient client) {
    switch (client) {
    case MemoryClient::DescriptorReader: return "descriptor_reader";
    case MemoryClient::Dma: return "dma";
    case MemoryClient::CommandFetch: return "command_fetch";
    case MemoryClient::CompletionWriter: return "completion_writer";
    case MemoryClient::ExecUnit: return "exec_unit";
    case MemoryClient::Test: return "test";
    case MemoryClient::Other: return "other";
    }
    return "unknown";
}

MemoryService::MemoryService(
    DeviceMemory& memory,
    MemoryServiceConfig config,
    TraceRecorder* trace)
    : memory_(memory), config_(config), trace_(trace) {}

bool MemoryService::submit(const MemoryRequest& request, std::uint64_t issue_cycle) {
    MemoryResponse response;
    response.request_id = request.request_id;
    response.client = request.client;
    response.issue_cycle = issue_cycle;
    response.accept_cycle = issue_cycle;
    response.epoch = epoch_;
    response.request = request;

    // Validation first: an illegal access never modifies memory and never
    // occupies the data port.
    const auto checked = memory_.check(
        request.address,
        request.length,
        request.kind,
        request.write);
    if (!checked.ok) {
        response.status = checked.status;
        response.reason = checked.reason;
        response.bytes = 0;
        ++error_count_;
        responses_.push_back(std::move(response));
        emit_trace(responses_.back());
        return true;
    }

    if (request.write && request.write_data.size() != request.length) {
        // Declared write length and payload must agree; otherwise the caller
        // could modify more bytes than the request advertises.
        response.status = hw::command::StatusCode::BadCommand;
        response.reason = "write data length does not match request length";
        response.bytes = 0;
        ++error_count_;
        responses_.push_back(std::move(response));
        emit_trace(responses_.back());
        return true;
    }

    if (checked.zero_length) {
        // Recognizable successful no-op: no memory traffic, no port occupancy.
        response.status = hw::command::StatusCode::Ok;
        response.bytes = 0;
        accepted_count_ += 1;
        responses_.push_back(std::move(response));
        emit_trace(responses_.back());
        return true;
    }

    if (inflight_.size() >= config_.max_inflight) {
        return false;  // normal backpressure; no side effect
    }

    const std::uint64_t beats =
        (request.length + config_.beat_bytes - 1) / config_.beat_bytes;
    const std::uint64_t service =
        config_.startup_cycles + beats * config_.beat_cycles;

    Inflight entry;
    entry.response = std::move(response);
    entry.response.status = hw::command::StatusCode::Ok;
    entry.response.bytes = request.length;
    entry.response.start_cycle = std::max(entry.response.issue_cycle, port_free_cycle_);
    entry.response.finish_cycle = entry.response.start_cycle + service;
    entry.response.request_wait_cycles =
        entry.response.start_cycle - entry.response.issue_cycle;
    entry.response.service_cycles = service;
    entry.write_data = request.write_data;

    port_free_cycle_ = entry.response.finish_cycle;
    port_busy_cycles_ += service;
    if (request.write) {
        write_bytes_ += request.length;
    } else {
        read_bytes_ += request.length;
    }
    ++accepted_count_;
    inflight_.push_back(std::move(entry));
    return true;
}

void MemoryService::tick(std::uint64_t cycle) {
    while (!inflight_.empty() &&
           inflight_.front().response.finish_cycle <= cycle) {
        Inflight entry = std::move(inflight_.front());
        inflight_.pop_front();

        MemoryResponse& response = entry.response;
        if (response.request.write) {
            std::string reason;
            const auto status = memory_.write(
                response.request.address,
                entry.write_data,
                response.request.kind,
                &reason);
            if (status != hw::command::StatusCode::Ok) {
                response.status = status;
                response.reason = std::move(reason);
                response.bytes = 0;
            }
        } else {
            std::string reason;
            const auto status = memory_.read(
                response.request.address,
                response.request.length,
                response.request.kind,
                response.read_data,
                &reason);
            if (status != hw::command::StatusCode::Ok) {
                response.status = status;
                response.reason = std::move(reason);
                response.bytes = 0;
                response.read_data.clear();
            }
        }
        responses_.push_back(std::move(response));
        emit_trace(responses_.back());
    }
}

bool MemoryService::try_pop_response(MemoryResponse& out) {
    if (responses_.empty()) {
        return false;
    }
    out = std::move(responses_.front());
    responses_.pop_front();
    return true;
}

void MemoryService::reset() {
    inflight_.clear();
    responses_.clear();
    ++epoch_;
    port_free_cycle_ = 0;
}

void MemoryService::emit_trace(const MemoryResponse& done) {
    if (trace_ == nullptr) {
        return;
    }
    TransactionRow row;
    row.layer = "memory";
    row.client = memory_client_name(done.client);
    row.request_id = done.request_id;
    row.has_command = done.request.has_command;
    row.command_id = done.request.command_id;
    row.node_id = done.request.node_id;
    row.region = done.request.label;
    row.address = done.request.address;
    row.length = done.request.length;
    row.write = done.request.write;
    row.status = done.status;
    row.reason = done.reason;
    row.issue_cycle = done.issue_cycle;
    row.accept_cycle = done.accept_cycle;
    row.data_cycle = done.finish_cycle;
    row.bytes = done.bytes;
    row.request_wait_cycles = done.request_wait_cycles;
    row.memory_service_cycles = done.service_cycles;
    trace_->add_transaction(std::move(row));
}

}  // namespace hw::dataplane
