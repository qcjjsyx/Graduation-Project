#include "dataplane/dataplane.hpp"

#include <utility>

namespace hw::dataplane {

DataPlane::DataPlane(LoadedArtifact artifact, DataPlaneConfig config)
    : artifact_(std::move(artifact)),
      memory_(artifact_.image, artifact_.config.regions),
      memory_service_(memory_, config.memory, &trace_),
      buffers_(config.buffers, &trace_),
      dma_(config.dma, memory_service_, buffers_, &trace_),
      reader_(memory_service_, artifact_.config, &trace_) {}

std::uint64_t DataPlane::begin_tick() {
    ++cycle_;
    return cycle_;
}

void DataPlane::tick_memory_and_route() {
    memory_service_.tick(cycle_);
    MemoryResponse response;
    while (memory_service_.try_pop_response(response)) {
        if (response.epoch != memory_service_.epoch()) {
            continue;  // response invalidated by reset
        }
        switch (response.client) {
        case MemoryClient::DescriptorReader:
            reader_.deliver_memory_response(response);
            break;
        case MemoryClient::Dma:
            dma_.deliver_memory_response(response);
            break;
        case MemoryClient::CommandFetch:
            if (command_fetch_sink_ != nullptr) {
                command_fetch_sink_->deliver_memory_response(response);
            }
            break;
        case MemoryClient::CompletionWriter:
            if (completion_writer_sink_ != nullptr) {
                completion_writer_sink_->deliver_memory_response(response);
            }
            break;
        case MemoryClient::ExecUnit:
            if (exec_unit_sink_ != nullptr) {
                exec_unit_sink_->deliver_memory_response(response);
            }
            break;
        case MemoryClient::Test:
        case MemoryClient::Other:
            break;
        }
    }
}

void DataPlane::tick_clients() {
    reader_.tick(cycle_);
    dma_.tick(cycle_);
}

std::uint64_t DataPlane::step() {
    begin_tick();
    tick_memory_and_route();
    tick_clients();
    return cycle_;
}

void DataPlane::set_memory_sink(MemoryClient client, MemoryResponseSink* sink) {
    switch (client) {
    case MemoryClient::CommandFetch:
        command_fetch_sink_ = sink;
        break;
    case MemoryClient::CompletionWriter:
        completion_writer_sink_ = sink;
        break;
    case MemoryClient::ExecUnit:
        exec_unit_sink_ = sink;
        break;
    default:
        break;
    }
}

bool DataPlane::run_until(
    const std::function<bool()>& done,
    std::uint64_t max_cycles) {
    if (done()) {
        return true;
    }
    const std::uint64_t deadline = cycle_ + max_cycles;
    while (cycle_ < deadline) {
        step();
        if (done()) {
            return true;
        }
    }
    return false;
}

void DataPlane::reset() {
    dma_.reset();
    buffers_.reset();
    reader_.reset();
    memory_service_.reset();
    command_fetch_sink_ = nullptr;
    completion_writer_sink_ = nullptr;
    exec_unit_sink_ = nullptr;
    trace_.clear();
}

bool DataPlane::run_dma_request(
    const DmaRequest& request,
    DmaResponse& response,
    std::uint64_t max_cycles) {
    dma_.set_request(request);
    dma_.set_request_valid(true);
    const bool got_response = run_until(
        [this]() { return dma_.response_valid(); }, max_cycles);
    if (!got_response) {
        dma_.set_request_valid(false);
        return false;
    }
    response = dma_.response();
    dma_.set_request_valid(false);
    dma_.set_response_ready(true);
    step();  // consume the response (one terminal response per request)
    dma_.set_response_ready(false);
    return true;
}

}  // namespace hw::dataplane
