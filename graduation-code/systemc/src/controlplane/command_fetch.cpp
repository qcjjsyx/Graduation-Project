#include "controlplane/command_fetch.hpp"

#include <stdexcept>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

CommandFetch::CommandFetch(
    dp::MemoryService& memory,
    Config config,
    dp::TraceRecorder* trace)
    : memory_(memory), config_(config), trace_(trace) {}

void CommandFetch::deliver_memory_response(const dp::MemoryResponse& response) {
    inbox_.push_back(response);
}

void CommandFetch::fault(const std::string& reason, std::uint64_t cycle) {
    if (!faulted_) {
        faulted_ = true;
        fault_reason_ = reason;
    }
    if (trace_ != nullptr) {
        dp::TransactionRow row;
        row.layer = "command_fetch";
        row.client = dp::memory_client_name(dp::MemoryClient::CommandFetch);
        row.request_id = 0;
        row.region = "command_buffer";
        row.write = false;
        row.status = cmd::StatusCode::BadCommand;
        row.reason = reason;
        row.issue_cycle = cycle;
        row.accept_cycle = cycle;
        row.data_cycle = cycle;
        trace_->add_transaction(row);
    }
}

void CommandFetch::update(std::uint64_t cycle) {
    while (!inbox_.empty()) {
        const auto response = std::move(inbox_.front());
        inbox_.pop_front();
        if (response.request_id != pending_request_id_) {
            continue;  // stale response (e.g. after reset)
        }
        read_pending_ = false;
        if (faulted_) {
            continue;
        }
        if (response.status != cmd::StatusCode::Ok) {
            fault(
                std::string("command fetch read failed: ") + response.reason, cycle);
            continue;
        }
        if (response.read_data.size() != cmd::COMMAND_RECORD_BYTES) {
            fault("command fetch returned a short record", cycle);
            continue;
        }
        try {
            auto command = cmd::CommandCodec::decode(
                response.read_data.data(), cmd::COMMAND_RECORD_BYTES);
            cmd::CommandCodec::validate(command);
            if (command.command_id != next_index_) {
                fault(
                    "command stream is not a dense 0..N-1 batch (unexpected "
                    "command_id)",
                    cycle);
                continue;
            }
            window_.push_back(command);
            ++next_index_;
            ++fetched_count_;
        } catch (const std::exception& exception) {
            fault(
                std::string("command record rejected at fetch: ") + exception.what(),
                cycle);
        }
    }
}

void CommandFetch::tick(std::uint64_t cycle) {
    if (faulted_) {
        return;
    }
    if (read_pending_) {
        ++read_wait_cycles_;
        return;
    }
    if (next_index_ >= config_.command_count) {
        return;
    }
    if (window_.size() >= config_.window_depth) {
        ++window_full_cycles_;  // bounded-window backpressure, not an error
        return;
    }
    dp::MemoryRequest request;
    request.request_id = memory_.next_request_id();
    request.client = dp::MemoryClient::CommandFetch;
    request.write = false;
    request.address =
        config_.base_addr +
        static_cast<std::uint64_t>(next_index_) * cmd::COMMAND_RECORD_BYTES;
    request.length = cmd::COMMAND_RECORD_BYTES;
    request.kind = dp::AccessKind::Command;
    request.label = "command_buffer";
    if (memory_.submit(request, cycle)) {
        read_pending_ = true;
        pending_request_id_ = request.request_id;
    } else {
        ++read_wait_cycles_;  // memory queue backpressure
    }
}

void CommandFetch::remove_at(std::size_t index) {
    window_.erase(window_.begin() + static_cast<std::ptrdiff_t>(index));
}

void CommandFetch::reset() {
    inbox_.clear();
    window_.clear();
    next_index_ = 0;
    fetched_count_ = 0;
    read_pending_ = false;
    pending_request_id_ = 0;
    faulted_ = false;
    fault_reason_.clear();
    window_full_cycles_ = 0;
    read_wait_cycles_ = 0;
}

bool CommandFetch::finished() const {
    return !faulted_ && next_index_ >= config_.command_count && window_.empty() &&
           !read_pending_ && inbox_.empty();
}

}  // namespace hw::controlplane
