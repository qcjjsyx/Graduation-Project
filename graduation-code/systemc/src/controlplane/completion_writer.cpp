#include "controlplane/completion_writer.hpp"

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

CompletionWriter::CompletionWriter(
    dp::MemoryService& memory,
    std::uint64_t base_addr,
    std::uint32_t queue_depth,
    dp::TraceRecorder* trace)
    : memory_(memory),
      base_addr_(base_addr),
      queue_depth_(queue_depth == 0 ? 1 : queue_depth) {
    (void)trace;
}

void CompletionWriter::deliver_memory_response(const dp::MemoryResponse& response) {
    inbox_.push_back(response);
}

bool CompletionWriter::enqueue(const Item& item, std::uint64_t cycle) {
    if (queue_.size() >= queue_depth_) {
        ++backpressure_cycles_;
        (void)cycle;
        return false;
    }
    queue_.push_back(item);
    return true;
}

void CompletionWriter::update(std::uint64_t cycle) {
    (void)cycle;
    while (!inbox_.empty()) {
        const auto response = std::move(inbox_.front());
        inbox_.pop_front();
        if (response.request_id != pending_request_id_ || !write_pending_) {
            continue;  // stale response
        }
        write_pending_ = false;
        if (response.status == cmd::StatusCode::Ok) {
            results_.emplace_back(pending_slot_, true);
            ++written_count_;
        } else {
            // One attempt only: a failing completion write is a batch fault,
            // never retried at the same address, and must not publish success.
            faulted_ = true;
            fault_reason_ =
                "completion write failed for slot " + std::to_string(pending_slot_) +
                ": " + response.reason;
            results_.emplace_back(pending_slot_, false);
        }
    }
}

void CompletionWriter::tick(std::uint64_t cycle) {
    if (faulted_ || write_pending_ || queue_.empty()) {
        return;
    }
    const Item& item = queue_.front();
    const auto record = cmd::CompletionCodec::encode(item.record);
    write_buffer_.assign(record.begin(), record.end());

    dp::MemoryRequest request;
    request.request_id = memory_.next_request_id();
    request.client = dp::MemoryClient::CompletionWriter;
    request.write = true;
    request.address = base_addr_ +
                      static_cast<std::uint64_t>(item.slot) * cmd::COMPLETION_RECORD_BYTES;
    request.length = cmd::COMPLETION_RECORD_BYTES;
    request.kind = dp::AccessKind::Completion;
    request.write_data = write_buffer_;
    request.command_id = item.record.command_id;
    request.node_id = item.record.node_id;
    request.label = "completion_status";
    if (!memory_.submit(request, cycle)) {
        ++backpressure_cycles_;
        return;
    }
    write_pending_ = true;
    pending_request_id_ = request.request_id;
    pending_slot_ = item.slot;
    queue_.pop_front();
}

void CompletionWriter::reset() {
    queue_.clear();
    inbox_.clear();
    results_.clear();
    write_pending_ = false;
    pending_request_id_ = 0;
    pending_slot_ = 0;
    faulted_ = false;
    fault_reason_.clear();
    written_count_ = 0;
    backpressure_cycles_ = 0;
}

bool CompletionWriter::take_result(std::uint32_t& slot, bool& confirmed) {
    if (results_.empty()) {
        return false;
    }
    slot = results_.front().first;
    confirmed = results_.front().second;
    results_.pop_front();
    return true;
}

}  // namespace hw::controlplane
