#pragma once

#include <cstdint>
#include <deque>
#include <string>

#include "command_codec.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/trace.hpp"

namespace hw::controlplane {

// Writes one terminal completion record per accepted command into its fixed
// slot (slot == command_id). Bounded queue; a failed write is a batch fault,
// is never retried at the same address and never publishes a READY token.
class CompletionWriter : public hw::dataplane::MemoryResponseSink {
public:
    struct Item {
        std::uint32_t slot{0};
        hw::command::Completion record{};
    };

    CompletionWriter(
        hw::dataplane::MemoryService& memory,
        std::uint64_t base_addr,
        std::uint32_t queue_depth,
        hw::dataplane::TraceRecorder* trace = nullptr);

    void deliver_memory_response(
        const hw::dataplane::MemoryResponse& response) override;
    void update(std::uint64_t cycle);
    void tick(std::uint64_t cycle);

    // Returns false while the bounded queue is full (stall, not an error).
    bool enqueue(const Item& item, std::uint64_t cycle);
    // One result per written record: slot + whether the write confirmed.
    bool take_result(std::uint32_t& slot, bool& confirmed);

    void reset();

    bool faulted() const { return faulted_; }
    const std::string& fault_reason() const { return fault_reason_; }
    std::uint32_t pending_count() const {
        return static_cast<std::uint32_t>(queue_.size() + (write_pending_ ? 1 : 0));
    }
    std::uint32_t written_count() const { return written_count_; }
    std::uint64_t written_bytes() const {
        return static_cast<std::uint64_t>(written_count_) *
               hw::command::COMPLETION_RECORD_BYTES;
    }
    std::uint64_t backpressure_cycles() const { return backpressure_cycles_; }

private:
    hw::dataplane::MemoryService& memory_;
    std::uint64_t base_addr_{0};
    std::uint32_t queue_depth_{0};
    std::deque<Item> queue_;
    std::deque<hw::dataplane::MemoryResponse> inbox_;
    std::deque<std::pair<std::uint32_t, bool>> results_;
    std::vector<std::uint8_t> write_buffer_;
    bool write_pending_{false};
    std::uint64_t pending_request_id_{0};
    std::uint32_t pending_slot_{0};
    bool faulted_{false};
    std::string fault_reason_;
    std::uint32_t written_count_{0};
    std::uint64_t backpressure_cycles_{0};
};

}  // namespace hw::controlplane
