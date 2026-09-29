#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/trace.hpp"

namespace hw::controlplane {

// Fixed-width command fetch from the static command batch region. A bounded
// FIFO doubles as the scheduler's command window: entries stay stable while
// blocked and can be removed out of order (earliest dispatchable first).
class CommandFetch : public hw::dataplane::MemoryResponseSink {
public:
    struct Config {
        std::uint64_t base_addr{0};
        std::uint32_t command_count{0};
        std::uint32_t window_depth{16};
    };

    CommandFetch(
        hw::dataplane::MemoryService& memory,
        Config config,
        hw::dataplane::TraceRecorder* trace = nullptr);

    void deliver_memory_response(
        const hw::dataplane::MemoryResponse& response) override;
    // Decodes buffered reads into the window (non-blocking).
    void update(std::uint64_t cycle);
    // Issues new reads when the window has room (non-blocking).
    void tick(std::uint64_t cycle);

    bool window_empty() const { return window_.empty(); }
    std::size_t window_size() const { return window_.size(); }
    const hw::command::Command& at(std::size_t index) const { return window_[index]; }
    void remove_at(std::size_t index);

    void reset();

    bool finished() const;
    bool faulted() const { return faulted_; }
    const std::string& fault_reason() const { return fault_reason_; }

    std::uint32_t fetched_count() const { return fetched_count_; }
    std::uint64_t fetched_bytes() const {
        return static_cast<std::uint64_t>(fetched_count_) *
               hw::command::COMMAND_RECORD_BYTES;
    }
    std::uint64_t window_full_cycles() const { return window_full_cycles_; }
    std::uint64_t read_wait_cycles() const { return read_wait_cycles_; }

private:
    void fault(const std::string& reason, std::uint64_t cycle);

    hw::dataplane::MemoryService& memory_;
    Config config_;
    hw::dataplane::TraceRecorder* trace_{nullptr};
    std::deque<hw::dataplane::MemoryResponse> inbox_;
    std::deque<hw::command::Command> window_;
    std::uint32_t next_index_{0};
    std::uint32_t fetched_count_{0};
    bool read_pending_{false};
    std::uint64_t pending_request_id_{0};
    bool faulted_{false};
    std::string fault_reason_;
    std::uint64_t window_full_cycles_{0};
    std::uint64_t read_wait_cycles_{0};
};

}  // namespace hw::controlplane
