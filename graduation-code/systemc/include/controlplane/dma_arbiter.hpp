#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "dataplane/dma_engine.hpp"

namespace hw::controlplane {

// Fair sharing of T04's single-active-transaction DMA channel across front
// contexts: one bounded job queue, round-robin between front tags.
class DmaArbiter {
public:
    struct Job {
        std::uint64_t job_id{0};
        std::uint32_t front_tag{hw::command::NONE};
        std::uint32_t command_id{hw::command::NONE};
        std::uint32_t node_id{hw::command::NONE};
        hw::dataplane::DmaRequest request;
        std::uint64_t enqueued_cycle{0};
    };

    struct Completion {
        std::uint64_t job_id{0};
        std::uint32_t command_id{hw::command::NONE};
        std::uint32_t node_id{hw::command::NONE};
        hw::dataplane::DmaResponse response;
        // Original request, so a transiently rejected job can be retried with
        // a fresh request id without side effects.
        hw::dataplane::DmaRequest request;
    };

    DmaArbiter(hw::dataplane::DmaEngine& engine, std::uint32_t queue_depth);

    // Queue-full is normal backpressure (stall), not an error. Returns the
    // assigned job id, or 0 when the queue rejected the job.
    std::uint64_t enqueue(Job job, std::uint64_t cycle);
    // Captures finished engine responses (called once per cycle).
    void update(std::uint64_t cycle);
    // Drives the engine request/response handshake (called once per cycle).
    void tick(std::uint64_t cycle);
    bool take_completion(Completion& out);

    bool can_enqueue(std::size_t count) const {
        return queue_.size() + count <= queue_depth_;
    }
    // Abort path: drops queued jobs of a front and returns their job ids; the
    // job already in the engine cannot be cancelled and finishes normally (its
    // writes are not rolled back, which the abort documentation must state).
    std::vector<std::uint64_t> drop_queued(std::uint32_t front_tag);
    std::vector<std::uint64_t> drop_queued_command(std::uint32_t command_id);

    void reset();

    bool idle() const;
    bool has_active_job() const { return active_job_.has_value(); }
    std::uint32_t active_command_id() const;
    std::uint32_t active_front_tag() const;
    std::uint64_t queued_count() const { return queue_.size(); }
    std::uint64_t completed_count() const { return completed_count_; }
    std::uint64_t dropped_count() const { return dropped_count_; }
    std::uint64_t job_wait_cycles() const { return job_wait_cycles_; }

private:
    hw::dataplane::DmaEngine& engine_;
    std::uint32_t queue_depth_;
    std::deque<Job> queue_;
    std::optional<Job> active_job_;
    std::deque<Completion> completions_;
    bool captured_{false};
    std::uint32_t last_tag_{hw::command::NONE};
    std::uint64_t next_job_id_{1};
    std::uint64_t completed_count_{0};
    std::uint64_t dropped_count_{0};
    std::uint64_t job_wait_cycles_{0};
};

}  // namespace hw::controlplane
