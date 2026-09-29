#include "controlplane/dma_arbiter.hpp"

#include <algorithm>

namespace hw::controlplane {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

DmaArbiter::DmaArbiter(dp::DmaEngine& engine, std::uint32_t queue_depth)
    : engine_(engine), queue_depth_(queue_depth == 0 ? 1 : queue_depth) {}

std::uint64_t DmaArbiter::enqueue(Job job, std::uint64_t cycle) {
    if (queue_.size() >= queue_depth_) {
        return 0;
    }
    job.job_id = next_job_id_++;
    job.enqueued_cycle = cycle;
    const std::uint64_t job_id = job.job_id;
    queue_.push_back(std::move(job));
    return job_id;
}

void DmaArbiter::update(std::uint64_t cycle) {
    (void)cycle;
    if (!captured_ && engine_.response_valid()) {
        Completion completion;
        if (active_job_.has_value()) {
            completion.job_id = active_job_->job_id;
            completion.command_id = active_job_->command_id;
            completion.node_id = active_job_->node_id;
            completion.request = active_job_->request;
        }
        completion.response = engine_.response();
        completions_.push_back(completion);
        if (active_job_.has_value()) {
            active_job_.reset();
        }
        ++completed_count_;
        captured_ = true;
    }
    if (!engine_.response_valid()) {
        engine_.set_response_ready(false);
    }
}

void DmaArbiter::tick(std::uint64_t cycle) {
    if (captured_) {
        // Consume the stable response exactly once.
        engine_.set_response_ready(true);
        captured_ = false;
    }
    if (active_job_.has_value() || !engine_.request_ready()) {
        if (!queue_.empty() && !active_job_.has_value()) {
            ++job_wait_cycles_;  // queued work waits for the shared channel
        }
        return;
    }
    if (queue_.empty()) {
        return;
    }
    // Fair sharing: prefer a job from a different front tag.
    auto chosen = queue_.begin();
    for (auto iterator = queue_.begin(); iterator != queue_.end(); ++iterator) {
        if (iterator->front_tag != last_tag_) {
            chosen = iterator;
            break;
        }
    }
    Job job = *chosen;
    queue_.erase(chosen);
    last_tag_ = job.front_tag;
    engine_.set_request(job.request);
    engine_.set_request_valid(true);
    active_job_ = job;
    (void)cycle;
}

bool DmaArbiter::take_completion(Completion& out) {
    if (completions_.empty()) {
        return false;
    }
    out = std::move(completions_.front());
    completions_.pop_front();
    return true;
}

std::vector<std::uint64_t> DmaArbiter::drop_queued(std::uint32_t front_tag) {
    std::vector<std::uint64_t> dropped;
    for (auto iterator = queue_.begin(); iterator != queue_.end();) {
        if (iterator->front_tag == front_tag) {
            dropped.push_back(iterator->job_id);
            iterator = queue_.erase(iterator);
            ++dropped_count_;
        } else {
            ++iterator;
        }
    }
    return dropped;
}

std::vector<std::uint64_t> DmaArbiter::drop_queued_command(std::uint32_t command_id) {
    std::vector<std::uint64_t> dropped;
    for (auto iterator = queue_.begin(); iterator != queue_.end();) {
        if (iterator->command_id == command_id) {
            dropped.push_back(iterator->job_id);
            iterator = queue_.erase(iterator);
            ++dropped_count_;
        } else {
            ++iterator;
        }
    }
    return dropped;
}

void DmaArbiter::reset() {
    queue_.clear();
    active_job_.reset();
    completions_.clear();
    captured_ = false;
    last_tag_ = cmd::NONE;
    completed_count_ = 0;
    dropped_count_ = 0;
    job_wait_cycles_ = 0;
}

bool DmaArbiter::idle() const {
    return queue_.empty() && !active_job_.has_value() && completions_.empty() &&
           !captured_ && !engine_.busy();
}

std::uint32_t DmaArbiter::active_command_id() const {
    return active_job_.has_value() ? active_job_->command_id : cmd::NONE;
}

std::uint32_t DmaArbiter::active_front_tag() const {
    return active_job_.has_value() ? active_job_->front_tag : cmd::NONE;
}

}  // namespace hw::controlplane
