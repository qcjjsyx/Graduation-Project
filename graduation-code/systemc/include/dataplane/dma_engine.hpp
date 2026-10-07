#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/buffer_manager.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/on_chip_arbiter.hpp"
#include "dataplane/region_view.hpp"
#include "dataplane/trace.hpp"

namespace hw::dataplane {

enum class DmaDirection {
    MemoryToBuffer,  // DDR -> scratchpad
    BufferToMemory,  // scratchpad -> DDR
};

const char* dma_direction_name(DmaDirection direction);

struct DmaConfig {
    std::uint32_t beat_bytes{16};
    std::uint32_t startup_cycles{1};
    std::uint32_t beat_interval_cycles{1};
};

struct DmaRequest {
    std::uint64_t request_id{0};
    bool has_command{false};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    DmaDirection direction{DmaDirection::MemoryToBuffer};

    // DDR side (source for MemoryToBuffer, destination for BufferToMemory).
    RegionView ddr_region;
    std::uint64_t ddr_offset{0};
    std::uint64_t ddr_row_stride{0};  // 0 => cols * element_bytes

    // Scratchpad side. `owner` must match the buffer slot owner (T04 default
    // "dma"; the control plane passes the front context owner).
    std::string owner{"dma"};
    BufferHandle buffer{};
    std::uint64_t buffer_offset{0};
    std::uint64_t buffer_row_stride{0};  // 0 => cols * element_bytes

    std::uint32_t rows{0};
    std::uint32_t cols{0};
    std::uint32_t element_bytes{4};
    std::string label;
};

struct DmaResponse {
    std::uint64_t request_id{0};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    // Set only for failures raised before the first beat (resource
    // contention); no data has been moved, so a retry is safe.
    bool retryable{false};
    std::uint64_t bytes{0};
    std::uint64_t read_bytes{0};
    std::uint64_t write_bytes{0};
    std::uint64_t issue_cycle{0};
    std::uint64_t accept_cycle{0};
    std::uint64_t first_beat_cycle{0};
    std::uint64_t last_beat_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint64_t request_wait_cycles{0};
    std::uint64_t memory_wait_cycles{0};
    std::uint64_t buffer_wait_cycles{0};
    std::uint64_t pipeline_wait_cycles{0};
    std::uint64_t response_backpressure_cycles{0};
    std::uint64_t memory_backpressure_cycles{0};
    bool has_command{false};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
};

// Single-channel DMA: one active transfer, sequential completion. Request and
// response use explicit valid/ready handshakes; an accepted request always gets
// exactly one terminal response unless reset() drops it.
class DmaEngine {
public:
    DmaEngine(
        DmaConfig config,
        MemoryService& memory,
        BufferManager& buffers,
        TraceRecorder* trace = nullptr);

    // Request handshake (testbench/consumer side).
    void set_request(const DmaRequest& request);
    void set_request_valid(bool valid);
    bool request_ready() const;
    bool request_valid() const { return request_valid_; }

    // Response handshake.
    bool response_valid() const { return response_valid_; }
    const DmaResponse& response() const { return response_; }
    void set_response_ready(bool ready) { response_ready_ = ready; }

    // Called once per DataPlane cycle. Never blocks.
    void tick(std::uint64_t cycle);
    // Routed by DataPlane from MemoryService responses.
    void deliver_memory_response(const MemoryResponse& response);

    // Optional unified on-chip bandwidth arbitration (nullptr = legacy T04
    // standalone behaviour with no budget gating).
    void set_on_chip_arbiter(OnChipArbiter* arbiter) { arbiter_ = arbiter; }

    void reset();

    // On-chip scratchpad traffic actually moved by this engine (test/diagnostic
    // instrumentation; one increment per accepted lease access).
    std::uint64_t scratch_read_bytes() const { return scratch_read_bytes_; }
    std::uint64_t scratch_write_bytes() const { return scratch_write_bytes_; }

    bool busy() const;
    std::uint64_t accepted_count() const { return accepted_count_; }
    std::uint64_t completed_count() const { return completed_count_; }
    std::uint64_t failed_count() const { return failed_count_; }
    std::uint64_t read_bytes() const { return read_bytes_; }
    std::uint64_t write_bytes() const { return write_bytes_; }
    std::uint64_t ddr_read_bytes() const { return ddr_read_bytes_; }
    std::uint64_t ddr_write_bytes() const { return ddr_write_bytes_; }
    const DmaConfig& config() const { return config_; }

private:
    enum class Phase { Idle, Active, ResponseHold };

    struct Beat {
        std::uint32_t row{0};
        std::uint32_t index_in_row{0};
        std::uint64_t ddr_address{0};
        std::uint64_t buffer_offset{0};
        std::uint64_t length{0};
    };

    void accept_request(std::uint64_t cycle);
    void fail_transfer(
        hw::command::StatusCode status,
        const std::string& reason,
        std::uint64_t cycle,
        bool retryable = false);
    void finish_transfer(std::uint64_t cycle);
    void start_next_beat(std::uint64_t cycle);
    void complete_beat(std::uint64_t cycle);
    Beat current_beat() const;
    std::uint64_t beats_per_row() const;
    std::uint64_t total_bytes() const;
    bool prepare_transfer(std::uint64_t cycle, std::string* reason);
    void publish_response(std::uint64_t cycle);
    void flush_trace_row();

    DmaConfig config_;
    OnChipArbiter* arbiter_{nullptr};
    MemoryService& memory_;
    BufferManager& buffers_;
    TraceRecorder* trace_{nullptr};

    bool request_valid_{false};
    DmaRequest request_;
    std::uint64_t request_issue_cycle_{0};
    bool issue_recorded_{false};

    bool response_valid_{false};
    DmaResponse response_;
    bool response_ready_{false};

    Phase phase_{Phase::Idle};
    DmaRequest active_;
    BufferLease lease_;
    bool have_lease_{false};
    std::uint64_t total_transfer_bytes_{0};
    std::uint64_t buffer_span_bytes_{0};
    std::uint64_t bytes_done_{0};
    std::uint32_t row_{0};
    std::uint32_t index_in_row_{0};
    std::uint64_t beats_per_row_{1};
    std::uint64_t next_beat_cycle_{0};
    bool first_beat_started_{false};

    bool memory_pending_{false};
    bool memory_response_ready_{false};
    std::uint64_t pending_memory_id_{0};
    MemoryResponse pending_memory_response_;
    std::vector<std::uint8_t> beat_buffer_data_;
    // On-chip budget progress inside one beat: when the per-cycle grant is
    // smaller than the beat, the beat is split into segments across cycles
    // (progress preserved, no replayed access).
    std::uint32_t beat_read_progress_{0};
    std::uint32_t beat_write_progress_{0};
    // Staging lifecycle: initialized once when a new beat starts; preserved
    // across budget waits and DDR backpressure retries.
    bool beat_staged_{false};
    std::uint64_t scratch_read_bytes_{0};
    std::uint64_t scratch_write_bytes_{0};

    std::uint64_t memory_wait_cycles_{0};
    std::uint64_t buffer_wait_cycles_{0};
    std::uint64_t pipeline_wait_cycles_{0};
    std::uint64_t memory_backpressure_cycles_{0};
    std::uint64_t response_backpressure_cycles_{0};

    TransactionRow pending_trace_row_;
    bool have_pending_trace_row_{false};

    std::uint64_t accepted_count_{0};
    std::uint64_t completed_count_{0};
    std::uint64_t failed_count_{0};
    std::uint64_t read_bytes_{0};
    std::uint64_t write_bytes_{0};
    std::uint64_t ddr_read_bytes_{0};
    std::uint64_t ddr_write_bytes_{0};
};

}  // namespace hw::dataplane
