#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "dataplane/device_memory.hpp"
#include "dataplane/trace.hpp"

namespace hw::dataplane {

enum class MemoryClient {
    DescriptorReader,
    Dma,
    CommandFetch,
    CompletionWriter,
    ExecUnit,
    Test,
    Other,
};

const char* memory_client_name(MemoryClient client);

struct MemoryServiceConfig {
    std::uint32_t startup_cycles{1};
    std::uint32_t beat_bytes{16};
    std::uint32_t beat_cycles{1};
    std::uint32_t max_inflight{8};
};

struct MemoryRequest {
    std::uint64_t request_id{0};
    MemoryClient client{MemoryClient::Other};
    bool write{false};
    std::uint64_t address{0};
    std::uint64_t length{0};
    AccessKind kind{AccessKind::ByteData};
    std::vector<std::uint8_t> write_data;
    bool has_command{false};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::string label;
};

struct MemoryResponse {
    std::uint64_t request_id{0};
    MemoryClient client{MemoryClient::Other};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    std::uint64_t bytes{0};
    std::uint64_t issue_cycle{0};
    std::uint64_t accept_cycle{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint64_t request_wait_cycles{0};
    std::uint64_t service_cycles{0};
    std::uint64_t epoch{0};  // MemoryService epoch at accept; reset invalidates
    MemoryRequest request;   // echo (address/length/kind/client/labels)
    std::vector<std::uint8_t> read_data;
};

// Sink for memory responses of clients other than the built-in T04
// DescriptorReader/Dma paths. Delivery is non-blocking; the receiving module
// processes the response in its own update phase.
class MemoryResponseSink {
public:
    virtual ~MemoryResponseSink() = default;
    virtual void deliver_memory_response(const MemoryResponse& response) = 0;
};

// Shared DDR service. Single port, FIFO, serialized; both DescriptorReader and
// DMA submit here, so arbitration is explicit and first-come-first-served.
// Queue-full is normal backpressure (submit returns false); illegal accesses
// are accepted and answered with one terminal AddressFault response.
// Latency ownership: this class charges the DDR service time; DMA does not add
// its own memory latency.
class MemoryService {
public:
    MemoryService(
        DeviceMemory& memory,
        MemoryServiceConfig config,
        TraceRecorder* trace = nullptr);

    // Returns true when the request was accepted (including terminal error and
    // zero-length no-op responses). Full queue returns false without any side
    // effect; callers retry. `issue_cycle` is the cycle in which the caller
    // asserted the request.
    bool submit(const MemoryRequest& request, std::uint64_t issue_cycle);

    // Completes transactions whose finish_cycle has been reached; applies
    // writes and captures read data. `cycle` is the DataPlane cycle.
    void tick(std::uint64_t cycle);

    bool try_pop_response(MemoryResponse& out);
    std::size_t inflight() const { return inflight_.size(); }

    // Drops queued/in-flight requests and pending responses; epoch is bumped so
    // stale responses cannot be mistaken for new ones. No response is
    // delivered for dropped requests.
    void reset();

    std::uint64_t accepted_count() const { return accepted_count_; }
    std::uint64_t error_count() const { return error_count_; }
    std::uint64_t read_bytes() const { return read_bytes_; }
    std::uint64_t write_bytes() const { return write_bytes_; }
    std::uint64_t port_busy_cycles() const { return port_busy_cycles_; }
    std::uint64_t port_free_cycle() const { return port_free_cycle_; }
    std::uint64_t epoch() const { return epoch_; }
    const MemoryServiceConfig& config() const { return config_; }

    std::uint64_t next_request_id() { return next_request_id_++; }

    // Validation only (no side effect); used by DMA to pre-validate a whole
    // 2D transfer before any beat is issued.
    AccessCheck check(
        std::uint64_t address,
        std::uint64_t length,
        AccessKind kind,
        bool write) const {
        return memory_.check(address, length, kind, write);
    }
    const DeviceMemory& device_memory() const { return memory_; }

private:
    struct Inflight {
        MemoryResponse response;
        std::vector<std::uint8_t> write_data;
    };

    void emit_trace(const MemoryResponse& response);

    DeviceMemory& memory_;
    MemoryServiceConfig config_;
    TraceRecorder* trace_{nullptr};
    std::uint64_t next_request_id_{1};
    std::uint64_t port_free_cycle_{0};
    std::uint64_t epoch_{0};
    std::deque<Inflight> inflight_;
    std::deque<MemoryResponse> responses_;
    std::uint64_t accepted_count_{0};
    std::uint64_t error_count_{0};
    std::uint64_t read_bytes_{0};
    std::uint64_t write_bytes_{0};
    std::uint64_t port_busy_cycles_{0};
};

}  // namespace hw::dataplane
