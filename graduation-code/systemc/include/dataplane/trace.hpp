#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "command_codec.hpp"

namespace hw::dataplane {

// One CSV row per transaction layer event. Layers:
//   "descriptor": DescriptorReader memory read (one per 64B record or payload)
//   "memory":     MemoryService transaction
//   "dma":        DmaEngine request (aggregates one transfer)
struct TransactionRow {
    std::string layer;
    std::string client;
    std::uint64_t request_id{0};
    bool has_command{false};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::string region;
    std::uint64_t address{0};
    std::uint64_t length{0};
    bool write{false};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    std::uint64_t issue_cycle{0};
    std::uint64_t accept_cycle{0};
    std::uint64_t data_cycle{0};
    std::uint64_t response_cycle{0};
    std::uint64_t bytes{0};
    std::uint64_t request_wait_cycles{0};
    std::uint64_t memory_service_cycles{0};
    std::uint64_t response_backpressure_cycles{0};
};

struct BufferRow {
    std::uint64_t cycle{0};
    std::uint32_t slot{0};
    std::uint32_t generation{0};
    std::string owner;
    std::string from_state;
    std::string to_state;
    std::string reason;
};

class TraceRecorder {
public:
    void add_transaction(TransactionRow row) {
        transactions_.push_back(std::move(row));
    }
    void add_buffer(BufferRow row) { buffers_.push_back(std::move(row)); }

    const std::vector<TransactionRow>& transactions() const { return transactions_; }
    const std::vector<BufferRow>& buffers() const { return buffers_; }

    void clear() {
        transactions_.clear();
        buffers_.clear();
    }

    // Machine-readable outputs. Cycles are architecture-model cycles counted
    // from DataPlane::step() (see docs/systemc/T04_SystemC_数据面设计.md).
    void write_csv(const std::string& output_dir) const;

private:
    std::vector<TransactionRow> transactions_;
    std::vector<BufferRow> buffers_;
};

const char* status_name(hw::command::StatusCode status);

}  // namespace hw::dataplane
