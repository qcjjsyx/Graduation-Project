#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/trace.hpp"

namespace hw::dataplane {

// On-chip matrix data buffer (scratchpad), not a command queue.
enum class BufferState {
    Free,
    Reserved,
    Loading,
    Ready,
    Computing,
    Dirty,
    WritingBack,
};

const char* buffer_state_name(BufferState state);

// slot_id + generation: handles become stale after release/reset.
struct BufferHandle {
    std::uint32_t slot{0};
    std::uint32_t generation{0};
};

inline bool operator==(const BufferHandle& lhs, const BufferHandle& rhs) {
    return lhs.slot == rhs.slot && lhs.generation == rhs.generation;
}
inline bool operator!=(const BufferHandle& lhs, const BufferHandle& rhs) {
    return !(lhs == rhs);
}

struct BufferConfig {
    std::uint32_t slot_count{2};
    std::uint64_t capacity_bytes{64 * 1024};
    std::uint32_t port_count{2};
    std::uint32_t access_cycles{1};
};

struct BufferLease {
    bool valid{false};
    std::uint64_t lease_id{0};
    std::uint32_t slot{0};
    std::uint32_t generation{0};
    bool write{false};
    std::uint64_t offset{0};
    std::uint64_t length{0};
    std::uint64_t acquire_cycle{0};
    std::uint64_t ready_cycle{0};
};

struct BufferResult {
    bool ok{false};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    // True when the failure is temporary contention (port busy, conflicting
    // access) and a later retry may succeed without side effects.
    bool transient{false};
    BufferHandle handle{};
    BufferLease lease{};
};

// Single scratchpad with explicit ownership, capacity, valid-data range and
// lifecycle. Conflicting or out-of-port accesses fail with BufferFull and an
// internal reason instead of silently overlapping.
class BufferManager {
public:
    BufferManager(BufferConfig config, TraceRecorder* trace = nullptr);

    BufferResult allocate(
        const std::string& owner,
        std::uint64_t required_bytes,
        std::uint64_t cycle);
    BufferResult release(
        const BufferHandle& handle,
        const std::string& owner,
        std::uint64_t cycle);

    // Explicit ownership handoff (e.g. DMA load finishes, consumer takes over
    // a READY buffer). Only allowed with no active lease.
    BufferResult transfer_owner(
        const BufferHandle& handle,
        const std::string& current_owner,
        const std::string& new_owner,
        std::uint64_t cycle);

    // DMA load path: Reserved -> Loading --(finish_load)--> Ready.
    BufferResult begin_load(
        const BufferHandle& handle,
        const std::string& owner,
        std::uint64_t offset,
        std::uint64_t bytes,
        std::uint64_t cycle);

    // Consumer compute path: Reserved/Ready/Dirty -> Computing
    // --(end_write)--> Dirty.
    BufferResult begin_write(
        const BufferHandle& handle,
        const std::string& owner,
        std::uint64_t offset,
        std::uint64_t bytes,
        std::uint64_t cycle);

    // Read-only consumer lease; state stays Ready/Dirty. Read-write conflicts
    // on the same slot are refused.
    BufferResult begin_read(
        const BufferHandle& handle,
        const std::string& owner,
        std::uint64_t offset,
        std::uint64_t bytes,
        std::uint64_t cycle);

    // DMA writeback path: Ready/Dirty -> WritingBack
    // --(finish_writeback)--> Ready.
    BufferResult begin_writeback(
        const BufferHandle& handle,
        const std::string& owner,
        std::uint64_t offset,
        std::uint64_t bytes,
        std::uint64_t cycle);

    hw::command::StatusCode lease_write(
        std::uint64_t lease_id,
        std::uint64_t offset,
        const std::vector<std::uint8_t>& data,
        std::uint64_t cycle,
        std::string* reason = nullptr);
    hw::command::StatusCode lease_read(
        std::uint64_t lease_id,
        std::uint64_t offset,
        std::uint64_t length,
        std::vector<std::uint8_t>& out,
        std::uint64_t cycle,
        std::string* reason = nullptr);

    // Aborts an in-flight load/writeback lease (used on reset or memory
    // failure); the slot returns to its pre-lease state and no data is
    // committed.
    // Returns a READY/DIRTY slot (without active leases) to RESERVED so the
    // DMA load path can refill it; generation is kept so the handle stays
    // valid. Used by the control plane to reuse scratch slots.
    BufferResult rearm(
        const BufferHandle& handle,
        const std::string& owner,
        std::uint64_t cycle);

    // Releases a read-only lease. Write leases must be completed with
    // end_write(); load/writeback leases with finish_load()/finish_writeback().
    hw::command::StatusCode release_lease(
        std::uint64_t lease_id,
        std::uint64_t cycle,
        std::string* reason = nullptr);

    hw::command::StatusCode abort_lease(
        std::uint64_t lease_id,
        std::uint64_t cycle,
        std::string* reason = nullptr);

    hw::command::StatusCode finish_load(
        std::uint64_t lease_id,
        std::uint64_t bytes_written,
        std::uint64_t cycle,
        std::string* reason = nullptr);
    hw::command::StatusCode end_write(
        std::uint64_t lease_id,
        std::uint64_t bytes_written,
        std::uint64_t cycle,
        std::string* reason = nullptr);
    hw::command::StatusCode finish_writeback(
        std::uint64_t lease_id,
        std::uint64_t bytes_read,
        std::uint64_t cycle,
        std::string* reason = nullptr);

    void reset();

    // Introspection for tests and diagnostics.
    BufferState state(std::uint32_t slot) const;
    std::uint32_t generation(std::uint32_t slot) const;
    std::uint64_t valid_bytes(std::uint32_t slot) const;
    std::uint64_t valid_offset(std::uint32_t slot) const;
    const std::vector<std::uint8_t>& bytes(std::uint32_t slot) const;
    std::uint32_t active_lease_count() const {
        return static_cast<std::uint32_t>(leases_.size());
    }
    const BufferConfig& config() const { return config_; }
    std::uint64_t conflicts() const { return conflicts_; }

private:
    struct Slot {
        BufferState state{BufferState::Free};
        std::uint32_t generation{0};
        std::string owner;
        std::uint64_t valid_offset{0};
        std::uint64_t valid_bytes{0};
        std::uint64_t reserved_bytes{0};
        std::vector<std::uint8_t> storage;
        std::uint64_t active_lease{0};
    };

    struct Lease {
        std::uint64_t id{0};
        std::uint32_t slot{0};
        std::uint32_t generation{0};
        bool write{false};
        std::string owner;
        std::uint64_t offset{0};
        std::uint64_t length{0};
        std::uint64_t ready_cycle{0};
        bool load_lease{false};
        bool writeback_lease{false};
        BufferState previous_state{BufferState::Reserved};
        std::uint64_t access_bytes{0};
    };

    BufferResult failure(
        hw::command::StatusCode status,
        std::string reason,
        std::uint64_t cycle,
        std::uint32_t slot = hw::command::NONE);
    const Slot* checked_slot(
        const BufferHandle& handle,
        const std::string& owner,
        hw::command::StatusCode* status,
        std::string* reason) const;
    BufferResult make_lease(
        const BufferHandle& handle,
        const std::string& owner,
        bool write,
        std::uint64_t offset,
        std::uint64_t bytes,
        std::uint64_t cycle,
        bool load_lease,
        bool writeback_lease);
    Lease* find_lease(std::uint64_t lease_id);
    void release_lease_record(std::uint64_t lease_id);
    bool range_conflicts(const Slot& slot, bool write) const;
    void log_transition(
        const Slot& slot,
        std::uint32_t slot_index,
        const std::string& owner,
        BufferState from,
        BufferState to,
        const std::string& reason,
        std::uint64_t cycle) const;

    BufferConfig config_;
    TraceRecorder* trace_{nullptr};
    std::vector<Slot> slots_;
    std::vector<Lease> leases_;
    std::uint64_t next_lease_id_{1};
    std::uint64_t conflicts_{0};
};

}  // namespace hw::dataplane
