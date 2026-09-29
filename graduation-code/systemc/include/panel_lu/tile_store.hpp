#pragma once

// Synthetic on-chip tile store for the T06 prototype. One slot holds one
// physical 16x16 FP32 tile (1024B) plus a committed version counter. Access
// bandwidth is not modeled here: the prototype owns per-cycle port budgets and
// feeds them to PEs/controllers, so contention is explicit and observable.

#include <cstdint>
#include <string>
#include <vector>

#include "panel_lu/task_types.hpp"

namespace hw::panel_lu {

class TileStore {
public:
    explicit TileStore(std::uint32_t slot_count = 0);

    void resize(std::uint32_t slot_count);

    // Assigns synthetic data to a slot (owner label, generation, values).
    // Values are copied into the fixed 16x16 tile storage (256 floats).
    void assign(
        std::uint32_t slot,
        const std::string& owner,
        std::uint32_t generation,
        const std::vector<float>& values);

    bool assigned(std::uint32_t slot) const;
    std::uint32_t slot_count() const { return static_cast<std::uint32_t>(slots_.size()); }
    const std::string& owner(std::uint32_t slot) const;
    std::uint32_t generation(std::uint32_t slot) const;
    std::uint32_t version(std::uint32_t slot) const;
    std::uint64_t commit_count(std::uint32_t slot) const;

    float* data(std::uint32_t slot);
    const float* data(std::uint32_t slot) const;

    // Commits a result version. Returns false on a stale expected version
    // (i.e. an out-of-order write would have occurred); the caller must treat
    // that as a control bug, not as a silent success.
    bool commit(std::uint32_t slot, std::uint32_t expected_version);

    // Invalidates all handles after a reset.
    void bump_generations();
    void clear();

    static constexpr std::uint32_t kTileFloats = 16 * 16;

private:
    struct Slot {
        bool assigned{false};
        std::string owner;
        std::uint32_t generation{0};
        std::uint32_t version{0};
        std::uint64_t commits{0};
        std::vector<float> values;
    };

    void check_slot(std::uint32_t slot) const;

    std::vector<Slot> slots_;
};

// One per-cycle bandwidth budget (bytes). The owner resets it at the start of
// every tick; requests beyond the budget fail and the requester stalls.
struct PortBudget {
    std::uint64_t capacity{0};
    std::uint64_t used{0};

    void reset(std::uint64_t capacity_bytes) {
        capacity = capacity_bytes;
        used = 0;
    }
    std::uint64_t remaining() const { return capacity > used ? capacity - used : 0; }
    bool request(std::uint64_t bytes) {
        if (bytes > remaining()) {
            return false;
        }
        used += bytes;
        return true;
    }
};

// Bounded shared writeback reservation. Tasks that finished executing hold a
// reservation while their result drains; the reservation is released when the
// writeback completes.
class WritebackReservation {
public:
    void configure(std::uint32_t capacity) { capacity_ = capacity; reset(); }
    bool acquire() {
        if (in_use_ >= capacity_) {
            return false;
        }
        ++in_use_;
        return true;
    }
    void release() {
        if (in_use_ > 0) {
            --in_use_;
        }
    }
    std::uint32_t in_use() const { return in_use_; }
    std::uint32_t capacity() const { return capacity_; }
    void reset() { in_use_ = 0; }

private:
    std::uint32_t capacity_{0};
    std::uint32_t in_use_{0};
};

}  // namespace hw::panel_lu
