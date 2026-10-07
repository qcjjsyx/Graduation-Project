#pragma once

// Unified on-chip scratchpad bandwidth arbiter.
//
// Read and write budgets are separate domains. Per cycle, each domain serves
// AT MOST ONE requester with the whole remaining budget: the remainder is not
// shared with a second requester. Requesters have stable identities
// (kind + id) and are selected by a deterministic rotation, so tick order can
// never give a module permanent priority. A requester that is not selected
// keeps its progress and retries later; modules must never move bytes first
// and account budget afterwards.
//
// Clients: Panel scan, PE operand loads, PE writeback, FactorStore reads, DMA
// on-chip accesses (and future assemble/solve). This class is data/architecture
// model only: DDR latency stays in MemoryService, lease authorization and
// ready cycles stay in BufferManager.

#include <cstdint>
#include <limits>
#include <map>
#include <set>

namespace hw::dataplane {

enum class OnChipClientKind : std::uint32_t {
    PanelScan = 1,
    PeLoad = 2,
    PeStore = 3,
    FactorRead = 4,
    Dma = 5,
    Assemble = 6,
    Solve = 7,
    Other = 8,
};

const char* on_chip_client_kind_name(OnChipClientKind kind);

enum class OnChipDomain : std::uint8_t {
    Read = 0,
    Write = 1,
};

const char* on_chip_domain_name(OnChipDomain domain);

class OnChipArbiter {
public:
    struct ClientKey {
        OnChipClientKind kind{OnChipClientKind::Other};
        std::uint32_t id{0};
    };

    struct Stats {
        std::uint64_t grants{0};
        std::uint64_t stalls{0};
        std::uint64_t bytes{0};
    };

    void configure(std::uint32_t read_bytes_per_cycle,
                   std::uint32_t write_bytes_per_cycle);
    std::uint32_t read_bytes_per_cycle() const { return domains_[0].budget; }
    std::uint32_t write_bytes_per_cycle() const { return domains_[1].budget; }

    // Initializes the per-cycle grant window. Idempotent for the same cycle:
    // the device owner and a standalone DataPlane may both call it safely, and
    // each domain still serves at most one requester per cycle.
    void begin_cycle(std::uint64_t cycle);

    // Requests up to `bytes` from the domain for this cycle. Returns the
    // granted byte count (0 = not the selected requester this cycle, or the
    // domain budget is exhausted). Granted bytes never exceed the budget
    // remaining this cycle.
    std::uint32_t acquire(
        OnChipDomain domain,
        OnChipClientKind kind,
        std::uint32_t client_id,
        std::uint32_t bytes);

    // Drops a stale pending need (e.g. a cancelled transfer).
    void clear_pending(
        OnChipDomain domain,
        OnChipClientKind kind,
        std::uint32_t client_id);

    const Stats& stats(
        OnChipDomain domain,
        OnChipClientKind kind,
        std::uint32_t client_id) const;
    std::uint64_t grants(OnChipDomain domain) const {
        return domain_grants_[static_cast<int>(domain)];
    }
    std::size_t pending(OnChipDomain domain) const {
        return domains_[static_cast<int>(domain)].pending.size();
    }
    bool has_grant(OnChipDomain domain) const {
        return domains_[static_cast<int>(domain)].granted_key != 0;
    }
    std::uint32_t remaining(OnChipDomain domain) const;
    std::uint64_t cycle() const { return cycle_; }
    void reset();

private:
    struct Domain {
        std::uint32_t budget{0};
        std::uint64_t granted_key{0};  // 0 = no requester selected yet
        std::uint32_t used{0};
        std::set<std::uint64_t> pending;
        std::map<std::uint64_t, std::uint64_t> last_seen;
        std::uint64_t cursor{0};  // last selected client key
    };

    static std::uint64_t key_of(OnChipClientKind kind, std::uint32_t id) {
        return (static_cast<std::uint64_t>(kind) << 32) | id;
    }
    Domain& domain_of(OnChipDomain domain) {
        return domains_[static_cast<int>(domain)];
    }
    const Domain& domain_of(OnChipDomain domain) const {
        return domains_[static_cast<int>(domain)];
    }
    std::uint64_t select_pending(const Domain& domain) const;
    Stats& stats_of(OnChipDomain domain, OnChipClientKind kind, std::uint32_t id);

    Domain domains_[2];
    std::uint64_t domain_grants_[2]{0, 0};
    std::map<std::uint64_t, std::map<std::uint64_t, Stats>> client_stats_;
    std::uint64_t cycle_{0};
    std::uint64_t last_begin_cycle_{std::numeric_limits<std::uint64_t>::max()};
    bool configured_{false};
};

}  // namespace hw::dataplane
