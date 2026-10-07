#include "dataplane/on_chip_arbiter.hpp"

#include <algorithm>

namespace hw::dataplane {

const char* on_chip_client_kind_name(OnChipClientKind kind) {
    switch (kind) {
        case OnChipClientKind::PanelScan:
            return "panel_scan";
        case OnChipClientKind::PeLoad:
            return "pe_load";
        case OnChipClientKind::PeStore:
            return "pe_store";
        case OnChipClientKind::FactorRead:
            return "factor_read";
        case OnChipClientKind::Dma:
            return "dma";
        case OnChipClientKind::Assemble:
            return "assemble";
        case OnChipClientKind::Solve:
            return "solve";
        case OnChipClientKind::Other:
            return "other";
    }
    return "unknown";
}

const char* on_chip_domain_name(OnChipDomain domain) {
    return domain == OnChipDomain::Read ? "read" : "write";
}

void OnChipArbiter::configure(
    std::uint32_t read_bytes_per_cycle,
    std::uint32_t write_bytes_per_cycle) {
    domains_[0] = Domain{};
    domains_[0].budget = read_bytes_per_cycle;
    domains_[1] = Domain{};
    domains_[1].budget = write_bytes_per_cycle;
    domain_grants_[0] = 0;
    domain_grants_[1] = 0;
    client_stats_.clear();
    cycle_ = 0;
    last_begin_cycle_ = std::numeric_limits<std::uint64_t>::max();
    configured_ = true;
}

void OnChipArbiter::begin_cycle(std::uint64_t cycle) {
    if (cycle == last_begin_cycle_) {
        return;  // already initialized this cycle (idempotent)
    }
    last_begin_cycle_ = cycle;
    cycle_ = cycle;
    for (Domain& domain : domains_) {
        domain.granted_key = 0;
        domain.used = 0;
        // A pending need is only kept alive while its owner keeps asking;
        // stale entries (cancelled transfers, finished tasks) are dropped so
        // they can never block the rotation.
        for (auto iterator = domain.pending.begin();
             iterator != domain.pending.end();) {
            const auto seen = domain.last_seen.find(*iterator);
            if (seen == domain.last_seen.end() || seen->second + 1 < cycle) {
                domain.last_seen.erase(*iterator);
                iterator = domain.pending.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }
}

std::uint64_t OnChipArbiter::select_pending(const Domain& domain) const {
    if (domain.pending.empty()) {
        return 0;
    }
    // First key strictly greater than the cursor, wrapping around.
    auto iterator = domain.pending.upper_bound(domain.cursor);
    if (iterator == domain.pending.end()) {
        iterator = domain.pending.begin();
    }
    return *iterator;
}

std::uint32_t OnChipArbiter::acquire(
    OnChipDomain domain,
    OnChipClientKind kind,
    std::uint32_t client_id,
    std::uint32_t bytes) {
    if (!configured_ || bytes == 0) {
        return 0;
    }
    Domain& state = domain_of(domain);
    const std::uint64_t key = key_of(kind, client_id);
    Stats& stats = stats_of(domain, kind, client_id);

    if (state.granted_key == 0) {
        // No requester selected yet this cycle.
        const std::uint64_t chosen = select_pending(state);
        if (chosen != 0 && chosen != key) {
            // A different pending requester owns this cycle's rotation slot.
            state.pending.insert(key);
            state.last_seen[key] = cycle_;
            ++stats.stalls;
            return 0;
        }
        state.granted_key = key;
        state.cursor = key;
        state.pending.erase(key);
        state.last_seen.erase(key);
    } else if (state.granted_key != key) {
        // Another requester already owns this cycle; the remainder is not
        // shared. Remember the need for the rotation.
        state.pending.insert(key);
        state.last_seen[key] = cycle_;
        ++stats.stalls;
        return 0;
    }

    const std::uint32_t remaining =
        state.used >= state.budget ? 0 : state.budget - state.used;
    const std::uint32_t granted = std::min(bytes, remaining);
    if (granted == 0) {
        state.pending.insert(key);
        state.last_seen[key] = cycle_;
        ++stats.stalls;
        return 0;
    }
    state.used += granted;
    stats.bytes += granted;
    ++stats.grants;
    ++domain_grants_[static_cast<int>(domain)];
    return granted;
}

void OnChipArbiter::clear_pending(
    OnChipDomain domain,
    OnChipClientKind kind,
    std::uint32_t client_id) {
    Domain& state = domain_of(domain);
    const std::uint64_t key = key_of(kind, client_id);
    state.pending.erase(key);
    state.last_seen.erase(key);
}

const OnChipArbiter::Stats& OnChipArbiter::stats(
    OnChipDomain domain,
    OnChipClientKind kind,
    std::uint32_t client_id) const {
    static const Stats empty{};
    const auto iterator = client_stats_.find(static_cast<int>(domain));
    if (iterator == client_stats_.end()) {
        return empty;
    }
    const auto found = iterator->second.find(key_of(kind, client_id));
    return found == iterator->second.end() ? empty : found->second;
}

OnChipArbiter::Stats& OnChipArbiter::stats_of(
    OnChipDomain domain,
    OnChipClientKind kind,
    std::uint32_t client_id) {
    return client_stats_[static_cast<int>(domain)][key_of(kind, client_id)];
}

std::uint32_t OnChipArbiter::remaining(OnChipDomain domain) const {
    const Domain& state = domain_of(domain);
    return state.used >= state.budget ? 0 : state.budget - state.used;
}

void OnChipArbiter::reset() {
    for (Domain& domain : domains_) {
        domain = Domain{};
    }
    domain_grants_[0] = 0;
    domain_grants_[1] = 0;
    client_stats_.clear();
    cycle_ = 0;
    last_begin_cycle_ = std::numeric_limits<std::uint64_t>::max();
}

}  // namespace hw::dataplane
