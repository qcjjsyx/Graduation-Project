#include "panel_lu/tile_store.hpp"

#include <stdexcept>

namespace hw::panel_lu {

TileStore::TileStore(std::uint32_t slot_count) {
    resize(slot_count);
}

void TileStore::resize(std::uint32_t slot_count) {
    slots_.assign(slot_count, Slot{});
}

void TileStore::check_slot(std::uint32_t slot) const {
    if (slot >= slots_.size()) {
        throw std::out_of_range("TileStore slot out of range: " + std::to_string(slot));
    }
}

void TileStore::assign(
    std::uint32_t slot,
    const std::string& owner,
    std::uint32_t generation,
    const std::vector<float>& values) {
    check_slot(slot);
    Slot& entry = slots_[slot];
    entry.assigned = true;
    entry.owner = owner;
    entry.generation = generation;
    entry.version = 0;
    entry.commits = 0;
    entry.values.assign(kTileFloats, 0.0f);
    const std::size_t count = values.size() < kTileFloats ? values.size() : kTileFloats;
    for (std::size_t i = 0; i < count; ++i) {
        entry.values[i] = values[i];
    }
}

bool TileStore::assigned(std::uint32_t slot) const {
    check_slot(slot);
    return slots_[slot].assigned;
}

const std::string& TileStore::owner(std::uint32_t slot) const {
    check_slot(slot);
    return slots_[slot].owner;
}

std::uint32_t TileStore::generation(std::uint32_t slot) const {
    check_slot(slot);
    return slots_[slot].generation;
}

std::uint32_t TileStore::version(std::uint32_t slot) const {
    check_slot(slot);
    return slots_[slot].version;
}

std::uint64_t TileStore::commit_count(std::uint32_t slot) const {
    check_slot(slot);
    return slots_[slot].commits;
}

float* TileStore::data(std::uint32_t slot) {
    check_slot(slot);
    return slots_[slot].values.data();
}

const float* TileStore::data(std::uint32_t slot) const {
    check_slot(slot);
    return slots_[slot].values.data();
}

bool TileStore::commit(std::uint32_t slot, std::uint32_t expected_version) {
    check_slot(slot);
    Slot& entry = slots_[slot];
    if (entry.version != expected_version) {
        return false;
    }
    ++entry.version;
    ++entry.commits;
    return true;
}

void TileStore::bump_generations() {
    for (Slot& entry : slots_) {
        if (entry.assigned) {
            ++entry.generation;
        }
        entry.version = 0;
        entry.commits = 0;
    }
}

void TileStore::clear() {
    slots_.clear();
}

}  // namespace hw::panel_lu
