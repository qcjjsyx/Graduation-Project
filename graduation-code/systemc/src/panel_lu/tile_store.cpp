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

std::uint32_t TileStore::version(std::uint64_t key) const {
    return version(static_cast<std::uint32_t>(key));
}

bool TileStore::commit(std::uint64_t key, std::uint32_t expected_version) {
    return commit(static_cast<std::uint32_t>(key), expected_version);
}

void TileStore::clear() {
    slots_.clear();
}

bool SyntheticTileAccess::read_operand_segment(
    const TileOperand& operand,
    std::uint32_t row,
    std::uint32_t col_begin,
    std::uint32_t col_count,
    float* out_row,
    std::uint64_t cycle) {
    (void)cycle;
    if (!operand.valid || !store_.assigned(operand.handle.slot)) {
        return false;
    }
    const float* base = store_.data(operand.handle.slot) + operand.offset / 4;
    const std::uint32_t stride_floats = operand.stride_bytes / 4;
    for (std::uint32_t c = 0; c < col_count; ++c) {
        out_row[col_begin + c] = base[row * stride_floats + col_begin + c];
    }
    return true;
}

bool SyntheticTileAccess::write_result(
    const TileTask& task,
    const float* data,
    std::uint64_t cycle) {
    (void)cycle;
    if (!store_.assigned(task.dst.handle.slot)) {
        return false;
    }
    float* base = store_.data(task.dst.handle.slot) + task.dst.offset / 4;
    const std::uint32_t stride_floats = task.dst.stride_bytes / 4;
    for (std::uint32_t r = 0; r < task.dims.m; ++r) {
        for (std::uint32_t c = 0; c < task.dims.n; ++c) {
            base[r * stride_floats + c] = data[r * 16 + c];
        }
    }
    return true;
}

}  // namespace hw::panel_lu
