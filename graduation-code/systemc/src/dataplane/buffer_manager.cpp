#include "dataplane/buffer_manager.hpp"

#include <algorithm>
#include <stdexcept>

namespace hw::dataplane {

const char* buffer_state_name(BufferState state) {
    switch (state) {
    case BufferState::Free: return "FREE";
    case BufferState::Reserved: return "RESERVED";
    case BufferState::Loading: return "LOADING";
    case BufferState::Ready: return "READY";
    case BufferState::Computing: return "COMPUTING";
    case BufferState::Dirty: return "DIRTY";
    case BufferState::WritingBack: return "WRITING_BACK";
    }
    return "UNKNOWN";
}

BufferManager::BufferManager(BufferConfig config, TraceRecorder* trace)
    : config_(config), trace_(trace) {
    if (config_.slot_count < 2) {
        throw std::invalid_argument("BufferManager requires at least two slots");
    }
    if (config_.capacity_bytes == 0) {
        throw std::invalid_argument("BufferManager capacity must be positive");
    }
    if (config_.port_count == 0) {
        throw std::invalid_argument("BufferManager requires at least one port");
    }
    slots_.resize(config_.slot_count);
    for (auto& slot : slots_) {
        slot.storage.assign(static_cast<std::size_t>(config_.capacity_bytes), 0);
    }
}

BufferResult BufferManager::failure(
    hw::command::StatusCode status,
    std::string reason,
    std::uint64_t cycle,
    std::uint32_t slot) {
    ++conflicts_;
    BufferResult result;
    result.ok = false;
    result.status = status;
    result.reason = std::move(reason);
    result.transient =
        result.reason.find("conflicting active access") != std::string::npos ||
        result.reason.find("ports are busy") != std::string::npos;
    if (trace_ != nullptr && slot != hw::command::NONE && slot < slots_.size()) {
        const Slot& entry = slots_[slot];
        trace_->add_buffer(BufferRow{
            cycle, slot, entry.generation, entry.owner,
            buffer_state_name(entry.state), buffer_state_name(entry.state),
            result.reason});
    }
    return result;
}

const BufferManager::Slot* BufferManager::checked_slot(
    const BufferHandle& handle,
    const std::string& owner,
    hw::command::StatusCode* status,
    std::string* reason) const {
    if (handle.slot >= slots_.size()) {
        *status = hw::command::StatusCode::BadDescriptor;
        *reason = "buffer handle slot is out of range";
        return nullptr;
    }
    const Slot& slot = slots_[handle.slot];
    if (slot.generation != handle.generation) {
        *status = hw::command::StatusCode::BadDescriptor;
        *reason = "buffer handle is stale (generation mismatch)";
        return nullptr;
    }
    if (slot.state == BufferState::Free) {
        *status = hw::command::StatusCode::BadDescriptor;
        *reason = "buffer slot is free; handle was released or reset";
        return nullptr;
    }
    if (slot.owner != owner) {
        *status = hw::command::StatusCode::BadDescriptor;
        *reason = "buffer handle owner mismatch (current owner " + slot.owner + ")";
        return nullptr;
    }
    return &slot;
}

bool BufferManager::range_conflicts(const Slot& slot, bool write) const {
    const auto index = static_cast<std::uint32_t>(&slot - slots_.data());
    for (const auto& lease : leases_) {
        if (lease.slot != index) {
            continue;
        }
        if (write || lease.write) {
            return true;
        }
    }
    return false;
}

BufferResult BufferManager::make_lease(
    const BufferHandle& handle,
    const std::string& owner,
    bool write,
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t cycle,
    bool load_lease,
    bool writeback_lease) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (offset > config_.capacity_bytes ||
        bytes > config_.capacity_bytes - offset) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "buffer range exceeds slot capacity",
            cycle,
            handle.slot);
    }
    if (range_conflicts(*slot, write)) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "buffer slot has a conflicting active access",
            cycle,
            handle.slot);
    }
    if (leases_.size() >= config_.port_count) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "all on-chip buffer ports are busy",
            cycle,
            handle.slot);
    }

    Lease lease;
    lease.id = next_lease_id_++;
    lease.slot = handle.slot;
    lease.generation = handle.generation;
    lease.write = write;
    lease.owner = owner;
    lease.offset = offset;
    lease.length = bytes;
    lease.ready_cycle = cycle + config_.access_cycles;
    lease.load_lease = load_lease;
    lease.writeback_lease = writeback_lease;
    leases_.push_back(lease);

    BufferResult result;
    result.ok = true;
    result.handle = handle;
    result.lease = BufferLease{
        true, lease.id, lease.slot, lease.generation, lease.write,
        lease.offset, lease.length, cycle, lease.ready_cycle};
    return result;
}

BufferResult BufferManager::allocate(
    const std::string& owner,
    std::uint64_t required_bytes,
    std::uint64_t cycle) {
    if (required_bytes > config_.capacity_bytes) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "requested bytes exceed buffer capacity",
            cycle);
    }
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        Slot& slot = slots_[index];
        if (slot.state != BufferState::Free) {
            continue;
        }
        const BufferState previous = slot.state;
        slot.state = BufferState::Reserved;
        slot.owner = owner;
        slot.reserved_bytes = required_bytes;
        slot.valid_offset = 0;
        slot.valid_bytes = 0;
        log_transition(
            slot, index, owner, previous, BufferState::Reserved, "allocate", cycle);
        BufferResult result;
        result.ok = true;
        result.handle = BufferHandle{index, slot.generation};
        return result;
    }
    return failure(
        hw::command::StatusCode::BufferFull,
        "no free buffer slot",
        cycle);
}

BufferResult BufferManager::release(
    const BufferHandle& handle,
    const std::string& owner,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->active_lease != 0 || range_conflicts(*slot, true)) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "cannot release a buffer with an active transaction or access",
            cycle,
            handle.slot);
    }
    if (slot->state != BufferState::Ready && slot->state != BufferState::Dirty &&
        slot->state != BufferState::Reserved) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "only RESERVED, READY or DIRTY buffers may be released",
            cycle,
            handle.slot);
    }
    Slot& mutable_slot = slots_[handle.slot];
    const BufferState previous = mutable_slot.state;
    mutable_slot.state = BufferState::Free;
    mutable_slot.owner.clear();
    mutable_slot.valid_offset = 0;
    mutable_slot.valid_bytes = 0;
    mutable_slot.reserved_bytes = 0;
    mutable_slot.generation += 1;
    mutable_slot.active_lease = 0;
    log_transition(
        mutable_slot, handle.slot, owner, previous, BufferState::Free, "release", cycle);
    BufferResult result;
    result.ok = true;
    return result;
}

BufferResult BufferManager::transfer_owner(
    const BufferHandle& handle,
    const std::string& current_owner,
    const std::string& new_owner,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, current_owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->state != BufferState::Ready && slot->state != BufferState::Dirty) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "ownership can only be transferred for READY or DIRTY buffers",
            cycle,
            handle.slot);
    }
    if (slot->active_lease != 0 || range_conflicts(*slot, true)) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "cannot transfer ownership while a transaction or access is active",
            cycle,
            handle.slot);
    }
    Slot& mutable_slot = slots_[handle.slot];
    mutable_slot.owner = new_owner;
    log_transition(
        mutable_slot, handle.slot, new_owner, mutable_slot.state,
        mutable_slot.state, "transfer_owner", cycle);
    BufferResult result;
    result.ok = true;
    result.handle = handle;
    return result;
}

BufferResult BufferManager::rearm(
    const BufferHandle& handle,
    const std::string& owner,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->state != BufferState::Ready && slot->state != BufferState::Dirty) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "rearm requires a READY or DIRTY buffer",
            cycle,
            handle.slot);
    }
    if (slot->active_lease != 0 || range_conflicts(*slot, true)) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "cannot rearm a buffer with an active transaction or access",
            cycle,
            handle.slot);
    }
    Slot& mutable_slot = slots_[handle.slot];
    const BufferState previous = mutable_slot.state;
    mutable_slot.state = BufferState::Reserved;
    mutable_slot.valid_offset = 0;
    mutable_slot.valid_bytes = 0;
    mutable_slot.reserved_bytes = config_.capacity_bytes;
    log_transition(
        mutable_slot, handle.slot, owner, previous, BufferState::Reserved, "rearm", cycle);
    BufferResult result;
    result.ok = true;
    result.handle = handle;
    return result;
}

BufferResult BufferManager::begin_load(
    const BufferHandle& handle,
    const std::string& owner,
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->state != BufferState::Reserved) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "begin_load requires a RESERVED buffer",
            cycle,
            handle.slot);
    }
    const std::uint64_t limit =
        slot->reserved_bytes == 0 ? config_.capacity_bytes : slot->reserved_bytes;
    if (offset > limit || bytes > limit - offset) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "load range exceeds the reserved buffer range",
            cycle,
            handle.slot);
    }
    BufferResult result = make_lease(
        handle, owner, true, offset, bytes, cycle, true, false);
    if (!result.ok) {
        return result;
    }
    Slot& mutable_slot = slots_[handle.slot];
    const BufferState previous = mutable_slot.state;
    mutable_slot.state = BufferState::Loading;
    mutable_slot.active_lease = result.lease.lease_id;
    find_lease(result.lease.lease_id)->previous_state = previous;
    log_transition(
        mutable_slot, handle.slot, owner, previous, BufferState::Loading,
        "begin_load", cycle);
    return result;
}

BufferResult BufferManager::begin_write(
    const BufferHandle& handle,
    const std::string& owner,
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->state != BufferState::Reserved && slot->state != BufferState::Ready &&
        slot->state != BufferState::Dirty) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "begin_write requires RESERVED, READY or DIRTY",
            cycle,
            handle.slot);
    }
    if (offset > config_.capacity_bytes ||
        bytes > config_.capacity_bytes - offset) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "write range exceeds slot capacity",
            cycle,
            handle.slot);
    }
    BufferResult result = make_lease(
        handle, owner, true, offset, bytes, cycle, false, false);
    if (!result.ok) {
        return result;
    }
    Slot& mutable_slot = slots_[handle.slot];
    const BufferState previous = mutable_slot.state;
    mutable_slot.state = BufferState::Computing;
    mutable_slot.active_lease = result.lease.lease_id;
    find_lease(result.lease.lease_id)->previous_state = previous;
    log_transition(
        mutable_slot, handle.slot, owner, previous, BufferState::Computing,
        "begin_write", cycle);
    return result;
}

BufferResult BufferManager::begin_read(
    const BufferHandle& handle,
    const std::string& owner,
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->state != BufferState::Ready && slot->state != BufferState::Dirty) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "read requires READY or DIRTY buffer data",
            cycle,
            handle.slot);
    }
    const std::uint64_t valid_end = slot->valid_offset + slot->valid_bytes;
    if (offset < slot->valid_offset || offset > valid_end ||
        bytes > valid_end - offset) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "read range exceeds the valid data range",
            cycle,
            handle.slot);
    }
    return make_lease(handle, owner, false, offset, bytes, cycle, false, false);
}

BufferResult BufferManager::begin_writeback(
    const BufferHandle& handle,
    const std::string& owner,
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    const Slot* slot = checked_slot(handle, owner, &status, &reason);
    if (slot == nullptr) {
        return failure(status, std::move(reason), cycle, handle.slot);
    }
    if (slot->state != BufferState::Ready && slot->state != BufferState::Dirty) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "writeback requires READY or DIRTY buffer data",
            cycle,
            handle.slot);
    }
    const std::uint64_t valid_end = slot->valid_offset + slot->valid_bytes;
    if (offset < slot->valid_offset || offset > valid_end ||
        bytes > valid_end - offset) {
        return failure(
            hw::command::StatusCode::BufferFull,
            "writeback range exceeds the valid data range",
            cycle,
            handle.slot);
    }
    BufferResult result = make_lease(
        handle, owner, false, offset, bytes, cycle, false, true);
    if (!result.ok) {
        return result;
    }
    Slot& mutable_slot = slots_[handle.slot];
    const BufferState previous = mutable_slot.state;
    mutable_slot.state = BufferState::WritingBack;
    mutable_slot.active_lease = result.lease.lease_id;
    find_lease(result.lease.lease_id)->previous_state = previous;
    log_transition(
        mutable_slot, handle.slot, owner, previous, BufferState::WritingBack,
        "begin_writeback", cycle);
    return result;
}

BufferManager::Lease* BufferManager::find_lease(std::uint64_t lease_id) {
    for (auto& lease : leases_) {
        if (lease.id == lease_id) {
            return &lease;
        }
    }
    return nullptr;
}

void BufferManager::release_lease_record(std::uint64_t lease_id) {
    leases_.erase(
        std::remove_if(
            leases_.begin(), leases_.end(),
            [lease_id](const Lease& lease) { return lease.id == lease_id; }),
        leases_.end());
}

hw::command::StatusCode BufferManager::lease_write(
    std::uint64_t lease_id,
    std::uint64_t offset,
    const std::vector<std::uint8_t>& data,
    std::uint64_t cycle,
    std::string* reason) {
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown buffer lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (!lease->write) {
        if (reason != nullptr) {
            *reason = "buffer lease is read-only";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (cycle < lease->ready_cycle) {
        if (reason != nullptr) {
            *reason = "buffer lease is not ready yet";
        }
        return hw::command::StatusCode::BufferFull;
    }
    const std::uint64_t lease_end = lease->offset + lease->length;
    if (offset < lease->offset || offset > lease_end ||
        data.size() > lease_end - offset) {
        if (reason != nullptr) {
            *reason = "buffer write exceeds the leased range";
        }
        return hw::command::StatusCode::AddressFault;
    }
    Slot& slot = slots_[lease->slot];
    std::copy(
        data.begin(), data.end(),
        slot.storage.begin() + static_cast<std::ptrdiff_t>(offset));
    lease->access_bytes += data.size();
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode BufferManager::lease_read(
    std::uint64_t lease_id,
    std::uint64_t offset,
    std::uint64_t length,
    std::vector<std::uint8_t>& out,
    std::uint64_t cycle,
    std::string* reason) {
    out.clear();
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown buffer lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (cycle < lease->ready_cycle) {
        if (reason != nullptr) {
            *reason = "buffer lease is not ready yet";
        }
        return hw::command::StatusCode::BufferFull;
    }
    const std::uint64_t lease_end = lease->offset + lease->length;
    if (offset < lease->offset || offset > lease_end ||
        length > lease_end - offset) {
        if (reason != nullptr) {
            *reason = "buffer read exceeds the leased range";
        }
        return hw::command::StatusCode::AddressFault;
    }
    const Slot& slot = slots_[lease->slot];
    out.assign(
        slot.storage.begin() + static_cast<std::ptrdiff_t>(offset),
        slot.storage.begin() + static_cast<std::ptrdiff_t>(offset + length));
    lease->access_bytes += length;
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode BufferManager::release_lease(
    std::uint64_t lease_id,
    std::uint64_t cycle,
    std::string* reason) {
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown buffer lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (lease->write || lease->load_lease || lease->writeback_lease) {
        if (reason != nullptr) {
            *reason = "write leases must complete through their owning path";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    const std::uint32_t slot_index = lease->slot;
    const std::string owner = lease->owner;
    const BufferState current = slots_[slot_index].state;
    release_lease_record(lease_id);
    log_transition(
        slots_[slot_index], slot_index, owner, current, current,
        "release_read_lease", cycle);
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode BufferManager::abort_lease(
    std::uint64_t lease_id,
    std::uint64_t cycle,
    std::string* reason) {
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown buffer lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    const std::uint32_t slot_index = lease->slot;
    const std::string owner = lease->owner;
    const BufferState target = lease->previous_state;
    Slot& slot = slots_[slot_index];
    const BufferState previous = slot.state;
    slot.state = target;
    slot.active_lease = 0;
    release_lease_record(lease_id);
    log_transition(
        slot, slot_index, owner, previous, target, "abort_lease", cycle);
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode BufferManager::finish_load(
    std::uint64_t lease_id,
    std::uint64_t bytes_written,
    std::uint64_t cycle,
    std::string* reason) {
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr || !lease->load_lease) {
        if (reason != nullptr) {
            *reason = "unknown load lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (bytes_written > lease->length) {
        if (reason != nullptr) {
            *reason = "load completion exceeds the leased range";
        }
        return hw::command::StatusCode::AddressFault;
    }
    const std::uint32_t slot_index = lease->slot;
    const std::string owner = lease->owner;
    const std::uint64_t offset = lease->offset;
    Slot& slot = slots_[slot_index];
    const BufferState previous = slot.state;
    slot.state = BufferState::Ready;
    slot.valid_offset = offset;
    slot.valid_bytes = bytes_written;
    slot.active_lease = 0;
    release_lease_record(lease_id);
    log_transition(
        slot, slot_index, owner, previous, BufferState::Ready, "finish_load", cycle);
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode BufferManager::end_write(
    std::uint64_t lease_id,
    std::uint64_t bytes_written,
    std::uint64_t cycle,
    std::string* reason) {
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr || lease->load_lease || lease->writeback_lease) {
        if (reason != nullptr) {
            *reason = "unknown compute lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (bytes_written > lease->length) {
        if (reason != nullptr) {
            *reason = "write completion exceeds the leased range";
        }
        return hw::command::StatusCode::AddressFault;
    }
    const std::uint32_t slot_index = lease->slot;
    const std::string owner = lease->owner;
    const std::uint64_t offset = lease->offset;
    Slot& slot = slots_[slot_index];
    const BufferState previous = slot.state;
    const std::uint64_t previous_end = slot.valid_offset + slot.valid_bytes;
    const std::uint64_t end = offset + bytes_written;
    slot.valid_offset = std::min(slot.valid_offset, offset);
    slot.valid_bytes = std::max(previous_end, end) - slot.valid_offset;
    slot.state = BufferState::Dirty;
    slot.active_lease = 0;
    release_lease_record(lease_id);
    log_transition(
        slot, slot_index, owner, previous, BufferState::Dirty, "end_write", cycle);
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode BufferManager::finish_writeback(
    std::uint64_t lease_id,
    std::uint64_t bytes_read,
    std::uint64_t cycle,
    std::string* reason) {
    Lease* lease = find_lease(lease_id);
    if (lease == nullptr || !lease->writeback_lease) {
        if (reason != nullptr) {
            *reason = "unknown writeback lease";
        }
        return hw::command::StatusCode::BadDescriptor;
    }
    if (bytes_read > lease->length) {
        if (reason != nullptr) {
            *reason = "writeback completion exceeds the leased range";
        }
        return hw::command::StatusCode::AddressFault;
    }
    const std::uint32_t slot_index = lease->slot;
    const std::string owner = lease->owner;
    Slot& slot = slots_[slot_index];
    const BufferState previous = slot.state;
    slot.state = BufferState::Ready;
    slot.active_lease = 0;
    release_lease_record(lease_id);
    log_transition(
        slot, slot_index, owner, previous, BufferState::Ready,
        "finish_writeback", cycle);
    return hw::command::StatusCode::Ok;
}

void BufferManager::reset() {
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        Slot& slot = slots_[index];
        const BufferState previous = slot.state;
        slot.state = BufferState::Free;
        slot.owner.clear();
        slot.valid_offset = 0;
        slot.valid_bytes = 0;
        slot.reserved_bytes = 0;
        slot.active_lease = 0;
        slot.generation += 1;
        log_transition(slot, index, "", previous, BufferState::Free, "reset", 0);
    }
    leases_.clear();
}

BufferState BufferManager::state(std::uint32_t slot) const {
    return slots_.at(slot).state;
}
std::uint32_t BufferManager::generation(std::uint32_t slot) const {
    return slots_.at(slot).generation;
}
std::uint64_t BufferManager::valid_bytes(std::uint32_t slot) const {
    return slots_.at(slot).valid_bytes;
}
std::uint64_t BufferManager::valid_offset(std::uint32_t slot) const {
    return slots_.at(slot).valid_offset;
}
const std::vector<std::uint8_t>& BufferManager::bytes(std::uint32_t slot) const {
    return slots_.at(slot).storage;
}

void BufferManager::log_transition(
    const Slot& slot,
    std::uint32_t slot_index,
    const std::string& owner,
    BufferState from,
    BufferState to,
    const std::string& reason,
    std::uint64_t cycle) const {
    if (trace_ == nullptr) {
        return;
    }
    trace_->add_buffer(BufferRow{
        cycle, slot_index, slot.generation, owner, buffer_state_name(from),
        buffer_state_name(to), reason});
}

}  // namespace hw::dataplane
