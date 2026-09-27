#include "dataplane/device_memory.hpp"

#include <algorithm>
#include <limits>

namespace hw::dataplane {

namespace {

bool role_allowed(AccessKind kind, RegionRole role) {
    switch (kind) {
    case AccessKind::Command:
        return role == RegionRole::Command;
    case AccessKind::Descriptor:
        return role == RegionRole::Descriptor;
    case AccessKind::Payload:
        return role == RegionRole::Payload;
    case AccessKind::Completion:
        return role == RegionRole::Completion;
    case AccessKind::Fp32Data:
    case AccessKind::Int32Data:
    case AccessKind::ByteData:
        // DMA data traffic must never touch command/descriptor/payload space.
        return role == RegionRole::Data || role == RegionRole::DataInput ||
               role == RegionRole::Completion || role == RegionRole::PVector;
    }
    return false;
}

bool write_allowed(RegionRole role) {
    // command/descriptor/payload are runtime read-only; numeric outputs and
    // status regions may be written according to purpose.
    return role == RegionRole::Completion || role == RegionRole::Data ||
           role == RegionRole::PVector;
}

std::string describe_region(const RegionInfo* region) {
    if (region == nullptr) {
        return "outside any region";
    }
    return "region " + region->name;
}

}  // namespace

const char* region_role_name(RegionRole role) {
    switch (role) {
    case RegionRole::Command: return "command";
    case RegionRole::Descriptor: return "descriptor";
    case RegionRole::Payload: return "payload";
    case RegionRole::Completion: return "completion";
    case RegionRole::Data: return "data";
    case RegionRole::DataInput: return "data_input";
    case RegionRole::PVector: return "pvector";
    }
    return "unknown";
}

const char* access_kind_name(AccessKind kind) {
    switch (kind) {
    case AccessKind::Command: return "command";
    case AccessKind::Descriptor: return "descriptor";
    case AccessKind::Payload: return "payload";
    case AccessKind::Completion: return "completion";
    case AccessKind::Fp32Data: return "fp32";
    case AccessKind::Int32Data: return "int32";
    case AccessKind::ByteData: return "byte";
    }
    return "unknown";
}

RegionRole classify_region_role(const std::string& name) {
    if (name == "command_buffer") {
        return RegionRole::Command;
    }
    if (name == "descriptor_table") {
        return RegionRole::Descriptor;
    }
    if (name == "descriptor_payload") {
        return RegionRole::Payload;
    }
    if (name == "completion_status") {
        return RegionRole::Completion;
    }
    if (name == "rhs_data" || name == "permutation_data") {
        return RegionRole::DataInput;
    }
    const std::string suffix = ".p_vector";
    if (name == "p_vector" ||
        (name.size() >= suffix.size() &&
         name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)) {
        return RegionRole::PVector;
    }
    return RegionRole::Data;
}

DeviceMemory::DeviceMemory(
    std::vector<std::uint8_t> image,
    std::vector<RegionInfo> regions)
    : image_(std::move(image)), regions_(std::move(regions)) {}

const RegionInfo* DeviceMemory::find_region(std::uint64_t address) const {
    for (const auto& region : regions_) {
        if (address >= region.offset && address < region.offset + region.size) {
            return &region;
        }
    }
    return nullptr;
}

const RegionInfo* DeviceMemory::find_region_by_name(const std::string& name) const {
    for (const auto& region : regions_) {
        if (region.name == name) {
            return &region;
        }
    }
    return nullptr;
}

AccessCheck DeviceMemory::check(
    std::uint64_t address,
    std::uint64_t length,
    AccessKind kind,
    bool write) const {
    AccessCheck result;

    if (length == 0) {
        // Zero-byte operations never touch memory but must still resolve to a
        // legal region (or the exact end of one); they may not bypass checks.
        const RegionInfo* region = nullptr;
        for (const auto& candidate : regions_) {
            if (address >= candidate.offset && address < candidate.offset + candidate.size) {
                region = &candidate;
                break;
            }
        }
        if (region == nullptr) {
            for (const auto& candidate : regions_) {
                if (address == candidate.offset + candidate.size) {
                    region = &candidate;
                    break;
                }
            }
        }
        if (region == nullptr) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "zero-length access is outside any region";
            return result;
        }
        if (!role_allowed(kind, region->role)) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = std::string("zero-length ") + access_kind_name(kind) +
                            " access is not allowed in " + region->name;
            return result;
        }
        if (write && !write_allowed(region->role)) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "zero-length write to read-only " + region->name;
            return result;
        }
        result.ok = true;
        result.zero_length = true;
        result.region = region;
        return result;
    }

    if (length > std::numeric_limits<std::uint64_t>::max() - address) {
        result.status = hw::command::StatusCode::AddressFault;
        result.reason = "address + length overflows u64";
        return result;
    }
    const std::uint64_t end = address + length;

    const RegionInfo* region = nullptr;
    for (const auto& candidate : regions_) {
        const std::uint64_t candidate_end = candidate.offset + candidate.size;
        if (address >= candidate.offset && end <= candidate_end) {
            region = &candidate;
            break;
        }
    }
    if (region == nullptr) {
        result.status = hw::command::StatusCode::AddressFault;
        result.reason = "access crosses a region boundary or lands in padding";
        return result;
    }

    switch (kind) {
    case AccessKind::Command:
        if (address % 32 != 0 || length % 32 != 0) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "command access must use 32-byte records";
            return result;
        }
        break;
    case AccessKind::Descriptor:
        if (address % 64 != 0 || length != 64) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "descriptor access must be exactly one aligned 64-byte record";
            return result;
        }
        break;
    case AccessKind::Completion:
        if (address % 64 != 0 || length % 64 != 0) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "completion access must use 64-byte records";
            return result;
        }
        break;
    case AccessKind::Payload:
        if (address % 4 != 0 || length % 4 != 0) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "descriptor payload access must be u32 aligned";
            return result;
        }
        break;
    case AccessKind::Fp32Data:
    case AccessKind::Int32Data:
        if (address % 4 != 0 || length % 4 != 0) {
            result.status = hw::command::StatusCode::AddressFault;
            result.reason = "element access must be 4-byte aligned";
            return result;
        }
        break;
    case AccessKind::ByteData:
        break;
    }

    if (!role_allowed(kind, region->role)) {
        result.status = hw::command::StatusCode::AddressFault;
        result.reason = std::string(access_kind_name(kind)) +
                        " access is not allowed in " + describe_region(region);
        return result;
    }
    if (write && !write_allowed(region->role)) {
        result.status = hw::command::StatusCode::AddressFault;
        result.reason = "write to read-only " + region->name;
        return result;
    }

    result.ok = true;
    result.region = region;
    return result;
}

hw::command::StatusCode DeviceMemory::read(
    std::uint64_t address,
    std::uint64_t length,
    AccessKind kind,
    std::vector<std::uint8_t>& out,
    std::string* reason) const {
    out.clear();
    const AccessCheck checked = check(address, length, kind, false);
    if (!checked.ok) {
        if (reason != nullptr) {
            *reason = checked.reason;
        }
        return checked.status;
    }
    if (checked.zero_length) {
        return hw::command::StatusCode::Ok;
    }
    out.assign(
        image_.begin() + static_cast<std::ptrdiff_t>(address),
        image_.begin() + static_cast<std::ptrdiff_t>(address + length));
    return hw::command::StatusCode::Ok;
}

hw::command::StatusCode DeviceMemory::write(
    std::uint64_t address,
    const std::vector<std::uint8_t>& data,
    AccessKind kind,
    std::string* reason) {
    const AccessCheck checked = check(
        address, static_cast<std::uint64_t>(data.size()), kind, true);
    if (!checked.ok) {
        if (reason != nullptr) {
            *reason = checked.reason;
        }
        return checked.status;
    }
    if (checked.zero_length) {
        return hw::command::StatusCode::Ok;
    }
    std::copy(
        data.begin(), data.end(),
        image_.begin() + static_cast<std::ptrdiff_t>(address));
    return hw::command::StatusCode::Ok;
}

}  // namespace hw::dataplane
