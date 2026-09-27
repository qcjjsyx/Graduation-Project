#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "command_codec.hpp"

namespace hw::dataplane {

// Device addresses are u64 byte offsets from the memory image base. Host
// pointers must never be used as device addresses.
enum class RegionRole {
    Command,     // command_buffer: runtime read-only
    Descriptor,  // descriptor_table: runtime read-only
    Payload,     // descriptor_payload: runtime read-only
    Completion,  // completion_status: executor writes (T05); readable here
    Data,        // matrix data / solution / workspace: writable DMA data region
    DataInput,   // rhs_data / permutation_data: read-only device input
    PVector,     // node.*.p_vector: INT32 device-local permutation
};

const char* region_role_name(RegionRole role);

struct RegionInfo {
    std::string name;
    RegionRole role{RegionRole::Data};
    std::uint64_t offset{0};
    std::uint64_t size{0};
};

// Access kinds define alignment and which region roles may be touched.
enum class AccessKind {
    Command,      // 32B records inside command_buffer
    Descriptor,   // 64B records inside descriptor_table
    Payload,      // u32 payload inside descriptor_payload
    Completion,   // 64B records inside completion_status
    Fp32Data,     // 4B elements, data regions only
    Int32Data,    // 4B elements, data/P-vector regions only
    ByteData,     // byte-exact DMA copies, data regions only
};

const char* access_kind_name(AccessKind kind);

struct AccessCheck {
    bool ok{false};
    bool zero_length{false};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    const RegionInfo* region{nullptr};
};

class DeviceMemory {
public:
    DeviceMemory() = default;
    DeviceMemory(std::vector<std::uint8_t> image, std::vector<RegionInfo> regions);

    std::uint64_t size() const { return static_cast<std::uint64_t>(image_.size()); }
    const std::vector<RegionInfo>& regions() const { return regions_; }

    const RegionInfo* find_region(std::uint64_t address) const;
    const RegionInfo* find_region_by_name(const std::string& name) const;

    // Validates address/length/kind/permission without touching memory.
    AccessCheck check(
        std::uint64_t address,
        std::uint64_t length,
        AccessKind kind,
        bool write) const;

    // Copies bytes out on success; on failure the output is cleared and the
    // image is untouched.
    hw::command::StatusCode read(
        std::uint64_t address,
        std::uint64_t length,
        AccessKind kind,
        std::vector<std::uint8_t>& out,
        std::string* reason = nullptr) const;

    // Copies bytes in on success; on failure the image is untouched.
    hw::command::StatusCode write(
        std::uint64_t address,
        const std::vector<std::uint8_t>& data,
        AccessKind kind,
        std::string* reason = nullptr);

    // Full read-only view for tests, loaders and final snapshots. Runtime
    // modules (descriptor reader, DMA, buffers) only receive MemoryService.
    const std::vector<std::uint8_t>& image() const { return image_; }
    std::vector<std::uint8_t> snapshot() const { return image_; }

private:
    std::vector<std::uint8_t> image_;
    std::vector<RegionInfo> regions_;
};

RegionRole classify_region_role(const std::string& name);

}  // namespace hw::dataplane
