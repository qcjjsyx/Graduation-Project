#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/device_memory.hpp"

namespace hw::dataplane {

class ArtifactLoadError : public std::runtime_error {
public:
    explicit ArtifactLoadError(const std::string& message)
        : std::runtime_error(message) {}
};

// Validated boot-time view of a schema v2 device artifact. Decoding is done for
// validation only; runtime descriptor reads must go through DescriptorReader.
struct DeviceBootConfig {
    std::uint64_t total_bytes{0};
    std::uint64_t alignment{64};
    hw::command::PivotPolicy pivot_policy{hw::command::PivotPolicy::Static};
    std::uint32_t pivot_threshold_bits{0};
    float pivot_threshold{0.0f};

    std::uint32_t command_count{0};
    std::uint32_t descriptor_count{0};
    std::uint32_t completion_count{0};
    std::uint32_t token_count{0};
    std::uint32_t max_wait_tokens{0};
    std::uint32_t node_count{0};

    std::uint64_t command_table_addr{0};
    std::uint64_t descriptor_table_addr{0};
    std::uint64_t completion_table_addr{0};

    std::vector<RegionInfo> regions;

    const RegionInfo* region(const std::string& name) const {
        for (const auto& entry : regions) {
            if (entry.name == name) {
                return &entry;
            }
        }
        return nullptr;
    }
};

struct LoadedArtifact {
    DeviceBootConfig config;
    std::vector<std::uint8_t> image;

    DeviceMemory device_memory() const {
        return DeviceMemory(image, config.regions);
    }
};

// Loads manifest.json and the memory image declared by it. The device run
// directory may contain only manifest.json and memory_image.bin: no golden or
// reference file is required, and no Python reference is invoked.
LoadedArtifact load_artifact(const std::string& device_dir);

// Test/loader entry point that does not touch the filesystem.
LoadedArtifact load_artifact_from_bytes(
    const std::string& manifest_json,
    std::vector<std::uint8_t> image,
    const std::string& source_label = "<memory>");

}  // namespace hw::dataplane
