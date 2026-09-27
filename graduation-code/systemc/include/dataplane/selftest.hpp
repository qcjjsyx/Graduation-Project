#pragma once

#include <cstdint>
#include <string>

#include "dataplane/dataplane.hpp"

namespace hw::dataplane {

struct DataPlaneSelftestOptions {
    std::string source_region;  // empty => first FRONT_DESC
    std::string target_region;  // empty => "node.0.update"
    std::uint32_t buffer_slots{2};
    std::uint64_t buffer_capacity_bytes{64 * 1024};
    std::uint32_t buffer_ports{2};
    std::uint32_t buffer_access_cycles{1};
    MemoryServiceConfig memory;
    DmaConfig dma;
    std::uint64_t timeout_cycles{2000000};
};

struct DataPlaneSelftestReport {
    bool ok{false};
    std::string failure;
    std::uint64_t image_bytes{0};
    std::string policy;
    std::uint32_t pivot_threshold_bits{0};
    std::string source_region;
    std::string target_region;
    std::uint32_t tile_rows{0};
    std::uint32_t tile_cols{0};
    std::uint64_t tile_bytes{0};
    std::uint64_t cycles{0};
    std::uint64_t dma_transfers{0};
    std::uint64_t dma_read_bytes{0};
    std::uint64_t dma_write_bytes{0};
    std::uint64_t descriptor_reads{0};
    std::uint64_t descriptor_read_bytes{0};
    bool target_verified{false};
    bool non_target_unchanged{false};
    bool control_regions_unchanged{false};
    // This writeback is test data moved through the data plane; it is NOT a
    // solved LU factor or solution.
    std::string writeback_note;
};

// Runs the descriptor -> DMA -> buffer -> consumer copy -> DMA writeback loop
// against a device run directory. `output_dir` may be empty; when set it
// receives transactions.csv, buffers.csv, summary.json and the final snapshot.
DataPlaneSelftestReport run_dataplane_selftest(
    const std::string& device_dir,
    const DataPlaneSelftestOptions& options,
    const std::string& output_dir);

}  // namespace hw::dataplane
