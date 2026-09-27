#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "dataplane/artifact_loader.hpp"
#include "dataplane/buffer_manager.hpp"
#include "dataplane/descriptor_reader.hpp"
#include "dataplane/dma_engine.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/trace.hpp"

namespace hw::dataplane {

struct DataPlaneConfig {
    MemoryServiceConfig memory;
    BufferConfig buffers;
    DmaConfig dma;
};

// Plain C++ data-plane orchestrator with deterministic tick order:
//   step(): ++cycle; memory.tick; route responses; reader.tick; dma.tick
// SystemC tests drive step() from sc_clock posegde events.
class DataPlane {
public:
    explicit DataPlane(LoadedArtifact artifact, DataPlaneConfig config = {});

    DataPlane(const DataPlane&) = delete;
    DataPlane& operator=(const DataPlane&) = delete;

    std::uint64_t step();
    bool run_until(
        const std::function<bool()>& done,
        std::uint64_t max_cycles = 1000000);
    void reset();

    std::uint64_t cycle() const { return cycle_; }

    DeviceMemory& memory() { return memory_; }
    const DeviceMemory& memory() const { return memory_; }
    MemoryService& memory_service() { return memory_service_; }
    BufferManager& buffers() { return buffers_; }
    DmaEngine& dma() { return dma_; }
    DescriptorReader& reader() { return reader_; }
    TraceRecorder& trace() { return trace_; }
    const DeviceBootConfig& boot() const { return artifact_.config; }

    // Convenience for tests/smoke: submits a DMA request and waits for one
    // terminal response, honoring the ready/valid handshake.
    bool run_dma_request(
        const DmaRequest& request,
        DmaResponse& response,
        std::uint64_t max_cycles = 1000000);

private:
    LoadedArtifact artifact_;
    DeviceMemory memory_;
    TraceRecorder trace_;
    MemoryService memory_service_;
    BufferManager buffers_;
    DmaEngine dma_;
    DescriptorReader reader_;
    std::uint64_t cycle_{0};
};

}  // namespace hw::dataplane
