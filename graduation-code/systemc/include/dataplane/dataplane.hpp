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

    // Full cycle step (T04 tests). ControlPlane drives the phases separately:
    //   begin_tick(); tick_memory_and_route(); <module updates>; tick_clients();
    std::uint64_t step();
    std::uint64_t begin_tick();
    void tick_memory_and_route();
    void tick_clients();
    bool run_until(
        const std::function<bool()>& done,
        std::uint64_t max_cycles = 1000000);
    void reset();

    // Route CommandFetch/CompletionWriter/ExecUnit responses to control-plane
    // modules. DescriptorReader and Dma keep their built-in routing; Test and
    // Other responses are dropped.
    void set_memory_sink(MemoryClient client, MemoryResponseSink* sink);

    std::uint64_t cycle() const { return cycle_; }

    DeviceMemory& memory() { return memory_; }
    const DeviceMemory& memory() const { return memory_; }
    MemoryService& memory_service() { return memory_service_; }
    const MemoryService& memory_service() const { return memory_service_; }
    BufferManager& buffers() { return buffers_; }
    const BufferManager& buffers() const { return buffers_; }
    DmaEngine& dma() { return dma_; }
    const DmaEngine& dma() const { return dma_; }
    DescriptorReader& reader() { return reader_; }
    const DescriptorReader& reader() const { return reader_; }
    TraceRecorder& trace() { return trace_; }
    const TraceRecorder& trace() const { return trace_; }
    const DeviceBootConfig& boot() const { return artifact_.config; }

    // Convenience for tests/smoke: submits a DMA request and waits for one
    // terminal response, honoring the ready/valid handshake.
    bool run_dma_request(
        const DmaRequest& request,
        DmaResponse& response,
        std::uint64_t max_cycles = 1000000);

private:
    LoadedArtifact artifact_;
    MemoryResponseSink* command_fetch_sink_{nullptr};
    MemoryResponseSink* completion_writer_sink_{nullptr};
    MemoryResponseSink* exec_unit_sink_{nullptr};
    DeviceMemory memory_;
    TraceRecorder trace_;
    MemoryService memory_service_;
    BufferManager buffers_;
    DmaEngine dma_;
    DescriptorReader reader_;
    std::uint64_t cycle_{0};
};

}  // namespace hw::dataplane
