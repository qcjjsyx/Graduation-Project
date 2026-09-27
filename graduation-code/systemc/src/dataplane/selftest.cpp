#include "dataplane/selftest.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <fstream>
#include <optional>
#include <sstream>

#include "nlohmann/json.hpp"

namespace hw::dataplane {

namespace {

namespace cmd = hw::command;

bool poll_reader(
    DataPlane& plane,
    std::uint64_t request_id,
    ReadOutcome& outcome,
    std::uint64_t timeout) {
    const bool done = plane.run_until(
        [&]() { return !plane.reader().pending(request_id); }, timeout);
    if (!done) {
        return false;
    }
    return plane.reader().poll(request_id, outcome);
}

bool read_region(
    DataPlane& plane,
    std::uint32_t descriptor_id,
    RegionView& view,
    std::uint64_t timeout,
    std::string* reason) {
    const auto request_id = plane.reader().request(
        ReadKind::Region, descriptor_id, plane.cycle());
    ReadOutcome outcome;
    if (!poll_reader(plane, request_id, outcome, timeout)) {
        *reason = "descriptor read timed out";
        return false;
    }
    if (outcome.status != cmd::StatusCode::Ok) {
        *reason = outcome.reason;
        return false;
    }
    view = outcome.region;
    return true;
}

bool read_front(
    DataPlane& plane,
    std::uint32_t descriptor_id,
    FrontView& view,
    std::uint64_t timeout,
    std::string* reason) {
    const auto request_id = plane.reader().request(
        ReadKind::Front, descriptor_id, plane.cycle());
    ReadOutcome outcome;
    if (!poll_reader(plane, request_id, outcome, timeout)) {
        *reason = "descriptor read timed out";
        return false;
    }
    if (outcome.status != cmd::StatusCode::Ok) {
        *reason = outcome.reason;
        return false;
    }
    view = outcome.front;
    return true;
}

bool wait_lease(DataPlane& plane, const BufferResult& result, std::uint64_t timeout) {
    if (!result.ok) {
        return false;
    }
    if (plane.cycle() >= result.lease.ready_cycle) {
        return true;
    }
    return plane.run_until(
        [&]() { return plane.cycle() >= result.lease.ready_cycle; }, timeout);
}

std::vector<std::uint8_t> expected_tile(
    const DeviceMemory& memory,
    const RegionView& region,
    std::uint64_t offset,
    std::uint32_t rows,
    std::uint32_t cols) {
    std::vector<std::uint8_t> expected;
    expected.reserve(
        static_cast<std::size_t>(rows) * cols * region.element_bytes);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint64_t address =
            region.base_addr + offset +
            static_cast<std::uint64_t>(row) * region.row_stride;
        std::vector<std::uint8_t> row_bytes;
        std::string reason;
        if (memory.read(
                address, static_cast<std::uint64_t>(cols) * region.element_bytes,
                AccessKind::Fp32Data, row_bytes, &reason) != cmd::StatusCode::Ok) {
            return {};
        }
        expected.insert(expected.end(), row_bytes.begin(), row_bytes.end());
    }
    return expected;
}

void write_summary(
    const std::string& output_dir,
    const DeviceBootConfig& boot,
    const DataPlaneSelftestOptions& options,
    const DataPlaneSelftestReport& report,
    DataPlane& plane) {
    nlohmann::json summary;
    summary["tool"] = "t04_dataplane_selftest";
    summary["scope"] =
        "descriptor -> DMA -> scratchpad -> consumer copy -> DMA writeback; "
        "no command execution and no numerical kernel";
    summary["ok"] = report.ok;
    summary["failure"] = report.failure;
    summary["artifact"] = {
        {"image_bytes", boot.total_bytes},
        {"command_count", boot.command_count},
        {"descriptor_count", boot.descriptor_count},
        {"completion_count", boot.completion_count},
        {"node_count", boot.node_count},
        {"pivot_policy",
         boot.pivot_policy == cmd::PivotPolicy::Static ? "STATIC" : "DYNAMIC_PANEL"},
        {"pivot_threshold_bits_fp32", boot.pivot_threshold_bits},
    };
    summary["transfer"] = {
        {"source_region", report.source_region},
        {"target_region", report.target_region},
        {"tile_rows", report.tile_rows},
        {"tile_cols", report.tile_cols},
        {"tile_bytes", report.tile_bytes},
        {"writeback_note", report.writeback_note},
    };
    summary["cycles"] = {
        {"finished_at", report.cycles},
        {"memory_port_busy", plane.memory_service().port_busy_cycles()},
    };
    summary["descriptor_reader"] = {
        {"completed", plane.reader().completed_count()},
        {"errors", plane.reader().error_count()},
        {"memory_reads", report.descriptor_reads},
        {"read_bytes", report.descriptor_read_bytes},
    };
    summary["dma"] = {
        {"transfers", report.dma_transfers},
        {"completed", plane.dma().completed_count()},
        {"failed", plane.dma().failed_count()},
        {"read_bytes", report.dma_read_bytes},
        {"write_bytes", report.dma_write_bytes},
    };
    summary["memory_service"] = {
        {"accepted", plane.memory_service().accepted_count()},
        {"errors", plane.memory_service().error_count()},
        {"read_bytes", plane.memory_service().read_bytes()},
        {"write_bytes", plane.memory_service().write_bytes()},
    };
    summary["verification"] = {
        {"target_verified", report.target_verified},
        {"non_target_unchanged", report.non_target_unchanged},
        {"control_regions_unchanged", report.control_regions_unchanged},
    };
    summary["config"] = {
        {"buffer_slots", options.buffer_slots},
        {"buffer_capacity_bytes", options.buffer_capacity_bytes},
        {"buffer_ports", options.buffer_ports},
        {"buffer_access_cycles", options.buffer_access_cycles},
        {"memory_startup_cycles", options.memory.startup_cycles},
        {"memory_beat_bytes", options.memory.beat_bytes},
        {"memory_beat_cycles", options.memory.beat_cycles},
        {"memory_max_inflight", options.memory.max_inflight},
        {"dma_beat_bytes", options.dma.beat_bytes},
        {"dma_startup_cycles", options.dma.startup_cycles},
        {"dma_beat_interval_cycles", options.dma.beat_interval_cycles},
    };
    std::ofstream out(std::filesystem::path(output_dir) / "summary.json");
    out << summary.dump(2) << '\n';
}

}  // namespace

DataPlaneSelftestReport run_dataplane_selftest(
    const std::string& device_dir,
    const DataPlaneSelftestOptions& options,
    const std::string& output_dir) {
    DataPlaneSelftestReport report;
    report.writeback_note =
        "the target region receives test bytes copied through the data plane; "
        "it is not a solved LU factor or solution and no command was executed";

    LoadedArtifact artifact;
    try {
        artifact = load_artifact(device_dir);
    } catch (const std::exception& exception) {
        report.failure = std::string("artifact load failed: ") + exception.what();
        return report;
    }

    DataPlaneConfig config;
    config.memory = options.memory;
    config.buffers.slot_count = options.buffer_slots;
    config.buffers.capacity_bytes = options.buffer_capacity_bytes;
    config.buffers.port_count = options.buffer_ports;
    config.buffers.access_cycles = options.buffer_access_cycles;
    config.dma = options.dma;
    DataPlane plane(std::move(artifact), config);

    report.image_bytes = plane.boot().total_bytes;
    report.policy = plane.boot().pivot_policy == cmd::PivotPolicy::Static
                        ? "STATIC"
                        : "DYNAMIC_PANEL";
    report.pivot_threshold_bits = plane.boot().pivot_threshold_bits;

    const auto fail = [&](const std::string& reason) {
        report.failure = reason;
        report.cycles = plane.cycle();
        report.dma_transfers = plane.dma().accepted_count();
        report.dma_read_bytes = plane.dma().read_bytes();
        report.dma_write_bytes = plane.dma().write_bytes();
        report.descriptor_reads = plane.reader().memory_reads();
        report.descriptor_read_bytes = plane.reader().read_bytes();
        if (!output_dir.empty()) {
            plane.trace().write_csv(output_dir);
            std::ofstream snapshot(
                std::filesystem::path(output_dir) / "final_memory_image.bin",
                std::ios::binary);
            const auto bytes = plane.memory().snapshot();
            snapshot.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
            write_summary(output_dir, plane.boot(), options, report, plane);
        }
        return report;
    };

    // 1. Runtime descriptor scan (every read is a device transaction).
    std::map<std::string, RegionView> region_views;
    for (std::uint32_t id = 0; id < plane.boot().descriptor_count; ++id) {
        RegionView view;
        std::string reason;
        if (!read_region(plane, id, view, options.timeout_cycles, &reason)) {
            continue;
        }
        region_views[view.name] = view;
    }
    std::vector<FrontView> fronts;
    for (std::uint32_t id = 0; id < plane.boot().descriptor_count; ++id) {
        FrontView view;
        std::string reason;
        if (!read_front(plane, id, view, options.timeout_cycles, &reason)) {
            continue;
        }
        fronts.push_back(view);
    }
    std::optional<FrontView> front;
    for (const auto& view : fronts) {
        if (!options.source_region.empty() && view.front.name != options.source_region) {
            continue;
        }
        if (!front.has_value() || view.total_dim > front->total_dim) {
            front = view;
        }
    }
    if (!front.has_value()) {
        return fail("no FRONT_DESC matched the requested source region");
    }
    // Node prefix of the chosen front, e.g. "node.7" from "node.7.front".
    std::string node_prefix{};
    const auto dot = front->front.name.rfind('.');
    if (dot != std::string::npos) {
        node_prefix = front->front.name.substr(0, dot);
    }

    std::optional<RegionView> target;
    if (!options.target_region.empty()) {
        const auto found = region_views.find(options.target_region);
        if (found == region_views.end()) {
            return fail("target region not found: " + options.target_region);
        }
        target = found->second;
    } else {
        // Prefer a writable scratch region of the same node that yields the
        // largest square tile within the buffer capacity; the writeback is
        // test data, never a solved factor or solution.
        std::vector<std::string> candidates;
        if (!node_prefix.empty()) {
            candidates.push_back(node_prefix + ".update");
            candidates.push_back(node_prefix + ".factor_l");
            candidates.push_back(node_prefix + ".solve_workspace");
        }
        candidates.push_back("node.0.update");
        candidates.push_back("solution_data");
        std::uint32_t best_tile = 0;
        for (const auto& name : candidates) {
            const auto found = region_views.find(name);
            if (found == region_views.end()) {
                continue;
            }
            const std::uint32_t tile = std::min<std::uint32_t>(
                {front->total_dim, found->second.rows, found->second.cols});
            const std::uint64_t bytes =
                static_cast<std::uint64_t>(tile) * tile * 4;
            if (tile > best_tile && bytes <= options.buffer_capacity_bytes) {
                best_tile = tile;
                target = found->second;
            }
        }
        if (!target.has_value()) {
            for (const auto& entry : region_views) {
                if (entry.second.rows > 0 && entry.second.cols > 0) {
                    const auto* info = plane.boot().region(entry.first);
                    if (info != nullptr && info->role == RegionRole::Data) {
                        target = entry.second;
                        break;
                    }
                }
            }
        }
        if (!target.has_value()) {
            return fail("no writable target region found");
        }
    }
    report.source_region = front->front.name;
    report.target_region = target->name;

    const std::uint32_t tile_rows = std::min<std::uint32_t>(
        {front->total_dim, target->rows, target->cols});
    const std::uint32_t tile_cols = tile_rows;
    if (tile_rows == 0) {
        return fail("target region cannot hold a tile");
    }
    report.tile_rows = tile_rows;
    report.tile_cols = tile_cols;
    report.tile_bytes =
        static_cast<std::uint64_t>(tile_rows) * tile_cols * front->front.element_bytes;
    if (report.tile_bytes > options.buffer_capacity_bytes) {
        return fail(
            "tile of " + std::to_string(report.tile_bytes) +
            " bytes exceeds the configured scratchpad capacity of " +
            std::to_string(options.buffer_capacity_bytes) + " bytes");
    }

    const std::vector<std::uint8_t> expected = expected_tile(
        plane.memory(), front->front, 0, tile_rows, tile_cols);
    if (expected.empty()) {
        return fail("could not read the expected source tile from device memory");
    }
    const std::vector<std::uint8_t> initial_image = plane.memory().snapshot();

    // 2. Allocate buffer A and DMA-load the source tile from DDR.
    auto allocate_a = plane.buffers().allocate("dma", report.tile_bytes, plane.cycle());
    if (!allocate_a.ok) {
        return fail("buffer A allocation failed: " + allocate_a.reason);
    }
    DmaRequest load;
    load.request_id = 1;
    load.has_command = false;
    load.direction = DmaDirection::MemoryToBuffer;
    load.ddr_region = front->front;
    load.ddr_offset = 0;
    load.ddr_row_stride = front->front.row_stride;
    load.buffer = allocate_a.handle;
    load.buffer_offset = 0;
    load.buffer_row_stride = static_cast<std::uint64_t>(tile_cols) * 4;
    load.rows = tile_rows;
    load.cols = tile_cols;
    load.element_bytes = front->front.element_bytes;
    load.label = "selftest_load";
    DmaResponse load_response;
    if (!plane.run_dma_request(load, load_response, options.timeout_cycles)) {
        return fail("DMA load timed out");
    }
    if (load_response.status != cmd::StatusCode::Ok) {
        return fail("DMA load failed: " + load_response.reason);
    }
    if (plane.buffers().bytes(allocate_a.handle.slot).size() <
        static_cast<std::size_t>(report.tile_bytes)) {
        return fail("buffer A capacity introspection failed");
    }
    {
        const auto& storage = plane.buffers().bytes(allocate_a.handle.slot);
        if (!std::equal(expected.begin(), expected.end(), storage.begin())) {
            return fail("DMA load did not reproduce the source tile bytes");
        }
    }

    // 3. Consumer copies A -> B with buffer leases (predictable byte copy).
    auto handoff_a = plane.buffers().transfer_owner(
        allocate_a.handle, "dma", "consumer", plane.cycle());
    if (!handoff_a.ok) {
        return fail("buffer A ownership handoff failed: " + handoff_a.reason);
    }
    auto allocate_b = plane.buffers().allocate("consumer", report.tile_bytes, plane.cycle());
    if (!allocate_b.ok) {
        return fail("buffer B allocation failed: " + allocate_b.reason);
    }
    auto write_b = plane.buffers().begin_write(
        allocate_b.handle, "consumer", 0, report.tile_bytes, plane.cycle());
    if (!wait_lease(plane, write_b, options.timeout_cycles)) {
        return fail("buffer B write lease failed: " + write_b.reason);
    }
    auto read_a = plane.buffers().begin_read(
        allocate_a.handle, "consumer", 0, report.tile_bytes, plane.cycle());
    if (!wait_lease(plane, read_a, options.timeout_cycles)) {
        return fail("buffer A read lease failed: " + read_a.reason);
    }
    const std::uint64_t chunk = 64;
    for (std::uint64_t offset = 0; offset < report.tile_bytes; offset += chunk) {
        const std::uint64_t length = std::min<std::uint64_t>(chunk, report.tile_bytes - offset);
        std::vector<std::uint8_t> data;
        std::string reason;
        if (plane.buffers().lease_read(
                read_a.lease.lease_id, offset, length, data, plane.cycle(), &reason) !=
            cmd::StatusCode::Ok) {
            return fail("consumer read failed: " + reason);
        }
        if (plane.buffers().lease_write(
                write_b.lease.lease_id, offset, data, plane.cycle(), &reason) !=
            cmd::StatusCode::Ok) {
            return fail("consumer write failed: " + reason);
        }
    }
    if (plane.buffers().release_lease(
            read_a.lease.lease_id, plane.cycle()) != cmd::StatusCode::Ok) {
        return fail("consumer read lease release failed");
    }
    if (plane.buffers().end_write(
            write_b.lease.lease_id, report.tile_bytes, plane.cycle()) !=
        cmd::StatusCode::Ok) {
        return fail("consumer write lease completion failed");
    }
    {
        const auto& storage = plane.buffers().bytes(allocate_b.handle.slot);
        if (!std::equal(expected.begin(), expected.end(), storage.begin())) {
            return fail("consumer copy did not reproduce the tile bytes in buffer B");
        }
    }

    // 4. DMA writeback B -> writable DDR region with test data.
    auto handoff_b = plane.buffers().transfer_owner(
        allocate_b.handle, "consumer", "dma", plane.cycle());
    if (!handoff_b.ok) {
        return fail("buffer B ownership handoff failed: " + handoff_b.reason);
    }
    DmaRequest writeback = load;
    writeback.request_id = 2;
    writeback.direction = DmaDirection::BufferToMemory;
    writeback.ddr_region = *target;
    writeback.ddr_offset = 0;
    writeback.ddr_row_stride = target->row_stride;
    writeback.buffer = allocate_b.handle;
    writeback.label = "selftest_writeback";
    DmaResponse writeback_response;
    if (!plane.run_dma_request(writeback, writeback_response, options.timeout_cycles)) {
        return fail("DMA writeback timed out");
    }
    if (writeback_response.status != cmd::StatusCode::Ok) {
        return fail("DMA writeback failed: " + writeback_response.reason);
    }

    // 5. Verify target and every non-target byte.
    const std::vector<std::uint8_t> target_expected = expected_tile(
        plane.memory(), *target, 0, tile_rows, tile_cols);
    {
        std::vector<std::uint8_t> target_after;
        std::string reason;
        if (plane.memory().read(
                target->base_addr, report.tile_bytes, AccessKind::Fp32Data,
                target_after, &reason) != cmd::StatusCode::Ok) {
            return fail("could not read back the target region: " + reason);
        }
        report.target_verified = target_after == target_expected;
    }
    {
        const auto final_image = plane.memory().snapshot();
        bool unchanged = final_image.size() == initial_image.size();
        if (unchanged) {
            for (std::uint64_t index = 0; index < final_image.size(); ++index) {
                const bool in_target = index >= target->base_addr &&
                                       index < target->base_addr + target->byte_size;
                if (!in_target && final_image[index] != initial_image[index]) {
                    unchanged = false;
                    break;
                }
            }
        }
        report.non_target_unchanged = unchanged;
    }
    {
        // Control regions are read-only for the data plane; compare the
        // initial and final snapshots directly (raw byte comparison, not a
        // device data access).
        const auto final_image = plane.memory().snapshot();
        bool unchanged = true;
        for (const std::string name : {
                 "command_buffer", "descriptor_table", "descriptor_payload",
                 "completion_status"}) {
            const RegionInfo* region = plane.boot().region(name);
            if (region == nullptr) {
                continue;
            }
            unchanged = unchanged && std::equal(
                initial_image.begin() + static_cast<std::ptrdiff_t>(region->offset),
                initial_image.begin() +
                    static_cast<std::ptrdiff_t>(region->offset + region->size),
                final_image.begin() + static_cast<std::ptrdiff_t>(region->offset));
        }
        report.control_regions_unchanged = unchanged;
    }

    // 6. Release buffers and finish.
    plane.buffers().release(allocate_a.handle, "consumer", plane.cycle());
    plane.buffers().release(allocate_b.handle, "dma", plane.cycle());

    report.cycles = plane.cycle();
    report.dma_transfers = plane.dma().accepted_count();
    report.dma_read_bytes = plane.dma().read_bytes();
    report.dma_write_bytes = plane.dma().write_bytes();
    report.descriptor_reads = plane.reader().memory_reads();
    report.descriptor_read_bytes = plane.reader().read_bytes();
    report.ok = report.target_verified && report.non_target_unchanged &&
                report.control_regions_unchanged;

    if (!output_dir.empty()) {
        plane.trace().write_csv(output_dir);
        std::ofstream snapshot(
            std::filesystem::path(output_dir) / "final_memory_image.bin",
            std::ios::binary);
        const auto bytes = plane.memory().snapshot();
        snapshot.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        write_summary(output_dir, plane.boot(), options, report, plane);
    }
    if (!report.ok && report.failure.empty()) {
        report.failure = "verification failed";
    }
    return report;
}

}  // namespace hw::dataplane
