// SystemC timing tests for the T04 data plane. Every step is driven by an
// sc_clock posedge: DataPlane is a plain C++ transaction model, and the SystemC
// testbench owns the clock/timing layer.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <systemc>

#include "dataplane/dataplane.hpp"
#include "dataplane/device_memory.hpp"

namespace {

namespace cmd = hw::command;
namespace dp = hw::dataplane;

constexpr std::uint64_t kTotalBytes = 4096;

dp::RegionView region_view(const dp::DeviceBootConfig& boot, const std::string& name) {
    dp::RegionView view;
    const auto* info = boot.region(name);
    if (info == nullptr) {
        throw std::runtime_error("unknown test region " + name);
    }
    view.descriptor_id = 0;
    view.base_addr = info->offset;
    view.byte_size = info->size;
    view.rows = 1;
    view.cols = static_cast<std::uint32_t>(info->size / 4);
    view.row_stride = info->size;
    view.format = cmd::DataFormat::Fp32;
    view.layout = cmd::DataLayout::RowMajor;
    view.element_bytes = 4;
    view.name = name;
    return view;
}

dp::LoadedArtifact make_artifact() {
    dp::LoadedArtifact artifact;
    artifact.config.total_bytes = kTotalBytes;
    artifact.config.regions = {
        {"command_buffer", dp::RegionRole::Command, 0, 64},
        {"descriptor_table", dp::RegionRole::Descriptor, 64, 128},
        {"descriptor_payload", dp::RegionRole::Payload, 192, 64},
        {"completion_status", dp::RegionRole::Completion, 256, 64},
        {"rhs_data", dp::RegionRole::DataInput, 320, 128},
        {"node.0.front", dp::RegionRole::Data, 448, 1024},
        {"node.0.update", dp::RegionRole::Data, 1472, 1024},
        {"node.0.p_vector", dp::RegionRole::PVector, 2496, 128},
        {"node.0.factor_l", dp::RegionRole::Data, 2624, 1024},
        {"node.0.factor_u", dp::RegionRole::Data, 3648, 448},
    };
    artifact.image.assign(kTotalBytes, 0);
    return artifact;
}

dp::DataPlaneConfig make_config() {
    dp::DataPlaneConfig config;
    config.memory.startup_cycles = 1;
    config.memory.beat_bytes = 8;
    config.memory.beat_cycles = 1;
    config.memory.max_inflight = 8;
    config.buffers.slot_count = 2;
    config.buffers.capacity_bytes = 512;
    config.buffers.port_count = 2;
    config.buffers.access_cycles = 1;
    config.dma.beat_bytes = 8;
    config.dma.startup_cycles = 1;
    config.dma.beat_interval_cycles = 1;
    return config;
}

std::vector<std::uint8_t> pattern(std::size_t size, std::uint8_t seed) {
    std::vector<std::uint8_t> values(size);
    for (std::size_t index = 0; index < size; ++index) {
        values[index] = static_cast<std::uint8_t>((index * 37 + seed) & 0xFF);
    }
    return values;
}

class T04SystemCTests : public sc_core::sc_module {
public:
    sc_core::sc_in<bool> clk;

    SC_CTOR(T04SystemCTests) { SC_THREAD(run); }

    int failures() const { return failures_; }

private:
    int failures_{0};
    int checks_{0};

    void check(bool condition, const std::string& message) {
        ++checks_;
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void wait_step(dp::DataPlane& plane) {
        wait(clk.posedge_event());
        plane.step();
    }

    bool run_until(
        dp::DataPlane& plane,
        const std::function<bool()>& done,
        std::uint64_t max_cycles) {
        const std::uint64_t start = plane.cycle();
        while (!done()) {
            if (plane.cycle() - start >= max_cycles) {
                return false;
            }
            wait_step(plane);
        }
        return true;
    }

    void start_dma(dp::DataPlane& plane, const dp::DmaRequest& request) {
        plane.dma().set_request(request);
        plane.dma().set_request_valid(true);
    }

    bool wait_dma(
        dp::DataPlane& plane,
        dp::DmaResponse& response,
        std::uint64_t max_cycles = 100000) {
        if (!run_until(plane, [&]() { return plane.dma().response_valid(); }, max_cycles)) {
            return false;
        }
        response = plane.dma().response();
        plane.dma().set_request_valid(false);
        plane.dma().set_response_ready(true);
        wait_step(plane);
        plane.dma().set_response_ready(false);
        return true;
    }

    bool run_dma(
        dp::DataPlane& plane,
        const dp::DmaRequest& request,
        dp::DmaResponse& response) {
        start_dma(plane, request);
        return wait_dma(plane, response);
    }

    bool lease_ready(dp::DataPlane& plane, dp::BufferResult& result) {
        if (!result.ok) {
            return false;
        }
        return run_until(
            plane,
            [&]() { return plane.cycle() >= result.lease.ready_cycle; },
            100000);
    }

    std::vector<std::uint8_t> read_storage(
        dp::DataPlane& plane,
        const dp::BufferHandle& handle,
        std::size_t length) {
        const auto& storage = plane.buffers().bytes(handle.slot);
        return std::vector<std::uint8_t>(storage.begin(), storage.begin() + length);
    }

    void test_dma_address_overflow() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const auto input = region_view(artifact.config, "rhs_data");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 64, 0x6D);
        std::fill(
            artifact.image.begin() + input.base_addr,
            artifact.image.begin() + input.base_addr + input.byte_size, 0x3C);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 64, plane.cycle());
        check(buffer.ok, "overflow: allocate");

        // offset + (rows-1)*stride + row_bytes wraps u64.
        dp::DmaRequest wrapped = {};
        wrapped.request_id = 201;
        wrapped.direction = dp::DmaDirection::MemoryToBuffer;
        wrapped.ddr_region = front;
        wrapped.ddr_offset = 0xFFFFFFFFFFFFFFC0ull;
        wrapped.rows = 1;
        wrapped.cols = 16;
        wrapped.element_bytes = 4;
        wrapped.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, wrapped, response), "overflow: terminal response");
        check(
            response.status == cmd::StatusCode::AddressFault,
            "overflow: wrapping offset is AddressFault");
        check(response.bytes == 0, "overflow: no bytes moved");
        check(
            plane.buffers().state(buffer.handle.slot) == dp::BufferState::Reserved,
            "overflow: buffer untouched");

        // Strided range that ends past the declared region.
        dp::DmaRequest past_end = wrapped;
        past_end.request_id = 202;
        past_end.ddr_offset = front.byte_size - 8;
        past_end.rows = 2;
        past_end.cols = 2;
        past_end.ddr_row_stride = 8;
        check(run_dma(plane, past_end, response), "overflow: second terminal response");
        check(
            response.status == cmd::StatusCode::AddressFault,
            "overflow: range past the region is AddressFault");

        std::vector<std::uint8_t> other;
        plane.memory().read(input.base_addr, 16, dp::AccessKind::Fp32Data, other);
        check(
            std::all_of(other.begin(), other.end(), [](std::uint8_t v) { return v == 0x3C; }),
            "overflow: no read escaped into another region");
    }

    void test_reset_drops_waiting_request() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 128, 0x18);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto first = plane.buffers().allocate("dma", 64, plane.cycle());
        auto second = plane.buffers().allocate("dma", 16, plane.cycle());
        dp::DmaRequest first_request = {};
        first_request.request_id = 211;
        first_request.direction = dp::DmaDirection::MemoryToBuffer;
        first_request.ddr_region = front;
        first_request.rows = 1;
        first_request.cols = 16;
        first_request.element_bytes = 4;
        first_request.buffer = first.handle;
        dp::DmaRequest second_request = first_request;
        second_request.request_id = 212;
        second_request.ddr_offset = 64;
        second_request.cols = 4;
        second_request.buffer = second.handle;

        start_dma(plane, first_request);
        wait_step(plane);  // first accepted
        plane.dma().set_request(second_request);
        plane.dma().set_request_valid(true);
        wait_step(plane);  // second is latched as waiting
        check(plane.dma().accepted_count() == 1, "reset-wait: only the first is accepted");

        plane.reset();
        for (int index = 0; index < 5; ++index) {
            wait_step(plane);
            check(!plane.dma().response_valid(), "reset-wait: no stale response");
        }
        check(
            plane.dma().accepted_count() == 1,
            "reset-wait: waiting request is not accepted after reset");
        check(
            plane.dma().failed_count() == 0,
            "reset-wait: no error response for the dropped waiting request");
        check(
            plane.buffers().state(first.handle.slot) == dp::BufferState::Free,
            "reset-wait: stale handles rejected");

        auto fresh = plane.buffers().allocate("dma", 16, plane.cycle());
        check(fresh.ok, "reset-wait: fresh allocation");
        dp::DmaRequest retry = second_request;
        retry.request_id = 213;
        retry.buffer = fresh.handle;
        dp::DmaResponse response;
        check(run_dma(plane, retry, response), "reset-wait: new request completes");
        check(response.status == cmd::StatusCode::Ok, "reset-wait: new request status");
    }

    void test_zero_size_after_transfer() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 32, 0x27);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 16, plane.cycle());
        dp::DmaRequest load = {};
        load.request_id = 221;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.rows = 1;
        load.cols = 4;
        load.element_bytes = 4;
        load.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, load, response), "zero-after: 16B transfer completes");
        check(response.bytes == 16, "zero-after: 16B moved");
        const auto read_bytes_after_transfer = plane.dma().read_bytes();

        auto empty_buffer = plane.buffers().allocate("dma", 8, plane.cycle());
        dp::DmaRequest empty = load;
        empty.request_id = 222;
        empty.cols = 0;
        empty.buffer = empty_buffer.handle;
        check(run_dma(plane, empty, response), "zero-after: zero-size response");
        check(response.bytes == 0, "zero-after: zero-size reports zero bytes");
        check(
            plane.dma().read_bytes() == read_bytes_after_transfer,
            "zero-after: counters do not inherit the previous transfer");

        dp::DmaRequest bad = load;
        bad.request_id = 223;
        bad.ddr_offset = 0xFFFFFFFFFFFFFFF0ull;
        check(run_dma(plane, bad, response), "zero-after: failing request response");
        check(response.status == cmd::StatusCode::AddressFault, "zero-after: failure status");
        check(response.bytes == 0, "zero-after: failed request reports zero bytes");
        check(
            plane.dma().read_bytes() == read_bytes_after_transfer,
            "zero-after: failed request does not add bytes");
    }

    void run() {
        test_dma_on_chip_budget_segmented_staging();
        test_contiguous_and_timing();
        test_2d_strides_and_tail();
        test_bit_patterns_preserved();
        test_request_backpressure();
        test_response_backpressure_stable();
        test_ping_pong_three_blocks();
        test_buffer_resource_errors();
        test_reset_interruption();
        test_no_lost_or_duplicated_transfers();
        test_illegal_request_no_partial_write();
        test_zero_size_transfer();
        test_unrelated_regions_unchanged();
        test_dma_address_overflow();
        test_reset_drops_waiting_request();
        test_zero_size_after_transfer();

        std::cout << checks_ << " checks, " << failures_ << " failures\n";
        sc_core::sc_stop();
    }


    // Regression: BufferToMemory beats are staged exactly once under the
    // unified read budget. Budget waits and DDR backpressure retries must not
    // re-read staged bytes or re-zero the staging buffer.
    void test_dma_on_chip_budget_segmented_staging() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const auto update = region_view(artifact.config, "node.0.update");
        const std::vector<std::uint8_t> source = pattern(64, 11);
        artifact.image.assign(kTotalBytes, 0);
        std::copy(source.begin(), source.end(), artifact.image.begin() + front.base_addr);

        dp::DataPlaneConfig config = make_config();
        config.dma.beat_bytes = 16;     // 16B beats: 4 beats over 64B
        config.memory.max_inflight = 1;  // tight DDR queue -> submit retries
        dp::DataPlane plane(std::move(artifact), config);

        dp::OnChipArbiter arbiter;
        arbiter.configure(16, 16);  // 16B/cycle read budget -> 4 grants/64B
        plane.set_on_chip_arbiter(&arbiter);


        auto buffer = plane.buffers().allocate("dma", 64, plane.cycle());
        check(buffer.ok, "dma budget: buffer allocated");
        dp::DmaRequest load;
        load.request_id = 71;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.rows = 2;
        load.cols = 8;
        load.element_bytes = 4;
        load.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, load, response), "dma budget: load completes");
        check(response.status == cmd::StatusCode::Ok, "dma budget: load status");

        // Occupy the single DDR queue slot with a longer transaction so the
        // writeback's DDR submit retries while the beat is staged.
        dp::MemoryRequest blocker;
        blocker.request_id = 900;
        blocker.client = dp::MemoryClient::Test;
        blocker.write = false;
        blocker.address = front.base_addr;
        blocker.length = 256;
        blocker.kind = dp::AccessKind::Fp32Data;
        check(
            plane.memory_service().submit(blocker, plane.cycle()),
            "dma budget: DDR blocker accepted");
        const std::uint64_t access_before =
            plane.buffers().total_lease_access_bytes();
        dp::DmaRequest writeback = load;
        writeback.request_id = 72;
        writeback.direction = dp::DmaDirection::BufferToMemory;
        writeback.ddr_region = update;
        check(run_dma(plane, writeback, response), "dma budget: writeback completes");
        const std::uint64_t writeback_access =
            plane.buffers().total_lease_access_bytes() - access_before;
        check(response.status == cmd::StatusCode::Ok, "dma budget: writeback status");

        check(
            plane.dma().scratch_read_bytes() == 64,
            "dma budget: exactly one 64B scratch read (no beat re-read)");
        check(
            writeback_access == 64,
            "dma budget: lease accounting shows exactly one 64B read");
        const auto& read_stats =
            arbiter.stats(dp::OnChipDomain::Read, dp::OnChipClientKind::Dma, 0);
        check(read_stats.bytes == 64, "dma budget: all reads charged to the budget");
        check(
            plane.dma().scratch_read_bytes() == read_stats.bytes,
            "dma budget: total scratch reads equal budgeted reads");
        check(read_stats.grants == 4, "dma budget: 16B budget -> four grants");
        check(
            arbiter.grants(dp::OnChipDomain::Read) <= plane.cycle(),
            "dma budget: at most one read grant per cycle");
        std::vector<std::uint8_t> written;
        plane.memory().read(
            update.base_addr, 64, dp::AccessKind::Fp32Data, written);
        check(written == source, "dma budget: staged bytes written exactly once");
    }

    void test_contiguous_and_timing() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const std::vector<std::uint8_t> source = pattern(32, 3);
        artifact.image.assign(kTotalBytes, 0);
        std::copy(source.begin(), source.end(), artifact.image.begin() + front.base_addr);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 32, plane.cycle());
        check(buffer.ok, "contiguous: allocate");
        dp::DmaRequest load;
        load.request_id = 11;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.rows = 1;
        load.cols = 8;
        load.element_bytes = 4;
        load.buffer = buffer.handle;
        load.label = "contiguous_load";
        dp::DmaResponse response;
        check(run_dma(plane, load, response), "contiguous: load completes");
        check(response.status == cmd::StatusCode::Ok, "contiguous: load status");
        // Independent cycle expectation for the fixed configuration:
        // accept=1 (first tick), first beat=2, each beat = 2 memory cycles plus
        // one interval cycle, so beat k starts at 2+3k and finishes at 4+3k.
        check(response.accept_cycle == 1, "contiguous: accept cycle");
        check(response.first_beat_cycle == 2, "contiguous: first beat cycle");
        check(response.last_beat_cycle == 13, "contiguous: last beat cycle");
        check(response.finish_cycle == 13, "contiguous: finish cycle");
        check(response.bytes == 32, "contiguous: byte count");
        check(
            read_storage(plane, buffer.handle, 32) == source,
            "contiguous: loaded bytes are byte-exact");

        const auto update = region_view(plane.boot(), "node.0.update");
        dp::DmaRequest writeback = load;
        writeback.request_id = 12;
        writeback.direction = dp::DmaDirection::BufferToMemory;
        writeback.ddr_region = update;
        check(run_dma(plane, writeback, response), "contiguous: writeback completes");
        check(response.status == cmd::StatusCode::Ok, "contiguous: writeback status");
        std::vector<std::uint8_t> written;
        check(
            plane.memory().read(update.base_addr, 32, dp::AccessKind::Fp32Data, written) ==
                cmd::StatusCode::Ok,
            "contiguous: read back target");
        check(written == source, "contiguous: writeback is byte-exact");
    }

    void test_2d_strides_and_tail() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        // Three rows of three FP32 values with a 16-byte row stride and a
        // 4-byte starting offset; each row has a 4-byte tail beat.
        std::vector<std::uint8_t> rows(3 * 12);
        for (std::size_t index = 0; index < rows.size(); ++index) {
            rows[index] = static_cast<std::uint8_t>(0x10 + index);
        }
        artifact.image.assign(kTotalBytes, 0);
        for (std::uint32_t row = 0; row < 3; ++row) {
            std::copy(
                rows.begin() + row * 12, rows.begin() + (row + 1) * 12,
                artifact.image.begin() + front.base_addr + 4 + row * 16);
        }
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 3 * 32, plane.cycle());
        check(buffer.ok, "2d: allocate");
        dp::DmaRequest load;
        load.request_id = 21;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.ddr_offset = 4;
        load.ddr_row_stride = 16;
        load.buffer = buffer.handle;
        load.buffer_offset = 8;
        load.buffer_row_stride = 32;
        load.rows = 3;
        load.cols = 3;
        load.element_bytes = 4;
        dp::DmaResponse response;
        check(run_dma(plane, load, response), "2d: transfer completes");
        check(response.status == cmd::StatusCode::Ok, "2d: status");
        check(response.bytes == 36, "2d: byte count with tails");
        // 6 beats (two per row): last beat starts at 2+3*5=17 and finishes 19.
        check(response.last_beat_cycle == 19, "2d: predictable last beat cycle");
        const auto storage = read_storage(plane, buffer.handle, 96);
        bool exact = true;
        for (std::uint32_t row = 0; row < 3; ++row) {
            exact = exact && std::equal(
                rows.begin() + row * 12, rows.begin() + (row + 1) * 12,
                storage.begin() + 8 + row * 32);
        }
        check(exact, "2d: strided rows and tail beats are byte-exact");

        // The valid range must be the layout span (76 bytes), not the moved
        // byte count (36); otherwise the last row cannot be read or written
        // back through the normal interface.
        check(
            plane.buffers().valid_bytes(buffer.handle.slot) == 76,
            "2d: valid range is the layout span");
        check(
            plane.buffers().transfer_owner(buffer.handle, "dma", "consumer", plane.cycle()).ok,
            "2d: handoff to consumer");
        auto read = plane.buffers().begin_read(
            buffer.handle, "consumer", 8, 76, plane.cycle());
        check(lease_ready(plane, read), "2d: consumer read lease over the span");
        bool read_back_exact = true;
        for (std::uint32_t row = 0; row < 3; ++row) {
            std::vector<std::uint8_t> row_bytes;
            const auto status = plane.buffers().lease_read(
                read.lease.lease_id, 8 + row * 32, 12, row_bytes, plane.cycle());
            read_back_exact = read_back_exact &&
                              status == cmd::StatusCode::Ok &&
                              std::equal(
                                  rows.begin() + row * 12, rows.begin() + (row + 1) * 12,
                                  row_bytes.begin());
        }
        check(read_back_exact, "2d: every row readable through the buffer interface");
        plane.buffers().release_lease(read.lease.lease_id, plane.cycle());
        check(
            plane.buffers().transfer_owner(buffer.handle, "consumer", "dma", plane.cycle()).ok,
            "2d: handoff back to DMA");

        const auto update = region_view(plane.boot(), "node.0.update");
        dp::DmaRequest writeback = load;
        writeback.request_id = 22;
        writeback.direction = dp::DmaDirection::BufferToMemory;
        writeback.ddr_region = update;
        writeback.ddr_offset = 0;
        writeback.ddr_row_stride = 32;
        check(run_dma(plane, writeback, response), "2d: writeback completes");
        check(response.status == cmd::StatusCode::Ok, "2d: writeback status");
        bool writeback_exact = true;
        for (std::uint32_t row = 0; row < 3; ++row) {
            std::vector<std::uint8_t> target;
            plane.memory().read(
                update.base_addr + row * 32, 12, dp::AccessKind::Fp32Data, target);
            writeback_exact = writeback_exact && std::equal(
                rows.begin() + row * 12, rows.begin() + (row + 1) * 12, target.begin());
        }
        check(writeback_exact, "2d: strided writeback through the buffer interface");
    }

    void test_bit_patterns_preserved() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const std::vector<std::uint8_t> bits{
            0x00, 0x00, 0x80, 0x7F,  // +inf-like FP32 pattern
            0x01, 0x00, 0x80, 0x7F,  // NaN-like pattern
            0x00, 0x00, 0x00, 0x80,  // -0.0
            0xFF, 0xFF, 0xFF, 0x7F,  // max float
            0x01, 0x00, 0x00, 0x00,  // subnormal
            0x00, 0x00, 0x00, 0x00,  // zero
            0xAA, 0x55, 0xAA, 0x55,  // arbitrary INT32 pattern
            0x01, 0x02, 0x03, 0x04,  // arbitrary bytes
        };
        artifact.image.assign(kTotalBytes, 0);
        std::copy(bits.begin(), bits.end(), artifact.image.begin() + front.base_addr);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", bits.size(), plane.cycle());
        dp::DmaRequest load;
        load.request_id = 31;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.rows = 1;
        load.cols = static_cast<std::uint32_t>(bits.size() / 4);
        load.element_bytes = 4;
        load.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, load, response), "bits: load completes");
        check(
            read_storage(plane, buffer.handle, bits.size()) == bits,
            "bits: FP32/INT32 patterns survive DMA unchanged");

        const auto update = region_view(plane.boot(), "node.0.update");
        dp::DmaRequest writeback = load;
        writeback.request_id = 32;
        writeback.direction = dp::DmaDirection::BufferToMemory;
        writeback.ddr_region = update;
        check(run_dma(plane, writeback, response), "bits: writeback completes");
        std::vector<std::uint8_t> written;
        plane.memory().read(update.base_addr, bits.size(), dp::AccessKind::Fp32Data, written);
        check(written == bits, "bits: no implicit numerical conversion");
    }

    void test_request_backpressure() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 128, 0x42);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 64, plane.cycle());
        dp::DmaRequest first;
        first.request_id = 41;
        first.direction = dp::DmaDirection::MemoryToBuffer;
        first.ddr_region = front;
        first.rows = 1;
        first.cols = 16;
        first.element_bytes = 4;
        first.buffer = buffer.handle;
        start_dma(plane, first);
        check(
            run_until(plane, [&]() { return plane.dma().accepted_count() == 1; }, 1000),
            "backpressure: first request accepted");
        check(!plane.dma().request_ready(), "backpressure: busy engine is not ready");

        dp::DmaRequest second = first;
        second.request_id = 42;
        second.ddr_offset = 64;
        plane.dma().set_request(second);
        plane.dma().set_request_valid(true);
        const std::uint64_t issue_cycle = plane.cycle() + 1;
        for (int index = 0; index < 3; ++index) {
            check(
                !plane.dma().request_ready(),
                "backpressure: second request waits while first is active");
            wait_step(plane);
        }
        check(plane.dma().accepted_count() == 1, "backpressure: no early acceptance");

        check(
            run_until(plane, [&]() { return plane.dma().response_valid(); }, 100000),
            "backpressure: first response asserted");
        const dp::DmaResponse first_response = plane.dma().response();
        check(first_response.status == cmd::StatusCode::Ok, "backpressure: first status");
        plane.dma().set_response_ready(true);
        wait_step(plane);
        plane.dma().set_response_ready(false);
        check(
            run_until(plane, [&]() { return plane.dma().accepted_count() == 2; }, 1000),
            "backpressure: second request accepted after the first response");
        dp::DmaResponse second_response;
        check(wait_dma(plane, second_response), "backpressure: second completes");
        check(second_response.request_id == 42, "backpressure: second response id");
        check(
            second_response.issue_cycle == issue_cycle,
            "backpressure: issue cycle recorded while waiting");
        check(
            second_response.request_wait_cycles > 0,
            "backpressure: request wait cycles are counted separately");
    }

    void test_response_backpressure_stable() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const auto update = region_view(artifact.config, "node.0.update");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 32, 0x21);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("consumer", 32, plane.cycle());
        dp::DmaRequest writeback;
        writeback.request_id = 51;
        writeback.direction = dp::DmaDirection::BufferToMemory;
        writeback.ddr_region = update;
        writeback.rows = 1;
        writeback.cols = 8;
        writeback.element_bytes = 4;
        writeback.buffer = buffer.handle;
        auto write = plane.buffers().begin_write(buffer.handle, "consumer", 0, 32, plane.cycle());
        check(lease_ready(plane, write), "response backpressure: write lease");
        std::vector<std::uint8_t> data = pattern(32, 9);
        plane.buffers().lease_write(write.lease.lease_id, 0, data, plane.cycle());
        plane.buffers().end_write(write.lease.lease_id, 32, plane.cycle());
        plane.buffers().transfer_owner(buffer.handle, "consumer", "dma", plane.cycle());

        start_dma(plane, writeback);
        check(
            run_until(plane, [&]() { return plane.dma().response_valid(); }, 100000),
            "response backpressure: response asserted");
        const dp::DmaResponse captured = plane.dma().response();
        const auto write_bytes_before = plane.memory_service().write_bytes();
        for (int index = 0; index < 3; ++index) {
            wait_step(plane);
            const auto& current = plane.dma().response();
            check(current.request_id == captured.request_id, "response stays stable");
            check(current.status == captured.status, "response status stays stable");
            check(current.bytes == captured.bytes, "response bytes stay stable");
            check(
                current.finish_cycle == captured.finish_cycle,
                "response finish cycle stays stable");
        }
        check(
            plane.memory_service().write_bytes() == write_bytes_before,
            "blocked response does not duplicate DDR writes");
        plane.dma().set_response_ready(true);
        wait_step(plane);
        plane.dma().set_response_ready(false);
        check(!plane.dma().response_valid(), "response consumed exactly once");
    }

    void test_ping_pong_three_blocks() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        artifact.image.assign(kTotalBytes, 0);
        std::vector<std::vector<std::uint8_t>> blocks(3);
        for (int index = 0; index < 3; ++index) {
            blocks[index] = pattern(32, static_cast<std::uint8_t>(0x30 + index));
            std::copy(
                blocks[index].begin(), blocks[index].end(),
                artifact.image.begin() + front.base_addr + index * 64);
        }
        dp::DataPlane plane(std::move(artifact), make_config());

        auto allocate = [&](const std::string& owner) {
            return plane.buffers().allocate(owner, 32, plane.cycle());
        };
        auto make_load = [&](std::uint32_t id, int block, const dp::BufferHandle& handle) {
            dp::DmaRequest request;
            request.request_id = id;
            request.direction = dp::DmaDirection::MemoryToBuffer;
            request.ddr_region = front;
            request.ddr_offset = static_cast<std::uint64_t>(block) * 64;
            request.rows = 1;
            request.cols = 8;
            request.element_bytes = 4;
            request.buffer = handle;
            return request;
        };
        auto consume = [&](const dp::BufferHandle& handle, const std::vector<std::uint8_t>& expected) {
            auto read = plane.buffers().begin_read(handle, "consumer", 0, 32, plane.cycle());
            if (!lease_ready(plane, read)) {
                check(false, "ping-pong: consumer read lease");
                return;
            }
            std::vector<std::uint8_t> copied;
            plane.buffers().lease_read(read.lease.lease_id, 0, 32, copied, plane.cycle());
            plane.buffers().release_lease(read.lease.lease_id, plane.cycle());
            check(copied == expected, "ping-pong: consumer read is byte-exact");
        };

        // Block 0 -> A.
        auto a = allocate("dma");
        check(a.ok, "ping-pong: allocate A");
        dp::DmaResponse response;
        check(run_dma(plane, make_load(61, 0, a.handle), response), "ping-pong: load A0");
        check(response.status == cmd::StatusCode::Ok, "ping-pong: load A0 status");
        check(read_storage(plane, a.handle, 32) == blocks[0], "ping-pong: A0 data");
        check(
            plane.buffers().transfer_owner(a.handle, "dma", "consumer", plane.cycle()).ok,
            "ping-pong: A ownership handoff");

        // Block 1 -> B while the consumer reads A.
        auto b = allocate("dma");
        check(b.ok, "ping-pong: allocate B");
        dp::DmaRequest load_b = make_load(62, 1, b.handle);
        start_dma(plane, load_b);
        wait_step(plane);
        check(plane.dma().busy(), "ping-pong: DMA busy loading B");
        consume(a.handle, blocks[0]);
        check(plane.dma().busy(), "ping-pong: consumer read overlapped the DMA load");
        dp::DmaResponse response_b;
        check(wait_dma(plane, response_b), "ping-pong: load B completes");
        check(response_b.status == cmd::StatusCode::Ok, "ping-pong: load B status");
        check(read_storage(plane, b.handle, 32) == blocks[1], "ping-pong: B1 data");

        // Release A, load block 2 into A while the consumer reads B.
        check(
            plane.buffers().release(a.handle, "consumer", plane.cycle()).ok,
            "ping-pong: release A");
        check(
            plane.buffers().transfer_owner(b.handle, "dma", "consumer", plane.cycle()).ok,
            "ping-pong: B ownership handoff");
        auto a2 = allocate("dma");
        check(a2.ok, "ping-pong: reallocate A");
        dp::DmaRequest load_a2 = make_load(63, 2, a2.handle);
        start_dma(plane, load_a2);
        wait_step(plane);
        check(plane.dma().busy(), "ping-pong: DMA busy loading A2");
        consume(b.handle, blocks[1]);
        dp::DmaResponse response_a2;
        check(wait_dma(plane, response_a2), "ping-pong: load A2 completes");
        check(read_storage(plane, a2.handle, 32) == blocks[2], "ping-pong: A2 data");
        check(
            plane.buffers().transfer_owner(a2.handle, "dma", "consumer", plane.cycle()).ok,
            "ping-pong: A2 ownership handoff");
        consume(a2.handle, blocks[2]);
        check(plane.dma().failed_count() == 0, "ping-pong: no failed transactions");
    }

    void test_buffer_resource_errors() {
        auto plane = dp::DataPlane(make_artifact(), make_config());
        auto a = plane.buffers().allocate("dma", 128, plane.cycle());
        auto b = plane.buffers().allocate("consumer", 128, plane.cycle());
        check(a.ok && b.ok, "resources: two buffers");
        auto c = plane.buffers().allocate("dma", 128, plane.cycle());
        check(!c.ok && c.status == cmd::StatusCode::BufferFull, "resources: third is full");
        auto load_a = plane.buffers().begin_load(a.handle, "dma", 0, 128, plane.cycle());
        check(load_a.ok, "resources: load A");
        check(
            plane.buffers().finish_load(load_a.lease.lease_id, 128, plane.cycle()) ==
                cmd::StatusCode::Ok,
            "resources: finish load A");
        check(
            !plane.buffers().begin_load(b.handle, "dma", 0, 128, plane.cycle()).ok,
            "resources: wrong owner");
        check(
            plane.buffers().release(a.handle, "dma", plane.cycle()).ok,
            "resources: release A");
        check(
            !plane.buffers().release(a.handle, "dma", plane.cycle()).ok,
            "resources: double release");
        check(
            !plane.buffers().begin_read(a.handle, "dma", 0, 4, plane.cycle()).ok,
            "resources: stale handle");
        plane.reset();
        check(
            plane.buffers().state(b.handle.slot) == dp::BufferState::Free,
            "resources: reset frees slots");
        check(!plane.dma().response_valid(), "resources: reset clears responses");
        check(
            !plane.buffers().begin_read(b.handle, "consumer", 0, 4, plane.cycle()).ok,
            "resources: handle across reset is stale");
    }

    void test_reset_interruption() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 64, 0x77);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 64, plane.cycle());
        dp::DmaRequest load;
        load.request_id = 71;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.rows = 2;
        load.cols = 8;
        load.element_bytes = 4;
        load.buffer = buffer.handle;
        start_dma(plane, load);
        for (int index = 0; index < 3; ++index) {
            wait_step(plane);
        }
        check(plane.dma().busy(), "reset: transfer is active before reset");
        const auto stale_handle = buffer.handle;
        plane.reset();
        check(!plane.dma().response_valid(), "reset: no stale response is delivered");
        check(plane.dma().accepted_count() == 1, "reset: old acceptance not replayed");
        check(
            plane.buffers().state(stale_handle.slot) == dp::BufferState::Free,
            "reset: buffer slots freed");
        check(
            !plane.buffers().begin_load(stale_handle, "dma", 0, 64, plane.cycle()).ok,
            "reset: old handle is rejected");
        dp::DmaResponse response;
        for (int index = 0; index < 5; ++index) {
            wait_step(plane);
            check(!plane.dma().response_valid(), "reset: no late response appears");
        }

        // A new transfer works and is byte-exact.
        auto fresh = plane.buffers().allocate("dma", 64, plane.cycle());
        check(fresh.ok, "reset: reallocate");
        dp::DmaRequest retry = load;
        retry.request_id = 72;
        retry.buffer = fresh.handle;
        check(run_dma(plane, retry, response), "reset: new transfer completes");
        check(response.status == cmd::StatusCode::Ok, "reset: new transfer status");
        check(
            read_storage(plane, fresh.handle, 64) == pattern(64, 0) ||
                std::all_of(
                    plane.buffers().bytes(fresh.handle.slot).begin(),
                    plane.buffers().bytes(fresh.handle.slot).begin() + 64,
                    [](std::uint8_t value) { return value == 0x77; }),
            "reset: new transfer loads expected bytes");
    }

    void test_no_lost_or_duplicated_transfers() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        artifact.image.assign(kTotalBytes, 0);
        std::vector<std::vector<std::uint8_t>> expected(5);
        for (int index = 0; index < 5; ++index) {
            expected[index] = pattern(16, static_cast<std::uint8_t>(index));
            std::copy(
                expected[index].begin(), expected[index].end(),
                artifact.image.begin() + front.base_addr + index * 16);
        }
        dp::DataPlane plane(std::move(artifact), make_config());

        for (int index = 0; index < 5; ++index) {
            auto buffer = plane.buffers().allocate("dma", 16, plane.cycle());
            check(buffer.ok, "no-loss: allocate");
            dp::DmaRequest load;
            load.request_id = static_cast<std::uint64_t>(100 + index);
            load.direction = dp::DmaDirection::MemoryToBuffer;
            load.ddr_region = front;
            load.ddr_offset = static_cast<std::uint64_t>(index) * 16;
            load.rows = 1;
            load.cols = 4;
            load.element_bytes = 4;
            load.buffer = buffer.handle;
            dp::DmaResponse response;
            check(run_dma(plane, load, response), "no-loss: transfer completes");
            check(response.status == cmd::StatusCode::Ok, "no-loss: status");
            check(
                read_storage(plane, buffer.handle, 16) == expected[index],
                "no-loss: bytes exact");
            check(
                plane.buffers().release(buffer.handle, "dma", plane.cycle()).ok,
                "no-loss: buffer released");
        }
        check(plane.dma().accepted_count() == 5, "no-loss: accepted count");
        check(plane.dma().completed_count() == 5, "no-loss: completed count");
        check(plane.dma().failed_count() == 0, "no-loss: failed count");
        check(plane.dma().read_bytes() == 80, "no-loss: total DMA read bytes");
        check(
            plane.memory_service().accepted_count() == 10,
            "no-loss: two memory beats per 16-byte transfer, none lost or repeated");
        check(
            plane.memory_service().read_bytes() == 80,
            "no-loss: memory read bytes match");
    }

    void test_illegal_request_no_partial_write() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const auto input = region_view(artifact.config, "rhs_data");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 256, 0x5C);
        dp::DataPlane plane(std::move(artifact), make_config());

        auto buffer = plane.buffers().allocate("dma", 32, plane.cycle());
        const auto storage_before = plane.buffers().bytes(buffer.handle.slot);

        dp::DmaRequest illegal;
        illegal.request_id = 81;
        illegal.direction = dp::DmaDirection::MemoryToBuffer;
        illegal.ddr_region = front;
        illegal.ddr_offset = front.byte_size - 8;
        illegal.rows = 2;
        illegal.cols = 2;
        illegal.element_bytes = 4;
        illegal.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, illegal, response), "illegal: terminal response");
        check(response.status == cmd::StatusCode::AddressFault, "illegal: AddressFault");
        check(response.bytes == 0, "illegal: no bytes moved");
        check(
            plane.buffers().state(buffer.handle.slot) == dp::BufferState::Reserved,
            "illegal: buffer state unchanged");
        check(
            plane.buffers().bytes(buffer.handle.slot) == storage_before,
            "illegal: buffer contents unchanged");

        const auto image_before = plane.memory().snapshot();
        dp::DmaRequest bad_writeback;
        bad_writeback.request_id = 82;
        bad_writeback.direction = dp::DmaDirection::BufferToMemory;
        bad_writeback.ddr_region = input;
        bad_writeback.rows = 1;
        bad_writeback.cols = 4;
        bad_writeback.element_bytes = 4;
        bad_writeback.buffer = buffer.handle;
        check(run_dma(plane, bad_writeback, response), "illegal: write to input answered");
        check(
            response.status == cmd::StatusCode::AddressFault,
            "illegal: write to read-only input is AddressFault");
        check(plane.memory().snapshot() == image_before, "illegal: DDR unchanged");
    }

    void test_zero_size_transfer() {
        auto plane = dp::DataPlane(make_artifact(), make_config());
        auto buffer = plane.buffers().allocate("dma", 8, plane.cycle());
        check(buffer.ok, "zero: allocate");
        const auto reads_before = plane.memory_service().read_bytes();
        dp::DmaRequest request;
        request.request_id = 91;
        request.direction = dp::DmaDirection::MemoryToBuffer;
        request.ddr_region = region_view(plane.boot(), "node.0.front");
        request.rows = 1;
        request.cols = 0;
        request.element_bytes = 4;
        request.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, request, response), "zero: response");
        check(response.status == cmd::StatusCode::Ok, "zero: success no-op");
        check(response.bytes == 0, "zero: no bytes");
        check(response.first_beat_cycle == response.accept_cycle, "zero: no beats");
        check(
            plane.buffers().state(buffer.handle.slot) == dp::BufferState::Ready &&
                plane.buffers().valid_bytes(buffer.handle.slot) == 0,
            "zero: empty valid range and READY state");
        check(
            plane.memory_service().read_bytes() == reads_before,
            "zero: no memory traffic");
    }

    void test_unrelated_regions_unchanged() {
        auto artifact = make_artifact();
        const auto front = region_view(artifact.config, "node.0.front");
        const auto update = region_view(artifact.config, "node.0.update");
        const auto factor_u = region_view(artifact.config, "node.0.factor_u");
        artifact.image.assign(kTotalBytes, 0);
        std::fill(
            artifact.image.begin() + front.base_addr,
            artifact.image.begin() + front.base_addr + 32, 0x11);
        std::fill(
            artifact.image.begin() + factor_u.base_addr,
            artifact.image.begin() + factor_u.base_addr + factor_u.byte_size, 0xEE);
        std::fill(
            artifact.image.begin() + 0, artifact.image.begin() + 64, 0xCD);  // command region
        dp::DataPlane plane(std::move(artifact), make_config());
        const auto before = plane.memory().snapshot();

        auto buffer = plane.buffers().allocate("dma", 32, plane.cycle());
        dp::DmaRequest load;
        load.request_id = 95;
        load.direction = dp::DmaDirection::MemoryToBuffer;
        load.ddr_region = front;
        load.rows = 1;
        load.cols = 8;
        load.element_bytes = 4;
        load.buffer = buffer.handle;
        dp::DmaResponse response;
        check(run_dma(plane, load, response), "guards: load");
        dp::DmaRequest writeback = load;
        writeback.request_id = 96;
        writeback.direction = dp::DmaDirection::BufferToMemory;
        writeback.ddr_region = update;
        check(run_dma(plane, writeback, response), "guards: writeback");

        const auto after = plane.memory().snapshot();
        bool only_target_changed = true;
        for (std::size_t index = 0; index < after.size(); ++index) {
            const bool in_target = index >= update.base_addr &&
                                   index < update.base_addr + 32;
            if (!in_target && after[index] != before[index]) {
                only_target_changed = false;
                break;
            }
        }
        check(only_target_changed, "guards: only the target bytes changed");
        check(
            std::equal(
                before.begin() + factor_u.base_addr,
                before.begin() + factor_u.base_addr + factor_u.byte_size,
                after.begin() + factor_u.base_addr),
            "guards: unrelated factor region unchanged");
        check(
            std::equal(before.begin(), before.begin() + 64, after.begin()),
            "guards: command region unchanged");
    }
};

}  // namespace

int sc_main(int, char**) {
    sc_core::sc_clock clock("clk", sc_core::sc_time(1, sc_core::SC_NS));
    T04SystemCTests tests("t04_systemc_tests");
    tests.clk(clock);
    sc_core::sc_start();
    return tests.failures() == 0 ? 0 : 1;
}
