#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "controlplane/assembly_exec_unit.hpp"

namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace cmd = hw::command;
int checks = 0, failures = 0;
void expect(bool ok, const char* message) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
std::vector<std::uint8_t> bytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> result(values.size() * 4);
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}
struct Harness {
    dp::BufferManager buffers{dp::BufferConfig{2, 1024, 4, 1}};
    dp::OnChipArbiter arbiter;
    cp::AssemblyExecutionUnit unit{0, &buffers, &arbiter, cp::AssemblyConfig{1, 3}};
    cp::ExecRequest request;
    Harness(std::vector<float> parent = {1,2,3,4,5,6,7,8,9},
            std::vector<float> child = {1,2,3,4}) {
        arbiter.configure(4, 4);
        request.opcode = cmd::Opcode::AssembleExtendAdd;
        request.txn_id = 7; request.epoch = 1; request.command_id = 3; request.node_id = 1;
        request.total_dim = 3; request.update_dim = 2;
        request.front_offset = 4; request.temp_offset = 8;
        request.front_slot = buffers.allocate("parent", 40, 0).handle;
        request.temp_slot = buffers.allocate("child", 24, 0).handle;
        request.front_lease = buffers.begin_write(request.front_slot, "parent", 0, 40, 0).lease;
        auto temp = buffers.begin_write(request.temp_slot, "child", 0, 24, 0).lease;
        expect(buffers.lease_write(request.front_lease.lease_id, 4, bytes(parent), 1) == cmd::StatusCode::Ok,
               "parent initialized");
        expect(buffers.lease_write(temp.lease_id, 8, bytes(child), 1) == cmd::StatusCode::Ok,
               "child initialized");
        buffers.end_write(temp.lease_id, 24, 1);
        request.temp_lease = buffers.begin_read(request.temp_slot, "child", 0, 24, 1).lease;
        request.has_front_lease = request.has_temp_lease = true;
        request.front.total_dim = 3;
        request.front.front.descriptor_id = 10;
        request.front.front.rows = request.front.front.cols = 3;
        request.front.front.row_stride = 12; request.front.front.byte_size = 36;
        auto& c = request.contribution;
        c.parent_id = 1; c.child_id = 0; c.row_count = c.col_count = 2;
        c.row_map = {1,0}; c.col_map = {2,0}; c.source.descriptor_id = 11;
        c.source.rows = c.source.cols = 2; c.source.row_stride = c.source_stride = 8;
        c.source.byte_size = 16; c.target = request.front.front; c.target_stride = 12;
    }
    std::vector<float> parent() const {
        std::vector<float> result(9);
        std::memcpy(result.data(), buffers.bytes(request.front_slot.slot).data() + 4, 36);
        return result;
    }
    cp::ExecResponse run(bool contention = false, std::uint64_t start = 2) {
        expect(unit.submit(request, start), "request accepted");
        cp::ExecResponse response;
        for (std::uint64_t cycle = start; cycle < start + 300; ++cycle) {
            arbiter.begin_cycle(cycle);
            if (contention) {
                arbiter.acquire(dp::OnChipDomain::Read, dp::OnChipClientKind::Other, 9, 4);
                arbiter.acquire(dp::OnChipDomain::Write, dp::OnChipClientKind::Other, 9, 4);
            }
            unit.tick(cycle);
            if (unit.poll_response(response)) return response;
        }
        response.status = cmd::StatusCode::Timeout;
        return response;
    }
};
int main() {
    for (bool contention : {false, true}) {
        Harness h;
        const auto before = h.buffers.total_lease_access_bytes();
        const auto r = h.run(contention);
        expect(r.status == cmd::StatusCode::Ok, "mapped extend-add succeeds under arbitration");
        expect(h.parent() == std::vector<float>({2,2,5,4,5,6,10,8,13}),
               "source permutation, target map, offsets and addition are honored");
        expect(r.bytes_read == 32 && r.bytes_written == 16 &&
               h.buffers.total_lease_access_bytes() - before == 48,
               "actual scratchpad accesses equal modeled byte counters");
        expect(h.arbiter.stats(dp::OnChipDomain::Read, dp::OnChipClientKind::Assemble, 0).bytes == 32 &&
               h.arbiter.stats(dp::OnChipDomain::Write, dp::OnChipClientKind::Assemble, 0).bytes == 16,
               "reads/writes pass through the shared arbiter");
        expect(r.finish_cycle - r.start_cycle >= 15, "latency comes from per-element phases");
        expect(!h.unit.busy(), "response drains the unit");
    }
    for (unsigned invalid = 0; invalid < 8; ++invalid) {
        Harness h;
        auto& c = h.request.contribution;
        switch (invalid) {
            case 0: c.target.descriptor_id = 99; break;
            case 1: c.target.base_addr = 64; break;
            case 2: c.parent_id = 4; break;
            case 3: c.row_map = {0,0}; break;
            case 4: c.col_map = {0,3}; break;
            case 5: c.source_stride = 4; break;
            case 6: c.col_count = 1; break;
            case 7: h.request.front_lease.write = false; break;
        }
        const auto original = h.parent();
        const auto r = h.run();
        expect(r.status == cmd::StatusCode::BadDescriptor && r.bytes_read == 0 &&
               r.bytes_written == 0 && h.parent() == original,
               "invalid identity/map/stride/lease rejected before any access");
    }
    for (unsigned kind = 0; kind < 3; ++kind) {
        std::vector<float> parent(9, 1), child(4, 1);
        if (kind == 0) child[3] = std::numeric_limits<float>::infinity();
        if (kind == 1) parent[8] = std::numeric_limits<float>::quiet_NaN();
        if (kind == 2) parent[8] = child[3] = std::numeric_limits<float>::max();
        Harness h(parent, child);
        const auto r = h.run();
        expect(r.status == cmd::StatusCode::NonFiniteValue && r.bytes_written == 0,
               "non-finite operand and FP32 addition overflow are explicit failures");
    }
    for (const auto abort_cycle : {4u, 5u}) {
        Harness h;
        const auto original = h.parent();
        h.unit.submit(h.request, 2);
        for (std::uint64_t cycle = 2; cycle <= abort_cycle; ++cycle) {
            h.arbiter.begin_cycle(cycle); h.unit.tick(cycle);
        }
        // Abort either before first write or immediately after it.
        h.unit.abort_txn(h.request.txn_id, abort_cycle);
        cp::ExecResponse response;
        for (std::uint64_t cycle = abort_cycle + 1; cycle < 20; ++cycle) h.unit.tick(cycle);
        expect(h.unit.poll_response(response) && response.status == cmd::StatusCode::Aborted,
               "directed abort produces one terminal response");
        expect((h.parent() == original) == (abort_cycle == 4),
               "abort blocks future writes, earlier writes need not roll back");
        expect(h.arbiter.pending(dp::OnChipDomain::Read) == 0 &&
               h.arbiter.pending(dp::OnChipDomain::Write) == 0,
               "abort clears pending bandwidth requests");
        h.request.txn_id = 42;
        expect(h.run(false, 100).status == cmd::StatusCode::Ok, "unit and leases reusable after abort");
    }
    {
        Harness h;
        h.unit.submit(h.request, 2);
        h.arbiter.begin_cycle(2); h.unit.tick(2);
        h.unit.abort_epoch(1, 2);
        cp::ExecResponse response;
        h.unit.tick(100);
        expect(!h.unit.busy() && !h.unit.poll_response(response), "reset drops old epoch response");
        expect(h.run(false, 101).status == cmd::StatusCode::Ok, "unit reusable after epoch reset");
    }
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
