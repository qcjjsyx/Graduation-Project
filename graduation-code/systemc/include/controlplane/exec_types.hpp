#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/buffer_manager.hpp"
#include "dataplane/descriptor_reader.hpp"
#include "dataplane/memory_service.hpp"

namespace hw::controlplane {

// Execution backend selection. Real numerical kernels are not part of T05;
// RequireRealKernels refuses to run instead of silently using the test stub.
enum class ExecBackend {
    TestStub,
    RequireRealKernels,
};

const char* exec_backend_name(ExecBackend backend);

enum class UnitKind {
    Assemble,
    Panel,
    Trsm,
    Gemm,
    Solve,
    Factor,  // real packed-LU -> persistent L/U conversion (STORE_FACTOR)
};

const char* unit_kind_name(UnitKind kind);

// Deterministic test-data transform applied by the stub to its buffer range.
enum class StubTransform {
    None,    // leave data untouched (byte-predictable data path)
    Pattern, // fill the declared range with command_id & 0xFF
};

struct StubConfig {
    // Fixed per-opcode latency in cycles (>= 1).
    std::map<hw::command::Opcode, std::uint32_t> latency_by_opcode;
    std::uint32_t default_latency{5};
    StubTransform transform{StubTransform::None};
    std::uint32_t pattern_bytes{64};
    // Extra cycles the unit holds its response stable before the consumer
    // takes it (models response backpressure).
    std::uint32_t response_hold_cycles{0};
};

struct ExecFailure {
    hw::command::StatusCode status{hw::command::StatusCode::StaticPivotFailure};
    std::uint32_t pivot_count{0};
    std::uint32_t overflow_count{0};
};

struct ExecRequest {
    std::uint64_t txn_id{0};
    std::uint64_t epoch{0};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    hw::command::Opcode opcode{hw::command::Opcode::NodeBegin};
    std::uint32_t unit_id{0};

    // Typed descriptor parameters (filled per opcode).
    hw::dataplane::FrontView front;
    hw::dataplane::FactorView factor;
    hw::dataplane::ContributionView contribution;
    hw::dataplane::KernelView kernel;
    hw::dataplane::SolveView solve;

    // Buffer operands and layout. Leases are acquired atomically by the
    // scheduler at dispatch time and released by the device when the command
    // finishes, so a unit never waits for buffer resources while holding a
    // unit slot.
    hw::dataplane::BufferHandle front_slot{};
    hw::dataplane::BufferHandle temp_slot{};
    hw::dataplane::BufferLease front_lease{};
    hw::dataplane::BufferLease temp_lease{};
    bool has_front_lease{false};
    bool has_temp_lease{false};
    std::uint32_t total_dim{0};
    std::uint32_t update_dim{0};
    std::uint64_t front_offset{0};
    std::uint64_t temp_offset{0};
    std::uint64_t range_bytes{0};

    hw::command::PivotPolicy policy{hw::command::PivotPolicy::Static};
    std::uint32_t threshold_bits{0};
};

struct ExecResponse {
    std::uint64_t txn_id{0};
    std::uint64_t epoch{0};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::uint32_t unit_id{0};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    std::uint32_t pivot_count{0};
    std::uint32_t overflow_count{0};
    std::uint64_t bytes_read{0};
    std::uint64_t bytes_written{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    bool needs_pvector_write{false};  // PANEL_LU success under STATIC
    std::uint32_t pvector_dim{0};
};

// Replaceable execution unit. The test stub is only used when explicitly
// enabled; formal runs without kernels must refuse instead of faking success.
class IExecutionUnit {
public:
    virtual ~IExecutionUnit() = default;
    virtual UnitKind kind() const = 0;
    virtual std::uint32_t unit_id() const = 0;
    // Returns false when the unit is busy (resource stall, not an error).
    virtual bool submit(const ExecRequest& request, std::uint64_t cycle) = 0;
    virtual void tick(std::uint64_t cycle) = 0;
    // Takes exactly one terminal response per accepted transaction.
    virtual bool poll_response(ExecResponse& response) = 0;
    virtual bool busy() const = 0;
    // Directed cancellation of one accepted transaction (node abort). After it
    // returns, the unit performs no further buffer access for that transaction;
    // already-completed writes are not rolled back. Exactly one terminal
    // response is still produced and consumed by the device.
    virtual void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) = 0;
    // Drops the active transaction of the given epoch; late responses vanish.
    // Only for global reset-style semantics; directed aborts use abort_txn.
    virtual void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) = 0;
    // Memory responses for writes this unit issued (default: not a client).
    virtual void deliver_memory_response(const hw::dataplane::MemoryResponse& response) {
        (void)response;
    }
    // True while the given memory request id belongs to an in-flight write of
    // this unit. Used for precise response routing instead of broadcasting.
    virtual bool handles_request(std::uint64_t request_id) const {
        (void)request_id;
        return false;
    }
    // Number of in-flight memory requests owned by this unit (diagnostics).
    virtual std::size_t pending_request_count() const { return 0; }
    // Command id of the currently active transaction (NONE when idle).
    virtual std::uint32_t active_command_id() const { return hw::command::NONE; }
    virtual void reset() = 0;
};

}  // namespace hw::controlplane
