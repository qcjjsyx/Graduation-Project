#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "controlplane/exec_types.hpp"
#include "dataplane/buffer_manager.hpp"

namespace hw::controlplane {

// Explicitly enabled test execution unit. It models dispatch/latency/busy/
// response behaviour and a checkable test-data transform; it never performs
// real Panel-LU/TRSM/GEMM/solve arithmetic and its outputs must be reported as
// execution_backend=test_stub, numerical_validation=not_performed.
class TestStubUnit : public IExecutionUnit {
public:
    TestStubUnit(
        UnitKind kind,
        std::uint32_t unit_id,
        StubConfig config,
        hw::dataplane::BufferManager* buffers = nullptr);

    UnitKind kind() const override { return kind_; }
    std::uint32_t unit_id() const override { return unit_id_; }

    bool submit(const ExecRequest& request, std::uint64_t cycle) override;
    void tick(std::uint64_t cycle) override;
    bool poll_response(ExecResponse& response) override;
    bool busy() const override;
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) override;
    void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) override;
    void reset() override;

    // Error injection: command_id -> terminal failure (status + failure
    // column). Injected failures still run their latency and never write data.
    void inject_failure(std::uint32_t command_id, ExecFailure failure);
    void clear_injections();

    std::uint32_t active_command_id() const override;
    std::uint64_t busy_start() const { return busy_start_; }
    std::uint32_t accepted_count() const { return accepted_count_; }

private:
    void finish_execution(std::uint64_t cycle);
    bool apply_transform(std::uint64_t cycle, std::string* reason);
    bool assemble_copy(std::uint64_t cycle, std::string* reason);

    UnitKind kind_;
    std::uint32_t unit_id_;
    StubConfig config_;
    hw::dataplane::BufferManager* buffers_;
    std::map<std::uint32_t, ExecFailure> failures_;

    std::optional<ExecRequest> active_;
    std::optional<ExecResponse> response_;
    bool response_visible_{false};
    std::uint64_t response_ready_cycle_{0};
    std::uint64_t busy_start_{0};
    std::uint64_t ready_cycle_{0};
    std::uint32_t accepted_count_{0};
};

}  // namespace hw::controlplane
