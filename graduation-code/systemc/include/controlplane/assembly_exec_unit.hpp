#pragma once

#include <optional>

#include "controlplane/exec_types.hpp"
#include "dataplane/on_chip_arbiter.hpp"

namespace hw::controlplane {

struct AssemblyConfig {
    std::uint32_t add_latency_cycles{1};
    std::uint32_t response_hold_cycles{0};
};

// Validate the runtime contribution before even starting its DDR load.
std::string validate_contribution(
    const hw::dataplane::ContributionView& contribution,
    const hw::dataplane::FrontView& front,
    std::uint32_t node_id);

// One scalar FP32 extend-add lane per unit, with 12 bytes of numeric staging.
// No full front/update copy and no PE pool: assembly is scratchpad read/add/write.
class AssemblyExecutionUnit : public IExecutionUnit {
public:
    AssemblyExecutionUnit(std::uint32_t unit_id,
                          hw::dataplane::BufferManager* buffers,
                          hw::dataplane::OnChipArbiter* arbiter,
                          AssemblyConfig config = {});
    UnitKind kind() const override { return UnitKind::Assemble; }
    std::uint32_t unit_id() const override { return unit_id_; }
    bool submit(const ExecRequest& request, std::uint64_t cycle) override;
    void tick(std::uint64_t cycle) override;
    bool poll_response(ExecResponse& response) override;
    bool busy() const override { return active_.has_value() || response_.has_value(); }
    void abort_txn(std::uint64_t txn_id, std::uint64_t cycle) override;
    void abort_epoch(std::uint64_t epoch, std::uint64_t cycle) override;
    std::uint32_t active_command_id() const override;
    void reset() override;

private:
    enum class Phase { ReadChild, ReadParent, Add, WriteParent };
    void finish(hw::command::StatusCode status, const std::string& reason,
                std::uint64_t cycle);
    void clear_pending();
    std::uint64_t source_offset() const;
    std::uint64_t target_offset() const;

    std::uint32_t unit_id_;
    hw::dataplane::BufferManager* buffers_;
    hw::dataplane::OnChipArbiter* arbiter_;
    AssemblyConfig config_;
    std::optional<ExecRequest> active_;
    std::optional<ExecResponse> response_;
    ExecResponse progress_;
    Phase phase_{Phase::ReadChild};
    std::uint32_t row_{0}, col_{0};
    float child_{0}, parent_{0}, result_{0};
    std::uint64_t add_ready_{0}, response_ready_{0}, cycle_{0};
};

}  // namespace hw::controlplane
