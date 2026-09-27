#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/artifact_loader.hpp"
#include "dataplane/memory_service.hpp"
#include "dataplane/region_view.hpp"
#include "dataplane/trace.hpp"

namespace hw::dataplane {

enum class ReadKind {
    Region,
    Front,
    Factor,
    Contribution,
    Kernel,
    Solve,
    Dependency,
};

const char* read_kind_name(ReadKind kind);

struct ContributionView {
    std::uint32_t descriptor_id{0};
    RegionView source;
    RegionView target;
    std::uint32_t child_id{0};
    std::uint32_t parent_id{0};
    std::uint32_t row_count{0};
    std::uint32_t col_count{0};
    std::uint64_t source_stride{0};
    std::uint64_t target_stride{0};
    hw::command::DataFormat format{hw::command::DataFormat::Fp32};
    std::vector<std::uint32_t> row_map;
    std::vector<std::uint32_t> col_map;
};

struct FactorView {
    std::uint32_t descriptor_id{0};
    RegionView l;
    RegionView u;
    RegionView p;
    RegionView workspace;
    bool has_workspace{false};
    std::uint32_t total_dim{0};
    std::uint32_t pivot_dim{0};
    hw::command::DataFormat format{hw::command::DataFormat::Fp32};
};

struct FrontView {
    std::uint32_t descriptor_id{0};
    RegionView front;
    RegionView p;
    RegionView workspace;
    std::uint32_t total_dim{0};
    std::uint32_t pivot_dim{0};
    std::uint32_t tile_size{0};
    std::uint32_t contribution_descriptor_id{hw::command::NONE};
    bool has_contribution{false};
    std::uint32_t factor_descriptor_id{0};
    hw::command::PivotPolicy policy{hw::command::PivotPolicy::Static};
    std::uint32_t threshold_bits{0};
    float threshold{0.0f};
};

struct KernelView {
    std::uint32_t descriptor_id{0};
    RegionView a;
    RegionView b;
    RegionView c;
    std::uint32_t m{0};
    std::uint32_t n{0};
    std::uint32_t k{0};
    hw::command::KernelBackend backend{
        hw::command::KernelBackend::SystemCFp32DeviceModel};
    hw::command::DataFormat format{hw::command::DataFormat::Fp32};
    std::uint32_t tile_size{0};
    std::uint32_t scale_descriptor_id{hw::command::NONE};
    bool has_scale{false};
};

struct SolveView {
    std::uint32_t descriptor_id{0};
    FactorView factor;
    RegionView rhs;
    RegionView solution;
    RegionView p;
    RegionView workspace;
    std::uint32_t rhs_count{0};
    hw::command::SolveDirection direction{hw::command::SolveDirection::Forward};
    hw::command::DataFormat format{hw::command::DataFormat::Fp32};
};

struct DependencyView {
    std::uint32_t descriptor_id{0};
    std::vector<std::uint32_t> tokens;
};

struct ReadOutcome {
    std::uint64_t request_id{0};
    ReadKind kind{ReadKind::Region};
    hw::command::StatusCode status{hw::command::StatusCode::Ok};
    std::string reason;
    std::uint64_t issue_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint64_t read_bytes{0};
    std::uint32_t memory_reads{0};
    hw::command::Descriptor descriptor{};
    RegionView region;
    FrontView front;
    FactorView factor;
    ContributionView contribution;
    KernelView kernel;
    SolveView solve;
    DependencyView dependency;
};

// Runtime descriptor/payload reads through MemoryService. No descriptor cache:
// every read is a device transaction and is counted in the trace.
class DescriptorReader {
public:
    DescriptorReader(
        MemoryService& memory,
        const DeviceBootConfig& boot,
        TraceRecorder* trace = nullptr);

    std::uint64_t request(
        ReadKind kind,
        std::uint32_t descriptor_id,
        std::uint64_t cycle);
    bool poll(std::uint64_t request_id, ReadOutcome& outcome);
    bool pending(std::uint64_t request_id) const;
    void tick(std::uint64_t cycle);
    void deliver_memory_response(const MemoryResponse& response);
    void reset();

    std::uint64_t completed_count() const { return completed_count_; }
    std::uint64_t error_count() const { return error_count_; }
    std::uint64_t read_bytes() const { return read_bytes_; }
    std::uint64_t memory_reads() const { return memory_reads_; }
    const DeviceBootConfig& boot() const { return boot_; }

private:
    enum class Stage { Record, References, Payload, Done };

    struct Ref {
        std::uint32_t id{hw::command::NONE};
        bool is_region{false};
        hw::command::Descriptor descriptor{};
        RegionView region;

        Ref() = default;
        explicit Ref(std::uint32_t descriptor_id) : id(descriptor_id) {}
    };

    struct Task {
        std::uint64_t request_id{0};
        ReadKind kind{ReadKind::Region};
        std::uint32_t descriptor_id{0};
        std::uint64_t issue_cycle{0};
        std::uint64_t read_bytes{0};
        std::uint32_t memory_reads{0};
        Stage stage{Stage::Record};
        bool waiting_submit{false};
        bool payload_pending{false};
        std::uint64_t pending_memory_id{0};
        bool waiting_memory{false};
        std::uint64_t payload_address{0};
        std::uint64_t payload_length{0};
        std::vector<std::uint8_t> payload;
        hw::command::Descriptor record{};
        std::vector<Ref> refs;
        std::size_t next_ref{0};
    };

    Task* find_task(std::uint64_t request_id);
    void submit_descriptor_read(Task& task, std::uint32_t descriptor_id, std::uint64_t cycle);
    void submit_payload_read(Task& task, std::uint64_t cycle);
    void plan_references(Task& task);
    void on_record(Task& task, const MemoryResponse& response, std::uint64_t cycle);
    void on_reference(Task& task, const MemoryResponse& response, std::uint64_t cycle);
    void on_payload(Task& task, const MemoryResponse& response, std::uint64_t cycle);
    void finish(Task& task, hw::command::StatusCode status, std::string reason, std::uint64_t cycle);
    bool finalize(Task& task, ReadOutcome& outcome, std::string* reason);
    bool build_region_view(
        std::uint32_t descriptor_id,
        const hw::command::Descriptor& descriptor,
        RegionView& out,
        std::string* reason) const;
    void trace_read(const MemoryResponse& response) const;

    MemoryService& memory_;
    DeviceBootConfig boot_;
    TraceRecorder* trace_{nullptr};
    std::unordered_map<std::uint64_t, Task> tasks_;
    std::unordered_map<std::uint64_t, std::uint64_t> memory_to_task_;
    std::deque<ReadOutcome> outcomes_;
    std::uint64_t next_request_id_{1};
    std::uint64_t completed_count_{0};
    std::uint64_t error_count_{0};
    std::uint64_t read_bytes_{0};
    std::uint64_t memory_reads_{0};
};

}  // namespace hw::dataplane
