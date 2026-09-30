#pragma once

// Synthetic schema-v2 artifact builder for control-plane tests. Produces a
// real manifest + memory image (the same binary contract the software
// compiler emits) with deterministic, independently computable data patterns
// so tests can verify byte outcomes without golden files.

#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "command_codec.hpp"
#include "dataplane/artifact_loader.hpp"

namespace hw::testing {

namespace cmd = hw::command;

struct TestNode {
    std::uint32_t node_id{0};
    std::int32_t parent_id{-1};
    std::uint32_t total_dim{2};
    std::uint32_t pivot_dim{1};
    // child node id -> col_map (length == child's update_dim).
    std::map<std::uint32_t, std::vector<std::uint32_t>> contributions;
    // Optional explicit row-major FP32 front values (total_dim * total_dim).
    // When set, the front region uses these values instead of the byte
    // pattern; used by real-numeric T07 tests.
    std::vector<float> front_values;

    TestNode() = default;
    TestNode(
        std::uint32_t id,
        std::int32_t parent,
        std::uint32_t total,
        std::uint32_t pivot,
        std::map<std::uint32_t, std::vector<std::uint32_t>> child_maps)
        : node_id(id),
          parent_id(parent),
          total_dim(total),
          pivot_dim(pivot),
          contributions(std::move(child_maps)) {}
};

struct TestArtifactOptions {
    cmd::PivotPolicy policy{cmd::PivotPolicy::Static};
    std::uint32_t threshold_bits{0};
    std::uint32_t matrix_dim{8};
    std::uint32_t tile_size{16};
    // Trailing SOLVE commands are skipped when false (real-backend numeric
    // tests of one panel chain do not need solve units).
    bool include_solve{true};
};

struct TestArtifact {
    hw::dataplane::LoadedArtifact artifact;
    std::vector<cmd::Command> commands;
    std::vector<cmd::Descriptor> descriptors;
    std::vector<TestNode> nodes;
    TestArtifactOptions options;

    std::uint64_t region_addr(const std::string& name) const {
        return artifact.config.region(name)->offset;
    }
    std::uint64_t region_size(const std::string& name) const {
        return artifact.config.region(name)->size;
    }
    const TestNode& node(std::uint32_t node_id) const { return nodes.at(node_id); }

    // Independent byte expectations for the stub data path: the only data
    // transformation before STORE_FACTOR/STORE_UPDATE write-back is the
    // assemble scatter-copy parent[col_map[i], col_map[j]] = child[i][j].
    std::vector<std::uint8_t> expected_l(std::uint32_t node_id) const;
    std::vector<std::uint8_t> expected_u(std::uint32_t node_id) const;
    std::vector<std::uint8_t> expected_update(std::uint32_t node_id) const;
    std::vector<std::uint8_t> expected_pvector(std::uint32_t pivot_dim) const;
    std::vector<std::uint8_t> expected_solution() const;
};

namespace detail {

inline std::uint8_t front_byte(std::uint32_t node, std::uint32_t row, std::uint32_t col) {
    return static_cast<std::uint8_t>((node * 31u + row * 7u + col * 13u + 3u) & 0xFFu);
}

inline std::uint8_t update_byte(std::uint32_t node, std::uint32_t row, std::uint32_t col) {
    return static_cast<std::uint8_t>((node * 17u + row * 5u + col * 11u + 91u) & 0xFFu);
}

inline std::uint8_t rhs_byte(std::uint32_t index) {
    return static_cast<std::uint8_t>((index * 19u + 41u) & 0xFFu);
}

inline void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        out.push_back(static_cast<std::uint8_t>((value >> (8 * index)) & 0xFF));
    }
}

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline std::string node_prefix(const std::string& name) {
    const auto first = name.find('.');
    const auto second = name.find('.', first == std::string::npos ? 0 : first + 1);
    return second == std::string::npos ? name : name.substr(0, second);
}

struct RegionSpec {
    std::string name;
    std::uint32_t rows;
    std::uint32_t cols;
    cmd::DataFormat format;
    std::vector<std::uint8_t> data;
};

struct DescriptorSpec {
    std::string name;
    cmd::DescriptorType type;
    std::vector<std::uint32_t> body;
    std::vector<std::uint8_t> payload;
};

struct CommandSpec {
    cmd::Opcode opcode;
    std::uint32_t node_id;
    std::string descriptor;
    std::vector<std::pair<std::uint32_t, std::string>> waits;
    std::string token;
};

}  // namespace detail

inline void set_element(
    std::vector<std::uint8_t>& out,
    std::size_t element_index,
    std::uint8_t pattern) {
    out[element_index * 4 + 0] = pattern;
    out[element_index * 4 + 1] = 0;
    out[element_index * 4 + 2] = 0;
    out[element_index * 4 + 3] = 0;
}

inline std::vector<std::uint8_t> TestArtifact::expected_l(std::uint32_t node_id) const {
    const auto& node = nodes.at(node_id);
    const std::uint32_t total = node.total_dim;
    const std::uint32_t pivot = node.pivot_dim;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(total) * pivot * 4, 0);
    for (std::uint32_t row = 0; row < total; ++row) {
        for (std::uint32_t col = 0; col < pivot; ++col) {
            set_element(
                out, static_cast<std::size_t>(row) * pivot + col,
                detail::front_byte(node_id, row, col));
        }
    }
    for (const auto& entry : node.contributions) {
        const auto& col_map = entry.second;
        for (std::size_t i = 0; i < col_map.size(); ++i) {
            for (std::size_t j = 0; j < col_map.size(); ++j) {
                if (col_map[j] < pivot) {
                    // The assembled value is the child's STORE_UPDATE output:
                    // its front update block element (pivot + i, pivot + j).
                    const std::uint32_t child_pivot = nodes.at(entry.first).pivot_dim;
                    set_element(
                        out, static_cast<std::size_t>(col_map[i]) * pivot + col_map[j],
                        detail::front_byte(
                            entry.first, child_pivot + static_cast<std::uint32_t>(i),
                            child_pivot + static_cast<std::uint32_t>(j)));
                }
            }
        }
    }
    return out;
}

inline std::vector<std::uint8_t> TestArtifact::expected_u(std::uint32_t node_id) const {
    const auto& node = nodes.at(node_id);
    const std::uint32_t total = node.total_dim;
    const std::uint32_t pivot = node.pivot_dim;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(pivot) * total * 4, 0);
    for (std::uint32_t row = 0; row < pivot; ++row) {
        for (std::uint32_t col = 0; col < total; ++col) {
            set_element(
                out, static_cast<std::size_t>(row) * total + col,
                detail::front_byte(node_id, row, col));
        }
    }
    for (const auto& entry : node.contributions) {
        const auto& col_map = entry.second;
        for (std::size_t i = 0; i < col_map.size(); ++i) {
            for (std::size_t j = 0; j < col_map.size(); ++j) {
                if (col_map[i] < pivot) {
                    const std::uint32_t child_pivot = nodes.at(entry.first).pivot_dim;
                    set_element(
                        out, static_cast<std::size_t>(col_map[i]) * total + col_map[j],
                        detail::front_byte(
                            entry.first, child_pivot + static_cast<std::uint32_t>(i),
                            child_pivot + static_cast<std::uint32_t>(j)));
                }
            }
        }
    }
    return out;
}

inline std::vector<std::uint8_t> TestArtifact::expected_update(std::uint32_t node_id) const {
    const auto& node = nodes.at(node_id);
    const std::uint32_t pivot = node.pivot_dim;
    const std::uint32_t update = node.total_dim - pivot;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(update) * update * 4, 0);
    for (std::uint32_t row = 0; row < update; ++row) {
        for (std::uint32_t col = 0; col < update; ++col) {
            set_element(
                out, static_cast<std::size_t>(row) * update + col,
                detail::front_byte(node_id, pivot + row, pivot + col));
        }
    }
    for (const auto& entry : node.contributions) {
        const auto& col_map = entry.second;
        for (std::size_t i = 0; i < col_map.size(); ++i) {
            for (std::size_t j = 0; j < col_map.size(); ++j) {
                if (col_map[i] >= pivot && col_map[j] >= pivot) {
                    const std::uint32_t child_pivot = nodes.at(entry.first).pivot_dim;
                    set_element(
                        out,
                        static_cast<std::size_t>(col_map[i] - pivot) * update +
                            (col_map[j] - pivot),
                        detail::front_byte(
                            entry.first, child_pivot + static_cast<std::uint32_t>(i),
                            child_pivot + static_cast<std::uint32_t>(j)));
                }
            }
        }
    }
    return out;
}

inline std::vector<std::uint8_t> TestArtifact::expected_pvector(std::uint32_t pivot_dim) const {
    std::vector<std::uint8_t> out;
    for (std::uint32_t index = 0; index < pivot_dim; ++index) {
        detail::append_u32(out, index);
    }
    return out;
}

inline std::vector<std::uint8_t> TestArtifact::expected_solution() const {
    // SOLVE must traverse the complete RHS; a prefix-only copy is a bug.
    std::vector<std::uint8_t> out(static_cast<std::size_t>(options.matrix_dim) * 4, 0);
    for (std::uint32_t index = 0; index < options.matrix_dim; ++index) {
        out[static_cast<std::size_t>(index) * 4] = detail::rhs_byte(index);
    }
    return out;
}

inline TestArtifact build_test_artifact(
    std::vector<TestNode> nodes,
    TestArtifactOptions options = {}) {
    using detail::CommandSpec;
    using detail::DescriptorSpec;
    using detail::RegionSpec;

    if (nodes.empty()) {
        throw std::runtime_error("need at least one node");
    }
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (nodes[index].node_id != index) {
            throw std::runtime_error("node ids must be dense 0..N-1 in order");
        }
    }

    TestArtifact result;
    result.nodes = nodes;
    result.options = options;

    // ---- region specs ----------------------------------------------------
    std::vector<RegionSpec> regions;
    {
        std::vector<std::uint8_t> data;
        for (std::uint32_t index = 0; index < options.matrix_dim; ++index) {
            detail::append_u32(data, index);
        }
        regions.push_back(
            {"permutation_data", 1, options.matrix_dim, cmd::DataFormat::Int32, data});
    }
    {
        std::vector<std::uint8_t> data;
        for (std::uint32_t index = 0; index < options.matrix_dim; ++index) {
            for (unsigned b = 0; b < 4; ++b) {
                data.push_back(b == 0 ? detail::rhs_byte(index) : 0);
            }
        }
        regions.push_back(
            {"rhs_data", 1, options.matrix_dim, cmd::DataFormat::Fp32, data});
    }
    regions.push_back(
        {"solution_data", 1, options.matrix_dim, cmd::DataFormat::Fp32,
         std::vector<std::uint8_t>(options.matrix_dim * 4, 0)});
    for (const auto& node : nodes) {
        const std::uint32_t total = node.total_dim;
        const std::uint32_t pivot = node.pivot_dim;
        const std::uint32_t update = total - pivot;
        const std::string prefix = "node." + std::to_string(node.node_id);
        RegionSpec front{prefix + ".front", total, total, cmd::DataFormat::Fp32, {}};
        if (node.front_values.size() == static_cast<std::size_t>(total) * total) {
            front.data.resize(static_cast<std::size_t>(total) * total * 4);
            std::memcpy(
                front.data.data(), node.front_values.data(), front.data.size());
        } else {
            for (std::uint32_t row = 0; row < total; ++row) {
                for (std::uint32_t col = 0; col < total; ++col) {
                    for (unsigned b = 0; b < 4; ++b) {
                        front.data.push_back(
                            b == 0 ? detail::front_byte(node.node_id, row, col) : 0);
                    }
                }
            }
        }
        regions.push_back(front);
        RegionSpec update_region{
            prefix + ".update", update, update, cmd::DataFormat::Fp32, {}};
        for (std::uint32_t row = 0; row < update; ++row) {
            for (std::uint32_t col = 0; col < update; ++col) {
                for (unsigned b = 0; b < 4; ++b) {
                    update_region.data.push_back(
                        b == 0 ? detail::update_byte(node.node_id, row, col) : 0);
                }
            }
        }
        regions.push_back(update_region);
        regions.push_back(
            {prefix + ".factor_l", total, pivot, cmd::DataFormat::Fp32,
             std::vector<std::uint8_t>(
                 static_cast<std::size_t>(total) * pivot * 4, 0xEE)});
        regions.push_back(
            {prefix + ".factor_u", pivot, total, cmd::DataFormat::Fp32,
             std::vector<std::uint8_t>(
                 static_cast<std::size_t>(pivot) * total * 4, 0xDD)});
        std::vector<std::uint8_t> p_data;
        for (std::uint32_t index = 0; index < pivot; ++index) {
            detail::append_u32(p_data, 0xFFFFFFFFu);  // sentinel -1
        }
        regions.push_back(
            {prefix + ".p_vector", 1, pivot, cmd::DataFormat::Int32, p_data});
        regions.push_back(
            {prefix + ".solve_workspace", 2, pivot, cmd::DataFormat::Fp32,
             std::vector<std::uint8_t>(static_cast<std::size_t>(2) * pivot * 4, 0)});
    }

    // ---- command templates (symbolic waits) ------------------------------
    std::map<std::uint32_t, std::vector<CommandSpec>> plans;
    for (const auto& node : nodes) {
        const std::string prefix = "node." + std::to_string(node.node_id);
        std::vector<CommandSpec> plan;
        std::vector<std::pair<std::uint32_t, std::string>> admission;
        for (const auto& entry : node.contributions) {
            admission.emplace_back(entry.first, "commit");
        }
        plan.push_back(
            {cmd::Opcode::NodeBegin, node.node_id, prefix + ".front", admission, "begin"});
        plan.push_back(
            {cmd::Opcode::LoadFront, node.node_id, prefix + ".front",
             {{node.node_id, "begin"}}, "loaded"});
        std::string current = "loaded";
        for (const auto& entry : node.contributions) {
            const std::string token = "assembled_" + std::to_string(entry.first);
            plan.push_back(
                {cmd::Opcode::AssembleExtendAdd, node.node_id,
                 prefix + ".contribution." + std::to_string(entry.first),
                 {{node.node_id, current}, {entry.first, "update_stored"}},
                 token});
            current = token;
        }
        plan.push_back(
            {cmd::Opcode::PanelLu, node.node_id, prefix + ".front",
             {{node.node_id, current}}, "panel"});
        if (node.total_dim > node.pivot_dim) {
            plan.push_back(
                {cmd::Opcode::TrsmLeft, node.node_id, prefix + ".kernel.trsm_left",
                 {{node.node_id, "panel"}}, "trsm_left"});
            plan.push_back(
                {cmd::Opcode::TrsmRight, node.node_id, prefix + ".kernel.trsm_right",
                 {{node.node_id, "panel"}}, "trsm_right"});
            plan.push_back(
                {cmd::Opcode::GemmSchur, node.node_id, prefix + ".kernel.gemm",
                 {{node.node_id, "trsm_left"}, {node.node_id, "trsm_right"}},
                 "schur"});
            plan.push_back(
                {cmd::Opcode::StoreFactor, node.node_id, prefix + ".factor",
                 {{node.node_id, "trsm_left"}, {node.node_id, "trsm_right"}},
                 "factor_stored"});
            plan.push_back(
                {cmd::Opcode::StoreUpdate, node.node_id, "region:" + prefix + ".update",
                 {{node.node_id, "schur"}}, "update_stored"});
            plan.push_back(
                {cmd::Opcode::NodeCommit, node.node_id, prefix + ".front",
                 {{node.node_id, "factor_stored"}, {node.node_id, "update_stored"}},
                 "commit"});
        } else {
            plan.push_back(
                {cmd::Opcode::StoreFactor, node.node_id, prefix + ".factor",
                 {{node.node_id, "panel"}}, "factor_stored"});
            plan.push_back(
                {cmd::Opcode::NodeCommit, node.node_id, prefix + ".front",
                 {{node.node_id, "factor_stored"}}, "commit"});
        }
        plans[node.node_id] = plan;
    }

    std::map<std::uint32_t, std::size_t> cursors;
    std::set<std::pair<std::uint32_t, std::string>> emitted;
    std::map<std::uint32_t, std::map<std::string, std::uint32_t>> node_tokens;
    std::vector<CommandSpec> command_specs;
    std::uint32_t rotation = 0;
    const std::uint32_t node_count = static_cast<std::uint32_t>(nodes.size());
    std::size_t total_commands = 0;
    for (const auto& entry : plans) {
        total_commands += entry.second.size();
    }
    auto ref_ready = [&](const CommandSpec& spec) {
        for (const auto& ref : spec.waits) {
            if (emitted.count({ref.first, ref.second}) == 0) {
                return false;
            }
        }
        return true;
    };
    auto emit = [&](const CommandSpec& spec) {
        const std::uint32_t command_id =
            static_cast<std::uint32_t>(command_specs.size());
        command_specs.push_back(spec);
        node_tokens[spec.node_id][spec.token] = command_id;
        emitted.insert({spec.node_id, spec.token});
        return command_id;
    };
    while (command_specs.size() < total_commands) {
        std::optional<std::uint32_t> chosen;
        std::uint32_t chosen_index = 0;
        for (std::uint32_t step = 0; step < node_count; ++step) {
            const std::uint32_t index = (rotation + step) % node_count;
            const std::uint32_t node_id = nodes[index].node_id;
            const auto cursor = cursors[node_id];
            if (cursor >= plans[node_id].size()) {
                continue;
            }
            if (ref_ready(plans[node_id][cursor])) {
                chosen = node_id;
                chosen_index = index;
                break;
            }
        }
        if (!chosen.has_value()) {
            throw std::runtime_error("synthetic command templates have a cycle");
        }
        const auto cursor = cursors[*chosen];
        emit(plans[*chosen][cursor]);
        cursors[*chosen] = cursor + 1;
        rotation = (chosen_index + 1) % node_count;
    }

    // Solve phase (trailing chain).
    std::optional<std::pair<std::uint32_t, std::string>> previous;
    if (options.include_solve) {
    for (const auto& node : nodes) {
        CommandSpec spec{cmd::Opcode::SolveForward, node.node_id,
                         "node." + std::to_string(node.node_id) + ".solve_forward",
                         {{node.node_id, "commit"}}, "solve_forward"};
        if (previous.has_value()) {
            spec.waits.push_back(*previous);
        }
        emit(spec);
        previous = std::make_pair(node.node_id, std::string("solve_forward"));
    }
    for (auto iterator = nodes.rbegin(); iterator != nodes.rend(); ++iterator) {
        const auto& node = *iterator;
        CommandSpec spec{cmd::Opcode::SolveBackward, node.node_id,
                         "node." + std::to_string(node.node_id) + ".solve_backward",
                         {{node.node_id, "commit"}}, "solve_backward"};
        if (previous.has_value()) {
            spec.waits.push_back(*previous);
        }
        emit(spec);
        previous = std::make_pair(node.node_id, std::string("solve_backward"));
    }
    }

    // ---- descriptor specs -----------------------------------------------
    std::vector<DescriptorSpec> descriptor_specs;
    for (const auto& region : regions) {
        descriptor_specs.push_back(
            {"region:" + region.name, cmd::DescriptorType::Region,
             {0, 0, 0, 0, region.cols * 4, region.rows, region.cols,
              static_cast<std::uint32_t>(region.format),
              static_cast<std::uint32_t>(cmd::DataLayout::RowMajor), 0},
             {}});
    }
    for (const auto& node : nodes) {
        const std::string prefix = "node." + std::to_string(node.node_id);
        descriptor_specs.push_back(
            {prefix + ".factor", cmd::DescriptorType::Factor,
             {0, 0, 0, node.total_dim, node.pivot_dim,
              static_cast<std::uint32_t>(cmd::DataFormat::Fp32), cmd::NONE, 0, 0, 0},
             {}});
        descriptor_specs.push_back(
            {prefix + ".front", cmd::DescriptorType::Front,
             {0, node.total_dim, node.pivot_dim, options.tile_size, cmd::NONE,
              cmd::NONE, cmd::NONE, cmd::NONE,
              static_cast<std::uint32_t>(options.policy), options.threshold_bits},
             {}});
        if (node.total_dim > node.pivot_dim) {
            const std::uint32_t update = node.total_dim - node.pivot_dim;
            descriptor_specs.push_back(
                {prefix + ".kernel.trsm_left", cmd::DescriptorType::Kernel,
                 {0, 0, 0, node.pivot_dim, update, node.pivot_dim,
                  static_cast<std::uint32_t>(
                      cmd::KernelBackend::SystemCFp32DeviceModel),
                  static_cast<std::uint32_t>(cmd::DataFormat::Fp32), cmd::NONE,
                  options.tile_size},
                 {}});
            descriptor_specs.push_back(
                {prefix + ".kernel.trsm_right", cmd::DescriptorType::Kernel,
                 {0, 0, 0, update, node.pivot_dim, node.pivot_dim,
                  static_cast<std::uint32_t>(
                      cmd::KernelBackend::SystemCFp32DeviceModel),
                  static_cast<std::uint32_t>(cmd::DataFormat::Fp32), cmd::NONE,
                  options.tile_size},
                 {}});
            descriptor_specs.push_back(
                {prefix + ".kernel.gemm", cmd::DescriptorType::Kernel,
                 {0, 0, 0, update, update, node.pivot_dim,
                  static_cast<std::uint32_t>(
                      cmd::KernelBackend::SystemCFp32DeviceModel),
                  static_cast<std::uint32_t>(cmd::DataFormat::Fp32), cmd::NONE,
                  options.tile_size},
                 {}});
        }
        descriptor_specs.push_back(
            {prefix + ".solve_forward", cmd::DescriptorType::Solve,
             {cmd::NONE, cmd::NONE, cmd::NONE, cmd::NONE, cmd::NONE, 1,
              static_cast<std::uint32_t>(cmd::SolveDirection::Forward),
              static_cast<std::uint32_t>(cmd::DataFormat::Fp32), 0, 0},
             {}});
        descriptor_specs.push_back(
            {prefix + ".solve_backward", cmd::DescriptorType::Solve,
             {cmd::NONE, cmd::NONE, cmd::NONE, cmd::NONE, cmd::NONE, 1,
              static_cast<std::uint32_t>(cmd::SolveDirection::Backward),
              static_cast<std::uint32_t>(cmd::DataFormat::Fp32), 0, 0},
             {}});
    }
    for (const auto& node : nodes) {
        const std::string prefix = "node." + std::to_string(node.node_id);
        for (const auto& entry : node.contributions) {
            const auto& child = nodes.at(entry.first);
            const std::uint32_t update_dim = child.total_dim - child.pivot_dim;
            std::vector<std::uint8_t> payload;
            for (std::uint32_t index = 0; index < update_dim; ++index) {
                detail::append_u32(payload, index);  // row_map: identity
            }
            for (const auto column : entry.second) {
                detail::append_u32(payload, column);
            }
            descriptor_specs.push_back(
                {prefix + ".contribution." + std::to_string(entry.first),
                 cmd::DescriptorType::Contribution,
                 {cmd::NONE, cmd::NONE, entry.first, node.node_id, update_dim, update_dim,
                  update_dim * 4, node.total_dim * 4,
                  static_cast<std::uint32_t>(cmd::DataFormat::Fp32), 0},
                 payload});
        }
    }

    // Dependency descriptors per command with waits (payloads appended last).
    std::map<std::uint32_t, std::vector<std::uint32_t>> command_waits;
    for (std::uint32_t index = 0; index < command_specs.size(); ++index) {
        const auto& spec = command_specs[index];
        std::vector<std::uint32_t> waits;
        for (const auto& ref : spec.waits) {
            waits.push_back(node_tokens.at(ref.first).at(ref.second));
        }
        command_waits[index] = waits;
        if (!waits.empty()) {
            std::vector<std::uint8_t> payload;
            for (const auto token : waits) {
                detail::append_u32(payload, token);
            }
            descriptor_specs.push_back(
                {"dependency." + std::to_string(index), cmd::DescriptorType::Dependency,
                 {static_cast<std::uint32_t>(waits.size()), 0, 0, 0, 0, 0, 0, 0, 0, 0},
                 payload});
        }
    }

    std::map<std::string, std::uint32_t> descriptor_ids;
    for (std::uint32_t index = 0; index < descriptor_specs.size(); ++index) {
        descriptor_ids[descriptor_specs[index].name] = index;
    }
    auto id_of = [&](const std::string& name) -> std::uint32_t {
        const auto found = descriptor_ids.find(name);
        return found == descriptor_ids.end() ? cmd::NONE : found->second;
    };

    // Resolve descriptor references.
    for (auto& spec : descriptor_specs) {
        const std::string prefix = detail::node_prefix(spec.name);
        if (spec.type == cmd::DescriptorType::Factor) {
            spec.body[0] = id_of("region:" + prefix + ".factor_l");
            spec.body[1] = id_of("region:" + prefix + ".factor_u");
            spec.body[2] = id_of("region:" + prefix + ".p_vector");
            spec.body[6] = id_of("region:" + prefix + ".solve_workspace");
        } else if (spec.type == cmd::DescriptorType::Front) {
            const std::uint32_t node_id = static_cast<std::uint32_t>(
                std::stoul(prefix.substr(prefix.find('.') + 1)));
            spec.body[0] = id_of("region:" + prefix + ".front");
            if (!nodes[node_id].contributions.empty()) {
                spec.body[4] = id_of(
                    prefix + ".contribution." +
                    std::to_string(nodes[node_id].contributions.begin()->first));
            }
            spec.body[5] = id_of(prefix + ".factor");
            spec.body[6] = id_of("region:" + prefix + ".p_vector");
            spec.body[7] = id_of("region:" + prefix + ".solve_workspace");
        } else if (spec.type == cmd::DescriptorType::Kernel) {
            spec.body[0] = id_of(prefix + ".front");
            spec.body[1] = id_of(prefix + ".front");
            spec.body[2] = id_of(prefix + ".front");
        } else if (spec.type == cmd::DescriptorType::Solve) {
            spec.body[0] = id_of(prefix + ".factor");
            spec.body[1] = id_of("region:rhs_data");
            spec.body[2] = id_of("region:solution_data");
            spec.body[3] = id_of("region:" + prefix + ".p_vector");
            spec.body[4] = id_of("region:" + prefix + ".solve_workspace");
        } else if (spec.type == cmd::DescriptorType::Contribution) {
            const std::string child_id = spec.name.substr(spec.name.rfind('.') + 1);
            spec.body[0] = id_of("region:node." + child_id + ".update");
            spec.body[1] = id_of("region:" + prefix + ".front");
        }
    }

    // ---- layout ---------------------------------------------------------
    struct LaidRegion {
        std::string name;
        std::uint64_t offset;
        std::uint64_t size;
        std::vector<std::uint8_t> data;
    };
    std::vector<LaidRegion> laid;
    std::uint64_t cursor = 0;
    auto place = [&](const std::string& name, std::uint64_t size,
                     std::vector<std::uint8_t> data) {
        cursor = detail::align_up(cursor, 64);
        laid.push_back({name, cursor, size, std::move(data)});
        cursor += size;
    };

    std::uint64_t payload_bytes = 0;
    for (const auto& spec : descriptor_specs) {
        if (!spec.payload.empty()) {
            payload_bytes = detail::align_up(payload_bytes, 4);
            payload_bytes += spec.payload.size();
        }
    }
    place("command_buffer",
          static_cast<std::uint64_t>(command_specs.size()) * cmd::COMMAND_RECORD_BYTES,
          {});
    place("descriptor_table",
          static_cast<std::uint64_t>(descriptor_specs.size()) * cmd::DESCRIPTOR_RECORD_BYTES,
          {});
    place("descriptor_payload", payload_bytes, {});
    place("completion_status",
          static_cast<std::uint64_t>(command_specs.size()) * cmd::COMPLETION_RECORD_BYTES,
          {});
    for (auto& region : regions) {
        place(region.name, region.data.size(), std::move(region.data));
    }
    const std::uint64_t total_bytes = detail::align_up(cursor, 64);

    std::vector<std::uint8_t> image(static_cast<std::size_t>(total_bytes), 0);
    std::map<std::string, const LaidRegion*> laid_by_name;
    for (const auto& region : laid) {
        laid_by_name[region.name] = &region;
        if (!region.data.empty()) {
            std::memcpy(
                image.data() + region.offset, region.data.data(), region.data.size());
        }
    }

    std::uint64_t payload_cursor = 0;
    std::map<std::string, std::uint64_t> payload_offsets;
    for (const auto& spec : descriptor_specs) {
        if (spec.payload.empty()) {
            continue;
        }
        payload_cursor = detail::align_up(payload_cursor, 4);
        payload_offsets[spec.name] = payload_cursor;
        std::memcpy(
            image.data() + laid_by_name.at("descriptor_payload")->offset + payload_cursor,
            spec.payload.data(), spec.payload.size());
        payload_cursor += spec.payload.size();
    }

    auto build_descriptor = [&](const DescriptorSpec& spec, std::uint32_t index) {
        (void)index;
        cmd::Descriptor descriptor{};
        descriptor.descriptor_type = spec.type;
        for (std::size_t word = 0; word < 10 && word < spec.body.size(); ++word) {
            descriptor.body_words[word] = spec.body[word];
        }
        if (spec.type == cmd::DescriptorType::Region) {
            const auto* region = laid_by_name.at(spec.name.substr(7));  // strip "region:"
            descriptor.body_words[0] = static_cast<std::uint32_t>(region->offset);
            descriptor.body_words[1] = static_cast<std::uint32_t>(region->offset >> 32);
            descriptor.body_words[2] = static_cast<std::uint32_t>(region->size);
            descriptor.body_words[3] = static_cast<std::uint32_t>(region->size >> 32);
        }
        if (!spec.payload.empty()) {
            descriptor.payload_offset =
                laid_by_name.at("descriptor_payload")->offset + payload_offsets.at(spec.name);
            descriptor.payload_bytes = spec.payload.size();
        }
        return descriptor;
    };

    for (std::uint32_t index = 0; index < descriptor_specs.size(); ++index) {
        const auto record =
            cmd::DescriptorCodec::encode(build_descriptor(descriptor_specs[index], index));
        std::memcpy(
            image.data() + laid_by_name.at("descriptor_table")->offset +
                static_cast<std::uint64_t>(index) * cmd::DESCRIPTOR_RECORD_BYTES,
            record.data(), record.size());
    }
    for (std::uint32_t index = 0; index < command_specs.size(); ++index) {
        const auto& spec = command_specs[index];
        cmd::Command command{};
        command.opcode = spec.opcode;
        command.command_id = index;
        command.node_id = spec.node_id;
        command.descriptor_id = id_of(spec.descriptor);
        command.wait_list_id = id_of("dependency." + std::to_string(index));
        command.signal_token = index;
        const auto record = cmd::CommandCodec::encode(command);
        std::memcpy(
            image.data() + laid_by_name.at("command_buffer")->offset +
                static_cast<std::uint64_t>(index) * cmd::COMMAND_RECORD_BYTES,
            record.data(), record.size());
    }
    for (std::uint32_t index = 0; index < command_specs.size(); ++index) {
        cmd::Completion completion{};
        completion.command_id = index;
        completion.node_id = command_specs[index].node_id;
        completion.status_code = cmd::StatusCode::Ok;
        const auto record = cmd::CompletionCodec::encode(completion);
        std::memcpy(
            image.data() + laid_by_name.at("completion_status")->offset +
                static_cast<std::uint64_t>(index) * cmd::COMPLETION_RECORD_BYTES,
            record.data(), record.size());
    }

    // ---- manifest -------------------------------------------------------
    std::string parent_list;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        parent_list += std::to_string(nodes[index].parent_id);
        parent_list += index + 1 == nodes.size() ? "" : ", ";
    }
    std::string order_list;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        order_list += std::to_string(index);
        order_list += index + 1 == nodes.size() ? "" : ", ";
    }

    std::string manifest = "{\n";
    manifest +=
        "  \"abi\": {\n"
        "    \"name\": \"command_descriptor\",\n"
        "    \"version\": 2,\n"
        "    \"endianness\": \"little\",\n"
        "    \"address_unit\": \"byte_offset_from_memory_image_base\",\n"
        "    \"command_record_bytes\": 32,\n"
        "    \"descriptor_record_bytes\": 64,\n"
        "    \"completion_record_bytes\": 64,\n"
        "    \"pivot_policy_field\": \"FRONT_DESC.body_words[8..9]\"\n"
        "  },\n";
    manifest +=
        "  \"compiler\": {\n"
        "    \"device_format\": \"FP32\",\n"
        "    \"kernel_backend\": \"SYSTEMC_FP32_DEVICE_MODEL\",\n"
        "    \"global_bfp\": false,\n"
        "    \"alignment\": 64,\n"
        "    \"tile_size\": " + std::to_string(options.tile_size) + ",\n"
        "    \"max_front_size\": 256,\n"
        "    \"max_wait_tokens\": 256\n"
        "  },\n";
    float threshold_ratio = 0.0f;
    std::memcpy(&threshold_ratio, &options.threshold_bits, sizeof(threshold_ratio));
    std::ostringstream threshold_text;
    threshold_text.precision(9);
    threshold_text << std::fixed << threshold_ratio;
    manifest += std::string("  \"pivot\": {\n    \"policy\": \"") +
                (options.policy == cmd::PivotPolicy::Static ? "STATIC" : "DYNAMIC_PANEL") +
                "\",\n"
                "    \"scope\": \"batch_level_uniform\",\n"
                "    \"threshold_ratio\": " + threshold_text.str() + ",\n"
                "    \"threshold_bits_fp32\": " +
                std::to_string(options.threshold_bits) +
                ",\n"
                "    \"threshold_scale\": \"front_pre_elimination_max_abs\",\n"
                "    \"threshold_scale_definition\": \"test\",\n"
                "    \"static_pivot_failure_policy\": \"report numeric failure\",\n"
                "    \"p_vector\": {\n"
                "      \"region_suffix\": \"p_vector\",\n"
                "      \"format\": \"INT32\",\n"
                "      \"length\": \"pivot_dim\",\n"
                "      \"initial_value\": -1,\n"
                "      \"direction\": \"gather_within_pivot_block\",\n"
                "      \"semantics\": \"test\",\n"
                "      \"omitted\": false,\n"
                "      \"none_reference_allowed\": false\n"
                "    }\n"
                "  },\n";
    manifest +=
        "  \"required_device_capabilities\": {\n"
        "    \"STATIC\": {\"hpu_atu_dynamic_mapping\": \"bypassed\"},\n"
        "    \"DYNAMIC_PANEL\": {\"hpu_atu_dynamic_mapping\": \"required\"}\n"
        "  },\n";
    manifest += "  \"total_bytes\": " + std::to_string(total_bytes) + ",\n";
    manifest += "  \"memory_image\": {\n    \"file\": \"memory_image.bin\",\n    \"size\": " +
                std::to_string(total_bytes) + ",\n    \"regions\": {\n";
    for (std::size_t index = 0; index < laid.size(); ++index) {
        manifest += "      \"" + laid[index].name + "\": {\"offset\": " +
                    std::to_string(laid[index].offset) + ", \"size\": " +
                    std::to_string(laid[index].size) + "}";
        manifest += index + 1 == laid.size() ? "\n" : ",\n";
    }
    manifest += "    }\n  },\n";
    manifest +=
        "  \"command_batch\": {\n"
        "    \"region\": \"command_buffer\",\n"
        "    \"command_count\": " + std::to_string(command_specs.size()) +
        ",\n"
        "    \"record_bytes\": 32,\n"
        "    \"token_count\": " + std::to_string(command_specs.size()) +
        ",\n"
        "    \"max_wait_tokens\": 256,\n"
        "    \"static_batch\": true,\n"
        "    \"ordering\": \"interleaved_topological\",\n"
        "    \"admission_dependencies\": \"NODE_BEGIN waits for child commits\"\n"
        "  },\n";
    manifest +=
        "  \"descriptor_table\": {\n"
        "    \"region\": \"descriptor_table\",\n"
        "    \"payload_region\": \"descriptor_payload\",\n"
        "    \"descriptor_count\": " + std::to_string(descriptor_specs.size()) +
        ",\n"
        "    \"record_bytes\": 64\n"
        "  },\n";
    manifest +=
        "  \"completion_queue\": {\n"
        "    \"region\": \"completion_status\",\n"
        "    \"slot_count\": " + std::to_string(command_specs.size()) +
        ",\n"
        "    \"record_bytes\": 64,\n"
        "    \"initialization\": \"expected_OK_templates\"\n"
        "  },\n";
    manifest +=
        "  \"symbolic\": {\n"
        "    \"pattern_source\": \"union_of_pattern_B_and_pattern_B_transpose\",\n"
        "    \"node_count\": " + std::to_string(nodes.size()) + ",\n"
        "    \"parent\": [" + parent_list + "],\n"
        "    \"permutation\": [0],\n"
        "    \"factorization_order\": [" + order_list + "]\n"
        "  },\n";

    manifest += "  \"nodes\": {\n";
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const auto& node = nodes[index];
        std::vector<std::string> ids;
        for (std::uint32_t command = 0; command < command_specs.size(); ++command) {
            if (command_specs[command].node_id == node.node_id) {
                ids.push_back(std::to_string(command));
            }
        }
        manifest += "    \"" + std::to_string(node.node_id) + "\": {\n";
        manifest += "      \"parent_id\": " + std::to_string(node.parent_id) + ",\n";
        manifest += "      \"range\": {\"start\": 0, \"end\": " +
                    std::to_string(node.pivot_dim) + "},\n";
        manifest += "      \"front_indices\": [0],\n";
        manifest += "      \"total_dim\": " + std::to_string(node.total_dim) + ",\n";
        manifest += "      \"pivot_dim\": " + std::to_string(node.pivot_dim) + ",\n";
        manifest += "      \"update_dim\": " +
                    std::to_string(node.total_dim - node.pivot_dim) + ",\n";
        manifest += "      \"regions\": {\n";
        for (const std::string suffix :
             {"front", "update", "factor_l", "factor_u", "p_vector", "solve_workspace"}) {
            manifest += "        \"" + suffix + "\": \"node." +
                        std::to_string(node.node_id) + "." + suffix + "\",\n";
        }
        manifest = manifest.substr(0, manifest.size() - 2) + "\n      },\n";
        manifest += "      \"descriptors\": {\"front\": 0, \"factor\": 0},\n";
        manifest += "      \"command_ids\": [";
        for (std::size_t id = 0; id < ids.size(); ++id) {
            manifest += ids[id];
            manifest += id + 1 == ids.size() ? "" : ", ";
        }
        manifest += "],\n";
        manifest += "      \"tokens\": {}\n";
        manifest += index + 1 == nodes.size() ? "    }\n" : "    },\n";
    }
    manifest += "  }\n";
    manifest += "}\n";

    result.artifact =
        hw::dataplane::load_artifact_from_bytes(manifest, image, "<test>");
    for (std::uint32_t index = 0; index < command_specs.size(); ++index) {
        const auto& spec = command_specs[index];
        cmd::Command command{};
        command.opcode = spec.opcode;
        command.command_id = index;
        command.node_id = spec.node_id;
        command.descriptor_id = id_of(spec.descriptor);
        command.wait_list_id = id_of("dependency." + std::to_string(index));
        command.signal_token = index;
        result.commands.push_back(command);
    }
    for (std::uint32_t index = 0; index < descriptor_specs.size(); ++index) {
        result.descriptors.push_back(
            build_descriptor(descriptor_specs[index], index));
    }
    return result;
}

}  // namespace hw::testing
