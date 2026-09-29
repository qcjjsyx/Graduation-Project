#include "dataplane/artifact_loader.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>

#include "nlohmann/json.hpp"

namespace hw::dataplane {

namespace {

using json = nlohmann::json;
namespace cmd = hw::command;

[[noreturn]] void fail(const std::string& message) {
    throw ArtifactLoadError(message);
}

const json& require_object(const json& parent, const char* key, const std::string& context) {
    if (!parent.contains(key) || !parent.at(key).is_object()) {
        fail(context + ": missing object " + key);
    }
    return parent.at(key);
}

const json& require_array(const json& parent, const char* key, const std::string& context) {
    if (!parent.contains(key) || !parent.at(key).is_array()) {
        fail(context + ": missing array " + key);
    }
    return parent.at(key);
}

const json& require_key(const json& parent, const char* key, const std::string& context) {
    if (!parent.contains(key)) {
        fail(context + ": missing key " + key);
    }
    return parent.at(key);
}

std::uint64_t as_u64(const json& value, const std::string& context) {
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>();
    }
    if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < 0) {
            fail(context + ": expected non-negative integer");
        }
        return static_cast<std::uint64_t>(signed_value);
    }
    fail(context + ": expected non-negative integer");
}

std::int64_t as_i64(const json& value, const std::string& context) {
    if (value.is_number_unsigned()) {
        const auto wide = value.get<std::uint64_t>();
        if (wide > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            fail(context + ": integer does not fit i64");
        }
        return static_cast<std::int64_t>(wide);
    }
    if (value.is_number_integer()) {
        return value.get<std::int64_t>();
    }
    fail(context + ": expected integer");
}

std::uint32_t as_u32(const json& value, const std::string& context) {
    const auto wide = as_u64(value, context);
    if (wide > 0xFFFFFFFFu) {
        fail(context + ": value does not fit u32");
    }
    return static_cast<std::uint32_t>(wide);
}

std::string as_string(const json& value, const std::string& context) {
    if (!value.is_string()) {
        fail(context + ": expected string");
    }
    return value.get<std::string>();
}

float decode_threshold(std::uint32_t bits) {
    float threshold = 0.0f;
    std::memcpy(&threshold, &bits, sizeof(threshold));
    return threshold;
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& image, std::uint64_t address) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(image[address + index]) << (8 * index);
    }
    return value;
}

const char* descriptor_type_name(cmd::DescriptorType type) {
    switch (type) {
    case cmd::DescriptorType::Region: return "REGION_DESC";
    case cmd::DescriptorType::Front: return "FRONT_DESC";
    case cmd::DescriptorType::Contribution: return "CONTRIBUTION_DESC";
    case cmd::DescriptorType::Factor: return "FACTOR_DESC";
    case cmd::DescriptorType::Kernel: return "KERNEL_DESC";
    case cmd::DescriptorType::Solve: return "SOLVE_DESC";
    case cmd::DescriptorType::Scale: return "SCALE_DESC";
    case cmd::DescriptorType::Dependency: return "DEPENDENCY_DESC";
    }
    return "UNKNOWN_DESC";
}

cmd::PivotPolicy pivot_policy_from_name(const std::string& name) {
    if (name == "STATIC") {
        return cmd::PivotPolicy::Static;
    }
    if (name == "DYNAMIC_PANEL") {
        return cmd::PivotPolicy::DynamicPanel;
    }
    fail("manifest pivot.policy has unknown value " + name);
}

struct RegionShape {
    std::uint64_t base_addr{0};
    std::uint64_t byte_size{0};
    std::uint64_t row_stride{0};
    std::uint32_t rows{0};
    std::uint32_t cols{0};
    cmd::DataFormat format{cmd::DataFormat::Fp32};
};

struct ArtifactProbe {
    const std::vector<std::uint8_t>* image{nullptr};
    const std::vector<cmd::Descriptor>* descriptors{nullptr};
    const std::vector<cmd::Command>* commands{nullptr};
    const DeviceBootConfig* config{nullptr};
};

cmd::Descriptor descriptor_at(
    const ArtifactProbe& probe,
    std::uint32_t descriptor_id,
    const std::string& context) {
    if (descriptor_id >= probe.descriptors->size()) {
        fail(context + ": descriptor id out of range");
    }
    return probe.descriptors->at(descriptor_id);
}

RegionShape region_shape(const cmd::Descriptor& descriptor) {
    RegionShape shape;
    shape.base_addr = static_cast<std::uint64_t>(descriptor.body_words[0]) |
                      (static_cast<std::uint64_t>(descriptor.body_words[1]) << 32);
    shape.byte_size = static_cast<std::uint64_t>(descriptor.body_words[2]) |
                      (static_cast<std::uint64_t>(descriptor.body_words[3]) << 32);
    shape.row_stride = descriptor.body_words[4];
    shape.rows = descriptor.body_words[5];
    shape.cols = descriptor.body_words[6];
    shape.format = static_cast<cmd::DataFormat>(descriptor.body_words[7]);
    return shape;
}

void require_descriptor_type(
    const ArtifactProbe& probe,
    std::uint32_t index,
    cmd::DescriptorType expected,
    const std::string& context) {
    if (index == cmd::NONE) {
        return;
    }
    if (index >= probe.descriptors->size()) {
        fail(context + ": descriptor reference out of range");
    }
    if (probe.descriptors->at(index).descriptor_type != expected) {
        fail(
            context + ": expected " + descriptor_type_name(expected) + ", found " +
            descriptor_type_name(probe.descriptors->at(index).descriptor_type));
    }
}

void require_mandatory_descriptor_type(
    const ArtifactProbe& probe,
    std::uint32_t index,
    cmd::DescriptorType expected,
    const std::string& context) {
    if (index == cmd::NONE) {
        fail(context + ": reference must not be NONE");
    }
    require_descriptor_type(probe, index, expected, context);
}

// Follows a REGION_DESC reference and checks INT32 P-vector capacity/sentinel.
void check_p_vector_region(
    const ArtifactProbe& probe,
    std::uint32_t index,
    std::uint32_t pivot_dim,
    const std::string& context) {
    if (index == cmd::NONE) {
        fail(context + ": P-vector reference must not be NONE");
    }
    const auto descriptor = descriptor_at(probe, index, context);
    if (descriptor.descriptor_type != cmd::DescriptorType::Region) {
        fail(context + ": P-vector reference is not a REGION_DESC");
    }
    const auto shape = region_shape(descriptor);
    if (shape.format != cmd::DataFormat::Int32) {
        fail(context + ": P-vector region must be INT32");
    }
    const std::uint64_t capacity =
        static_cast<std::uint64_t>(shape.rows) * shape.cols;
    if (pivot_dim == 0 || capacity != pivot_dim ||
        shape.byte_size != static_cast<std::uint64_t>(pivot_dim) * 4) {
        fail(context + ": P-vector region capacity must equal pivot_dim");
    }
    const std::uint64_t end = shape.base_addr + shape.byte_size;
    if (end > probe.image->size()) {
        fail(context + ": P-vector region exceeds the memory image");
    }
    for (std::uint64_t offset = 0; offset < shape.byte_size; offset += 4) {
        std::int32_t value = 0;
        std::memcpy(&value, probe.image->data() + shape.base_addr + offset, 4);
        if (value != cmd::PVECTOR_INIT_VALUE) {
            fail(
                context + ": P-vector must be initialized to the INT32 sentinel "
                "-1; a zero array is never identity");
        }
    }
}

std::optional<std::set<cmd::DescriptorType>> allowed_descriptor_types(
    cmd::Opcode opcode) {
    switch (opcode) {
    case cmd::Opcode::NodeBegin:
    case cmd::Opcode::LoadFront:
    case cmd::Opcode::PanelLu:
    case cmd::Opcode::NodeCommit:
        return std::set<cmd::DescriptorType>{cmd::DescriptorType::Front};
    case cmd::Opcode::AssembleExtendAdd:
        return std::set<cmd::DescriptorType>{cmd::DescriptorType::Contribution};
    case cmd::Opcode::TrsmLeft:
    case cmd::Opcode::TrsmRight:
    case cmd::Opcode::GemmSchur:
        return std::set<cmd::DescriptorType>{cmd::DescriptorType::Kernel};
    case cmd::Opcode::StoreFactor:
        return std::set<cmd::DescriptorType>{cmd::DescriptorType::Factor};
    case cmd::Opcode::StoreUpdate:
        return std::set<cmd::DescriptorType>{cmd::DescriptorType::Region};
    case cmd::Opcode::SolveForward:
    case cmd::Opcode::SolveBackward:
        return std::set<cmd::DescriptorType>{cmd::DescriptorType::Solve};
    case cmd::Opcode::AbortNode:
        return std::nullopt;
    }
    return std::nullopt;
}

std::vector<RegionInfo> build_regions(
    const json& region_table,
    std::uint64_t total_bytes,
    std::uint64_t alignment) {
    std::vector<RegionInfo> regions;
    for (auto iterator = region_table.begin(); iterator != region_table.end(); ++iterator) {
        const std::string name = iterator.key();
        const json& raw = iterator.value();
        if (!raw.is_object()) {
            fail("memory_image.regions." + name + ": expected object");
        }
        RegionInfo info;
        info.name = name;
        info.role = classify_region_role(name);
        info.offset = as_u64(require_key(raw, "offset", name), name + ".offset");
        info.size = as_u64(require_key(raw, "size", name), name + ".size");
        if (info.offset % alignment != 0) {
            fail(name + ": region offset is not aligned");
        }
        if (info.size > total_bytes || info.offset > total_bytes - info.size) {
            fail(name + ": region exceeds total_bytes");
        }
        regions.push_back(std::move(info));
    }
    std::vector<const RegionInfo*> ordered;
    ordered.reserve(regions.size());
    for (const auto& region : regions) {
        if (region.size != 0) {
            ordered.push_back(&region);
        }
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const RegionInfo* lhs, const RegionInfo* rhs) {
            return lhs->offset < rhs->offset;
        });
    for (std::size_t index = 1; index < ordered.size(); ++index) {
        const auto* previous = ordered[index - 1];
        const auto* current = ordered[index];
        if (previous->offset + previous->size > current->offset) {
            fail(
                "regions " + previous->name + " and " + current->name +
                " overlap");
        }
    }
    return regions;
}

std::uint64_t required_region_size(
    const DeviceBootConfig& config,
    const std::string& name,
    const std::string& context) {
    const RegionInfo* region = config.region(name);
    if (region == nullptr) {
        fail(context + ": missing memory region " + name);
    }
    return region->size;
}

void validate_descriptor_references(const ArtifactProbe& probe) {
    for (std::uint32_t index = 0; index < probe.descriptors->size(); ++index) {
        const auto& descriptor = probe.descriptors->at(index);
        const auto body = descriptor.body_words;
        const std::string context =
            std::string("descriptor ") + std::to_string(index) + " (" +
            descriptor_type_name(descriptor.descriptor_type) + ")";
        switch (descriptor.descriptor_type) {
        case cmd::DescriptorType::Region:
        case cmd::DescriptorType::Scale:
            break;
        case cmd::DescriptorType::Front:
            require_descriptor_type(probe, body[0], cmd::DescriptorType::Region, context + " front region");
            require_descriptor_type(probe, body[4], cmd::DescriptorType::Contribution, context + " contribution");
            require_mandatory_descriptor_type(probe, body[5], cmd::DescriptorType::Factor, context + " factor");
            require_mandatory_descriptor_type(probe, body[6], cmd::DescriptorType::Region, context + " P-vector");
            require_mandatory_descriptor_type(probe, body[7], cmd::DescriptorType::Region, context + " workspace");
            break;
        case cmd::DescriptorType::Contribution:
            require_descriptor_type(probe, body[0], cmd::DescriptorType::Region, context + " source");
            require_descriptor_type(probe, body[1], cmd::DescriptorType::Region, context + " target");
            break;
        case cmd::DescriptorType::Factor:
            require_descriptor_type(probe, body[0], cmd::DescriptorType::Region, context + " L");
            require_descriptor_type(probe, body[1], cmd::DescriptorType::Region, context + " U");
            require_mandatory_descriptor_type(probe, body[2], cmd::DescriptorType::Region, context + " P-vector");
            require_descriptor_type(probe, body[6], cmd::DescriptorType::Region, context + " workspace");
            break;
        case cmd::DescriptorType::Kernel:
            for (std::size_t operand = 0; operand < 3; ++operand) {
                require_descriptor_type(
                    probe, body[operand], cmd::DescriptorType::Front,
                    context + " operand " + std::to_string(operand));
            }
            require_descriptor_type(probe, body[8], cmd::DescriptorType::Scale, context + " scale");
            break;
        case cmd::DescriptorType::Solve:
            require_mandatory_descriptor_type(probe, body[0], cmd::DescriptorType::Factor, context + " factor");
            require_descriptor_type(probe, body[1], cmd::DescriptorType::Region, context + " RHS");
            require_descriptor_type(probe, body[2], cmd::DescriptorType::Region, context + " solution");
            require_mandatory_descriptor_type(probe, body[3], cmd::DescriptorType::Region, context + " P-vector");
            require_descriptor_type(probe, body[4], cmd::DescriptorType::Region, context + " workspace");
            break;
        case cmd::DescriptorType::Dependency:
            break;
        }
    }
}

void validate_pivot_contract(
    const ArtifactProbe& probe,
    const json& manifest,
    DeviceBootConfig& config) {
    const json& pivot = require_object(manifest, "pivot", "manifest");
    const auto policy_name = as_string(require_key(pivot, "policy", "pivot"), "pivot.policy");
    const auto policy = pivot_policy_from_name(policy_name);
    config.pivot_policy = policy;

    const std::uint32_t threshold_bits =
        as_u32(require_key(pivot, "threshold_bits_fp32", "pivot"), "pivot.threshold_bits_fp32");
    const float decoded = decode_threshold(threshold_bits);
    if (!(decoded >= 0.0f) || !(decoded < 1.0f)) {
        fail("manifest pivot threshold must be a binary32 value in [0, 1)");
    }
    const float ratio = static_cast<float>(
        require_key(pivot, "threshold_ratio", "pivot").get<double>());
    if (ratio != decoded) {
        fail("manifest threshold_ratio does not equal the binary32 threshold bits");
    }
    if (as_string(require_key(pivot, "scope", "pivot"), "pivot.scope") !=
        "batch_level_uniform") {
        fail("manifest pivot policy scope must be batch_level_uniform");
    }
    if (as_string(require_key(pivot, "threshold_scale", "pivot"), "pivot.threshold_scale") !=
        "front_pre_elimination_max_abs") {
        fail("manifest pivot threshold scale must be explicit");
    }
    if (as_string(require_key(pivot, "static_pivot_failure_policy", "pivot"),
                  "pivot.static_pivot_failure_policy")
            .empty()) {
        fail("manifest must declare the static pivot failure policy");
    }
    const json& p_vector = require_object(pivot, "p_vector", "pivot");
    if (p_vector.contains("omitted") && p_vector.at("omitted").get<bool>() != false) {
        fail("v2 artifacts must keep explicit P-vector regions");
    }
    if (p_vector.contains("none_reference_allowed") &&
        p_vector.at("none_reference_allowed").get<bool>() != false) {
        fail("NONE P-vector references are not allowed in v2");
    }

    const json& capabilities =
        require_object(manifest, "required_device_capabilities", "manifest");
    const json& policy_caps =
        require_object(capabilities, policy_name.c_str(), "required_device_capabilities");
    const std::string mapping = as_string(
        require_key(policy_caps, "hpu_atu_dynamic_mapping", policy_name),
        policy_name + ".hpu_atu_dynamic_mapping");
    if (policy == cmd::PivotPolicy::Static && mapping != "bypassed") {
        fail("STATIC must bypass the HPU/ATU dynamic mapping");
    }
    if (policy == cmd::PivotPolicy::DynamicPanel && mapping != "required") {
        fail("DYNAMIC_PANEL requires the HPU/ATU dynamic mapping");
    }

    std::uint32_t front_count = 0;
    for (std::uint32_t index = 0; index < probe.descriptors->size(); ++index) {
        const auto& descriptor = probe.descriptors->at(index);
        const auto body = descriptor.body_words;
        if (descriptor.descriptor_type == cmd::DescriptorType::Front) {
            ++front_count;
            if (body[8] != static_cast<std::uint32_t>(policy)) {
                fail("FRONT_DESC pivot policy disagrees with the manifest policy");
            }
            if (body[9] != threshold_bits) {
                fail("FRONT_DESC pivot threshold disagrees with the manifest threshold");
            }
            const auto front_p = body[6];
            check_p_vector_region(
                probe, front_p, body[2],
                "descriptor " + std::to_string(index) + " front");
            const auto factor = descriptor_at(probe, body[5], "front factor");
            if (factor.body_words[2] != front_p) {
                fail("FRONT_DESC and FACTOR_DESC must reference one P-vector region");
            }
        } else if (descriptor.descriptor_type == cmd::DescriptorType::Factor) {
            check_p_vector_region(
                probe, body[2], body[4],
                "descriptor " + std::to_string(index) + " factor");
        } else if (descriptor.descriptor_type == cmd::DescriptorType::Solve) {
            const auto factor = descriptor_at(probe, body[0], "solve factor");
            const auto solve_p = body[3];
            check_p_vector_region(
                probe, solve_p, factor.body_words[4],
                "descriptor " + std::to_string(index) + " solve");
            if (factor.body_words[2] != solve_p) {
                fail("SOLVE_DESC and FACTOR_DESC must reference one P-vector region");
            }
        }
    }
    if (front_count == 0) {
        fail("artifact contains no FRONT_DESC");
    }
}

void validate_command_semantics(
    const ArtifactProbe& probe,
    const std::vector<std::uint8_t>& image) {
    for (const auto& command : *probe.commands) {
        const auto allowed = allowed_descriptor_types(command.opcode);
        if (allowed.has_value()) {
            if (command.descriptor_id == cmd::NONE) {
                fail("command " + std::to_string(command.command_id) + " is missing its descriptor");
            }
            if (command.descriptor_id >= probe.descriptors->size()) {
                fail("command descriptor reference out of range");
            }
            const auto type = probe.descriptors->at(command.descriptor_id).descriptor_type;
            if (allowed->count(type) == 0) {
                fail(
                    "command " + std::to_string(command.command_id) +
                    " uses the wrong descriptor type");
            }
        }
        if (command.wait_list_id != cmd::NONE) {
            const auto& wait_descriptor = descriptor_at(
                probe, command.wait_list_id, "wait list");
            if (wait_descriptor.descriptor_type != cmd::DescriptorType::Dependency) {
                fail("wait_list_id is not a DEPENDENCY_DESC");
            }
            const auto count = wait_descriptor.body_words[0];
            for (std::uint32_t token_index = 0; token_index < count; ++token_index) {
                const std::uint64_t offset =
                    wait_descriptor.payload_offset + token_index * 4;
                if (offset + 4 > image.size()) {
                    fail("dependency payload exceeds the memory image");
                }
                const std::uint32_t token = read_u32(image, offset);
                if (token >= command.command_id) {
                    fail("command waits on a non-preceding token");
                }
            }
        }
    }
}

void validate_nodes(const json& manifest, const ArtifactProbe& probe) {
    const json& symbolic = require_object(manifest, "symbolic", "manifest");
    const std::uint32_t node_count = as_u32(
        require_key(symbolic, "node_count", "symbolic"), "symbolic.node_count");
    const json& parent = require_array(symbolic, "parent", "symbolic");
    if (parent.size() != node_count) {
        fail("symbolic.parent length mismatch");
    }
    const json& order = require_array(symbolic, "factorization_order", "symbolic");
    if (order.size() != node_count) {
        fail("symbolic.factorization_order length mismatch");
    }
    std::set<std::uint32_t> ordered;
    for (const auto& value : order) {
        ordered.insert(as_u32(value, "factorization_order"));
    }
    for (std::uint32_t index = 0; index < node_count; ++index) {
        if (ordered.count(index) == 0) {
            fail("factorization_order is not a permutation of node ids");
        }
    }
    const json& nodes = require_object(manifest, "nodes", "manifest");
    if (nodes.size() != node_count) {
        fail("manifest nodes do not match symbolic.node_count");
    }
    std::set<std::uint32_t> seen_commands;
    std::uint64_t max_front_size = 0;
    if (manifest.contains("compiler") && manifest.at("compiler").is_object() &&
        manifest.at("compiler").contains("max_front_size")) {
        max_front_size = as_u64(
            manifest.at("compiler").at("max_front_size"), "compiler.max_front_size");
    }
    for (std::uint32_t node_id = 0; node_id < node_count; ++node_id) {
        const std::string key = std::to_string(node_id);
        if (!nodes.contains(key)) {
            fail("manifest nodes are not a dense 0..N-1 range");
        }
        const json& node = nodes.at(key);
        const std::int32_t parent_id = static_cast<std::int32_t>(
            as_i64(require_key(node, "parent_id", key), key + ".parent_id"));
        const std::int64_t parent_value = as_i64(parent.at(node_id), "symbolic.parent");
        if (parent_value < -1 ||
            parent_value >= static_cast<std::int64_t>(node_count)) {
            fail("symbolic.parent contains an out-of-range node id");
        }
        if (parent_id != parent_value) {
            fail("node " + key + " parent mismatch");
        }
        const std::uint32_t total = as_u32(require_key(node, "total_dim", key), key + ".total_dim");
        const std::uint32_t pivot = as_u32(require_key(node, "pivot_dim", key), key + ".pivot_dim");
        const std::uint32_t update = as_u32(require_key(node, "update_dim", key), key + ".update_dim");
        if (pivot == 0 || total != pivot + update || total == 0) {
            fail("node " + key + " dimensions are invalid");
        }
        if (max_front_size != 0 && total > max_front_size) {
            fail("node " + key + " front exceeds compiler.max_front_size");
        }
        const json& regions = require_object(node, "regions", key);
        for (auto iterator = regions.begin(); iterator != regions.end(); ++iterator) {
            const std::string region_name = iterator.value().get<std::string>();
            if (probe.config->region(region_name) == nullptr) {
                fail("node " + key + " references unknown region " + region_name);
            }
        }
        const json& command_ids = require_array(node, "command_ids", key);
        std::set<cmd::Opcode> opcodes;
        for (const auto& value : command_ids) {
            const std::uint32_t command_id = as_u32(value, key + ".command_ids");
            if (command_id >= probe.commands->size()) {
                fail("node references an unknown command");
            }
            if (!seen_commands.insert(command_id).second) {
                fail("a command is owned by multiple nodes");
            }
            const auto& command = probe.commands->at(command_id);
            if (command.node_id != node_id) {
                fail("command node id disagrees with the manifest");
            }
            opcodes.insert(command.opcode);
        }
        for (const auto required : {
                 cmd::Opcode::NodeBegin,
                 cmd::Opcode::LoadFront,
                 cmd::Opcode::PanelLu,
                 cmd::Opcode::StoreFactor,
                 cmd::Opcode::NodeCommit,
                 cmd::Opcode::SolveForward,
                 cmd::Opcode::SolveBackward,
             }) {
            if (opcodes.count(required) == 0) {
                fail("node " + key + " is missing a required opcode");
            }
        }
    }
    if (seen_commands.size() != probe.commands->size()) {
        fail("manifest does not assign every command to a node");
    }

    // Admission contract: NODE_BEGIN must wait on the completion token of
    // every child so the device never has to infer the elimination forest.
    std::map<std::uint32_t, std::uint32_t> begin_command;
    std::map<std::uint32_t, std::uint32_t> commit_token;
    for (const auto& command : *probe.commands) {
        if (command.opcode == cmd::Opcode::NodeBegin) {
            begin_command[command.node_id] = command.command_id;
        } else if (command.opcode == cmd::Opcode::NodeCommit) {
            commit_token[command.node_id] = command.signal_token;
        }
    }
    for (std::uint32_t node_id = 0; node_id < node_count; ++node_id) {
        std::vector<std::uint32_t> children;
        for (std::uint32_t child = 0; child < node_count; ++child) {
            if (parent.at(child).get<std::int64_t>() == static_cast<std::int64_t>(node_id)) {
                children.push_back(child);
            }
        }
        if (children.empty()) {
            continue;
        }
        const auto begin = begin_command.find(node_id);
        if (begin == begin_command.end() || commit_token.find(node_id) == commit_token.end()) {
            fail("node " + std::to_string(node_id) + " is missing NODE_BEGIN or NODE_COMMIT");
        }
        const auto& begin_cmd = probe.commands->at(begin->second);
        if (begin_cmd.wait_list_id == cmd::NONE) {
            fail("node " + std::to_string(node_id) +
                 " NODE_BEGIN is missing its admission dependency");
        }
        const auto dependency = descriptor_at(probe, begin_cmd.wait_list_id, "admission");
        if (dependency.descriptor_type != cmd::DescriptorType::Dependency) {
            fail("NODE_BEGIN wait_list_id is not a DEPENDENCY_DESC");
        }
        const auto waits = cmd::DescriptorCodec::dependency_tokens(
            dependency, *probe.image, probe.config->token_count, probe.config->max_wait_tokens);
        const std::set<std::uint32_t> wait_set(waits.begin(), waits.end());
        for (const auto child : children) {
            const auto token = commit_token.find(child);
            if (token == commit_token.end() || wait_set.count(token->second) == 0) {
                fail(
                    "node " + std::to_string(node_id) +
                    " NODE_BEGIN must wait for child " + std::to_string(child) +
                    " to complete");
            }
        }
    }

    if (probe.config->token_count != 0) {
        std::set<std::uint32_t> signals;
        for (const auto& command : *probe.commands) {
            if (command.signal_token != cmd::NONE) {
                signals.insert(command.signal_token);
            }
        }
        if (signals.size() != probe.config->token_count) {
            fail("signal tokens are not a dense 0..token_count-1 set");
        }
    }
}

void validate_completion_templates(
    const std::vector<cmd::Command>& commands,
    const std::vector<cmd::Completion>& completions) {
    if (completions.size() != commands.size()) {
        fail("completion slot count must equal command count");
    }
    for (std::size_t slot = 0; slot < commands.size(); ++slot) {
        const auto& command = commands[slot];
        const auto& completion = completions[slot];
        if (completion.command_id != command.command_id ||
            completion.node_id != command.node_id) {
            fail("completion template does not match its command");
        }
        if (completion.status_code != cmd::StatusCode::Ok) {
            fail("completion template status must be OK before execution");
        }
        if (completion.pivot_count != 0 || completion.start_cycle != 0 ||
            completion.finish_cycle != 0 || completion.read_bytes != 0 ||
            completion.write_bytes != 0 || completion.stall_cycles != 0 ||
            completion.overflow_count != 0 || completion.retry_count != 0) {
            fail("completion template counters must start at zero");
        }
    }
}

}  // namespace

LoadedArtifact load_artifact(const std::string& device_dir) {
    const std::filesystem::path directory(device_dir);
    const auto manifest_path = directory / "manifest.json";
    std::ifstream manifest_stream(manifest_path, std::ios::binary);
    if (!manifest_stream) {
        fail("cannot open " + manifest_path.string());
    }
    const std::string manifest_text{
        std::istreambuf_iterator<char>(manifest_stream),
        std::istreambuf_iterator<char>()};

    json manifest;
    try {
        manifest = json::parse(manifest_text);
    } catch (const json::exception& exception) {
        fail(std::string("manifest.json is not valid JSON: ") + exception.what());
    }
    if (!manifest.is_object()) {
        fail("manifest.json must contain a JSON object");
    }
    const json& image_meta = require_object(manifest, "memory_image", "manifest");
    const std::string file = as_string(require_key(image_meta, "file", "memory_image"), "memory_image.file");
    if (file.empty() || file.find("..") != std::string::npos ||
        file.find('/') != std::string::npos || file.find('\\') != std::string::npos) {
        fail("memory_image.file must be a plain file name");
    }
    const auto image_path = directory / file;
    std::ifstream image_stream(image_path, std::ios::binary);
    if (!image_stream) {
        fail("cannot open " + image_path.string());
    }
    std::vector<std::uint8_t> image{
        std::istreambuf_iterator<char>(image_stream),
        std::istreambuf_iterator<char>()};
    return load_artifact_from_bytes(manifest_text, std::move(image), device_dir);
}

LoadedArtifact load_artifact_from_bytes(
    const std::string& manifest_json,
    std::vector<std::uint8_t> image,
    const std::string& source_label) {
    json manifest;
    try {
        manifest = json::parse(manifest_json);
    } catch (const json::exception& exception) {
        fail(source_label + ": manifest is not valid JSON: " + exception.what());
    }
    if (!manifest.is_object()) {
        fail(source_label + ": manifest must be a JSON object");
    }

    DeviceBootConfig config;

    const json& abi = require_object(manifest, "abi", "manifest");
    if (as_string(require_key(abi, "name", "abi"), "abi.name") != "command_descriptor") {
        fail("manifest is not a Command/Descriptor artifact");
    }
    if (as_u32(require_key(abi, "version", "abi"), "abi.version") != cmd::SCHEMA_VERSION) {
        fail("manifest schema version is not Command/Descriptor v2");
    }
    if (as_string(require_key(abi, "endianness", "abi"), "abi.endianness") != "little") {
        fail("only little-endian artifacts are supported");
    }
    if (as_string(require_key(abi, "address_unit", "abi"), "abi.address_unit") !=
        "byte_offset_from_memory_image_base") {
        fail("device addresses must be byte offsets from the memory image base");
    }
    if (as_u32(require_key(abi, "command_record_bytes", "abi"), "abi.command_record_bytes") !=
            cmd::COMMAND_RECORD_BYTES ||
        as_u32(require_key(abi, "descriptor_record_bytes", "abi"), "abi.descriptor_record_bytes") !=
            cmd::DESCRIPTOR_RECORD_BYTES ||
        as_u32(require_key(abi, "completion_record_bytes", "abi"), "abi.completion_record_bytes") !=
            cmd::COMPLETION_RECORD_BYTES) {
        fail("manifest record sizes do not match schema v2");
    }
    if (as_string(require_key(abi, "pivot_policy_field", "abi"), "abi.pivot_policy_field") !=
        "FRONT_DESC.body_words[8..9]") {
        fail("manifest pivot_policy_field is inconsistent");
    }

    const json& compiler = require_object(manifest, "compiler", "manifest");
    config.alignment = as_u64(require_key(compiler, "alignment", "compiler"), "compiler.alignment");
    if (config.alignment < 64) {
        fail("compiler.alignment must be at least 64 bytes");
    }
    if (as_string(require_key(compiler, "device_format", "compiler"), "compiler.device_format") !=
        "FP32") {
        fail("Command v2 main path must be FP32");
    }
    if (require_key(compiler, "global_bfp", "compiler").get<bool>() != false) {
        fail("Command v2 main path must not use global BFP");
    }

    config.total_bytes = as_u64(require_key(manifest, "total_bytes", "manifest"), "total_bytes");
    if (config.total_bytes != image.size()) {
        fail("memory image length does not match manifest total_bytes");
    }

    const json& image_meta = require_object(manifest, "memory_image", "manifest");
    if (as_u64(require_key(image_meta, "size", "memory_image"), "memory_image.size") !=
        config.total_bytes) {
        fail("memory_image.size does not match total_bytes");
    }
    const json& region_table = require_object(image_meta, "regions", "memory_image");
    config.regions = build_regions(region_table, config.total_bytes, config.alignment);

    const json& batch = require_object(manifest, "command_batch", "manifest");
    if (as_string(require_key(batch, "region", "command_batch"), "command_batch.region") !=
        "command_buffer") {
        fail("command_batch.region must be command_buffer");
    }
    config.command_count = as_u32(
        require_key(batch, "command_count", "command_batch"), "command_batch.command_count");
    config.token_count = as_u32(
        require_key(batch, "token_count", "command_batch"), "command_batch.token_count");
    config.max_wait_tokens = as_u32(
        require_key(batch, "max_wait_tokens", "command_batch"), "command_batch.max_wait_tokens");
    if (config.command_count == 0) {
        fail("command_count must be positive");
    }
    if (as_u32(require_key(batch, "record_bytes", "command_batch"), "command_batch.record_bytes") !=
        cmd::COMMAND_RECORD_BYTES) {
        fail("command_batch.record_bytes mismatch");
    }
    if (batch.contains("static_batch") && batch.at("static_batch").get<bool>() != true) {
        fail("only a static command batch is supported by schema v2");
    }
    if (config.token_count != config.command_count) {
        fail("token_count must equal command_count in the static batch");
    }

    const json& descriptor_meta = require_object(manifest, "descriptor_table", "manifest");
    if (as_string(require_key(descriptor_meta, "region", "descriptor_table"), "descriptor_table.region") !=
        "descriptor_table") {
        fail("descriptor_table.region mismatch");
    }
    if (as_string(require_key(descriptor_meta, "payload_region", "descriptor_table"),
                  "descriptor_table.payload_region") != "descriptor_payload") {
        fail("descriptor_table.payload_region mismatch");
    }
    config.descriptor_count = as_u32(
        require_key(descriptor_meta, "descriptor_count", "descriptor_table"),
        "descriptor_table.descriptor_count");
    if (config.descriptor_count == 0) {
        fail("descriptor_count must be positive");
    }
    if (as_u32(require_key(descriptor_meta, "record_bytes", "descriptor_table"),
               "descriptor_table.record_bytes") != cmd::DESCRIPTOR_RECORD_BYTES) {
        fail("descriptor_table.record_bytes mismatch");
    }

    const json& completion_meta = require_object(manifest, "completion_queue", "manifest");
    if (as_string(require_key(completion_meta, "region", "completion_queue"),
                  "completion_queue.region") != "completion_status") {
        fail("completion_queue.region mismatch");
    }
    config.completion_count = as_u32(
        require_key(completion_meta, "slot_count", "completion_queue"),
        "completion_queue.slot_count");
    if (as_u32(require_key(completion_meta, "record_bytes", "completion_queue"),
               "completion_queue.record_bytes") != cmd::COMPLETION_RECORD_BYTES) {
        fail("completion_queue.record_bytes mismatch");
    }
    if (config.completion_count != config.command_count) {
        fail("completion slot count must equal command count");
    }

    const auto command_size = required_region_size(config, "command_buffer", "memory");
    const auto descriptor_size = required_region_size(config, "descriptor_table", "memory");
    const auto payload_size = required_region_size(config, "descriptor_payload", "memory");
    const auto completion_size = required_region_size(config, "completion_status", "memory");
    if (command_size != static_cast<std::uint64_t>(config.command_count) * cmd::COMMAND_RECORD_BYTES) {
        fail("command_buffer has the wrong size for command_count");
    }
    if (descriptor_size !=
        static_cast<std::uint64_t>(config.descriptor_count) * cmd::DESCRIPTOR_RECORD_BYTES) {
        fail("descriptor_table has the wrong size for descriptor_count");
    }
    if (payload_size > config.total_bytes) {
        fail("descriptor_payload exceeds the image");
    }
    if (completion_size !=
        static_cast<std::uint64_t>(config.completion_count) * cmd::COMPLETION_RECORD_BYTES) {
        fail("completion_status has the wrong size for slot_count");
    }

    config.command_table_addr = config.region("command_buffer")->offset;
    config.descriptor_table_addr = config.region("descriptor_table")->offset;
    config.completion_table_addr = config.region("completion_status")->offset;

    std::vector<cmd::Command> commands;
    commands.reserve(config.command_count);
    for (std::uint32_t index = 0; index < config.command_count; ++index) {
        try {
            commands.push_back(cmd::CommandCodec::decode(
                image.data() +
                    static_cast<std::size_t>(config.command_table_addr) +
                    index * cmd::COMMAND_RECORD_BYTES,
                cmd::COMMAND_RECORD_BYTES));
        } catch (const std::runtime_error& error) {
            fail("command record decode failed: " + std::string(error.what()));
        }
    }
    std::vector<cmd::Descriptor> descriptors;
    descriptors.reserve(config.descriptor_count);
    for (std::uint32_t index = 0; index < config.descriptor_count; ++index) {
        try {
            descriptors.push_back(cmd::DescriptorCodec::decode(
                image.data() +
                    static_cast<std::size_t>(config.descriptor_table_addr) +
                    index * cmd::DESCRIPTOR_RECORD_BYTES,
                cmd::DESCRIPTOR_RECORD_BYTES,
                config.total_bytes));
        } catch (const std::runtime_error& error) {
            fail("descriptor record decode failed: " + std::string(error.what()));
        }
    }
    std::vector<cmd::Completion> completions;
    completions.reserve(config.completion_count);
    for (std::uint32_t index = 0; index < config.completion_count; ++index) {
        try {
            completions.push_back(cmd::CompletionCodec::decode(
                image.data() +
                    static_cast<std::size_t>(config.completion_table_addr) +
                    index * cmd::COMPLETION_RECORD_BYTES,
                cmd::COMPLETION_RECORD_BYTES));
        } catch (const std::runtime_error& error) {
            fail("completion record decode failed: " + std::string(error.what()));
        }
    }

    try {
        cmd::validate_command_batch(
            commands, descriptors, image, config.token_count, config.max_wait_tokens);
    } catch (const std::runtime_error& error) {
        fail("command batch validation failed: " + std::string(error.what()));
    }

    for (const auto& descriptor : descriptors) {
        if (descriptor.descriptor_type == cmd::DescriptorType::Region) {
            const auto format = static_cast<cmd::DataFormat>(descriptor.body_words[7]);
            if (format != cmd::DataFormat::Fp32 && format != cmd::DataFormat::Int32) {
                fail("device image contains a non-current data format");
            }
        }
    }

    config.node_count = as_u32(
        require_key(require_object(manifest, "symbolic", "manifest"), "node_count", "symbolic"),
        "symbolic.node_count");

    ArtifactProbe probe;
    probe.image = &image;
    probe.descriptors = &descriptors;
    probe.commands = &commands;
    probe.config = &config;

    if (manifest.contains("verification")) {
        const json& verification = manifest.at("verification");
        if (verification.is_object() &&
            verification.contains("device_memory_contains_fp64_reference") &&
            verification.at("device_memory_contains_fp64_reference").get<bool>() != false) {
            fail("FP64 reference must not be device input");
        }
    }

    validate_descriptor_references(probe);
    validate_pivot_contract(probe, manifest, config);
    validate_command_semantics(probe, image);
    validate_nodes(manifest, probe);
    validate_completion_templates(commands, completions);

    if (manifest.contains("expected_completions")) {
        const json& expected = manifest.at("expected_completions");
        if (!expected.is_array() || expected.size() != completions.size()) {
            fail("expected_completions does not match the completion slots");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const json& entry = expected.at(index);
            if (as_u32(require_key(entry, "slot", "expected_completions"), "slot") != index ||
                as_u32(require_key(entry, "command_id", "expected_completions"), "command_id") !=
                    completions[index].command_id ||
                as_u32(require_key(entry, "node_id", "expected_completions"), "node_id") !=
                    completions[index].node_id) {
                fail("expected_completions entry does not match the completion template");
            }
        }
    }

    LoadedArtifact artifact;
    artifact.config = config;
    artifact.image = std::move(image);
    return artifact;
}

}  // namespace hw::dataplane
