#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "model_types.hpp"
#include "node_task_codec.hpp"

namespace hw {

struct MemoryRegion {
    std::uint64_t offset{0};
    std::uint64_t size{0};
};

struct ManifestNode {
    std::uint16_t node_id{0};
    std::uint32_t range_start{0};
    std::uint32_t range_end{0};
    std::vector<std::uint32_t> front_indices{};
    std::uint64_t reference_front_file_offset{0};
    MemoryRegion front_q{};
    MemoryRegion front_e{};
    MemoryRegion update_q{};
    MemoryRegion update_e{};
    MemoryRegion l_factor{};
    MemoryRegion u_factor{};
    MemoryRegion map_table{};
    MemoryRegion p_vector{};
    MemoryRegion node_meta{};
    MemoryRegion solve_workspace{};
};

struct ArtifactManifest {
    std::filesystem::path manifest_path{};
    std::filesystem::path directory{};
    std::filesystem::path memory_image_path{};
    std::filesystem::path reference_front_path{};
    std::filesystem::path rhs_reference_path{};
    std::filesystem::path original_matrix_reference_path{};
    std::filesystem::path original_rhs_reference_path{};
    std::filesystem::path original_solution_reference_path{};
    std::filesystem::path row_scale_exponents_path{};
    std::filesystem::path column_scale_exponents_path{};
    std::filesystem::path solution_reference_path{};
    std::string equilibration_mode{"none"};
    bool solution_requires_unscale{false};
    std::uint32_t matrix_dim{0};
    std::uint32_t node_count{0};
    std::uint64_t total_bytes{0};
    std::uint32_t alignment{64};
    std::int32_t q_limit{0};
    std::uint32_t bfp_tile_size{0};
    std::vector<std::int32_t> parent{};
    std::vector<std::uint32_t> permutation{};
    std::vector<std::uint16_t> task_order{};
    std::map<std::string, MemoryRegion> global_regions{};
    std::vector<ManifestNode> nodes{};
};

inline MemoryRegion parse_region(
    const nlohmann::json& value,
    const std::string& label) {
    if (!value.is_object() || !value.contains("offset") || !value.contains("size")) {
        throw std::runtime_error(label + " must contain offset and size");
    }
    MemoryRegion region{
        value.at("offset").get<std::uint64_t>(),
        value.at("size").get<std::uint64_t>(),
    };
    if (region.offset > std::numeric_limits<std::uint64_t>::max() - region.size) {
        throw std::runtime_error(label + " address range overflows uint64");
    }
    return region;
}

inline std::vector<std::uint8_t> read_binary_file(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open binary file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot determine file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
    if (!data.empty() &&
        !input.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("failed to read binary file: " + path.string());
    }
    return data;
}

inline void write_binary_file(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& data) {
    std::ofstream output(path, std::ios::binary);
    if (!output ||
        (!data.empty() &&
         !output.write(
             reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size())))) {
        throw std::runtime_error("cannot write binary file: " + path.string());
    }
}

inline ArtifactManifest load_manifest(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open manifest: " + path.string());
    }
    nlohmann::json json;
    input >> json;

    const auto& abi = json.at("abi");
    if (abi.at("version").get<unsigned>() != 2u ||
        abi.at("node_task_byte_size").get<unsigned>() !=
            NodeTaskCodec::RECORD_SIZE ||
        abi.at("endianness").get<std::string>() != "little") {
        throw std::runtime_error(
            "unsupported artifact ABI (requires little-endian v2, 128-byte NodeTask)");
    }

    ArtifactManifest manifest{};
    manifest.manifest_path = std::filesystem::absolute(path);
    manifest.directory = manifest.manifest_path.parent_path();
    manifest.matrix_dim = json.at("matrix").at("n").get<std::uint32_t>();
    manifest.node_count =
        json.at("symbolic").at("node_count").get<std::uint32_t>();
    manifest.total_bytes = json.at("total_bytes").get<std::uint64_t>();
    manifest.alignment =
        json.at("config").at("memory").at("alignment").get<std::uint32_t>();
    manifest.q_limit = json.at("quantization").at("q_limit").get<std::int32_t>();
    manifest.bfp_tile_size =
        json.at("quantization").value("bfp_tile_size", 0u);
    if (manifest.matrix_dim == 0 || manifest.node_count == 0 ||
        manifest.q_limit <= 0 ||
        (manifest.bfp_tile_size != 0 && manifest.bfp_tile_size != 16)) {
        throw std::runtime_error("manifest dimensions and q_limit must be positive");
    }

    manifest.parent =
        json.at("symbolic").at("parent").get<std::vector<std::int32_t>>();
    manifest.permutation =
        json.at("symbolic").at("permutation").get<std::vector<std::uint32_t>>();
    const auto raw_order =
        json.at("task_order").get<std::vector<std::uint32_t>>();
    if (manifest.parent.size() != manifest.node_count ||
        manifest.permutation.size() != manifest.matrix_dim ||
        raw_order.size() != manifest.node_count) {
        throw std::runtime_error("manifest array length mismatch");
    }
    manifest.task_order.reserve(raw_order.size());
    for (const auto node_id : raw_order) {
        if (node_id > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("task order node id exceeds uint16");
        }
        manifest.task_order.push_back(static_cast<std::uint16_t>(node_id));
    }

    const auto& memory = json.at("memory_image");
    manifest.memory_image_path =
        manifest.directory / memory.at("file").get<std::string>();
    if (memory.at("size").get<std::uint64_t>() != manifest.total_bytes) {
        throw std::runtime_error("memory image size metadata mismatch");
    }
    for (auto it = memory.at("global_regions").begin();
         it != memory.at("global_regions").end(); ++it) {
        manifest.global_regions.emplace(
            it.key(), parse_region(it.value(), "global region " + it.key()));
    }

    const auto& verification = json.at("verification");
    manifest.reference_front_path =
        manifest.directory /
        verification.at("reference_front_file").get<std::string>();
    manifest.rhs_reference_path =
        manifest.directory /
        verification.at("rhs_reference_file").get<std::string>();
    manifest.solution_reference_path =
        manifest.directory /
        verification.at("solution_reference_file").get<std::string>();
    manifest.original_matrix_reference_path =
        manifest.directory /
        verification.at("original_matrix_reference_file").get<std::string>();
    manifest.original_rhs_reference_path =
        manifest.directory /
        verification.at("original_rhs_reference_file").get<std::string>();
    manifest.original_solution_reference_path =
        manifest.directory /
        verification.at("original_solution_reference_file").get<std::string>();
    const auto& equilibration = json.at("equilibration");
    manifest.equilibration_mode =
        equilibration.at("mode").get<std::string>();
    if (manifest.equilibration_mode != "none" &&
        manifest.equilibration_mode != "pow2-row" &&
        manifest.equilibration_mode != "pow2-row-column" &&
        manifest.equilibration_mode != "pow2-ruiz") {
        throw std::runtime_error("unsupported equilibration mode");
    }
    manifest.solution_requires_unscale =
        equilibration.at("solution_requires_unscale").get<bool>();
    const auto expected_unscale =
        manifest.equilibration_mode == "pow2-row-column" ||
        manifest.equilibration_mode == "pow2-ruiz";
    if (manifest.solution_requires_unscale != expected_unscale) {
        throw std::runtime_error(
            "equilibration solution-unscale metadata mismatch");
    }
    if (equilibration.at("row_scale_exponent_count").get<std::uint32_t>() !=
        manifest.matrix_dim) {
        throw std::runtime_error("row scale exponent count mismatch");
    }
    manifest.row_scale_exponents_path =
        manifest.directory /
        equilibration.at("row_scale_exponent_file").get<std::string>();
    if (equilibration.at("column_scale_exponent_count")
            .get<std::uint32_t>() != manifest.matrix_dim) {
        throw std::runtime_error("column scale exponent count mismatch");
    }
    manifest.column_scale_exponents_path =
        manifest.directory /
        equilibration.at("column_scale_exponent_file").get<std::string>();

    const auto& nodes_json = json.at("nodes");
    manifest.nodes.resize(manifest.node_count);
    for (std::uint32_t node_id = 0; node_id < manifest.node_count; ++node_id) {
        const auto key = std::to_string(node_id);
        if (!nodes_json.contains(key)) {
            throw std::runtime_error("manifest is missing node " + key);
        }
        const auto& source = nodes_json.at(key);
        ManifestNode node{};
        node.node_id = static_cast<std::uint16_t>(node_id);
        node.range_start = source.at("range").at("start").get<std::uint32_t>();
        node.range_end = source.at("range").at("end").get<std::uint32_t>();
        node.front_indices =
            source.at("front_indices").get<std::vector<std::uint32_t>>();
        node.reference_front_file_offset =
            source.at("reference_front_file_offset").get<std::uint64_t>();
        node.front_q = parse_region(source.at("front_q"), key + ".front_q");
        node.front_e = parse_region(source.at("front_e"), key + ".front_e");
        node.update_q = parse_region(source.at("update_q"), key + ".update_q");
        node.update_e = parse_region(source.at("update_e"), key + ".update_e");
        node.l_factor = parse_region(source.at("l_factor"), key + ".l_factor");
        node.u_factor = parse_region(source.at("u_factor"), key + ".u_factor");
        node.map_table =
            parse_region(source.at("map_table"), key + ".map_table");
        node.p_vector =
            parse_region(source.at("p_vector"), key + ".p_vector");
        node.node_meta =
            parse_region(source.at("node_meta"), key + ".node_meta");
        node.solve_workspace =
            parse_region(source.at("solve_workspace"), key + ".solve_workspace");
        if (node.range_end <= node.range_start ||
            node.range_end > manifest.matrix_dim ||
            node.front_indices.size() < node.range_end - node.range_start) {
            throw std::runtime_error("invalid node range/front dimensions for node " + key);
        }
        manifest.nodes[node_id] = std::move(node);
    }

    return manifest;
}

inline void validate_regions(const ArtifactManifest& manifest) {
    struct NamedRange {
        std::uint64_t start;
        std::uint64_t end;
        std::string name;
    };
    std::vector<NamedRange> ranges;
    const auto append = [&](const MemoryRegion& region, const std::string& name) {
        if (region.offset % manifest.alignment != 0) {
            throw std::runtime_error(name + " is not aligned");
        }
        if (region.offset + region.size > manifest.total_bytes) {
            throw std::runtime_error(name + " exceeds DDR image");
        }
        if (region.size != 0) {
            ranges.push_back({region.offset, region.offset + region.size, name});
        }
    };

    for (const auto& [name, region] : manifest.global_regions) {
        append(region, "global." + name);
    }
    for (const auto& node : manifest.nodes) {
        const auto prefix = "node." + std::to_string(node.node_id) + ".";
        append(node.front_q, prefix + "front_q");
        append(node.front_e, prefix + "front_e");
        append(node.update_q, prefix + "update_q");
        append(node.update_e, prefix + "update_e");
        append(node.l_factor, prefix + "l_factor");
        append(node.u_factor, prefix + "u_factor");
        append(node.map_table, prefix + "map_table");
        append(node.p_vector, prefix + "p_vector");
        append(node.node_meta, prefix + "node_meta");
        append(node.solve_workspace, prefix + "solve_workspace");
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.start < rhs.start;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i - 1].end > ranges[i].start) {
            throw std::runtime_error(
                "DDR regions overlap: " + ranges[i - 1].name +
                " and " + ranges[i].name);
        }
    }

    const auto require_global =
        [&](const std::string& name, std::uint64_t expected_size) {
            const auto it = manifest.global_regions.find(name);
            if (it == manifest.global_regions.end() ||
                it->second.size != expected_size) {
                throw std::runtime_error(
                    "global." + name + " has an invalid size");
            }
        };
    require_global(
        "task_queue", manifest.node_count * NodeTaskCodec::RECORD_SIZE);
    require_global("permutation", manifest.matrix_dim * sizeof(std::uint32_t));
    require_global("rhs_q", manifest.matrix_dim * sizeof(std::int32_t));
    require_global("rhs_e", sizeof(std::int16_t));
    require_global("solution_q", manifest.matrix_dim * sizeof(std::int64_t));
    require_global("solution_e", manifest.node_count * sizeof(std::int16_t));

    std::set<std::uint32_t> permutation(
        manifest.permutation.begin(), manifest.permutation.end());
    if (permutation.size() != manifest.matrix_dim ||
        *permutation.begin() != 0 ||
        *permutation.rbegin() != manifest.matrix_dim - 1) {
        throw std::runtime_error("symbolic permutation is not a bijection");
    }

    std::size_t roots = 0;
    const auto exponent_count =
        [&](std::uint32_t rows, std::uint32_t cols) -> std::uint64_t {
            if (rows == 0 || cols == 0) return 0;
            if (manifest.bfp_tile_size == 0) return 1;
            const auto tile = manifest.bfp_tile_size;
            return static_cast<std::uint64_t>((rows + tile - 1) / tile) *
                   ((cols + tile - 1) / tile);
        };
    for (std::uint32_t node_id = 0;
         node_id < manifest.node_count; ++node_id) {
        const auto& node = manifest.nodes[node_id];
        const auto pivot = node.range_end - node.range_start;
        const auto total = static_cast<std::uint32_t>(node.front_indices.size());
        const auto update = total - pivot;
        if (node.front_q.size !=
                static_cast<std::uint64_t>(total) * total * sizeof(std::int32_t) ||
            node.front_e.size !=
                exponent_count(total, total) * sizeof(std::int16_t) ||
            node.update_q.size !=
                static_cast<std::uint64_t>(update) * update *
                    sizeof(std::int32_t) ||
            node.update_e.size !=
                exponent_count(update, update) * sizeof(std::int16_t) ||
            node.l_factor.size !=
                static_cast<std::uint64_t>(total) * pivot *
                    sizeof(std::int32_t) ||
            node.u_factor.size !=
                static_cast<std::uint64_t>(pivot) * total *
                    sizeof(std::int32_t) ||
            node.p_vector.size !=
                static_cast<std::uint64_t>(pivot) * sizeof(std::uint16_t) ||
            node.node_meta.size <
                std::max<std::uint64_t>(
                    64,
                    16 + exponent_count(pivot, total) *
                        sizeof(std::int16_t))) {
            throw std::runtime_error(
                "node region sizes disagree with front dimensions for node " +
                std::to_string(node_id));
        }
        const auto parent = manifest.parent[node_id];
        if (parent < 0) {
            ++roots;
        } else if (parent >= static_cast<std::int32_t>(manifest.node_count) ||
                   parent == static_cast<std::int32_t>(node_id)) {
            throw std::runtime_error("invalid parent id in elimination forest");
        }
        std::set<std::uint32_t> indices(
            node.front_indices.begin(), node.front_indices.end());
        if (indices.size() != node.front_indices.size() ||
            (!indices.empty() && *indices.rbegin() >= manifest.matrix_dim)) {
            throw std::runtime_error("front index list is invalid");
        }
    }
    if (roots == 0) {
        throw std::runtime_error("elimination forest has no root");
    }
    for (std::uint32_t origin = 0;
         origin < manifest.node_count; ++origin) {
        std::set<std::uint32_t> visited;
        auto cursor = static_cast<std::int32_t>(origin);
        while (cursor >= 0) {
            if (!visited.insert(static_cast<std::uint32_t>(cursor)).second) {
                throw std::runtime_error("elimination forest contains a cycle");
            }
            cursor = manifest.parent.at(static_cast<std::size_t>(cursor));
        }
    }
}

class DdrMemory {
public:
    DdrMemory(
        std::vector<std::uint8_t> image,
        const ModelConfig& config,
        SimulationStats& stats)
        : image_(std::move(image)),
          config_(config),
          stats_(stats),
          rng_(config.seed) {}

    std::size_t size() const { return image_.size(); }

    const std::vector<std::uint8_t>& bytes() const { return image_; }

    std::vector<std::uint8_t> read_bytes(
        std::uint64_t address,
        std::size_t size,
        bool account = true) {
        check_range(address, size);
        if (account) {
            account_transaction(size, false);
        }
        return std::vector<std::uint8_t>(
            image_.begin() + static_cast<std::ptrdiff_t>(address),
            image_.begin() + static_cast<std::ptrdiff_t>(address + size));
    }

    std::uint64_t write_bytes(
        std::uint64_t address,
        const std::vector<std::uint8_t>& data,
        bool account = true) {
        check_range(address, data.size());
        std::uint64_t cycles = 0;
        if (account) {
            cycles = account_transaction(data.size(), true);
        }
        std::copy(data.begin(), data.end(),
                  image_.begin() + static_cast<std::ptrdiff_t>(address));
        return cycles;
    }

    std::uint64_t account_read(std::size_t size) {
        return account_transaction(size, false);
    }

    std::uint64_t account_write(std::size_t size) {
        return account_transaction(size, true);
    }

    std::int16_t read_i16(std::uint64_t address, bool account = true) {
        const auto data = read_bytes(address, 2, account);
        return static_cast<std::int16_t>(
            static_cast<std::uint16_t>(data[0]) |
            (static_cast<std::uint16_t>(data[1]) << 8));
    }

    std::vector<std::int16_t> read_i16_vector(
        std::uint64_t address,
        std::size_t count,
        bool account = true) {
        const auto data = read_bytes(address, count * 2, account);
        std::vector<std::int16_t> output(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto raw =
                static_cast<std::uint16_t>(data[i * 2]) |
                (static_cast<std::uint16_t>(data[i * 2 + 1]) << 8);
            output[i] = static_cast<std::int16_t>(raw);
        }
        return output;
    }

    std::vector<std::int32_t> read_i32_vector(
        std::uint64_t address,
        std::size_t count,
        bool account = true) {
        const auto data = read_bytes(address, count * 4, account);
        std::vector<std::int32_t> output(count);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t raw = 0;
            for (unsigned byte = 0; byte < 4; ++byte) {
                raw |= static_cast<std::uint32_t>(data[i * 4 + byte]) << (8 * byte);
            }
            output[i] = static_cast<std::int32_t>(raw);
        }
        return output;
    }

    std::uint64_t write_i16(std::uint64_t address, std::int16_t value) {
        const auto raw = static_cast<std::uint16_t>(value);
        return write_bytes(address, {
            static_cast<std::uint8_t>(raw & 0xffu),
            static_cast<std::uint8_t>((raw >> 8) & 0xffu),
        });
    }

    std::uint64_t write_i32_vector(
        std::uint64_t address,
        const std::vector<std::int32_t>& values) {
        std::vector<std::uint8_t> data(values.size() * 4);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto raw = static_cast<std::uint32_t>(values[i]);
            for (unsigned byte = 0; byte < 4; ++byte) {
                data[i * 4 + byte] =
                    static_cast<std::uint8_t>((raw >> (8 * byte)) & 0xffu);
            }
        }
        return write_bytes(address, data);
    }

    std::uint64_t write_u16_vector(
        std::uint64_t address,
        const std::vector<std::uint16_t>& values) {
        std::vector<std::uint8_t> data(values.size() * 2);
        for (std::size_t i = 0; i < values.size(); ++i) {
            data[i * 2] = static_cast<std::uint8_t>(values[i] & 0xffu);
            data[i * 2 + 1] =
                static_cast<std::uint8_t>((values[i] >> 8) & 0xffu);
        }
        return write_bytes(address, data);
    }

    std::uint64_t write_i16_vector(
        std::uint64_t address,
        const std::vector<std::int16_t>& values) {
        std::vector<std::uint8_t> data(values.size() * 2);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto raw = static_cast<std::uint16_t>(values[i]);
            data[i * 2] = static_cast<std::uint8_t>(raw & 0xffu);
            data[i * 2 + 1] =
                static_cast<std::uint8_t>((raw >> 8) & 0xffu);
        }
        return write_bytes(address, data);
    }

    std::uint64_t write_i64_vector(
        std::uint64_t address,
        const std::vector<std::int64_t>& values) {
        std::vector<std::uint8_t> data(values.size() * 8);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto raw = static_cast<std::uint64_t>(values[i]);
            for (unsigned byte = 0; byte < 8; ++byte) {
                data[i * 8 + byte] =
                    static_cast<std::uint8_t>((raw >> (8 * byte)) & 0xffu);
            }
        }
        return write_bytes(address, data);
    }

    std::uint64_t write_f64_vector(
        std::uint64_t address,
        const std::vector<double>& values) {
        std::vector<std::uint8_t> data(values.size() * 8);
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::uint64_t raw = 0;
            std::memcpy(&raw, &values[i], sizeof(double));
            for (unsigned byte = 0; byte < 8; ++byte) {
                data[i * 8 + byte] =
                    static_cast<std::uint8_t>((raw >> (8 * byte)) & 0xffu);
            }
        }
        return write_bytes(address, data);
    }

private:
    std::vector<std::uint8_t> image_;
    const ModelConfig& config_;
    SimulationStats& stats_;
    std::mt19937_64 rng_;
    std::uint64_t data_bus_available_cycle_{0};

    void check_range(std::uint64_t address, std::size_t size) const {
        if (address > image_.size() || size > image_.size() - address) {
            throw std::out_of_range("DDR access is outside memory image");
        }
    }

    std::uint64_t account_transaction(std::size_t size, bool write) {
        if (size == 0) {
            return 0;
        }
        const auto bursts =
            (size + config_.ddr_burst_bytes - 1) / config_.ddr_burst_bytes;
        const auto waves =
            (bursts + config_.ddr_outstanding - 1) / config_.ddr_outstanding;
        std::uint64_t latency = waves * config_.ddr_base_latency;
        const auto transfer_cycles =
            (size + config_.ddr_bytes_per_cycle - 1) /
            config_.ddr_bytes_per_cycle;
        if (config_.ddr_jitter_cycles != 0) {
            std::uniform_int_distribution<unsigned> jitter(
                0, config_.ddr_jitter_cycles);
            latency += jitter(rng_);
        }
        if (config_.backpressure_probability > 0.0) {
            std::binomial_distribution<std::uint64_t> stalls(
                std::max<std::uint64_t>(latency + transfer_cycles, 1),
                std::min(config_.backpressure_probability, 1.0));
            latency += stalls(rng_);
        }
        const auto data_ready = stats_.cycle + latency;
        const auto transfer_start =
            std::max(data_ready, data_bus_available_cycle_);
        const auto completion = transfer_start + transfer_cycles;
        data_bus_available_cycle_ = completion;
        const auto cycles = completion - stats_.cycle;

        if (write) {
            stats_.memory.write_bytes += size;
            stats_.memory.write_transactions += bursts;
            stats_.memory.write_cycles += cycles;
        } else {
            stats_.memory.read_bytes += size;
            stats_.memory.read_transactions += bursts;
            stats_.memory.read_cycles += cycles;
        }
        return cycles;
    }
};

inline ModelConfig load_model_config(
    const std::optional<std::filesystem::path>& path) {
    ModelConfig config{};
    if (!path) {
        return config;
    }
    std::ifstream input(*path);
    if (!input) {
        throw std::runtime_error("cannot open simulation config: " + path->string());
    }
    nlohmann::json json;
    input >> json;
    const auto assign_u = [&](const char* key, unsigned& target) {
        if (json.contains(key)) target = json.at(key).get<unsigned>();
    };
    const auto assign_u64 = [&](const char* key, std::uint64_t& target) {
        if (json.contains(key)) target = json.at(key).get<std::uint64_t>();
    };
    const auto assign_bool = [&](const char* key, bool& target) {
        if (json.contains(key)) target = json.at(key).get<bool>();
    };
    assign_u("clock_period_ns", config.clock_period_ns);
    assign_u("tile_size", config.tile_size);
    assign_u("bfp_tile_size", config.bfp_tile_size);
    assign_u("buffer_count", config.buffer_count);
    assign_u64("buffer_capacity_bytes", config.buffer_capacity_bytes);
    assign_u("fifo_depth", config.fifo_depth);
    assign_u("ddr_base_latency", config.ddr_base_latency);
    assign_u("ddr_bytes_per_cycle", config.ddr_bytes_per_cycle);
    assign_u("ddr_burst_bytes", config.ddr_burst_bytes);
    assign_u("ddr_outstanding", config.ddr_outstanding);
    assign_u("ddr_jitter_cycles", config.ddr_jitter_cycles);
    if (json.contains("backpressure_probability")) {
        config.backpressure_probability =
            json.at("backpressure_probability").get<double>();
    }
    assign_u("panel_units", config.panel_units);
    assign_u("panel_startup", config.panel_startup);
    assign_u("panel_ops_per_cycle", config.panel_ops_per_cycle);
    assign_u("trsm_units", config.trsm_units);
    assign_u("trsm_startup", config.trsm_startup);
    assign_u("trsm_macs_per_cycle", config.trsm_macs_per_cycle);
    assign_u("gemm_units", config.gemm_units);
    assign_u("gemm_startup", config.gemm_startup);
    assign_u("gemm_macs_per_cycle", config.gemm_macs_per_cycle);
    assign_u("writeback_latency", config.writeback_latency);
    assign_u64("timeout_cycles", config.timeout_cycles);
    assign_u("q_use_bits", config.q_use_bits);
    assign_u("frac_bits", config.frac_bits);
    assign_bool("adaptive_frac_retry", config.adaptive_frac_retry);
    assign_u("retry_frac_bits", config.retry_frac_bits);
    assign_u("accumulator_bits", config.accumulator_bits);
    assign_u("workspace_guard_bits", config.workspace_guard_bits);
    assign_u("vector_use_bits", config.vector_use_bits);
    assign_bool("adaptive_factor_scaling", config.adaptive_factor_scaling);
    if (json.contains("fixed_pivot_rel_tol")) {
        config.fixed_pivot_rel_tol =
            json.at("fixed_pivot_rel_tol").get<double>();
    }
    if (json.contains("fixed_factor_rel_tol")) {
        config.fixed_factor_rel_tol =
            json.at("fixed_factor_rel_tol").get<double>();
    }
    if (json.contains("fixed_rescue_mode")) {
        config.fixed_rescue_mode =
            json.at("fixed_rescue_mode").get<std::string>();
    }
    if (json.contains("rescue_pivot_rel_tol")) {
        config.rescue_pivot_rel_tol =
            json.at("rescue_pivot_rel_tol").get<double>();
    }
    assign_u("precision_rescue_startup", config.precision_rescue_startup);
    assign_u(
        "precision_rescue_macs_per_cycle",
        config.precision_rescue_macs_per_cycle);
    assign_bool("iterative_refinement", config.iterative_refinement);
    assign_u("ir_max_iters", config.ir_max_iters);
    if (json.contains("ir_tolerance")) {
        config.ir_tolerance = json.at("ir_tolerance").get<double>();
    }
    if (json.contains("ir_min_improvement")) {
        config.ir_min_improvement =
            json.at("ir_min_improvement").get<double>();
    }
    assign_u(
        "ir_residual_macs_per_cycle",
        config.ir_residual_macs_per_cycle);
    if (json.contains("pivot_rel_tol")) {
        config.pivot_rel_tol = json.at("pivot_rel_tol").get<double>();
    }
    if (json.contains("scheduler_policy")) {
        config.scheduler_policy = json.at("scheduler_policy").get<std::string>();
    }
    assign_u64("seed", config.seed);

    if (config.clock_period_ns == 0 ||
        config.tile_size == 0 || config.buffer_count == 0 ||
        (config.bfp_tile_size != 0 && config.bfp_tile_size != 16) ||
        config.fifo_depth == 0 || config.ddr_bytes_per_cycle == 0 ||
        config.ddr_burst_bytes == 0 || config.ddr_outstanding == 0 ||
        config.panel_units == 0 || config.panel_ops_per_cycle == 0 ||
        config.trsm_units == 0 || config.trsm_macs_per_cycle == 0 ||
        config.gemm_units == 0 || config.gemm_macs_per_cycle == 0 ||
        config.q_use_bits == 0 || config.q_use_bits > 30 ||
        config.frac_bits > 30 ||
        config.retry_frac_bits > 30 ||
        (config.adaptive_frac_retry &&
         (config.bfp_tile_size == 0 ||
          config.retry_frac_bits <= config.frac_bits)) ||
        config.accumulator_bits < 32 || config.accumulator_bits > 64 ||
        config.workspace_guard_bits > 24 ||
        config.q_use_bits + config.workspace_guard_bits > 61 ||
        config.vector_use_bits == 0 || config.vector_use_bits > 62 ||
        config.fixed_pivot_rel_tol < 0.0 ||
        config.fixed_factor_rel_tol < 0.0 ||
        config.rescue_pivot_rel_tol < 0.0 ||
        (config.fixed_rescue_mode != "fail" &&
         config.fixed_rescue_mode != "fp64") ||
        config.precision_rescue_macs_per_cycle == 0 ||
        config.ir_tolerance < 0.0 ||
        config.ir_min_improvement < 0.0 ||
        config.ir_min_improvement >= 1.0 ||
        config.ir_residual_macs_per_cycle == 0 ||
        config.backpressure_probability < 0.0 ||
        config.backpressure_probability > 1.0 ||
        (config.scheduler_policy != "serial" &&
         config.scheduler_policy != "resource_aware")) {
        throw std::runtime_error("simulation config contains an invalid value");
    }
    return config;
}

inline std::vector<NodeTask> decode_task_queue(
    const ArtifactManifest& manifest,
    const std::vector<std::uint8_t>& image) {
    const auto it = manifest.global_regions.find("task_queue");
    if (it == manifest.global_regions.end()) {
        throw std::runtime_error("manifest has no task_queue global region");
    }
    const auto& region = it->second;
    if (region.size != manifest.node_count * NodeTaskCodec::RECORD_SIZE ||
        region.offset + region.size > image.size()) {
        throw std::runtime_error("invalid task queue region");
    }
    std::vector<NodeTask> tasks;
    tasks.reserve(manifest.node_count);
    std::set<std::uint16_t> ids;
    for (std::uint32_t index = 0; index < manifest.node_count; ++index) {
        NodeTaskCodec::Record record{};
        const auto offset =
            region.offset + index * NodeTaskCodec::RECORD_SIZE;
        std::copy_n(image.begin() + static_cast<std::ptrdiff_t>(offset),
                    NodeTaskCodec::RECORD_SIZE, record.begin());
        auto task = NodeTaskCodec::decode(record);
        if (task.node_id >= manifest.node_count || !ids.insert(task.node_id).second) {
            throw std::runtime_error("task queue contains invalid or duplicate node id");
        }
        const auto& node = manifest.nodes[task.node_id];
        if (task.node_id != manifest.task_order[index]) {
            throw std::runtime_error(
                "task queue order disagrees with manifest task_order");
        }
        std::uint16_t expected_children = 0;
        for (const auto parent : manifest.parent) {
            if (parent == task.node_id) ++expected_children;
        }
        const auto expected_flags =
            static_cast<std::uint16_t>(
                (expected_children == 0 ? 1u : 0u) |
                (manifest.parent[task.node_id] < 0 ? 2u : 0u));
        if (task.total_dim != node.front_indices.size() ||
            task.pivot_dim != node.range_end - node.range_start ||
            task.children_count != expected_children ||
            task.flags != expected_flags ||
            task.front_q_addr != node.front_q.offset ||
            task.front_e_addr != node.front_e.offset ||
            task.update_q_addr != node.update_q.offset ||
            task.update_e_addr != node.update_e.offset ||
            task.map_table_addr != node.map_table.offset ||
            task.map_table_bytes != node.map_table.size ||
            task.l_factor_addr != node.l_factor.offset ||
            task.u_factor_addr != node.u_factor.offset ||
            task.p_vector_addr != node.p_vector.offset ||
            task.node_meta_addr != node.node_meta.offset ||
            task.solve_workspace_addr != node.solve_workspace.offset) {
            throw std::runtime_error(
                "NodeTask fields disagree with manifest for node " +
                std::to_string(task.node_id));
        }
        const auto parent = manifest.parent[task.node_id];
        if ((parent < 0 && task.parent_id != ROOT_PARENT_ID) ||
            (parent >= 0 && task.parent_id != static_cast<std::uint16_t>(parent))) {
            throw std::runtime_error("NodeTask parent does not match manifest");
        }
        tasks.push_back(task);
    }
    return tasks;
}

}  // namespace hw
