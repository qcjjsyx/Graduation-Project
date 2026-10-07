#include "tree_artifact_check.hpp"
#include "fp64_reference.hpp"
#include "dataplane/artifact_loader.hpp"
#include "command_codec.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace hw::reference_fp64 {
namespace cmd = hw::command;
using json = nlohmann::json;

TreeValidation check_tree_artifact(const std::string& device_dir,
                                   const std::string& report_dir, double tolerance) {
    TreeValidation result;
    json node_results = json::array();
    try {
        if (!(tolerance > 0) || !std::isfinite(tolerance))
            throw ReferenceError("checker tolerance must be finite and positive");
        auto artifact = hw::dataplane::load_artifact(device_dir);
        std::ifstream input(device_dir + "/manifest.json"), summary(report_dir + "/summary.json");
        json manifest, device_status;
        input >> manifest;
        summary >> device_status;
        if (manifest.at("pivot").at("policy") != "STATIC")
            throw ReferenceError("T11 checker currently requires STATIC");
        if (!device_status.at("numeric_status_pass").get<bool>())
            throw ReferenceError("device reported a failed/incomplete numeric command batch");
        std::ifstream binary(report_dir + "/final_memory_image.bin", std::ios::binary);
        if (!binary) throw ReferenceError("device final memory image is missing");
        const std::vector<std::uint8_t> output{
            std::istreambuf_iterator<char>(binary), std::istreambuf_iterator<char>()};
        if (output.size() != artifact.image.size()) throw ReferenceError("output image size mismatch");
        const auto descriptor = [&](std::uint32_t id) {
            if (id >= artifact.config.descriptor_count) throw ReferenceError("invalid checker descriptor id");
            return cmd::DescriptorCodec::decode(
                artifact.image.data() + artifact.config.descriptor_table_addr + id * 64ull,
                64, artifact.image.size());
        };
        const auto address = [](const cmd::Descriptor& d) {
            return static_cast<std::uint64_t>(d.body_words[0]) |
                   (static_cast<std::uint64_t>(d.body_words[1]) << 32);
        };
        const auto read_matrix = [&](const std::vector<std::uint8_t>& image,
                                     std::uint32_t region, std::uint32_t rows,
                                     std::uint32_t cols) {
            const auto d = descriptor(region);
            DenseMatrix matrix(rows, cols);
            for (std::uint32_t i = 0; i < rows; ++i) {
                for (std::uint32_t j = 0; j < cols; ++j) {
                    const auto offset = address(d) + i * static_cast<std::uint64_t>(d.body_words[4]) + j * 4ull;
                    if (offset > image.size() || image.size() - offset < 4)
                        throw ReferenceError("checker matrix read exceeds image");
                    float value;
                    std::memcpy(&value, image.data() + offset, 4);
                    matrix.at(i, j) = value;
                }
            }
            return matrix;
        };
        const auto count = artifact.config.node_count;
        std::vector<FrontInput> fronts(count);
        std::vector<int> parent(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& node = manifest.at("nodes").at(std::to_string(i));
            const auto f = descriptor(node.at("descriptors").at("front"));
            fronts[i].variables = node.at("front_indices").get<std::vector<std::size_t>>();
            fronts[i].pivot_columns = f.body_words[2];
            fronts[i].local_front = read_matrix(artifact.image, f.body_words[0], f.body_words[1], f.body_words[1]);
            parent[i] = node.at("parent_id");
            for (const auto& id : node.at("descriptors").at("contributions")) {
                const auto c = descriptor(id);
                ContributionMap map;
                map.child = c.body_words[2];
                for (std::uint32_t j = 0; j < c.body_words[4] + c.body_words[5]; ++j) {
                    const auto offset = c.payload_offset + j * 4ull;
                    if (offset > artifact.image.size() || artifact.image.size() - offset < 4)
                        throw ReferenceError("checker map read exceeds image");
                    std::uint32_t value;
                    std::memcpy(&value, artifact.image.data() + offset, 4);
                    if (j < c.body_words[4]) map.row_map.push_back(value);
                    else map.col_map.push_back(value);
                }
                fronts[i].contributions.push_back(map);
            }
        }
        const auto expected = factor_front_tree(fronts, parent, PivotPolicy::Static,
                                                manifest.at("pivot").at("threshold_ratio"));
        bool pass = true;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& node = manifest.at("nodes").at(std::to_string(i));
            const auto factor = descriptor(node.at("descriptors").at("factor"));
            const auto& golden = expected[i].factorization;
            const auto compare = [&](const DenseMatrix& actual, const DenseMatrix& reference) {
                double max_error = 0, scale = 1;
                for (std::size_t j = 0; j < reference.values.size(); ++j) {
                    if (!std::isfinite(actual.values[j])) return std::numeric_limits<double>::infinity();
                    max_error = std::max(max_error, std::fabs(actual.values[j] - reference.values[j]));
                    scale = std::max(scale, std::fabs(reference.values[j]));
                }
                return max_error / scale;
            };
            const auto n = factor.body_words[3], p = factor.body_words[4], u = n - p;
            double error = std::max(
                compare(read_matrix(output, factor.body_words[0], n, p), golden.l),
                compare(read_matrix(output, factor.body_words[1], p, n), golden.u));
            if (u != 0) {
                const auto* region = artifact.config.region(node.at("regions").at("update"));
                if (!region) throw ReferenceError("update region missing");
                // Resolve by validated REGION_DESC base, supporting padded rows.
                bool found = false;
                for (std::uint32_t id = 0; id < artifact.config.descriptor_count; ++id) {
                    const auto d = descriptor(id);
                    if (d.descriptor_type == cmd::DescriptorType::Region && address(d) == region->offset) {
                        error = std::max(error, compare(read_matrix(output, id, u, u), golden.schur_update));
                        found = true;
                        break;
                    }
                }
                if (!found) throw ReferenceError("update descriptor missing");
            }
            const auto permutation = descriptor(factor.body_words[2]);
            bool identity = true;
            for (std::uint32_t j = 0; j < p; ++j) {
                std::int32_t value;
                std::memcpy(&value, output.data() + address(permutation) + j * 4ull, 4);
                identity = identity && value == static_cast<std::int32_t>(j);
            }
            const bool node_pass = error <= tolerance && identity;
            pass = pass && node_pass;
            result.max_relative_error = std::max(result.max_relative_error, error);
            ++result.nodes_checked;
            node_results.push_back({{"node_id", i}, {"relative_factor_update_error", error},
                                    {"identity_pvector", identity}, {"pass", node_pass}});
        }
        result.pass = pass;
        if (!pass) result.reason = "factor/update mismatch or invalid STATIC P-vector";
    } catch (const std::exception& error) {
        result.reason = error.what();
    }
    json report{{"checker", "independent_fp64_front_tree"}, {"numerical_validation", result.pass ? "passed" : "failed"},
                {"numeric_pass", result.pass}, {"nodes_checked", result.nodes_checked},
                {"max_relative_error", result.max_relative_error}, {"tolerance", tolerance},
                {"reason", result.reason}, {"nodes", node_results}, {"solve_validation", "not_performed"}};
    std::ofstream stream(report_dir + "/validation.json");
    stream << report.dump(2) << '\n';
    return result;
}
}
