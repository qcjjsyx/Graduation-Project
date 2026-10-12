#include "tree_solve_check.hpp"
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
#include <sstream>
#include <vector>

namespace hw::reference_fp64 {
namespace cmd = hw::command;
using json = nlohmann::json;

namespace {

// Thrown after `reason` is already set; the catch below must not overwrite it.
struct StatusGateError : std::exception {};

double ratio_or_zero(double numerator, double denominator) {
    if (denominator != 0.0) {
        return numerator / denominator;
    }
    return numerator == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

float read_f32(const std::vector<std::uint8_t>& image, std::uint64_t offset) {
    if (offset + 4 > image.size()) {
        throw ReferenceError("solve checker read exceeds the memory image");
    }
    float value = 0.0f;
    std::memcpy(&value, image.data() + offset, 4);
    return value;
}

std::int32_t read_i32(const std::vector<std::uint8_t>& image, std::uint64_t offset) {
    if (offset + 4 > image.size()) {
        throw ReferenceError("solve checker read exceeds the memory image");
    }
    std::int32_t value = 0;
    std::memcpy(&value, image.data() + offset, 4);
    return value;
}

std::uint64_t region_base(const cmd::Descriptor& descriptor) {
    return static_cast<std::uint64_t>(descriptor.body_words[0]) |
           (static_cast<std::uint64_t>(descriptor.body_words[1]) << 32);
}

double read_f64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    double value = 0.0;
    std::memcpy(&value, bytes.data() + offset, 8);
    return value;
}

// commands.csv: fixed header; every field before `status` is unquoted.
void scan_solve_command_status(
    const std::string& path,
    std::uint32_t* solve_commands,
    std::uint32_t* solve_ok) {
    std::ifstream stream(path);
    if (!stream) {
        throw ReferenceError("device commands.csv is missing");
    }
    std::string line;
    std::getline(stream, line);  // header
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        std::string field;
        bool quoted = false;
        for (const char c : line) {
            if (c == '"') {
                quoted = !quoted;
            } else if (c == ',' && !quoted) {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
        fields.push_back(field);
        if (fields.size() < 11) {
            throw ReferenceError("commands.csv row has too few fields");
        }
        const std::string& opcode = fields[2];
        const std::string& status = fields[10];
        const bool is_solve =
            opcode == std::to_string(
                          static_cast<std::uint32_t>(cmd::Opcode::SolveForward)) ||
            opcode == std::to_string(
                          static_cast<std::uint32_t>(cmd::Opcode::SolveBackward));
        if (!is_solve) {
            continue;
        }
        ++*solve_commands;
        if (status == "OK") {
            ++*solve_ok;
        }
    }
}

}  // namespace

TriangularBackwardErrors triangular_backward_errors(
    const std::vector<SolveNodeData>& data,
    const std::vector<double>& b_hat,
    const std::vector<double>& x_hat) {
    const auto dimension = static_cast<std::uint32_t>(b_hat.size());
    TriangularBackwardErrors result;
    // (1) forward backward error: assemble L~ y row by row.
    std::vector<double> lhs(dimension, 0.0);
    std::vector<double> lhs_abs(dimension, 0.0);
    for (const auto& node : data) {
        const std::uint32_t total = node.total;
        const std::uint32_t pivot = node.pivot;
        for (std::uint32_t row = 0; row < total; ++row) {
            std::uint32_t source_position = row;
            if (row < pivot) {
                const auto position = node.p[row];
                if (position < 0 ||
                    static_cast<std::uint32_t>(position) >= pivot) {
                    throw ReferenceError(
                        "P-vector is not a permutation inside the pivot block");
                }
                source_position = static_cast<std::uint32_t>(position);
            }
            const std::uint32_t target = node.variables[source_position];
            double sum = 0.0;
            double sum_abs = 0.0;
            for (std::uint32_t col = 0; col < pivot; ++col) {
                const double coefficient =
                    node.l[static_cast<std::size_t>(row) * pivot + col];
                sum += coefficient * node.z[col];
                sum_abs += std::fabs(coefficient) * std::fabs(node.z[col]);
            }
            lhs[target] += sum;
            lhs_abs[target] += sum_abs;
        }
    }
    double eta_forward = 0.0;
    for (std::uint32_t index = 0; index < dimension; ++index) {
        const double residual = b_hat[index] - lhs[index];
        eta_forward = std::max(
            eta_forward,
            ratio_or_zero(
                std::fabs(residual), lhs_abs[index] + std::fabs(b_hat[index])));
    }
    result.eta_forward = eta_forward;

    // (2) backward backward error: y[g[i]] - U_g x_hat, per pivot row.
    double eta_backward = 0.0;
    for (const auto& node : data) {
        const std::uint32_t total = node.total;
        const std::uint32_t pivot = node.pivot;
        for (std::uint32_t row = 0; row < pivot; ++row) {
            double product = 0.0;
            double product_abs = 0.0;
            for (std::uint32_t col = 0; col < total; ++col) {
                const double coefficient =
                    node.u[static_cast<std::size_t>(row) * total + col];
                product += coefficient * x_hat[node.variables[col]];
                product_abs += std::fabs(coefficient) *
                               std::fabs(x_hat[node.variables[col]]);
            }
            const double y = node.z[row];
            const double residual = y - product;
            eta_backward = std::max(
                eta_backward,
                ratio_or_zero(
                    std::fabs(residual), product_abs + std::fabs(y)));
        }
    }
    result.eta_backward = eta_backward;
    return result;
}

bool triangular_errors_within(const TriangularBackwardErrors& errors, double tolerance) {
    return errors.eta_forward <= tolerance && errors.eta_backward <= tolerance;
}

SolveValidation check_tree_solve_artifact(
    const std::string& device_dir,
    const std::string& report_dir,
    double tolerance) {
    SolveValidation result;
    result.tolerance = tolerance;
    try {
        if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
            throw ReferenceError("solve checker tolerance must be finite and positive");
        }
        auto artifact = hw::dataplane::load_artifact(device_dir);
        std::ifstream manifest_stream(device_dir + "/manifest.json");
        json manifest;
        manifest_stream >> manifest;
        std::ifstream summary_stream(report_dir + "/summary.json");
        json summary;
        if (!summary_stream) {
            throw ReferenceError("device summary.json is missing");
        }
        summary_stream >> summary;
        std::ifstream final_stream(
            report_dir + "/final_memory_image.bin", std::ios::binary);
        if (!final_stream) {
            throw ReferenceError("device final memory image is missing");
        }
        const std::vector<std::uint8_t> final_image{
            std::istreambuf_iterator<char>(final_stream),
            std::istreambuf_iterator<char>()};
        if (final_image.size() != artifact.image.size()) {
            throw ReferenceError("output image size mismatch");
        }
        const auto* rhs_region = artifact.config.region("rhs_data");
        const auto* solution_region = artifact.config.region("solution_data");
        if (rhs_region == nullptr || solution_region == nullptr) {
            throw ReferenceError("rhs_data/solution_data regions are missing");
        }
        const std::uint32_t dimension =
            static_cast<std::uint32_t>(rhs_region->size / 4);
        if (dimension == 0 || rhs_region->size % 4 != 0) {
            throw ReferenceError("rhs_data has an invalid size");
        }
        if (solution_region->size < static_cast<std::uint64_t>(dimension) * 4) {
            throw ReferenceError("solution_data is smaller than the RHS");
        }

        // Device status gate first: on a failed command batch the factor/
        // accumulator images are not meaningful inputs for items 1-3, so the
        // checker refuses to run them (and never indexes a -1 P-vector).
        scan_solve_command_status(
            report_dir + "/commands.csv", &result.solve_commands,
            &result.solve_commands_ok);
        const bool summary_ok =
            summary.value("numeric_status_pass", false) &&
            summary.value("control_flow_pass", false);
        if (result.solve_commands == 0) {
            result.reason = "artifact contains no solve commands";
            throw StatusGateError();
        }
        if (result.solve_commands_ok != result.solve_commands) {
            result.reason = "some solve commands did not complete OK";
            throw StatusGateError();
        }
        if (!summary_ok) {
            result.reason = "device reported a failed/incomplete command batch";
            throw StatusGateError();
        }

        const auto descriptor = [&](std::uint32_t id) {
            if (id >= artifact.config.descriptor_count) {
                throw ReferenceError("solve checker descriptor id is invalid");
            }
            return cmd::DescriptorCodec::decode(
                artifact.image.data() + artifact.config.descriptor_table_addr +
                    static_cast<std::uint64_t>(id) * cmd::DESCRIPTOR_RECORD_BYTES,
                cmd::DESCRIPTOR_RECORD_BYTES, artifact.image.size());
        };
        const auto node_count = artifact.config.node_count;
        const json& nodes = manifest.at("nodes");

        // b_hat: the FP32 RHS the device actually consumed (initial image).
        std::vector<double> b_hat(dimension, 0.0);
        for (std::uint32_t index = 0; index < dimension; ++index) {
            b_hat[index] = read_f32(artifact.image, rhs_region->offset + index * 4);
        }

        // Per-node device data: L, U, P-vector, forward z, backtracked x_hat.
        std::vector<SolveNodeData> data(node_count);
        for (std::uint32_t node_id = 0; node_id < node_count; ++node_id) {
            const std::string key = std::to_string(node_id);
            const json& node = nodes.at(key);
            SolveNodeData& node_data = data[node_id];
            node_data.variables =
                node.at("front_indices").get<std::vector<std::uint32_t>>();
            const auto factor = descriptor(
                node.at("descriptors").at("factor").get<std::uint32_t>());
            if (factor.descriptor_type != cmd::DescriptorType::Factor) {
                throw ReferenceError("node factor reference is not a FACTOR_DESC");
            }
            const std::uint32_t total = factor.body_words[3];
            const std::uint32_t pivot = factor.body_words[4];
            node_data.total = total;
            node_data.pivot = pivot;
            if (node_data.variables.size() != total) {
                throw ReferenceError("node front_indices do not match total_dim");
            }
            const auto region_of = [&](std::uint32_t id) {
                const auto region = descriptor(id);
                if (region.descriptor_type != cmd::DescriptorType::Region) {
                    throw ReferenceError("solve checker reference is not a region");
                }
                return region;
            };
            const auto l_region = region_of(factor.body_words[0]);
            const auto u_region = region_of(factor.body_words[1]);
            const auto p_region = region_of(factor.body_words[2]);
            node_data.l.assign(static_cast<std::size_t>(total) * pivot, 0.0);
            for (std::uint32_t row = 0; row < total; ++row) {
                for (std::uint32_t col = 0; col < pivot; ++col) {
                    node_data.l[static_cast<std::size_t>(row) * pivot + col] =
                        read_f32(
                            final_image,
                            region_base(l_region) +
                                row * static_cast<std::uint64_t>(l_region.body_words[4]) +
                                col * 4ull);
                }
            }
            node_data.u.assign(static_cast<std::size_t>(pivot) * total, 0.0);
            for (std::uint32_t row = 0; row < pivot; ++row) {
                for (std::uint32_t col = 0; col < total; ++col) {
                    node_data.u[static_cast<std::size_t>(row) * total + col] =
                        read_f32(
                            final_image,
                            region_base(u_region) +
                                row * static_cast<std::uint64_t>(u_region.body_words[4]) +
                                col * 4ull);
                }
            }
            const std::uint64_t p_base = region_base(p_region);
            node_data.p.resize(pivot);
            for (std::uint32_t index = 0; index < pivot; ++index) {
                node_data.p[index] = read_i32(final_image, p_base + index * 4);
            }
            const auto* workspace = artifact.config.region(
                node.at("regions").at("solve_workspace").get<std::string>());
            if (workspace == nullptr) {
                throw ReferenceError("node solve_workspace region is missing");
            }
            node_data.z.resize(pivot);
            for (std::uint32_t index = 0; index < pivot; ++index) {
                node_data.z[index] =
                    read_f32(final_image, workspace->offset + index * 4);
            }
        }

        // (1)-(2) triangular backward errors from the device data.
        std::vector<double> x_hat(dimension, 0.0);
        for (std::uint32_t index = 0; index < dimension; ++index) {
            x_hat[index] =
                read_f32(final_image, solution_region->offset + index * 4);
        }
        const auto triangular = triangular_backward_errors(data, b_hat, x_hat);
        result.eta_forward = triangular.eta_forward;
        result.eta_backward = triangular.eta_backward;

        // (3) transformed system: A_hat scattered from the initial fronts.
        DenseMatrix a_hat(dimension, dimension);
        std::vector<FrontInput> fronts(node_count);
        std::vector<int> parent(node_count);
        for (std::uint32_t node_id = 0; node_id < node_count; ++node_id) {
            const json& node = nodes.at(std::to_string(node_id));
            const SolveNodeData& node_data = data[node_id];
            const auto front_descriptor = descriptor(
                node.at("descriptors").at("front").get<std::uint32_t>());
            const auto front_region = descriptor(front_descriptor.body_words[0]);
            if (front_region.descriptor_type != cmd::DescriptorType::Region) {
                throw ReferenceError("node front reference is not a region");
            }
            const std::uint64_t front_base = region_base(front_region);
            const std::uint64_t front_stride = front_region.body_words[4];
            for (std::uint32_t row = 0; row < node_data.total; ++row) {
                for (std::uint32_t col = 0; col < node_data.total; ++col) {
                    const double value =
                        read_f32(artifact.image, front_base + row * front_stride + col * 4ull);
                    a_hat.at(node_data.variables[row], node_data.variables[col]) +=
                        value;
                }
            }
            // FrontInput for the independent FP64 reference re-derivation.
            fronts[node_id].variables.assign(
                node_data.variables.begin(), node_data.variables.end());
            fronts[node_id].pivot_columns = node_data.pivot;
            fronts[node_id].local_front = DenseMatrix(
                node_data.total, node_data.total);
            for (std::uint32_t row = 0; row < node_data.total; ++row) {
                for (std::uint32_t col = 0; col < node_data.total; ++col) {
                    fronts[node_id].local_front.at(row, col) =
                        read_f32(artifact.image, front_base + row * front_stride + col * 4ull);
                }
            }
            parent[node_id] = node.at("parent_id").get<int>();
            for (const auto& id : node.at("descriptors").at("contributions")) {
                const auto contribution = descriptor(id.get<std::uint32_t>());
                if (contribution.descriptor_type != cmd::DescriptorType::Contribution) {
                    throw ReferenceError("node contribution reference is invalid");
                }
                ContributionMap map;
                map.child = contribution.body_words[2];
                for (std::uint32_t index = 0;
                     index < contribution.body_words[4] + contribution.body_words[5]; ++index) {
                    std::uint32_t value = 0;
                    const std::uint64_t offset = contribution.payload_offset + index * 4ull;
                    std::memcpy(&value, artifact.image.data() + offset, 4);
                    if (index < contribution.body_words[4]) {
                        map.row_map.push_back(value);
                    } else {
                        map.col_map.push_back(value);
                    }
                }
                fronts[node_id].contributions.push_back(std::move(map));
            }
        }
        std::vector<double> residual(dimension, 0.0);
        double a_norm = 0.0;
        double x_norm = 0.0;
        double b_norm = 0.0;
        for (std::uint32_t row = 0; row < dimension; ++row) {
            double value = -b_hat[row];
            double row_abs = 0.0;
            for (std::uint32_t col = 0; col < dimension; ++col) {
                row_abs += std::fabs(a_hat.at(row, col));
                value += a_hat.at(row, col) * x_hat[col];
            }
            residual[row] = value;
            a_norm = std::max(a_norm, row_abs);
            x_norm = std::max(x_norm, std::fabs(x_hat[row]));
            b_norm = std::max(b_norm, std::fabs(b_hat[row]));
        }
        double residual_inf = 0.0;
        double transformed_backward = 0.0;
        for (std::uint32_t row = 0; row < dimension; ++row) {
            residual_inf = std::max(residual_inf, std::fabs(residual[row]));
            double denominator = std::fabs(b_hat[row]);
            for (std::uint32_t col = 0; col < dimension; ++col) {
                denominator += std::fabs(a_hat.at(row, col)) * std::fabs(x_hat[col]);
            }
            transformed_backward = std::max(
                transformed_backward,
                ratio_or_zero(std::fabs(residual[row]), denominator));
        }
        result.transformed_relative_residual =
            ratio_or_zero(residual_inf, a_norm * x_norm + b_norm);
        result.transformed_backward_error = transformed_backward;

        // Independent FP64 re-derivation from the initial fronts.
        const std::string policy_name = manifest.at("pivot").at("policy").get<std::string>();
        const PivotPolicy policy =
            policy_name == "STATIC" ? PivotPolicy::Static : PivotPolicy::DynamicPanel;
        const double threshold =
            manifest.at("pivot").at("threshold_ratio").get<double>();
        const auto factors =
            factor_front_tree(fronts, parent, policy, threshold);
        const std::vector<double> reference =
            solve_front_tree(factors, b_hat, 0.0);
        result.reference_solution_relative_error =
            relative_solution_error(x_hat, reference);
        result.nodes_checked = node_count;

        // (4) original-equation metrics when the host reference exists.
        const std::string matrix_path = device_dir + "/original_matrix_f64.bin";
        const std::string rhs_path = device_dir + "/original_rhs_f64.bin";
        std::ifstream matrix_stream(matrix_path, std::ios::binary);
        std::ifstream original_rhs_stream(rhs_path, std::ios::binary);
        if (matrix_stream && original_rhs_stream) {
            std::vector<std::uint8_t> matrix_bytes{
                std::istreambuf_iterator<char>(matrix_stream),
                std::istreambuf_iterator<char>()};
            std::vector<std::uint8_t> original_rhs_bytes{
                std::istreambuf_iterator<char>(original_rhs_stream),
                std::istreambuf_iterator<char>()};
            if (matrix_bytes.size() !=
                    static_cast<std::size_t>(dimension) * dimension * 8 ||
                original_rhs_bytes.size() != static_cast<std::size_t>(dimension) * 8) {
                throw ReferenceError("host original-system files have the wrong size");
            }
            const auto& transform = manifest.at("preprocess").at("transform");
            const auto ordering =
                transform.at("ordering_perm").get<std::vector<std::uint32_t>>();
            if (ordering.size() != dimension) {
                throw ReferenceError("ordering_perm length mismatch");
            }
            std::vector<double> x(dimension, 0.0);
            for (std::uint32_t index = 0; index < dimension; ++index) {
                x[ordering[index]] = x_hat[index];
            }
            double original_residual_inf = 0.0;
            double original_backward = 0.0;
            double original_a_norm = 0.0;
            double original_x_norm = 0.0;
            double original_b_norm = 0.0;
            std::vector<double> original_residual(dimension, 0.0);
            for (std::uint32_t row = 0; row < dimension; ++row) {
                const double b = read_f64(original_rhs_bytes, static_cast<std::size_t>(row) * 8);
                double value = -b;
                double row_abs = 0.0;
                for (std::uint32_t col = 0; col < dimension; ++col) {
                    const double a = read_f64(
                        matrix_bytes,
                        (static_cast<std::size_t>(row) * dimension + col) * 8);
                    value += a * x[col];
                    row_abs += std::fabs(a);
                }
                original_residual[row] = value;
                original_a_norm = std::max(original_a_norm, row_abs);
                original_x_norm = std::max(original_x_norm, std::fabs(x[row]));
                original_b_norm = std::max(original_b_norm, std::fabs(b));
            }
            for (std::uint32_t row = 0; row < dimension; ++row) {
                original_residual_inf =
                    std::max(original_residual_inf, std::fabs(original_residual[row]));
                const double b = read_f64(original_rhs_bytes, static_cast<std::size_t>(row) * 8);
                double denominator = std::fabs(b);
                for (std::uint32_t col = 0; col < dimension; ++col) {
                    denominator += std::fabs(read_f64(
                                       matrix_bytes,
                                       (static_cast<std::size_t>(row) * dimension + col) * 8)) *
                                   std::fabs(x[col]);
                }
                original_backward = std::max(
                    original_backward,
                    ratio_or_zero(std::fabs(original_residual[row]), denominator));
            }
            result.original_metrics_available = true;
            result.original_relative_residual = ratio_or_zero(
                original_residual_inf,
                original_a_norm * original_x_norm + original_b_norm);
            result.original_backward_error = original_backward;
            std::ifstream golden_stream(device_dir + "/x_true_f64.bin", std::ios::binary);
            if (golden_stream) {
                std::vector<std::uint8_t> golden{
                    std::istreambuf_iterator<char>(golden_stream),
                    std::istreambuf_iterator<char>()};
                if (golden.size() == static_cast<std::size_t>(dimension) * 8) {
                    std::vector<double> x_true(dimension, 0.0);
                    double difference = 0.0;
                    double reference_norm = 0.0;
                    for (std::uint32_t index = 0; index < dimension; ++index) {
                        x_true[index] = read_f64(golden, static_cast<std::size_t>(index) * 8);
                        const double delta = x[index] - x_true[index];
                        difference += delta * delta;
                        reference_norm += x_true[index] * x_true[index];
                    }
                    result.has_golden_solution = true;
                    result.original_solution_relative_error = ratio_or_zero(
                        std::sqrt(difference), std::sqrt(reference_norm));
                }
            }
        }

        // solve_pass = all solve commands OK && both triangular backward
        // errors within tolerance; items 3-4 are reported, not gating.
        result.pass = triangular_errors_within(triangular, tolerance);
        if (!result.pass) {
            result.reason = "triangular backward error exceeds the tolerance";
        }
    } catch (const StatusGateError&) {
        // `reason` was set by the status gate; no numeric validation ran.
    } catch (const std::exception& error) {
        result.reason = error.what();
    }

    json report{
        {"checker", "independent_fp64_tree_solve"},
        {"numerical_validation", result.pass ? "passed" : "failed"},
        {"solve_pass", result.pass},
        {"eta_forward", result.eta_forward},
        {"eta_backward", result.eta_backward},
        {"transformed_relative_residual", result.transformed_relative_residual},
        {"transformed_backward_error", result.transformed_backward_error},
        {"reference_solution_relative_error",
         result.reference_solution_relative_error},
        {"original_metrics_available", result.original_metrics_available},
        {"original_relative_residual", result.original_relative_residual},
        {"original_backward_error", result.original_backward_error},
        {"original_solution_relative_error",
         result.original_solution_relative_error},
        {"has_golden_solution", result.has_golden_solution},
        {"solve_commands", result.solve_commands},
        {"solve_commands_ok", result.solve_commands_ok},
        {"nodes_checked", result.nodes_checked},
        {"tolerance", result.tolerance},
        {"reason", result.reason},
    };
    std::ofstream stream(report_dir + "/solve_validation.json");
    stream << report.dump(2) << '\n';
    return result;
}

}  // namespace hw::reference_fp64
