#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hw::reference_fp64 {

// T12 independent solve validation (design doc §5). Reads the initial memory
// image, the final device image, the manifest and (for the status gate) the
// device reports; never reads host golden factors/solutions for items 1-3.
//
//   1. forward backward error eta_f = max_rho |b_hat - L~ y| /
//      (|L~| |y| + |b_hat|)_rho, with L~ assembled from the device L and
//      P-vector (pivot rows at g[p[i]], columns g[j]) and y from the device
//      forward workspace z;
//   2. backward backward error eta_b = max_i |y - U_g x_hat| /
//      (|U_g| |x_hat| + |y|), U_g from the device U;
//   3. transformed-system residual of A_hat x_hat = b_hat, where A_hat is
//      scattered from the initial local fronts (every original element lives
//      in exactly one front);
//   4. original-equation metrics when the host reference files exist.
//
// solve_pass = (eta_f <= tol) && (eta_b <= tol) && every solve command OK;
// 3 and 4 are reported but never the pass condition (the tests assert them
// separately on well-conditioned fixtures).
struct SolveValidation {
    bool pass{false};
    double eta_forward{0.0};
    double eta_backward{0.0};
    double transformed_relative_residual{0.0};
    double transformed_backward_error{0.0};
    double reference_solution_relative_error{0.0};
    bool original_metrics_available{false};
    double original_relative_residual{0.0};
    double original_backward_error{0.0};
    double original_solution_relative_error{0.0};
    bool has_golden_solution{false};
    std::uint32_t solve_commands{0};
    std::uint32_t solve_commands_ok{0};
    std::uint32_t nodes_checked{0};
    double tolerance{1e-4};
    std::string reason;
};

// Device data of one node used by the triangular backward errors (items 1-2):
// front variables g, L (n x p), U (p x n), gather P-vector p and forward z.
struct SolveNodeData {
    std::vector<std::uint32_t> variables;
    std::vector<double> l;  // n x p
    std::vector<double> u;  // p x n
    std::vector<std::int32_t> p;
    std::vector<double> z;  // p
    std::uint32_t total{0};
    std::uint32_t pivot{0};
};

struct TriangularBackwardErrors {
    double eta_forward{0.0};
    double eta_backward{0.0};
};

// Items 1-2 of check_tree_solve_artifact as a pure function of the per-node
// device data, b_hat and x_hat (both indexed by global variable).  L~ places
// the pivot row i of node k at global row g[p_k[i]] and the update row t at
// g[p+t].  Throws ReferenceError if a P-vector entry is outside the block.
TriangularBackwardErrors triangular_backward_errors(
    const std::vector<SolveNodeData>& nodes,
    const std::vector<double>& b_hat,
    const std::vector<double>& x_hat);

// The numeric half of solve_pass (the other half is "every solve command OK").
bool triangular_errors_within(const TriangularBackwardErrors& errors, double tolerance);

SolveValidation check_tree_solve_artifact(
    const std::string& device_dir,
    const std::string& report_dir,
    double tolerance = 1e-4);

}  // namespace hw::reference_fp64
