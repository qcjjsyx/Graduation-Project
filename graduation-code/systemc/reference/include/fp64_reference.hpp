#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace hw::reference_fp64 {

class ReferenceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Pivot policy of the Command/Descriptor contract.  The two policies may
// produce different L/U/P for the same input; factors must only be compared
// across runs of the same policy.
enum class PivotPolicy {
    // Fixed post-ordering pivot order: no candidate search, no row interchange.
    Static,
    // Pivot search restricted to rows [k, pivot_columns) of the pivot block;
    // update rows are never promoted and never move.  Ties pick the lowest row.
    DynamicPanel,
};

// Numeric pivot failure of one factorization step.  A Static failure means the
// fixed elimination order failed; it is not a claim that the matrix is singular.
class PivotFailure : public ReferenceError {
public:
    enum class Kind {
        ZeroPivot,
        BelowThreshold,
        NonFinite,
        NoCandidate,
    };

    PivotFailure(Kind kind, std::size_t column, const std::string& detail)
        : ReferenceError(
              "pivot failure in column " + std::to_string(column) + ": " + detail),
          kind_(kind),
          column_(column) {}

    Kind kind() const { return kind_; }
    std::size_t column() const { return column_; }

private:
    Kind kind_;
    std::size_t column_;
};

struct DenseMatrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<double> values{};

    DenseMatrix() = default;
    DenseMatrix(std::size_t row_count, std::size_t col_count);
    DenseMatrix(
        std::size_t row_count,
        std::size_t col_count,
        std::vector<double> row_major_values);

    double& at(std::size_t row, std::size_t col);
    const double& at(std::size_t row, std::size_t col) const;
};

struct PanelLuResult {
    std::size_t pivot_columns{0};
    DenseMatrix l{};
    DenseMatrix u{};
    DenseMatrix schur_update{};
    // P*A row i is sourced from original row permutation[i].
    std::vector<std::size_t> permutation{};
    // Selected logical row before each pivot swap. Ties choose the lowest row.
    std::vector<std::size_t> pivot_rows{};
    double minimum_pivot_ratio{1.0};
    double pivot_growth{1.0};
};

struct AccuracyMetrics {
    double relative_residual{0.0};
    double componentwise_backward_error{0.0};
    double relative_solution_error{0.0};
};

struct SolveResult {
    PanelLuResult factorization{};
    std::vector<double> solution{};
    AccuracyMetrics metrics{};
};

struct FrontFactor {
    // First factorization.pivot_columns entries are eliminated by this front.
    std::vector<std::size_t> variables{};
    PanelLuResult factorization{};
};

// One child-update to parent-front contribution (see CONTRIBUTION_DESC):
// target[col_map[i], col_map[j]] += child_schur[row_map[i], row_map[j]].
struct ContributionMap {
    std::size_t child{0};
    std::vector<std::size_t> row_map{};
    std::vector<std::size_t> col_map{};
};

// One node's numeric input before assembly: local contribution (update block
// already zeroed) plus the contributions of its children.
struct FrontInput {
    std::vector<std::size_t> variables{};
    std::size_t pivot_columns{0};
    DenseMatrix local_front{};
    std::vector<ContributionMap> contributions{};
};

// Factor one assembled front into FrontFactor form under the declared policy.
// The produced permutation is confined to the pivot block, matching the scope
// enforced by solve_front_tree.
FrontFactor factor_front(
    const std::vector<std::size_t>& variables,
    const DenseMatrix& assembled_front,
    std::size_t pivot_columns,
    PivotPolicy policy,
    double relative_pivot_tolerance = 0.0);

// Assemble child Schur updates into their parents and factor every front in
// children-before-parents order under the declared policy.  Same permutation
// scope as solve_front_tree, so factorization and solve stay consistent.
std::vector<FrontFactor> factor_front_tree(
    const std::vector<FrontInput>& fronts,
    const std::vector<int>& parent,
    PivotPolicy policy,
    double relative_pivot_tolerance = 0.0);

// Generic dense LU with partial pivoting over all remaining rows of the
// rectangle (update rows can be promoted to pivots).  This API keeps its full
// mathematical capability and is *not* the device pivot contract.
PanelLuResult panel_lu(
    const DenseMatrix& input,
    std::size_t pivot_columns,
    double relative_pivot_tolerance = 0.0);

// Device-contract reference factorization under the declared pivot policy.
// The relative threshold is measured against the pre-elimination max |input|
// of the block ("front_pre_elimination_max_abs"); zero is always rejected.
// The ratio is normalized once to its IEEE-754 binary32 value (the device
// storage format), so host and device share one effective threshold; ratios
// that underflow to zero or round to 1.0 in binary32 are rejected.
// Throws PivotFailure on zero / too-small pivots and non-finite values.
PanelLuResult panel_lu_policy(
    const DenseMatrix& input,
    std::size_t pivot_columns,
    PivotPolicy policy,
    double relative_pivot_tolerance = 0.0);

DenseMatrix trsm_left_lower(
    const DenseMatrix& lower,
    const DenseMatrix& rhs,
    bool unit_diagonal = true,
    double singular_tolerance = 0.0);

DenseMatrix trsm_right_upper(
    const DenseMatrix& rhs,
    const DenseMatrix& upper,
    double singular_tolerance = 0.0);

DenseMatrix gemm_schur(
    const DenseMatrix& c,
    const DenseMatrix& a,
    const DenseMatrix& b);

std::vector<double> solve_from_lu(
    const PanelLuResult& factorization,
    const std::vector<double>& rhs,
    double singular_tolerance = 0.0);

std::vector<double> solve_front_tree(
    const std::vector<FrontFactor>& fronts_child_to_parent,
    const std::vector<double>& rhs,
    double singular_tolerance = 0.0);

double relative_residual(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs);

double componentwise_backward_error(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs);

double relative_solution_error(
    const std::vector<double>& x,
    const std::vector<double>& reference_solution);

AccuracyMetrics compute_accuracy_metrics(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs,
    const std::vector<double>& reference_solution);

SolveResult solve_linear_system(
    const DenseMatrix& matrix,
    const std::vector<double>& rhs,
    const std::vector<double>& reference_solution,
    double relative_pivot_tolerance = 0.0);

}  // namespace hw::reference_fp64
