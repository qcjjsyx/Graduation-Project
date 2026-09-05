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

PanelLuResult panel_lu(
    const DenseMatrix& input,
    std::size_t pivot_columns,
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
