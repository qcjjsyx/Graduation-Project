#include "fp64_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace hw::reference_fp64 {
namespace {

void validate_shape(const DenseMatrix& matrix, const std::string& name) {
    if (matrix.rows != 0 &&
        matrix.cols > std::numeric_limits<std::size_t>::max() / matrix.rows) {
        throw ReferenceError(name + " dimensions overflow size_t");
    }
    if (matrix.values.size() != matrix.rows * matrix.cols) {
        throw ReferenceError(name + " row-major storage size mismatch");
    }
    for (const auto value : matrix.values) {
        if (!std::isfinite(value)) {
            throw ReferenceError(name + " contains a non-finite value");
        }
    }
}

void validate_vector(const std::vector<double>& values, const std::string& name) {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw ReferenceError(name + " contains a non-finite value");
        }
    }
}

void validate_tolerance(double tolerance, const std::string& name) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw ReferenceError(name + " must be finite and non-negative");
    }
}

long double vector_l2_norm(const std::vector<double>& values) {
    long double sum = 0.0L;
    for (const auto value : values) {
        const auto wide = static_cast<long double>(value);
        sum += wide * wide;
    }
    return std::sqrt(sum);
}

long double matrix_frobenius_norm(const DenseMatrix& matrix) {
    return vector_l2_norm(matrix.values);
}

double ratio_or_limit(long double numerator, long double denominator) {
    if (denominator != 0.0L) {
        return static_cast<double>(numerator / denominator);
    }
    return numerator == 0.0L ? 0.0 : std::numeric_limits<double>::infinity();
}

std::vector<long double> residual_vector(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    if (matrix.rows != matrix.cols || matrix.rows != x.size() ||
        matrix.rows != rhs.size()) {
        throw ReferenceError("residual dimension mismatch");
    }
    std::vector<long double> residual(matrix.rows, 0.0L);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        long double value = -static_cast<long double>(rhs[row]);
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            value += static_cast<long double>(matrix.at(row, col)) * x[col];
        }
        residual[row] = value;
    }
    return residual;
}

long double residual_l2_norm(const std::vector<long double>& residual) {
    long double sum = 0.0L;
    for (const auto value : residual) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

void validate_lu_for_solve(
    const PanelLuResult& factorization,
    std::size_t dimension) {
    const auto& l = factorization.l;
    const auto& u = factorization.u;
    validate_shape(l, "L");
    validate_shape(u, "U");
    if (factorization.pivot_columns != dimension || l.rows != dimension ||
        l.cols != dimension || u.rows != dimension || u.cols != dimension ||
        factorization.permutation.size() != dimension) {
        throw ReferenceError("solve requires a complete square LU factorization");
    }
    std::vector<bool> seen(dimension, false);
    for (const auto source_row : factorization.permutation) {
        if (source_row >= dimension || seen[source_row]) {
            throw ReferenceError("LU permutation is invalid");
        }
        seen[source_row] = true;
    }
}

bool singular(double diagonal, double tolerance) {
    return std::abs(diagonal) <= tolerance;
}

}  // namespace

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t col_count)
    : rows(row_count), cols(col_count) {
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
        throw ReferenceError("matrix dimensions overflow size_t");
    }
    values.assign(rows * cols, 0.0);
}

DenseMatrix::DenseMatrix(
    std::size_t row_count,
    std::size_t col_count,
    std::vector<double> row_major_values)
    : rows(row_count), cols(col_count), values(std::move(row_major_values)) {
    validate_shape(*this, "matrix");
}

double& DenseMatrix::at(std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) {
        throw ReferenceError("matrix index out of range");
    }
    return values[row * cols + col];
}

const double& DenseMatrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows || col >= cols) {
        throw ReferenceError("matrix index out of range");
    }
    return values[row * cols + col];
}

PanelLuResult panel_lu(
    const DenseMatrix& input,
    std::size_t pivot_columns,
    double relative_pivot_tolerance) {
    validate_shape(input, "panel input");
    validate_tolerance(relative_pivot_tolerance, "relative pivot tolerance");
    if (input.rows == 0 || input.cols == 0 || pivot_columns == 0 ||
        pivot_columns > std::min(input.rows, input.cols)) {
        throw ReferenceError("invalid panel dimensions");
    }

    DenseMatrix work = input;
    PanelLuResult result{};
    result.pivot_columns = pivot_columns;
    result.permutation.resize(input.rows);
    std::iota(result.permutation.begin(), result.permutation.end(), 0);
    result.minimum_pivot_ratio = std::numeric_limits<double>::infinity();

    double initial_max_abs = 0.0;
    for (const auto value : input.values) {
        initial_max_abs = std::max(initial_max_abs, std::abs(value));
    }
    double maximum_workspace_abs = initial_max_abs;

    for (std::size_t column = 0; column < pivot_columns; ++column) {
        std::size_t best_row = column;
        double best_abs = -1.0;
        for (std::size_t row = column; row < input.rows; ++row) {
            const auto magnitude = std::abs(work.at(row, column));
            // Strict comparison preserves the lowest logical row on ties.
            if (magnitude > best_abs) {
                best_abs = magnitude;
                best_row = row;
            }
        }
        result.pivot_rows.push_back(best_row);
        const auto pivot_ratio =
            initial_max_abs == 0.0 ? 0.0 : best_abs / initial_max_abs;
        result.minimum_pivot_ratio =
            std::min(result.minimum_pivot_ratio, pivot_ratio);
        if (best_abs == 0.0 ||
            best_abs <= relative_pivot_tolerance * initial_max_abs) {
            throw ReferenceError(
                "pivot below threshold in column " + std::to_string(column));
        }

        if (best_row != column) {
            for (std::size_t col = 0; col < input.cols; ++col) {
                std::swap(work.at(column, col), work.at(best_row, col));
            }
            std::swap(result.permutation[column], result.permutation[best_row]);
        }

        const auto pivot = work.at(column, column);
        for (std::size_t row = column + 1; row < input.rows; ++row) {
            const auto multiplier = work.at(row, column) / pivot;
            if (!std::isfinite(multiplier)) {
                throw ReferenceError("non-finite panel multiplier");
            }
            work.at(row, column) = multiplier;
            maximum_workspace_abs =
                std::max(maximum_workspace_abs, std::abs(multiplier));
            for (std::size_t col = column + 1; col < input.cols; ++col) {
                const auto updated =
                    work.at(row, col) - multiplier * work.at(column, col);
                if (!std::isfinite(updated)) {
                    throw ReferenceError("non-finite panel update");
                }
                work.at(row, col) = updated;
                maximum_workspace_abs =
                    std::max(maximum_workspace_abs, std::abs(updated));
            }
        }
    }

    result.l = DenseMatrix(input.rows, pivot_columns);
    result.u = DenseMatrix(pivot_columns, input.cols);
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t col = 0; col < pivot_columns; ++col) {
            if (row == col) {
                result.l.at(row, col) = 1.0;
            } else if (row > col) {
                result.l.at(row, col) = work.at(row, col);
            }
        }
    }
    for (std::size_t row = 0; row < pivot_columns; ++row) {
        for (std::size_t col = row; col < input.cols; ++col) {
            result.u.at(row, col) = work.at(row, col);
        }
    }

    result.schur_update = DenseMatrix(
        input.rows - pivot_columns, input.cols - pivot_columns);
    for (std::size_t row = pivot_columns; row < input.rows; ++row) {
        for (std::size_t col = pivot_columns; col < input.cols; ++col) {
            result.schur_update.at(row - pivot_columns, col - pivot_columns) =
                work.at(row, col);
        }
    }
    result.pivot_growth = initial_max_abs == 0.0 ?
        1.0 : maximum_workspace_abs / initial_max_abs;
    return result;
}

DenseMatrix trsm_left_lower(
    const DenseMatrix& lower,
    const DenseMatrix& rhs,
    bool unit_diagonal,
    double singular_tolerance) {
    validate_shape(lower, "left TRSM lower matrix");
    validate_shape(rhs, "left TRSM RHS");
    validate_tolerance(singular_tolerance, "singular tolerance");
    if (lower.rows == 0 || lower.rows != lower.cols || rhs.rows != lower.rows) {
        throw ReferenceError("left TRSM dimension mismatch");
    }

    DenseMatrix solution(rhs.rows, rhs.cols);
    for (std::size_t row = 0; row < lower.rows; ++row) {
        for (std::size_t col = 0; col < rhs.cols; ++col) {
            long double value = rhs.at(row, col);
            for (std::size_t inner = 0; inner < row; ++inner) {
                value -= static_cast<long double>(lower.at(row, inner)) *
                    solution.at(inner, col);
            }
            if (!unit_diagonal) {
                const auto diagonal = lower.at(row, row);
                if (singular(diagonal, singular_tolerance)) {
                    throw ReferenceError("left TRSM singular diagonal");
                }
                value /= diagonal;
            }
            solution.at(row, col) = static_cast<double>(value);
        }
    }
    return solution;
}

DenseMatrix trsm_right_upper(
    const DenseMatrix& rhs,
    const DenseMatrix& upper,
    double singular_tolerance) {
    validate_shape(rhs, "right TRSM RHS");
    validate_shape(upper, "right TRSM upper matrix");
    validate_tolerance(singular_tolerance, "singular tolerance");
    if (upper.rows == 0 || upper.rows != upper.cols || rhs.cols != upper.rows) {
        throw ReferenceError("right TRSM dimension mismatch");
    }

    DenseMatrix solution(rhs.rows, rhs.cols);
    for (std::size_t row = 0; row < rhs.rows; ++row) {
        // X*U=B has a forward dependency across U columns.
        for (std::size_t col = 0; col < upper.cols; ++col) {
            long double value = rhs.at(row, col);
            for (std::size_t inner = 0; inner < col; ++inner) {
                value -= static_cast<long double>(solution.at(row, inner)) *
                    upper.at(inner, col);
            }
            const auto diagonal = upper.at(col, col);
            if (singular(diagonal, singular_tolerance)) {
                throw ReferenceError("right TRSM singular diagonal");
            }
            solution.at(row, col) = static_cast<double>(value / diagonal);
        }
    }
    return solution;
}

DenseMatrix gemm_schur(
    const DenseMatrix& c,
    const DenseMatrix& a,
    const DenseMatrix& b) {
    validate_shape(c, "GEMM C");
    validate_shape(a, "GEMM A");
    validate_shape(b, "GEMM B");
    if (a.cols != b.rows || c.rows != a.rows || c.cols != b.cols) {
        throw ReferenceError("GEMM-Schur dimension mismatch");
    }

    DenseMatrix result = c;
    for (std::size_t row = 0; row < c.rows; ++row) {
        for (std::size_t col = 0; col < c.cols; ++col) {
            long double product = 0.0L;
            for (std::size_t inner = 0; inner < a.cols; ++inner) {
                product += static_cast<long double>(a.at(row, inner)) *
                    b.at(inner, col);
            }
            result.at(row, col) =
                static_cast<double>(static_cast<long double>(c.at(row, col)) - product);
        }
    }
    return result;
}

std::vector<double> solve_from_lu(
    const PanelLuResult& factorization,
    const std::vector<double>& rhs,
    double singular_tolerance) {
    validate_vector(rhs, "RHS");
    validate_tolerance(singular_tolerance, "singular tolerance");
    validate_lu_for_solve(factorization, rhs.size());

    const auto dimension = rhs.size();
    std::vector<double> y(dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        long double value = rhs[factorization.permutation[row]];
        for (std::size_t col = 0; col < row; ++col) {
            value -= static_cast<long double>(factorization.l.at(row, col)) * y[col];
        }
        y[row] = static_cast<double>(value);
    }

    std::vector<double> x(dimension, 0.0);
    for (std::size_t reverse_row = dimension; reverse_row > 0; --reverse_row) {
        const auto row = reverse_row - 1;
        long double value = y[row];
        for (std::size_t col = row + 1; col < dimension; ++col) {
            value -= static_cast<long double>(factorization.u.at(row, col)) * x[col];
        }
        const auto diagonal = factorization.u.at(row, row);
        if (singular(diagonal, singular_tolerance)) {
            throw ReferenceError("backward solve singular diagonal");
        }
        x[row] = static_cast<double>(value / diagonal);
    }
    return x;
}

std::vector<double> solve_front_tree(
    const std::vector<FrontFactor>& fronts_child_to_parent,
    const std::vector<double>& rhs,
    double singular_tolerance) {
    validate_vector(rhs, "tree RHS");
    validate_tolerance(singular_tolerance, "singular tolerance");
    if (rhs.empty() || fronts_child_to_parent.empty()) {
        throw ReferenceError("tree solve requires RHS and fronts");
    }

    std::vector<bool> pivot_owner(rhs.size(), false);
    for (const auto& front : fronts_child_to_parent) {
        const auto dimension = front.variables.size();
        const auto pivots = front.factorization.pivot_columns;
        validate_shape(front.factorization.l, "tree front L");
        validate_shape(front.factorization.u, "tree front U");
        if (dimension == 0 || pivots == 0 || pivots > dimension ||
            front.factorization.l.rows != dimension ||
            front.factorization.l.cols != pivots ||
            front.factorization.u.rows != pivots ||
            front.factorization.u.cols != dimension ||
            front.factorization.permutation.size() != dimension) {
            throw ReferenceError("tree front factor dimensions are invalid");
        }
        std::vector<bool> local_seen(dimension, false);
        std::vector<bool> variable_seen(rhs.size(), false);
        for (std::size_t local = 0; local < dimension; ++local) {
            const auto variable = front.variables[local];
            const auto source = front.factorization.permutation[local];
            if (variable >= rhs.size() || source >= dimension || local_seen[source] ||
                variable_seen[variable]) {
                throw ReferenceError("tree front variables or permutation are invalid");
            }
            local_seen[source] = true;
            variable_seen[variable] = true;
        }
        for (std::size_t row = 0; row < pivots; ++row) {
            if (front.factorization.permutation[row] >= pivots) {
                throw ReferenceError(
                    "tree solve does not support pivoting across the front boundary");
            }
            const auto variable = front.variables[row];
            if (pivot_owner[variable]) {
                throw ReferenceError("tree variable has multiple pivot owners");
            }
            pivot_owner[variable] = true;
        }
        for (std::size_t row = pivots; row < dimension; ++row) {
            if (front.factorization.permutation[row] != row) {
                throw ReferenceError(
                    "tree solve does not support permuted update rows");
            }
        }
    }
    if (std::find(pivot_owner.begin(), pivot_owner.end(), false) != pivot_owner.end()) {
        throw ReferenceError("tree factors do not eliminate every variable");
    }

    auto work = rhs;
    std::vector<double> y(rhs.size(), 0.0);
    for (const auto& front : fronts_child_to_parent) {
        const auto pivots = front.factorization.pivot_columns;
        const auto dimension = front.variables.size();
        std::vector<double> pivot_rhs(pivots, 0.0);
        for (std::size_t row = 0; row < pivots; ++row) {
            pivot_rhs[row] = work[front.variables[front.factorization.permutation[row]]];
        }
        for (std::size_t row = 0; row < pivots; ++row) {
            long double value = pivot_rhs[row];
            for (std::size_t col = 0; col < row; ++col) {
                value -= static_cast<long double>(front.factorization.l.at(row, col)) *
                    y[front.variables[col]];
            }
            y[front.variables[row]] = static_cast<double>(value);
        }
        for (std::size_t row = pivots; row < dimension; ++row) {
            long double value = work[front.variables[row]];
            for (std::size_t col = 0; col < pivots; ++col) {
                value -= static_cast<long double>(front.factorization.l.at(row, col)) *
                    y[front.variables[col]];
            }
            work[front.variables[row]] = static_cast<double>(value);
        }
    }

    std::vector<double> x(rhs.size(), 0.0);
    for (auto front_it = fronts_child_to_parent.rbegin();
         front_it != fronts_child_to_parent.rend(); ++front_it) {
        const auto& front = *front_it;
        const auto pivots = front.factorization.pivot_columns;
        const auto dimension = front.variables.size();
        for (std::size_t reverse_row = pivots; reverse_row > 0; --reverse_row) {
            const auto row = reverse_row - 1;
            long double value = y[front.variables[row]];
            for (std::size_t col = row + 1; col < dimension; ++col) {
                value -= static_cast<long double>(front.factorization.u.at(row, col)) *
                    x[front.variables[col]];
            }
            const auto diagonal = front.factorization.u.at(row, row);
            if (singular(diagonal, singular_tolerance)) {
                throw ReferenceError("tree backward solve singular diagonal");
            }
            x[front.variables[row]] = static_cast<double>(value / diagonal);
        }
    }
    return x;
}

double relative_residual(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    validate_shape(matrix, "residual matrix");
    validate_vector(x, "solution");
    validate_vector(rhs, "RHS");
    const auto residual = residual_vector(matrix, x, rhs);
    const auto denominator =
        matrix_frobenius_norm(matrix) * vector_l2_norm(x) + vector_l2_norm(rhs);
    return ratio_or_limit(residual_l2_norm(residual), denominator);
}

double componentwise_backward_error(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    validate_shape(matrix, "backward-error matrix");
    validate_vector(x, "solution");
    validate_vector(rhs, "RHS");
    const auto residual = residual_vector(matrix, x, rhs);
    double maximum = 0.0;
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        long double denominator = std::abs(static_cast<long double>(rhs[row]));
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            denominator += std::abs(static_cast<long double>(matrix.at(row, col))) *
                std::abs(static_cast<long double>(x[col]));
        }
        maximum = std::max(
            maximum,
            ratio_or_limit(std::abs(residual[row]), denominator));
    }
    return maximum;
}

double relative_solution_error(
    const std::vector<double>& x,
    const std::vector<double>& reference_solution) {
    validate_vector(x, "solution");
    validate_vector(reference_solution, "reference solution");
    if (x.size() != reference_solution.size()) {
        throw ReferenceError("solution/reference dimension mismatch");
    }
    std::vector<double> difference(x.size(), 0.0);
    for (std::size_t index = 0; index < x.size(); ++index) {
        difference[index] = x[index] - reference_solution[index];
    }
    return ratio_or_limit(
        vector_l2_norm(difference), vector_l2_norm(reference_solution));
}

AccuracyMetrics compute_accuracy_metrics(
    const DenseMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs,
    const std::vector<double>& reference_solution) {
    return {
        relative_residual(matrix, x, rhs),
        componentwise_backward_error(matrix, x, rhs),
        relative_solution_error(x, reference_solution),
    };
}

SolveResult solve_linear_system(
    const DenseMatrix& matrix,
    const std::vector<double>& rhs,
    const std::vector<double>& reference_solution,
    double relative_pivot_tolerance) {
    validate_shape(matrix, "solve matrix");
    if (matrix.rows == 0 || matrix.rows != matrix.cols || rhs.size() != matrix.rows ||
        reference_solution.size() != matrix.rows) {
        throw ReferenceError("linear solve dimension mismatch");
    }
    auto factorization =
        panel_lu(matrix, matrix.cols, relative_pivot_tolerance);
    auto solution = solve_from_lu(factorization, rhs);
    auto metrics =
        compute_accuracy_metrics(matrix, solution, rhs, reference_solution);
    return {std::move(factorization), std::move(solution), metrics};
}

}  // namespace hw::reference_fp64
