#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "fp64_reference.hpp"
#include "nlohmann/json.hpp"

namespace {

namespace ref = hw::reference_fp64;
using json = nlohmann::json;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_throws(const std::function<void()>& action, const std::string& message) {
    try {
        action();
    } catch (const ref::ReferenceError&) {
        return;
    }
    throw std::runtime_error(message);
}

bool close(double actual, double expected, double tolerance = 1e-12) {
    const auto scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    return std::abs(actual - expected) <= tolerance * scale;
}

void expect_vector_close(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& message,
    double tolerance = 1e-12) {
    expect(actual.size() == expected.size(), message + " size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!close(actual[index], expected[index], tolerance)) {
            throw std::runtime_error(
                message + " at index " + std::to_string(index) +
                ": got " + std::to_string(actual[index]) +
                ", expected " + std::to_string(expected[index]));
        }
    }
}

ref::DenseMatrix matrix_from_json(const json& source) {
    return {
        source.at("rows").get<std::size_t>(),
        source.at("cols").get<std::size_t>(),
        source.at("values").get<std::vector<double>>(),
    };
}

void expect_matrix_close(
    const ref::DenseMatrix& actual,
    const json& expected,
    const std::string& message,
    double tolerance = 1e-12) {
    expect(actual.rows == expected.at("rows").get<std::size_t>(),
           message + " row count mismatch");
    expect(actual.cols == expected.at("cols").get<std::size_t>(),
           message + " column count mismatch");
    expect_vector_close(
        actual.values,
        expected.at("values").get<std::vector<double>>(),
        message,
        tolerance);
}

void expect_size_vector(
    const std::vector<std::size_t>& actual,
    const json& expected,
    const std::string& message) {
    expect(actual == expected.get<std::vector<std::size_t>>(), message);
}

void expect_panel_reconstruction(
    const ref::DenseMatrix& input,
    const ref::PanelLuResult& factorization,
    const std::string& name) {
    const auto pivots = factorization.pivot_columns;
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t col = 0; col < input.cols; ++col) {
            long double reconstructed = 0.0L;
            for (std::size_t inner = 0; inner < pivots; ++inner) {
                reconstructed +=
                    static_cast<long double>(factorization.l.at(row, inner)) *
                    factorization.u.at(inner, col);
            }
            if (row >= pivots && col >= pivots) {
                reconstructed += factorization.schur_update.at(
                    row - pivots, col - pivots);
            }
            const auto expected = input.at(factorization.permutation[row], col);
            expect(
                close(static_cast<double>(reconstructed), expected, 2e-12),
                name + " does not reconstruct P*A");
        }
    }
}

void test_panel_and_dense_solve(const json& fixture) {
    bool saw_one_by_one = false;
    bool saw_row_swap = false;
    bool saw_tie_break = false;
    bool saw_four_by_four = false;
    bool saw_near_singular = false;
    bool saw_asymmetric = false;

    for (const auto& test_case : fixture.at("panel_cases")) {
        const auto name = test_case.at("name").get<std::string>();
        const auto input = matrix_from_json(test_case.at("matrix"));
        const auto pivots = test_case.at("pivot_columns").get<std::size_t>();
        const auto factorization = ref::panel_lu(input, pivots);
        const auto& expected = test_case.at("expected");

        expect_size_vector(
            factorization.pivot_rows, expected.at("pivot_rows"),
            name + " pivot rows mismatch");
        expect_size_vector(
            factorization.permutation, expected.at("permutation"),
            name + " permutation mismatch");
        expect_matrix_close(factorization.l, expected.at("l"), name + " L mismatch");
        expect_matrix_close(factorization.u, expected.at("u"), name + " U mismatch");
        expect_matrix_close(
            factorization.schur_update,
            expected.at("schur_update"),
            name + " Schur update mismatch");
        expect(close(
                   factorization.minimum_pivot_ratio,
                   expected.at("minimum_pivot_ratio").get<double>()),
               name + " minimum pivot ratio mismatch");
        expect(close(
                   factorization.pivot_growth,
                   expected.at("pivot_growth").get<double>()),
               name + " pivot growth mismatch");
        expect_panel_reconstruction(input, factorization, name);

        if (test_case.contains("rhs")) {
            const auto rhs = test_case.at("rhs").get<std::vector<double>>();
            const auto reference_solution =
                test_case.at("reference_solution").get<std::vector<double>>();
            const auto solution = ref::solve_from_lu(factorization, rhs);
            expect_vector_close(
                solution,
                expected.at("solution").get<std::vector<double>>(),
                name + " solve mismatch",
                2e-11);
            const auto solve =
                ref::solve_linear_system(input, rhs, reference_solution);
            expect_vector_close(
                solve.solution, solution, name + " solve API disagreement", 2e-11);
            expect(solve.metrics.relative_residual <= 1e-14,
                   name + " relative residual too large");
            expect(solve.metrics.componentwise_backward_error <= 1e-14,
                   name + " backward error too large");
            expect(solve.metrics.relative_solution_error <= 2e-11,
                   name + " solution error too large");
        }

        saw_one_by_one |= name == "one_by_one";
        saw_row_swap |= name == "two_by_two_row_swap";
        saw_tie_break |= name == "two_by_two_tie_break";
        saw_four_by_four |= name == "four_by_four_partial_panel";
        saw_near_singular |= name == "near_singular";
        saw_asymmetric |= name == "structurally_asymmetric_four_by_four";

        if (name == "near_singular") {
            expect_throws(
                [&] { (void)ref::panel_lu(input, pivots, 1e-10); },
                "near-singular fixture must honor the pivot threshold");
        } else if (name == "structurally_asymmetric_four_by_four") {
            expect(input.at(0, 3) != 0.0 && input.at(3, 0) == 0.0,
                   "asymmetric fixture must have an asymmetric sparsity pattern");
        }
    }

    expect(saw_one_by_one && saw_row_swap && saw_tie_break && saw_four_by_four &&
               saw_near_singular && saw_asymmetric,
           "required panel fixtures are missing");
}

void test_trsm_and_gemm(const json& fixture) {
    const auto& operators = fixture.at("operators");
    const auto& left = operators.at("trsm_left_lower");
    expect_matrix_close(
        ref::trsm_left_lower(
            matrix_from_json(left.at("lower")), matrix_from_json(left.at("rhs"))),
        left.at("expected"),
        "left lower TRSM mismatch");

    const auto& right = operators.at("trsm_right_upper");
    expect_matrix_close(
        ref::trsm_right_upper(
            matrix_from_json(right.at("rhs")), matrix_from_json(right.at("upper"))),
        right.at("expected"),
        "right upper TRSM mismatch");

    const auto& gemm = operators.at("gemm_schur");
    expect_matrix_close(
        ref::gemm_schur(
            matrix_from_json(gemm.at("c")),
            matrix_from_json(gemm.at("a")),
            matrix_from_json(gemm.at("b"))),
        gemm.at("expected"),
        "GEMM-Schur mismatch");
}

void test_accuracy_metrics(const json& fixture) {
    const auto& test_case = fixture.at("metric_case");
    const auto metrics = ref::compute_accuracy_metrics(
        matrix_from_json(test_case.at("matrix")),
        test_case.at("solution").get<std::vector<double>>(),
        test_case.at("rhs").get<std::vector<double>>(),
        test_case.at("reference_solution").get<std::vector<double>>());
    const auto& expected = test_case.at("expected");
    expect(close(
               metrics.relative_residual,
               expected.at("relative_residual").get<double>()),
           "relative residual formula mismatch");
    expect(close(
               metrics.componentwise_backward_error,
               expected.at("componentwise_backward_error").get<double>()),
           "componentwise backward error formula mismatch");
    expect(close(
               metrics.relative_solution_error,
               expected.at("relative_solution_error").get<double>()),
           "relative solution error formula mismatch");
}

void test_two_front_tree(const json& fixture) {
    const auto& tree = fixture.at("tree_case");
    const auto& child_json = tree.at("child");
    const auto child = ref::panel_lu(
        matrix_from_json(child_json.at("local_matrix")),
        child_json.at("pivot_columns").get<std::size_t>());
    expect_size_vector(
        child.pivot_rows, child_json.at("expected_pivot_rows"),
        "tree child pivot rows mismatch");
    expect_size_vector(
        child.permutation, child_json.at("expected_permutation"),
        "tree child permutation mismatch");
    expect_matrix_close(child.l, child_json.at("expected_l"), "tree child L mismatch");
    expect_matrix_close(child.u, child_json.at("expected_u"), "tree child U mismatch");
    expect_matrix_close(
        child.schur_update,
        child_json.at("expected_update"),
        "tree child update mismatch");

    const auto& parent_json = tree.at("parent");
    auto parent_assembled =
        matrix_from_json(parent_json.at("local_matrix_before_extend_add"));
    parent_assembled.at(0, 0) += child.schur_update.at(0, 0);
    expect_matrix_close(
        parent_assembled,
        parent_json.at("expected_assembled_matrix"),
        "tree parent assembly mismatch");
    const auto parent = ref::panel_lu(
        parent_assembled,
        parent_json.at("pivot_columns").get<std::size_t>());
    expect_size_vector(
        parent.pivot_rows, parent_json.at("expected_pivot_rows"),
        "tree parent pivot rows mismatch");
    expect_size_vector(
        parent.permutation, parent_json.at("expected_permutation"),
        "tree parent permutation mismatch");
    expect_matrix_close(parent.l, parent_json.at("expected_l"), "tree parent L mismatch");
    expect_matrix_close(parent.u, parent_json.at("expected_u"), "tree parent U mismatch");

    const std::vector<ref::FrontFactor> fronts{
        {child_json.at("variables").get<std::vector<std::size_t>>(), child},
        {parent_json.at("variables").get<std::vector<std::size_t>>(), parent},
    };
    const auto rhs = tree.at("rhs").get<std::vector<double>>();
    const auto expected_solution =
        tree.at("reference_solution").get<std::vector<double>>();
    const auto solution = ref::solve_front_tree(fronts, rhs);
    expect_vector_close(solution, expected_solution, "tree solve mismatch");

    const auto metrics = ref::compute_accuracy_metrics(
        matrix_from_json(tree.at("global_matrix")), solution, rhs, expected_solution);
    expect(metrics.relative_residual <= 1e-14, "tree relative residual too large");
    expect(metrics.componentwise_backward_error <= 1e-14,
           "tree backward error too large");
    expect(metrics.relative_solution_error <= 1e-14,
           "tree solution error too large");
}

void test_rejections() {
    expect_throws(
        [] {
            (void)ref::panel_lu(ref::DenseMatrix(2, 2, {0.0, 1.0, 0.0, 2.0}), 2);
        },
        "zero pivot column must fail");
    expect_throws(
        [] {
            (void)ref::panel_lu(ref::DenseMatrix(2, 2, {1.0, 0.0, 0.0, 1.0}), 3);
        },
        "invalid pivot count must fail");
    expect_throws(
        [] {
            (void)ref::DenseMatrix(
                1, 1, {std::numeric_limits<double>::infinity()});
        },
        "non-finite input must fail");
    expect_throws(
        [] {
            (void)ref::trsm_right_upper(
                ref::DenseMatrix(1, 1, {1.0}),
                ref::DenseMatrix(1, 1, {0.0}));
        },
        "singular TRSM must fail");
    expect_throws(
        [] {
            (void)ref::gemm_schur(
                ref::DenseMatrix(1, 1, {0.0}),
                ref::DenseMatrix(1, 2, {1.0, 2.0}),
                ref::DenseMatrix(1, 1, {1.0}));
        },
        "GEMM dimension mismatch must fail");
    expect_throws(
        [] {
            const auto cross_boundary = ref::panel_lu(
                ref::DenseMatrix(2, 2, {1.0, 0.0, 2.0, 3.0}), 1);
            const std::vector<ref::FrontFactor> fronts{
                {{0, 1}, cross_boundary},
            };
            (void)ref::solve_front_tree(fronts, {1.0, 1.0});
        },
        "tree solve must reject a cross-front pivot");
}

void expect_pivot_failure(
    const std::function<void()>& action,
    ref::PivotFailure::Kind kind,
    std::size_t column,
    const std::string& message) {
    try {
        action();
    } catch (const ref::PivotFailure& failure) {
        expect(failure.kind() == kind, message + " wrong failure kind");
        expect(failure.column() == column, message + " wrong failure column");
        return;
    }
    throw std::runtime_error(message);
}

void expect_lu_reconstruction(
    const ref::PanelLuResult& result,
    const ref::DenseMatrix& permuted,
    const std::string& message) {
    expect(result.l.rows == permuted.rows, message + " L rows");
    expect(result.u.cols == permuted.cols, message + " U cols");
    for (std::size_t row = 0; row < permuted.rows; ++row) {
        for (std::size_t col = 0; col < permuted.cols; ++col) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < result.l.cols; ++inner) {
                value += result.l.at(row, inner) * result.u.at(inner, col);
            }
            expect(
                close(value, permuted.at(row, col), 1e-12),
                message + " L*U mismatch");
        }
    }
}

void test_policy_no_swap_success() {
    const ref::DenseMatrix front(2, 2, {2.0, 1.0, 0.0, 3.0});
    const auto result = ref::panel_lu_policy(
        front, 2, ref::PivotPolicy::Static);
    expect(result.permutation == std::vector<std::size_t>({0, 1}),
           "static P must be identity without swaps");
    expect(result.pivot_rows == std::vector<std::size_t>({0, 1}),
           "static pivot rows must follow the fixed order");
    expect_lu_reconstruction(result, front, "static triangular");
}

void test_policy_requires_preprocessing_row_swap() {
    const ref::DenseMatrix swap_only(2, 2, {0.0, 1.0, 1.0, 0.0});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(swap_only, 2, ref::PivotPolicy::Static);
        },
        ref::PivotFailure::Kind::ZeroPivot, 0,
        "static on the un-preprocessed matrix must fail");

    const ref::DenseMatrix preprocessed(2, 2, {1.0, 0.0, 0.0, 1.0});
    const auto result = ref::panel_lu_policy(
        preprocessed, 2, ref::PivotPolicy::Static);
    expect(result.permutation == std::vector<std::size_t>({0, 1}),
           "preprocessed static solve must not swap");
    expect_lu_reconstruction(result, preprocessed, "preprocessed static");
}

void test_policy_nonzero_diagonal_can_still_need_interchange() {
    const ref::DenseMatrix dense(3, 3, {1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(dense, 3, ref::PivotPolicy::Static);
        },
        ref::PivotFailure::Kind::ZeroPivot, 1,
        "static must fail at the second pivot of the trap matrix");

    const auto dynamic = ref::panel_lu_policy(dense, 3, ref::PivotPolicy::DynamicPanel);
    expect(dynamic.permutation == std::vector<std::size_t>({0, 2, 1}),
           "dynamic panel P must swap within the pivot block");
    ref::DenseMatrix permuted(3, 3, dense.values);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            permuted.at(row, col) = dense.at(dynamic.permutation[row], col);
        }
    }
    expect_lu_reconstruction(dynamic, permuted, "dynamic trap matrix");

    const std::vector<double> rhs{1.0, 2.0, 3.0};
    const std::vector<ref::FrontFactor> fronts{
        {{0, 1, 2}, dynamic},
    };
    const auto solution = ref::solve_front_tree(fronts, rhs);
    const std::vector<double> expected{-1.0, 2.0, 1.0};
    expect_vector_close(solution, expected, "dynamic trap solution");

    const auto dynamic_again = ref::panel_lu_policy(
        dense, 3, ref::PivotPolicy::DynamicPanel);
    expect(
        dynamic_again.permutation == dynamic.permutation &&
            dynamic_again.l.values == dynamic.l.values &&
            dynamic_again.u.values == dynamic.u.values,
        "same policy must be deterministic");
}

void test_policy_cross_pivot_block_is_rejected() {
    const ref::DenseMatrix front(
        3, 3, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(front, 2, ref::PivotPolicy::DynamicPanel);
        },
        ref::PivotFailure::Kind::NoCandidate, 1,
        "dynamic panel must refuse a pivot that lives past the pivot block");

    const auto full_block = ref::panel_lu_policy(
        front, 3, ref::PivotPolicy::DynamicPanel);
    expect(
        full_block.permutation == std::vector<std::size_t>({0, 2, 1}),
        "full pivot block may interchange inside the block");
}

void test_policy_threshold_nonfinite_and_tie_break() {
    const ref::DenseMatrix near_zero(2, 2, {100.0, 1.0, 1.0, 1e-4});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(
                near_zero, 2, ref::PivotPolicy::Static, 1e-3);
        },
        ref::PivotFailure::Kind::BelowThreshold, 1,
        "static must reject a pivot below the relative threshold");
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(
                near_zero, 2, ref::PivotPolicy::DynamicPanel, 1e-3);
        },
        ref::PivotFailure::Kind::NoCandidate, 1,
        "dynamic must report no admissible pivot inside the block");
    (void)ref::panel_lu_policy(near_zero, 2, ref::PivotPolicy::Static, 1e-6);

    const ref::DenseMatrix overflow_multiplier(
        2, 2, {1e-30, 1e308, 1e308, 1.0});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(
                overflow_multiplier, 2, ref::PivotPolicy::Static);
        },
        ref::PivotFailure::Kind::NonFinite, 0,
        "non-finite multiplier must be reported");

    const ref::DenseMatrix overflow_update(
        2, 2, {-1e308, 1e308, 1e308, 1e308});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(
                overflow_update, 2, ref::PivotPolicy::DynamicPanel);
        },
        ref::PivotFailure::Kind::NonFinite, 0,
        "non-finite trailing update must be reported");

    const ref::DenseMatrix tie_low(
        3, 3, {1.0, 5.0, 5.0, 5.0, 1.0, 2.0, 5.0, 3.0, 4.0});
    const auto tied = ref::panel_lu_policy(
        tie_low, 3, ref::PivotPolicy::DynamicPanel);
    expect(tied.pivot_rows[0] == 1, "tie must pick the lowest logical row");
    const ref::DenseMatrix tie_high(
        3, 3, {5.0, 1.0, 2.0, 5.0, 3.0, 4.0, 1.0, 5.0, 5.0});
    const auto tied_high = ref::panel_lu_policy(
        tie_high, 3, ref::PivotPolicy::DynamicPanel);
    expect(tied_high.pivot_rows[0] == 0, "tie must pick the lowest logical row");
}

void test_policy_static_and_dynamic_may_differ_but_solve_equally() {
    const ref::DenseMatrix dense(2, 2, {1.0, 2.0, 3.0, 4.0});
    const auto static_factor = ref::panel_lu_policy(
        dense, 2, ref::PivotPolicy::Static);
    const auto dynamic_factor = ref::panel_lu_policy(
        dense, 2, ref::PivotPolicy::DynamicPanel);
    expect(
        static_factor.permutation != dynamic_factor.permutation,
        "policies may produce different P");
    expect(
        static_factor.l.values != dynamic_factor.l.values ||
            static_factor.u.values != dynamic_factor.u.values,
        "policies may produce different L/U");

    const std::vector<double> rhs{5.0, 11.0};
    const std::vector<double> expected{1.0, 2.0};
    for (const auto& factor : {static_factor, dynamic_factor}) {
        const auto x = ref::solve_from_lu(factor, rhs);
        expect_vector_close(x, expected, "policy solution");
    }
}

void test_policy_multi_root_forest_static_solve() {
    const std::vector<std::size_t> left_vars{0, 1, 2};
    const std::vector<std::size_t> mid_vars{1, 2};
    const std::vector<std::size_t> leaf_vars{2};
    const std::vector<std::size_t> right_vars{3, 4};
    const std::vector<std::size_t> right_leaf_vars{4};

    const ref::DenseMatrix left(
        3, 3, {4.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0});
    const ref::DenseMatrix mid(2, 2, {3.0, 2.0, 2.0, 0.0});
    const ref::DenseMatrix leaf(1, 1, {5.0});
    const ref::DenseMatrix right(2, 2, {2.0, 0.5, 0.5, 0.0});
    const ref::DenseMatrix right_leaf(1, 1, {1.5});

    auto child = ref::factor_front(
        left_vars, left, 1, ref::PivotPolicy::Static);
    ref::DenseMatrix mid_assembled = mid;
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 2; ++col) {
            mid_assembled.at(row, col) +=
                child.factorization.schur_update.at(row, col);
        }
    }
    auto mid_factor = ref::factor_front(
        mid_vars, mid_assembled, 1, ref::PivotPolicy::Static);
    ref::DenseMatrix leaf_assembled = leaf;
    leaf_assembled.at(0, 0) += mid_factor.factorization.schur_update.at(0, 0);
    auto root = ref::factor_front(
        leaf_vars, leaf_assembled, 1, ref::PivotPolicy::Static);

    auto right_child = ref::factor_front(
        right_vars, right, 1, ref::PivotPolicy::Static);
    ref::DenseMatrix right_assembled = right_leaf;
    right_assembled.at(0, 0) +=
        right_child.factorization.schur_update.at(0, 0);
    auto right_root = ref::factor_front(
        right_leaf_vars, right_assembled, 1, ref::PivotPolicy::Static);

    const std::vector<ref::FrontFactor> fronts{
        child, mid_factor, root, right_child, right_root,
    };
    const std::vector<double> rhs{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto x = ref::solve_front_tree(fronts, rhs);

    ref::DenseMatrix dense(5, 5, std::vector<double>(25, 0.0));
    const double left_values[9]{4.0, 1.0, 0.5, 1.0, 3.0, 2.0, 0.5, 2.0, 5.0};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            dense.at(row, col) = left_values[row * 3 + col];
        }
    }
    dense.at(3, 3) = 2.0;
    dense.at(3, 4) = 0.5;
    dense.at(4, 3) = 0.5;
    dense.at(4, 4) = 1.5;

    const auto residual = ref::relative_residual(dense, x, rhs);
    expect(residual < 1e-12, "multi-root forest residual too large");
    const auto metrics = ref::compute_accuracy_metrics(dense, x, rhs, x);
    expect(metrics.componentwise_backward_error < 1e-12,
           "multi-root forest backward error too large");
}

void test_policy_threshold_normalizes_to_binary32() {
    const ref::DenseMatrix front(2, 2, {1.0, 0.5, 0.0, 0.1000000005});
    expect_pivot_failure(
        [&] {
            (void)ref::panel_lu_policy(front, 2, ref::PivotPolicy::Static, 0.1);
        },
        ref::PivotFailure::Kind::BelowThreshold, 1,
        "host threshold must equal the binary32 device threshold");
    expect_throws(
        [&] {
            (void)ref::panel_lu_policy(front, 2, ref::PivotPolicy::Static, 1e-50);
        },
        "underflowing threshold must fail");
    expect_throws(
        [&] {
            (void)ref::panel_lu_policy(
                front, 2, ref::PivotPolicy::Static, 0.99999999);
        },
        "threshold rounding to 1.0 must fail");
}

void test_policy_static_and_dynamic_factor_scopes() {
    const ref::DenseMatrix front(
        3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0});
    const auto factor = ref::panel_lu_policy(
        front, 2, ref::PivotPolicy::DynamicPanel);
    for (std::size_t row = 0; row < 2; ++row) {
        expect(factor.permutation[row] < 2, "pivot rows stay inside the block");
    }
    expect(factor.permutation[2] == 2, "update rows must not move");
    const auto front_factor = ref::factor_front(
        {0, 1, 2}, front, 2, ref::PivotPolicy::Static);
    expect(
        front_factor.factorization.permutation ==
            std::vector<std::size_t>({0, 1, 2}),
        "static front permutation must be identity");
}

std::string failure_kind_name(ref::PivotFailure::Kind kind) {
    switch (kind) {
    case ref::PivotFailure::Kind::ZeroPivot: return "zero_pivot";
    case ref::PivotFailure::Kind::BelowThreshold: return "below_threshold";
    case ref::PivotFailure::Kind::NonFinite: return "non_finite";
    case ref::PivotFailure::Kind::NoCandidate: return "no_candidate";
    }
    throw std::runtime_error("unknown pivot failure kind");
}

void test_tree_policy_fixture(const json& fixture) {
    expect(
        fixture.at("schema").get<std::string>() == "fp64_reference_tree_v1",
        "unexpected tree fixture schema");
    for (const auto& test_case : fixture.at("cases")) {
        const auto name = test_case.at("name").get<std::string>();
        const auto parent = test_case.at("parent").get<std::vector<int>>();
        std::vector<ref::FrontInput> fronts;
        for (const auto& node : test_case.at("nodes")) {
            ref::FrontInput input{};
            input.variables = node.at("variables").get<std::vector<std::size_t>>();
            input.pivot_columns = node.at("pivot_dim").get<std::size_t>();
            const auto dimension = input.variables.size();
            input.local_front = ref::DenseMatrix(
                dimension, dimension, node.at("front").get<std::vector<double>>());
            fronts.push_back(std::move(input));
        }
        const auto& maps = test_case.at("maps");
        expect(maps.size() == fronts.size(), name + " map table count mismatch");
        for (std::size_t node = 0; node < maps.size(); ++node) {
            for (const auto& entry : maps[node]) {
                ref::ContributionMap map{};
                map.child = entry.at("child").get<std::size_t>();
                map.row_map = entry.at("row_map").get<std::vector<std::size_t>>();
                map.col_map = entry.at("col_map").get<std::vector<std::size_t>>();
                fronts[node].contributions.push_back(std::move(map));
            }
        }
        const auto policy =
            test_case.at("policy").get<std::string>() == "STATIC"
                ? ref::PivotPolicy::Static
                : ref::PivotPolicy::DynamicPanel;
        const auto threshold = test_case.at("threshold_ratio").get<double>();
        const auto rhs = test_case.at("rhs").get<std::vector<double>>();
        const auto& expected = test_case.at("expected");
        try {
            const auto factors =
                ref::factor_front_tree(fronts, parent, policy, threshold);
            expect(
                expected.at("status").get<std::string>() == "ok",
                name + ": expected a pivot failure");
            const auto x = ref::solve_front_tree(factors, rhs);
            expect_vector_close(
                x,
                expected.at("x_hat").get<std::vector<double>>(),
                name + " x_hat");
            const auto& expected_p = expected.at("p_local");
            expect(expected_p.size() == factors.size(), name + " p_local count");
            for (std::size_t node = 0; node < factors.size(); ++node) {
                for (std::size_t row = 0; row < fronts[node].pivot_columns; ++row) {
                    expect(
                        factors[node].factorization.permutation[row] ==
                            expected_p[node][row].get<std::size_t>(),
                        name + " p_local mismatch");
                }
            }
        } catch (const ref::PivotFailure& failure) {
            expect(
                expected.at("status").get<std::string>() == "pivot_failure",
                name + ": unexpected pivot failure");
            expect(
                failure_kind_name(failure.kind()) ==
                    expected.at("kind").get<std::string>(),
                name + " failure kind mismatch");
            expect(
                failure.column() == expected.at("column").get<std::size_t>(),
                name + " failure column mismatch");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error(
                "usage: fp64_reference_tests FIXTURE_JSON TREE_FIXTURE_JSON");
        }
        std::ifstream input(argv[1]);
        if (!input) {
            throw std::runtime_error("cannot open FP64 fixture JSON");
        }
        json fixture;
        input >> fixture;
        expect(
            fixture.at("schema").get<std::string>() ==
                "fp64_reference_fixture_v1",
            "unexpected FP64 fixture schema");

        test_panel_and_dense_solve(fixture);
        test_trsm_and_gemm(fixture);
        test_accuracy_metrics(fixture);
        test_two_front_tree(fixture);
        test_rejections();
        test_policy_no_swap_success();
        test_policy_requires_preprocessing_row_swap();
        test_policy_nonzero_diagonal_can_still_need_interchange();
        test_policy_cross_pivot_block_is_rejected();
        test_policy_threshold_nonfinite_and_tie_break();
        test_policy_static_and_dynamic_may_differ_but_solve_equally();
        test_policy_multi_root_forest_static_solve();
        test_policy_threshold_normalizes_to_binary32();
        test_policy_static_and_dynamic_factor_scopes();
        {
            std::ifstream tree_input(argv[2]);
            if (!tree_input) {
                throw std::runtime_error("cannot open tree fixture JSON");
            }
            json tree_fixture;
            tree_input >> tree_fixture;
            test_tree_policy_fixture(tree_fixture);
        }
        std::cout << "[FP64_REFERENCE_TESTS] ALL PASSED\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FP64_REFERENCE_TESTS] FAILED: " << exception.what() << '\n';
        return 1;
    }
}
