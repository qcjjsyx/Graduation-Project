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

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: fp64_reference_tests FIXTURE_JSON");
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
        std::cout << "[FP64_REFERENCE_TESTS] ALL PASSED\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FP64_REFERENCE_TESTS] FAILED: " << exception.what() << '\n';
        return 1;
    }
}
