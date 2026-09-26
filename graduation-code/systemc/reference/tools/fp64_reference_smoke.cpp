// Numerical smoke runner for the independent FP64 reference.
//
// Reads a dense row-major FP64 matrix, an RHS and a high-precision reference
// solution (host-side binary dumps), solves, and prints the accuracy metrics
// against the *original* system.  It never touches device images and never
// feeds compiler artifacts.
//
// The default "generic" mode uses the classic dense partial-pivot LU and is
// only an independent solve reference.  With `static` or `dynamic_panel` the
// solve uses the device-contract pivot policy (panel_lu_policy), matching the
// Command/Descriptor semantics; same-policy full-tree cross-validation lives
// in fp64_reference_tests against the shared tree fixture.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fp64_reference.hpp"

namespace {

std::vector<double> read_f64(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    input.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    if (bytes % sizeof(double) != 0) {
        throw std::runtime_error(path + " size is not a multiple of 8 bytes");
    }
    std::vector<std::uint8_t> raw(bytes);
    input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
    std::vector<double> values(bytes / sizeof(double));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::memcpy(&values[index], raw.data() + index * sizeof(double), sizeof(double));
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace hw::reference_fp64;
    try {
        if (argc < 4 || argc > 6) {
            throw std::runtime_error(
                "usage: fp64_reference_smoke MATRIX.bin RHS.bin X_TRUE.bin "
                "[n] [generic|static|dynamic_panel]");
        }
        const auto values = read_f64(argv[1]);
        const auto rhs = read_f64(argv[2]);
        const auto x_true = read_f64(argv[3]);
        const std::size_t n = argc >= 5 ? std::stoul(argv[4]) : rhs.size();
        const std::string mode = argc >= 6 ? argv[5] : "generic";
        if (values.size() != n * n || rhs.size() != n || x_true.size() != n) {
            throw std::runtime_error("input size mismatch");
        }
        const DenseMatrix matrix(n, n, values);
        PanelLuResult factorization{};
        if (mode == "generic") {
            factorization = panel_lu(matrix, n);
        } else if (mode == "static" || mode == "dynamic_panel") {
            factorization = panel_lu_policy(
                matrix, n,
                mode == "static" ? PivotPolicy::Static : PivotPolicy::DynamicPanel);
        } else {
            throw std::runtime_error("unknown solve mode: " + mode);
        }
        const auto x = solve_from_lu(factorization, rhs);
        const auto metrics = compute_accuracy_metrics(matrix, x, rhs, x_true);
        std::cout << "n=" << n << " mode=" << mode
                  << " relative_residual=" << metrics.relative_residual
                  << " componentwise_backward_error="
                  << metrics.componentwise_backward_error
                  << " relative_solution_error=" << metrics.relative_solution_error
                  << " pivot_growth=" << factorization.pivot_growth
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fp64_reference_smoke: " << error.what() << std::endl;
        return 1;
    }
}
