#include <iostream>
#include "tree_solve_check.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tree_solve_check DEVICE_DIR REPORT_DIR\n";
        return 2;
    }
    const auto result = hw::reference_fp64::check_tree_solve_artifact(argv[1], argv[2]);
    std::cout << "solve_pass=" << result.pass
              << " eta_forward=" << result.eta_forward
              << " eta_backward=" << result.eta_backward
              << " transformed_backward_error=" << result.transformed_backward_error
              << " reference_solution_relative_error="
              << result.reference_solution_relative_error
              << " original_relative_residual="
              << result.original_relative_residual
              << " original_backward_error=" << result.original_backward_error
              << " original_solution_relative_error="
              << result.original_solution_relative_error
              << " solve_commands=" << result.solve_commands
              << " solve_commands_ok=" << result.solve_commands_ok
              << " reason=" << result.reason << '\n';
    return result.pass ? 0 : 1;
}
