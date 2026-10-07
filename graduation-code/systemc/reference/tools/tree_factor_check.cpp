#include <iostream>
#include "tree_artifact_check.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tree_factor_check DEVICE_DIR REPORT_DIR\n";
        return 2;
    }
    const auto result = hw::reference_fp64::check_tree_artifact(argv[1], argv[2]);
    std::cout << "numeric_pass=" << result.pass << " nodes_checked=" << result.nodes_checked
              << " max_relative_error=" << result.max_relative_error << " reason=" << result.reason << '\n';
    return result.pass ? 0 : 1;
}
