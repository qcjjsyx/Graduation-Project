#pragma once
#include <cstdint>
#include <string>

namespace hw::reference_fp64 {
struct TreeValidation {
    bool pass{false};
    std::uint32_t nodes_checked{0};
    double max_relative_error{0};
    std::string reason;
};
// Checker only: read immutable compiler inputs and device DDR output, never
// feed reference fronts/factors/pivots into CommandDevice.
TreeValidation check_tree_artifact(const std::string& device_dir,
                                   const std::string& report_dir,
                                   double tolerance = 1e-4);
}
