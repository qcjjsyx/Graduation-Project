// T12 revision: checker-level test of the P-vector convention used by the
// triangular backward errors of tree_solve_check (design doc §5 items 1-2).
//
// Same node data as solve_real_tests' test_pvector_three_cycle_manual (3-cycle
// gather P-vector, non-trivial L11, update segment), embedded in a two-node
// tree so that the global system is complete:
//   node A: variables g = {1,2,3,5,7}, p = 3, P = [1,2,0]
//   node B: variables {5,7} (A's update variables), p = 2, L = U = I, P = id
// All values are hand-computed below; neither the device nor the checker
// produces its own expectation.
//
// Checks:
//   - consistent data: eta_f = eta_b = 0 (exact dyadic data) <= 1e-6, pass;
//   - the same data with A's P-vector replaced by its inverse [2,0,1]:
//     eta_f > 1e-3 and the numeric pass is false.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "tree_solve_check.hpp"

namespace {

namespace ref = hw::reference_fp64;

int checks = 0;
int failures = 0;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

ref::SolveNodeData node_a() {
    ref::SolveNodeData node;
    node.variables = {1, 2, 3, 5, 7};
    node.total = 5;
    node.pivot = 3;
    // L11 = [[1,0,0],[2,1,0],[-1,0.5,1]], L21 = [[1,0,2],[0.5,-1,0]].
    node.l = {
        1.0,  0.0, 0.0,
        2.0,  1.0, 0.0,
        -1.0, 0.5, 1.0,
        1.0,  0.0, 2.0,
        0.5, -1.0, 0.0,
    };
    // U11 = [[2,1,-1],[0,4,2],[0,0,-2]], U12 = [[1,0],[0,-1],[0.5,0]].
    node.u = {
        2.0, 1.0, -1.0, 1.0,  0.0,
        0.0, 4.0,  2.0, 0.0, -1.0,
        0.0, 0.0, -2.0, 0.5,  0.0,
    };
    node.p = {1, 2, 0};
    // Forward z (hand computation in solve_real_tests): [3, -5, 9.5].
    node.z = {3.0, -5.0, 9.5};
    return node;
}

ref::SolveNodeData node_b() {
    ref::SolveNodeData node;
    node.variables = {5, 7};
    node.total = 2;
    node.pivot = 2;
    node.l = {1.0, 0.0, 0.0, 1.0};
    node.u = {1.0, 0.0, 0.0, 1.0};
    node.p = {0, 1};
    // z_B = b_hat[{5,7}] - L21_A z_A = [0.5 - 22, -1.5 - 6.5] = [-21.5, -8].
    node.z = {-21.5, -8.0};
    return node;
}

// b_hat, by global variable 0..7:
//   A's pivot rows: L~ row of local row i sits at g[p[i]]:
//     i=0 -> g[1] = var 2: L11[0] . z = 3
//     i=1 -> g[2] = var 3: 2*3 + (-5) = 1
//     i=2 -> g[0] = var 1: -3 + 0.5*(-5) + 9.5 = 4
//     so b_hat[1..3] = [4, 3, 1]  (= c in the unit test: b + w0 per variable)
//   var 5: L21_A[0] . z_A + z_B[0] = 22 + (-21.5) = 0.5
//   var 7: L21_A[1] . z_A + z_B[1] = 6.5 + (-8)   = -1.5
//   variables 0, 4, 6 are owned by no node: b_hat = x_hat = 0 there.
const std::vector<double> kBHat = {0.0, 4.0, 3.0, 1.0, 0.0, 0.5, 0.0, -1.5};
// x_hat: B gives x5 = -21.5, x7 = -8; A's backward (unit test) gives
// x1..x3 = [6.28125, 1.8125, -10.125].
const std::vector<double> kXHat = {
    0.0, 6.28125, 1.8125, -10.125, 0.0, -21.5, 0.0, -8.0};

void test_consistent_three_cycle() {
    const std::vector<ref::SolveNodeData> nodes = {node_a(), node_b()};
    const auto errors = ref::triangular_backward_errors(nodes, kBHat, kXHat);
    expect(
        errors.eta_forward <= 1e-6,
        "consistent 3-cycle: eta_forward <= 1e-6 (got " +
            std::to_string(errors.eta_forward) + ")");
    expect(
        errors.eta_backward <= 1e-6,
        "consistent 3-cycle: eta_backward <= 1e-6 (got " +
            std::to_string(errors.eta_backward) + ")");
    expect(
        ref::triangular_errors_within(errors, 1e-4),
        "consistent 3-cycle: numeric solve_pass is true");
}

void test_inverse_pvector_is_rejected() {
    auto a = node_a();
    // Inverse of [1,2,0]: q[p[i]] = i -> [2,0,1].  With q, local row 0 lands
    // on g[2] = var 3: lhs 3 vs b_hat 1 -> |2| / (3 + 1) = 0.5.
    a.p = {2, 0, 1};
    const std::vector<ref::SolveNodeData> nodes = {a, node_b()};
    const auto errors = ref::triangular_backward_errors(nodes, kBHat, kXHat);
    expect(
        errors.eta_forward > 1e-3,
        "inverse P: eta_forward > 1e-3 (got " + std::to_string(errors.eta_forward) + ")");
    expect(
        !ref::triangular_errors_within(errors, 1e-4),
        "inverse P: numeric solve_pass is false");
}

}  // namespace

int main() {
    test_consistent_three_cycle();
    test_inverse_pvector_is_rejected();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
