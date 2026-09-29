#pragma once

// Synthetic FP32 tile kernels for the T06 architecture prototype ONLY.
//
// These kernels exist so that the prototype can produce data-dependent pivot
// failures and dependency-consistent tile versions for explicitly synthetic
// workloads. They are *not* a device numerical backend, are not validated
// against the FP64 reference, do not implement the frozen rounding/FMA
// contract of the production path and must never be wired into a real
// CommandDevice execution slot. Prototype runs are marked
// architecture_prototype / numerical_validation=not_performed.

#include <cstdint>
#include <string>

#include "panel_lu/task_types.hpp"

namespace hw::panel_lu::synthetic {

struct KernelOutcome {
    StatusCode status{StatusCode::Ok};
    std::uint32_t failed_column{0};
    // True when a TRSM_RIGHT failure was detected during the reciprocal
    // precompute phase (bad diagonal) rather than during the solve phase.
    bool diagonal_failure{false};
    std::string reason;
};

// Static packed LU of one diagonal tile (row stride = stride_floats).
// Pivots are checked after elimination: NaN/Inf -> NON_FINITE_VALUE, exact
// zero -> STATIC_PIVOT_FAILURE, |pivot| <= threshold_product -> PIVOT_UNSTABLE.
// The unit lower diagonal is implicit; the diagonal stores U.
KernelOutcome run_lu_diag(
    float* data,
    std::uint32_t stride_floats,
    std::uint32_t k,
    double threshold_product,
    std::uint32_t pivot_base);

// X = L^-1 * A with unit lower triangular L (k x k, implicit unit diagonal)
// packed in `l`, and right-hand-side `rhs` (k x n) modified in place.
KernelOutcome run_trsm_left(
    float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t k,
    std::uint32_t n,
    const float* l,
    std::uint32_t l_stride);

// X = A * U^-1 with upper triangular U (k x k, non-unit diagonal) packed in
// `u`, and `rhs` (m x k) modified in place.
KernelOutcome run_trsm_right(
    float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t k,
    const float* u,
    std::uint32_t u_stride);

// dst (m x n) -= lhs (m x k) * rhs (k x n) using fused multiply-add.
KernelOutcome run_gemm_update(
    float* dst,
    std::uint32_t dst_stride,
    const float* lhs,
    std::uint32_t lhs_stride,
    const float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k);

}  // namespace hw::panel_lu::synthetic
