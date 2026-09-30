#pragma once

// Formal FP32 tile numerical primitives (T07). These are the production
// primitives used by the real Panel engine and reused by T09/T10:
//   - STATIC packed LU_DIAG with post-elimination pivot checks,
//   - unit-lower TRSM_LEFT (forward substitution),
//   - non-unit-upper TRSM_RIGHT (reciprocal precompute + multiply substitution),
//   - GEMM_UPDATE with fused multiply-add.
//
// Semantics: explicit FP32 FMA for all multiply-accumulate updates, gradual
// underflow (subnormals are preserved, never flushed to zero), no fast-math,
// and per-column finiteness checks so the first failing column/stage is
// reported consistently with the modeled execution-time contract. The T06
// architecture prototype calls the same primitives on synthetic workloads;
// the primitive semantics themselves are the production ones.

#include <cstdint>
#include <string>

#include "panel_lu/task_types.hpp"

namespace hw::panel_lu::kernels {

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

// Explicit FP64 module diagnostic primitives: same algorithms, same status
// semantics and failure columns, but binary64 accumulation. Only reachable
// when PanelNumericMode::Fp64Diagnostic is declared; results are rounded back
// to binary32 storage by the caller.
namespace fp64_diagnostic {

KernelOutcome run_lu_diag(
    double* data,
    std::uint32_t stride_floats,
    std::uint32_t k,
    double threshold_product,
    std::uint32_t pivot_base);

KernelOutcome run_trsm_left(
    double* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t k,
    std::uint32_t n,
    const double* l,
    std::uint32_t l_stride);

KernelOutcome run_trsm_right(
    double* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t k,
    const double* u,
    std::uint32_t u_stride);

KernelOutcome run_gemm_update(
    double* dst,
    std::uint32_t dst_stride,
    const double* lhs,
    std::uint32_t lhs_stride,
    const double* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k);

}  // namespace fp64_diagnostic

}  // namespace hw::panel_lu::kernels
