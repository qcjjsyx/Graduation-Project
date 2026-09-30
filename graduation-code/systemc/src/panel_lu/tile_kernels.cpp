#include "panel_lu/tile_kernels.hpp"

#include <cmath>

namespace hw::panel_lu::kernels {

namespace {

bool finite_f32(float value) {
    return std::isfinite(value);
}

}  // namespace

KernelOutcome run_lu_diag(
    float* data,
    std::uint32_t stride_floats,
    std::uint32_t k,
    double threshold_product,
    std::uint32_t pivot_base) {
    KernelOutcome outcome;
    for (std::uint32_t j = 0; j < k; ++j) {
        float* pivot_row = data + static_cast<std::size_t>(j) * stride_floats;
        const float pivot = pivot_row[j];
        if (!finite_f32(pivot)) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = pivot_base + j;
            outcome.reason = "non-finite LU_DIAG pivot";
            return outcome;
        }
        if (pivot == 0.0f) {
            outcome.status = StatusCode::StaticPivotFailure;
            outcome.failed_column = pivot_base + j;
            outcome.reason = "zero pivot after elimination";
            return outcome;
        }
        if (std::fabs(static_cast<double>(pivot)) <= threshold_product) {
            outcome.status = StatusCode::PivotUnstable;
            outcome.failed_column = pivot_base + j;
            outcome.reason = "pivot at or below threshold * front_pre_elimination_max_abs";
            return outcome;
        }
        const float inverse = 1.0f / pivot;
        // A finite non-zero pivot can still overflow its reciprocal (subnormal
        // magnitude). Report it at this column, consistent with the stage
        // timing, instead of letting Inf leak into later pivots.
        if (!finite_f32(inverse)) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = pivot_base + j;
            outcome.reason = "LU_DIAG pivot reciprocal overflowed to a non-finite value";
            return outcome;
        }
        for (std::uint32_t i = j + 1; i < k; ++i) {
            float* row = data + static_cast<std::size_t>(i) * stride_floats;
            const float multiplier = row[j] * inverse;
            row[j] = multiplier;
            for (std::uint32_t t = j + 1; t < k; ++t) {
                row[t] = std::fma(-multiplier, pivot_row[t], row[t]);
            }
        }
    }
    return outcome;
}

KernelOutcome run_trsm_left(
    float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t k,
    std::uint32_t n,
    const float* l,
    std::uint32_t l_stride) {
    KernelOutcome outcome;
    for (std::uint32_t i = 0; i < k; ++i) {
        float* row = rhs + static_cast<std::size_t>(i) * rhs_stride;
        for (std::uint32_t t = 0; t < i; ++t) {
            const float scale = l[static_cast<std::size_t>(i) * l_stride + t];
            const float* prev = rhs + static_cast<std::size_t>(t) * rhs_stride;
            for (std::uint32_t c = 0; c < n; ++c) {
                row[c] = std::fma(-scale, prev[c], row[c]);
            }
        }
        for (std::uint32_t c = 0; c < n; ++c) {
            if (!finite_f32(row[c])) {
                outcome.status = StatusCode::NonFiniteValue;
                outcome.failed_column = i;
                outcome.reason = "non-finite value produced by TRSM_LEFT";
                return outcome;
            }
        }
    }
    return outcome;
}

KernelOutcome run_trsm_right(
    float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t k,
    const float* u,
    std::uint32_t u_stride) {
    KernelOutcome outcome;
    // Phase 1: reciprocal precompute through the corner divide/reciprocal path
    // (pipelined, one issue per cycle). All U[j][j] are available before the
    // solve starts, so the k reciprocals are independent; a zero/non-finite
    // diagonal OR a non-finite reciprocal (finite subnormal diagonal whose
    // reciprocal overflows) is a *precompute* failure (diagonal_failure = true)
    // reported at the first affected column.
    std::vector<float> reciprocal(k, 0.0f);
    for (std::uint32_t j = 0; j < k; ++j) {
        const float diagonal = u[static_cast<std::size_t>(j) * u_stride + j];
        if (diagonal == 0.0f) {
            outcome.status = StatusCode::StaticPivotFailure;
            outcome.failed_column = j;
            outcome.reason = "zero diagonal in TRSM_RIGHT reciprocal precompute";
            outcome.diagonal_failure = true;
            return outcome;
        }
        if (!finite_f32(diagonal)) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = j;
            outcome.reason = "non-finite diagonal in TRSM_RIGHT reciprocal precompute";
            outcome.diagonal_failure = true;
            return outcome;
        }
        reciprocal[j] = 1.0f / diagonal;
        // Subnormal inputs are legal; the reciprocal itself must still be
        // finite. Without this check a tiny finite diagonal would be detected
        // late (during the solve) or mask an earlier zero diagonal.
        if (!finite_f32(reciprocal[j])) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = j;
            outcome.reason =
                "reciprocal of TRSM_RIGHT diagonal overflowed to a non-finite value";
            outcome.diagonal_failure = true;
            return outcome;
        }
    }

    // Phase 2: multiply-only forward substitution over increasing j:
    //   x[j] = (a[j] - sum_{t<j} x[t] * U[t][j]) * reciprocal[j]
    // Previously computed x[t] (t < j) feed back into the sum, so the direction
    // must be increasing j. A non-finite result here is a solve failure.
    for (std::uint32_t i = 0; i < m; ++i) {
        float* row = rhs + static_cast<std::size_t>(i) * rhs_stride;
        for (std::uint32_t j = 0; j < k; ++j) {
            float acc = row[j];
            for (std::uint32_t t = 0; t < j; ++t) {
                acc = std::fma(-row[t], u[static_cast<std::size_t>(t) * u_stride + j], acc);
            }
            row[j] = acc * reciprocal[j];
            if (!finite_f32(row[j])) {
                outcome.status = StatusCode::NonFiniteValue;
                outcome.failed_column = j;
                outcome.reason = "non-finite value produced by TRSM_RIGHT solve";
                outcome.diagonal_failure = false;
                return outcome;
            }
        }
    }
    return outcome;
}

KernelOutcome run_gemm_update(
    float* dst,
    std::uint32_t dst_stride,
    const float* lhs,
    std::uint32_t lhs_stride,
    const float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    KernelOutcome outcome;
    for (std::uint32_t i = 0; i < m; ++i) {
        float* dst_row = dst + static_cast<std::size_t>(i) * dst_stride;
        const float* lhs_row = lhs + static_cast<std::size_t>(i) * lhs_stride;
        for (std::uint32_t c = 0; c < n; ++c) {
            float acc = dst_row[c];
            for (std::uint32_t t = 0; t < k; ++t) {
                acc = std::fma(-lhs_row[t], rhs[static_cast<std::size_t>(t) * rhs_stride + c], acc);
            }
            dst_row[c] = acc;
            if (!finite_f32(acc)) {
                outcome.status = StatusCode::NonFiniteValue;
                outcome.failed_column = c;
                outcome.reason = "non-finite value produced by GEMM_UPDATE";
                return outcome;
            }
        }
    }
    return outcome;
}

}  // namespace hw::panel_lu::kernels

namespace hw::panel_lu::kernels::fp64_diagnostic {

namespace {

bool finite_f64(double value) {
    return std::isfinite(value);
}

}  // namespace

KernelOutcome run_lu_diag(
    double* data,
    std::uint32_t stride_floats,
    std::uint32_t k,
    double threshold_product,
    std::uint32_t pivot_base) {
    KernelOutcome outcome;
    for (std::uint32_t j = 0; j < k; ++j) {
        double* pivot_row = data + static_cast<std::size_t>(j) * stride_floats;
        const double pivot = pivot_row[j];
        if (!finite_f64(pivot)) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = pivot_base + j;
            outcome.reason = "non-finite LU_DIAG pivot (fp64 diagnostic)";
            return outcome;
        }
        if (pivot == 0.0) {
            outcome.status = StatusCode::StaticPivotFailure;
            outcome.failed_column = pivot_base + j;
            outcome.reason = "zero pivot after elimination (fp64 diagnostic)";
            return outcome;
        }
        if (std::fabs(pivot) <= threshold_product) {
            outcome.status = StatusCode::PivotUnstable;
            outcome.failed_column = pivot_base + j;
            outcome.reason =
                "pivot at or below threshold * front_pre_elimination_max_abs "
                "(fp64 diagnostic)";
            return outcome;
        }
        const double inverse = 1.0 / pivot;
        if (!finite_f64(inverse)) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = pivot_base + j;
            outcome.reason =
                "LU_DIAG pivot reciprocal overflowed (fp64 diagnostic)";
            return outcome;
        }
        for (std::uint32_t i = j + 1; i < k; ++i) {
            double* row = data + static_cast<std::size_t>(i) * stride_floats;
            const double multiplier = row[j] * inverse;
            row[j] = multiplier;
            for (std::uint32_t t = j + 1; t < k; ++t) {
                row[t] = std::fma(-multiplier, pivot_row[t], row[t]);
            }
        }
    }
    return outcome;
}

KernelOutcome run_trsm_left(
    double* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t k,
    std::uint32_t n,
    const double* l,
    std::uint32_t l_stride) {
    KernelOutcome outcome;
    for (std::uint32_t i = 0; i < k; ++i) {
        double* row = rhs + static_cast<std::size_t>(i) * rhs_stride;
        for (std::uint32_t t = 0; t < i; ++t) {
            const double scale = l[static_cast<std::size_t>(i) * l_stride + t];
            const double* prev = rhs + static_cast<std::size_t>(t) * rhs_stride;
            for (std::uint32_t c = 0; c < n; ++c) {
                row[c] = std::fma(-scale, prev[c], row[c]);
            }
        }
        for (std::uint32_t c = 0; c < n; ++c) {
            if (!finite_f64(row[c])) {
                outcome.status = StatusCode::NonFiniteValue;
                outcome.failed_column = i;
                outcome.reason =
                    "non-finite value produced by TRSM_LEFT (fp64 diagnostic)";
                return outcome;
            }
        }
    }
    return outcome;
}

KernelOutcome run_trsm_right(
    double* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t k,
    const double* u,
    std::uint32_t u_stride) {
    KernelOutcome outcome;
    std::vector<double> reciprocal(k, 0.0);
    for (std::uint32_t j = 0; j < k; ++j) {
        const double diagonal = u[static_cast<std::size_t>(j) * u_stride + j];
        if (diagonal == 0.0) {
            outcome.status = StatusCode::StaticPivotFailure;
            outcome.failed_column = j;
            outcome.reason =
                "zero diagonal in TRSM_RIGHT reciprocal precompute (fp64 diagnostic)";
            outcome.diagonal_failure = true;
            return outcome;
        }
        if (!finite_f64(diagonal)) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = j;
            outcome.reason =
                "non-finite diagonal in TRSM_RIGHT precompute (fp64 diagnostic)";
            outcome.diagonal_failure = true;
            return outcome;
        }
        reciprocal[j] = 1.0 / diagonal;
        if (!finite_f64(reciprocal[j])) {
            outcome.status = StatusCode::NonFiniteValue;
            outcome.failed_column = j;
            outcome.reason =
                "reciprocal of TRSM_RIGHT diagonal overflowed (fp64 diagnostic)";
            outcome.diagonal_failure = true;
            return outcome;
        }
    }
    for (std::uint32_t i = 0; i < m; ++i) {
        double* row = rhs + static_cast<std::size_t>(i) * rhs_stride;
        for (std::uint32_t j = 0; j < k; ++j) {
            double acc = row[j];
            for (std::uint32_t t = 0; t < j; ++t) {
                acc = std::fma(
                    -row[t], u[static_cast<std::size_t>(t) * u_stride + j], acc);
            }
            row[j] = acc * reciprocal[j];
            if (!finite_f64(row[j])) {
                outcome.status = StatusCode::NonFiniteValue;
                outcome.failed_column = j;
                outcome.reason =
                    "non-finite value produced by TRSM_RIGHT (fp64 diagnostic)";
                outcome.diagonal_failure = false;
                return outcome;
            }
        }
    }
    return outcome;
}

KernelOutcome run_gemm_update(
    double* dst,
    std::uint32_t dst_stride,
    const double* lhs,
    std::uint32_t lhs_stride,
    const double* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k) {
    KernelOutcome outcome;
    for (std::uint32_t i = 0; i < m; ++i) {
        double* dst_row = dst + static_cast<std::size_t>(i) * dst_stride;
        const double* lhs_row = lhs + static_cast<std::size_t>(i) * lhs_stride;
        for (std::uint32_t c = 0; c < n; ++c) {
            double acc = dst_row[c];
            for (std::uint32_t t = 0; t < k; ++t) {
                acc = std::fma(
                    -lhs_row[t],
                    rhs[static_cast<std::size_t>(t) * rhs_stride + c],
                    acc);
            }
            dst_row[c] = acc;
            if (!finite_f64(acc)) {
                outcome.status = StatusCode::NonFiniteValue;
                outcome.failed_column = c;
                outcome.reason =
                    "non-finite value produced by GEMM_UPDATE (fp64 diagnostic)";
                return outcome;
            }
        }
    }
    return outcome;
}

}  // namespace hw::panel_lu::kernels::fp64_diagnostic
