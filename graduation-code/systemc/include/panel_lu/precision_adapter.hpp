#pragma once

// T10 local PrecisionAdapter + INT32 GEMM behavior model.
//
// Scope: ONLY the top-level GEMM_SCHUR command may switch to the local INT32
// path. Panel/TRSM internal GemmUpdate tasks always stay FP32; the backend is
// an explicit property of the command/tile, never inferred from the tile mode
// or from the stored data format.
//
// Data model (documented; report must use the same wording):
//   INT32 storage / 30 effective quantization bits / INT64 accumulator.
// Default effective range is symmetric: q in [-(2^29-1), +(2^29-1)]. The
// supported configuration must satisfy
//   k_block_max * (2^(bits-1)-1)^2 < 2^(accumulator_bits-2)
// so the default 16-term dot product is < 2^62 and can never overflow the
// INT64 accumulator. This is NOT a claim that an arbitrary full INT32 range is
// safe; the accumulation primitive still checks every partial step explicitly
// (no signed overflow, no post-hoc inference) and reports NUMERIC_OVERFLOW.
//
// Per K-block task:
//   load FP32 A/B -> maxabs (per operand) -> power-of-two scale -> quantize
//   -> INT32xINT32 with INT64 accumulation -> dequantize one rounded FP32
//   partial product -> FP32 C := C - partial.
//
// Rounding: round-to-nearest, ties-to-even, implemented with exact integer
// arithmetic on the dyadic value x / 2^e, so it does not depend on the host
// process rounding mode. Finiteness and range checks happen BEFORE any
// float-to-integer cast. Dequantization is the correctly rounded binary32 of
// the exact integer value times 2^(eA+eB) (gradual underflow, overflow to
// infinity is reported as NON_FINITE_VALUE).

#include <cstdint>
#include <string>
#include <vector>

#include "panel_lu/gemm_types.hpp"
#include "panel_lu/task_types.hpp"

namespace hw::panel_lu {

using hw::dataplane::GemmScalePolicy;
using hw::dataplane::ScaleSaturationPolicy;
using hw::dataplane::kScaleExponentMax;
using hw::dataplane::kScaleExponentMin;

// PrecisionConfig is declared in panel_lu/gemm_types.hpp (included above) so
// the engine configuration can embed it without a header cycle.

// Validates the whole precision configuration. Rejects zero rates, unsupported
// accumulator widths, effective_bits > 30 (unsafe 16-term dot product) and any
// k_block_max whose supported dot products could not be bounded.
ConfigCheck validate_precision_config(const PrecisionConfig& config);

// ---------------------------------------------------------------------------
// Element primitives (used by the tile kernel and by independent tests)
// ---------------------------------------------------------------------------

// Smallest exponent e (power-of-two scale 2^e) such that every element of the
// tile can be represented with |q| <= qmax = 2^(bits-1)-1. maxabs == 0 returns
// the deterministic e = 0 (all-zero integers). The selection is exact integer
// arithmetic around the only two candidate exponents, so power boundaries,
// subnormals and values near the FP32 maximum are handled without an
// uncorrected ceil(log2(...)).
std::int32_t choose_power_of_two_exponent(float maxabs, std::uint32_t effective_bits);

// Exact round-to-nearest, ties-to-even of the dyadic value x / 2^exponent.
// The caller must have verified that x is finite and |x / 2^e| <= 2^31 (range
// checks before the cast; out-of-range casts are forbidden).
std::int64_t round_half_to_even_scaled(float x, std::int32_t exponent);

// Quantizes one element. Returns:
//   Ok                   - q written,
//   NonFiniteValue       - input not finite (checked first),
//   QuantizationSaturation - |round(x/2^e)| > qmax under the FAIL policy
//                          (q is set to the clamped value for determinism but
//                          the caller must not commit the tile);
// with CLAMP_AND_REPORT the clamped value is returned as Ok and
// *saturation_count is incremented.
StatusCode quantize_element(
    float x,
    std::int32_t exponent,
    const PrecisionConfig& config,
    ScaleSaturationPolicy policy,
    std::int32_t* q,
    std::uint64_t* saturation_count);

// Correctly rounded binary32 of the exact value `value * 2^exponent`
// (round-to-nearest, ties-to-even, gradual underflow). Overflow returns
// +/-infinity; the caller maps a non-finite result to NON_FINITE_VALUE. This is
// the documented dequantization of one exact INT64 partial dot product.
float fp32_from_int64_scaled(std::int64_t value, std::int32_t exponent);

// Explicit widened accumulation with a per-step range check (no signed
// overflow, no post-hoc inference). Returns false when the next partial sum no
// longer fits -2^(guard_bits-1) <= acc < 2^(guard_bits-1) with guard_bits in
// [1,64]; guard_bits = 64 is exactly the INT64 range.
bool checked_accumulate(
    std::int64_t qa,
    std::int64_t qb,
    std::uint32_t guard_bits,
    std::int64_t* accumulator,
    std::uint64_t* overflow_count);

// Full INT32 dot product with the same explicit guard. `guard_bits` is
// validated at the primitive boundary so tests can exercise full-range INT32
// operands (where the configured effective range would reject them) and verify
// that the guard itself cannot silently overflow.
StatusCode int32_dot_checked(
    const std::int32_t* a,
    const std::int32_t* b,
    std::uint32_t k,
    std::uint32_t guard_bits,
    std::int64_t* result);

// ---------------------------------------------------------------------------
// Tile-level INT32 GEMM update
// ---------------------------------------------------------------------------

// Per-phase model cycles and counters produced by one local tile task.
struct Int32GemmStats {
    std::uint64_t scale_reduction_cycles{0};
    std::uint64_t quantize_cycles{0};
    std::uint64_t integer_compute_cycles{0};
    std::uint64_t dequantize_cycles{0};
    std::uint64_t saturation_count{0};
    std::uint64_t overflow_count{0};
    // PE-local conversion bytes (maxabs read + quantize read/write + dequant
    // read/update); never counted as DDR or scratchpad traffic.
    std::uint64_t local_conversion_bytes{0};
    std::int32_t exponent_a{0};
    std::int32_t exponent_b{0};
    GemmPhase failed_phase{GemmPhase::None};
    std::uint32_t failed_local_row{0};
    std::uint32_t failed_local_col{0};
};

// dst (m x n) := dst - dequantize(quantize(lhs) * quantize(rhs)).
// Operands are the staged FP32 tiles (row stride in floats). Only the effective
// m/n/k region is touched. Failure semantics:
//   NonFiniteValue         - non-finite input or non-finite dequantized/output
//                            value,
//   QuantizationSaturation - saturation under the FAIL policy (no partial
//                            writeback; the caller must not commit the tile),
//   NumericOverflow        - accumulation exceeded the configured range,
//   BadDescriptor          - k outside [1, k_block_max],
//   PolicyNotSupported     - a fixed exponent outside the supported range or a
//                            fixed output scale.
// `stats` (optional) receives the per-phase model cycles and counters. The
// update is applied element-wise in row-major order; a failure leaves already
// updated elements of this tile present in the scratch buffer, but the tile is
// never written back / committed on failure.
StatusCode run_int32_gemm_update(
    float* dst,
    std::uint32_t dst_stride,
    const float* lhs,
    std::uint32_t lhs_stride,
    const float* rhs,
    std::uint32_t rhs_stride,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const PrecisionConfig& config,
    const PeTimingConfig& timing,
    const GemmScalePolicy& scale,
    Int32GemmStats* stats,
    std::string* reason);

// Model cycles of a successful INT32 tile task (conversion phases + integer
// array occupancy; the integer phase reuses the PE array pipeline parameters
// fill/k_ii/drain_row_cycles). `k` must not exceed config.k_block_max.
std::uint64_t int32_gemm_exec_cycles(
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const PrecisionConfig& config,
    const PeTimingConfig& timing);

// Model cycles spent before a mid-task failure is detected. `failed_phase`
// selects the conversion stage; `failed_element_index` is the 0-based index of
// the failing element in that stage's processing order (A operand elements
// first, then B; output elements for the dequantize stage), so the operand
// identity never has to be inferred from a row value.
std::uint64_t int32_gemm_failure_cycles(
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const PrecisionConfig& config,
    const PeTimingConfig& timing,
    GemmPhase failed_phase,
    std::uint32_t failed_element_index);

}  // namespace hw::panel_lu
