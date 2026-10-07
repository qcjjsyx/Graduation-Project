#include "panel_lu/precision_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace hw::panel_lu {

namespace {

std::uint64_t ceil_div(std::uint64_t value, std::uint32_t rate) {
    return rate == 0 ? value : (value + rate - 1) / rate;
}

// Round the checked dyadic value y (|y| <= 2^31) to nearest, ties-to-even.
// Implemented with floor/fraction comparison on an exactly representable
// dyadic value, so it never depends on the process rounding mode.
std::int64_t round_dyadic_half_to_even(double y) {
    const double floor_value = std::floor(y);
    const double fraction = y - floor_value;
    const std::int64_t base = static_cast<std::int64_t>(floor_value);
    if (fraction > 0.5) {
        return base + 1;
    }
    if (fraction < 0.5) {
        return base;
    }
    return (base % 2 == 0) ? base : base + 1;
}

bool finite_f32(float value) {
    return std::isfinite(value);
}

}  // namespace

ConfigCheck validate_precision_config(const PrecisionConfig& config) {
    ConfigCheck check;
    if (config.effective_bits < 1 || config.effective_bits > 30) {
        check.ok = false;
        check.status = StatusCode::PolicyNotSupported;
        check.reason =
            "effective quantization bits must be in [1, 30]; INT32 storage / "
            "30 effective bits is the largest range whose 16-term dot product "
            "is provably < 2^62";
        return check;
    }
    if (config.accumulator_bits != 64) {
        check.ok = false;
        check.status = StatusCode::PolicyNotSupported;
        check.reason = "only the INT64 accumulator (64) is supported";
        return check;
    }
    if (config.k_block_max < 1 || config.k_block_max > 16) {
        check.ok = false;
        check.status = StatusCode::PolicyNotSupported;
        check.reason = "K block must be in [1, 16]";
        return check;
    }
    const std::int64_t qmax = config.saturation_bound();
    const __int128 worst =
        static_cast<__int128>(config.k_block_max) *
        static_cast<__int128>(qmax) * static_cast<__int128>(qmax);
    const __int128 safe_limit = static_cast<__int128>(1)
                                << (config.accumulator_bits - 2);
    if (worst >= safe_limit) {
        check.ok = false;
        check.status = StatusCode::PolicyNotSupported;
        check.reason =
            "unsafe quantization config: k_block_max * qmax^2 must stay below "
            "2^(accumulator_bits-2)";
        return check;
    }
    if (config.maxabs_elements_per_cycle == 0 ||
        config.quantize_elements_per_cycle == 0 ||
        config.dequantize_elements_per_cycle == 0) {
        check.ok = false;
        check.status = StatusCode::BadDescriptor;
        check.reason = "conversion rates must be >= 1 element per cycle";
        return check;
    }
    check.ok = true;
    return check;
}

std::int32_t choose_power_of_two_exponent(
    float maxabs,
    std::uint32_t effective_bits) {
    if (maxabs == 0.0f) {
        return 0;  // deterministic all-zero representation
    }
    if (!std::isfinite(maxabs)) {
        return 0;  // callers reject non-finite operands before scale selection
    }
    if (effective_bits < 1 || effective_bits > 30) {
        return 0;
    }
    const std::int64_t qmax =
        (static_cast<std::int64_t>(1) << (effective_bits - 1)) - 1;
    int significand_exponent = 0;
    const double fraction = std::frexp(
        static_cast<double>(std::fabs(maxabs)), &significand_exponent);
    (void)fraction;
    // maxabs = f * 2^m with f in [0.5, 1). The smallest candidate is
    // e0 = m - (bits-1): qmax * 2^e0 = 2^m - 2^(m-(bits-1)) covers everything
    // except the top sliver [2^m - 2^(m-29), 2^m), which needs e0 + 1. The two
    // checks below are exact in binary64 (24-bit FP32 mantissa and a 29-bit
    // qmax are both exact, and scaling by a power of two is exact).
    const std::int32_t e0 =
        significand_exponent - static_cast<std::int32_t>(effective_bits - 1);
    const double bound = std::ldexp(static_cast<double>(qmax), e0);
    if (static_cast<double>(std::fabs(maxabs)) <= bound) {
        return e0;
    }
    return e0 + 1;
}

std::int64_t round_half_to_even_scaled(float x, std::int32_t exponent) {
    const double y = std::ldexp(static_cast<double>(x), -exponent);
    return round_dyadic_half_to_even(y);
}

StatusCode quantize_element(
    float x,
    std::int32_t exponent,
    const PrecisionConfig& config,
    ScaleSaturationPolicy policy,
    std::int32_t* q,
    std::uint64_t* saturation_count) {
    if (!finite_f32(x)) {
        return StatusCode::NonFiniteValue;  // finiteness before any cast
    }
    const double y = std::ldexp(static_cast<double>(x), -exponent);
    const double qmax = static_cast<double>(config.saturation_bound());
    // |round(y)| > qmax  <=>  |y| >= qmax + 0.5 (a tie at exactly qmax + 0.5
    // rounds away from zero because qmax is odd and qmax+1 is even).
    if (std::fabs(y) >= qmax + 0.5) {
        const bool positive = y > 0.0;
        if (saturation_count != nullptr) {
            ++(*saturation_count);
        }
        if (policy == ScaleSaturationPolicy::Fail) {
            *q = static_cast<std::int32_t>(
                positive ? config.saturation_bound() : -config.saturation_bound());
            return StatusCode::QuantizationSaturation;
        }
        *q = static_cast<std::int32_t>(
            positive ? config.saturation_bound() : -config.saturation_bound());
        return StatusCode::Ok;
    }
    *q = static_cast<std::int32_t>(round_dyadic_half_to_even(y));
    return StatusCode::Ok;
}

float fp32_from_int64_scaled(std::int64_t value, std::int32_t exponent) {
    if (value == 0) {
        return 0.0f;
    }
    const bool negative = value < 0;
    // Safe magnitude including INT64_MIN.
    const std::uint64_t magnitude =
        negative ? (~static_cast<std::uint64_t>(value) + 1ull)
                 : static_cast<std::uint64_t>(value);
    const int bit_length =
        64 - __builtin_clzll(magnitude);  // magnitude != 0
    const int leading_exponent = exponent + bit_length - 1;

    auto make_bits = [&](std::uint32_t bits) {
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    };
    auto infinity = [&]() {
        return negative ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
    };

    if (leading_exponent > 127) {
        return infinity();
    }

    if (leading_exponent >= -126) {
        // Normal range: round the 24-bit significand.
        const int shift = bit_length - 24;
        std::uint64_t significand = 0;
        int result_exponent = leading_exponent;
        if (shift > 0) {
            significand = magnitude >> shift;
            const std::uint64_t remainder =
                magnitude & ((static_cast<std::uint64_t>(1) << shift) - 1);
            const std::uint64_t half = static_cast<std::uint64_t>(1) << (shift - 1);
            const bool round_up =
                remainder > half || (remainder == half && (significand & 1ull) != 0);
            if (round_up) {
                ++significand;
                if (significand == (static_cast<std::uint64_t>(1) << 24)) {
                    significand >>= 1;
                    ++result_exponent;
                }
            }
        } else {
            significand = magnitude << (-shift);
        }
        if (result_exponent > 127) {
            return infinity();
        }
        const std::uint32_t bits =
            (negative ? 0x80000000u : 0u) |
            (static_cast<std::uint32_t>(result_exponent + 127) << 23) |
            static_cast<std::uint32_t>(significand & 0x7FFFFFu);
        return make_bits(bits);
    }

    // Subnormal range: represent as k * 2^-149 with round-to-nearest-even.
    const int shift_left = exponent + 149;
    std::uint64_t subnormal = 0;
    if (shift_left >= 0) {
        subnormal = magnitude << shift_left;
    } else {
        const int shift_right = -shift_left;
        std::uint64_t quotient = 0;
        bool round_up = false;
        if (shift_right < 64) {
            quotient = magnitude >> shift_right;
            const std::uint64_t remainder =
                magnitude & ((static_cast<std::uint64_t>(1) << shift_right) - 1);
            const std::uint64_t half =
                static_cast<std::uint64_t>(1) << (shift_right - 1);
            round_up =
                remainder > half || (remainder == half && (quotient & 1ull) != 0);
        } else if (shift_right == 64) {
            const std::uint64_t half = 0x8000000000000000ull;
            round_up = magnitude > half;  // exact half rounds to even (0)
        }
        subnormal = quotient + (round_up ? 1ull : 0ull);
    }
    if (subnormal >= (static_cast<std::uint64_t>(1) << 23)) {
        // Rounded up to the minimum normal value 2^-126.
        return make_bits(negative ? 0x80800000u : 0x00800000u);
    }
    if (subnormal == 0) {
        return negative ? -0.0f : 0.0f;
    }
    return make_bits((negative ? 0x80000000u : 0u) |
                     static_cast<std::uint32_t>(subnormal));
}

bool checked_accumulate(
    std::int64_t qa,
    std::int64_t qb,
    std::uint32_t guard_bits,
    std::int64_t* accumulator,
    std::uint64_t* overflow_count) {
    if (guard_bits < 1 || guard_bits > 64) {
        return false;
    }
    const __int128 product = static_cast<__int128>(qa) * static_cast<__int128>(qb);
    const __int128 next = static_cast<__int128>(*accumulator) + product;
    const __int128 limit = static_cast<__int128>(1) << (guard_bits - 1);
    if (next >= limit || next < -limit) {
        if (overflow_count != nullptr) {
            ++(*overflow_count);
        }
        return false;
    }
    *accumulator = static_cast<std::int64_t>(next);
    return true;
}

StatusCode int32_dot_checked(
    const std::int32_t* a,
    const std::int32_t* b,
    std::uint32_t k,
    std::uint32_t guard_bits,
    std::int64_t* result) {
    if (result == nullptr || k == 0) {
        return StatusCode::BadDescriptor;
    }
    std::int64_t accumulator = 0;
    std::uint64_t overflows = 0;
    for (std::uint32_t index = 0; index < k; ++index) {
        if (!checked_accumulate(
                a[index], b[index], guard_bits, &accumulator, &overflows)) {
            return StatusCode::NumericOverflow;
        }
    }
    *result = accumulator;
    return StatusCode::Ok;
}

std::uint64_t int32_gemm_exec_cycles(
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const PrecisionConfig& config,
    const PeTimingConfig& timing) {
    const std::uint64_t input_elements =
        static_cast<std::uint64_t>(m) * k + static_cast<std::uint64_t>(k) * n;
    const std::uint64_t output_elements = static_cast<std::uint64_t>(m) * n;
    return ceil_div(input_elements, config.maxabs_elements_per_cycle) +
           ceil_div(input_elements, config.quantize_elements_per_cycle) +
           timing.fill_cycles +
           static_cast<std::uint64_t>(k) * timing.gemm_k_ii +
           static_cast<std::uint64_t>(m) * timing.drain_row_cycles +
           ceil_div(output_elements, config.dequantize_elements_per_cycle);
}

std::uint64_t int32_gemm_failure_cycles(
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const PrecisionConfig& config,
    const PeTimingConfig& timing,
    GemmPhase failed_phase,
    std::uint32_t failed_element_index) {
    const std::uint64_t input_elements =
        static_cast<std::uint64_t>(m) * k + static_cast<std::uint64_t>(k) * n;
    const std::uint64_t output_elements = static_cast<std::uint64_t>(m) * n;
    const std::uint64_t index = failed_element_index;
    switch (failed_phase) {
        case GemmPhase::ScaleReduction:
            return ceil_div(
                std::min(static_cast<std::uint64_t>(index) + 1, input_elements),
                config.maxabs_elements_per_cycle);
        case GemmPhase::Quantize:
            return ceil_div(input_elements, config.maxabs_elements_per_cycle) +
                   ceil_div(
                       std::min(static_cast<std::uint64_t>(index) + 1, input_elements),
                       config.quantize_elements_per_cycle);
        case GemmPhase::IntegerCompute: {
            const std::uint64_t row =
                output_elements == 0
                    ? 0
                    : std::min<std::uint64_t>(index / std::max(1u, n),
                                              std::max(1u, m) - 1);
            return ceil_div(input_elements, config.maxabs_elements_per_cycle) +
                   ceil_div(input_elements, config.quantize_elements_per_cycle) +
                   timing.fill_cycles +
                   static_cast<std::uint64_t>(k) * timing.gemm_k_ii +
                   (row + 1) * timing.drain_row_cycles;
        }
        case GemmPhase::DequantizeUpdate:
            return ceil_div(input_elements, config.maxabs_elements_per_cycle) +
                   ceil_div(input_elements, config.quantize_elements_per_cycle) +
                   timing.fill_cycles +
                   static_cast<std::uint64_t>(k) * timing.gemm_k_ii +
                   static_cast<std::uint64_t>(m) * timing.drain_row_cycles +
                   ceil_div(std::min(static_cast<std::uint64_t>(index) + 1,
                                     output_elements),
                            config.dequantize_elements_per_cycle);
        default:
            return 0;
    }
}

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
    std::string* reason) {
    if (stats != nullptr) {
        *stats = Int32GemmStats{};
    }
    if (m == 0 || n == 0 || k == 0 || k > config.k_block_max) {
        if (reason != nullptr) {
            *reason = "INT32 GEMM requires 1 <= k <= k_block_max";
        }
        return StatusCode::BadDescriptor;
    }
    if ((!scale.auto_exponent_a &&
         (scale.exponent_a < kScaleExponentMin ||
          scale.exponent_a > kScaleExponentMax)) ||
        (!scale.auto_exponent_b &&
         (scale.exponent_b < kScaleExponentMin ||
          scale.exponent_b > kScaleExponentMax))) {
        if (reason != nullptr) {
            *reason = "fixed INT32 scale exponent outside [-256, 256]";
        }
        return StatusCode::PolicyNotSupported;
    }
    if (!scale.auto_output_scale) {
        if (reason != nullptr) {
            *reason = "fixed output scale is not supported by this version";
        }
        return StatusCode::PolicyNotSupported;
    }

    const std::uint64_t input_elements =
        static_cast<std::uint64_t>(m) * k + static_cast<std::uint64_t>(k) * n;
    const std::uint64_t output_elements = static_cast<std::uint64_t>(m) * n;

    auto fail = [&](StatusCode status, GemmPhase phase, std::uint32_t row,
                    std::uint32_t column, const char* text) {
        if (stats != nullptr) {
            stats->failed_phase = phase;
            stats->failed_local_row = row;
            stats->failed_local_col = column;
            // Only the cycles actually consumed before the failure are kept;
            // the caller fills the partial counters of the failing phase and
            // later phases stay zero.
        }
        if (reason != nullptr) {
            *reason = text;
        }
        return status;
    };

    // Phase 1: maxabs + finiteness over the valid region, A then B.
    double max_abs_a = 0.0;
    std::uint64_t scanned = 0;
    for (std::uint32_t row = 0; row < m; ++row) {
        for (std::uint32_t column = 0; column < k; ++column) {
            const float value = lhs[static_cast<std::size_t>(row) * lhs_stride + column];
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;  // maxabs FP32 read
            }
            if (!finite_f32(value)) {
                if (stats != nullptr) {
                    stats->scale_reduction_cycles = ceil_div(
                        std::min(scanned + 1, input_elements),
                        config.maxabs_elements_per_cycle);
                }
                return fail(
                    StatusCode::NonFiniteValue, GemmPhase::ScaleReduction, row,
                    column, "non-finite GEMM operand A in the INT32 scale reduction");
            }
            max_abs_a = std::max(max_abs_a, std::fabs(static_cast<double>(value)));
            ++scanned;
        }
    }
    double max_abs_b = 0.0;
    for (std::uint32_t row = 0; row < k; ++row) {
        for (std::uint32_t column = 0; column < n; ++column) {
            const float value = rhs[static_cast<std::size_t>(row) * rhs_stride + column];
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;
            }
            if (!finite_f32(value)) {
                if (stats != nullptr) {
                    stats->scale_reduction_cycles = ceil_div(
                        std::min(scanned + 1, input_elements),
                        config.maxabs_elements_per_cycle);
                }
                return fail(
                    StatusCode::NonFiniteValue, GemmPhase::ScaleReduction, row,
                    column, "non-finite GEMM operand B in the INT32 scale reduction");
            }
            max_abs_b = std::max(max_abs_b, std::fabs(static_cast<double>(value)));
            ++scanned;
        }
    }
    const std::uint64_t scale_reduction_cycles =
        ceil_div(input_elements, config.maxabs_elements_per_cycle);

    // Phase 2: per-operand power-of-two scale. AUTO selects the smallest
    // exponent that represents the local maxabs exactly within the symmetric
    // range; a fixed exponent comes from the validated SCALE_DESC.
    const std::int32_t exponent_a =
        scale.auto_exponent_a
            ? choose_power_of_two_exponent(static_cast<float>(max_abs_a),
                                           config.effective_bits)
            : scale.exponent_a;
    const std::int32_t exponent_b =
        scale.auto_exponent_b
            ? choose_power_of_two_exponent(static_cast<float>(max_abs_b),
                                           config.effective_bits)
            : scale.exponent_b;
    if (stats != nullptr) {
        stats->exponent_a = exponent_a;
        stats->exponent_b = exponent_b;
    }

    // Phase 3: quantization (A then B) into the reused staging buffers.
    std::vector<std::int32_t> qa(static_cast<std::size_t>(m) * k, 0);
    std::vector<std::int32_t> qb(static_cast<std::size_t>(k) * n, 0);
    std::uint64_t saturation_count = 0;
    std::uint64_t quantized = 0;
    for (std::uint32_t row = 0; row < m; ++row) {
        for (std::uint32_t column = 0; column < k; ++column) {
            const float value = lhs[static_cast<std::size_t>(row) * lhs_stride + column];
            std::int32_t q = 0;
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;  // quantize FP32 read
            }
            const StatusCode status = quantize_element(
                value, exponent_a, config, scale.saturation, &q, &saturation_count);
            if (status != StatusCode::Ok) {
                if (stats != nullptr) {
                    stats->scale_reduction_cycles = scale_reduction_cycles;
                    stats->quantize_cycles = ceil_div(
                        std::min(quantized + 1, input_elements),
                        config.quantize_elements_per_cycle);
                    stats->saturation_count = saturation_count;
                }
                return fail(
                    status, GemmPhase::Quantize, row, column,
                    status == StatusCode::QuantizationSaturation
                        ? "INT32 quantization saturated operand A (FAIL policy); "
                          "tile not committed"
                        : "non-finite value during INT32 quantization");
            }
            qa[static_cast<std::size_t>(row) * k + column] = q;
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;  // INT32 staging write
            }
            ++quantized;
        }
    }
    for (std::uint32_t row = 0; row < k; ++row) {
        for (std::uint32_t column = 0; column < n; ++column) {
            const float value = rhs[static_cast<std::size_t>(row) * rhs_stride + column];
            std::int32_t q = 0;
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;
            }
            const StatusCode status = quantize_element(
                value, exponent_b, config, scale.saturation, &q, &saturation_count);
            if (status != StatusCode::Ok) {
                if (stats != nullptr) {
                    stats->scale_reduction_cycles = scale_reduction_cycles;
                    stats->quantize_cycles = ceil_div(
                        std::min(quantized + 1, input_elements),
                        config.quantize_elements_per_cycle);
                    stats->saturation_count = saturation_count;
                }
                return fail(
                    status, GemmPhase::Quantize, row, column,
                    status == StatusCode::QuantizationSaturation
                        ? "INT32 quantization saturated operand B (FAIL policy); "
                          "tile not committed"
                        : "non-finite value during INT32 quantization");
            }
            qb[static_cast<std::size_t>(row) * n + column] = q;
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;
            }
            ++quantized;
        }
    }
    if (stats != nullptr) {
        stats->scale_reduction_cycles = scale_reduction_cycles;
        stats->quantize_cycles =
            ceil_div(input_elements, config.quantize_elements_per_cycle);
        stats->saturation_count = saturation_count;
    }

    // Phase 4: INT32 x INT32 with explicit widened, range-checked INT64
    // accumulation.
    const std::uint64_t integer_cycles =
        timing.fill_cycles + static_cast<std::uint64_t>(k) * timing.gemm_k_ii +
        static_cast<std::uint64_t>(m) * timing.drain_row_cycles;
    if (stats != nullptr) {
        stats->integer_compute_cycles = integer_cycles;
    }
    const std::int32_t exponent_sum = exponent_a + exponent_b;
    std::uint64_t overflow_count = 0;
    // One bounded INT64 output tile. Complete the integer phase before any
    // dequantization, matching the sequential phase timing/resource contract.
    std::vector<std::int64_t> accumulators(output_elements, 0);
    for (std::uint32_t row = 0; row < m; ++row) {
        for (std::uint32_t column = 0; column < n; ++column) {
            std::int64_t accumulator = 0;
            for (std::uint32_t t = 0; t < k; ++t) {
                if (!checked_accumulate(
                        qa[static_cast<std::size_t>(row) * k + t],
                        qb[static_cast<std::size_t>(t) * n + column],
                        config.accumulator_bits,
                        &accumulator,
                        &overflow_count)) {
                    if (stats != nullptr) {
                        stats->overflow_count = overflow_count;
                        stats->integer_compute_cycles =
                            timing.fill_cycles +
                            static_cast<std::uint64_t>(t + 1) * timing.gemm_k_ii +
                            static_cast<std::uint64_t>(row) *
                                timing.drain_row_cycles;
                    }
                    return fail(
                        StatusCode::NumericOverflow, GemmPhase::IntegerCompute, row,
                        column, "INT64 accumulation exceeded the configured range");
                }
            }
            accumulators[static_cast<std::size_t>(row) * n + column] = accumulator;
        }
    }

    for (std::uint32_t row = 0; row < m; ++row) {
        for (std::uint32_t column = 0; column < n; ++column) {
            if (stats != nullptr) {
                stats->local_conversion_bytes += 8;  // INT64 accumulator read
            }
            const float partial = fp32_from_int64_scaled(
                accumulators[static_cast<std::size_t>(row) * n + column], exponent_sum);
            if (!finite_f32(partial)) {
                if (stats != nullptr) {
                    stats->overflow_count = overflow_count;
                    stats->dequantize_cycles = ceil_div(
                        std::min(static_cast<std::uint64_t>(row) * n + column + 1,
                                 output_elements),
                        config.dequantize_elements_per_cycle);
                }
                return fail(
                    StatusCode::NonFiniteValue, GemmPhase::DequantizeUpdate, row,
                    column, "dequantized INT32 partial product is non-finite");
            }
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;  // FP32 C read
            }
            float updated =
                dst[static_cast<std::size_t>(row) * dst_stride + column] - partial;
            if (!finite_f32(updated)) {
                if (stats != nullptr) {
                    stats->overflow_count = overflow_count;
                    stats->dequantize_cycles = ceil_div(
                        std::min(static_cast<std::uint64_t>(row) * n + column + 1,
                                 output_elements),
                        config.dequantize_elements_per_cycle);
                }
                return fail(
                    StatusCode::NonFiniteValue, GemmPhase::DequantizeUpdate, row,
                    column, "FP32 Schur update produced a non-finite value");
            }
            dst[static_cast<std::size_t>(row) * dst_stride + column] = updated;
            if (stats != nullptr) {
                stats->local_conversion_bytes += 4;  // FP32 C write
            }
        }
    }
    if (stats != nullptr) {
        stats->overflow_count = overflow_count;
        stats->dequantize_cycles =
            ceil_div(output_elements, config.dequantize_elements_per_cycle);
        stats->failed_phase = GemmPhase::None;
    }
    return StatusCode::Ok;
}

}  // namespace hw::panel_lu
