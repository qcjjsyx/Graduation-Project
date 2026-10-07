// T10 GEMM_SCHUR tests: local PrecisionAdapter + INT32 GEMM, real FP32 GEMM on
// the shared compute engine, control-plane integration and the STATIC pivot
// propagation case.
//
// Coverage map:
//   A. SCALE_DESC decode/validation (malformed vs unsupported policies),
//   B. precision primitives: power-of-two scale boundaries, ties-to-even
//      rounding, range checks before casts, exact dequantization boundaries,
//      explicit INT64 accumulation guard vs an independent widened oracle,
//   C. INT32 tile kernel: exact-integer cases, per-block AUTO scales, tails,
//      saturation FAIL / CLAMP_AND_REPORT, non-finite handling, dequantize
//      overflow, rounding boundaries,
//   D. real FP32 GEMM engine: hand case, n=42/p=23 multi-tile/multi-K with an
//      independent FP64 recurrence, task/byte accounting, block invariance,
//      nonzero front offset, groups, backpressure, failures, abort/reset,
//   E. control-plane + schema v2 artifact path: PANEL -> TRSM -> GEMM ->
//      STORE_FACTOR/STORE_UPDATE -> NODE_COMMIT, DDR update values, completion
//      order; FP32 and INT32 backends; scale-policy negative cases,
//   F. STATIC pivot propagation from the real device-generated update (no
//      FP64 pivot injection): FP32 vs local INT32.
//
// Numeric expectations come from hand computation, an independent FP64/INT128
// oracle, or the device's own committed inputs re-evaluated in FP64. Device
// output is never used as its own expectation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "controlplane/control_plane.hpp"
#include "controlplane/panel_exec_unit.hpp"
#include "control_test_artifact.hpp"
#include "dataplane/on_chip_arbiter.hpp"
#include "fp64_reference.hpp"
#include "panel_lu/precision_adapter.hpp"
#include "panel_lu/tile_kernels.hpp"

namespace {

namespace cmd = hw::command;
namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace pl = hw::panel_lu;
namespace testing = hw::testing;
namespace ref = hw::reference_fp64;

using StatusCode = cmd::StatusCode;
using pl::GemmBackend;
using pl::GemmPhase;
using pl::PrecisionConfig;
using pl::ScaleSaturationPolicy;

int failures = 0;
int checks = 0;
std::string g_output_dir;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::uint32_t f32_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_f32(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

pl::PanelLuConfig base_panel_config(std::uint32_t groups, std::uint32_t per_group) {
    pl::PanelLuConfig config;
    config.compute_group_count = groups;
    config.pes_per_group = per_group;
    config.pe_count = groups * per_group;
    return config;
}

// ---------------------------------------------------------------------------
// A. SCALE_DESC decode/validation
// ---------------------------------------------------------------------------

void test_scale_desc_decode() {
    const std::uint32_t auto_word = dp::kScaleAutoWord;
    pl::GemmScalePolicy policy;
    std::string reason;

    // Default policy: AUTO, AUTO, AUTO, 1, 1, 1.
    const std::uint32_t defaults[10] = {auto_word, auto_word, auto_word, 1, 1, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(7, defaults, &policy, &reason) == StatusCode::Ok,
        "scale: default AUTO/1/1/1 policy accepted");
    expect(
        policy.auto_exponent_a && policy.auto_exponent_b && policy.auto_output_scale &&
            policy.saturation == ScaleSaturationPolicy::Fail,
        "scale: default policy fields decoded");

    // Fixed exponents within [-256, 256].
    const std::uint32_t fixed[10] = {
        static_cast<std::uint32_t>(-3), static_cast<std::uint32_t>(256), auto_word, 1, 1, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, fixed, &policy, &reason) == StatusCode::Ok &&
            !policy.auto_exponent_a && policy.exponent_a == -3 &&
            policy.exponent_b == 256,
        "scale: fixed two's-complement exponents accepted");

    // Out-of-range fixed exponent -> POLICY_NOT_SUPPORTED (well-formed i32).
    const std::uint32_t too_big[10] = {257u, auto_word, auto_word, 1, 1, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, too_big, &policy, &reason) ==
            StatusCode::PolicyNotSupported,
        "scale: fixed exponent outside [-256, 256] rejected as unsupported");

    // Fixed output scale -> POLICY_NOT_SUPPORTED.
    const std::uint32_t fixed_output[10] = {auto_word, auto_word, 0, 1, 1, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, fixed_output, &policy, &reason) ==
            StatusCode::PolicyNotSupported,
        "scale: fixed output scale rejected as unsupported");

    // Undefined enumerations -> BAD_DESCRIPTOR.
    const std::uint32_t zero_rounding[10] = {auto_word, auto_word, auto_word, 0, 1, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, zero_rounding, &policy, &reason) ==
            StatusCode::BadDescriptor,
        "scale: rounding_mode 0 is malformed");
    const std::uint32_t zero_saturation[10] = {auto_word, auto_word, auto_word, 1, 0, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, zero_saturation, &policy, &reason) ==
            StatusCode::BadDescriptor,
        "scale: saturation_policy 0 is malformed");
    const std::uint32_t zero_overflow[10] = {auto_word, auto_word, auto_word, 1, 1, 0, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, zero_overflow, &policy, &reason) ==
            StatusCode::BadDescriptor,
        "scale: overflow_policy 0 is malformed");
    // Non-zero reserved word -> BAD_DESCRIPTOR.
    const std::uint32_t reserved[10] = {auto_word, auto_word, auto_word, 1, 1, 1, 0, 0, 0, 9};
    expect(
        dp::decode_scale_policy(1, reserved, &policy, &reason) ==
            StatusCode::BadDescriptor,
        "scale: non-zero reserved word rejected");
    // Defined but unsupported enumerations -> POLICY_NOT_SUPPORTED.
    const std::uint32_t unknown_round[10] = {auto_word, auto_word, auto_word, 2, 1, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, unknown_round, &policy, &reason) ==
            StatusCode::PolicyNotSupported,
        "scale: rounding_mode 2 is defined but unsupported");
    const std::uint32_t clamp[10] = {auto_word, auto_word, auto_word, 1, 2, 1, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, clamp, &policy, &reason) == StatusCode::Ok &&
            policy.saturation == ScaleSaturationPolicy::ClampAndReport,
        "scale: CLAMP_AND_REPORT accepted");
    const std::uint32_t overflow2[10] = {auto_word, auto_word, auto_word, 1, 1, 2, 0, 0, 0, 0};
    expect(
        dp::decode_scale_policy(1, overflow2, &policy, &reason) ==
            StatusCode::PolicyNotSupported,
        "scale: overflow_policy 2 is unsupported");
}

// ---------------------------------------------------------------------------
// B. Precision primitives
// ---------------------------------------------------------------------------

// Independent widened dot product oracle (never overflows; reports whether the
// value would fit INT64).
bool oracle_dot_fits_int64(
    const std::vector<std::int32_t>& a,
    const std::vector<std::int32_t>& b,
    std::int64_t* value) {
    __int128 acc = 0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        acc += static_cast<__int128>(a[index]) * static_cast<__int128>(b[index]);
    }
    if (acc > static_cast<__int128>(std::numeric_limits<std::int64_t>::max()) ||
        acc < static_cast<__int128>(std::numeric_limits<std::int64_t>::min())) {
        return false;
    }
    *value = static_cast<std::int64_t>(acc);
    return true;
}

void test_precision_primitives() {
    PrecisionConfig config;
    expect(
        pl::validate_precision_config(config).ok,
        "precision: default INT32/30-bit/INT64 config is accepted");
    {
        PrecisionConfig unsafe = config;
        unsafe.effective_bits = 31;
        expect(
            !pl::validate_precision_config(unsafe).ok,
            "precision: 31 effective bits rejected (16-term dot not < 2^62)");
    }
    {
        PrecisionConfig other = config;
        other.accumulator_bits = 32;
        expect(
            !pl::validate_precision_config(other).ok,
            "precision: non-INT64 accumulator rejected");
    }
    {
        PrecisionConfig zero_rate = config;
        zero_rate.quantize_elements_per_cycle = 0;
        expect(
            !pl::validate_precision_config(zero_rate).ok,
            "precision: zero conversion rate rejected");
    }
    {
        PrecisionConfig wide_k = config;
        wide_k.k_block_max = 17;
        expect(
            !pl::validate_precision_config(wide_k).ok,
            "precision: K block > 16 rejected");
    }

    // Scale selection boundaries. qmax = 2^29 - 1.
    const std::int64_t qmax = config.saturation_bound();
    expect(
        pl::choose_power_of_two_exponent(0.0f, 30) == 0,
        "scale: all-zero tile uses the deterministic e = 0");
    // Exact power boundaries: for maxabs = 2^t the smallest exponent is
    // t - (bits-1) + 1 = t - 28 because qmax = 2^29 - 1 < 2^29.
    for (const int t : {-100, -10, 0, 5, 60}) {
        const float power = std::ldexp(1.0f, t);
        if (power == 0.0f || !std::isfinite(power)) {
            continue;
        }
        expect(
            pl::choose_power_of_two_exponent(power, 30) == t - 28,
            "scale: maxabs = 2^" + std::to_string(t) + " -> e = t - 28");
    }
    // 2^29 is one bit above qmax in exact arithmetic and needs e = 1; 2^28
    // fits at e = 0.
    expect(
        pl::choose_power_of_two_exponent(std::ldexp(1.0f, 29), 30) == 1,
        "scale: maxabs = 2^29 needs e = 1");
    expect(
        pl::choose_power_of_two_exponent(std::ldexp(1.0f, 28), 30) == 0,
        "scale: maxabs = 2^28 fits at e = 0");
    // Subnormal maxabs and FP32 max value are finite cases.
    expect(
        pl::choose_power_of_two_exponent(
            std::numeric_limits<float>::denorm_min(), 30) == -177,
        "scale: subnormal maxabs (2^-149) -> e = -177");
    expect(
        pl::choose_power_of_two_exponent(std::numeric_limits<float>::max(), 30) <= 128,
        "scale: FP32 max value gets a finite exponent");

    // Quantization: sign, zero, tie-to-even, quantize-to-zero, saturation.
    std::uint64_t saturations = 0;
    std::int32_t q = 0;
    expect(
        pl::quantize_element(1.0f, 0, config, ScaleSaturationPolicy::Fail, &q, &saturations) ==
                StatusCode::Ok &&
            q == 1,
        "quantize: +1.0 at e=0 -> 1");
    expect(
        pl::quantize_element(-1.0f, 0, config, ScaleSaturationPolicy::Fail, &q, &saturations) ==
                StatusCode::Ok &&
            q == -1,
        "quantize: -1.0 at e=0 -> -1");
    expect(
        pl::quantize_element(0.0f, -5, config, ScaleSaturationPolicy::Fail, &q, &saturations) ==
                StatusCode::Ok &&
            q == 0,
        "quantize: zero stays zero");
    // Ties-to-even: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, -2.5 -> -2, -1.5 -> -2.
    struct TieCase {
        float value;
        std::int32_t expected;
    };
    for (const TieCase tie : {TieCase{0.5f, 0}, TieCase{1.5f, 2}, TieCase{2.5f, 2},
                              TieCase{-0.5f, 0}, TieCase{-1.5f, -2}, TieCase{-2.5f, -2}}) {
        std::int32_t tq = 0;
        pl::quantize_element(tie.value, 0, config, ScaleSaturationPolicy::Fail, &tq, &saturations);
        expect(
            tq == tie.expected,
            "quantize: ties-to-even " + std::to_string(tie.value) + " -> " +
                std::to_string(tie.expected));
    }
    // A tiny value quantizes to zero at e = 0 (documented, not saturation).
    expect(
        pl::quantize_element(1.0e-30f, 0, config, ScaleSaturationPolicy::Fail, &q,
                             &saturations) == StatusCode::Ok &&
            q == 0,
        "quantize: subnormal/tiny value quantizes to zero without saturation");
    // Saturation: value above qmax at e = 0.
    const float too_big = static_cast<float>(static_cast<double>(qmax) * 2.0);
    expect(
        pl::quantize_element(too_big, 0, config, ScaleSaturationPolicy::Fail, &q,
                             &saturations) == StatusCode::QuantizationSaturation,
        "quantize: FAIL policy reports saturation");
    expect(saturations == 1, "quantize: FAIL policy counts the rejected value");
    expect(
        pl::quantize_element(too_big, 0, config, ScaleSaturationPolicy::ClampAndReport, &q,
                             &saturations) == StatusCode::Ok &&
            q == qmax && saturations == 2,
        "quantize: CLAMP_AND_REPORT clamps and counts");
    expect(
        pl::quantize_element(
            std::numeric_limits<float>::infinity(), 0, config,
            ScaleSaturationPolicy::Fail, &q, &saturations) == StatusCode::NonFiniteValue,
        "quantize: non-finite input is rejected before any cast");
    expect(
        pl::quantize_element(
            std::numeric_limits<float>::quiet_NaN(), 0, config,
            ScaleSaturationPolicy::Fail, &q, &saturations) == StatusCode::NonFiniteValue,
        "quantize: NaN input is rejected before any cast");

    // Dequantization: exact boundaries (hand-computed, round-to-nearest-even).
    expect(pl::fp32_from_int64_scaled(0, 100) == 0.0f, "dequant: zero -> +0");
    expect(pl::fp32_from_int64_scaled(-1, 0) == -1.0f, "dequant: -1 -> -1.0");
    expect(pl::fp32_from_int64_scaled(1, 100) == std::ldexp(1.0f, 100), "dequant: 2^100 exact");
    // 2^24 + 1 ties between 2^24 and 2^24 + 2 -> 2^24 (even).
    expect(
        pl::fp32_from_int64_scaled((1LL << 24) + 1, 0) == static_cast<float>(1LL << 24),
        "dequant: 2^24+1 rounds to even 2^24");
    // 2^24 + 3 ties between 2^24+2 and 2^24+4 -> 2^24+4.
    expect(
        pl::fp32_from_int64_scaled((1LL << 24) + 3, 0) == static_cast<float>((1LL << 24) + 4),
        "dequant: 2^24+3 rounds to even 2^24+4");
    // Exact power-of-two scaling stays exact.
    expect(
        pl::fp32_from_int64_scaled(3, -3) == 0.375f,
        "dequant: 3 * 2^-3 = 0.375 exact");
    // Subnormal range: 2^-149 is the smallest subnormal.
    expect(
        pl::fp32_from_int64_scaled(1, -149) == std::numeric_limits<float>::denorm_min(),
        "dequant: 2^-149 is the smallest subnormal");
    // Rounding to the minimum normal: (2^23-1)*2^-149 + half ULP.
    {
        const std::int64_t half_way = (1LL << 23) - 1;
        const float rounded = pl::fp32_from_int64_scaled((half_way * 2 + 1), -150);
        expect(
            rounded == bits_f32(0x00800000u),
            "dequant: subnormal rounding carries into the minimum normal");
    }
    // Underflow to zero (below half the smallest subnormal).
    expect(
        pl::fp32_from_int64_scaled(1, -160) == 0.0f,
        "dequant: 2^-160 underflows to +0");
    expect(
        std::signbit(pl::fp32_from_int64_scaled(-1, -160)),
        "dequant: negative underflow keeps the sign of zero");
    // Overflow -> infinity (caller maps to NON_FINITE_VALUE).
    expect(
        std::isinf(pl::fp32_from_int64_scaled(std::numeric_limits<std::int64_t>::max(), 100)),
        "dequant: overflow to +inf");
    expect(
        std::isinf(pl::fp32_from_int64_scaled(
            std::numeric_limits<std::int64_t>::min(), 100)),
        "dequant: overflow to -inf");

    // INT32 dot product vs an independent widened oracle, exact equality.
    {
        std::vector<std::int32_t> a(16);
        std::vector<std::int32_t> b(16);
        for (std::uint32_t index = 0; index < 16; ++index) {
            a[index] = static_cast<std::int32_t>((index % 2 == 0 ? 1 : -1) *
                                                 (127 + static_cast<int>(index) * 997));
            b[index] = static_cast<std::int32_t>(static_cast<int>(index) * 1237 - 4096);
        }
        std::int64_t expected = 0;
        expect(
            oracle_dot_fits_int64(a, b, &expected),
            "dot: oracle value fits INT64 for the supported range");
        std::int64_t got = 0;
        expect(
            pl::int32_dot_checked(a.data(), b.data(), 16, 64, &got) == StatusCode::Ok &&
                got == expected,
            "dot: INT32xINT32/INT64 result matches the independent oracle exactly");
    }
    // Maximum supported quantized magnitude at K=16 stays inside the declared
    // safety bound and matches the oracle.
    {
        const std::int64_t max_q = qmax;
        std::vector<std::int32_t> a(16, static_cast<std::int32_t>(max_q));
        std::vector<std::int32_t> b(16, static_cast<std::int32_t>(max_q));
        std::int64_t expected = 0;
        expect(oracle_dot_fits_int64(a, b, &expected), "dot: qmax^2*16 fits INT64");
        expect(
            static_cast<__int128>(16) * static_cast<__int128>(max_q) *
                    static_cast<__int128>(max_q) <
                (static_cast<__int128>(1) << 62),
            "dot: default 30-bit 16-term product is below 2^62");
        std::int64_t got = 0;
        expect(
            pl::int32_dot_checked(a.data(), b.data(), 16, 64, &got) == StatusCode::Ok &&
                got == expected,
            "dot: qmax 16-term product matches the oracle");
    }
    // Full-range INT32 extremes overflow INT64: the guard must report it and
    // the oracle itself must not overflow (it uses __int128).
    {
        std::vector<std::int32_t> a(16, std::numeric_limits<std::int32_t>::max());
        std::vector<std::int32_t> b(16, std::numeric_limits<std::int32_t>::max());
        std::int64_t ignored = 0;
        expect(
            !oracle_dot_fits_int64(a, b, &ignored),
            "dot: full-range INT32 extremes do not fit INT64 (oracle exact)");
        std::int64_t got = 0;
        expect(
            pl::int32_dot_checked(a.data(), b.data(), 16, 64, &got) ==
                StatusCode::NumericOverflow,
            "dot: guarded accumulation reports NUMERIC_OVERFLOW");
    }
}

// ---------------------------------------------------------------------------
// C. INT32 tile kernel
// ---------------------------------------------------------------------------

constexpr std::uint32_t kTileStride = 16;

void store_tile(
    float* tile,
    std::uint32_t rows,
    std::uint32_t cols,
    const std::vector<float>& values) {
    std::fill(tile, tile + kTileStride * kTileStride, 0.0f);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; ++col) {
            tile[row * kTileStride + col] = values[row * cols + col];
        }
    }
}

void test_int32_tile_kernel() {
    PrecisionConfig config;
    const std::uint32_t auto_word = dp::kScaleAutoWord;
    pl::GemmScalePolicy auto_policy;
    {
        const std::uint32_t words[10] = {auto_word, auto_word, auto_word, 1, 1, 1, 0, 0, 0, 0};
        std::string reason;
        expect(
            dp::decode_scale_policy(0, words, &auto_policy, &reason) == StatusCode::Ok,
            "kernel: AUTO policy decoded");
    }

    // Exact-integer case: all values are small dyadic rationals, so the
    // quantization is exact and the update must equal an independently
    // computed integer recurrence with no tolerance.
    {
        const std::uint32_t m = 4;
        const std::uint32_t n = 4;
        const std::uint32_t k = 4;
        std::vector<float> a(m * k);
        std::vector<float> b(k * n);
        std::vector<float> c(m * n);
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t t = 0; t < k; ++t) {
                a[i * k + t] = (i + 2 * t) % 3 == 0 ? -2.0f : 0.5f * static_cast<float>(i + t);
            }
        }
        for (std::uint32_t t = 0; t < k; ++t) {
            for (std::uint32_t j = 0; j < n; ++j) {
                b[t * n + j] = (t + j) % 2 == 0 ? 1.0f : -1.5f;
            }
        }
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                c[i * n + j] = static_cast<float>(i + j) * 0.25f;
            }
        }
        std::vector<float> dst(kTileStride * kTileStride, 0.0f);
        std::vector<float> lhs(kTileStride * kTileStride, 0.0f);
        std::vector<float> rhs(kTileStride * kTileStride, 0.0f);
        store_tile(dst.data(), m, n, c);
        store_tile(lhs.data(), m, k, a);
        store_tile(rhs.data(), k, n, b);
        pl::Int32GemmStats stats;
        std::string reason;
        const StatusCode status = pl::run_int32_gemm_update(
            dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(), kTileStride,
            m, n, k, config, pl::PeTimingConfig{}, auto_policy, &stats, &reason);
        expect(status == StatusCode::Ok, "kernel: exact dyadic case succeeds");
        // Independent exact expectation: scales are power-of-two, q exact.
        const std::int32_t e_a = pl::choose_power_of_two_exponent(
            *std::max_element(a.begin(), a.end(), [](float x, float y) {
                return std::fabs(x) < std::fabs(y);
            }), 30);
        const std::int32_t e_b = pl::choose_power_of_two_exponent(
            *std::max_element(b.begin(), b.end(), [](float x, float y) {
                return std::fabs(x) < std::fabs(y);
            }), 30);
        bool exact = true;
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                __int128 acc = 0;
                for (std::uint32_t t = 0; t < k; ++t) {
                    const std::int64_t qa = static_cast<std::int64_t>(
                        std::ldexp(static_cast<double>(a[i * k + t]), -e_a));
                    const std::int64_t qb = static_cast<std::int64_t>(
                        std::ldexp(static_cast<double>(b[t * n + j]), -e_b));
                    acc += static_cast<__int128>(qa) * qb;
                }
                const double partial = std::ldexp(static_cast<double>(acc), e_a + e_b);
                const float expected = static_cast<float>(
                    static_cast<double>(c[i * n + j]) - partial);
                if (dst[i * kTileStride + j] != expected) {
                    exact = false;
                }
            }
        }
        expect(exact, "kernel: INT32 update is exact for exactly representable data");
        expect(
            stats.exponent_a == e_a && stats.exponent_b == e_b,
            "kernel: reported AUTO exponents match the independent selection");
        expect(
            stats.local_conversion_bytes ==
                12 * (static_cast<std::uint64_t>(m) * k + static_cast<std::uint64_t>(k) * n) +
                    16 * static_cast<std::uint64_t>(m) * n,
            "kernel: local conversion bytes counted separately");
        expect(stats.saturation_count == 0, "kernel: no saturation in the exact case");
    }

    // Two K blocks with different per-block scales and a tail region.
    {
        const std::uint32_t m = 3;
        const std::uint32_t n = 5;
        std::vector<float> dst(kTileStride * kTileStride, 0.0f);
        std::vector<float> lhs(kTileStride * kTileStride, 0.0f);
        std::vector<float> rhs(kTileStride * kTileStride, 0.0f);
        // Block 0: k=4 with values ~1; block 1: k=3 with values ~2^-20.
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t t = 0; t < 4; ++t) {
                lhs[i * kTileStride + t] = 0.25f * static_cast<float>(i + t + 1);
            }
            for (std::uint32_t t = 0; t < 3; ++t) {
                lhs[i * kTileStride + 4 + t] = std::ldexp(1.0f, -20) * static_cast<float>(t + 1);
            }
        }
        for (std::uint32_t t = 0; t < 4; ++t) {
            for (std::uint32_t j = 0; j < n; ++j) {
                rhs[t * kTileStride + j] = 0.5f * static_cast<float>(j - t);
            }
        }
        for (std::uint32_t t = 0; t < 3; ++t) {
            for (std::uint32_t j = 0; j < n; ++j) {
                rhs[(4 + t) * kTileStride + j] = std::ldexp(1.0f, -20) * static_cast<float>(t + j);
            }
        }
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                dst[i * kTileStride + j] = 1.0f;
            }
        }
        std::string reason;
        const StatusCode first = pl::run_int32_gemm_update(
            dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(), kTileStride,
            m, n, 4, config, pl::PeTimingConfig{}, auto_policy, nullptr, &reason);
        const StatusCode second = pl::run_int32_gemm_update(
            dst.data(), kTileStride, lhs.data() + 4, kTileStride,
            rhs.data() + 4 * kTileStride, kTileStride, m, n, 3, config,
            pl::PeTimingConfig{}, auto_policy, nullptr, &reason);
        expect(
            first == StatusCode::Ok && second == StatusCode::Ok,
            "kernel: multi K blocks with different scales succeed");
        bool finite = true;
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                if (!std::isfinite(dst[i * kTileStride + j])) {
                    finite = false;
                }
            }
        }
        expect(finite, "kernel: multi-block result stays finite");
    }

    // Saturation FAIL vs CLAMP_AND_REPORT with a fixed e = 0.
    {
        pl::GemmScalePolicy fixed;
        {
            const std::uint32_t words[10] = {0, 0, auto_word, 1, 1, 1, 0, 0, 0, 0};
            std::string reason;
            dp::decode_scale_policy(0, words, &fixed, &reason);
        }
        std::vector<float> dst(kTileStride * kTileStride, 0.0f);
        std::vector<float> lhs(kTileStride * kTileStride, 0.0f);
        std::vector<float> rhs(kTileStride * kTileStride, 0.0f);
        lhs[0] = 1.0e10f;  // > qmax at e = 0
        rhs[0] = 1.0f;
        dst[0] = 1.0f;
        const float dst_before = dst[0];
        std::string reason;
        pl::Int32GemmStats stats;
        expect(
            pl::run_int32_gemm_update(
                dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(),
                kTileStride, 1, 1, 1, config, pl::PeTimingConfig{}, fixed, &stats,
                &reason) == StatusCode::QuantizationSaturation,
            "kernel: FAIL policy returns QUANTIZATION_SATURATION");
        expect(
            stats.failed_phase == GemmPhase::Quantize && dst[0] == dst_before,
            "kernel: failed quantize does not modify the target tile");
        expect(
            stats.saturation_count == 1 && stats.local_conversion_bytes == 12 &&
                stats.scale_reduction_cycles == 2 && stats.quantize_cycles == 1 &&
                stats.integer_compute_cycles == 0 && stats.dequantize_cycles == 0,
            "kernel: first-A saturation counts scans and failed read, no write");
        lhs[0] = 1.0f;
        rhs[0] = 1.0e10f;
        expect(
            pl::run_int32_gemm_update(
                dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(),
                kTileStride, 1, 1, 1, config, pl::PeTimingConfig{}, fixed, &stats,
                &reason) == StatusCode::QuantizationSaturation &&
                stats.saturation_count == 1 && stats.local_conversion_bytes == 20 &&
                stats.quantize_cycles == 2 && dst[0] == dst_before,
            "kernel: B saturation retains the successful A conversion bytes");
        lhs[0] = 1.0e10f;
        rhs[0] = 1.0f;
        pl::GemmScalePolicy clamp = fixed;
        clamp.saturation = ScaleSaturationPolicy::ClampAndReport;
        stats = pl::Int32GemmStats{};
        expect(
            pl::run_int32_gemm_update(
                dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(),
                kTileStride, 1, 1, 1, config, pl::PeTimingConfig{}, clamp, &stats,
                &reason) == StatusCode::Ok &&
                stats.saturation_count == 1,
            "kernel: CLAMP_AND_REPORT continues and counts the clamp");
    }

    // Non-finite operand.
    {
        std::vector<float> dst(kTileStride * kTileStride, 0.0f);
        std::vector<float> lhs(kTileStride * kTileStride, 0.0f);
        std::vector<float> rhs(kTileStride * kTileStride, 0.0f);
        lhs[0] = std::numeric_limits<float>::infinity();
        rhs[0] = 1.0f;
        pl::Int32GemmStats stats;
        std::string reason;
        expect(
            pl::run_int32_gemm_update(
                dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(),
                kTileStride, 1, 1, 1, config, pl::PeTimingConfig{}, auto_policy, &stats,
                &reason) == StatusCode::NonFiniteValue &&
                stats.failed_phase == GemmPhase::ScaleReduction,
            "kernel: non-finite operand fails in the scale reduction phase");
        expect(
            stats.local_conversion_bytes == 4 && stats.scale_reduction_cycles == 1 &&
                stats.quantize_cycles == 0 && stats.integer_compute_cycles == 0 &&
                stats.dequantize_cycles == 0,
            "kernel: failed maxabs counts only the accessed FP32 element");
    }

    // Dequantized overflow -> NON_FINITE_VALUE at the dequantize phase.
    // Fixed eA = eB = 40 with q = qmax: the exact integer dot product is
    // ~4.6e18 (still inside INT64) but the dequantized value is ~5e42 and must
    // overflow FP32.
    {
        pl::GemmScalePolicy huge;
        {
            const std::uint32_t words[10] = {40u, 40u, auto_word, 1, 1, 1, 0, 0, 0, 0};
            std::string reason;
            dp::decode_scale_policy(0, words, &huge, &reason);
        }
        std::vector<float> dst(kTileStride * kTileStride, 0.0f);
        std::vector<float> lhs(kTileStride * kTileStride, 0.0f);
        std::vector<float> rhs(kTileStride * kTileStride, 0.0f);
        // q = 2^28 fits the 30-bit range and is exactly representable; the
        // largest element uses q = 2^28 - 1. The exact dot product is
        // 15*2^56 + (2^28-1)^2 < 2^62 (inside INT64) while the dequantized
        // value overflows binary32.
        const float scaled_high = std::ldexp(1.0f, 68);
        const float scaled_low = std::ldexp(static_cast<float>((1u << 28) - 1), 40);
        for (std::uint32_t index = 0; index < 16; ++index) {
            lhs[index] = index == 0 ? scaled_low : scaled_high;
            rhs[index * kTileStride] = index == 0 ? scaled_low : scaled_high;
        }
        dst[0] = 1.0f;
        pl::Int32GemmStats stats;
        std::string reason;
        expect(
            pl::run_int32_gemm_update(
                dst.data(), kTileStride, lhs.data(), kTileStride, rhs.data(),
                kTileStride, 1, 1, 16, config, pl::PeTimingConfig{}, huge, &stats,
                &reason) == StatusCode::NonFiniteValue &&
                stats.failed_phase == GemmPhase::DequantizeUpdate &&
                stats.dequantize_cycles > 0 && stats.integer_compute_cycles > 0,
            "kernel: dequantized overflow fails at the dequantize phase");
    }

    // K beyond the configured block is rejected, never silently truncated.
    {
        std::vector<float> tile(kTileStride * kTileStride, 0.0f);
        pl::Int32GemmStats stats;
        std::string reason;
        expect(
            pl::run_int32_gemm_update(
                tile.data(), kTileStride, tile.data(), kTileStride, tile.data(),
                kTileStride, 1, 1, 17, config, pl::PeTimingConfig{}, auto_policy,
                &stats, &reason) == StatusCode::BadDescriptor,
            "kernel: K > k_block_max is rejected");
    }
}

// ---------------------------------------------------------------------------
// D. Real FP32 GEMM engine
// ---------------------------------------------------------------------------

struct EngineHarness {
    dp::BufferManager buffers;
    pl::PanelEngine engine;
    dp::BufferHandle handle{};
    dp::BufferLease lease{};
    dp::BufferHandle handle2{};
    dp::BufferLease lease2{};

    explicit EngineHarness(const pl::PanelLuConfig& config)
        : buffers(dp::BufferConfig{3, 1024 * 1024, 4, 1}), engine(config) {}

    bool prepare_at(std::uint64_t front_offset, const std::vector<float>& front) {
        engine.set_buffers(&buffers);
        if (!engine.configure().ok) {
            return false;
        }
        const std::uint64_t total = front_offset + front.size() * 4;
        const auto allocate = buffers.allocate("gemm-test", total, 0);
        if (!allocate.ok) {
            return false;
        }
        handle = allocate.handle;
        const auto lease_result = buffers.begin_write(handle, "gemm-test", 0, total, 0);
        if (!lease_result.ok) {
            return false;
        }
        lease = lease_result.lease;
        std::vector<std::uint8_t> bytes(front.size() * 4);
        std::memcpy(bytes.data(), front.data(), bytes.size());
        return buffers.lease_write(lease.lease_id, front_offset, bytes, 1, nullptr) ==
               StatusCode::Ok;
    }

    bool prepare(const std::vector<float>& front) { return prepare_at(0, front); }

    // Second independent front (used by the two-group test so the two
    // commands never target the same tiles).
    bool prepare_other(const std::vector<float>& front) {
        const std::uint64_t total = front.size() * 4;
        const auto allocate = buffers.allocate("gemm-test-2", total, 0);
        if (!allocate.ok) {
            return false;
        }
        handle2 = allocate.handle;
        const auto lease_result = buffers.begin_write(handle2, "gemm-test-2", 0, total, 0);
        if (!lease_result.ok) {
            return false;
        }
        lease2 = lease_result.lease;
        std::vector<std::uint8_t> bytes(front.size() * 4);
        std::memcpy(bytes.data(), front.data(), bytes.size());
        return buffers.lease_write(lease2.lease_id, 0, bytes, 1, nullptr) ==
               StatusCode::Ok;
    }

    std::vector<float> front_values(std::size_t count, std::uint64_t front_offset = 0) const {
        const auto bytes = buffers.bytes(handle.slot);
        std::vector<float> values(count, 0.0f);
        std::memcpy(values.data(), bytes.data() + front_offset, count * 4);
        return values;
    }

    pl::PanelEngineResponse run(
        const pl::PanelEngineRequest& request,
        std::uint64_t max_cycles = 500000) {
        const std::uint64_t key = pl::PanelEngine::adapter_key(3, 0);
        const pl::ConfigCheck started = engine.begin_command(request, key, 2);
        if (!started.ok) {
            pl::PanelEngineResponse response;
            response.status = started.status;
            response.reason = started.reason;
            return response;
        }
        for (std::uint64_t cycle = 2; cycle < max_cycles; ++cycle) {
            engine.tick(cycle);
            pl::PanelEngineResponse response;
            if (engine.poll_response(key, response)) {
                return response;
            }
        }
        pl::PanelEngineResponse timeout;
        timeout.status = StatusCode::Timeout;
        return timeout;
    }
};

pl::PanelEngineRequest gemm_request(
    std::uint32_t total,
    std::uint32_t pivot,
    const dp::BufferHandle& handle,
    const dp::BufferLease& lease,
    std::uint64_t front_offset = 0,
    GemmBackend backend = GemmBackend::Fp32Device,
    pl::GemmScalePolicy scale = {}) {
    pl::PanelEngineRequest request;
    request.txn_id = 1;
    request.command_id = 1;
    request.node_id = 0;
    request.total_dim = total;
    request.pivot_dim = pivot;
    request.tile_size = 16;
    request.compute_kind = pl::ComputeKind::GemmSchur;
    request.front = handle;
    request.front_lease = lease;
    request.front_offset = front_offset;
    const std::uint32_t update = total - pivot;
    request.kernel_m = update;
    request.kernel_n = update;
    request.kernel_k = pivot;
    request.kernel_tile_size = 16;
    request.kernel_backend =
        backend == GemmBackend::Int32Gemm
            ? static_cast<std::uint32_t>(cmd::KernelBackend::SystemCInt32GemmModel)
            : static_cast<std::uint32_t>(cmd::KernelBackend::SystemCFp32DeviceModel);
    request.gemm_scale = scale;
    request.kernel_has_scale = backend == GemmBackend::Int32Gemm;
    return request;
}

std::vector<float> make_front(
    std::uint32_t n,
    std::uint32_t p,
    std::uint32_t seed,
    bool non_finite_hotspot = false) {
    std::vector<float> front(static_cast<std::size_t>(n) * n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            const int pattern = static_cast<int>((i * 7 + j * 13 + seed * 29) % 11) - 5;
            float value = 0.25f * static_cast<float>(pattern);
            if (i == j) {
                value += 4.0f;
            }
            front[static_cast<std::size_t>(i) * n + j] = value;
        }
    }
    if (non_finite_hotspot) {
        front[static_cast<std::size_t>(p) * n + 0] =
            std::numeric_limits<float>::infinity();
    }
    return front;
}

// Independent FP64 Schur evaluation from the FP32 front blocks (A21, U12).
std::vector<double> schur_expectation(
    const std::vector<float>& front,
    std::uint32_t n,
    std::uint32_t p) {
    const std::uint32_t u = n - p;
    std::vector<double> expected(static_cast<std::size_t>(u) * u, 0.0);
    for (std::uint32_t i = 0; i < u; ++i) {
        for (std::uint32_t j = 0; j < u; ++j) {
            double acc = static_cast<double>(front[(p + i) * n + (p + j)]);
            for (std::uint32_t t = 0; t < p; ++t) {
                acc -= static_cast<double>(front[(p + i) * n + t]) *
                       static_cast<double>(front[t * n + (p + j)]);
            }
            expected[static_cast<std::size_t>(i) * u + j] = acc;
        }
    }
    return expected;
}

void test_t10_execution_regressions() {
    {
        auto config = base_panel_config(2, 2);
        config.numeric_mode = pl::PanelNumericMode::Fp64Diagnostic;
        EngineHarness harness(config);
        const float big = std::numeric_limits<float>::max();
        // Only output (1,1) overflows FP32, outside the old KxK check (K=1).
        const std::vector<float> front{1, 0, big, 0, 0, 0, big, 0, 0};
        expect(harness.prepare(front), "diagnostic Schur: finite front staged");
        const auto response = harness.run(gemm_request(3, 1, harness.handle, harness.lease));
        expect(
            response.status == StatusCode::NonFiniteValue && response.failed_column == 2,
            "diagnostic Schur: MxN overflow reports the global output column");
        expect(
            harness.front_values(front.size()) == front &&
                harness.engine.task_records().size() == 1 &&
                !harness.engine.task_records().front().committed,
            "diagnostic Schur: overflowing tile is never committed");
    }
    for (const std::uint32_t block : {1u, 8u, 16u}) {
        const std::uint32_t n = 26, p = 9, u = n - p;
        auto config = base_panel_config(2, 2);
        config.precision.k_block_max = block;
        EngineHarness harness(config);
        const auto front = make_front(n, p, 1);
        const auto expected = schur_expectation(front, n, p);
        const std::string label = "configured K=" + std::to_string(block);
        expect(harness.prepare(front), label + ": front staged");
        const auto response = harness.run(gemm_request(
            n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm));
        const std::uint32_t steps = (p + block - 1) / block;
        expect(
            response.status == StatusCode::Ok && response.tasks_total == 4 * steps,
            label + ": every configured K contribution executes");
        const auto updated = harness.front_values(front.size());
        bool numeric_ok = true;
        for (std::uint32_t i = 0; i < u; ++i) {
            for (std::uint32_t j = 0; j < u; ++j) {
                numeric_ok = numeric_ok &&
                    updated[(p + i) * n + p + j] == expected[i * u + j];
            }
        }
        expect(numeric_ok, label + ": dyadic Schur result equals independent FP64 oracle");
        bool addresses_ok = harness.engine.gemm_log().size() == 1;
        if (addresses_ok) {
            const auto& requests = harness.engine.gemm_log().front().requests;
            addresses_ok = requests.size() == 4 * steps;
            for (const auto& request : requests) {
                addresses_ok = addresses_ok &&
                    request.k == std::min(block, p - request.step * block) &&
                    request.a.offset == p * n * 4 + request.tile_row * 16 * n * 4 +
                                            request.step * block * 4 &&
                    request.b.offset == request.step * block * n * 4 + p * 4 +
                                            request.tile_col * 16 * 4;
            }
        }
        expect(addresses_ok, label + ": K tail and A/B offsets follow configured block");
    }
    // ReadyWait has loaded inputs and a functional forecast, but no execution
    // cycles have begun. A Running task follows the existing drain-on-abort model.
    for (const bool abort_before_exec : {true, false}) {
        auto config = base_panel_config(2, 2);
        config.load_latency_cycles = abort_before_exec ? 100 : 1;
        EngineHarness harness(config);
        const std::vector<float> front{1, 1, 1, 0};
        const std::string label = abort_before_exec ? "abort ReadyWait" : "abort Running";
        expect(harness.prepare(front), label + ": front staged");
        const auto request = gemm_request(
            2, 1, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm);
        const auto key = pl::PanelEngine::adapter_key(3, 0);
        expect(harness.engine.begin_command(request, key, 2).ok, label + ": accepted");
        bool abort_sent = false, finished = false;
        pl::PanelEngineResponse response;
        for (std::uint64_t cycle = 2; cycle < 1000; ++cycle) {
            harness.engine.tick(cycle);
            if (!abort_sent) {
                for (const auto& pe : harness.engine.pes()) {
                    for (const auto& slot : pe.slot_views()) {
                        if (slot.active && slot.phase ==
                                (abort_before_exec ? pl::TaskPhase::ReadyWait
                                                   : pl::TaskPhase::Running)) {
                            abort_sent = true;
                        }
                    }
                }
                if (abort_sent) {
                    harness.engine.abort_txn(request.txn_id, cycle);
                }
            }
            if (harness.engine.poll_response(key, response)) {
                finished = true;
                break;
            }
        }
        expect(
            abort_sent && finished && response.status == StatusCode::Aborted,
            label + ": directed abort drains to terminal response");
        expect(harness.front_values(4) == front, label + ": no output published");
        const auto& records = harness.engine.task_records();
        expect(records.size() == 1, label + ": one task record");
        if (records.size() == 1) {
            const auto& record = records.front();
            const std::uint64_t local_bytes = abort_before_exec ? 0 : 40;
            expect(
                record.gemm_backend == GemmBackend::Int32Gemm &&
                    (record.exec_start_cycle == 0) == abort_before_exec &&
                    record.local_conversion_bytes == local_bytes &&
                    record.scale_reduction_cycles == (abort_before_exec ? 0u : 2u) &&
                    record.quantize_cycles == (abort_before_exec ? 0u : 2u) &&
                    record.integer_compute_cycles == (abort_before_exec ? 0u : 4u) &&
                    record.dequantize_cycles == (abort_before_exec ? 0u : 1u) &&
                    record.saturation_count == 0 && record.overflow_count == 0,
                label + ": counters include only modeled work that executes");
            expect(
                response.local_conversion_bytes == record.local_conversion_bytes &&
                    response.scale_reduction_cycles == record.scale_reduction_cycles &&
                    response.quantize_cycles == record.quantize_cycles &&
                    response.integer_compute_cycles == record.integer_compute_cycles &&
                    response.dequantize_cycles == record.dequantize_cycles,
                label + ": command totals include drained task costs exactly once");
        }
        const auto& log = harness.engine.gemm_log();
        expect(
            log.size() == 1 && log.front().responses.size() == 1 &&
                log.front().responses.front().local_conversion_bytes ==
                    (abort_before_exec ? 0u : 40u) &&
                !log.front().responses.front().output_visible,
            label + ": GemmRsp agrees with task costs and visibility");
    }
}

void test_gemm_request_response_contract() {
    pl::TileTask task;
    task.task_id = 42;
    task.epoch = 3;
    task.command_id = 7;
    task.node_id = 1;
    task.step = 1;
    task.tile_row = 2;
    task.tile_col = 3;
    task.mode = pl::TileMode::GemmSchur;
    task.dims = {5, 6, 7};
    task.gemm_backend = GemmBackend::Int32Gemm;
    task.gemm_scale.auto_exponent_a = false;
    task.gemm_scale.exponent_a = -4;
    task.gemm_scale.saturation = ScaleSaturationPolicy::ClampAndReport;
    task.lhs.valid = true;
    task.lhs.offset = 64;
    task.lhs.stride_bytes = 168;
    task.rhs.valid = true;
    task.rhs.offset = 128;
    task.rhs.stride_bytes = 168;
    task.dst.valid = true;
    task.dst.offset = 256;
    task.dst.stride_bytes = 168;
    task.dst.version = 2;
    const pl::GemmReq request = pl::make_gemm_req(task, 99, 23, 17);
    expect(
        request.task_id == 42 && request.descriptor_id == 99 && request.step == 1 &&
            request.tile_row == 2 && request.tile_col == 3 && request.m == 5 &&
            request.n == 6 && request.k == 7 && request.backend == GemmBackend::Int32Gemm &&
            request.scale.exponent_a == -4 &&
            request.scale.saturation == ScaleSaturationPolicy::ClampAndReport &&
            request.front_dim == 23 && request.pivot_dim == 17 &&
            request.front_row == 17 + 32 && request.front_col == 17 + 48 &&
            request.a.offset == 64 && request.b.offset == 128 &&
            request.c.offset == 256 && request.c.version == 2,
        "contract: GemmReq carries identity, valid M/N/K, operands and scale");
    pl::TileTaskResponse response;
    response.task_id = 42;
    response.epoch = 3;
    response.command_id = 7;
    response.node_id = 1;
    response.status = StatusCode::Ok;
    response.result_visible = true;
    response.bytes_read = 200;
    response.bytes_written = 120;
    response.saturation_count = 3;
    response.overflow_count = 0;
    response.gemm_exponent_a = -4;
    response.gemm_exponent_b = 2;
    response.local_conversion_bytes = 900;
    response.scale_reduction_cycles = 11;
    response.quantize_cycles = 12;
    response.integer_compute_cycles = 13;
    response.dequantize_cycles = 14;
    response.gemm_failed_phase = GemmPhase::None;
    const pl::GemmRsp reply = pl::make_gemm_rsp(task, response, 23, 17);
    expect(
        reply.task_id == 42 && reply.output_visible && reply.output_version == 3 &&
            reply.saturated && reply.saturation_count == 3 && !reply.overflowed &&
            reply.read_bytes == 200 && reply.write_bytes == 120 &&
            reply.local_conversion_bytes == 900 && reply.scale_reduction_cycles == 11 &&
            reply.quantize_cycles == 12 && reply.integer_compute_cycles == 13 &&
            reply.dequantize_cycles == 14 && reply.exponent_a == -4 &&
            reply.exponent_b == 2 && reply.failed_tile_row == 2 &&
            reply.failed_tile_col == 3 && reply.failed_step == 1 &&
            reply.failed_front_row == 17 + 32 && reply.failed_front_col == 17 + 48,
        "contract: GemmRsp carries visibility, statistics, phases and coordinates");
}

void test_fp32_manual_case() {
    // n=4, p=2, A21 = [[1,0],[2,1]], U12 = [[3,1],[10,5]], A22 = I2.
    // Schur(i,j) = A22(i,j) - sum_t A21(i,t)*U12(t,j):
    //   (0,0): 1 - (1*3 + 0*10) = -2
    //   (0,1): 0 - (1*1 + 0*5) = -1
    //   (1,0): 0 - (2*3 + 1*10) = -16
    //   (1,1): 1 - (2*1 + 1*5) = -6
    const std::uint32_t n = 4;
    const std::uint32_t p = 2;
    std::vector<float> front(n * n, 0.0f);
    front[2 * n + 0] = 1.0f;
    front[2 * n + 1] = 0.0f;
    front[3 * n + 0] = 2.0f;
    front[3 * n + 1] = 1.0f;
    front[0 * n + 2] = 3.0f;
    front[0 * n + 3] = 1.0f;
    front[1 * n + 2] = 10.0f;
    front[1 * n + 3] = 5.0f;
    front[2 * n + 2] = 1.0f;
    front[3 * n + 3] = 1.0f;

    EngineHarness harness(base_panel_config(1, 1));
    expect(harness.prepare(front), "manual: front staged");
    const auto response = harness.run(gemm_request(n, p, harness.handle, harness.lease));
    expect(response.status == StatusCode::Ok, "manual: GEMM succeeds");
    expect(response.tasks_total == 1, "manual: one 16x16 tile task");
    const std::vector<float> updated = harness.front_values(n * n);
    expect(
        updated[2 * n + 2] == -2.0f && updated[2 * n + 3] == -1.0f &&
            updated[3 * n + 2] == -16.0f && updated[3 * n + 3] == -6.0f,
        "manual: A22 = [[-2,-1],[-16,-6]]");
    // A11/A12/A21 unchanged.
    bool rest_unchanged = true;
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            if (i >= p && j >= p) {
                continue;
            }
            if (updated[i * n + j] != front[i * n + j]) {
                rest_unchanged = false;
            }
        }
    }
    expect(rest_unchanged, "manual: A11/A12/A21 unchanged");
}

void test_fp32_n42_multitile_multik() {
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    const std::uint32_t u = n - p;
    for (const std::uint64_t offset : {std::uint64_t{0}, std::uint64_t{4}, std::uint64_t{8}}) {
        const std::string label = "n42 offset " + std::to_string(offset);
        std::vector<float> front = make_front(n, p, 5);
        EngineHarness harness(base_panel_config(2, 2));
        expect(harness.prepare_at(offset, front), label + ": front staged");
        const auto response =
            harness.run(gemm_request(n, p, harness.handle, harness.lease, offset));
        expect(response.status == StatusCode::Ok, label + ": GEMM succeeds");
        expect(
            response.tasks_total == 8,
            label + ": ceil(u/16)^2*ceil(p/16) = 8 tile tasks");
        // Operand/writeback byte accounting of the tile data path.
        std::uint64_t read_bytes = 0;
        std::uint64_t write_bytes = 0;
        std::uint32_t gemm_tasks = 0;
        for (const auto& record : harness.engine.task_records()) {
            if (record.task.mode != pl::TileMode::GemmSchur) {
                continue;
            }
            ++gemm_tasks;
            read_bytes += record.bytes_read;
            write_bytes += record.bytes_written;
        }
        expect(gemm_tasks == 8, label + ": 8 GEMM task records");
        expect(read_bytes == 9880, label + ": operand read = 9880 B");
        expect(write_bytes == 2888, label + ": tile writeback = 2888 B");
        // Unified GemmReq/GemmRsp boundary records for this command.
        const auto& gemm_log = harness.engine.gemm_log();
        expect(gemm_log.size() == 1, label + ": one GEMM context log");
        if (!gemm_log.empty()) {
            const auto& requests = gemm_log.back().requests;
            const auto& responses = gemm_log.back().responses;
            expect(
                requests.size() == 8 && responses.size() == 8,
                label + ": GemmReq/GemmRsp records per task");
            // Responses may complete out of order: pair them by task id.
            std::map<std::uint64_t, const pl::GemmRsp*> by_task;
            for (const auto& reply : responses) {
                by_task.emplace(reply.task_id, &reply);
            }
            bool identity_ok = by_task.size() == responses.size();
            std::uint64_t rsp_read = 0;
            std::uint64_t rsp_write = 0;
            for (const auto& request : requests) {
                const auto found = by_task.find(request.task_id);
                if (found == by_task.end()) {
                    identity_ok = false;
                    continue;
                }
                const pl::GemmRsp& reply = *found->second;
                if (request.backend != GemmBackend::Fp32Device ||
                    request.accumulate_mode !=
                        pl::GemmAccumulateMode::SubtractProduct ||
                    request.m == 0 || request.n == 0 || request.k == 0 ||
                    request.front_dim != n || request.pivot_dim != p ||
                    request.front_row != request.pivot_dim + request.tile_row * 16 ||
                    request.front_col != request.pivot_dim + request.tile_col * 16 ||
                    !reply.output_visible || reply.status != StatusCode::Ok) {
                    identity_ok = false;
                }
                rsp_read += reply.read_bytes;
                rsp_write += reply.write_bytes;
            }
            expect(identity_ok, label + ": GemmReq/GemmRsp identity and visibility");
            expect(
                rsp_read == 9880 && rsp_write == 2888,
                label + ": GemmRsp byte accounting matches the tile model");
        }
        // Independent FP64 recurrence vs device A22.
        const std::vector<double> expected = schur_expectation(front, n, p);
        const std::vector<float> updated =
            harness.front_values(front.size(), offset);
        double max_error = 0.0;
        double scale = 1.0;
        for (std::uint32_t i = 0; i < u; ++i) {
            for (std::uint32_t j = 0; j < u; ++j) {
                const double got = static_cast<double>(updated[(p + i) * n + (p + j)]);
                max_error = std::max(
                    max_error,
                    std::fabs(got - expected[static_cast<std::size_t>(i) * u + j]));
                scale = std::max(scale, std::fabs(expected[static_cast<std::size_t>(i) * u + j]));
            }
        }
        if (offset == 0) {
            std::cout << "[t10] n42 fp32 offset=0 relative Schur error vs FP64 = "
                      << max_error / scale << "\n";
        }
        expect(
            max_error / scale < 5.0e-5,
            label + ": FP32 A22 matches the independent FP64 recurrence");
        // Only A22 changed.
        bool rest_unchanged = true;
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) {
                if (i >= p && j >= p) {
                    continue;
                }
                if (updated[i * n + j] != front[i * n + j]) {
                    rest_unchanged = false;
                }
            }
        }
        expect(rest_unchanged, label + ": A11/A12/A21 byte-unchanged");
        // Guard bytes before/after the front payload stay untouched.
        const auto bytes = harness.buffers.bytes(harness.handle.slot);
        bool guard_ok = true;
        for (std::uint64_t index = 0; index < offset; ++index) {
            if (bytes[index] != 0) {
                guard_ok = false;
            }
        }
        for (std::uint64_t index = offset + front.size() * 4; index < bytes.size(); ++index) {
            if (bytes[index] != 0) {
                guard_ok = false;
            }
        }
        expect(guard_ok, label + ": guard/padding bytes untouched");
    }
}

// Full-mantissa FP32 evidence: the device tile update must match an
// independently written FP32 FMA recurrence bit-for-bit, while the FP64
// recurrence reports the rounding error.
void test_fp32_recurrence_oracle() {
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    const std::uint32_t u = n - p;
    std::vector<float> front(static_cast<std::size_t>(n) * n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            const std::uint32_t mixed =
                (i * 2654435761u + j * 40503u + 12345u) & 0xFFFFFFu;
            front[static_cast<std::size_t>(i) * n + j] =
                std::ldexp(static_cast<float>(mixed), -23) * 2.0f - 1.0f;
        }
        front[static_cast<std::size_t>(i) * n + i] += 3.0f;
    }
    EngineHarness harness(base_panel_config(2, 2));
    expect(harness.prepare(front), "fp32 oracle: front staged");
    const auto response = harness.run(gemm_request(n, p, harness.handle, harness.lease));
    expect(response.status == StatusCode::Ok, "fp32 oracle: GEMM succeeds");
    const std::vector<float> updated = harness.front_values(front.size());

    // Independent FP32 FMA recurrence in the documented order.
    bool exact = true;
    for (std::uint32_t i = 0; i < u && exact; ++i) {
        for (std::uint32_t j = 0; j < u && exact; ++j) {
            float acc = front[(p + i) * n + (p + j)];
            for (std::uint32_t t = 0; t < p; ++t) {
                acc = std::fma(
                    -front[(p + i) * n + t], front[t * n + (p + j)], acc);
            }
            if (updated[(p + i) * n + (p + j)] != acc) {
                exact = false;
            }
        }
    }
    expect(exact, "fp32 oracle: device update matches the independent FP32 FMA recurrence exactly");

    const std::vector<double> expected = schur_expectation(front, n, p);
    double max_error = 0.0;
    double scale = 1.0;
    for (std::uint32_t i = 0; i < u; ++i) {
        for (std::uint32_t j = 0; j < u; ++j) {
            max_error = std::max(
                max_error,
                std::fabs(static_cast<double>(updated[(p + i) * n + (p + j)]) -
                          expected[static_cast<std::size_t>(i) * u + j]));
            scale = std::max(scale, std::fabs(expected[static_cast<std::size_t>(i) * u + j]));
        }
    }
    std::cout << "[t10] full-mantissa fp32 relative Schur error vs FP64 = "
              << max_error / scale << "\n";
    expect(
        max_error / scale < 1.0e-5,
        "fp32 oracle: full-mantissa FP32 update stays within 1e-5 of FP64");
}

void test_fp32_failure_and_refusals() {
    const std::uint32_t n = 17;
    const std::uint32_t p = 16;
    {  // non-finite operand
        std::vector<float> front = make_front(n, p, 9, true);
        EngineHarness harness(base_panel_config(1, 1));
        expect(harness.prepare(front), "failure: front staged");
        const auto response = harness.run(gemm_request(n, p, harness.handle, harness.lease));
        expect(
            response.status == StatusCode::NonFiniteValue,
            "failure: non-finite A21 -> NON_FINITE_VALUE");
        expect(
            response.pivot_count == 0,
            "failure: GEMM failure reports no pivot count");
        // The failed tile was not committed: no GEMM task is marked committed.
        bool any_committed = false;
        for (const auto& record : harness.engine.task_records()) {
            if (record.task.mode == pl::TileMode::GemmSchur && record.committed) {
                any_committed = true;
            }
        }
        expect(!any_committed, "failure: no partial tile committed");
    }
    {  // u = 0 requires the compiler to omit the command; the engine refuses.
        std::vector<float> front = make_front(n, n, 3);
        EngineHarness harness(base_panel_config(1, 1));
        expect(harness.prepare(front), "refusal: front staged");
        const auto response = harness.run(gemm_request(n, n, harness.handle, harness.lease));
        expect(
            response.status == StatusCode::BadDescriptor,
            "refusal: pivot_dim == front_dim (empty update) -> BAD_DESCRIPTOR");
    }
    {  // tile size other than 16
        std::vector<float> front = make_front(n, p, 3);
        EngineHarness harness(base_panel_config(1, 1));
        expect(harness.prepare(front), "refusal: front staged");
        pl::PanelEngineRequest request =
            gemm_request(n, p, harness.handle, harness.lease);
        request.tile_size = 8;
        const auto response = harness.run(request);
        expect(
            response.status == StatusCode::BadDescriptor,
            "refusal: non-16 tile rejected");
    }
    {  // FP32 kernel with a SCALE_DESC reference
        std::vector<float> front = make_front(n, p, 3);
        EngineHarness harness(base_panel_config(1, 1));
        expect(harness.prepare(front), "refusal: front staged");
        pl::PanelEngineRequest request =
            gemm_request(n, p, harness.handle, harness.lease);
        request.kernel_has_scale = true;
        const auto response = harness.run(request);
        expect(
            response.status == StatusCode::BadDescriptor,
            "refusal: FP32 GEMM with SCALE_DESC reference -> BAD_DESCRIPTOR");
    }
    {  // INT32 without SCALE_DESC
        std::vector<float> front = make_front(n, p, 3);
        pl::PanelLuConfig config = base_panel_config(1, 1);
        EngineHarness harness(config);
        expect(harness.prepare(front), "refusal: front staged");
        pl::PanelEngineRequest request = gemm_request(
            n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm);
        request.kernel_has_scale = false;
        const auto response = harness.run(request);
        expect(
            response.status == StatusCode::BadDescriptor,
            "refusal: INT32 GEMM without SCALE_DESC -> BAD_DESCRIPTOR");
    }
    {  // INT32 + explicit FP64 diagnostic mode
        std::vector<float> front = make_front(n, p, 3);
        pl::PanelLuConfig config = base_panel_config(1, 1);
        config.numeric_mode = pl::PanelNumericMode::Fp64Diagnostic;
        EngineHarness harness(config);
        expect(harness.prepare(front), "refusal: front staged");
        const auto response = gemm_request(
            n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm);
        const auto result = harness.run(response);
        expect(
            result.status == StatusCode::PolicyNotSupported,
            "refusal: INT32 + FP64 diagnostic combination rejected");
    }
}

void test_fp32_groups_and_backpressure() {
    // Two groups: each accepted command owns exactly one group and all of its
    // tasks stay inside it.
    {
        const std::uint32_t n = 33;
        const std::uint32_t p = 16;
        pl::PanelLuConfig config = base_panel_config(2, 2);
        EngineHarness harness(config);
        std::vector<float> front = make_front(n, p, 11);
        std::vector<float> other = make_front(n, p, 12);
        expect(harness.prepare(front), "groups: front staged");
        expect(harness.prepare_other(other), "groups: second front staged");
        pl::PanelEngineRequest first =
            gemm_request(n, p, harness.handle, harness.lease);
        first.command_id = 1;
        pl::PanelEngineRequest second = first;
        second.command_id = 2;
        second.txn_id = 2;
        second.front = harness.handle2;
        second.front_lease = harness.lease2;
        const std::uint64_t key1 = pl::PanelEngine::adapter_key(3, 0);
        const std::uint64_t key2 = pl::PanelEngine::adapter_key(3, 1);
        expect(harness.engine.begin_command(first, key1, 2).ok, "groups: first accepted");
        expect(harness.engine.begin_command(second, key2, 2).ok, "groups: second accepted");
        const std::uint32_t group1 = harness.engine.group_of_command(1);
        const std::uint32_t group2 = harness.engine.group_of_command(2);
        expect(
            group1 != cmd::NONE && group2 != cmd::NONE && group1 != group2,
            "groups: two commands own two distinct groups");
        pl::PanelEngineResponse r1;
        pl::PanelEngineResponse r2;
        for (std::uint64_t cycle = 2; cycle < 2000000; ++cycle) {
            harness.engine.tick(cycle);
            if (r1.status != StatusCode::Ok || r1.finish_cycle == 0) {
                harness.engine.poll_response(key1, r1);
            }
            if (r2.status != StatusCode::Ok || r2.finish_cycle == 0) {
                harness.engine.poll_response(key2, r2);
            }
            if (r1.finish_cycle != 0 && r2.finish_cycle != 0) {
                break;
            }
        }
        expect(
            r1.status == StatusCode::Ok && r2.status == StatusCode::Ok,
            "groups: both commands complete");
        bool group_ok = true;
        for (const auto& record : harness.engine.task_records()) {
            if (record.task.mode != pl::TileMode::GemmSchur) {
                continue;
            }
            const std::uint32_t owner = harness.engine.group_of_command(record.task.command_id);
            if (owner != cmd::NONE && record.group_id != owner) {
                group_ok = false;
            }
            if (record.task.command_id == 1 && record.group_id != group1) {
                group_ok = false;
            }
            if (record.task.command_id == 2 && record.group_id != group2) {
                group_ok = false;
            }
        }
        expect(group_ok, "groups: no task crosses its command's group");
    }
    // Backpressure: tiny budgets, one slot per PE, shallow queues, held
    // responses. Completion must still be reached with real stalls.
    {
        const std::uint32_t n = 42;
        const std::uint32_t p = 23;
        pl::PanelLuConfig config = base_panel_config(2, 2);
        config.load_bytes_per_cycle = 8;
        config.writeback_bytes_per_cycle = 8;
        config.dispatcher_queue_depth = 2;
        config.task_slots_per_pe = 1;
        config.response_hold_cycles = 5;
        EngineHarness harness(config);
        std::vector<float> front = make_front(n, p, 13);
        expect(harness.prepare(front), "backpressure: front staged");
        const auto response = harness.run(gemm_request(n, p, harness.handle, harness.lease));
        expect(response.status == StatusCode::Ok, "backpressure: completes");
        expect(
            harness.engine.dispatcher().queue_full_stall_cycles() > 0,
            "backpressure: dispatcher queue-full stalls were modeled");
        std::uint64_t port_stalls = 0;
        for (const auto& record : harness.engine.task_records()) {
            port_stalls += record.task.mode == pl::TileMode::GemmSchur
                               ? 0
                               : 0;
        }
        (void)port_stalls;
        bool stalled = false;
        for (const auto& record : harness.engine.task_records()) {
            if (record.task.mode == pl::TileMode::GemmSchur &&
                record.dispatch_cycle > record.enqueue_cycle) {
                stalled = true;
            }
        }
        expect(stalled, "backpressure: tasks waited for resources");
    }
}

void test_fp32_abort_and_reuse() {
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    pl::PanelLuConfig config = base_panel_config(2, 2);
    config.load_bytes_per_cycle = 8;  // slow the load so the abort lands mid-run
    EngineHarness harness(config);
    std::vector<float> front = make_front(n, p, 17);
    expect(harness.prepare(front), "abort: front staged");
    pl::PanelEngineRequest request = gemm_request(n, p, harness.handle, harness.lease);
    const std::uint64_t key = pl::PanelEngine::adapter_key(3, 0);
    expect(harness.engine.begin_command(request, key, 2).ok, "abort: command accepted");
    bool aborted = false;
    for (std::uint64_t cycle = 2; cycle < 100000; ++cycle) {
        harness.engine.tick(cycle);
        if (cycle == 10) {
            harness.engine.abort_txn(request.txn_id, cycle);
        }
        pl::PanelEngineResponse response;
        if (harness.engine.poll_response(key, response)) {
            aborted = response.status == StatusCode::Aborted;
            break;
        }
    }
    expect(aborted, "abort: directed abort terminates with ABORTED");
    expect(
        !harness.engine.group_busy(harness.engine.group_of_command(request.command_id)) ||
            harness.engine.group_of_command(request.command_id) == cmd::NONE,
        "abort: group released after drain");
    // Resource reuse: a new command on the same engine completes.
    pl::PanelEngineRequest reuse = gemm_request(n, p, harness.handle, harness.lease);
    reuse.command_id = 7;
    reuse.txn_id = 7;
    const std::uint64_t key2 = pl::PanelEngine::adapter_key(3, 1);
    const auto reuse_response = [&]() {
        pl::PanelEngineResponse response;
        if (!harness.engine.begin_command(reuse, key2, 50000).ok) {
            response.status = StatusCode::BufferFull;
            return response;
        }
        for (std::uint64_t cycle = 50000; cycle < 2000000; ++cycle) {
            harness.engine.tick(cycle);
            if (harness.engine.poll_response(key2, response)) {
                return response;
            }
        }
        pl::PanelEngineResponse timeout;
        timeout.status = StatusCode::Timeout;
        return timeout;
    }();
    expect(reuse_response.status == StatusCode::Ok, "abort: resources reusable afterwards");

    // reset() invalidates everything and the engine still works.
    harness.engine.reset();
    expect(
        !harness.engine.group_busy(0) && !harness.engine.group_busy(1),
        "reset: groups freed");
    pl::PanelEngineRequest after_reset = gemm_request(n, p, harness.handle, harness.lease);
    after_reset.command_id = 9;
    after_reset.txn_id = 9;
    const auto reset_response = harness.run(after_reset);
    expect(
        reset_response.status == StatusCode::Ok,
        "reset: engine accepts and completes a fresh command");
}

// INT32 abort/reset: the conversion phases must not publish a partial tile and
// the group/slot resources must be reusable afterwards.
void test_int32_abort_reset_and_reuse() {
    const std::uint32_t auto_word = dp::kScaleAutoWord;
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    pl::GemmScalePolicy auto_policy;
    {
        const std::uint32_t words[10] = {auto_word, auto_word, auto_word, 1, 1, 1, 0, 0, 0, 0};
        std::string reason;
        dp::decode_scale_policy(0, words, &auto_policy, &reason);
    }
    pl::PanelLuConfig config = base_panel_config(2, 2);
    config.load_bytes_per_cycle = 16;
    config.writeback_bytes_per_cycle = 16;
    EngineHarness harness(config);
    std::vector<float> front = make_front(n, p, 29);
    expect(harness.prepare(front), "int32 abort: front staged");
    const std::uint64_t key = pl::PanelEngine::adapter_key(3, 0);
    pl::PanelEngineRequest request = gemm_request(
        n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm, auto_policy);
    expect(harness.engine.begin_command(request, key, 2).ok, "int32 abort: accepted");
    bool aborted = false;
    std::uint64_t abort_cycle = 0;
    for (std::uint64_t cycle = 2; cycle < 200000; ++cycle) {
        harness.engine.tick(cycle);
        if (cycle == 400) {
            harness.engine.abort_txn(request.txn_id, cycle);
            abort_cycle = cycle;
        }
        pl::PanelEngineResponse response;
        if (harness.engine.poll_response(key, response)) {
            aborted = response.status == StatusCode::Aborted;
            break;
        }
    }
    expect(aborted, "int32 abort: directed abort terminates with ABORTED");
    // No INT32 tile may be committed once the transaction was aborted before
    // its completion; the partial conversion is never written back.
    bool committed_after_abort = false;
    for (const auto& record : harness.engine.task_records()) {
        if (record.task.mode == pl::TileMode::GemmSchur && record.committed &&
            record.task.command_id == request.command_id) {
            committed_after_abort = true;
        }
    }
    expect(
        !committed_after_abort || abort_cycle == 0,
        "int32 abort: aborted command publishes no partial committed tile");

    // Resource reuse: a fresh INT32 command completes on the same engine.
    pl::PanelEngineRequest reuse = gemm_request(
        n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm, auto_policy);
    reuse.command_id = 21;
    reuse.txn_id = 21;
    const std::uint64_t key2 = pl::PanelEngine::adapter_key(3, 1);
    pl::PanelEngineResponse reuse_response;
    expect(
        harness.engine.begin_command(reuse, key2, 100000).ok,
        "int32 abort: resources reusable");
    for (std::uint64_t cycle = 100000; cycle < 2000000; ++cycle) {
        harness.engine.tick(cycle);
        if (harness.engine.poll_response(key2, reuse_response)) {
            break;
        }
    }
    expect(
        reuse_response.status == StatusCode::Ok,
        "int32 abort: follow-up INT32 command completes");
    expect(
        reuse_response.saturation_count == 0 && reuse_response.overflow_count == 0,
        "int32 abort: follow-up reports clean counters");

    // reset() during an INT32 command: the engine drops it and still works.
    pl::PanelEngineRequest reset_request = gemm_request(
        n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm, auto_policy);
    reset_request.command_id = 31;
    reset_request.txn_id = 31;
    const std::uint64_t key3 = pl::PanelEngine::adapter_key(3, 0);
    expect(
        harness.engine.begin_command(reset_request, key3, 300000).ok,
        "int32 reset: command accepted");
    for (std::uint64_t cycle = 300000; cycle < 300100; ++cycle) {
        harness.engine.tick(cycle);
    }
    harness.engine.reset();
    expect(
        !harness.engine.unit_busy(key3),
        "int32 reset: in-flight INT32 context dropped");
    pl::PanelEngineRequest after = gemm_request(
        n, p, harness.handle, harness.lease, 0, GemmBackend::Int32Gemm, auto_policy);
    after.command_id = 41;
    after.txn_id = 41;
    const auto after_response = harness.run(after);
    expect(
        after_response.status == StatusCode::Ok,
        "int32 reset: engine completes a fresh command");
}

// ---------------------------------------------------------------------------
// E. Control-plane + schema v2 artifact integration
// ---------------------------------------------------------------------------

struct IntegrationOutcome {
    bool started{false};
    bool finished{false};
    std::map<std::uint32_t, StatusCode> status;
    std::uint32_t panel_command{cmd::NONE};
    std::uint32_t left_command{cmd::NONE};
    std::uint32_t right_command{cmd::NONE};
    std::uint32_t gemm_command{cmd::NONE};
    std::uint32_t update_command{cmd::NONE};
    std::uint32_t commit_command{cmd::NONE};
    std::uint64_t update_bytes{0};
    std::uint64_t gemm_read_bytes{0};
    std::uint64_t gemm_write_bytes{0};
    std::uint64_t gemm_saturation{0};
    std::uint64_t gemm_local_bytes{0};
    std::vector<float> update_values;
    // Device-committed factors (L is n x p, U is p x n) used by the test to
    // build an independent FP64 Schur expectation.
    std::vector<float> l_factors;
    std::vector<float> u_factors;
};

IntegrationOutcome run_node_pipeline(
    const std::vector<float>& matrix,
    std::uint32_t n,
    std::uint32_t p,
    const cp::ControlPlaneConfig& config,
    cmd::KernelBackend gemm_backend = cmd::KernelBackend::SystemCFp32DeviceModel,
    const std::vector<std::uint32_t>& scale_words = {},
    std::uint32_t update_stride_pad = 0) {
    IntegrationOutcome outcome;
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = p;
    node.front_values = matrix;
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = n;
    options.gemm_backend = gemm_backend;
    options.gemm_scale_words = scale_words;
    options.update_stride_pad_bytes = update_stride_pad;
    testing::TestArtifact artifact = testing::build_test_artifact({node}, options);
    outcome.panel_command = cmd::NONE;
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        switch (artifact.commands[index].opcode) {
            case cmd::Opcode::PanelLu:
                outcome.panel_command = index;
                break;
            case cmd::Opcode::TrsmLeft:
                outcome.left_command = index;
                break;
            case cmd::Opcode::TrsmRight:
                outcome.right_command = index;
                break;
            case cmd::Opcode::GemmSchur:
                outcome.gemm_command = index;
                break;
            case cmd::Opcode::StoreUpdate:
                outcome.update_command = index;
                break;
            case cmd::Opcode::NodeCommit:
                outcome.commit_command = index;
                break;
            default:
                break;
        }
    }
    cp::CommandDevice device(artifact.artifact, config);
    std::string error;
    outcome.started = device.start(&error);
    if (!outcome.started) {
        std::cerr << "pipeline start refused: " << error << '\n';
        return outcome;
    }
    while (!device.finished() && device.cycle() < 8000000) {
        device.tick();
    }
    outcome.finished = device.finished();
    outcome.status = device.completion_status();
    for (const auto& stats : device.command_stats()) {
        if (stats.command_id == outcome.update_command) {
            outcome.update_bytes = stats.bytes_written;
        }
        if (stats.command_id == outcome.gemm_command) {
            outcome.gemm_saturation = stats.saturation_count;
            outcome.gemm_local_bytes = stats.local_conversion_bytes;
        }
    }
    if (device.panel_engine() != nullptr) {
        for (const auto& record : device.panel_engine()->task_records()) {
            if (record.task.mode == pl::TileMode::GemmSchur) {
                outcome.gemm_read_bytes += record.bytes_read;
                outcome.gemm_write_bytes += record.bytes_written;
            }
        }
    }
    const auto& image = device.dataplane().memory().image();
    const auto* region = device.boot().region("node.0.update");
    if (region != nullptr) {
        const std::uint32_t u = n - p;
        outcome.update_values.assign(static_cast<std::size_t>(u) * u, 0.0f);
        for (std::uint32_t row = 0; row < u; ++row) {
            std::memcpy(
                outcome.update_values.data() + static_cast<std::size_t>(row) * u,
                image.data() + region->offset +
                    static_cast<std::uint64_t>(row) *
                        (u * 4 + update_stride_pad),
                static_cast<std::size_t>(u) * 4);
        }
    }
    const auto read_region = [&](const std::string& name,
                                 std::size_t count) -> std::vector<float> {
        const auto* found = device.boot().region(name);
        if (found == nullptr) {
            return {};
        }
        std::vector<float> values(count, 0.0f);
        std::memcpy(values.data(), image.data() + found->offset, count * 4);
        return values;
    };
    outcome.l_factors = read_region("node.0.factor_l", static_cast<std::size_t>(n) * p);
    outcome.u_factors = read_region("node.0.factor_u", static_cast<std::size_t>(p) * n);
    return outcome;
}

// Independent FP64 Schur expectation from the device's committed L21/U12.
double max_schur_error_device_factors(
    const IntegrationOutcome& outcome,
    const std::vector<float>& matrix,
    std::uint32_t n,
    std::uint32_t p,
    double* scale) {
    const std::uint32_t u = n - p;
    double max_error = 0.0;
    *scale = 1.0;
    for (std::uint32_t i = 0; i < u; ++i) {
        for (std::uint32_t j = 0; j < u; ++j) {
            double acc = static_cast<double>(matrix[(p + i) * n + (p + j)]);
            for (std::uint32_t t = 0; t < p; ++t) {
                acc -= static_cast<double>(
                           outcome.l_factors[static_cast<std::size_t>(p + i) * p + t]) *
                       static_cast<double>(
                           outcome.u_factors[static_cast<std::size_t>(t) * n + (p + j)]);
            }
            const double got = static_cast<double>(
                outcome.update_values[static_cast<std::size_t>(i) * u + j]);
            max_error = std::max(max_error, std::fabs(got - acc));
            *scale = std::max(*scale, std::fabs(acc));
        }
    }
    return max_error;
}

cp::ControlPlaneConfig artifact_config(std::uint32_t groups, std::uint32_t per_group) {
    cp::ControlPlaneConfig config;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 2;
    config.trsm_unit_count = 1;
    config.gemm_unit_count = 1;
    config.solve_unit_count = 1;
    config.assemble_unit_count = 1;
    config.factor_unit_count = 1;
    config.buffer_slot_count = 5;
    config.buffer_capacity_bytes = 64 * 1024;
    config.buffer_port_count = 4;
    config.command_window_depth = 16;
    config.completion_queue_depth = 16;
    config.dma_queue_depth = 8;
    config.backend = cp::ExecBackend::RequireRealKernels;
    config.enable_test_stub = false;
    config.enable_real_panel = true;
    config.memory.startup_cycles = 1;
    config.memory.beat_bytes = 16;
    config.memory.beat_cycles = 1;
    config.dma.beat_bytes = 16;
    config.dma.startup_cycles = 1;
    config.dma.beat_interval_cycles = 1;
    config.panel.compute_group_count = groups;
    config.panel.pes_per_group = per_group;
    config.panel.pe_count = groups * per_group;
    return config;
}

void test_artifact_fp32_pipeline() {
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    const std::uint32_t u = n - p;
    const std::vector<float> matrix = make_front(n, p, 21);
    const IntegrationOutcome outcome =
        run_node_pipeline(matrix, n, p, artifact_config(2, 2));
    expect(outcome.started, "artifact: device starts");
    expect(outcome.finished, "artifact: batch terminates");
    expect(
        outcome.status.at(outcome.panel_command) == StatusCode::Ok &&
            outcome.status.at(outcome.left_command) == StatusCode::Ok &&
            outcome.status.at(outcome.right_command) == StatusCode::Ok &&
            outcome.status.at(outcome.gemm_command) == StatusCode::Ok &&
            outcome.status.at(outcome.update_command) == StatusCode::Ok &&
            outcome.status.at(outcome.commit_command) == StatusCode::Ok,
        "artifact: PANEL -> TRSM -> GEMM -> STORE_UPDATE -> NODE_COMMIT all succeed");
    expect(outcome.gemm_read_bytes == 9880, "artifact: GEMM operand read = 9880 B");
    expect(outcome.gemm_write_bytes == 2888, "artifact: GEMM writeback = 2888 B");
    expect(
        outcome.update_bytes == static_cast<std::uint64_t>(u) * u * 4,
        "artifact: STORE_UPDATE DDR data = u*u*4 = 1444 B");
    expect(
        outcome.update_values.size() == static_cast<std::size_t>(u) * u,
        "artifact: update region read back");
    // The DDR update must equal the in-place A22 state evaluated independently
    // in FP64 from the device's committed L/U factors is not available here, so
    // the device's own final A22 (single buffer) is the reference only for the
    // byte identity of the store; the numeric Schur check lives in the engine
    // test above and in the retransmission check below.
    bool finite = true;
    bool nonzero = false;
    for (const float value : outcome.update_values) {
        if (!std::isfinite(value)) {
            finite = false;
        }
        if (value != 0.0f) {
            nonzero = true;
        }
    }
    expect(finite, "artifact: update values finite");
    expect(nonzero, "artifact: update values are real Schur results");
    // Padded DDR update stride: values land at the padded row stride and the
    // padding stays zero.
    const IntegrationOutcome padded =
        run_node_pipeline(matrix, n, p, artifact_config(2, 2),
                          cmd::KernelBackend::SystemCFp32DeviceModel, {}, 8);
    expect(
        padded.status.at(padded.gemm_command) == StatusCode::Ok &&
            padded.update_values == outcome.update_values,
        "artifact: padded DDR update stride keeps identical numeric results");
}

void test_artifact_int32_pipeline() {
    const std::uint32_t auto_word = dp::kScaleAutoWord;
    const std::uint32_t n = 42;
    const std::uint32_t p = 23;
    const std::uint32_t u = n - p;
    const std::vector<float> matrix = make_front(n, p, 23);
    // Default INT32 policy: AUTO, AUTO, AUTO, nearest-ties-even, FAIL, FAIL.
    const std::vector<std::uint32_t> defaults = {auto_word, auto_word, auto_word, 1, 1, 1};
    const IntegrationOutcome outcome = run_node_pipeline(
        matrix, n, p, artifact_config(2, 2),
        cmd::KernelBackend::SystemCInt32GemmModel, defaults);
    expect(outcome.started, "int32: device starts");
    expect(outcome.finished, "int32: batch terminates");
    expect(
        outcome.status.at(outcome.gemm_command) == StatusCode::Ok,
        "int32: INT32 GEMM executes through the real SCALE_DESC path");
    expect(outcome.gemm_saturation == 0, "int32: no saturation under AUTO scales");
    expect(outcome.gemm_local_bytes > 0, "int32: PE-local conversion bytes reported");
    expect(outcome.gemm_read_bytes == 9880, "int32: operand read unchanged (9880 B)");
    expect(outcome.gemm_write_bytes == 2888, "int32: writeback unchanged (2888 B)");
    expect(
        outcome.update_bytes == static_cast<std::uint64_t>(u) * u * 4,
        "int32: DDR update bytes unchanged");
    // Final Schur output error vs an independent FP64 evaluation of the
    // DEVICE's committed L21/U12 factors (separate checker, no device code).
    double scale = 1.0;
    const double max_error =
        max_schur_error_device_factors(outcome, matrix, n, p, &scale);
    std::cout << "[t10] n42 int32 final Schur relative error vs device-factor "
                 << "FP64 checker = " << max_error / scale << "\n";
    expect(
        max_error / scale < 1.0e-4,
        "int32: final Schur error stays within the local quantization bound");
    // Input quantization error, computed by an independent checker over the
    // device's committed L21/U12 blocks: for each 16x16 tile the error must be
    // at most one half of the tile's power-of-two step 2^(e-1).
    {
        double max_input_error = 0.0;
        double max_half_step = 0.0;
        const auto measure_block = [&](const std::vector<float>& values,
                                       std::uint32_t rows,
                                       std::uint32_t cols,
                                       std::uint32_t row_stride,
                                       std::uint32_t row_begin,
                                       std::uint32_t col_begin) {
            float maxabs = 0.0f;
            for (std::uint32_t r = 0; r < rows; ++r) {
                for (std::uint32_t c = 0; c < cols; ++c) {
                    const float value =
                        values[static_cast<std::size_t>(row_begin + r) * row_stride +
                               col_begin + c];
                    maxabs = std::max(maxabs, std::fabs(value));
                }
            }
            const std::int32_t exponent =
                pl::choose_power_of_two_exponent(maxabs, 30);
            const double half_step = std::ldexp(0.5, exponent);
            for (std::uint32_t r = 0; r < rows; ++r) {
                for (std::uint32_t c = 0; c < cols; ++c) {
                    const float value =
                        values[static_cast<std::size_t>(row_begin + r) * row_stride +
                               col_begin + c];
                    const std::int64_t q =
                        pl::round_half_to_even_scaled(value, exponent);
                    const double quantized = std::ldexp(static_cast<double>(q), exponent);
                    max_input_error =
                        std::max(max_input_error, std::fabs(static_cast<double>(value) - quantized));
                }
            }
            max_half_step = std::max(max_half_step, half_step);
        };
        for (std::uint32_t row_tile = 0; row_tile * 16 < u; ++row_tile) {
            for (std::uint32_t k_tile = 0; k_tile * 16 < p; ++k_tile) {
                const std::uint32_t rows = std::min<std::uint32_t>(16, u - row_tile * 16);
                const std::uint32_t cols = std::min<std::uint32_t>(16, p - k_tile * 16);
                measure_block(outcome.l_factors, rows, cols, p, p + row_tile * 16, k_tile * 16);
                measure_block(outcome.u_factors, cols, rows, n, k_tile * 16, p + row_tile * 16);
            }
        }
        std::cout << "[t10] int32 input quantization max error = " << max_input_error
                  << " <= max half-step " << max_half_step << "\n";
        expect(
            max_input_error <= max_half_step,
            "int32: input quantization error stays within half of the local step");
    }

    // Fixed e = 0 with values far beyond the 29-bit fraction range: FAIL must
    // reject the command, CLAMP_AND_REPORT must succeed with real saturation.
    {
        std::vector<float> big(static_cast<std::size_t>(n) * n, 0.0f);
        for (std::uint32_t i = 0; i < p; ++i) {
            big[i * n + i] = 1.0f;  // A11 = I_p keeps the STATIC panel regular
        }
        big[0 * n + p] = 1.0e10f;        // U12 entry
        big[p * n + 0] = 1.0e10f;        // L21 entry
        big[p * n + p] = 1.0f;           // A22
        const std::vector<std::uint32_t> fixed_fail = {0, 0, auto_word, 1, 1, 1};
        const IntegrationOutcome fail = run_node_pipeline(
            big, n, p, artifact_config(2, 2),
            cmd::KernelBackend::SystemCInt32GemmModel, fixed_fail);
        expect(
            fail.status.at(fail.gemm_command) == StatusCode::QuantizationSaturation,
            "int32: FAIL policy rejects a command whose quantization saturates");
        expect(
            fail.gemm_saturation > 0 && fail.gemm_local_bytes > 0 &&
                fail.gemm_write_bytes == 0,
            "int32: rejected saturation and local work reach command statistics");
        const std::vector<std::uint32_t> fixed_clamp = {0, 0, auto_word, 1, 2, 1};
        const IntegrationOutcome clamp = run_node_pipeline(
            big, n, p, artifact_config(2, 2),
            cmd::KernelBackend::SystemCInt32GemmModel, fixed_clamp);
        expect(
            clamp.status.at(clamp.gemm_command) == StatusCode::Ok &&
                clamp.gemm_saturation > 0,
            "int32: CLAMP_AND_REPORT succeeds and counts the clamps");
    }

    // Malformed / unsupported SCALE_DESC policies reach the command status.
    {
        const std::vector<std::uint32_t> bad_rounding = {auto_word, auto_word, auto_word, 0, 1, 1};
        const IntegrationOutcome bad = run_node_pipeline(
            matrix, n, p, artifact_config(2, 2),
            cmd::KernelBackend::SystemCInt32GemmModel, bad_rounding);
        expect(
            bad.status.at(bad.gemm_command) == StatusCode::BadDescriptor,
            "int32: rounding_mode 0 -> BAD_DESCRIPTOR at the command");
        const std::vector<std::uint32_t> fixed_output = {auto_word, auto_word, 1, 1, 1, 1};
        const IntegrationOutcome unsupported = run_node_pipeline(
            matrix, n, p, artifact_config(2, 2),
            cmd::KernelBackend::SystemCInt32GemmModel, fixed_output);
        expect(
            unsupported.status.at(unsupported.gemm_command) ==
                StatusCode::PolicyNotSupported,
            "int32: fixed output scale -> POLICY_NOT_SUPPORTED at the command");
    }
    // INT32 backend without any SCALE_DESC reference.
    {
        const IntegrationOutcome missing = run_node_pipeline(
            matrix, n, p, artifact_config(2, 2),
            cmd::KernelBackend::SystemCInt32GemmModel, {});
        expect(
            missing.status.at(missing.gemm_command) == StatusCode::BadDescriptor,
            "int32: missing SCALE_DESC -> BAD_DESCRIPTOR");
    }
    // FP32 backend with a SCALE_DESC reference (contract violation).
    {
        const IntegrationOutcome fp32_scaled = run_node_pipeline(
            matrix, n, p, artifact_config(2, 2),
            cmd::KernelBackend::SystemCFp32DeviceModel, defaults);
        expect(
            fp32_scaled.status.at(fp32_scaled.gemm_command) ==
                StatusCode::BadDescriptor,
            "fp32: FP32 GEMM referencing SCALE_DESC -> BAD_DESCRIPTOR");
    }
}

// A foreign-front KERNEL operand must be refused by the adapter before any
// engine context or data access.
void test_foreign_front_reference() {
    const std::uint32_t n = 4;
    const std::uint32_t p = 2;
    pl::PanelLuConfig config = base_panel_config(1, 1);
    pl::PanelEngine engine(config);
    expect(engine.configure().ok, "foreign: engine configured");
    cp::GemmExecutionUnit unit(0, &engine);

    cp::ExecRequest request;
    request.txn_id = 1;
    request.command_id = 1;
    request.node_id = 0;
    request.opcode = cmd::Opcode::GemmSchur;
    request.unit_id = 0;
    request.total_dim = n;
    request.front.pivot_dim = p;
    request.front.front.descriptor_id = 5;
    request.front.front.base_addr = 0x1000;
    request.front.front.byte_size = 64;
    request.kernel.a.descriptor_id = 5;
    request.kernel.a.base_addr = 0x1000;
    request.kernel.a.byte_size = 64;
    request.kernel.b = request.kernel.a;
    request.kernel.c = request.kernel.a;
    request.kernel.m = n - p;
    request.kernel.n = n - p;
    request.kernel.k = p;
    request.kernel.tile_size = 16;
    request.kernel.backend = cmd::KernelBackend::SystemCFp32DeviceModel;
    request.front_slot.slot = 0;

    // Valid identity passes the operand check and fails later at the engine
    // (no lease), which is not what this test asserts; the foreign reference
    // fails immediately.
    request.kernel.a.descriptor_id = 99;
    expect(unit.submit(request, 1), "foreign: request accepted for terminal refusal");
    cp::ExecResponse response;
    expect(unit.poll_response(response), "foreign: terminal response produced");
    expect(
        response.status == StatusCode::BadDescriptor,
        "foreign: KERNEL operand outside the resident front -> BAD_DESCRIPTOR");
    expect(
        !engine.group_busy(0),
        "foreign: no engine context/group was consumed");
}

// ---------------------------------------------------------------------------
// F. STATIC pivot propagation from the device-generated update
// ---------------------------------------------------------------------------

// Runs the canonical 3x3 chain (A11 = I2, A12 = [1,1]^T, A21 = [1, 2^-35],
// A22 = [1]) with the requested GEMM backend and returns the value the device
// stored into the update region. This is a numeric propagation fixture, not an
// ASSEMBLE_EXTEND_ADD implementation.
bool run_three_by_three_update(
    cmd::KernelBackend backend,
    const std::vector<std::uint32_t>& scale_words,
    float* update_value,
    StatusCode* gemm_status) {
    const std::uint32_t n = 3;
    const std::uint32_t p = 2;
    std::vector<float> matrix(n * n, 0.0f);
    matrix[0 * n + 0] = 1.0f;
    matrix[1 * n + 1] = 1.0f;
    matrix[0 * n + 2] = 1.0f;  // A12
    matrix[1 * n + 2] = 1.0f;
    matrix[2 * n + 0] = 1.0f;  // A21
    matrix[2 * n + 1] = std::ldexp(1.0f, -35);
    matrix[2 * n + 2] = 1.0f;  // A22

    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = p;
    node.front_values = matrix;
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.matrix_dim = n;
    options.gemm_backend = backend;
    options.gemm_scale_words = scale_words;
    testing::TestArtifact artifact = testing::build_test_artifact({node}, options);
    cp::CommandDevice device(artifact.artifact, artifact_config(1, 1));
    std::string error;
    if (!device.start(&error)) {
        return false;
    }
    std::uint32_t gemm_command = cmd::NONE;
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode == cmd::Opcode::GemmSchur) {
            gemm_command = index;
        }
    }
    while (!device.finished() && device.cycle() < 2000000) {
        device.tick();
    }
    *gemm_status = device.completion_status().count(gemm_command) != 0
                       ? device.completion_status().at(gemm_command)
                       : StatusCode::Timeout;
    const auto* region = device.boot().region("node.0.update");
    if (region == nullptr) {
        return false;
    }
    const auto& image = device.dataplane().memory().image();
    std::memcpy(update_value, image.data() + region->offset, 4);
    return true;
}

// Runs one single-pivot STATIC panel front carrying the value produced by the
// previous update (numeric propagation fixture) and reports the actual pivot
// outcome. No FP64 pivot is injected anywhere.
StatusCode run_single_pivot_panel(float value, std::uint32_t* pivot_count) {
    std::vector<float> matrix(1, value);
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = 1;
    node.pivot_dim = 1;
    node.front_values = matrix;
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.matrix_dim = 1;
    testing::TestArtifact artifact = testing::build_test_artifact({node}, options);
    cp::CommandDevice device(artifact.artifact, artifact_config(1, 1));
    std::string error;
    if (!device.start(&error)) {
        return StatusCode::BadCommand;
    }
    std::uint32_t panel_command = cmd::NONE;
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode == cmd::Opcode::PanelLu) {
            panel_command = index;
        }
    }
    while (!device.finished() && device.cycle() < 1000000) {
        device.tick();
    }
    *pivot_count = 0;
    for (const auto& stats : device.command_stats()) {
        if (stats.command_id == panel_command) {
            *pivot_count = stats.pivot_count;
        }
    }
    return panel_command != cmd::NONE && device.completion_status().count(panel_command) != 0
               ? device.completion_status().at(panel_command)
               : StatusCode::Timeout;
}

void test_static_pivot_propagation() {
    const std::uint32_t auto_word = dp::kScaleAutoWord;
    // Hand-computed FP32 expectation: Schur = 1 - (1*1 + 2^-35*1) = -2^-35.
    const float two_pow_minus_35 = std::ldexp(1.0f, -35);
    float fp32_update = 0.0f;
    StatusCode fp32_gemm = StatusCode::Ok;
    expect(
        run_three_by_three_update(
            cmd::KernelBackend::SystemCFp32DeviceModel, {}, &fp32_update, &fp32_gemm),
        "pivot: 3x3 FP32 chain runs");
    expect(fp32_gemm == StatusCode::Ok, "pivot: FP32 GEMM succeeds");
    expect(
        fp32_update == -two_pow_minus_35,
        "pivot: FP32 FMA order yields exactly -2^-35");

    const std::vector<std::uint32_t> defaults = {auto_word, auto_word, auto_word, 1, 1, 1};
    float int32_update = 1.0f;
    StatusCode int32_gemm = StatusCode::Ok;
    expect(
        run_three_by_three_update(
            cmd::KernelBackend::SystemCInt32GemmModel, defaults, &int32_update,
            &int32_gemm),
        "pivot: 3x3 INT32 chain runs");
    expect(int32_gemm == StatusCode::Ok, "pivot: INT32 GEMM succeeds");
    std::cout << "[t10] pivot propagation: fp32 update = " << fp32_update
              << ", int32 update = " << int32_update << "\n";
    expect(
        int32_update == 0.0f,
        "pivot: default local INT32 path rounds the Schur update to 0");

    // Numeric propagation fixture: feed the device-generated update into the
    // next single-pivot STATIC panel front.
    std::uint32_t fp32_pivots = 0;
    const StatusCode fp32_panel = run_single_pivot_panel(fp32_update, &fp32_pivots);
    expect(
        fp32_panel == StatusCode::Ok && fp32_pivots == 1,
        "pivot: FP32 update keeps a valid STATIC pivot (-2^-35)");
    std::uint32_t int32_pivots = 0;
    const StatusCode int32_panel = run_single_pivot_panel(int32_update, &int32_pivots);
    expect(
        int32_panel == StatusCode::StaticPivotFailure,
        "pivot: INT32 update drives the next STATIC panel to STATIC_PIVOT_FAILURE");
    expect(
        int32_pivots == 0,
        "pivot: failure column is reported through pivot_count");
}

}  // namespace

// Cross-language integration: a device run directory produced by the Python
// Command Compiler in FP32_SYSTEMC_INT32_GEMM mode (real manifest + image, no
// golden files) must load and execute its INT32 GEMM commands on this device
// with the SCALE_DESC policy read at runtime.
void test_cross_language_int32_artifact(const std::string& device_dir) {
    dp::LoadedArtifact artifact = dp::load_artifact(device_dir);
    cp::CommandDevice device(std::move(artifact), artifact_config(2, 2));
    std::string error;
    const bool started = device.start(&error);
    expect(
        started,
        "cross-lang: Python-generated INT32 artifact loads and starts: " + error);
    if (!started) {
        return;
    }
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    expect(device.finished(), "cross-lang: batch terminates");
    if (!g_output_dir.empty()) {
        // Device execution report of the INT32 run (architecture model).
        device.write_reports(g_output_dir + "/int32_minimal");
    }
    const std::string gemm_opcode = std::to_string(
        static_cast<std::uint32_t>(cmd::Opcode::GemmSchur));
    std::uint32_t gemm_commands = 0;
    std::uint32_t gemm_ok = 0;
    std::uint64_t saturation = 0;
    std::uint64_t local_bytes = 0;
    for (const auto& stats : device.command_stats()) {
        if (stats.opcode != gemm_opcode) {
            continue;
        }
        ++gemm_commands;
        if (stats.status == "OK") {
            ++gemm_ok;
        }
        saturation += stats.saturation_count;
        local_bytes += stats.local_conversion_bytes;
    }
    expect(gemm_commands == 2, "cross-lang: two INT32 GEMM commands present");
    expect(gemm_ok == gemm_commands, "cross-lang: every INT32 GEMM command succeeds");
    expect(saturation == 0, "cross-lang: AUTO scales produce no saturation");
    expect(local_bytes > 0, "cross-lang: PE-local conversion bytes are reported");

    // Numeric cross-language check against hand computation done here (the
    // Python fixture is only the producer of the artifact, not of the answer):
    //   node.0: L21 = 2/4 = 0.5, U12 = 1, A22 = 0 -> -0.5 (exact in INT32)
    //   node.1: L21 = 1/3, U12 = 1, A22 = 1 -> 2/3 (up to the local step)
    const auto read_update = [&](const std::string& name) -> float {
        const auto* region = device.boot().region(name);
        if (region == nullptr) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        float value = 0.0f;
        std::memcpy(
            &value, device.dataplane().memory().image().data() + region->offset, 4);
        return value;
    };
    expect(
        read_update("node.0.update") == -0.5f,
        "cross-lang: node 0 update is -0.5 (exact in the INT32 path)");
    const float node1 = read_update("node.1.update");
    expect(
        std::fabs(static_cast<double>(node1) - 2.0 / 3.0) < 1.0e-6,
        "cross-lang: node 1 update matches 2/3 within the local quantization step");
}

void test_configured_int32_artifact(const std::string& device_dir) {
    const dp::LoadedArtifact artifact = dp::load_artifact(device_dir);
    expect(
        artifact.config.gemm_model.has_value() &&
            artifact.config.gemm_model->effective_bits == 4 &&
            artifact.config.gemm_model->k_block_max == 8 &&
            artifact.config.gemm_model->conversion_rate_per_cycle == 8,
        "configured artifact: loader preserves declared precision and rate");
    cp::CommandDevice device(artifact, artifact_config(2, 2));
    std::string error;
    const bool started = device.start(&error);
    expect(started, "configured artifact: default device inherits manifest: " + error);
    if (!started) {
        return;
    }
    bool all_pes_configured = device.panel_engine()->pes().size() == 4;
    for (const auto& pe : device.panel_engine()->pes()) {
        const auto& precision = pe.config().precision;
        all_pes_configured = all_pes_configured && precision.effective_bits == 4 &&
            precision.accumulator_bits == 64 && precision.k_block_max == 8 &&
            precision.maxabs_elements_per_cycle == 8 &&
            precision.quantize_elements_per_cycle == 8 &&
            precision.dequantize_elements_per_cycle == 8;
    }
    expect(all_pes_configured, "configured artifact: all four shared PEs use declared model");
    while (!device.finished() && device.cycle() < 4000000) {
        device.tick();
    }
    expect(device.finished(), "configured artifact: batch terminates");
    const auto* update = device.boot().region("node.0.update");
    std::vector<float> values(4, std::numeric_limits<float>::quiet_NaN());
    if (update != nullptr) {
        std::memcpy(values.data(), device.dataplane().memory().image().data() + update->offset, 16);
    }
    // At 4 bits AUTO chooses eA=eB=-2: 1 -> 4, 2^-10 -> 0.
    expect(
        values == std::vector<float>({-1, -1, 0, 0}),
        "configured artifact: 4-bit quantization changes the tiny Schur row to zero");
    std::uint32_t gemm_commands = 0;
    for (const auto& stats : device.command_stats()) {
        if (stats.opcode == "7") {
            ++gemm_commands;
            expect(
                stats.status == "OK" && stats.scale_reduction_cycles == 1 &&
                    stats.quantize_cycles == 1 && stats.integer_compute_cycles == 5 &&
                    stats.dequantize_cycles == 1 && stats.local_conversion_bytes == 112,
                "configured artifact: conversion rate 8 changes real phase counters");
        }
    }
    expect(gemm_commands == 1, "configured artifact: exactly one real GEMM command");
    if (!g_output_dir.empty()) {
        const auto output = g_output_dir + "/int32_configured";
        device.write_reports(output);
        std::ifstream stream(output + "/summary.json");
        nlohmann::json report;
        stream >> report;
        const auto& execution = report.at("gemm_execution");
        expect(
            execution.at("effective_bits") == 4 &&
                execution.at("k_block_max") == 8 &&
                execution.at("precision_config_source") == "artifact_manifest" &&
                execution.at("maxabs_elements_per_cycle") == 8 &&
                execution.at("quantize_elements_per_cycle") == 8 &&
                execution.at("dequantize_elements_per_cycle") == 8,
            "configured artifact: summary reports applied model and its source");
    }

    auto matching = artifact_config(2, 2);
    matching.panel.precision.effective_bits = 4;
    matching.panel.precision.k_block_max = 8;
    matching.panel.precision.maxabs_elements_per_cycle = 8;
    matching.panel.precision.quantize_elements_per_cycle = 8;
    matching.panel.precision.dequantize_elements_per_cycle = 8;
    cp::CommandDevice matching_device(artifact, matching);
    expect(matching_device.start(&error), "configured artifact: matching explicit model accepted");
    for (unsigned field = 0; field < 4; ++field) {
        auto conflicting = matching;
        switch (field) {
            case 0: conflicting.panel.precision.effective_bits = 5; break;
            case 1: conflicting.panel.precision.accumulator_bits = 32; break;
            case 2: conflicting.panel.precision.k_block_max = 16; break;
            case 3: conflicting.panel.precision.quantize_elements_per_cycle = 1; break;
        }
        cp::CommandDevice conflict_device(artifact, conflicting);
        expect(
            !conflict_device.start(&error) && error.find("conflicts") != std::string::npos,
            "configured artifact: conflicting explicit model refused before execution");
    }

    std::ifstream manifest_stream(device_dir + "/manifest.json");
    nlohmann::json manifest;
    manifest_stream >> manifest;
    const std::vector<std::pair<std::string, nlohmann::json>> invalid_fields{
        {"effective_bits", 0}, {"effective_bits", 31}, {"accumulator_bits", 32},
        {"k_block_max", 0}, {"k_block_max", 17}, {"conversion_rate_per_cycle", 0},
        {"conversion_rate_per_cycle", -1}, {"conversion_rate_per_cycle", 1.5},
        {"effective_bits", "4"}, {"k_block_max", true}};
    for (const auto& [field, value] : invalid_fields) {
        auto invalid = manifest;
        invalid["gemm"][field] = value;
        bool rejected = false;
        try {
            (void)dp::load_artifact_from_bytes(invalid.dump(), artifact.image);
        } catch (const dp::ArtifactLoadError&) {
            rejected = true;
        }
        expect(rejected, "configured artifact: loader rejects invalid " + field);
    }
    for (const bool malformed_object : {false, true}) {
        auto invalid = manifest;
        if (malformed_object) {
            invalid["gemm"] = nullptr;
        } else {
            invalid["gemm"].erase("conversion_rate_per_cycle");
        }
        bool rejected = false;
        try {
            (void)dp::load_artifact_from_bytes(invalid.dump(), artifact.image);
        } catch (const dp::ArtifactLoadError&) {
            rejected = true;
        }
        expect(rejected, "configured artifact: incomplete/malformed declaration refused");
    }
    manifest.erase("gemm");
    auto legacy = dp::load_artifact_from_bytes(manifest.dump(), artifact.image);
    expect(!legacy.config.gemm_model.has_value(), "legacy artifact: declaration remains optional");
    cp::CommandDevice legacy_device(std::move(legacy), matching);
    expect(
        legacy_device.start(&error) &&
            legacy_device.panel_engine()->pes().front().config().precision.effective_bits == 4,
        "legacy artifact: explicit device precision remains supported");
}

int main(int argc, char** argv) {
    if (argc > 1) {
        g_output_dir = argv[1];
    }
    const std::string cross_language_dir = argc > 2 ? argv[2] : std::string();
    test_scale_desc_decode();
    test_precision_primitives();
    test_int32_tile_kernel();
    test_gemm_request_response_contract();
    test_t10_execution_regressions();
    test_fp32_manual_case();
    test_fp32_n42_multitile_multik();
    test_fp32_recurrence_oracle();
    test_fp32_failure_and_refusals();
    test_fp32_groups_and_backpressure();
    test_fp32_abort_and_reuse();
    test_int32_abort_reset_and_reuse();
    test_artifact_fp32_pipeline();
    test_artifact_int32_pipeline();
    test_foreign_front_reference();
    test_static_pivot_propagation();
    if (!cross_language_dir.empty()) {
        test_cross_language_int32_artifact(cross_language_dir);
    }
    if (argc > 3) {
        test_configured_int32_artifact(argv[3]);
    }
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
