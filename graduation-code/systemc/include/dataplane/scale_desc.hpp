#pragma once

// SCALE_DESC body decoding and validation (schema v2, frozen by T10).
//
// This lives in the dataplane layer because it describes a descriptor record,
// not a compute-engine type: the runtime DescriptorReader parses and validates
// the policy before any execution context exists, and both the FP32 and INT32
// backends consume the same decoded value.
//
// SCALE_DESC word semantics (words 6..9 must be zero):
//   word 0: signed power-of-two exponent for A, AUTO = 0x80000000
//   word 1: signed power-of-two exponent for B, AUTO = 0x80000000
//   word 2: output scale, AUTO = 0x80000000 (derived from the actual eA+eB)
//   word 3: rounding mode, 1 = nearest, ties-to-even
//   word 4: saturation policy, 1 = FAIL, 2 = CLAMP_AND_REPORT
//   word 5: overflow policy, 1 = FAIL
// Fixed exponents are i32 two's-complement words in [-256, 256]. A malformed
// encoding (undefined zero enumeration, non-zero reserved word) is
// BAD_DESCRIPTOR; a well-formed but unsupported value is POLICY_NOT_SUPPORTED.

#include <cstdint>
#include <string>

#include "command_codec.hpp"

namespace hw::dataplane {

using StatusCode = hw::command::StatusCode;

inline constexpr std::uint32_t kScaleAutoWord = 0x80000000u;
inline constexpr std::int32_t kScaleExponentMin = -256;
inline constexpr std::int32_t kScaleExponentMax = 256;

enum class ScaleRoundingMode : std::uint32_t {
    // 0 is undefined/malformed, never a default.
    NearestTiesEven = 1,
};

enum class ScaleSaturationPolicy : std::uint32_t {
    Fail = 1,
    ClampAndReport = 2,
};

enum class ScaleOverflowPolicy : std::uint32_t {
    Fail = 1,
};

const char* scale_saturation_policy_name(ScaleSaturationPolicy policy);

// Decoded and validated SCALE_DESC policy.
struct GemmScalePolicy {
    std::uint32_t descriptor_id{hw::command::NONE};
    bool auto_exponent_a{true};
    bool auto_exponent_b{true};
    std::int32_t exponent_a{0};  // valid when !auto_exponent_a
    std::int32_t exponent_b{0};  // valid when !auto_exponent_b
    bool auto_output_scale{true};
    ScaleRoundingMode rounding{ScaleRoundingMode::NearestTiesEven};
    ScaleSaturationPolicy saturation{ScaleSaturationPolicy::Fail};
    ScaleOverflowPolicy overflow{ScaleOverflowPolicy::Fail};
};

// Decodes and validates the 10 SCALE_DESC body words. Returns:
//   Ok                 - policy is fully supported,
//   BadDescriptor      - malformed encoding (undefined enum value, reserved
//                        word non-zero),
//   PolicyNotSupported - well-formed but outside this version's support
//                        (fixed output scale, exponent out of [-256, 256]).
StatusCode decode_scale_policy(
    std::uint32_t descriptor_id,
    const std::uint32_t* body_words,
    GemmScalePolicy* out,
    std::string* reason);

}  // namespace hw::dataplane
