#include "dataplane/scale_desc.hpp"

namespace hw::dataplane {

const char* scale_saturation_policy_name(ScaleSaturationPolicy policy) {
    switch (policy) {
        case ScaleSaturationPolicy::Fail:
            return "FAIL";
        case ScaleSaturationPolicy::ClampAndReport:
            return "CLAMP_AND_REPORT";
    }
    return "unknown";
}

StatusCode decode_scale_policy(
    std::uint32_t descriptor_id,
    const std::uint32_t* body_words,
    GemmScalePolicy* out,
    std::string* reason) {
    if (body_words == nullptr || out == nullptr) {
        if (reason != nullptr) {
            *reason = "SCALE_DESC body is missing";
        }
        return StatusCode::BadDescriptor;
    }
    for (std::uint32_t word = 6; word < 10; ++word) {
        if (body_words[word] != 0) {
            if (reason != nullptr) {
                *reason = "SCALE_DESC reserved word " + std::to_string(word) +
                          " must be zero";
            }
            return StatusCode::BadDescriptor;
        }
    }
    GemmScalePolicy policy;
    policy.descriptor_id = descriptor_id;

    const auto decode_exponent = [&](std::uint32_t word,
                                     bool* is_auto,
                                     std::int32_t* value) -> StatusCode {
        if (word == kScaleAutoWord) {
            *is_auto = true;
            *value = 0;
            return StatusCode::Ok;
        }
        const std::int32_t decoded = static_cast<std::int32_t>(word);
        if (decoded < kScaleExponentMin || decoded > kScaleExponentMax) {
            if (reason != nullptr) {
                *reason =
                    "fixed SCALE_DESC exponent outside the supported range "
                    "[-256, 256]";
            }
            return StatusCode::PolicyNotSupported;
        }
        *is_auto = false;
        *value = decoded;
        return StatusCode::Ok;
    };
    StatusCode status =
        decode_exponent(body_words[0], &policy.auto_exponent_a, &policy.exponent_a);
    if (status != StatusCode::Ok) {
        return status;
    }
    status =
        decode_exponent(body_words[1], &policy.auto_exponent_b, &policy.exponent_b);
    if (status != StatusCode::Ok) {
        return status;
    }

    if (body_words[2] == kScaleAutoWord) {
        policy.auto_output_scale = true;
    } else {
        if (reason != nullptr) {
            *reason =
                "fixed output_scale is defined but not supported in this "
                "version; use AUTO";
        }
        return StatusCode::PolicyNotSupported;
    }

    switch (body_words[3]) {
        case 0:
            if (reason != nullptr) {
                *reason = "SCALE_DESC rounding_mode 0 is undefined";
            }
            return StatusCode::BadDescriptor;
        case static_cast<std::uint32_t>(ScaleRoundingMode::NearestTiesEven):
            policy.rounding = ScaleRoundingMode::NearestTiesEven;
            break;
        default:
            if (reason != nullptr) {
                *reason =
                    "SCALE_DESC rounding_mode is defined but not supported; "
                    "only 1 = nearest ties-to-even";
            }
            return StatusCode::PolicyNotSupported;
    }

    switch (body_words[4]) {
        case 0:
            if (reason != nullptr) {
                *reason = "SCALE_DESC saturation_policy 0 is undefined";
            }
            return StatusCode::BadDescriptor;
        case static_cast<std::uint32_t>(ScaleSaturationPolicy::Fail):
            policy.saturation = ScaleSaturationPolicy::Fail;
            break;
        case static_cast<std::uint32_t>(ScaleSaturationPolicy::ClampAndReport):
            policy.saturation = ScaleSaturationPolicy::ClampAndReport;
            break;
        default:
            if (reason != nullptr) {
                *reason = "SCALE_DESC saturation_policy is unknown";
            }
            return StatusCode::PolicyNotSupported;
    }

    switch (body_words[5]) {
        case 0:
            if (reason != nullptr) {
                *reason = "SCALE_DESC overflow_policy 0 is undefined";
            }
            return StatusCode::BadDescriptor;
        case static_cast<std::uint32_t>(ScaleOverflowPolicy::Fail):
            policy.overflow = ScaleOverflowPolicy::Fail;
            break;
        default:
            if (reason != nullptr) {
                *reason =
                    "SCALE_DESC overflow_policy is defined but not supported; "
                    "only 1 = FAIL";
            }
            return StatusCode::PolicyNotSupported;
    }

    *out = policy;
    return StatusCode::Ok;
}

}  // namespace hw::dataplane
