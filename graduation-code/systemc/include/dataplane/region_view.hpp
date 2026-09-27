#pragma once

#include <cstdint>
#include <string>

#include "command_codec.hpp"

namespace hw::dataplane {

// Typed view of a REGION_DESC resolved at runtime through DescriptorReader.
// Data type/shape/stride come from descriptor fields, never from region names.
struct RegionView {
    std::uint32_t descriptor_id{0};
    std::uint64_t base_addr{0};
    std::uint64_t byte_size{0};
    std::uint32_t rows{0};
    std::uint32_t cols{0};
    std::uint64_t row_stride{0};
    hw::command::DataFormat format{hw::command::DataFormat::Fp32};
    hw::command::DataLayout layout{hw::command::DataLayout::RowMajor};
    std::uint32_t element_bytes{4};
    std::string name;  // diagnostic only (looked up by address)
};

inline const char* data_format_name(hw::command::DataFormat format) {
    switch (format) {
    case hw::command::DataFormat::Fp64: return "FP64";
    case hw::command::DataFormat::Fp32: return "FP32";
    case hw::command::DataFormat::Int32: return "INT32";
    case hw::command::DataFormat::LegacyInt32Exp: return "LEGACY_INT32_EXP";
    }
    return "UNKNOWN";
}

}  // namespace hw::dataplane
