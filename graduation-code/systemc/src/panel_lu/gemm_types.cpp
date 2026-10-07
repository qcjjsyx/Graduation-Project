#include "panel_lu/gemm_types.hpp"

namespace hw::panel_lu {

const char* gemm_backend_name(GemmBackend backend) {
    switch (backend) {
        case GemmBackend::Fp32Device:
            return "fp32_device";
        case GemmBackend::Int32Gemm:
            return "int32_gemm";
    }
    return "unknown";
}

GemmBackend gemm_backend_from_wire(std::uint32_t wire_code) {
    return wire_code == static_cast<std::uint32_t>(hw::command::KernelBackend::SystemCInt32GemmModel)
               ? GemmBackend::Int32Gemm
               : GemmBackend::Fp32Device;
}

std::uint32_t gemm_backend_to_wire(GemmBackend backend) {
    return backend == GemmBackend::Int32Gemm
               ? static_cast<std::uint32_t>(hw::command::KernelBackend::SystemCInt32GemmModel)
               : static_cast<std::uint32_t>(hw::command::KernelBackend::SystemCFp32DeviceModel);
}

const char* gemm_accumulate_mode_name(GemmAccumulateMode mode) {
    switch (mode) {
        case GemmAccumulateMode::SubtractProduct:
            return "C_minus_A_times_B";
    }
    return "unknown";
}

const char* gemm_phase_name(GemmPhase phase) {
    switch (phase) {
        case GemmPhase::None:
            return "none";
        case GemmPhase::Load:
            return "load";
        case GemmPhase::ScaleReduction:
            return "scale_reduction";
        case GemmPhase::Quantize:
            return "quantize";
        case GemmPhase::IntegerCompute:
            return "integer_compute";
        case GemmPhase::DequantizeUpdate:
            return "dequantize_update";
        case GemmPhase::Writeback:
            return "writeback";
    }
    return "unknown";
}

}  // namespace hw::panel_lu
