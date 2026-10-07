#pragma once

// T10 unified internal GEMM request/response contract and SCALE_DESC policy
// decoding. These types are INTERNAL to the device compute engine: they are
// not a new external ABI record and do not change the schema v2 binary layout
// (Command 32B / Descriptor 64B / Completion 64B).
//
// The same contract serves both backends:
//   Fp32Device : C -= A * B with the formal FP32 FMA tile primitive,
//   Int32Gemm  : C -= dequantize(quantize(A, eA) * quantize(B, eB)).
//
// SCALE_DESC word semantics frozen by T10 (words 6..9 must be zero):
//   word 0: signed power-of-two exponent for A, AUTO = 0x80000000
//   word 1: signed power-of-two exponent for B, AUTO = 0x80000000
//   word 2: output scale, AUTO = 0x80000000 (derived from the actual eA+eB)
//   word 3: rounding mode, 1 = nearest, ties-to-even
//   word 4: saturation policy, 1 = FAIL, 2 = CLAMP_AND_REPORT
//   word 5: overflow policy, 1 = FAIL
// Fixed exponents are i32 two's-complement words in [-256, 256]. A malformed
// encoding (unknown zero enumeration, non-zero reserved word) is
// BAD_DESCRIPTOR; a well-formed but unsupported value is POLICY_NOT_SUPPORTED.

#include <cstdint>
#include <string>

#include "command_codec.hpp"
#include "dataplane/scale_desc.hpp"

namespace hw::panel_lu {

using StatusCode = hw::command::StatusCode;
// SCALE_DESC decoding lives in the dataplane layer (descriptor semantics);
// the compute engine consumes the validated policy.
using hw::dataplane::GemmScalePolicy;
using hw::dataplane::ScaleOverflowPolicy;
using hw::dataplane::ScaleRoundingMode;
using hw::dataplane::ScaleSaturationPolicy;

// Configurable precision/conversion parameters (T10 defaults). Validated by
// validate_precision_config(); unsupported or unsafe values are rejected
// explicitly, never silently ignored. This struct lives here (and not in the
// adapter header) so the engine configuration can embed it without creating a
// header cycle.
struct PrecisionConfig {
    // Effective signed quantization bits (INT32 storage). Default 30 ->
    // symmetric range [-(2^29-1), +(2^29-1)].
    std::uint32_t effective_bits{30};
    // Architectural accumulator width. Only INT64 (64) is supported.
    std::uint32_t accumulator_bits{64};
    // Maximum K handled by one local task. The T10 tile grid uses 16.
    std::uint32_t k_block_max{16};
    // Conversion pipeline rates (valid elements per cycle). Default: one
    // element per cycle per stage; the conversion stages are modeled
    // sequentially (one PE works on one compute/conversion task at a time).
    std::uint32_t maxabs_elements_per_cycle{1};
    std::uint32_t quantize_elements_per_cycle{1};
    std::uint32_t dequantize_elements_per_cycle{1};

    // qmax = 2^(effective_bits-1) - 1 (symmetric saturation bound).
    std::int64_t saturation_bound() const {
        return (static_cast<std::int64_t>(1) << (effective_bits - 1)) - 1;
    }
};

// Numeric backend selected for one GEMM command/tile.
enum class GemmBackend : std::uint8_t {
    Fp32Device = 0,  // SYSTEMC_FP32_DEVICE_MODEL (0x01 on the wire)
    Int32Gemm = 1,   // SYSTEMC_INT32_GEMM_MODEL (0x02 on the wire)
};

const char* gemm_backend_name(GemmBackend backend);

// Wire backend code -> internal backend. Unknown codes are rejected by the
// descriptor reader/execution unit before this point.
GemmBackend gemm_backend_from_wire(std::uint32_t wire_code);
std::uint32_t gemm_backend_to_wire(GemmBackend backend);

enum class GemmAccumulateMode : std::uint8_t {
    SubtractProduct = 0,  // C := C - A*B (the only mode this version supports)
};

const char* gemm_accumulate_mode_name(GemmAccumulateMode mode);

// One operand of a tile GEMM request: buffer reference, byte offset and row
// stride are independent of any host pointer; versions express the
// read-after-commit dependency the dispatcher enforces.
struct GemmOperand {
    std::uint64_t buffer_id{0};    // BufferHandle.slot
    std::uint64_t offset{0};       // byte offset relative to the front base
    std::uint32_t stride_bytes{0};
    std::uint32_t version{0};      // committed version required before use
    std::uint64_t version_key{0};  // tile version-table key
    bool valid{false};
};

// Where a failure was detected, for internal tracing (not the Completion ABI).
enum class GemmPhase : std::uint8_t {
    None = 0,
    Load = 1,
    ScaleReduction = 2,
    Quantize = 3,
    IntegerCompute = 4,
    DequantizeUpdate = 5,
    Writeback = 6,
};

const char* gemm_phase_name(GemmPhase phase);

// Unified internal tile GEMM request. One request describes exactly one
// 16x16-target tile contribution `C(i,j) -= A(i,s) * B(s,j)` with effective
// dimensions <= 16 (tail tiles carry the real M/N/K).
struct GemmReq {
    std::uint64_t txn_id{0};
    std::uint64_t task_id{0};
    std::uint64_t epoch{0};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    std::uint32_t descriptor_id{hw::command::NONE};  // KERNEL_DESC id
    std::uint32_t front_dim{0};                      // n
    std::uint32_t pivot_dim{0};                      // p
    std::uint32_t step{0};                           // K step s (pivot block)
    std::uint32_t tile_row{0};                       // A22 tile coordinates
    std::uint32_t tile_col{0};
    std::uint32_t m{0};
    std::uint32_t n{0};
    std::uint32_t k{0};
    GemmOperand a;  // M x K (A21 block)
    GemmOperand b;  // K x N (U12 block)
    GemmOperand c;  // M x N (A22 tile, in-place)
    GemmBackend backend{GemmBackend::Fp32Device};
    GemmScalePolicy scale;
    GemmAccumulateMode accumulate_mode{GemmAccumulateMode::SubtractProduct};
    // Front coordinates of the tile origin (p + tile_row*16, p + tile_col*16),
    // kept for failure localization and trace.
    std::uint32_t front_row{0};
    std::uint32_t front_col{0};
};

// Unified internal tile GEMM response. Cycles are absolute engine cycles;
// per-phase counters are additive model terms, not measured silicon time.
struct GemmRsp {
    std::uint64_t txn_id{0};
    std::uint64_t task_id{0};
    std::uint64_t epoch{0};
    std::uint32_t command_id{hw::command::NONE};
    std::uint32_t node_id{hw::command::NONE};
    StatusCode status{StatusCode::Ok};
    std::string reason;
    // Output visibility: only true after the result tile was actually written
    // back and its version committed.
    bool output_visible{false};
    std::uint32_t output_version{0};
    // Quantization statistics (INT32 backend; always zero for FP32).
    std::uint64_t saturation_count{0};
    std::uint64_t overflow_count{0};
    bool saturated{false};
    bool overflowed{false};
    // Scratchpad traffic (operand load + result writeback).
    std::uint64_t read_bytes{0};
    std::uint64_t write_bytes{0};
    // PE-local conversion traffic (never counted as DDR/scratchpad traffic).
    std::uint64_t local_conversion_bytes{0};
    // Per-phase model cycles.
    std::uint64_t scale_reduction_cycles{0};
    std::uint64_t quantize_cycles{0};
    std::uint64_t integer_compute_cycles{0};
    std::uint64_t dequantize_cycles{0};
    // Resolved exponents actually used (AUTO resolved to concrete values).
    std::int32_t exponent_a{0};
    std::int32_t exponent_b{0};
    // Failure localization: tile grid, K step, front coordinates, phase.
    GemmPhase failed_phase{GemmPhase::None};
    std::uint32_t failed_tile_row{0};
    std::uint32_t failed_tile_col{0};
    std::uint32_t failed_step{0};
    std::uint32_t failed_front_row{0};
    std::uint32_t failed_front_col{0};
};

}  // namespace hw::panel_lu
