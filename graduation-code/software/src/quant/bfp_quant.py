from __future__ import annotations

"""Quantization contract between the software pipeline and hardware.

Software-owned work:
- run symbolic analysis, build each node's front, and extract the local original
  matrix contribution A_local;
- quantize each A_local into S_format mantissa/exponent pairs;
- write those pairs and metadata into DDR-facing binary artifacts.

Hardware-owned work:
- assemble A_local sources and child updates into the node front;
- choose the final node-scale after assembly;
- execute integer panel LU/TRSM/GEMM and generate child update payloads.

The assembly helpers in this module are reference models for validation only.
They do not mean the software pipeline is responsible for doing integer LU or
producing numeric child updates.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.config import QuantConfig


@dataclass(frozen=True)
class QuantizationStats:
    total_elements: int
    clip_count: int
    sat_count: int
    max_abs: float
    clip_bound: float
    q_limit: int


@dataclass(frozen=True)
class QuantizedSource:
    """DDR S_format source consumed by hardware front assembly.

    Values are represented as mantissa * 2**exponent. The pipeline uses this
    for each node's local frontal contribution A_local.
    """

    mantissa: np.ndarray
    exponent: int
    shape: tuple[int, int]
    stats: QuantizationStats


@dataclass(frozen=True)
class AssemblyStats:
    source_count: int
    assembly_exponent: int
    node_exponent: int
    align_shift_max: int
    align_drop_count: int
    requant_sat_count: int


@dataclass(frozen=True)
class AssemblyResult:
    mantissa: np.ndarray
    exponent: int
    accumulator: np.ndarray
    stats: AssemblyStats


def quant_limit(effective_bits: int) -> int:
    if not 1 <= effective_bits <= 30:
        raise ValueError(f"effective_bits must be in [1, 30], got {effective_bits}")
    return (1 << effective_bits) - 1


def quantize_local_contribution(
    values: np.ndarray,
    config: QuantConfig | None = None,
) -> QuantizedSource:
    """Quantize one node's local frontal contribution into S_format."""
    config = config or QuantConfig()
    matrix = _as_2d_float(values)
    q_limit = quant_limit(config.effective_bits)
    max_abs = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    clip_bound = _clip_bound(matrix, config.clip_percentile)

    if clip_bound == 0.0:
        mantissa = np.zeros(matrix.shape, dtype=np.int32)
        exponent = 0
        clip_count = 0
        sat_count = 0
    else:
        exponent = _scale_exponent(clip_bound, q_limit)
        clipped = np.clip(matrix, -clip_bound, clip_bound)
        scaled = np.rint(clipped / np.ldexp(1.0, exponent))
        mantissa = np.clip(scaled, -q_limit, q_limit).astype(np.int32)
        clip_count = int(np.count_nonzero(np.abs(matrix) > clip_bound))
        sat_count = int(np.count_nonzero(np.abs(mantissa) == q_limit))

    return QuantizedSource(
        mantissa=mantissa,
        exponent=exponent,
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        stats=QuantizationStats(
            total_elements=int(matrix.size),
            clip_count=clip_count,
            sat_count=sat_count,
            max_abs=max_abs,
            clip_bound=clip_bound,
            q_limit=q_limit,
        ),
    )


def dequantize_source(source: QuantizedSource) -> np.ndarray:
    return source.mantissa.astype(np.float64) * np.ldexp(1.0, source.exponent)


def flatten_quantized_source(source: QuantizedSource) -> tuple[list[int], list[int]]:
    q_values = source.mantissa.reshape(-1).astype(np.int32).tolist()
    return q_values, [int(source.exponent)]


def assemble_sources(
    sources: Sequence[QuantizedSource],
    config: QuantConfig | None = None,
) -> AssemblyResult:
    """Reference model for hardware online assembly and node-scale requantization."""
    if not sources:
        raise ValueError("at least one source is required")
    config = config or QuantConfig()
    shape = sources[0].shape
    if any(source.shape != shape for source in sources):
        raise ValueError("all sources must have the same shape")

    assembly_exponent = max(source.exponent for source in sources)
    accumulator = np.zeros(shape, dtype=np.int64)
    align_shift_max = 0
    align_drop_count = 0

    for source in sources:
        shift = assembly_exponent - source.exponent
        align_shift_max = max(align_shift_max, shift)
        aligned = round_shift(source.mantissa.astype(np.int64), shift)
        align_drop_count += int(
            np.count_nonzero((source.mantissa != 0) & (aligned == 0))
        )
        accumulator += aligned

    node_mantissa, node_exponent, sat_count = requantize_accumulator(
        accumulator,
        assembly_exponent,
        config,
    )

    return AssemblyResult(
        mantissa=node_mantissa,
        exponent=node_exponent,
        accumulator=accumulator,
        stats=AssemblyStats(
            source_count=len(sources),
            assembly_exponent=assembly_exponent,
            node_exponent=node_exponent,
            align_shift_max=align_shift_max,
            align_drop_count=align_drop_count,
            requant_sat_count=sat_count,
        ),
    )


def requantize_accumulator(
    accumulator: np.ndarray,
    assembly_exponent: int,
    config: QuantConfig | None = None,
) -> tuple[np.ndarray, int, int]:
    config = config or QuantConfig()
    q_limit = quant_limit(config.effective_bits)
    max_abs = int(np.max(np.abs(accumulator))) if accumulator.size else 0
    if max_abs == 0:
        return np.zeros(accumulator.shape, dtype=np.int32), assembly_exponent, 0

    node_exponent = assembly_exponent + _scale_exponent(float(max_abs), q_limit)
    shift = node_exponent - assembly_exponent
    mantissa = round_shift(accumulator.astype(np.int64), shift)
    mantissa = np.clip(mantissa, -q_limit, q_limit).astype(np.int32)
    sat_count = int(np.count_nonzero(np.abs(mantissa) == q_limit))
    return mantissa, node_exponent, sat_count


def round_shift(values: np.ndarray, shift: int) -> np.ndarray:
    """Round signed integers while shifting right; negative shift means left shift."""
    arr = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return arr.copy()
    if shift < 0:
        return arr << abs(shift)

    offset = np.int64(1 << (shift - 1))
    positive = arr >= 0
    rounded_abs = (np.abs(arr) + offset) >> shift
    return np.where(positive, rounded_abs, -rounded_abs).astype(np.int64)


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"values must be a 2D matrix, got shape {matrix.shape}")
    return matrix


def _clip_bound(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    abs_values = np.abs(values)
    if percentile == 100:
        return float(np.max(abs_values))
    return float(np.percentile(abs_values, percentile))


def _scale_exponent(max_abs: float, q_limit: int) -> int:
    if max_abs <= 0:
        return 0
    return int(np.ceil(np.log2(max_abs / q_limit)))
