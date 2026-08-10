import numpy as np

from src.config import QuantConfig
from src.quant.bfp_quant import (
    assemble_sources,
    dequantize_source,
    flatten_quantized_source,
    quant_limit,
    quantize_local_contribution,
    round_shift,
)


def test_local_contribution_uses_single_source_exponent():
    values = np.array([[1.0, -2.0], [4.0, 8.0]], dtype=np.float32)
    config = QuantConfig(effective_bits=3)

    source = quantize_local_contribution(values, config)
    restored = dequantize_source(source)

    assert source.exponent == 1
    assert source.mantissa.dtype == np.int32
    assert source.mantissa.tolist() == [[0, -1], [2, 4]]
    assert source.stats.q_limit == quant_limit(config.effective_bits)
    assert source.stats.sat_count == 0
    assert restored.shape == values.shape


def test_flatten_source_writes_one_exponent_per_local_front():
    source = quantize_local_contribution(np.eye(3, dtype=np.float32))
    q_values, e_values = flatten_quantized_source(source)

    assert len(q_values) == 9
    assert e_values == [source.exponent]


def test_reference_assembly_aligns_sources_with_round_shift():
    large = quantize_local_contribution(
        np.array([[16.0, 0.0]], dtype=np.float32),
        QuantConfig(effective_bits=3),
    )
    small = quantize_local_contribution(
        np.array([[1.0, -1.0]], dtype=np.float32),
        QuantConfig(effective_bits=3),
    )

    assembled = assemble_sources([large, small], QuantConfig(effective_bits=3))

    assert assembled.stats.source_count == 2
    assert assembled.stats.assembly_exponent == max(large.exponent, small.exponent)
    assert assembled.stats.align_shift_max == large.exponent - small.exponent
    assert assembled.stats.align_drop_count >= 1
    assert assembled.mantissa.shape == large.mantissa.shape


def test_round_shift_is_symmetric_for_signed_values():
    values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int64)
    assert round_shift(values, 1).tolist() == [-2, -1, -1, 0, 1, 1, 2]
