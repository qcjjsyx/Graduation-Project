import numpy as np

from src.quant.bfp_quant import dequantize, quantize_matrix


def test_quant_dequant_mse():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 48)).astype(np.float32) * 10.0
    qr = quantize_matrix(x)
    x_hat = dequantize(qr.q, qr.e)[: x.shape[0], : x.shape[1]]
    mse = np.mean((x - x_hat) ** 2)
    assert mse > 0
    assert mse < 5.0