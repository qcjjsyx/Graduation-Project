from pathlib import Path
import logging

import numpy as np

from src.quant.bfp_quant import dequantize, quantize_matrix
from src.matrix_compress.compress import read_mat_file

# 创建日志目录
log_dir = Path(__file__).parent / 'tests_log'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'test_quant.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


def test_quant_dequant_mse():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 48)).astype(np.float32) * 10.0
    qr = quantize_matrix(x)
    x_hat = dequantize(qr.q, qr.e)[: x.shape[0], : x.shape[1]]
    mse = np.mean((x - x_hat) ** 2)
    assert mse > 0
    assert mse < 5.0


def _example_mat_path() -> Path:
    return Path(__file__).resolve().parents[1] / "example" / "256X256JJ.mat"


def test_quant_dequant_on_mat():
    mat_path = _example_mat_path()
    a = read_mat_file(str(mat_path)).toarray().astype(np.float32)
    # print(a)
    qr = quantize_matrix(a)
    logger.info(f"qr: {qr}")
    
    a_hat = dequantize(qr.q, qr.e)[: a.shape[0], : a.shape[1]]
    logger.info(f"a_hat: {a_hat}")
    assert a_hat.shape == a.shape
    assert np.isfinite(a_hat).all()
