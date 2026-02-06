from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.matrix_compress.compress import (
    CompressionFormat,
    compress_mat_file,
    compress_sparse,
    decompress_sparse,
    load_compressed_file,
    read_mat_file,
)


def _example_mat_path() -> Path:
    return Path(__file__).resolve().parents[1] / "example" / "256X256JJ.mat"


def test_read_mat_file_loads_csr():
    mat_path = _example_mat_path()
    a = read_mat_file(str(mat_path))
    print(a)
    assert sp.isspmatrix_csr(a)
    assert a.shape == (256, 256)


def test_compress_decompress_roundtrip():
    mat_path = _example_mat_path()
    a = read_mat_file(str(mat_path))

    for fmt in (CompressionFormat.COO, CompressionFormat.CSR, CompressionFormat.BCSR):
        compressed = compress_sparse(a, fmt=fmt, blocksize=(2, 2))
        restored = decompress_sparse(compressed, out=CompressionFormat.CSR)

        diff = (a - restored).tocoo()
        assert diff.shape == a.shape
        assert np.allclose(diff.data, 0.0)


def test_compress_mat_file_and_reload(tmp_path: Path):
    mat_path = _example_mat_path()
    out_path = tmp_path / "compressed.npz"

    compressed = compress_mat_file(
        str(mat_path),
        str(out_path),
        fmt=CompressionFormat.CSR,
    )
    assert out_path.exists()

    loaded = load_compressed_file(str(out_path))
    restored = decompress_sparse(loaded, out=CompressionFormat.CSR)
    diff = (compressed.matrix.tocsr() - restored).tocoo() # type: ignore
    assert np.allclose(diff.data, 0.0)
