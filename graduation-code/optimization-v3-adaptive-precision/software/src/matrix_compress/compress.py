from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import scipy.io
import scipy.sparse as sp


class CompressionFormat(str, Enum):
    COO = "coo"
    CSR = "csr"
    BCSR = "bcsr"  # SciPy uses BSR for block CSR


@dataclass(frozen=True)
class CompressedMatrix:
    format: CompressionFormat
    matrix: sp.spmatrix
    blocksize: Tuple[int, int] | None = None


def read_mat_file(path: str, key: Optional[str] = None) -> sp.csr_matrix:
    """Load a .mat file and return a CSR sparse matrix.

    If key is None, the first 2D array-like entry is used.
    """
    data = scipy.io.loadmat(path)
    if key is not None:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in .mat file")
        value = data[key]
        return _to_csr(value)

    for value in data.values():
        if _is_matrix_like(value):
            return _to_csr(value)
    raise ValueError("No 2D matrix found in .mat file")


def compress_sparse(
    matrix: sp.spmatrix | np.ndarray,
    fmt: str | CompressionFormat = CompressionFormat.CSR,
    blocksize: Tuple[int, int] = (4, 4),
) -> CompressedMatrix:
    """Compress a matrix into COO/CSR/BCSR format."""
    fmt = _normalize_format(fmt)
    if fmt not in {CompressionFormat.COO, CompressionFormat.CSR, CompressionFormat.BCSR}:
        raise ValueError("fmt must be one of: 'coo', 'csr', 'bcsr'")

    base = _to_sparse(matrix)
    if fmt == CompressionFormat.COO:
        return CompressedMatrix(format=fmt, matrix=base.tocoo(), blocksize=None)
    if fmt == CompressionFormat.CSR:
        return CompressedMatrix(format=fmt, matrix=base.tocsr(), blocksize=None)

    # BCSR -> SciPy BSR
    bsr = sp.bsr_matrix(base, blocksize=blocksize)
    return CompressedMatrix(format=fmt, matrix=bsr, blocksize=blocksize)


def decompress_sparse(
    compressed: CompressedMatrix | sp.spmatrix,
    out: str | CompressionFormat = CompressionFormat.CSR,
) -> sp.spmatrix:
    """Decompress a COO/CSR/BCSR matrix into the requested sparse format."""
    out = _normalize_format(out)
    if out not in {CompressionFormat.COO, CompressionFormat.CSR}:
        raise ValueError("out must be 'coo' or 'csr'")

    matrix = compressed.matrix if isinstance(compressed, CompressedMatrix) else compressed
    if out == CompressionFormat.COO:
        return matrix.tocoo()
    return matrix.tocsr()


def compress_mat_file(
    path: str,
    out_path: str,
    fmt: str | CompressionFormat = CompressionFormat.CSR,
    blocksize: Tuple[int, int] = (4, 4),
    key: Optional[str] = None,
) -> CompressedMatrix:
    """Load a .mat file, compress it, and save to .npz."""
    matrix = read_mat_file(path, key=key)
    compressed = compress_sparse(matrix, fmt=fmt, blocksize=blocksize)
    sp.save_npz(out_path, compressed.matrix)
    return compressed


def load_compressed_file(path: str) -> sp.spmatrix:
    """Load a compressed sparse matrix saved via save_npz."""
    return sp.load_npz(path)


def _is_matrix_like(value) -> bool:
    if sp.isspmatrix(value):
        return value.ndim == 2
    if isinstance(value, np.ndarray):
        return value.ndim == 2
    return False


def _to_csr(value) -> sp.csr_matrix:
    if sp.isspmatrix(value):
        return value.tocsr()
    if isinstance(value, np.ndarray):
        return sp.csr_matrix(value)
    raise TypeError("Unsupported matrix type")


def _to_sparse(matrix: sp.spmatrix | np.ndarray) -> sp.spmatrix:
    if sp.isspmatrix(matrix):
        return matrix
    if isinstance(matrix, np.ndarray):
        return sp.csr_matrix(matrix)
    raise TypeError("matrix must be a numpy array or scipy sparse matrix")


def _normalize_format(fmt: str | CompressionFormat) -> CompressionFormat:
    if isinstance(fmt, CompressionFormat):
        return fmt
    try:
        return CompressionFormat(str(fmt).lower())
    except ValueError:
        raise ValueError("unsupported sparse compression format") from None
