from __future__ import annotations

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from src.matrix_io import load_matrix, load_vector, read_mat_file


def test_read_mat_file_sparse_and_dense_roundtrip(tmp_path):
    dense = np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
    sparse_path = tmp_path / "sparse.mat"
    dense_path = tmp_path / "dense.mat"
    scipy.io.savemat(sparse_path, {"A": sp.csr_matrix(dense)})
    scipy.io.savemat(dense_path, {"A": dense})

    for path in (sparse_path, dense_path):
        matrix = read_mat_file(str(path))
        assert sp.isspmatrix_csr(matrix)
        assert np.array_equal(matrix.toarray(), dense)
        assert np.array_equal(read_mat_file(str(path), key="A").toarray(), dense)


def test_read_mat_file_rejects_missing_key_and_empty_file(tmp_path):
    path = tmp_path / "vector_only.mat"
    scipy.io.savemat(path, {"name": "x"})
    with pytest.raises(ValueError):
        read_mat_file(str(path))
    with pytest.raises(KeyError):
        read_mat_file(str(path), key="A")


def test_load_matrix_and_vector_from_mat(tmp_path):
    dense = np.array([[2.0, 1.0], [0.0, 3.0]])
    matrix_path = tmp_path / "a.mat"
    rhs_path = tmp_path / "b.mat"
    scipy.io.savemat(matrix_path, {"A": sp.csr_matrix(dense)})
    scipy.io.savemat(rhs_path, {"b": np.array([1.0, 2.0])})

    matrix = load_matrix(str(matrix_path), n=0, density=0.1, seed=0)
    assert np.array_equal(matrix.toarray(), dense)
    assert np.array_equal(load_vector(str(rhs_path), 2), [1.0, 2.0])
    with pytest.raises(ValueError):
        load_vector(str(rhs_path), 3)


def test_load_matrix_rejects_non_square(tmp_path):
    path = tmp_path / "rect.mat"
    scipy.io.savemat(path, {"A": np.ones((2, 3))})
    with pytest.raises(ValueError, match="square"):
        load_matrix(str(path), n=0, density=0.1, seed=0)


def test_example_matrix_loads_as_square_csr():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "example" / "256X256JJ.mat"
    matrix = load_matrix(str(path), n=0, density=0.1, seed=0)
    assert sp.isspmatrix_csr(matrix)
    assert matrix.shape == (256, 256)
