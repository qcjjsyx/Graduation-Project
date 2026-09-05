from __future__ import annotations

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_matrix(path: str | None, n: int, density: float, seed: int) -> sp.csr_matrix:
    if path is None:
        return generate_random_spd(n=n, density=density, seed=seed)

    if path.endswith(".mat"):
        from src.matrix_compress.compress import read_mat_file

        matrix = read_mat_file(path)
    else:
        matrix = sp.csr_matrix(scipy.io.mmread(path))

    if matrix.shape[0] != matrix.shape[1]: # type: ignore
        raise ValueError("matrix must be square")
    matrix = matrix.tocsr().astype(np.float64)
    if matrix.shape[0] == 0:
        raise ValueError("matrix must not be empty")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("matrix contains non-finite values")
    return matrix


def load_vector(path: str, expected_size: int) -> np.ndarray:
    if path.endswith(".mat"):
        data = scipy.io.loadmat(path)
        candidates = [
            value
            for key, value in data.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        ]
        if not candidates:
            raise ValueError(f"no vector found in {path}")
        value = candidates[0]
    else:
        value = np.load(path)

    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("RHS must not be empty")
    if vector.size != expected_size:
        raise ValueError(
            f"RHS length {vector.size} does not match matrix dimension {expected_size}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("RHS contains non-finite values")
    return vector


def generate_random_spd(n: int, density: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    random = sp.random(n, n, density=density, format="csr", random_state=rng) # type: ignore
    matrix = random + random.T
    matrix = matrix + sp.eye(n, format="csr") * n
    return matrix.tocsr()
