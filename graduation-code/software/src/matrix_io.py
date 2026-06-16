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

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix.tocsr()


def generate_random_spd(n: int, density: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    random = sp.random(n, n, density=density, format="csr", random_state=rng)
    matrix = random + random.T
    matrix = matrix + sp.eye(n, format="csr") * n
    return matrix.tocsr()
