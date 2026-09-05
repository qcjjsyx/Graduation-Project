import numpy as np
import scipy.sparse as sp

from src.config import EquilibrationConfig
from src.equilibration import equilibrate_system


def test_pow2_row_equilibration_preserves_solution_and_sparsity():
    matrix = sp.csr_matrix(
        np.array(
            [
                [1024.0, -256.0, 0.0],
                [0.125, 0.5, -0.25],
                [0.0, 4.0, 8.0],
            ]
        )
    )
    solution = np.array([1.5, -2.0, 0.25])
    rhs = np.asarray(matrix @ solution).reshape(-1)

    result = equilibrate_system(
        matrix,
        rhs,
        EquilibrationConfig(mode="pow2-row"),
    )

    assert np.array_equal(
        result.matrix.toarray() != 0.0,
        matrix.toarray() != 0.0,
    )
    assert np.allclose(result.matrix @ solution, result.rhs)
    assert np.allclose(
        result.rhs,
        np.exp2(result.row_scale_exponents.astype(np.float64)) * rhs,
    )
    row_max = np.asarray(np.abs(result.matrix).max(axis=1).toarray()).reshape(-1)
    assert np.all(row_max >= 2.0**-0.5)
    assert np.all(row_max <= 2.0**0.5)


def test_none_equilibration_is_identity():
    matrix = sp.eye(4, format="csr") * 3.0
    rhs = np.arange(4, dtype=np.float64)
    result = equilibrate_system(
        matrix,
        rhs,
        EquilibrationConfig(mode="none"),
    )
    assert np.array_equal(result.matrix.toarray(), matrix.toarray())
    assert np.array_equal(result.rhs, rhs)
    assert np.array_equal(result.row_scale_exponents, np.zeros(4, dtype=np.int16))
