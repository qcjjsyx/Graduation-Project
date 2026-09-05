import numpy as np
import scipy.sparse as sp

from src.symbolic.ordering import compute_ordering


def _assert_permutation(perm, n: int):
    assert perm.dtype == np.int32
    assert sorted(perm.tolist()) == list(range(n))


def test_ordering_identity():
    a = sp.eye(5, format="csr")
    perm = compute_ordering(a, "identity")
    assert perm.tolist() == [0, 1, 2, 3, 4]


def test_ordering_amd_returns_valid_permutation():
    a = sp.csr_matrix(
        [
            [4, 1, 0, 0],
            [1, 4, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 4],
        ],
        dtype=np.float64,
    )
    perm = compute_ordering(a, "amd")
    _assert_permutation(perm, 4)


def test_ordering_rejects_unknown_method():
    a = sp.eye(3, format="csr")
    try:
        compute_ordering(a, "nested_dissection")
    except ValueError as exc:
        assert "unknown ordering method" in str(exc)
    else:
        raise AssertionError("unknown ordering method should be rejected")
