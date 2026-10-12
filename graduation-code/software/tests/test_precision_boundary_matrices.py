"""Test-matrix families of the FP32 precision-boundary experiment.

Expectations are hand-computed (diagonal example) or checked with an
independent dense eigen/SVD computation, never with the generator itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tools.precision_boundary.matrices import (
    MatrixSpec,
    build_matrix,
    load_pattern,
    laplacian_on_pattern,
    shift_to_kappa,
    skew_on_pattern,
)


def test_shift_to_kappa_hand_example():
    # B = diag(1, 2, 4), target kappa 10: s = (10*1 - 4) / 9 = 2/3,
    # A = diag(1/3, 4/3, 10/3), kappa = (10/3) / (1/3) = 10.
    a = shift_to_kappa(sp.diags([1.0, 2.0, 4.0]).tocsr(), 10.0).toarray()
    np.testing.assert_allclose(np.diag(a), [1 / 3, 4 / 3, 10 / 3], rtol=1e-14)
    assert np.count_nonzero(a - np.diag(np.diag(a))) == 0


def test_laplacian_and_skew_respect_the_pattern():
    pattern = load_pattern("random", 40, 0.1, seed=3)
    rng = np.random.default_rng(3)
    b = laplacian_on_pattern(pattern, rng).toarray()
    k = skew_on_pattern(pattern, rng).toarray()
    allowed = pattern.toarray() != 0
    assert np.all(allowed[b != 0]) and np.all(allowed[k != 0])
    np.testing.assert_array_equal(b, b.T)
    off = b - np.diag(np.diag(b))
    assert np.all(off <= 0)  # M-matrix sign pattern
    np.testing.assert_allclose(b @ np.ones(40), 0.0, atol=1e-12)  # L*1 = 0
    assert np.linalg.eigvalsh(b)[0] >= -1e-12  # positive semidefinite
    np.testing.assert_array_equal(k, -k.T)
    assert np.linalg.norm(k, 2) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("kappa", [1e2, 1e8, 1e12])
def test_f1_hits_kappa_and_stays_spd(kappa):
    a = build_matrix(MatrixSpec("F1", "random", 60, 0.08, kappa, 0.0, 1)).toarray()
    w = np.linalg.eigvalsh(a)
    assert w[0] > 0
    assert w[-1] / w[0] == pytest.approx(kappa, rel=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f1_ill_conditioning_survives_row_scaling(seed):
    # Regression (2026-10-11): kappa must not be removable by row
    # equilibration.  Independent check: scale every row by 1/max|a_ij|.
    a = build_matrix(MatrixSpec("F1", "random", 256, 0.015, 1e8, 0.0, seed)).toarray()
    scaled = a / np.abs(a).max(axis=1)[:, None]
    assert np.linalg.cond(scaled) >= 1e8 / 10


def test_f2_symmetric_part_is_positive_definite():
    a = build_matrix(MatrixSpec("F2", "random", 60, 0.08, 1e6, 1.0, 2)).toarray()
    assert np.linalg.eigvalsh((a + a.T) / 2)[0] > 0
    assert not np.allclose(a, a.T)


@pytest.mark.parametrize("kappa", [1e3, 1e10])
def test_f3_hits_kappa(kappa):
    a = build_matrix(MatrixSpec("F3", "random", 60, 0.08, kappa, 0.3, 4)).toarray()
    s = np.linalg.svd(a, compute_uv=False)
    assert s[0] / s[-1] == pytest.approx(kappa, rel=0.02)


def test_case_ids_are_unique_across_a_plan():
    from tools.precision_boundary.__main__ import plan

    specs = plan("main") + plan("large") + plan("real") + plan("smoke")
    ids = [spec.case_id for spec in specs]
    assert len(ids) == len(set(ids))
