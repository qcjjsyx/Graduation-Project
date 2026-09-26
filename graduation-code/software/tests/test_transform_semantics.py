import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.config import (
    MatchingConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
    PivotPolicyConfig,
)
from src.matching import compute_row_matching, gather_rows
from src.pipeline import run_pipeline, run_symbolic_analysis
from src.symbolic.ordering import compute_ordering
from src.symbolic.pattern import symmetric_sparsity_pattern

ASYMMETRIC = np.array(
    [
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [5.0, 0.0, 2.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 7.0, 1.0, 5.0],
        [0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 0.0, 7.0],
    ]
)


def _run(tmp_path: Path, matching: str = "weighted_matching", policy: str = "STATIC"):
    matrix_path = tmp_path / "asym.mtx"
    sp_path = sp.csr_matrix(ASYMMETRIC)
    import scipy.io

    scipy.io.mmwrite(matrix_path, sp_path)
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=str(matrix_path), n=6),
            ordering=OrderingConfig(method="amd"),
            matching=MatchingConfig(method=matching),
            pivot=PivotPolicyConfig(policy=policy),
            reference_check=True,
            out_dir=tmp_path / f"out-{matching}-{policy}",
        )
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    return outputs, manifest


def _read_f64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def test_transform_identities_on_asymmetric_input(tmp_path: Path):
    assert not np.array_equal(ASYMMETRIC, ASYMMETRIC.T)
    outputs, manifest = _run(tmp_path)
    transform = manifest["preprocess"]["transform"]
    p_match = np.asarray(transform["row_perm_match"], dtype=np.int64)
    q = np.asarray(transform["ordering_perm"], dtype=np.int64)
    rhs_map = np.asarray(transform["rhs_row_map"], dtype=np.int64)
    assert p_match.tolist() == [2, 4, 0, 3, 1, 5]
    assert q.tolist() == [1, 2, 0, 4, 3, 5]
    assert not np.array_equal(q, np.argsort(q))

    exponents = np.fromfile(outputs.row_scale_exponents_path, dtype="<i2")
    scales = np.exp2(exponents.astype(np.float64))
    scaled = sp.diags(scales) @ sp.csr_matrix(ASYMMETRIC)
    matrix_b = np.asarray(gather_rows(scaled, p_match).todense())
    matrix_a_hat = matrix_b[np.ix_(q, q)]

    b_hat = _read_f64(outputs.rhs_reference_path)
    original_b = _read_f64(outputs.original_rhs_reference_path)
    np.testing.assert_allclose(b_hat, (scales * original_b)[p_match][q])
    np.testing.assert_allclose(rhs_map, p_match[q])

    fronts = _read_f64(outputs.reference_front_path)
    offsets = manifest["verification"]["reference_front_offsets"]
    for name, node in manifest["nodes"].items():
        index = np.asarray(node["front_indices"], dtype=np.int64)
        pivot_dim = node["pivot_dim"]
        expected = matrix_a_hat[np.ix_(index, index)].copy()
        expected[pivot_dim:, pivot_dim:] = 0.0
        size = expected.size
        offset = offsets[int(name)] // 8
        np.testing.assert_allclose(
            fronts[offset : offset + size].reshape(expected.shape), expected
        )

    x_true = _read_f64(outputs.solution_reference_path)
    x_hat = _read_f64(outputs.x_hat_reference_path)
    np.testing.assert_allclose(x_hat, x_true[q], atol=1e-8)

    recovered = np.empty_like(x_hat)
    recovered[q] = x_hat
    original_matrix = sp.csr_matrix(
        _read_f64(outputs.original_matrix_reference_path).reshape(6, 6)
    )
    assert np.linalg.norm(original_b - original_matrix @ recovered) < 1e-8


def test_wrong_permutation_directions_fail_the_checks(tmp_path: Path):
    outputs, manifest = _run(tmp_path)
    transform = manifest["preprocess"]["transform"]
    p_match = np.asarray(transform["row_perm_match"], dtype=np.int64)
    q = np.asarray(transform["ordering_perm"], dtype=np.int64)
    exponents = np.fromfile(outputs.row_scale_exponents_path, dtype="<i2")
    scales = np.exp2(exponents.astype(np.float64))
    original_b = _read_f64(outputs.original_rhs_reference_path)
    b_hat = _read_f64(outputs.rhs_reference_path)

    assert not np.allclose(b_hat, (scales * original_b)[q])
    assert not np.allclose(b_hat, (scales * original_b)[np.argsort(q)])
    assert not np.allclose(b_hat, (scales * original_b)[p_match])

    x_true = _read_f64(outputs.solution_reference_path)
    x_hat = _read_f64(outputs.x_hat_reference_path)
    assert not np.allclose(x_hat, x_true, atol=1e-8)
    assert not np.allclose(x_hat, x_true[p_match], atol=1e-8)

    original_matrix = sp.csr_matrix(
        _read_f64(outputs.original_matrix_reference_path).reshape(6, 6)
    )
    wrong_solution = x_true[p_match]
    assert np.linalg.norm(original_b - original_matrix @ wrong_solution) > 1e-2
    wrong_inverse = np.empty_like(x_hat)
    wrong_inverse[np.argsort(q)] = x_hat
    assert np.linalg.norm(original_b - original_matrix @ wrong_inverse) > 1e-2


def test_symbolic_envelope_is_recomputed_on_transformed_matrix(tmp_path: Path):
    outputs, manifest = _run(tmp_path)
    p_match = np.asarray(
        manifest["preprocess"]["transform"]["row_perm_match"], dtype=np.int64
    )
    q = np.asarray(manifest["preprocess"]["transform"]["ordering_perm"], dtype=np.int64)
    scaled = sp.csr_matrix(ASYMMETRIC)
    matrix_b = gather_rows(scaled, p_match)
    q_from_b = compute_ordering(symmetric_sparsity_pattern(matrix_b), "amd")
    q_from_a = compute_ordering(symmetric_sparsity_pattern(scaled), "amd")
    np.testing.assert_array_equal(q, q_from_b)
    assert not np.array_equal(q_from_a, q_from_b)

    symbolic = run_symbolic_analysis(
        matrix_b, PipelineConfig(ordering=OrderingConfig(method="amd"))
    )
    assert symbolic.permutation == q.tolist()


def test_matching_change_rebuilds_the_symbolic_structure(tmp_path: Path):
    outputs, manifest_match = _run(tmp_path, matching="weighted_matching")
    matrix = sp.csr_matrix(ASYMMETRIC)
    matching = compute_row_matching(matrix, MatchingConfig())
    assert not np.array_equal(matching.row_perm, np.arange(6))
    config = PipelineConfig(ordering=OrderingConfig(method="amd"))
    symbolic_plain = run_symbolic_analysis(matrix, config)
    symbolic_matched = run_symbolic_analysis(
        gather_rows(matrix, matching.row_perm), config
    )
    assert symbolic_plain.permutation != symbolic_matched.permutation
    assert (
        manifest_match["preprocess"]["transform"]["ordering_perm"]
        == symbolic_matched.permutation
    )
    assert (
        manifest_match["preprocess"]["transform"]["ordering_perm"]
        != symbolic_plain.permutation
    )
    assert manifest_match["preprocess"]["matching"]["full_matching"] is True
    assert outputs.residual_norm < 1e-8
