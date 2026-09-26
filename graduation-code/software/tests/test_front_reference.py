import numpy as np
import pytest

from src.config import PivotPolicyConfig
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange
from src.verify.front_reference import (
    PivotFailureError,
    PivotFailureKind,
    factor_front,
    solve_front_tree_reference,
)

STATIC = PivotPolicyConfig(policy="STATIC")
DYNAMIC = PivotPolicyConfig(policy="DYNAMIC_PANEL")


def single_node(front: np.ndarray, pivot_dim: int | None = None):
    size = front.shape[0]
    node = NodeCompileRecord(0, -1, NodeRange(0, size), tuple(range(size)))
    return [node], [front], [[]], [-1], pivot_dim or size


def test_static_succeeds_without_swaps_and_matches_lu_identity():
    front = np.array([[2.0, 1.0], [0.0, 3.0]])
    result = factor_front(front, 2, STATIC)
    assert result.p_local.tolist() == [0, 1]
    assert np.allclose(result.l @ result.u, front)


def test_static_fails_on_zero_diagonal_but_matching_preprocessing_solves_it():
    swap_only = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(PivotFailureError) as failure:
        factor_front(swap_only, 2, STATIC)
    assert failure.value.kind == PivotFailureKind.ZERO_PIVOT
    assert failure.value.column == 0

    preprocessed = swap_only[[1, 0], :]
    result = factor_front(preprocessed, 2, STATIC)
    assert result.p_local.tolist() == [0, 1]
    assert np.allclose(result.l @ result.u, preprocessed)


def test_nonzero_diagonal_still_needs_interchange_and_matrix_is_nonsingular():
    dense = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    assert abs(np.linalg.det(dense)) == pytest.approx(1.0)

    with pytest.raises(PivotFailureError) as failure:
        factor_front(dense, 3, STATIC, node_id=7)
    assert failure.value.kind == PivotFailureKind.ZERO_PIVOT
    assert failure.value.column == 1
    assert failure.value.node_id == 7

    dynamic = factor_front(dense, 3, DYNAMIC)
    assert dynamic.p_local.tolist() == [0, 2, 1]
    assert np.allclose(dynamic.l @ dynamic.u, dense[dynamic.p_local, :])

    rhs = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.solve(dense, rhs)
    nodes, fronts, maps, parent, _ = single_node(dense)
    solution = solve_front_tree_reference(
        nodes, fronts, maps, parent, rhs, DYNAMIC
    )
    assert np.allclose(solution.x_hat, expected)


def test_dynamic_panel_refuses_to_cross_the_pivot_block():
    front = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    with pytest.raises(PivotFailureError) as failure:
        factor_front(front, 2, DYNAMIC)
    assert failure.value.kind == PivotFailureKind.NO_CANDIDATE
    assert failure.value.column == 1

    dynamic_full = factor_front(front, 3, DYNAMIC)
    assert dynamic_full.p_local.tolist() == [0, 2, 1]


def test_threshold_scale_is_the_front_pre_elimination_max_abs():
    front = np.array([[100.0, 1.0], [1.0, 1e-4]])
    strict = PivotPolicyConfig(policy="STATIC", threshold_ratio=1e-3)
    with pytest.raises(PivotFailureError) as failure:
        factor_front(front, 2, strict)
    assert failure.value.kind == PivotFailureKind.BELOW_THRESHOLD
    assert failure.value.column == 1

    relaxed = PivotPolicyConfig(policy="STATIC", threshold_ratio=1e-6)
    assert factor_front(front, 2, relaxed).pivot_count == 2

    dynamic_strict = PivotPolicyConfig(policy="DYNAMIC_PANEL", threshold_ratio=1e-3)
    with pytest.raises(PivotFailureError) as failure:
        factor_front(front, 2, dynamic_strict)
    assert failure.value.kind == PivotFailureKind.NO_CANDIDATE

    with pytest.raises(PivotFailureError) as failure:
        factor_front(np.array([[1.0, 1.0], [1.0, 1.0]]), 2, strict)
    assert failure.value.kind == PivotFailureKind.ZERO_PIVOT
    assert failure.value.column == 1


def test_non_finite_inputs_and_elimination_are_rejected():
    with pytest.raises(PivotFailureError) as failure:
        factor_front(np.array([[np.nan, 1.0], [1.0, 1.0]]), 2, STATIC)
    assert failure.value.kind == PivotFailureKind.NON_FINITE

    overflow_multiplier = np.array([[1e-30, 1e308], [1e308, 1.0]])
    with pytest.raises(PivotFailureError) as failure:
        factor_front(overflow_multiplier, 2, STATIC)
    assert failure.value.kind == PivotFailureKind.NON_FINITE

    overflow_update = np.array([[-1e308, 1e308], [1e308, 1e308]])
    with pytest.raises(PivotFailureError) as failure:
        factor_front(overflow_update, 2, DYNAMIC)
    assert failure.value.kind == PivotFailureKind.NON_FINITE


def test_dynamic_tie_break_is_the_lowest_row():
    tie_low = np.array([[1.0, 5.0, 5.0], [5.0, 1.0, 2.0], [5.0, 3.0, 4.0]])
    result = factor_front(tie_low, 3, DYNAMIC)
    assert result.p_local[0] == 1

    tie_high = np.array([[5.0, 1.0, 2.0], [5.0, 3.0, 4.0], [1.0, 5.0, 5.0]])
    result = factor_front(tie_high, 3, DYNAMIC)
    assert result.p_local[0] == 0


def test_static_and_dynamic_may_produce_different_factors():
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    static = factor_front(dense, 2, STATIC)
    dynamic = factor_front(dense, 2, DYNAMIC)
    assert static.p_local.tolist() == [0, 1]
    assert dynamic.p_local.tolist() == [1, 0]
    assert not np.allclose(static.l, dynamic.l)
    assert not np.allclose(static.u, dynamic.u)
    assert np.allclose(static.l @ static.u, dense)
    assert np.allclose(dynamic.l @ dynamic.u, dense[dynamic.p_local, :])

    rhs = np.array([5.0, 11.0])
    expected = np.linalg.solve(dense, rhs)
    nodes, fronts, maps, parent, _ = single_node(dense)
    for policy in (STATIC, DYNAMIC):
        solution = solve_front_tree_reference(
            nodes, fronts, maps, parent, rhs, policy
        )
        assert np.allclose(solution.x_hat, expected)


def test_multi_root_forest_static_solve_and_original_equation_error():
    left = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 2.0], [0.5, 2.0, 5.0]])
    right = np.array([[2.0, 0.5], [0.5, 1.5]])
    nodes = [
        NodeCompileRecord(0, 1, NodeRange(0, 1), (0, 1, 2)),
        NodeCompileRecord(1, 2, NodeRange(1, 2), (1, 2)),
        NodeCompileRecord(2, -1, NodeRange(2, 3), (2,)),
        NodeCompileRecord(3, 4, NodeRange(3, 4), (3, 4)),
        NodeCompileRecord(4, -1, NodeRange(4, 5), (4,)),
    ]
    maps = [
        [],
        [MapTableEntry(0, [0, 1], [0, 1])],
        [MapTableEntry(1, [0], [0])],
        [],
        [MapTableEntry(3, [0], [0])],
    ]

    def local(block: np.ndarray, pivot_dim: int) -> np.ndarray:
        front = block.astype(np.float64).copy()
        front[pivot_dim:, pivot_dim:] = 0.0
        return front

    fronts = [
        local(left, 1),
        local(left[1:, 1:], 1),
        local(left[2:, 2:], 1),
        local(right, 1),
        local(right[1:, 1:], 1),
    ]
    parent = [1, 2, -1, 4, -1]
    rhs_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    solution = solve_front_tree_reference(
        nodes, fronts, maps, parent, rhs_hat, STATIC
    )
    dense = np.zeros((5, 5))
    dense[np.ix_([0, 1, 2], [0, 1, 2])] = left
    dense[np.ix_([3, 4], [3, 4])] = right
    expected = np.linalg.solve(dense, rhs_hat)
    assert np.allclose(solution.x_hat, expected)
    assert np.linalg.norm(rhs_hat - dense @ solution.x_hat) < 1e-12
    assert [factor.node_id for factor in solution.factors] == [0, 1, 2, 3, 4]


def test_local_p_vector_is_a_pivot_block_gather():
    dense = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 9.0]])
    result = factor_front(dense, 3, DYNAMIC)
    assert sorted(result.p_local.tolist()) == [0, 1, 2]
    assert np.allclose(result.l @ result.u, dense[result.p_local, :])
    static = factor_front(np.array([[2.0, 1.0], [1.0, 2.0]]), 2, STATIC)
    assert static.p_local.dtype == np.int32
    assert static.p_local.tolist() == [0, 1]


def test_host_threshold_matches_the_device_binary32_value():
    from src.command_codec import decode_pivot_threshold, encode_pivot_threshold

    config = PivotPolicyConfig(policy="STATIC", threshold_ratio=0.1)
    assert config.threshold_ratio == decode_pivot_threshold(
        encode_pivot_threshold(0.1)
    )
    front = np.array([[1.0, 0.5], [0.0, 0.1000000005]])
    with pytest.raises(PivotFailureError) as failure:
        factor_front(front, 2, config)
    assert failure.value.kind == PivotFailureKind.BELOW_THRESHOLD
    assert failure.value.column == 1
