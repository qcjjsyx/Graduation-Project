"""T12: FP64 accumulator-form tree solve reference (design doc §1.3).

The accelerator reference must agree with the established array-form
``solve_front_tree_reference`` on the same IR, and both must agree with a dense
solve of the global matrix assembled from the initial front contributions.
Expectations here are hand-computed or come from an independent dense solve,
never from the code under test alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import PivotPolicyConfig
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange
from src.verify.front_reference import (
    PivotFailureError,
    factor_front,
    solve_front_tree_accumulator,
    solve_front_tree_reference,
)


def _two_level_tree():
    """Child front (pivot {0}, update {1}) + root front (pivot {1})."""

    nodes = [
        NodeCompileRecord(0, 1, NodeRange(0, 1), (0, 1)),
        NodeCompileRecord(1, -1, NodeRange(1, 2), (1,)),
    ]
    maps = [[], [MapTableEntry(0, [0], [0])]]
    fronts = [
        np.array([[4.0, 1.0], [2.0, 0.0]]),
        np.array([[5.0]]),
    ]
    return nodes, maps, fronts, [1, -1]


def _chain_tree():
    nodes = [
        NodeCompileRecord(0, 1, NodeRange(0, 1), (0, 1, 2)),
        NodeCompileRecord(1, 2, NodeRange(1, 2), (1, 2)),
        NodeCompileRecord(2, -1, NodeRange(2, 3), (2,)),
    ]
    maps = [
        [],
        [MapTableEntry(0, [0, 1], [0, 1])],
        [MapTableEntry(1, [0], [0])],
    ]
    fronts = [
        np.array([[3.0, 0.5, 0.25], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[4.0, 1.0], [0.5, 0.0]]),
        np.array([[2.5]]),
    ]
    return nodes, maps, fronts, [1, 2, -1]


def _scatter_local_fronts(nodes, fronts):
    """Global matrix: every original element lives in exactly one local front
    (update/update blocks are zero in the local contributions)."""

    dimension = max(node.node_range.end for node in nodes)
    matrix = np.zeros((dimension, dimension))
    for node, front in zip(nodes, fronts):
        variables = np.asarray(node.front_indices, dtype=np.int64)
        matrix[np.ix_(variables, variables)] += front
    return matrix


def test_accumulator_matches_array_reference_and_hand_dense_solve():
    nodes, maps, fronts, parent = _two_level_tree()
    rhs = np.array([1.0, 3.0])
    config = PivotPolicyConfig()
    accumulator = solve_front_tree_accumulator(nodes, fronts, maps, parent, rhs, config)
    array_form = solve_front_tree_reference(nodes, fronts, maps, parent, rhs, config)
    assert np.array_equal(accumulator, array_form.x_hat)
    # Hand arithmetic: global M = [[4,1],[2,5]], det = 18:
    # x = [(1*5 - 1*3), (4*3 - 2*1)] / 18 = [2/18, 10/18].
    expected = np.array([2.0 / 18.0, 10.0 / 18.0])
    assert np.array_equal(accumulator, expected)
    assert np.allclose(accumulator, np.linalg.solve(_scatter_local_fronts(nodes, fronts), rhs))


def test_accumulator_matches_array_reference_on_chain_tree_and_dense_solve():
    nodes, maps, fronts, parent = _chain_tree()
    rhs = np.array([1.0, -2.0, 0.5])
    config = PivotPolicyConfig()
    accumulator = solve_front_tree_accumulator(nodes, fronts, maps, parent, rhs, config)
    array_form = solve_front_tree_reference(nodes, fronts, maps, parent, rhs, config)
    assert np.array_equal(accumulator, array_form.x_hat)
    matrix = _scatter_local_fronts(nodes, fronts)
    expected = np.linalg.solve(matrix, rhs)
    assert np.allclose(accumulator, expected, rtol=0.0, atol=1e-12)


def test_accumulator_matches_array_reference_with_pivot_swaps():
    """DYNAMIC_PANEL produces a non-identity P-vector; the accumulator must
    apply the gather direction exactly like the array reference."""

    nodes = [
        NodeCompileRecord(0, 1, NodeRange(0, 2), (0, 1, 2)),
        NodeCompileRecord(1, -1, NodeRange(2, 3), (2,)),
    ]
    maps = [[], [MapTableEntry(0, [0], [0])]]
    fronts = [
        np.array([[0.5, 2.0, 1.0], [3.0, 0.5, 2.0], [0.0, 0.0, 0.0]]),
        np.array([[1.5]]),
    ]
    rhs = np.array([1.0, 2.0, 3.0])
    config = PivotPolicyConfig(policy="DYNAMIC_PANEL")
    array_form = solve_front_tree_reference(nodes, fronts, maps, [1, -1], rhs, config)
    assert list(array_form.factors[0].p_local) != [0, 1]  # a swap really happened
    accumulator = solve_front_tree_accumulator(nodes, fronts, maps, [1, -1], rhs, config)
    assert np.array_equal(accumulator, array_form.x_hat)
    # Scatter of local fronts: front0 update/update block is zero, so entry
    # (2,2) comes only from the root's local front.
    matrix = _scatter_local_fronts(nodes, fronts)
    assert np.allclose(accumulator, np.linalg.solve(matrix, rhs), rtol=0.0, atol=1e-12)


def test_accumulator_rejects_zero_pivot_and_non_finite_rhs():
    # A single-node front with a zero diagonal is refused by the shared
    # factorization reference before any substitution runs.
    with pytest.raises(PivotFailureError):
        solve_front_tree_accumulator(
            [NodeCompileRecord(0, -1, NodeRange(0, 1), (0,))],
            [np.array([[0.0]])],
            [[]],
            [-1],
            np.array([1.0]),
            PivotPolicyConfig(),
        )
    nodes, maps, fronts, parent = _two_level_tree()
    with pytest.raises(ValueError):
        solve_front_tree_accumulator(
            nodes, fronts, maps, parent, np.array([1.0, np.inf]), PivotPolicyConfig()
        )


def test_accumulator_gathers_with_a_three_cycle_dynamic_pivot():
    """T12 revision: non-involutive P-vector through the accumulator reference.

    DYNAMIC_PANEL on F picks the column-0 pivot in row 1 (|4|) and then the
    column-1 pivot in original row 2 (|7.5| vs |1.75|), so the gather
    permutation is the 3-cycle [1, 2, 0], whose inverse [2, 0, 1] differs
    (gather and scatter are distinguishable).  numpy.linalg.solve on the
    unpermuted front is the independent reference.
    """

    front = np.array([[1.0, 2.0, 1.0], [4.0, 1.0, 0.0], [2.0, 8.0, 1.0]])
    rhs = np.array([1.0, 3.0, -4.0])  # F @ [1, -1, 2]
    config = PivotPolicyConfig(policy="DYNAMIC_PANEL")
    p_local = [int(v) for v in factor_front(front, 3, config).p_local]
    assert p_local == [1, 2, 0]
    assert any(p_local[p_local[i]] != i for i in range(3))

    nodes = [NodeCompileRecord(0, -1, NodeRange(0, 3), (0, 1, 2))]
    x_hat = solve_front_tree_accumulator(nodes, [front], [[]], [-1], rhs, config)
    expected = np.linalg.solve(front, rhs)
    np.testing.assert_allclose(expected, [1.0, -1.0, 2.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(x_hat, expected, rtol=1e-12, atol=1e-12)
