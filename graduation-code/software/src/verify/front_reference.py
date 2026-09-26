"""FP64 host-side front-tree reference with explicit pivot policy.

This module is a *checker*, not a device-data source: nothing it computes is
written into the device memory image.  Its jobs are

* implement the same pivot-policy semantics that the Command/Descriptor
  contract declares (``STATIC`` and ``DYNAMIC_PANEL``),
* preflight the compiled fixed pivot order at compile time so a hopeless
  static image is rejected with a diagnostic instead of being emitted,
* produce the recovered solution used for residual / backward-error checks
  against the *original* ``A x = b``.

The independent C++ FP64 reference (``graduation-code/systemc/reference``)
remains the primary reference implementation; this module mirrors its policy
semantics for host-side use.

Pivot policy semantics
----------------------
``STATIC``
    Eliminate in the fixed post-ordering diagonal order.  No candidate search,
    no row interchange.  The actual pivot of every elimination step is checked
    against zero, the relative threshold, and non-finite values.

``DYNAMIC_PANEL``
    At step ``k`` the candidate rows are ``k .. pivot_dim-1``; the pivot is the
    candidate with the largest absolute value, ties resolved by the lowest row
    index.  Update rows (``row >= pivot_dim``) are never promoted to pivots and
    never move.

Both policies share one threshold rule: a pivot is rejected when
``|pivot| <= threshold_ratio * scale`` where ``scale`` is the max absolute
value of the assembled front *before* elimination of that front.  Zero is
always rejected (any ratio).  A failed fixed elimination order is reported as
a numeric failure of that order; it is not a claim that the matrix is
singular.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from src.config import PivotPolicyConfig
from src.dataStruct import MapTableEntry, NodeCompileRecord
from src.scheduler.task_queue import sibling_friendly_order


class PivotFailureKind(str, Enum):
    ZERO_PIVOT = "zero_pivot"
    BELOW_THRESHOLD = "below_threshold"
    NON_FINITE = "non_finite"
    NO_CANDIDATE = "no_candidate"


class PivotFailureError(ValueError):
    """Numeric pivot failure of a factorization step."""

    def __init__(
        self,
        kind: PivotFailureKind,
        *,
        node_id: int,
        column: int,
        detail: str = "",
    ) -> None:
        self.kind = kind
        self.node_id = node_id
        self.column = column
        message = (
            f"pivot failure ({kind.value}) at node {node_id}, "
            f"pivot column {column}"
        )
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class FrontFactorization:
    """Factorization of one assembled front."""

    node_id: int
    variables: tuple[int, ...]
    pivot_dim: int
    l: np.ndarray
    u: np.ndarray
    # Gather direction inside the pivot block: pivot position ``i`` is sourced
    # from original pivot-block row ``p_local[i]``.  ``STATIC`` writes identity.
    p_local: np.ndarray
    schur: np.ndarray
    pivot_count: int
    scale: float


@dataclass(frozen=True)
class FrontTreeSolution:
    factors: tuple[FrontFactorization, ...]
    x_hat: np.ndarray
    order: tuple[int, ...]


def factor_front(
    front: np.ndarray,
    pivot_dim: int,
    config: PivotPolicyConfig,
    *,
    node_id: int = 0,
) -> FrontFactorization:
    """Factor one assembled dense front under the declared pivot policy."""

    work = np.array(front, dtype=np.float64, copy=True)
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError("front must be a square dense array")
    total_dim = work.shape[0]
    if not 0 < pivot_dim <= total_dim:
        raise ValueError("pivot_dim must be in [1, total_dim]")
    if not np.all(np.isfinite(work)):
        raise PivotFailureError(
            PivotFailureKind.NON_FINITE,
            node_id=node_id,
            column=0,
            detail="assembled front contains non-finite values",
        )

    scale = float(np.max(np.abs(work))) if work.size else 0.0
    threshold = config.threshold_ratio * scale
    p_local = np.arange(pivot_dim, dtype=np.int32)

    for column in range(pivot_dim):
        pivot_row = column
        if config.policy == "STATIC":
            pivot = work[column, column]
            if not np.isfinite(pivot):
                raise PivotFailureError(
                    PivotFailureKind.NON_FINITE, node_id=node_id, column=column
                )
            if pivot == 0.0:
                raise PivotFailureError(
                    PivotFailureKind.ZERO_PIVOT,
                    node_id=node_id,
                    column=column,
                    detail="fixed elimination order produced a zero pivot",
                )
            if abs(pivot) <= threshold:
                raise PivotFailureError(
                    PivotFailureKind.BELOW_THRESHOLD,
                    node_id=node_id,
                    column=column,
                    detail=(
                        f"|pivot|={abs(pivot):.6e} <= "
                        f"{config.threshold_ratio:.6e} * scale={scale:.6e}"
                    ),
                )
        elif config.policy == "DYNAMIC_PANEL":
            best_abs = -1.0
            found = False
            for row in range(column, pivot_dim):
                candidate = work[row, column]
                if not np.isfinite(candidate):
                    continue
                magnitude = abs(candidate)
                if magnitude > best_abs:
                    best_abs = magnitude
                    pivot_row = row
                    found = True
            if not found:
                raise PivotFailureError(
                    PivotFailureKind.NON_FINITE,
                    node_id=node_id,
                    column=column,
                    detail="pivot block column has no finite candidate",
                )
            if best_abs <= threshold:
                raise PivotFailureError(
                    PivotFailureKind.NO_CANDIDATE,
                    node_id=node_id,
                    column=column,
                    detail=(
                        f"best |candidate|={best_abs:.6e} <= "
                        f"{config.threshold_ratio:.6e} * scale={scale:.6e}"
                    ),
                )
            if pivot_row != column:
                work[[column, pivot_row], :] = work[[pivot_row, column], :]
                p_local[[column, pivot_row]] = p_local[[pivot_row, column]]
            pivot = work[column, column]
        else:  # pragma: no cover - config validation rejects this
            raise ValueError(f"unknown pivot policy {config.policy!r}")

        for row in range(column + 1, total_dim):
            multiplier = work[row, column] / pivot
            if not np.isfinite(multiplier):
                raise PivotFailureError(
                    PivotFailureKind.NON_FINITE,
                    node_id=node_id,
                    column=column,
                    detail="elimination produced a non-finite multiplier",
                )
            work[row, column] = multiplier
            with np.errstate(over="ignore", invalid="ignore"):
                work[row, column + 1 :] -= multiplier * work[column, column + 1 :]
            if not np.all(np.isfinite(work[row, column + 1 :])):
                raise PivotFailureError(
                    PivotFailureKind.NON_FINITE,
                    node_id=node_id,
                    column=column,
                    detail="elimination produced non-finite trailing values",
                )

    lower = np.zeros((total_dim, pivot_dim), dtype=np.float64)
    upper = np.zeros((pivot_dim, total_dim), dtype=np.float64)
    for row in range(total_dim):
        for col in range(pivot_dim):
            if row == col:
                lower[row, col] = 1.0
            elif row > col:
                lower[row, col] = work[row, col]
    for row in range(pivot_dim):
        upper[row, row:] = work[row, row:]

    schur = work[pivot_dim:, pivot_dim:].copy()
    return FrontFactorization(
        node_id=node_id,
        variables=(),
        pivot_dim=pivot_dim,
        l=lower,
        u=upper,
        p_local=p_local,
        schur=schur,
        pivot_count=pivot_dim,
        scale=scale,
    )


def factor_front_tree(
    nodes: Sequence[NodeCompileRecord],
    local_fronts: Sequence[np.ndarray],
    map_tables: Sequence[Sequence[MapTableEntry]],
    config: PivotPolicyConfig,
) -> tuple[FrontFactorization, ...]:
    """Assemble child contributions and factor every front (children first)."""

    if not (len(nodes) == len(local_fronts) == len(map_tables)):
        raise ValueError("nodes, local fronts and map tables must have equal length")
    order = sibling_friendly_order([node.parent_id for node in nodes])
    assembled = [np.array(front, dtype=np.float64, copy=True) for front in local_fronts]
    factors: list[FrontFactorization | None] = [None] * len(nodes)

    for node_id in order:
        node = nodes[node_id]
        for entry in map_tables[node_id]:
            child_factor = factors[entry.child_id]
            if child_factor is None:  # pragma: no cover - order guarantees this
                raise ValueError("child front is not factored before its parent")
            columns = np.asarray(entry.col_map, dtype=np.int64)
            rows = np.asarray(entry.row_map, dtype=np.int64)
            contribution = child_factor.schur[np.ix_(rows, rows)]
            target = assembled[node_id]
            target[np.ix_(columns, columns)] += contribution
        factor = factor_front(
            assembled[node_id],
            node.pivot_dim,
            config,
            node_id=node_id,
        )
        factors[node_id] = FrontFactorization(
            node_id=factor.node_id,
            variables=tuple(node.front_indices),
            pivot_dim=factor.pivot_dim,
            l=factor.l,
            u=factor.u,
            p_local=factor.p_local,
            schur=factor.schur,
            pivot_count=factor.pivot_count,
            scale=factor.scale,
        )

    return tuple(factor for factor in factors if factor is not None)


def solve_front_tree(
    factors: Sequence[FrontFactorization],
    parent: Sequence[int],
    rhs_hat: np.ndarray,
) -> np.ndarray:
    """Solve with factors whose permutation scope is the pivot block only.

    Mirrors the independent C++ ``solve_front_tree`` validation: pivot rows must
    stay inside the pivot block, update rows must not move, and every variable
    must be eliminated by exactly one front.
    """

    rhs = np.asarray(rhs_hat, dtype=np.float64).reshape(-1)
    if rhs.size == 0:
        raise ValueError("RHS must not be empty")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("RHS contains non-finite values")
    order = sibling_friendly_order(list(parent))
    by_node = {factor.node_id: factor for factor in factors}
    if sorted(by_node) != list(range(len(parent))):
        raise ValueError("front factors must cover every node exactly once")

    pivot_owner = np.zeros(rhs.size, dtype=bool)
    for factor in factors:
        variables = np.asarray(factor.variables, dtype=np.int64)
        pivot_dim = factor.pivot_dim
        total_dim = variables.size
        if (
            total_dim == 0
            or pivot_dim <= 0
            or pivot_dim > total_dim
            or factor.l.shape != (total_dim, pivot_dim)
            or factor.u.shape != (pivot_dim, total_dim)
            or factor.p_local.shape != (pivot_dim,)
        ):
            raise ValueError("front factor dimensions are invalid")
        if np.unique(variables).size != total_dim or np.any(variables >= rhs.size):
            raise ValueError("front variables are invalid")
        for row in range(pivot_dim):
            if int(factor.p_local[row]) >= pivot_dim:
                raise ValueError(
                    "front solve does not support pivoting across the front boundary"
                )
            variable = variables[row]
            if pivot_owner[variable]:
                raise ValueError("front variable has multiple pivot owners")
            pivot_owner[variable] = True
    if not bool(np.all(pivot_owner)):
        raise ValueError("front factors do not eliminate every variable")

    work = rhs.copy()
    y = np.zeros(rhs.size, dtype=np.float64)
    for node_id in order:
        factor = by_node[node_id]
        variables = np.asarray(factor.variables, dtype=np.int64)
        pivot_dim = factor.pivot_dim
        if variables.size != factor.l.shape[0]:
            raise ValueError("front variable count does not match its factors")
        pivot_rhs = np.empty(pivot_dim, dtype=np.float64)
        for row in range(pivot_dim):
            pivot_rhs[row] = work[variables[factor.p_local[row]]]
        for row in range(pivot_dim):
            value = pivot_rhs[row]
            for col in range(row):
                value -= factor.l[row, col] * y[variables[col]]
            y[variables[row]] = value
        for row in range(pivot_dim, variables.size):
            value = work[variables[row]]
            for col in range(pivot_dim):
                value -= factor.l[row, col] * y[variables[col]]
            work[variables[row]] = value

    x = np.zeros(rhs.size, dtype=np.float64)
    for node_id in reversed(order):
        factor = by_node[node_id]
        variables = np.asarray(factor.variables, dtype=np.int64)
        pivot_dim = factor.pivot_dim
        for row in range(pivot_dim - 1, -1, -1):
            value = y[variables[row]]
            for col in range(row + 1, variables.size):
                value -= factor.u[row, col] * x[variables[col]]
            diagonal = factor.u[row, row]
            if diagonal == 0.0 or not np.isfinite(diagonal):
                raise PivotFailureError(
                    PivotFailureKind.ZERO_PIVOT,
                    node_id=factor.node_id,
                    column=row,
                    detail="backward solve met a zero or non-finite diagonal",
                )
            x[variables[row]] = value / diagonal
    return x


def solve_front_tree_reference(
    nodes: Sequence[NodeCompileRecord],
    local_fronts: Sequence[np.ndarray],
    map_tables: Sequence[Sequence[MapTableEntry]],
    parent: Sequence[int],
    rhs_hat: np.ndarray,
    config: PivotPolicyConfig,
) -> FrontTreeSolution:
    """Factor and solve the front tree; used for preflight and verification."""

    factors = factor_front_tree(nodes, local_fronts, map_tables, config)
    x_hat = solve_front_tree(factors, parent, rhs_hat)
    return FrontTreeSolution(
        factors=factors,
        x_hat=x_hat,
        order=tuple(sibling_friendly_order(list(parent))),
    )
