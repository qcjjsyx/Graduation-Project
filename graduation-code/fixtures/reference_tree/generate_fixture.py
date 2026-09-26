"""Deterministic generator for the shared FP64 front-tree policy fixture.

Regenerate with::

    python generate_fixture.py

The fixture pins the *same-policy* front-tree semantics (assembly, pivot
policy, local P-vector, solve) so the Python host reference
(``src/verify/front_reference.py``) and the independent C++ reference
(``graduation-code/systemc/reference``) can be cross-validated against one
committed expectation.  After regenerating, both test suites must be re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOFTWARE_ROOT = HERE.parents[1] / "software"
sys.path.insert(0, str(SOFTWARE_ROOT))

from src.config import PivotPolicyConfig  # noqa: E402
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange  # noqa: E402
from src.verify.front_reference import (  # noqa: E402
    PivotFailureError,
    factor_front_tree,
    solve_front_tree,
)

TRAP = [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
NEAR_ZERO = [[100.0, 1.0], [1.0, 1e-4]]
CROSS_BLOCK = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
SWAP_BLOCK = [[0.0, 2.0, 3.0], [5.0, 1.0, 6.0], [7.0, 8.0, 9.0]]
STATIC_OK = [[4.0, 1.0, 0.5], [1.0, 3.0, 2.0], [0.5, 2.0, 5.0]]
FOREST_LEFT = [[4.0, 1.0, 0.5], [1.0, 3.0, 2.0], [0.5, 2.0, 5.0]]
FOREST_MID = [[3.0, 2.0], [2.0, 5.0]]
FOREST_RIGHT = [[2.0, 0.5], [0.5, 1.5]]


def front(values, pivot_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy()
    array[pivot_dim:, pivot_dim:] = 0.0
    return array


def single_node_case(name, values, pivot_dim, policy, threshold, rhs):
    size = len(values)
    return {
        "name": name,
        "policy": policy,
        "threshold_ratio": threshold,
        "parent": [-1],
        "rhs": list(rhs),
        "nodes": [
            {
                "variables": list(range(size)),
                "pivot_dim": pivot_dim,
                "front": front(values, pivot_dim).reshape(-1).tolist(),
            }
        ],
        "maps": [[]],
    }


def forest_case(name, policy, threshold, rhs):
    return {
        "name": name,
        "policy": policy,
        "threshold_ratio": threshold,
        "parent": [1, 2, -1, 4, -1],
        "rhs": list(rhs),
        "nodes": [
            {
                "variables": [0, 1, 2],
                "pivot_dim": 1,
                "front": front(FOREST_LEFT, 1).reshape(-1).tolist(),
            },
            {
                "variables": [1, 2],
                "pivot_dim": 1,
                "front": front(FOREST_MID, 1).reshape(-1).tolist(),
            },
            {
                "variables": [2],
                "pivot_dim": 1,
                "front": front(np.array([[FOREST_LEFT[2][2]]]), 1).reshape(-1).tolist(),
            },
            {
                "variables": [3, 4],
                "pivot_dim": 1,
                "front": front(FOREST_RIGHT, 1).reshape(-1).tolist(),
            },
            {
                "variables": [4],
                "pivot_dim": 1,
                "front": front(np.array([[FOREST_RIGHT[1][1]]]), 1).reshape(-1).tolist(),
            },
        ],
        "maps": [
            [],
            [{"child": 0, "row_map": [0, 1], "col_map": [0, 1]}],
            [{"child": 1, "row_map": [0], "col_map": [0]}],
            [],
            [{"child": 3, "row_map": [0], "col_map": [0]}],
        ],
    }


def build_case(case: dict) -> dict:
    parent = [int(value) for value in case["parent"]]
    nodes = [
        NodeCompileRecord(
            node_id=index,
            parent_id=parent[index],
            node_range=NodeRange(0, int(node["pivot_dim"])),
            front_indices=tuple(int(value) for value in node["variables"]),
        )
        for index, node in enumerate(case["nodes"])
    ]
    for child, ancestor in enumerate(parent):
        if ancestor >= 0 and child > ancestor:
            raise ValueError(
                f"case {case['name']}: nodes must be listed children-first"
            )
    map_tables = [
        [
            MapTableEntry(
                child_id=int(entry["child"]),
                row_map=[int(value) for value in entry["row_map"]],
                col_map=[int(value) for value in entry["col_map"]],
            )
            for entry in table
        ]
        for table in case["maps"]
    ]
    local_fronts = [
        np.asarray(node["front"], dtype=np.float64).reshape(
            len(node["variables"]), len(node["variables"])
        )
        for node in case["nodes"]
    ]
    config = PivotPolicyConfig(
        policy=case["policy"], threshold_ratio=float(case["threshold_ratio"])
    )
    case["threshold_ratio"] = config.threshold_ratio
    rhs = np.asarray(case["rhs"], dtype=np.float64)

    try:
        factors = factor_front_tree(nodes, local_fronts, map_tables, config)
    except PivotFailureError as failure:
        case["expected"] = {
            "status": "pivot_failure",
            "kind": failure.kind.value,
            "column": int(failure.column),
            "node_id": int(failure.node_id),
        }
        return case

    x_hat = solve_front_tree(factors, parent, rhs)
    case["expected"] = {
        "status": "ok",
        "x_hat": [float(value) for value in x_hat],
        "p_local": [
            [int(value) for value in factor.p_local] for factor in factors
        ],
    }
    return case


def main() -> None:
    cases = [
        single_node_case(
            "static_identity_front", STATIC_OK, 3, "STATIC", 0.0, [1.0, 2.0, 3.0]
        ),
        single_node_case(
            "static_trap_fails", TRAP, 3, "STATIC", 0.0, [1.0, 2.0, 3.0]
        ),
        single_node_case(
            "dynamic_trap_succeeds", TRAP, 3, "DYNAMIC_PANEL", 0.0, [1.0, 2.0, 3.0]
        ),
        single_node_case(
            "dynamic_cross_block_rejected",
            CROSS_BLOCK,
            2,
            "DYNAMIC_PANEL",
            0.0,
            [1.0, 2.0, 3.0],
        ),
        single_node_case(
            "static_threshold_below", NEAR_ZERO, 2, "STATIC", 1e-3, [1.0, 2.0]
        ),
        single_node_case(
            "dynamic_threshold_no_candidate",
            NEAR_ZERO,
            2,
            "DYNAMIC_PANEL",
            1e-3,
            [1.0, 2.0],
        ),
        {
            "name": "dynamic_swap_within_block",
            "policy": "DYNAMIC_PANEL",
            "threshold_ratio": 0.0,
            "parent": [1, -1],
            "rhs": [1.0, 2.0, 3.0],
            "nodes": [
                {
                    "variables": [0, 1, 2],
                    "pivot_dim": 2,
                    "front": front(SWAP_BLOCK, 2).reshape(-1).tolist(),
                },
                {
                    "variables": [2],
                    "pivot_dim": 1,
                    "front": front([[SWAP_BLOCK[2][2]]], 1).reshape(-1).tolist(),
                },
            ],
            "maps": [[], [{"child": 0, "row_map": [0], "col_map": [0]}]],
        },
        forest_case("multi_root_forest_static", "STATIC", 0.0, [1.0, 2.0, 3.0, 4.0, 5.0]),
        forest_case(
            "multi_root_forest_dynamic", "DYNAMIC_PANEL", 0.0, [1.0, 2.0, 3.0, 4.0, 5.0]
        ),
    ]
    payload = {
        "schema": "fp64_reference_tree_v1",
        "cases": [build_case(case) for case in cases],
    }
    (HERE / "tree_policy_cases.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
