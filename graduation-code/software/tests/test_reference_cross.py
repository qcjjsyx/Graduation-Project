"""Cross-validation of the Python host reference against the shared fixture.

``graduation-code/fixtures/reference_tree/tree_policy_cases.json`` pins the
same-policy front-tree semantics (assembly, pivot policy, local P-vector,
solve, failure kinds) for the Python host reference
(``src/verify/front_reference.py``) and the independent C++ reference
(``fp64_reference_tests``).  Both suites must reproduce the committed
expectations, which cross-validates the two implementations under the *same*
pivot policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import PivotPolicyConfig
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange
from src.verify.front_reference import (
    PivotFailureError,
    solve_front_tree_reference,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "reference_tree"
    / "tree_policy_cases.json"
)


def _load_cases():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema"] == "fp64_reference_tree_v1"
    return payload["cases"]


def _run_case(case):
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
    rhs = np.asarray(case["rhs"], dtype=np.float64)
    return solve_front_tree_reference(
        nodes, local_fronts, map_tables, parent, rhs, config
    )


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_python_reference_matches_shared_tree_fixture(case):
    expected = case["expected"]
    if expected["status"] == "pivot_failure":
        with pytest.raises(PivotFailureError) as failure:
            _run_case(case)
        assert failure.value.kind.value == expected["kind"]
        assert failure.value.column == expected["column"]
        assert failure.value.node_id == expected["node_id"]
        return

    solution = _run_case(case)
    np.testing.assert_allclose(
        solution.x_hat, np.asarray(expected["x_hat"]), rtol=1e-10, atol=1e-12
    )
    got_p_local = [[int(value) for value in factor.p_local] for factor in solution.factors]
    assert got_p_local == expected["p_local"]
