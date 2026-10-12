#!/usr/bin/env python3
"""T12 solve production compiler fixtures.

Same hand-built compile IR as T11 (the tree cases are imported from
``t11_fixture_gen``) but compiled with the default
``factorization_and_solve`` scope so the device executes the full
``SOLVE_FORWARD`` / ``SOLVE_BACKWARD`` chain.  Device directories contain
manifest + image only; the independent checkers re-derive everything from
those inputs.

``cases.json`` carries one entry per (case, backend) with the factorization
and solve expectations; the INT32 pivot-propagation case is expected to fail
in the factorization, which must propagate as ``DEPENDENCY_FAILED`` to every
solve command.

``wide_tree/device`` (not part of ``cases.json``) is a production-CLI
artifact with more nodes than the default ``max_wait_tokens`` (256), the
regression case for removing the global solve gate (2026-10-11): a
three-level tree of 33 groups x 8 leaves under group roots and one top root,
so every node's fan-in stays small while the node count exceeds 256.

``slot_alias/device`` (not part of ``cases.json``) is the production-CLI
random matrix ``--n 120 --density 0.005 --seed 0`` (102 nodes), the
regression case for the PANEL_LU tile-version key alias (2026-10-11): before
the fix a PANEL_LU on front buffer slot 2 bumped slot 0's A21 tile versions,
a TRSM_RIGHT on slot 0 never dispatched its task and the device stalled
(30k-cycle batch timing out at 300k cycles).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

SOFTWARE = Path(__file__).resolve().parents[2] / "software"
sys.path.insert(0, str(SOFTWARE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import CommandCompilerConfig, GemmConfig, PivotPolicyConfig  # noqa: E402
from src.scheduler.command_compiler import compile_command_artifact  # noqa: E402
from src.verify.manifest import validate_manifest  # noqa: E402
from t11_fixture_gen import cases  # noqa: E402


WIDE_TREE_GROUPS = 33
WIDE_TREE_FAN = 8


def write_wide_tree(out: Path) -> Path:
    """Compile the >256-node bounded-fan-in tree with the default CLI config."""
    groups, fan = WIDE_TREE_GROUPS, WIDE_TREE_FAN
    n = groups * (fan + 1) + 1
    top = n - 1
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add(i: int, j: int, value: float) -> None:
        rows.append(i)
        cols.append(j)
        vals.append(value)

    for group in range(groups):
        root = group * (fan + 1) + fan
        for k in range(fan):
            leaf = group * (fan + 1) + k
            add(leaf, leaf, 4.0 + leaf % 3)
            add(leaf, root, 1.0)
            add(root, leaf, 0.5)
        add(root, root, 8.0)
        add(root, top, 1.0)
        add(top, root, 0.25)
    add(top, top, float(groups))
    base = out / "wide_tree"
    base.mkdir(parents=True, exist_ok=True)
    matrix_path = base / "wide_tree.mtx"
    scipy.io.mmwrite(str(matrix_path), sp.coo_matrix((vals, (rows, cols)), shape=(n, n)))
    directory = base / "device"
    subprocess.run(
        [sys.executable, "-m", "src.main", "--mtx", str(matrix_path), "--out", str(directory)],
        cwd=SOFTWARE,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    node_count = len(manifest["nodes"])
    max_wait = manifest["command_batch"]["max_wait_tokens"]
    if max_wait != 256 or node_count <= max_wait:
        raise SystemExit(
            f"wide_tree must exceed the default max_wait_tokens: nodes={node_count} "
            f"max_wait_tokens={max_wait}"
        )
    validate_manifest(directory / "manifest.json")
    return directory


SLOT_ALIAS_ARGS = ["--n", "120", "--density", "0.005", "--seed", "0"]


def write_slot_alias(out: Path) -> Path:
    """Compile the random matrix that stalled on the aliased version key."""
    directory = out / "slot_alias" / "device"
    directory.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "src.main", *SLOT_ALIAS_ARGS, "--out", str(directory)],
        cwd=SOFTWARE,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    validate_manifest(directory / "manifest.json")
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    index = []
    for name, nodes, fronts, maps in cases():
        dimension = max(node.node_range.end for node in nodes)
        for mode in ("FP32_SYSTEMC", "FP32_SYSTEMC_INT32_GEMM"):
            short = "fp32" if mode == "FP32_SYSTEMC" else "int32"
            factor_pass = name != "pivot_propagation" or short == "fp32"
            artifact = compile_command_artifact(
                nodes=nodes,
                map_tables=maps,
                local_fronts=fronts,
                permutation=list(range(dimension)),
                rhs=np.ones(dimension),
                config=CommandCompilerConfig(emit_solve_commands=True),
                pivot=PivotPolicyConfig(),
                gemm=GemmConfig(backend=mode),
            )
            directory = out / name / short / "solve" / "device"
            # Hand-built IR already uses identity ordering; declare that
            # truthfully for the production manifest validator.
            identity = list(range(dimension))
            artifact.manifest["preprocess"] = {
                "matching": {"method": "none"},
                "transform": {
                    "row_perm_match": identity,
                    "ordering_perm": identity,
                    "rhs_row_map": identity,
                },
            }
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "manifest.json").write_text(
                json.dumps(artifact.manifest, indent=2) + "\n"
            )
            (directory / "memory_image.bin").write_bytes(artifact.image)
            validate_manifest(directory / "manifest.json")
            index.append(
                {
                    "name": name,
                    "mode": short,
                    "device_dir": str(directory),
                    "factor_numeric_pass": factor_pass,
                    "solve_numeric_pass": factor_pass,
                }
            )
    (out / "cases.json").write_text(json.dumps(index, indent=2) + "\n")
    wide = write_wide_tree(out)
    alias = write_slot_alias(out)
    print(f"Generated {len(index)} T12 solve artifacts, {wide} and {alias} in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
