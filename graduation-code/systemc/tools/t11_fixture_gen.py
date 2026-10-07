#!/usr/bin/env python3
"""T11 production compiler fixtures. Device directories contain manifest/image only.

Local fronts are numeric INPUTS. No factor, pivot or reference front is emitted.
The independent C++ checker factors these inputs after device execution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOFTWARE = Path(__file__).resolve().parents[2] / "software"
sys.path.insert(0, str(SOFTWARE))
from src.config import CommandCompilerConfig, GemmConfig, PivotPolicyConfig  # noqa: E402
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange  # noqa: E402
from src.scheduler.command_compiler import compile_command_artifact  # noqa: E402
from src.verify.manifest import validate_manifest  # noqa: E402


def numeric_front(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=(n, n)).astype(np.float64) / 32.0
    a[p:, p:] = 0
    for i in range(p):
        a[i, i] += 8.0
    return a


def cases():
    # Each spec: parent id, pivot range, full ordered front indices.
    specs = {
        "single": [(-1, (0, 3), (0, 1, 2))],
        "two_level": [(1, (0, 1), (0, 1)), (-1, (1, 2), (1,))],
        "three_level": [(1, (0, 1), (0, 1, 2)), (2, (1, 2), (1, 2)),
                        (-1, (2, 3), (2,))],
        "fork": [(2, (0, 1), (0, 2, 3)), (2, (1, 2), (1, 2, 3)),
                 (-1, (2, 4), (2, 3))],
        "forest": [(2, (0, 1), (0, 2)), (3, (1, 2), (1, 3)),
                   (-1, (2, 3), (2,)), (-1, (3, 4), (3,))],
        "mapped_tail": [(2, (0, 1), tuple([0, *range(2, 19)])),
                        (2, (1, 2), tuple([1, *reversed(range(2, 19))])),
                        (-1, (2, 19), tuple(range(2, 19)))],
        "late_child0": [(2, (0, 17), tuple([*range(17), 18, 19])),
                        (2, (17, 18), (17, 18, 19)), (-1, (18, 20), (18, 19))],
        "late_child1": [(2, (17, 18), (17, 18, 19)),
                        (2, (0, 17), tuple([*range(17), 18, 19])),
                        (-1, (18, 20), (18, 19))],
        "pivot_propagation": [(1, (0, 1), (0, 1, 2)), (-1, (1, 3), (1, 2))],
    }
    # Deterministic random input variations exercise sibling completion/order
    # without turning this into an unbounded performance experiment.
    for seed in (7, 23, 49):
        order = np.random.default_rng(seed).permutation(2)
        base = specs["late_child0"]
        specs[f"fork_seed{seed}"] = [base[int(i)] for i in order] + [base[2]]
    for name, spec in specs.items():
        nodes = [NodeCompileRecord(i, parent, NodeRange(*pivot), indices)
                 for i, (parent, pivot, indices) in enumerate(spec)]
        seed = int(name.split("seed")[-1]) if "seed" in name else 13
        fronts = [numeric_front(node.total_dim, node.pivot_dim, seed + i) for i, node in enumerate(nodes)]
        if name == "two_level":
            fronts = [np.array([[4., 1.], [2., 0.]]), np.array([[5.]])]
        if name == "pivot_propagation":
            fronts = [np.array([[1., 0., 1.], [1., 0., 0.], [2.**-35, 0., 0.]]),
                      np.array([[2., 0.], [0., 0.]])]
        maps: list[list[MapTableEntry]] = [[] for _ in nodes]
        for child in nodes:
            if child.parent_id < 0:
                continue
            parent = nodes[child.parent_id]
            col_map = [parent.front_indices.index(v) for v in child.front_indices[child.pivot_dim:]]
            maps[child.parent_id].append(MapTableEntry(child.node_id, list(range(child.update_dim)), col_map))
        yield name, nodes, fronts, maps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    index = []
    for name, nodes, fronts, maps in cases():
        dimension = max(node.node_range.end for node in nodes)
        for mode in ("FP32_SYSTEMC", "FP32_SYSTEMC_INT32_GEMM"):
            short = "fp32" if mode == "FP32_SYSTEMC" else "int32"
            artifact = compile_command_artifact(
                nodes=nodes, map_tables=maps, local_fronts=fronts,
                permutation=list(range(dimension)), rhs=np.ones(dimension),
                config=CommandCompilerConfig(emit_solve_commands=False),
                pivot=PivotPolicyConfig(), gemm=GemmConfig(backend=mode))
            directory = out / name / short / "device"
            # These hand-built compile IR inputs already use identity ordering;
            # declare that truthfully for the production manifest validator.
            identity = list(range(dimension))
            artifact.manifest["preprocess"] = {"matching": {"method": "none"},
                "transform": {"row_perm_match": identity, "ordering_perm": identity,
                              "rhs_row_map": identity}}
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "manifest.json").write_text(json.dumps(artifact.manifest, indent=2) + "\n")
            (directory / "memory_image.bin").write_bytes(artifact.image)
            validate_manifest(directory / "manifest.json")
            index.append({"name": name, "mode": short, "device_dir": str(directory),
                          "numeric_pass": name != "pivot_propagation" or short == "fp32"})
    (out / "cases.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Generated {len(index)} T11 factorization artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
