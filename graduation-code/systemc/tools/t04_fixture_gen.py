#!/usr/bin/env python3
"""Generate T04 device run directories with the current software compiler.

Each device run directory contains only manifest.json and memory_image.bin, so
the C++ ArtifactLoader must not depend on golden/reference files. Cases cover a
single node, multiple nodes, a multi-root forest, DYNAMIC_PANEL and the
existing 256x256 input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

SYSTEMC_DIR = Path(__file__).resolve().parents[1]
SOFTWARE_DIR = SYSTEMC_DIR.parent / "software"
if str(SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_DIR))

from src.config import (  # noqa: E402
    CommandCompilerConfig,
    EquilibrationConfig,
    GemmConfig,
    MatchingConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
    PivotPolicyConfig,
    SolveInputConfig,
)
from src.pipeline import run_pipeline  # noqa: E402


def banded_matrix(n: int, bandwidth: int = 2) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row in range(n):
        for offset in range(-bandwidth, bandwidth + 1):
            col = row + offset
            if 0 <= col < n:
                value = float(4 * bandwidth + 1)
                if offset != 0:
                    value = 1.0 / (abs(offset) + 1.0)
                rows.append(row)
                cols.append(col)
                values.append(value)
    return sp.csr_matrix((values, (rows, cols)), shape=(n, n))


def block_diagonal_matrix(sizes: tuple[int, ...]) -> sp.csr_matrix:
    blocks = [banded_matrix(size, bandwidth=1) for size in sizes]
    return sp.block_diag(blocks, format="csr")


def dense_diagonally_dominant(n: int) -> sp.csr_matrix:
    rng = np.random.default_rng(7)
    matrix = rng.uniform(-0.5, 0.5, size=(n, n))
    matrix = matrix + np.diag(np.full(n, float(n) + 4.0))
    return sp.csr_matrix(matrix)


def write_matrix(path: Path, matrix: sp.csr_matrix) -> None:
    scipy.io.mmwrite(str(path), matrix)


def build_case(
    *,
    name: str,
    matrix_path: Path,
    matrix_dim: int,
    rhs_path: Path,
    out_root: Path,
    pivot_policy: str,
    pivot_threshold: float = 0.0,
    max_front_size: int = 256,
    tile_size: int = 16,
    gemm: GemmConfig | None = None,
) -> dict:
    staging = out_root / "_staging" / name
    device = out_root / name / "device"
    if staging.exists():
        shutil.rmtree(staging)
    if device.exists():
        shutil.rmtree(device)
    staging.mkdir(parents=True)

    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(
                path=str(matrix_path), n=matrix_dim, density=0.1, seed=0
            ),
            ordering=OrderingConfig(method="amd", max_supernode_size=max_front_size),
            equilibration=EquilibrationConfig(mode="pow2-row", max_scale_exponent=60),
            matching=MatchingConfig(method="weighted_matching"),
            pivot=PivotPolicyConfig(policy=pivot_policy, threshold_ratio=pivot_threshold),
            gemm=gemm or GemmConfig(),
            command=CommandCompilerConfig(
                tile_size=tile_size,
                max_front_size=max_front_size,
                max_wait_tokens=256,
            ),
            solve=SolveInputConfig(rhs_path=str(rhs_path), seed=1),
            reference_check=False,
            out_dir=staging,
        )
    )

    device.mkdir(parents=True)
    shutil.copyfile(staging / "manifest.json", device / "manifest.json")
    shutil.copyfile(staging / "memory_image.bin", device / "memory_image.bin")
    shutil.rmtree(staging)

    manifest = json.loads((device / "manifest.json").read_text(encoding="utf-8"))
    node_count = manifest["symbolic"]["node_count"]
    roots = sum(1 for parent in manifest["symbolic"]["parent"] if parent == -1)
    info = {
        "name": name,
        "device_dir": str(device),
        "policy": manifest["pivot"]["policy"],
        "pivot_threshold_bits_fp32": manifest["pivot"]["threshold_bits_fp32"],
        "node_count": node_count,
        "roots": roots,
        "command_count": outputs.command_count,
        "descriptor_count": outputs.descriptor_count,
        "image_bytes": (device / "memory_image.bin").stat().st_size,
        "gemm_backend": manifest["gemm"]["backend"],
        "gemm_storage_format": manifest["gemm"]["storage_format"],
        "rhs_source": manifest["verification"]["rhs_source"],
        "host_reference_performed": manifest["verification"]["host_reference"]["performed"],
    }
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--with-large",
        action="store_true",
        help="also compile the existing 256x256 input as a capacity smoke case",
    )
    args = parser.parse_args()

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = out_root / "_inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []

    single_matrix = dense_diagonally_dominant(6)
    single_path = inputs / "single.mtx"
    write_matrix(single_path, single_matrix)
    single_rhs = inputs / "single_rhs.npy"
    np.save(single_rhs, np.linspace(0.5, -0.5, 6))
    cases.append(
        build_case(
            name="single_static",
            matrix_path=single_path,
            matrix_dim=6,
            rhs_path=single_rhs,
            out_root=out_root,
            pivot_policy="STATIC",
        )
    )

    multi_matrix = banded_matrix(24, bandwidth=2)
    multi_path = inputs / "multi.mtx"
    write_matrix(multi_path, multi_matrix)
    multi_rhs = inputs / "multi_rhs.npy"
    rng = np.random.default_rng(11)
    np.save(multi_rhs, rng.uniform(-1.0, 1.0, 24))
    cases.append(
        build_case(
            name="multi_static",
            matrix_path=multi_path,
            matrix_dim=24,
            rhs_path=multi_rhs,
            out_root=out_root,
            pivot_policy="STATIC",
        )
    )

    multiroot_matrix = block_diagonal_matrix((5, 7))
    multiroot_path = inputs / "multiroot.mtx"
    write_matrix(multiroot_path, multiroot_matrix)
    multiroot_rhs = inputs / "multiroot_rhs.npy"
    np.save(multiroot_rhs, np.linspace(-1.0, 1.0, 12))
    cases.append(
        build_case(
            name="multiroot_static",
            matrix_path=multiroot_path,
            matrix_dim=12,
            rhs_path=multiroot_rhs,
            out_root=out_root,
            pivot_policy="STATIC",
        )
    )

    cases.append(
        build_case(
            name="dynamic_panel",
            matrix_path=multi_path,
            matrix_dim=24,
            rhs_path=multi_rhs,
            out_root=out_root,
            pivot_policy="DYNAMIC_PANEL",
        )
    )

    if args.with_large:
        large_matrix = SOFTWARE_DIR / "example" / "256X256JJ.mat"
        large_rhs = inputs / "large_rhs.npy"
        np.save(large_rhs, np.linspace(-0.25, 0.25, 256))
        cases.append(
            build_case(
                name="large_256",
                matrix_path=large_matrix,
                matrix_dim=256,
                rhs_path=large_rhs,
                out_root=out_root,
                pivot_policy="STATIC",
            )
        )

    (out_root / "fixtures.json").write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"cases": cases}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
