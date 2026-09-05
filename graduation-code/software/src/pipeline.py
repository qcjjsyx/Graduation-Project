from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import MatrixRankWarning

from src.config import PipelineConfig
from src.dataStruct import NodeCompileRecord, NodeRange, SymbolicResult
from src.equilibration import equilibrate_system
from src.io import write_manifest
from src.matrix_io import load_matrix, load_vector
from src.scheduler.command_compiler import compile_command_artifact
from src.scheduler.map_gen import generate_map_tables
from src.symbolic.fill import symbolic_fill_pattern
from src.symbolic.ordering import apply_permutation, compute_ordering
from src.symbolic.pattern import (
    is_structurally_symmetric,
    symmetric_sparsity_pattern,
)
from src.symbolic.supernode import (
    build_front_indices_from_filled,
    build_supernode_parent,
    build_supernodes_from_filled,
)
from src.verify.manifest import validate_manifest
from src.verify.metrics import residual_norm


@dataclass(frozen=True)
class PipelineOutputs:
    out_dir: Path
    manifest_path: Path
    memory_image_path: Path
    reference_front_path: Path
    rhs_reference_path: Path
    original_matrix_reference_path: Path
    original_rhs_reference_path: Path
    row_scale_exponents_path: Path
    solution_reference_path: Path
    residual_norm: float
    original_residual_norm: float
    node_count: int
    command_count: int
    descriptor_count: int


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    original_matrix = load_matrix(
        path=config.matrix.path,
        n=config.matrix.n,
        density=config.matrix.density,
        seed=config.matrix.seed,
    )
    input_structurally_symmetric = is_structurally_symmetric(original_matrix)
    rhs_original, solution_original, rhs_source_kind = _prepare_rhs(
        original_matrix, config
    )

    equilibration = equilibrate_system(
        original_matrix, rhs_original, config.equilibration
    )
    matrix = equilibration.matrix
    rhs_scaled = equilibration.rhs
    symbolic = run_symbolic_analysis(matrix, config)
    permutation = np.asarray(symbolic.permutation, dtype=np.int32)
    permuted = apply_permutation(matrix, permutation)
    rhs_permuted = rhs_scaled[permutation]
    solution_permuted = solution_original[permutation]

    map_tables = generate_map_tables(
        [node_range.as_tuple() for node_range in symbolic.node_ranges],
        symbolic.parent,
        symbolic.front_indices,
    )
    local_fronts = [
        extract_local_contribution(permuted, front_indices, node_range)
        for node_range, front_indices in zip(
            symbolic.node_ranges, symbolic.front_indices
        )
    ]
    nodes = [
        NodeCompileRecord(
            node_id=node_id,
            parent_id=symbolic.parent[node_id],
            node_range=symbolic.node_ranges[node_id],
            front_indices=tuple(symbolic.front_indices[node_id]),
        )
        for node_id in range(len(symbolic.node_ranges))
    ]
    artifact = compile_command_artifact(
        nodes=nodes,
        map_tables=map_tables,
        local_fronts=local_fronts,
        permutation=symbolic.permutation,
        rhs=rhs_permuted,
        config=config.command,
    )

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    memory_image_path = out_dir / "memory_image.bin"
    reference_front_path = out_dir / "reference_front_f64.bin"
    rhs_reference_path = out_dir / "rhs_f64.bin"
    original_matrix_reference_path = out_dir / "original_matrix_f64.bin"
    original_rhs_reference_path = out_dir / "original_rhs_f64.bin"
    row_scale_exponents_path = out_dir / "row_scale_e.bin"
    solution_reference_path = out_dir / "x_reference_f64.bin"

    memory_image_path.write_bytes(artifact.image)
    reference_front_path.write_bytes(
        b"".join(front.astype("<f8", copy=False).tobytes(order="C") for front in local_fronts)
    )
    rhs_reference_path.write_bytes(rhs_permuted.astype("<f8").tobytes(order="C"))
    original_matrix_reference_path.write_bytes(
        original_matrix.toarray().astype("<f8", copy=False).tobytes(order="C")
    )
    original_rhs_reference_path.write_bytes(
        rhs_original.astype("<f8", copy=False).tobytes(order="C")
    )
    row_scale_exponents_path.write_bytes(
        equilibration.row_scale_exponents.astype("<i2", copy=False).tobytes(order="C")
    )
    solution_reference_path.write_bytes(
        solution_permuted.astype("<f8").tobytes(order="C")
    )

    residual = residual_norm(permuted, solution_permuted, rhs_permuted)
    original_residual = residual_norm(
        original_matrix, solution_original, rhs_original
    )
    reference_offsets: list[int] = []
    cursor = 0
    for front in local_fronts:
        reference_offsets.append(cursor)
        cursor += front.size * 8

    manifest = dict(artifact.manifest)
    manifest["matrix"] = {
        "path": config.matrix.path,
        "n": int(original_matrix.shape[0]),
        "density": config.matrix.density,
        "seed": config.matrix.seed,
        "structurally_symmetric": input_structurally_symmetric,
    }
    manifest["equilibration"] = {
        "mode": config.equilibration.mode,
        "equation": "D_r * A * x = D_r * b",
        "solution_requires_unscale": False,
        "row_scale_exponent_file": row_scale_exponents_path.name,
        "row_scale_exponent_dtype": "int16",
        "row_scale_exponent_count": int(original_matrix.shape[0]),
        "row_max_before": {
            "min_nonzero": equilibration.row_max_before_min,
            "max": equilibration.row_max_before_max,
        },
        "row_max_after": {
            "min_nonzero": equilibration.row_max_after_min,
            "max": equilibration.row_max_after_max,
        },
    }
    manifest["verification"] = {
        "rhs_source": rhs_source_kind,
        "reference_residual_norm": residual,
        "original_reference_residual_norm": original_residual,
        "reference_front_file": reference_front_path.name,
        "reference_front_offsets": reference_offsets,
        "rhs_reference_file": rhs_reference_path.name,
        "original_matrix_reference_file": original_matrix_reference_path.name,
        "original_rhs_reference_file": original_rhs_reference_path.name,
        "row_scale_exponent_file": row_scale_exponents_path.name,
        "solution_reference_file": solution_reference_path.name,
        "device_memory_contains_fp64_reference": False,
    }
    manifest["pipeline_config"] = {
        "ordering": asdict(config.ordering),
        "equilibration": asdict(config.equilibration),
        "command": asdict(config.command),
    }
    write_manifest(str(manifest_path), manifest)
    validate_manifest(manifest_path)

    return PipelineOutputs(
        out_dir=out_dir,
        manifest_path=manifest_path,
        memory_image_path=memory_image_path,
        reference_front_path=reference_front_path,
        rhs_reference_path=rhs_reference_path,
        original_matrix_reference_path=original_matrix_reference_path,
        original_rhs_reference_path=original_rhs_reference_path,
        row_scale_exponents_path=row_scale_exponents_path,
        solution_reference_path=solution_reference_path,
        residual_norm=residual,
        original_residual_norm=original_residual,
        node_count=len(nodes),
        command_count=len(artifact.commands),
        descriptor_count=len(artifact.descriptors),
    )


def run_symbolic_analysis(matrix, config: PipelineConfig) -> SymbolicResult:
    symbolic_pattern = symmetric_sparsity_pattern(matrix)
    permutation = compute_ordering(symbolic_pattern, config.ordering.method)
    permuted_pattern = apply_permutation(symbolic_pattern, permutation)
    filled = symbolic_fill_pattern(permuted_pattern)
    supernodes = build_supernodes_from_filled(
        filled.parent,
        filled.columns,
        max_size=config.ordering.max_supernode_size,
    )
    node_ranges = build_node_ranges(supernodes)
    return SymbolicResult(
        permutation=permutation.astype(np.int32).tolist(),
        parent=build_supernode_parent(filled.parent, supernodes),
        supernodes=supernodes,
        node_ranges=node_ranges,
        front_indices=build_front_indices_from_filled(supernodes, filled.columns),
    )


def build_node_ranges(supernodes: List[List[int]]) -> List[NodeRange]:
    ranges: List[NodeRange] = []
    start = 0
    for supernode in supernodes:
        end = start + len(supernode)
        ranges.append(NodeRange(start=start, end=end))
        start = end
    return ranges


def extract_local_contribution(
    matrix, front_indices: List[int], node_range: NodeRange
) -> np.ndarray:
    pivot_dim = node_range.size
    if front_indices[:pivot_dim] != list(range(node_range.start, node_range.end)):
        raise ValueError("front pivot prefix does not match node range")
    local = matrix[front_indices][:, front_indices].toarray().astype(np.float64)
    if pivot_dim < len(front_indices):
        local[pivot_dim:, pivot_dim:] = 0.0
    return local


def _prepare_rhs(
    matrix, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    dimension = int(matrix.shape[0])
    if config.solve.rhs_path:
        rhs = load_vector(config.solve.rhs_path, dimension)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MatrixRankWarning)
            solution = np.asarray(spla.spsolve(matrix.tocsc(), rhs), dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            raise ValueError("reference solve produced non-finite values for supplied RHS")
        return rhs, solution, "file"

    rng = np.random.default_rng(config.solve.seed)
    solution = rng.uniform(-1.0, 1.0, dimension).astype(np.float64)
    rhs = np.asarray(matrix @ solution, dtype=np.float64).reshape(-1)
    return rhs, solution, "generated_from_x_true"
