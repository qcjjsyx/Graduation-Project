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
from src.matching import RowMatching, canonicalize_csr, compute_row_matching, gather_rows
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
from src.verify.front_reference import solve_front_tree_reference
from src.verify.manifest import validate_manifest
from src.verify.metrics import componentwise_backward_error, residual_norm

TRANSFORM_EQUATIONS = (
    "B = P_match * D_r * A; A_hat = Q^T * B * Q; "
    "b_hat = Q^T * P_match * D_r * b; x = Q * x_hat"
)


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
    solution_reference_path: Path | None
    x_hat_reference_path: Path | None
    reference_check_performed: bool
    residual_norm: float | None
    original_residual_norm: float | None
    componentwise_backward_error: float | None
    node_count: int
    command_count: int
    descriptor_count: int
    max_front_size: int


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    original_matrix = canonicalize_csr(
        load_matrix(
            path=config.matrix.path,
            n=config.matrix.n,
            density=config.matrix.density,
            seed=config.matrix.seed,
        )
    )
    input_structurally_symmetric = is_structurally_symmetric(original_matrix)
    rhs_original, solution_original, rhs_source_kind = _prepare_rhs(
        original_matrix, config
    )

    equilibration = equilibrate_system(
        original_matrix, rhs_original, config.equilibration
    )
    matching = compute_row_matching(equilibration.matrix, config.matching)
    matrix_b = gather_rows(equilibration.matrix, matching.row_perm)
    rhs_m = equilibration.rhs[matching.row_perm]

    symbolic = run_symbolic_analysis(matrix_b, config)
    permutation = np.asarray(symbolic.permutation, dtype=np.int32)
    permuted = apply_permutation(matrix_b, permutation)
    rhs_permuted = rhs_m[permutation]
    x_hat_true = (
        solution_original[permutation] if solution_original is not None else None
    )

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

    x_hat_reference: np.ndarray | None = None
    if config.reference_check:
        reference = solve_front_tree_reference(
            nodes,
            local_fronts,
            map_tables,
            symbolic.parent,
            rhs_permuted,
            config.pivot,
        )
        x_hat_reference = reference.x_hat

    artifact = compile_command_artifact(
        nodes=nodes,
        map_tables=map_tables,
        local_fronts=local_fronts,
        permutation=symbolic.permutation,
        rhs=rhs_permuted,
        config=config.command,
        pivot=config.pivot,
    )

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    memory_image_path = out_dir / "memory_image.bin"
    reference_front_path = out_dir / "front_f64.bin"
    rhs_reference_path = out_dir / "rhs_hat_f64.bin"
    original_matrix_reference_path = out_dir / "original_matrix_f64.bin"
    original_rhs_reference_path = out_dir / "original_rhs_f64.bin"
    row_scale_exponents_path = out_dir / "row_scale_e.bin"
    solution_reference_path = (
        out_dir / "x_true_f64.bin" if solution_original is not None else None
    )
    x_hat_reference_path = (
        out_dir / "x_hat_f64.bin" if x_hat_reference is not None else None
    )

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
    if solution_original is not None and solution_reference_path is not None:
        solution_reference_path.write_bytes(
            solution_original.astype("<f8", copy=False).tobytes(order="C")
        )
    if x_hat_reference is not None and x_hat_reference_path is not None:
        x_hat_reference_path.write_bytes(
            x_hat_reference.astype("<f8", copy=False).tobytes(order="C")
        )

    residual = (
        residual_norm(permuted, x_hat_true, rhs_permuted)
        if x_hat_true is not None
        else None
    )
    original_residual: float | None = None
    backward_error: float | None = None
    if x_hat_reference is not None:
        x_reference = np.empty_like(x_hat_reference)
        x_reference[permutation] = x_hat_reference
        original_residual = residual_norm(original_matrix, x_reference, rhs_original)
        backward_error = componentwise_backward_error(
            original_matrix, x_reference, rhs_original
        )
    reference_offsets: list[int] = []
    cursor = 0
    for front in local_fronts:
        reference_offsets.append(cursor)
        cursor += front.size * 8

    rhs_row_map = matching.row_perm[permutation]
    manifest = dict(artifact.manifest)
    manifest["matrix"] = {
        "path": config.matrix.path,
        "n": int(original_matrix.shape[0]),
        "density": config.matrix.density,
        "seed": config.matrix.seed,
        "structurally_symmetric": input_structurally_symmetric,
    }
    manifest["preprocess"] = {
        "equations": TRANSFORM_EQUATIONS,
        "array_permutation_convention": "gather: y[i] = x[p[i]]",
        "equilibration": {
            "mode": config.equilibration.mode,
            "max_scale_exponent": config.equilibration.max_scale_exponent,
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
        },
        "matching": {
            "method": matching.method,
            "objective": matching.objective,
            "objective_note": (
                "max_product matching is solved through the exactly "
                "equivalent log objective sum log|a_j,sigma(j)|"
            ),
            "algorithm": "scipy.sparse.csgraph.min_weight_full_bipartite_matching",
            "full_matching": matching.full_matching,
            "log_product": matching.log_product,
            "row_perm": [int(value) for value in matching.row_perm],
            "matched_column_of_row": [
                int(value) for value in matching.matched_column
            ],
            "matched_value": [float(value) for value in matching.matched_value],
        },
        "transform": {
            "row_perm_match": [int(value) for value in matching.row_perm],
            "ordering_perm": [int(value) for value in permutation],
            "rhs_row_map": [int(value) for value in rhs_row_map],
            "rhs_row_map_definition": (
                "b_hat[i] = (D_r * b)[rhs_row_map[i]] with "
                "rhs_row_map[i] = row_perm_match[ordering_perm[i]]"
            ),
            "solution_map": "x_hat[i] = x[ordering_perm[i]]",
            "solution_recover": "x[ordering_perm[i]] = x_hat[i]",
            "local_p_vector_vs_global": (
                "the device P-vector is local to one pivot block; it is not "
                "the global row_perm_match or ordering_perm"
            ),
        },
    }
    manifest["verification"] = {
        "rhs_source": rhs_source_kind,
        "reference_residual_norm": residual,
        "residual_target": "original_A_x_equals_b",
        "reference_front_file": reference_front_path.name,
        "reference_front_offsets": reference_offsets,
        "rhs_reference_file": rhs_reference_path.name,
        "original_matrix_reference_file": original_matrix_reference_path.name,
        "original_rhs_reference_file": original_rhs_reference_path.name,
        "row_scale_exponent_file": row_scale_exponents_path.name,
        "solution_reference_file": (
            solution_reference_path.name if solution_reference_path else None
        ),
        "solution_reference_space": (
            "original_variable_order" if solution_reference_path else None
        ),
        "device_memory_contains_fp64_reference": False,
        "host_reference": {
            "performed": x_hat_reference is not None,
            "role": (
                "preflight_and_checker_only; never feeds the device memory image"
            ),
            "policy": config.pivot.policy,
            "x_hat_reference_file": (
                x_hat_reference_path.name if x_hat_reference_path else None
            ),
            "x_hat_reference_space": (
                "ordering_perm_variable_order" if x_hat_reference_path else None
            ),
            "original_reference_residual_norm": original_residual,
            "original_reference_backward_error": backward_error,
        },
    }
    manifest["pipeline_config"] = {
        "ordering": asdict(config.ordering),
        "equilibration": asdict(config.equilibration),
        "matching": asdict(config.matching),
        "pivot": asdict(config.pivot),
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
        x_hat_reference_path=x_hat_reference_path,
        reference_check_performed=x_hat_reference is not None,
        residual_norm=residual,
        original_residual_norm=original_residual,
        componentwise_backward_error=backward_error,
        node_count=len(nodes),
        command_count=len(artifact.commands),
        descriptor_count=len(artifact.descriptors),
        max_front_size=max(node.total_dim for node in nodes),
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
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return ``(rhs, golden_solution_or_None, source)``.

    The golden solution of a user-supplied RHS needs a full reference solve.
    That solve is part of the verification data generation and therefore only
    runs when ``reference_check`` is explicitly enabled; otherwise compilation
    must not pay for (or be blocked by) it.
    """

    dimension = int(matrix.shape[0])
    if config.solve.rhs_path:
        rhs = load_vector(config.solve.rhs_path, dimension)
        if not config.reference_check:
            return rhs, None, "file"
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
