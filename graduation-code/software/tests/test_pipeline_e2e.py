import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from src.config import (
    EquilibrationConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
)
from src.dataStruct import NODE_TASK_BYTE_SIZE
from src.pipeline import run_pipeline
from src.verify.manifest import ManifestValidationError, validate_manifest


def test_pipeline_outputs_are_manifest_consistent(tmp_path: Path):
    config = PipelineConfig(
        matrix=MatrixInputConfig(path=None, n=12, density=0.25, seed=4),
        ordering=OrderingConfig(method="amd"),
        out_dir=tmp_path,
    )

    outputs = run_pipeline(config)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    validation = validate_manifest(outputs.manifest_path)

    assert outputs.tasks_path.exists()
    assert outputs.map_table_path.exists()
    assert outputs.front_q_path.exists()
    assert outputs.front_e_path.exists()
    assert outputs.original_matrix_reference_path.exists()
    assert outputs.original_rhs_reference_path.exists()
    assert outputs.row_scale_exponents_path.exists()
    assert outputs.task_count == outputs.node_count
    assert validation.node_count == outputs.node_count
    assert validation.task_count == outputs.task_count
    assert outputs.tasks_path.stat().st_size == outputs.task_count * NODE_TASK_BYTE_SIZE

    assert manifest["abi"]["node_task_byte_size"] == NODE_TASK_BYTE_SIZE
    assert manifest["symbolic"]["node_count"] == outputs.node_count
    assert manifest["quantization"]["format"] == "S_format_local_contribution"
    assert manifest["quantization"]["exponent_dtype"] == "int16"
    assert manifest["equilibration"]["mode"] == "pow2-row"
    assert manifest["equilibration"]["solution_requires_unscale"] is False
    assert outputs.row_scale_exponents_path.stat().st_size == 12 * 2
    assert manifest["output_sizes"]["tasks.bin"] == outputs.tasks_path.stat().st_size
    assert manifest["output_sizes"]["front_q.bin"] == outputs.front_q_path.stat().st_size
    assert manifest["output_sizes"]["front_e.bin"] == outputs.front_e_path.stat().st_size
    assert manifest["output_sizes"]["map_table.bin"] == outputs.map_table_path.stat().st_size

    for node_id, node in manifest["nodes"].items():
        front_q = node["front_q"]
        front_e = node["front_e"]
        map_table = node["map_table"]
        local_source = node["local_source"]
        front_dim = len(node["front_indices"])
        assert front_q["offset"] % config.memory.alignment == 0
        assert front_e["offset"] % config.memory.alignment == 0
        assert map_table["offset"] % config.memory.alignment == 0
        assert node["update_q"]["offset"] % config.memory.alignment == 0
        assert node["l_factor"]["offset"] % config.memory.alignment == 0
        assert node["u_factor"]["offset"] % config.memory.alignment == 0
        assert node["front_q_file_offset"] + front_q["size"] <= outputs.front_q_path.stat().st_size
        assert node["front_e_file_offset"] + front_e["size"] <= outputs.front_e_path.stat().st_size
        assert node["map_table_file_offset"] + map_table["size"] <= outputs.map_table_path.stat().st_size
        assert local_source["shape"] == [front_dim, front_dim]
        assert front_q["size"] == front_dim * front_dim * 4
        assert front_e["size"] == 2

    positions = {node_id: idx for idx, node_id in enumerate(manifest["task_order"])}
    for child, parent in enumerate(manifest["symbolic"]["parent"]):
        if parent >= 0:
            assert positions[child] < positions[parent]


def test_pipeline_artifacts_are_reproducible(tmp_path: Path):
    base = PipelineConfig(
        matrix=MatrixInputConfig(path=None, n=10, density=0.3, seed=9),
        ordering=OrderingConfig(method="amd"),
        out_dir=tmp_path / "first",
    )
    first = run_pipeline(base)
    second = run_pipeline(replace(base, out_dir=tmp_path / "second"))
    for filename in (
        "tasks.bin",
        "map_table.bin",
        "front_q.bin",
        "front_e.bin",
        "rhs_q.bin",
        "rhs_e.bin",
        "memory_image.bin",
        "reference_front_f64.bin",
        "rhs_f64.bin",
        "original_matrix_f64.bin",
        "original_rhs_f64.bin",
        "row_scale_e.bin",
        "x_reference_f64.bin",
        "manifest.json",
    ):
        assert (first.out_dir / filename).read_bytes() == (
            second.out_dir / filename
        ).read_bytes()


def test_pipeline_supports_asymmetric_structure_without_losing_values(
    tmp_path: Path,
):
    matrix = sp.csr_matrix(
        np.array(
            [
                [10.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 11.0, 3.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 12.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 13.0, 5.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 14.0, 6.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 15.0],
            ]
        )
    )
    matrix_path = tmp_path / "asymmetric.mtx"
    scipy.io.mmwrite(matrix_path, matrix)
    config = PipelineConfig(
        matrix=MatrixInputConfig(path=str(matrix_path), n=6),
        ordering=OrderingConfig(method="rcm"),
        equilibration=EquilibrationConfig(mode="none"),
        out_dir=tmp_path / "artifact",
    )

    outputs = run_pipeline(config)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    validate_manifest(outputs.manifest_path)

    assert manifest["matrix"]["structurally_symmetric"] is False
    assert (
        manifest["symbolic"]["pattern_source"]
        == "union_of_A_and_transpose_nonzero_patterns"
    )
    assert manifest["symbolic"]["pattern_structurally_symmetric"] is True
    assert outputs.original_residual_norm <= 1e-12

    permutation = np.asarray(manifest["symbolic"]["permutation"])
    expected = matrix[permutation][:, permutation].toarray()
    reconstructed = np.zeros_like(expected)
    reference_data = outputs.reference_front_path.read_bytes()
    for node_id in range(outputs.node_count):
        node = manifest["nodes"][str(node_id)]
        front = np.asarray(node["front_indices"])
        dimension = len(front)
        offset = node["reference_front_file_offset"]
        local = np.frombuffer(
            reference_data,
            dtype="<f8",
            count=dimension * dimension,
            offset=offset,
        ).reshape(dimension, dimension)
        reconstructed[np.ix_(front, front)] += local

    np.testing.assert_array_equal(reconstructed, expected)

    manifest["symbolic"]["pattern_source"] = "original_matrix"
    outputs.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="symbolic pattern"):
        validate_manifest(outputs.manifest_path)
