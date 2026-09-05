import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from src.command_codec import COMMAND_RECORD_BYTES, DESCRIPTOR_RECORD_BYTES
from src.config import MatrixInputConfig, OrderingConfig, PipelineConfig, SolveInputConfig
from src.pipeline import run_pipeline
from src.verify.manifest import ManifestValidationError, validate_manifest


def test_pipeline_emits_command_v1_artifact(tmp_path: Path):
    config = PipelineConfig(
        matrix=MatrixInputConfig(path=None, n=12, density=0.25, seed=4),
        ordering=OrderingConfig(method="amd"),
        out_dir=tmp_path,
    )
    outputs = run_pipeline(config)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    result = validate_manifest(outputs.manifest_path)

    assert manifest["abi"]["version"] == 1
    assert manifest["abi"]["command_record_bytes"] == COMMAND_RECORD_BYTES
    assert manifest["abi"]["descriptor_record_bytes"] == DESCRIPTOR_RECORD_BYTES
    assert manifest["compiler"]["device_format"] == "FP32"
    assert manifest["compiler"]["global_bfp"] is False
    assert manifest["verification"]["device_memory_contains_fp64_reference"] is False
    assert result.node_count == outputs.node_count
    assert result.command_count == outputs.command_count
    assert result.descriptor_count == outputs.descriptor_count
    assert not (tmp_path / "tasks.bin").exists()
    assert not (tmp_path / "front_q.bin").exists()
    assert not (tmp_path / "front_e.bin").exists()


def test_command_artifacts_are_reproducible(tmp_path: Path):
    base = PipelineConfig(
        matrix=MatrixInputConfig(path=None, n=10, density=0.3, seed=9),
        ordering=OrderingConfig(method="amd"),
        out_dir=tmp_path / "first",
    )
    first = run_pipeline(base)
    second = run_pipeline(replace(base, out_dir=tmp_path / "second"))
    for filename in (
        "manifest.json",
        "memory_image.bin",
        "reference_front_f64.bin",
        "rhs_f64.bin",
        "original_matrix_f64.bin",
        "original_rhs_f64.bin",
        "row_scale_e.bin",
        "x_reference_f64.bin",
    ):
        assert (first.out_dir / filename).read_bytes() == (
            second.out_dir / filename
        ).read_bytes()


def test_pipeline_rejects_nonfinite_matrix_and_empty_rhs(tmp_path: Path):
    bad_matrix = sp.csr_matrix(np.array([[1.0, np.nan], [0.0, 2.0]]))
    matrix_path = tmp_path / "bad.mtx"
    scipy.io.mmwrite(matrix_path, bad_matrix)
    with pytest.raises(ValueError, match="matrix contains non-finite"):
        run_pipeline(
            PipelineConfig(
                matrix=MatrixInputConfig(path=str(matrix_path), n=2),
                out_dir=tmp_path / "bad-out",
            )
        )

    valid_path = tmp_path / "valid.mtx"
    scipy.io.mmwrite(valid_path, sp.eye(2, format="csr"))
    rhs_path = tmp_path / "empty.npy"
    np.save(rhs_path, np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="RHS must not be empty"):
        run_pipeline(
            PipelineConfig(
                matrix=MatrixInputConfig(path=str(valid_path), n=2),
                solve=SolveInputConfig(rhs_path=str(rhs_path)),
                out_dir=tmp_path / "empty-out",
            )
        )


def test_manifest_rejects_overlapping_region(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    manifest["memory_image"]["regions"]["rhs_data"]["offset"] = manifest[
        "memory_image"
    ]["regions"]["permutation_data"]["offset"]
    outputs.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="overlap"):
        validate_manifest(outputs.manifest_path)
