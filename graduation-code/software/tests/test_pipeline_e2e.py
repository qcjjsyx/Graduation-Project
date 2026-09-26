import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from src.command_codec import COMMAND_RECORD_BYTES, DESCRIPTOR_RECORD_BYTES
from src.config import (
    MatchingConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
    PivotPolicyConfig,
    SolveInputConfig,
)
from src.pipeline import run_pipeline
from src.verify.front_reference import PivotFailureError, PivotFailureKind
from src.verify.manifest import ManifestValidationError, validate_manifest

ARTIFACT_FILES = (
    "manifest.json",
    "memory_image.bin",
    "front_f64.bin",
    "rhs_hat_f64.bin",
    "original_matrix_f64.bin",
    "original_rhs_f64.bin",
    "row_scale_e.bin",
    "x_true_f64.bin",
    "x_hat_f64.bin",
)


def test_pipeline_emits_command_v2_artifact(tmp_path: Path):
    config = PipelineConfig(
        matrix=MatrixInputConfig(path=None, n=12, density=0.25, seed=4),
        ordering=OrderingConfig(method="amd"),
        out_dir=tmp_path,
    )
    outputs = run_pipeline(config)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    result = validate_manifest(outputs.manifest_path)

    assert manifest["abi"]["version"] == 2
    assert manifest["abi"]["command_record_bytes"] == COMMAND_RECORD_BYTES
    assert manifest["abi"]["descriptor_record_bytes"] == DESCRIPTOR_RECORD_BYTES
    assert manifest["compiler"]["device_format"] == "FP32"
    assert manifest["compiler"]["global_bfp"] is False
    assert manifest["pivot"]["policy"] == "STATIC"
    assert manifest["pivot"]["threshold_scale"] == "front_pre_elimination_max_abs"
    assert manifest["pivot"]["p_vector"]["initial_value"] == -1
    assert manifest["preprocess"]["matching"]["method"] == "weighted_matching"
    assert manifest["preprocess"]["equations"].startswith("B = P_match * D_r * A")
    assert manifest["required_device_capabilities"]["STATIC"][
        "hpu_atu_dynamic_mapping"
    ] == "bypassed"
    assert manifest["verification"]["device_memory_contains_fp64_reference"] is False
    assert manifest["verification"]["residual_target"] == "original_A_x_equals_b"
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
        reference_check=True,
        out_dir=tmp_path / "first",
    )
    first = run_pipeline(base)
    second = run_pipeline(replace(base, out_dir=tmp_path / "second"))
    for filename in ARTIFACT_FILES:
        assert (first.out_dir / filename).read_bytes() == (
            second.out_dir / filename
        ).read_bytes()


def test_dynamic_panel_policy_is_expressed_end_to_end(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=None, n=10, density=0.3, seed=9),
            pivot=PivotPolicyConfig(policy="DYNAMIC_PANEL", threshold_ratio=0.125),
            out_dir=tmp_path,
        )
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pivot"]["policy"] == "DYNAMIC_PANEL"
    assert manifest["required_device_capabilities"]["DYNAMIC_PANEL"][
        "hpu_atu_dynamic_mapping"
    ] == "required"
    assert manifest["required_device_capabilities"]["DYNAMIC_PANEL"][
        "cross_front_pivot"
    ] is False
    validate_manifest(outputs.manifest_path)


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


def test_compile_survives_numeric_failure_unless_reference_check(tmp_path: Path):
    dense = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    matrix_path = tmp_path / "trap.mtx"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix(dense))
    config = PipelineConfig(
        matrix=MatrixInputConfig(path=str(matrix_path), n=3),
        ordering=OrderingConfig(method="identity"),
        matching=MatchingConfig(method="none"),
        pivot=PivotPolicyConfig(policy="STATIC"),
        out_dir=tmp_path / "static-out",
    )
    outputs = run_pipeline(config)
    assert outputs.command_count > 0
    assert outputs.reference_check_performed is False
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["verification"]["host_reference"]["performed"] is False
    assert manifest["verification"]["host_reference"]["x_hat_reference_file"] is None
    assert not (outputs.out_dir / "x_hat_f64.bin").exists()
    validate_manifest(outputs.manifest_path)


def test_reference_check_rejects_failed_fixed_order_without_artifacts(tmp_path: Path):
    dense = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    matrix_path = tmp_path / "trap.mtx"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix(dense))
    out_dir = tmp_path / "checked-out"
    with pytest.raises(PivotFailureError) as failure:
        run_pipeline(
            PipelineConfig(
                matrix=MatrixInputConfig(path=str(matrix_path), n=3),
                ordering=OrderingConfig(method="identity"),
                matching=MatchingConfig(method="none"),
                pivot=PivotPolicyConfig(policy="STATIC"),
                reference_check=True,
                out_dir=out_dir,
            )
        )
    assert failure.value.kind == PivotFailureKind.ZERO_PIVOT
    assert not out_dir.exists()

    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=str(matrix_path), n=3),
            ordering=OrderingConfig(method="identity"),
            matching=MatchingConfig(method="none"),
            pivot=PivotPolicyConfig(policy="DYNAMIC_PANEL"),
            reference_check=True,
            out_dir=out_dir,
        )
    )
    assert outputs.command_count > 0
    assert outputs.reference_check_performed is True


def test_matching_preprocessing_rescues_static_order(tmp_path: Path):
    swap_only = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrix_path = tmp_path / "swap.mtx"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix(swap_only))
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=str(matrix_path), n=2),
            ordering=OrderingConfig(method="identity"),
            matching=MatchingConfig(method="weighted_matching"),
            pivot=PivotPolicyConfig(policy="STATIC"),
            reference_check=True,
            out_dir=tmp_path / "matched",
        )
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["preprocess"]["matching"]["row_perm"] == [1, 0]
    assert (
        manifest["verification"]["host_reference"]["original_reference_residual_norm"]
        < 1e-10
    )


def _corrupt_manifest(outputs, mutate):
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    outputs.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_rejects_overlapping_region(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )

    def mutate(manifest):
        manifest["memory_image"]["regions"]["rhs_data"]["offset"] = manifest[
            "memory_image"
        ]["regions"]["permutation_data"]["offset"]

    _corrupt_manifest(outputs, mutate)
    with pytest.raises(ManifestValidationError, match="overlap"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_wrong_schema_version(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    _corrupt_manifest(outputs, lambda manifest: manifest["abi"].update(version=1))
    with pytest.raises(ManifestValidationError, match="schema version"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_policy_mismatch_and_unknown_policy(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    _corrupt_manifest(
        outputs,
        lambda manifest: manifest["pivot"].update(policy="DYNAMIC_PANEL"),
    )
    with pytest.raises(ManifestValidationError, match="disagrees with the manifest"):
        validate_manifest(outputs.manifest_path)

    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path / "second")
    )
    _corrupt_manifest(
        outputs, lambda manifest: manifest["pivot"].update(policy="ADAPTIVE")
    )
    with pytest.raises(ManifestValidationError, match="unknown pivot policy"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_inconsistent_permutation_maps(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )

    def mutate(manifest):
        transform = manifest["preprocess"]["transform"]
        transform["rhs_row_map"] = list(reversed(transform["rhs_row_map"]))

    _corrupt_manifest(outputs, mutate)
    with pytest.raises(ManifestValidationError, match="rhs_row_map is inconsistent"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_zero_filled_p_vector_and_v1_front_descriptor(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    image_path = outputs.memory_image_path
    original = image_path.read_bytes()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    name = "node.0.p_vector"
    region = manifest["memory_image"]["regions"][name]
    patched = bytearray(original)
    patched[region["offset"] : region["offset"] + region["size"]] = bytes(
        region["size"]
    )
    image_path.write_bytes(bytes(patched))
    with pytest.raises(ManifestValidationError, match="sentinel"):
        validate_manifest(outputs.manifest_path)

    image_path.write_bytes(original)
    descriptor_table = manifest["memory_image"]["regions"]["descriptor_table"]
    front = None
    for index in range(descriptor_table["size"] // DESCRIPTOR_RECORD_BYTES):
        offset = descriptor_table["offset"] + index * DESCRIPTOR_RECORD_BYTES
        record = original[offset : offset + DESCRIPTOR_RECORD_BYTES]
        if struct.unpack_from("<H", record, 0)[0] == 0x02:
            front = offset
            break
    assert front is not None
    patched = bytearray(original)
    struct.pack_into("<2I", patched, front + 56, 0, 0)
    image_path.write_bytes(bytes(patched))
    with pytest.raises(ValueError, match="unknown pivot policy"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_none_p_vector_reference(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    original = outputs.memory_image_path.read_bytes()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    table = manifest["memory_image"]["regions"]["descriptor_table"]
    patched = bytearray(original)
    for index in range(table["size"] // DESCRIPTOR_RECORD_BYTES):
        offset = table["offset"] + index * DESCRIPTOR_RECORD_BYTES
        if struct.unpack_from("<H", patched, offset)[0] == 0x02:
            struct.pack_into("<I", patched, offset + 24 + 6 * 4, 0xFFFFFFFF)
            break
    outputs.memory_image_path.write_bytes(bytes(patched))
    with pytest.raises(ManifestValidationError, match="P-vector must not be NONE"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_p_vector_reference_to_wrong_region(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    original = outputs.memory_image_path.read_bytes()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    table = manifest["memory_image"]["regions"]["descriptor_table"]
    rhs_desc = None
    front_offset = None
    for index in range(table["size"] // DESCRIPTOR_RECORD_BYTES):
        offset = table["offset"] + index * DESCRIPTOR_RECORD_BYTES
        kind = struct.unpack_from("<H", original, offset)[0]
        body = struct.unpack_from("<10I", original, offset + 24)
        if kind == 0x01 and body[7] == 0x02:
            rhs_desc = index
        if kind == 0x02 and front_offset is None:
            front_offset = offset
    assert rhs_desc is not None and front_offset is not None

    patched = bytearray(original)
    struct.pack_into("<I", patched, front_offset + 24 + 6 * 4, rhs_desc)
    outputs.memory_image_path.write_bytes(bytes(patched))
    with pytest.raises(ManifestValidationError, match="INT32"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_inconsistent_p_vector_across_factor_and_solve(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    original = outputs.memory_image_path.read_bytes()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    table = manifest["memory_image"]["regions"]["descriptor_table"]
    factor_offsets = []
    solve_offsets = []
    for index in range(table["size"] // DESCRIPTOR_RECORD_BYTES):
        offset = table["offset"] + index * DESCRIPTOR_RECORD_BYTES
        kind = struct.unpack_from("<H", original, offset)[0]
        if kind == 0x04:
            factor_offsets.append(offset)
        elif kind == 0x06:
            solve_offsets.append(offset)
    assert factor_offsets and solve_offsets

    patched = bytearray(original)
    struct.pack_into("<I", patched, factor_offsets[0] + 24 + 2 * 4, 0xFFFFFFFF)
    outputs.memory_image_path.write_bytes(bytes(patched))
    with pytest.raises(ManifestValidationError, match="P-vector must not be NONE"):
        validate_manifest(outputs.manifest_path)

    patched = bytearray(original)
    struct.pack_into("<I", patched, solve_offsets[0] + 24 + 3 * 4, factor_offsets[0])
    outputs.memory_image_path.write_bytes(bytes(patched))
    with pytest.raises(ManifestValidationError, match="P-vector"):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_p_vector_region_with_wrong_capacity(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(matrix=MatrixInputConfig(n=6), out_dir=tmp_path)
    )
    original = outputs.memory_image_path.read_bytes()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    table = manifest["memory_image"]["regions"]["descriptor_table"]
    patched = bytearray(original)
    for index in range(table["size"] // DESCRIPTOR_RECORD_BYTES):
        offset = table["offset"] + index * DESCRIPTOR_RECORD_BYTES
        if struct.unpack_from("<H", patched, offset)[0] != 0x01:
            continue
        body = list(struct.unpack_from("<10I", patched, offset + 24))
        if body[7] != 0x03:
            continue
        body[6] = body[6] + 1
        struct.pack_into("<10I", patched, offset + 24, *body)
        break
    outputs.memory_image_path.write_bytes(bytes(patched))
    with pytest.raises(ManifestValidationError):
        validate_manifest(outputs.manifest_path)


def test_manifest_rejects_threshold_ratio_disagreeing_with_bits(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(n=6),
            pivot=PivotPolicyConfig(policy="STATIC", threshold_ratio=0.1),
            out_dir=tmp_path,
        )
    )
    _corrupt_manifest(
        outputs,
        lambda manifest: manifest["pivot"].update(threshold_ratio=0.10000001),
    )
    with pytest.raises(ManifestValidationError, match="threshold_ratio"):
        validate_manifest(outputs.manifest_path)


def test_matching_none_manifest_is_strict_json(tmp_path: Path):
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(n=6),
            matching=MatchingConfig(method="none"),
            out_dir=tmp_path,
        )
    )
    text = outputs.manifest_path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text

    def reject_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    manifest = json.loads(text, parse_constant=reject_constant)
    assert manifest["preprocess"]["matching"]["method"] == "none"
    assert manifest["preprocess"]["matching"]["log_product"] is None
    validate_manifest(outputs.manifest_path)


def test_external_rhs_without_reference_check_never_solves(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("spsolve must not be called during plain compilation")

    monkeypatch.setattr("scipy.sparse.linalg.spsolve", forbidden)
    matrix_path = tmp_path / "square.mtx"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix(np.eye(4) * 2.0 + np.eye(4, k=1)))
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, np.arange(1, 5, dtype=np.float64))
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=str(matrix_path), n=4),
            solve=SolveInputConfig(rhs_path=str(rhs_path)),
            reference_check=False,
            out_dir=tmp_path / "plain",
        )
    )
    assert outputs.command_count > 0
    assert outputs.solution_reference_path is None
    assert outputs.residual_norm is None
    assert outputs.x_hat_reference_path is None
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["verification"]["solution_reference_file"] is None
    assert manifest["verification"]["solution_reference_space"] is None
    assert manifest["verification"]["reference_residual_norm"] is None
    assert manifest["verification"]["host_reference"]["performed"] is False
    validate_manifest(outputs.manifest_path)


def test_external_rhs_reference_check_produces_golden_solution(tmp_path):
    matrix_path = tmp_path / "square.mtx"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix(np.eye(4) * 2.0 + np.eye(4, k=1)))
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, np.arange(1, 5, dtype=np.float64))
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=str(matrix_path), n=4),
            solve=SolveInputConfig(rhs_path=str(rhs_path)),
            reference_check=True,
            out_dir=tmp_path / "checked",
        )
    )
    assert outputs.solution_reference_path is not None
    assert outputs.solution_reference_path.stat().st_size == 4 * 8
    assert outputs.x_hat_reference_path is not None
    assert outputs.original_residual_norm is not None
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["verification"]["solution_reference_file"] == "x_true_f64.bin"
    assert manifest["verification"]["host_reference"]["performed"] is True
    validate_manifest(outputs.manifest_path)
