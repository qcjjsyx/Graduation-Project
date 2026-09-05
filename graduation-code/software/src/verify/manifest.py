from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.command_codec import (
    COMMAND_RECORD_BYTES,
    COMPLETION_RECORD_BYTES,
    DESCRIPTOR_RECORD_BYTES,
    DataFormat,
    DescriptorType,
    Opcode,
    StatusCode,
    decode_commands,
    decode_completions,
    decode_descriptors,
    validate_command_batch,
)
from src.dataStruct import MemoryRegion
from src.memory.planner import validate_region_layout
from src.scheduler.task_queue import sibling_friendly_order


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestValidationResult:
    node_count: int
    command_count: int
    descriptor_count: int
    completion_count: int


def validate_manifest(manifest_path: str | Path) -> ManifestValidationResult:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _validate_abi(manifest)
        total_bytes = _as_int(manifest["total_bytes"], "total_bytes")
        image_meta = manifest["memory_image"]
        image_path = path.parent / image_meta["file"]
        image = image_path.read_bytes()
        if len(image) != total_bytes or _as_int(image_meta["size"], "image.size") != total_bytes:
            raise ManifestValidationError("memory image size does not match manifest")

        alignment = _as_int(manifest["compiler"]["alignment"], "alignment")
        regions = {
            name: MemoryRegion(
                offset=_as_int(value["offset"], f"{name}.offset"),
                size=_as_int(value["size"], f"{name}.size"),
            )
            for name, value in image_meta["regions"].items()
        }
        validate_region_layout(regions, total_bytes, alignment)
        for required in (
            "command_buffer",
            "descriptor_table",
            "descriptor_payload",
            "completion_status",
            "permutation_data",
            "rhs_data",
            "solution_data",
        ):
            if required not in regions:
                raise ManifestValidationError(f"missing memory region: {required}")

        batch = manifest["command_batch"]
        table = manifest["descriptor_table"]
        completion_meta = manifest["completion_queue"]
        command_count = _as_int(batch["command_count"], "command_count")
        descriptor_count = _as_int(table["descriptor_count"], "descriptor_count")
        completion_count = _as_int(completion_meta["slot_count"], "slot_count")
        if command_count <= 0 or descriptor_count <= 0:
            raise ManifestValidationError("command and descriptor counts must be positive")
        if completion_count != command_count:
            raise ManifestValidationError("completion slot count must equal command count")
        _require_region_size(
            regions, "command_buffer", command_count * COMMAND_RECORD_BYTES
        )
        _require_region_size(
            regions, "descriptor_table", descriptor_count * DESCRIPTOR_RECORD_BYTES
        )
        _require_region_size(
            regions, "completion_status", completion_count * COMPLETION_RECORD_BYTES
        )

        commands = decode_commands(_slice(image, regions["command_buffer"]))
        descriptors = decode_descriptors(
            _slice(image, regions["descriptor_table"]),
            memory_image_bytes=total_bytes,
        )
        completions = decode_completions(_slice(image, regions["completion_status"]))
        if len(commands) != command_count or len(descriptors) != descriptor_count:
            raise ManifestValidationError("decoded record count mismatch")
        token_count = _as_int(batch["token_count"], "token_count")
        validate_command_batch(
            commands,
            descriptors,
            image,
            token_count=token_count,
            max_wait_tokens=_as_int(batch["max_wait_tokens"], "max_wait_tokens"),
        )
        _validate_command_semantics(commands, descriptors, image, token_count)
        _validate_completion_templates(commands, completions)
        node_count = _validate_symbolic_and_nodes(manifest, commands, regions)
        _validate_descriptor_references(descriptors)
        _validate_device_precision(manifest, descriptors)
        _validate_reference_files(path.parent, manifest)
        return ManifestValidationResult(
            node_count=node_count,
            command_count=command_count,
            descriptor_count=descriptor_count,
            completion_count=completion_count,
        )
    except ManifestValidationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(str(exc)) from exc


def _validate_abi(manifest: dict) -> None:
    abi = manifest.get("abi", {})
    if abi.get("name") != "command_descriptor" or _as_int(
        abi.get("version", -1), "abi.version"
    ) != 1:
        raise ManifestValidationError("manifest is not Command/Descriptor v1")
    if abi.get("endianness") != "little":
        raise ManifestValidationError("only little-endian artifacts are supported")
    expected = {
        "command_record_bytes": COMMAND_RECORD_BYTES,
        "descriptor_record_bytes": DESCRIPTOR_RECORD_BYTES,
        "completion_record_bytes": COMPLETION_RECORD_BYTES,
    }
    for name, value in expected.items():
        if _as_int(abi.get(name, -1), f"abi.{name}") != value:
            raise ManifestValidationError(f"manifest {name} mismatch")


def _validate_completion_templates(commands, completions) -> None:
    for slot, (command, completion) in enumerate(zip(commands, completions)):
        if completion.command_id != command.command_id or completion.node_id != command.node_id:
            raise ManifestValidationError(
                f"completion template {slot} does not match its command"
            )
        if completion.status_code != StatusCode.OK:
            raise ManifestValidationError("completion template status must be OK")
        if any(
            (
                completion.pivot_count,
                completion.start_cycle,
                completion.finish_cycle,
                completion.read_bytes,
                completion.write_bytes,
                completion.stall_cycles,
                completion.overflow_count,
                completion.retry_count,
            )
        ):
            raise ManifestValidationError("completion template counters must start at zero")


def _validate_command_semantics(commands, descriptors, image: bytes, token_count: int) -> None:
    expected_descriptor_types = {
        Opcode.NODE_BEGIN: {DescriptorType.FRONT_DESC},
        Opcode.LOAD_FRONT: {DescriptorType.FRONT_DESC},
        Opcode.ASSEMBLE_EXTEND_ADD: {DescriptorType.CONTRIBUTION_DESC},
        Opcode.PANEL_LU: {DescriptorType.KERNEL_DESC},
        Opcode.TRSM_LEFT: {DescriptorType.KERNEL_DESC},
        Opcode.TRSM_RIGHT: {DescriptorType.KERNEL_DESC},
        Opcode.GEMM_SCHUR: {DescriptorType.KERNEL_DESC},
        Opcode.STORE_FACTOR: {DescriptorType.FACTOR_DESC},
        Opcode.STORE_UPDATE: {DescriptorType.REGION_DESC},
        Opcode.SOLVE_FORWARD: {DescriptorType.SOLVE_DESC},
        Opcode.SOLVE_BACKWARD: {DescriptorType.SOLVE_DESC},
        Opcode.NODE_COMMIT: {DescriptorType.FRONT_DESC},
    }
    signals = {command.signal_token for command in commands}
    if signals != set(range(token_count)):
        raise ManifestValidationError("T03 requires one dense signal token per command")
    for command in commands:
        allowed = expected_descriptor_types.get(Opcode(command.opcode))
        if allowed is not None:
            if command.descriptor_id == 0xFFFFFFFF:
                raise ManifestValidationError(
                    f"command {command.command_id} is missing its descriptor"
                )
            actual = DescriptorType(descriptors[command.descriptor_id].descriptor_type)
            if actual not in allowed:
                raise ManifestValidationError(
                    f"command {command.command_id} uses the wrong descriptor type"
                )
        if command.wait_list_id != 0xFFFFFFFF:
            waits = descriptors[command.wait_list_id].dependency_tokens(
                image, token_count=token_count
            )
            if any(token >= command.command_id for token in waits):
                raise ManifestValidationError(
                    f"command {command.command_id} waits on a non-preceding token"
                )


def _validate_symbolic_and_nodes(manifest, commands, regions) -> int:
    symbolic = manifest["symbolic"]
    if symbolic.get("pattern_source") != "union_of_A_and_transpose_nonzero_patterns":
        raise ManifestValidationError("unsupported symbolic pattern source")
    node_count = _as_int(symbolic["node_count"], "node_count")
    nodes = manifest["nodes"]
    if node_count <= 0 or set(nodes) != {str(index) for index in range(node_count)}:
        raise ManifestValidationError("node IDs must form a dense unique 0..N-1 range")
    parent = [_as_int(value, "parent") for value in symbolic["parent"]]
    if len(parent) != node_count:
        raise ManifestValidationError("parent array length mismatch")
    try:
        expected_order = sibling_friendly_order(parent)
    except ValueError as exc:
        raise ManifestValidationError(f"invalid elimination forest: {exc}") from exc
    if symbolic["factorization_order"] != expected_order:
        raise ManifestValidationError("factorization order is inconsistent with forest")

    seen_commands: set[int] = set()
    max_front = _as_int(manifest["compiler"]["max_front_size"], "max_front_size")
    for node_id in range(node_count):
        node = nodes[str(node_id)]
        if _as_int(node["parent_id"], "parent_id") != parent[node_id]:
            raise ManifestValidationError(f"node {node_id} parent mismatch")
        total = _as_int(node["total_dim"], "total_dim")
        pivot = _as_int(node["pivot_dim"], "pivot_dim")
        update = _as_int(node["update_dim"], "update_dim")
        if pivot <= 0 or total != pivot + update or total > max_front:
            raise ManifestValidationError(f"node {node_id} dimensions are invalid")
        for region_name in node["regions"].values():
            if region_name not in regions:
                raise ManifestValidationError(
                    f"node {node_id} references unknown region {region_name}"
                )
        command_ids = [_as_int(value, "command_id") for value in node["command_ids"]]
        if seen_commands.intersection(command_ids):
            raise ManifestValidationError("a command is owned by multiple nodes")
        seen_commands.update(command_ids)
        if any(command_id >= len(commands) for command_id in command_ids):
            raise ManifestValidationError("node references an unknown command")
        if any(commands[command_id].node_id != node_id for command_id in command_ids):
            raise ManifestValidationError("command node ID disagrees with manifest")
        opcodes = {commands[command_id].opcode for command_id in command_ids}
        for required in (
            Opcode.NODE_BEGIN,
            Opcode.LOAD_FRONT,
            Opcode.PANEL_LU,
            Opcode.STORE_FACTOR,
            Opcode.NODE_COMMIT,
            Opcode.SOLVE_FORWARD,
            Opcode.SOLVE_BACKWARD,
        ):
            if required not in opcodes:
                raise ManifestValidationError(
                    f"node {node_id} is missing command {required.name}"
                )
    if seen_commands != set(range(len(commands))):
        raise ManifestValidationError("manifest does not assign every command to a node")
    return node_count


def _validate_descriptor_references(descriptors) -> None:
    count = len(descriptors)

    def require(index: int, allowed: set[DescriptorType], label: str) -> None:
        if index == 0xFFFFFFFF:
            return
        if index >= count or DescriptorType(descriptors[index].descriptor_type) not in allowed:
            raise ManifestValidationError(f"{label} has invalid descriptor reference")

    for descriptor in descriptors:
        kind = DescriptorType(descriptor.descriptor_type)
        body = descriptor.body_words
        if kind == DescriptorType.FRONT_DESC:
            require(body[0], {DescriptorType.REGION_DESC}, "front region")
            require(body[4], {DescriptorType.CONTRIBUTION_DESC}, "front contribution")
            require(body[5], {DescriptorType.FACTOR_DESC}, "front factor")
            require(body[6], {DescriptorType.REGION_DESC}, "front P-vector")
            require(body[7], {DescriptorType.REGION_DESC}, "front workspace")
        elif kind == DescriptorType.CONTRIBUTION_DESC:
            require(body[0], {DescriptorType.REGION_DESC}, "contribution source")
            require(body[1], {DescriptorType.REGION_DESC}, "contribution target")
        elif kind == DescriptorType.FACTOR_DESC:
            for index in (0, 1, 2, 6):
                require(body[index], {DescriptorType.REGION_DESC}, "factor region")
        elif kind == DescriptorType.KERNEL_DESC:
            for index in (0, 1, 2):
                require(body[index], {DescriptorType.FRONT_DESC}, "kernel operand")
            require(body[8], {DescriptorType.SCALE_DESC}, "kernel scale")
        elif kind == DescriptorType.SOLVE_DESC:
            require(body[0], {DescriptorType.FACTOR_DESC}, "solve factor")
            for index in (1, 2, 3, 4):
                require(body[index], {DescriptorType.REGION_DESC}, "solve region")


def _validate_device_precision(manifest, descriptors) -> None:
    compiler = manifest["compiler"]
    if compiler.get("device_format") != "FP32" or compiler.get("global_bfp") is not False:
        raise ManifestValidationError("Command v1 main path must be FP32 without global BFP")
    for descriptor in descriptors:
        if descriptor.descriptor_type == DescriptorType.REGION_DESC:
            data_format = DataFormat(descriptor.body_words[7])
            if data_format not in {DataFormat.FP32, DataFormat.INT32}:
                raise ManifestValidationError(
                    "device image contains a non-current data format"
                )


def _validate_reference_files(out_dir: Path, manifest: dict) -> None:
    verification = manifest.get("verification")
    if verification is None:
        return
    if verification.get("device_memory_contains_fp64_reference") is not False:
        raise ManifestValidationError("FP64 reference must not be device input")
    matrix_dim = _as_int(manifest["matrix"]["n"], "matrix.n")
    expected_sizes = {
        verification["rhs_reference_file"]: matrix_dim * 8,
        verification["original_matrix_reference_file"]: matrix_dim * matrix_dim * 8,
        verification["original_rhs_reference_file"]: matrix_dim * 8,
        verification["row_scale_exponent_file"]: matrix_dim * 2,
        verification["solution_reference_file"]: matrix_dim * 8,
    }
    for name, expected in expected_sizes.items():
        if (out_dir / name).stat().st_size != expected:
            raise ManifestValidationError(f"reference file {name} has wrong size")


def _require_region_size(regions, name: str, expected: int) -> None:
    if regions[name].size != expected:
        raise ManifestValidationError(f"{name} has wrong size")


def _slice(image: bytes, region: MemoryRegion) -> bytes:
    return image[region.offset : region.offset + region.size]


def _as_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestValidationError(f"{label}: expected integer")
    return value
