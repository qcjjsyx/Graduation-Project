from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from src.dataStruct import NODE_TASK_BYTE_SIZE, ROOT_PARENT_ID
from src.io import read_map_tables, read_tasks


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestValidationResult:
    node_count: int
    task_count: int
    map_table_count: int


def validate_manifest(manifest_path: str | Path) -> ManifestValidationResult:
    manifest_path = Path(manifest_path)
    out_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _validate_abi(manifest)
    _validate_output_sizes(out_dir, manifest)

    node_count = int(manifest["symbolic"]["node_count"])
    nodes = manifest["nodes"]
    if len(nodes) != node_count:
        raise ManifestValidationError("nodes section length does not match node_count")

    tasks = read_tasks(str(out_dir / "tasks.bin"))
    if len(tasks) != node_count:
        raise ManifestValidationError("tasks.bin record count does not match node_count")

    task_ids = sorted(task.node_id for task in tasks)
    if task_ids != list(range(node_count)):
        raise ManifestValidationError("task node ids are not a dense 0..N-1 range")

    task_order = manifest["task_order"]
    if sorted(task_order) != list(range(node_count)):
        raise ManifestValidationError("task_order is not a permutation of node ids")

    parent = manifest["symbolic"]["parent"]
    if len(parent) != node_count:
        raise ManifestValidationError("parent array length does not match node_count")

    _validate_task_fields(tasks, parent, nodes)
    _validate_quantization(nodes)
    _validate_node_file_ranges(out_dir, manifest)
    _validate_memory_regions(nodes, int(manifest["config"]["memory"]["alignment"]))
    map_tables = _validate_map_tables(out_dir, manifest)

    return ManifestValidationResult(
        node_count=node_count,
        task_count=len(tasks),
        map_table_count=len(map_tables),
    )


def _validate_abi(manifest: dict) -> None:
    abi = manifest.get("abi", {})
    if int(abi.get("node_task_byte_size", -1)) != NODE_TASK_BYTE_SIZE:
        raise ManifestValidationError("manifest NodeTask byte size does not match code")


def _validate_output_sizes(out_dir: Path, manifest: dict) -> None:
    for name, expected_size in manifest.get("output_sizes", {}).items():
        path = out_dir / name
        if not path.exists():
            raise ManifestValidationError(f"missing output file: {name}")
        actual_size = path.stat().st_size
        if actual_size != int(expected_size):
            raise ManifestValidationError(
                f"{name} size mismatch: manifest={expected_size}, actual={actual_size}"
            )


def _validate_task_fields(tasks, parent: list[int], nodes: dict) -> None:
    by_id = {task.node_id: task for task in tasks}
    for node_id, parent_id in enumerate(parent):
        task = by_id[node_id]
        node = nodes[str(node_id)]
        node_range = node["range"]
        pivot_dim = int(node_range["end"]) - int(node_range["start"])
        total_dim = len(node.get("front_indices", [])) or pivot_dim

        if task.total_dim != total_dim or task.pivot_dim != pivot_dim:
            raise ManifestValidationError(f"task {node_id} dimensions do not match node range")
        if parent_id < 0:
            if task.parent_id != ROOT_PARENT_ID:
                raise ManifestValidationError(f"root task {node_id} has wrong parent sentinel")
        elif task.parent_id != parent_id:
            raise ManifestValidationError(f"task {node_id} parent_id mismatch")


def _validate_node_file_ranges(out_dir: Path, manifest: dict) -> None:
    file_sizes = manifest["output_sizes"]
    for node_id, node in manifest["nodes"].items():
        _check_file_range(
            f"node {node_id} front_q",
            int(node["front_q_file_offset"]),
            int(node["front_q"]["size"]),
            int(file_sizes["front_q.bin"]),
        )
        _check_file_range(
            f"node {node_id} front_e",
            int(node["front_e_file_offset"]),
            int(node["front_e"]["size"]),
            int(file_sizes["front_e.bin"]),
        )
        _check_file_range(
            f"node {node_id} map_table",
            int(node["map_table_file_offset"]),
            int(node["map_table"]["size"]),
            int(file_sizes["map_table.bin"]),
        )


def _check_file_range(name: str, offset: int, size: int, file_size: int) -> None:
    if offset < 0 or size < 0 or offset + size > file_size:
        raise ManifestValidationError(f"{name} file range is out of bounds")


def _validate_memory_regions(nodes: dict, alignment: int) -> None:
    ranges: list[tuple[int, int, str]] = []
    for node_id, node in nodes.items():
        for region_name in (
            "front_q",
            "front_e",
            "update_q",
            "update_e",
            "l_factor",
            "u_factor",
            "map_table",
            "task_desc",
        ):
            region = node[region_name]
            offset = int(region["offset"])
            size = int(region["size"])
            if offset % alignment != 0:
                raise ManifestValidationError(
                    f"node {node_id} {region_name} offset is not aligned"
                )
            if size:
                ranges.append((offset, offset + size, f"node {node_id} {region_name}"))

    ranges.sort()
    for prev, curr in zip(ranges, ranges[1:]):
        if prev[1] > curr[0]:
            raise ManifestValidationError(f"memory regions overlap: {prev[2]} and {curr[2]}")


def _validate_map_tables(out_dir: Path, manifest: dict):
    offsets = [
        int(manifest["nodes"][str(node_id)]["map_table_file_offset"])
        for node_id in range(int(manifest["symbolic"]["node_count"]))
    ]
    map_tables = read_map_tables(str(out_dir / "map_table.bin"), offsets)
    node_count = int(manifest["symbolic"]["node_count"])
    for parent_id, entries in enumerate(map_tables):
        parent_front = set(manifest["nodes"][str(parent_id)].get("front_indices", []))
        for entry in entries:
            if not (0 <= entry.child_id < node_count):
                raise ManifestValidationError("map_table child_id out of range")
            if len(entry.row_map) != len(entry.col_map):
                raise ManifestValidationError("map_table row/col map lengths differ")
            if parent_front:
                parent_size = len(parent_front)
                if any(col < 0 or col >= parent_size for col in entry.col_map):
                    raise ManifestValidationError("map_table parent column index out of range")
    return map_tables


def _validate_quantization(nodes: dict) -> None:
    for node_id, node in nodes.items():
        front_dim = len(node.get("front_indices", []))
        local_source = node.get("local_source")
        if not local_source:
            raise ManifestValidationError(f"node {node_id} missing local_source metadata")
        if local_source.get("format") != "S_format":
            raise ManifestValidationError(f"node {node_id} local_source has wrong format")

        shape = local_source.get("shape")
        if shape != [front_dim, front_dim]:
            raise ManifestValidationError(f"node {node_id} local_source shape mismatch")

        front_q_size = int(node["front_q"]["size"])
        if front_q_size != front_dim * front_dim * 4:
            raise ManifestValidationError(f"node {node_id} front_q size mismatch")
        if int(node["front_e"]["size"]) != 2:
            raise ManifestValidationError(f"node {node_id} front_e must store one int16 exponent")
