from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.dataStruct import ABI_VERSION, NODE_TASK_BYTE_SIZE, ROOT_PARENT_ID
from src.io import read_map_tables, read_tasks


class ManifestValidationError(ValueError):
    pass


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{label}: expected integer, got {value!r}") from exc


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
    _validate_equilibration(out_dir, manifest)

    node_count = _as_int(manifest["symbolic"]["node_count"], "node_count")
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
    if [task.node_id for task in tasks] != task_order:
        raise ManifestValidationError("tasks.bin order does not match task_order")

    parent = manifest["symbolic"]["parent"]
    if len(parent) != node_count:
        raise ManifestValidationError("parent array length does not match node_count")

    _validate_task_fields(tasks, parent, nodes)
    _validate_forest(parent)
    _validate_quantization(nodes)
    _validate_node_file_ranges(out_dir, manifest)
    _validate_memory_regions(
        manifest,
        _as_int(manifest["config"]["memory"]["alignment"], "alignment"),
    )
    map_tables = _validate_map_tables(out_dir, manifest)
    _validate_memory_image(out_dir, manifest, tasks)

    return ManifestValidationResult(
        node_count=node_count,
        task_count=len(tasks),
        map_table_count=len(map_tables),
    )


def _validate_abi(manifest: dict) -> None:
    abi = manifest.get("abi", {})
    if _as_int(abi.get("version", -1), "abi.version") != ABI_VERSION:
        raise ManifestValidationError("manifest ABI version does not match code")
    if _as_int(abi.get("node_task_byte_size", -1), "node_task_byte_size") != NODE_TASK_BYTE_SIZE:
        raise ManifestValidationError("manifest NodeTask byte size does not match code")
    if abi.get("endianness") != "little":
        raise ManifestValidationError("only little-endian artifacts are supported")


def _validate_output_sizes(out_dir: Path, manifest: dict) -> None:
    for name, expected_size in manifest.get("output_sizes", {}).items():
        path = out_dir / name
        if not path.exists():
            raise ManifestValidationError(f"missing output file: {name}")
        actual_size = path.stat().st_size
        if actual_size != _as_int(expected_size, f"output_size[{name}]"):
            raise ManifestValidationError(
                f"{name} size mismatch: manifest={expected_size}, actual={actual_size}"
            )


def _validate_equilibration(out_dir: Path, manifest: dict) -> None:
    matrix_dim = _as_int(manifest["matrix"]["n"], "matrix.n")
    metadata = manifest.get("equilibration", {})
    mode = metadata.get("mode")
    if mode not in {
        "none", "pow2-row", "pow2-row-column", "pow2-ruiz"
    }:
        raise ManifestValidationError("unsupported equilibration mode")
    expected_equation = (
        "D_r * A * D_c * y = D_r * b; x = D_c * y"
        if mode in {"pow2-row-column", "pow2-ruiz"}
        else "D_r * A * x = D_r * b"
    )
    if metadata.get("equation") != expected_equation:
        raise ManifestValidationError("equilibration equation metadata mismatch")
    expected_unscale = mode in {"pow2-row-column", "pow2-ruiz"}
    if metadata.get("solution_requires_unscale") is not expected_unscale:
        raise ManifestValidationError(
            "equilibration solution-unscale metadata mismatch"
        )
    exponent_count = _as_int(
        metadata.get("row_scale_exponent_count", -1),
        "row_scale_exponent_count",
    )
    if exponent_count != matrix_dim:
        raise ManifestValidationError("row scale exponent count mismatch")
    exponent_path = out_dir / metadata.get("row_scale_exponent_file", "")
    if not exponent_path.exists() or exponent_path.stat().st_size != matrix_dim * 2:
        raise ManifestValidationError("row scale exponent file size mismatch")
    column_exponent_count = _as_int(
        metadata.get("column_scale_exponent_count", -1),
        "column_scale_exponent_count",
    )
    if column_exponent_count != matrix_dim:
        raise ManifestValidationError("column scale exponent count mismatch")
    column_exponent_path = (
        out_dir / metadata.get("column_scale_exponent_file", "")
    )
    if (
        not column_exponent_path.exists()
        or column_exponent_path.stat().st_size != matrix_dim * 2
    ):
        raise ManifestValidationError(
            "column scale exponent file size mismatch"
        )

    verification = manifest.get("verification", {})
    required_sizes = {
        verification.get("original_matrix_reference_file", ""):
            matrix_dim * matrix_dim * 8,
        verification.get("original_rhs_reference_file", ""):
            matrix_dim * 8,
        verification.get("original_solution_reference_file", ""):
            matrix_dim * 8,
    }
    for name, expected_size in required_sizes.items():
        path = out_dir / name
        if not name or not path.exists() or path.stat().st_size != expected_size:
            raise ManifestValidationError(
                f"original-coordinate reference file {name!r} has wrong size"
            )


def _validate_task_fields(tasks, parent: list[int], nodes: dict) -> None:
    by_id = {task.node_id: task for task in tasks}
    for node_id, parent_id in enumerate(parent):
        task = by_id[node_id]
        node = nodes[str(node_id)]
        node_range = node["range"]
        pivot_dim = _as_int(node_range["end"], "range.end") - _as_int(node_range["start"], "range.start")
        total_dim = len(node.get("front_indices", [])) or pivot_dim
        expected_children = sum(candidate == node_id for candidate in parent)
        expected_flags = (1 if expected_children == 0 else 0) | (
            2 if parent_id < 0 else 0
        )

        if task.total_dim != total_dim or task.pivot_dim != pivot_dim:
            raise ManifestValidationError(f"task {node_id} dimensions do not match node range")
        if task.children_count != expected_children or task.flags != expected_flags:
            raise ManifestValidationError(
                f"task {node_id} dependency metadata mismatch"
            )
        if task.reserved or task.reserved_addr0 or task.reserved_addr1:
            raise ManifestValidationError(
                f"task {node_id} reserved ABI fields must be zero"
            )
        expected_tiles = (pivot_dim + 15) // 16
        expected_tail = pivot_dim % 16 or min(pivot_dim, 16)
        if task.tile_count != expected_tiles or task.tail_dim != expected_tail:
            raise ManifestValidationError(f"task {node_id} tile metadata mismatch")
        if task.map_table_bytes != _as_int(
            node["map_table"]["size"], f"node_{node_id}_map_table_size"
        ):
            raise ManifestValidationError(f"task {node_id} map table byte count mismatch")
        address_fields = {
            "front_q": task.front_q_addr,
            "front_e": task.front_e_addr,
            "update_q": task.update_q_addr,
            "update_e": task.update_e_addr,
            "map_table": task.map_table_addr,
            "l_factor": task.l_factor_addr,
            "u_factor": task.u_factor_addr,
            "p_vector": task.p_vector_addr,
            "node_meta": task.node_meta_addr,
            "solve_workspace": task.solve_workspace_addr,
        }
        for region_name, address in address_fields.items():
            expected = _as_int(
                node[region_name]["offset"],
                f"node_{node_id}_{region_name}_offset",
            )
            if address != expected:
                raise ManifestValidationError(
                    f"task {node_id} {region_name} address mismatch"
                )
        if parent_id < 0:
            if task.parent_id != ROOT_PARENT_ID:
                raise ManifestValidationError(f"root task {node_id} has wrong parent sentinel")
        elif task.parent_id != parent_id:
            raise ManifestValidationError(f"task {node_id} parent_id mismatch")


def _validate_forest(parent: list[int]) -> None:
    node_count = len(parent)
    roots = 0
    for node_id, parent_id in enumerate(parent):
        if parent_id < 0:
            roots += 1
        elif parent_id >= node_count or parent_id == node_id:
            raise ManifestValidationError("parent array contains an invalid node id")
    if roots == 0:
        raise ManifestValidationError("elimination forest has no root")
    for origin in range(node_count):
        visited: set[int] = set()
        cursor = origin
        while cursor >= 0:
            if cursor in visited:
                raise ManifestValidationError("elimination forest contains a cycle")
            visited.add(cursor)
            cursor = parent[cursor]


def _validate_node_file_ranges(out_dir: Path, manifest: dict) -> None:
    file_sizes = manifest["output_sizes"]
    for node_id, node in manifest["nodes"].items():
        _check_file_range(
            f"node {node_id} front_q",
            _as_int(node["front_q_file_offset"], f"node_{node_id}_front_q_offset"),
            _as_int(node["front_q"]["size"], f"node_{node_id}_front_q_size"),
            _as_int(file_sizes["front_q.bin"], "front_q.bin_size"),
        )
        _check_file_range(
            f"node {node_id} front_e",
            _as_int(node["front_e_file_offset"], f"node_{node_id}_front_e_offset"),
            _as_int(node["front_e"]["size"], f"node_{node_id}_front_e_size"),
            _as_int(file_sizes["front_e.bin"], "front_e.bin_size"),
        )
        _check_file_range(
            f"node {node_id} map_table",
            _as_int(node["map_table_file_offset"], f"node_{node_id}_map_offset"),
            _as_int(node["map_table"]["size"], f"node_{node_id}_map_size"),
            _as_int(file_sizes["map_table.bin"], "map_table.bin_size"),
        )


def _check_file_range(name: str, offset: int, size: int, file_size: int) -> None:
    if offset < 0 or size < 0 or offset + size > file_size:
        raise ManifestValidationError(f"{name} file range is out of bounds")


def _validate_memory_regions(manifest: dict, alignment: int) -> None:
    nodes = manifest["nodes"]
    ranges: list[tuple[int, int, str]] = []
    global_regions = manifest["memory_image"]["global_regions"]
    matrix_dim = _as_int(manifest["matrix"]["n"], "matrix.n")
    node_count = _as_int(manifest["symbolic"]["node_count"], "node_count")
    expected_global_sizes = {
        "task_queue": node_count * NODE_TASK_BYTE_SIZE,
        "permutation": matrix_dim * 4,
        "rhs_q": matrix_dim * 4,
        "rhs_e": 2,
        "solution_q": matrix_dim * 8,
        "solution_e": node_count * 2,
    }
    if set(global_regions) != set(expected_global_sizes):
        raise ManifestValidationError("global DDR region set is incomplete")
    for region_name, region in global_regions.items():
        offset = _as_int(region["offset"], f"global_{region_name}_offset")
        size = _as_int(region["size"], f"global_{region_name}_size")
        if offset < 0 or size < 0:
            raise ManifestValidationError("global memory region has a negative range")
        if size != expected_global_sizes[region_name]:
            raise ManifestValidationError(
                f"global region {region_name} has the wrong size"
            )
        if offset % alignment != 0:
            raise ManifestValidationError(
                f"global region {region_name} offset is not aligned"
            )
        if size:
            ranges.append((offset, offset + size, f"global {region_name}"))

    for node_id, node in nodes.items():
        for region_name in (
            "front_q",
            "front_e",
            "update_q",
            "update_e",
            "l_factor",
            "u_factor",
            "map_table",
            "p_vector",
            "node_meta",
            "solve_workspace",
        ):
            region = node[region_name]
            offset = _as_int(region["offset"], f"node_{node_id}_{region_name}_offset")
            size = _as_int(region["size"], f"node_{node_id}_{region_name}_size")
            if offset < 0 or size < 0:
                raise ManifestValidationError("node memory region has a negative range")
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
    total_bytes = _as_int(manifest["total_bytes"], "total_bytes")
    if any(start < 0 or end > total_bytes for start, end, _ in ranges):
        raise ManifestValidationError("memory region extends beyond total_bytes")


def _validate_memory_image(out_dir: Path, manifest: dict, tasks) -> None:
    image_path = out_dir / manifest["memory_image"]["file"]
    image = image_path.read_bytes()
    total_bytes = _as_int(manifest["total_bytes"], "total_bytes")
    if len(image) != total_bytes:
        raise ManifestValidationError("memory image size does not match total_bytes")

    task_region = manifest["memory_image"]["global_regions"]["task_queue"]
    task_offset = _as_int(task_region["offset"], "task_queue.offset")
    task_size = _as_int(task_region["size"], "task_queue.size")
    encoded_tasks = b"".join(task.to_bytes() for task in tasks)
    if task_size != len(encoded_tasks):
        raise ManifestValidationError("task queue region size mismatch")
    if image[task_offset : task_offset + task_size] != encoded_tasks:
        raise ManifestValidationError("memory image task queue differs from tasks.bin")

    for node_id, node in manifest["nodes"].items():
        for region_name, file_name, file_offset_name in (
            ("front_q", "front_q.bin", "front_q_file_offset"),
            ("front_e", "front_e.bin", "front_e_file_offset"),
            ("map_table", "map_table.bin", "map_table_file_offset"),
        ):
            region = node[region_name]
            image_offset = _as_int(region["offset"], f"node_{node_id}_{region_name}.offset")
            size = _as_int(region["size"], f"node_{node_id}_{region_name}.size")
            file_offset = _as_int(
                node[file_offset_name], f"node_{node_id}_{file_offset_name}"
            )
            file_data = (out_dir / file_name).read_bytes()
            if image[image_offset : image_offset + size] != file_data[
                file_offset : file_offset + size
            ]:
                raise ManifestValidationError(
                    f"memory image node {node_id} {region_name} differs from {file_name}"
                )


def _validate_map_tables(out_dir: Path, manifest: dict):
    offsets = [
        _as_int(manifest["nodes"][str(node_id)]["map_table_file_offset"], f"node_{node_id}_map_table_offset")
        for node_id in range(_as_int(manifest["symbolic"]["node_count"], "node_count"))
    ]
    map_tables = read_map_tables(str(out_dir / "map_table.bin"), offsets)
    node_count = _as_int(manifest["symbolic"]["node_count"], "node_count")
    parent = manifest["symbolic"]["parent"]
    for parent_id, entries in enumerate(map_tables):
        parent_front = set(manifest["nodes"][str(parent_id)].get("front_indices", []))
        for entry in entries:
            if not (0 <= entry.child_id < node_count):
                raise ManifestValidationError("map_table child_id out of range")
            if len(entry.row_map) != len(entry.col_map):
                raise ManifestValidationError("map_table row/col map lengths differ")
            if parent[entry.child_id] != parent_id:
                raise ManifestValidationError(
                    "map_table child does not belong to the parent node"
                )
            child = manifest["nodes"][str(entry.child_id)]
            child_pivot = (
                _as_int(child["range"]["end"], "child.range.end")
                - _as_int(child["range"]["start"], "child.range.start")
            )
            child_update = len(child["front_indices"]) - child_pivot
            if sorted(entry.row_map) != list(range(child_update)):
                raise ManifestValidationError(
                    "map_table does not cover the complete child update"
                )
            if parent_front:
                parent_size = len(parent_front)
                if any(col < 0 or col >= parent_size for col in entry.col_map):
                    raise ManifestValidationError("map_table parent column index out of range")
                if len(set(entry.col_map)) != child_update:
                    raise ManifestValidationError(
                        "map_table parent column indices are not unique"
                    )
        expected_children = sorted(
            child for child, owner in enumerate(parent) if owner == parent_id
        )
        if sorted(entry.child_id for entry in entries) != expected_children:
            raise ManifestValidationError(
                "map_table entries do not match the parent child list"
            )
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

        front_q_size = _as_int(node["front_q"]["size"], f"node_{node_id}_front_q_size")
        if front_q_size != front_dim * front_dim * 4:
            raise ManifestValidationError(f"node {node_id} front_q size mismatch")
        if _as_int(node["front_e"]["size"], f"node_{node_id}_front_e_size") != 2:
            raise ManifestValidationError(f"node {node_id} front_e must store one int16 exponent")

        node_range = node["range"]
        pivot_dim = _as_int(
            node_range["end"], f"node_{node_id}_range_end"
        ) - _as_int(node_range["start"], f"node_{node_id}_range_start")
        if pivot_dim <= 0 or pivot_dim > front_dim:
            raise ManifestValidationError(
                f"node {node_id} has invalid pivot/front dimensions"
            )
        update_dim = front_dim - pivot_dim
        expected_sizes = {
            "update_q": update_dim * update_dim * 4,
            "update_e": 2 if update_dim else 0,
            "l_factor": front_dim * pivot_dim * 4,
            "u_factor": pivot_dim * front_dim * 4,
            "p_vector": pivot_dim * 2,
            "node_meta": 64,
            "solve_workspace": pivot_dim * 16,
        }
        for region_name, expected_size in expected_sizes.items():
            actual_size = _as_int(
                node[region_name]["size"],
                f"node_{node_id}_{region_name}_size",
            )
            if actual_size != expected_size:
                raise ManifestValidationError(
                    f"node {node_id} {region_name} size mismatch"
                )
