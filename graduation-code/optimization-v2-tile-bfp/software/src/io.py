from __future__ import annotations

import json
import struct
from typing import Iterable, List, Tuple

from src.dataStruct import NODE_TASK_BYTE_SIZE, MapTableEntry, NodeTask


def write_tasks(path: str, tasks: Iterable[NodeTask]) -> None:
    with open(path, "wb") as f:
        for task in tasks:
            f.write(task.to_bytes())


def read_tasks(path: str) -> List[NodeTask]:
    data = open(path, "rb").read()
    if len(data) % NODE_TASK_BYTE_SIZE != 0:
        raise ValueError(
            f"tasks file size {len(data)} is not a multiple of {NODE_TASK_BYTE_SIZE}"
        )
    return [
        NodeTask.from_bytes(data[offset : offset + NODE_TASK_BYTE_SIZE])
        for offset in range(0, len(data), NODE_TASK_BYTE_SIZE)
    ]


def _encode_map_table(entries: List[MapTableEntry]) -> bytes:
    parts = [struct.pack("<I", len(entries))]
    for entry in entries:
        parts.append(struct.pack("<III", entry.child_id, len(entry.row_map), len(entry.col_map)))
        if entry.row_map:
            parts.append(struct.pack("<" + "I" * len(entry.row_map), *entry.row_map))
        if entry.col_map:
            parts.append(struct.pack("<" + "I" * len(entry.col_map), *entry.col_map))
    return b"".join(parts)


def _decode_map_table(data: bytes, offset: int = 0) -> tuple[List[MapTableEntry], int]:
    cursor = offset
    entry_count, cursor = _unpack_from("<I", data, cursor)
    entries: List[MapTableEntry] = []
    for _ in range(entry_count):
        child_id, row_count, col_count, cursor = _unpack_entry_header(data, cursor)
        row_map, cursor = _unpack_uint32_list(data, cursor, row_count)
        col_map, cursor = _unpack_uint32_list(data, cursor, col_count)
        entries.append(MapTableEntry(child_id=child_id, row_map=row_map, col_map=col_map))
    return entries, cursor


def decode_map_table(data: bytes) -> List[MapTableEntry]:
    entries, cursor = _decode_map_table(data, 0)
    if cursor != len(data):
        raise ValueError(f"map table has {len(data) - cursor} trailing bytes")
    return entries


def read_map_tables(path: str, offsets: List[int]) -> List[List[MapTableEntry]]:
    data = open(path, "rb").read()
    tables: List[List[MapTableEntry]] = []
    for idx, offset in enumerate(offsets):
        next_offset = offsets[idx + 1] if idx + 1 < len(offsets) else len(data)
        if offset < 0 or next_offset < offset or next_offset > len(data):
            raise ValueError(f"invalid map table offset range [{offset}, {next_offset})")
        tables.append(decode_map_table(data[offset:next_offset]))
    return tables


def write_map_table(path: str, map_tables: List[List[MapTableEntry]]) -> List[int]:
    offsets: List[int] = []
    offset = 0
    with open(path, "wb") as f:
        for entries in map_tables:
            offsets.append(offset)
            data = _encode_map_table(entries)
            f.write(data)
            offset += len(data)
    return offsets


def write_front_data(
    path_q: str,
    path_e: str,
    q_values: List[int],
    e_values: List[int],
) -> Tuple[int, int]:
    with open(path_q, "ab") as fq:
        fq.write(struct.pack("<" + "i" * len(q_values), *q_values) if q_values else b"")
    with open(path_e, "ab") as fe:
        fe.write(struct.pack("<" + "h" * len(e_values), *e_values) if e_values else b"")
    return (len(q_values) * 4, len(e_values) * 2)


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _unpack_from(fmt: str, data: bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError("truncated map table data")
    values = struct.unpack_from(fmt, data, offset)
    if len(values) == 1:
        return values[0], offset + size
    return (*values, offset + size)


def _unpack_entry_header(data: bytes, offset: int) -> tuple[int, int, int, int]:
    child_id, row_count, col_count, cursor = _unpack_from("<III", data, offset)
    return int(child_id), int(row_count), int(col_count), cursor


def _unpack_uint32_list(data: bytes, offset: int, count: int) -> tuple[List[int], int]:
    if count == 0:
        return [], offset
    fmt = "<" + ("I" * count)
    unpacked = _unpack_from(fmt, data, offset)
    values = unpacked[:-1]
    cursor = unpacked[-1]
    return [int(value) for value in values], cursor
