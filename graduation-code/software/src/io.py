from __future__ import annotations

import json
import struct
from typing import List

from src.dataStruct import MapTableEntry


def encode_map_entries(entries: List[MapTableEntry]) -> bytes:
    parts = [struct.pack("<I", len(entries))]
    for entry in entries:
        parts.append(
            struct.pack(
                "<III", entry.child_id, len(entry.row_map), len(entry.col_map)
            )
        )
        if entry.row_map:
            parts.append(struct.pack(f"<{len(entry.row_map)}I", *entry.row_map))
        if entry.col_map:
            parts.append(struct.pack(f"<{len(entry.col_map)}I", *entry.col_map))
    return b"".join(parts)


def decode_map_table(data: bytes) -> List[MapTableEntry]:
    entries, cursor = _decode_map_table(data, 0)
    if cursor != len(data):
        raise ValueError(f"map table has {len(data) - cursor} trailing bytes")
    return entries


def read_map_tables(path: str, offsets: List[int]) -> List[List[MapTableEntry]]:
    data = open(path, "rb").read()
    tables: List[List[MapTableEntry]] = []
    for index, offset in enumerate(offsets):
        next_offset = offsets[index + 1] if index + 1 < len(offsets) else len(data)
        if offset < 0 or next_offset < offset or next_offset > len(data):
            raise ValueError(f"invalid map table offset range [{offset}, {next_offset})")
        tables.append(decode_map_table(data[offset:next_offset]))
    return tables


def write_map_table(path: str, map_tables: List[List[MapTableEntry]]) -> List[int]:
    offsets: List[int] = []
    encoded = bytearray()
    for entries in map_tables:
        offsets.append(len(encoded))
        encoded.extend(encode_map_entries(entries))
    with open(path, "wb") as output:
        output.write(encoded)
    return offsets


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")


def _decode_map_table(
    data: bytes, offset: int
) -> tuple[List[MapTableEntry], int]:
    entry_count, cursor = _unpack_from("<I", data, offset)
    entries: List[MapTableEntry] = []
    for _ in range(entry_count):
        child_id, row_count, col_count, cursor = _unpack_from("<III", data, cursor)
        row_map, cursor = _unpack_uint32_list(data, cursor, row_count)
        col_map, cursor = _unpack_uint32_list(data, cursor, col_count)
        entries.append(
            MapTableEntry(
                child_id=int(child_id), row_map=row_map, col_map=col_map
            )
        )
    return entries, cursor


def _unpack_from(fmt: str, data: bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError("truncated map table data")
    values = struct.unpack_from(fmt, data, offset)
    if len(values) == 1:
        return values[0], offset + size
    return (*values, offset + size)


def _unpack_uint32_list(
    data: bytes, offset: int, count: int
) -> tuple[List[int], int]:
    if count == 0:
        return [], offset
    unpacked = _unpack_from(f"<{count}I", data, offset)
    return [int(value) for value in unpacked[:-1]], int(unpacked[-1])
