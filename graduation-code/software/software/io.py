from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class NodeTask:
    node_id: int
    flags: int
    parent_id: int
    children_count: int
    total_dim: int
    pivot_dim: int
    data_addr: int
    parent_address: int
    map_table_addr: int
    l_factor_addr: int
    u_factor_addr: int
    p_vector_addr: int
    flag: int

    def to_bytes(self) -> bytes:
        # Little-endian, fixed layout to match dataStruct.cpp
        packed = struct.pack(
            NODETASK_PACK_FMT,
            self.node_id,
            self.flags,
            self.parent_id,
            self.children_count,
            self.total_dim,
            self.pivot_dim,
            self.data_addr,
            self.parent_address,
            self.map_table_addr,
            self.l_factor_addr,
            self.u_factor_addr,
            self.p_vector_addr,
            self.flag,
        )
        # Pad to 8-byte alignment
        pad_len = (-len(packed)) % 8
        return packed + (b"\x00" * pad_len)


@dataclass
class MapTableEntry:
    child_id: int
    row_map: List[int]
    col_map: List[int]


# struct.pack format for tasks.bin
NODETASK_PACK_FMT = "<IIIIHHQQQQQQH"


def write_tasks(path: str, tasks: Iterable[NodeTask]) -> None:
    with open(path, "wb") as f:
        for task in tasks:
            f.write(task.to_bytes())


def _encode_map_table(entries: List[MapTableEntry]) -> bytes:
    parts = [struct.pack("<I", len(entries))]
    for entry in entries:
        parts.append(struct.pack("<III", entry.child_id, len(entry.row_map), len(entry.col_map)))
        if entry.row_map:
            parts.append(struct.pack("<" + "I" * len(entry.row_map), *entry.row_map))
        if entry.col_map:
            parts.append(struct.pack("<" + "I" * len(entry.col_map), *entry.col_map))
    return b"".join(parts)


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


def write_front_data(path_q: str, path_e: str, q_tiles: List[int], e_tiles: List[int]) -> Tuple[int, int]:
    with open(path_q, "ab") as fq:
        fq.write(struct.pack("<" + "i" * len(q_tiles), *q_tiles) if q_tiles else b"")
    with open(path_e, "ab") as fe:
        fe.write(struct.pack("<" + "b" * len(e_tiles), *e_tiles) if e_tiles else b"")
    return (len(q_tiles) * 4, len(e_tiles))


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
