from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar, List, Tuple


ABI_VERSION = 2
ROOT_PARENT_ID = 0xFFFF

_FMT_RANGE = {"H": 0xFFFF, "I": 0xFFFFFFFF, "Q": 0xFFFFFFFFFFFFFFFF}

_NODE_TASK_FIELDS = [
    ("node_id",              "H"),
    ("flags",                "H"),
    ("parent_id",            "H"),
    ("children_count",       "H"),
    ("total_dim",            "I"),
    ("pivot_dim",            "I"),
    ("tile_count",           "I"),
    ("tail_dim",             "I"),
    ("map_table_bytes",      "I"),
    ("reserved",             "I"),
    ("front_q_addr",         "Q"),
    ("front_e_addr",         "Q"),
    ("update_q_addr",        "Q"),
    ("update_e_addr",        "Q"),
    ("map_table_addr",       "Q"),
    ("l_factor_addr",        "Q"),
    ("u_factor_addr",        "Q"),
    ("p_vector_addr",        "Q"),
    ("node_meta_addr",       "Q"),
    ("solve_workspace_addr", "Q"),
    ("reserved_addr0",       "Q"),
    ("reserved_addr1",       "Q"),
]

NODE_TASK_PACK_FMT = "<" + "".join(fmt for _, fmt in _NODE_TASK_FIELDS)
NODE_TASK_PACKED_SIZE = struct.calcsize(NODE_TASK_PACK_FMT)
NODE_TASK_BYTE_SIZE = (NODE_TASK_PACKED_SIZE + 7) // 8 * 8


def _check_uint(name: str, value: int, max_val: int) -> None:
    if not (0 <= value <= max_val):
        raise ValueError(f"{name}={value:#x} out of range [0, {max_val:#x}]")


@dataclass(frozen=True)
class MemoryRegion:
    offset: int
    size: int


@dataclass(frozen=True)
class NodeMemoryPlan:
    front_q: MemoryRegion
    front_e: MemoryRegion
    update_q: MemoryRegion
    update_e: MemoryRegion
    l_factor: MemoryRegion
    u_factor: MemoryRegion
    map_table: MemoryRegion
    p_vector: MemoryRegion
    node_meta: MemoryRegion
    solve_workspace: MemoryRegion


@dataclass(frozen=True)
class GlobalMemoryPlan:
    task_queue: MemoryRegion
    permutation: MemoryRegion
    rhs_q: MemoryRegion
    rhs_e: MemoryRegion
    solution_q: MemoryRegion
    solution_e: MemoryRegion


@dataclass(frozen=True)
class NodeRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True)
class NodeTask:
    node_id: int
    flags: int
    parent_id: int
    children_count: int
    total_dim: int
    pivot_dim: int
    tile_count: int
    tail_dim: int
    map_table_bytes: int
    reserved: int
    front_q_addr: int
    front_e_addr: int
    update_q_addr: int
    update_e_addr: int
    map_table_addr: int
    l_factor_addr: int
    u_factor_addr: int
    p_vector_addr: int
    node_meta_addr: int
    solve_workspace_addr: int
    reserved_addr0: int = 0
    reserved_addr1: int = 0

    PACK_FMT: ClassVar[str] = NODE_TASK_PACK_FMT
    PACKED_SIZE: ClassVar[int] = NODE_TASK_PACKED_SIZE
    BYTE_SIZE: ClassVar[int] = NODE_TASK_BYTE_SIZE

    def __post_init__(self) -> None:
        for name, fmt in _NODE_TASK_FIELDS:
            _check_uint(name, getattr(self, name), _FMT_RANGE[fmt])

    def to_bytes(self) -> bytes:
        packed = struct.pack(
            self.PACK_FMT,
            self.node_id,
            self.flags,
            self.parent_id,
            self.children_count,
            self.total_dim,
            self.pivot_dim,
            self.tile_count,
            self.tail_dim,
            self.map_table_bytes,
            self.reserved,
            self.front_q_addr,
            self.front_e_addr,
            self.update_q_addr,
            self.update_e_addr,
            self.map_table_addr,
            self.l_factor_addr,
            self.u_factor_addr,
            self.p_vector_addr,
            self.node_meta_addr,
            self.solve_workspace_addr,
            self.reserved_addr0,
            self.reserved_addr1,
        )
        return packed + (b"\x00" * (self.BYTE_SIZE - len(packed)))

    @classmethod
    def from_bytes(cls, data: bytes) -> "NodeTask":
        if len(data) < cls.PACKED_SIZE:
            raise ValueError(f"expected at least {cls.PACKED_SIZE} bytes")
        values = struct.unpack(cls.PACK_FMT, data[: cls.PACKED_SIZE])
        return cls(*values)


@dataclass(frozen=True)
class MapTableEntry:
    child_id: int
    row_map: List[int]
    col_map: List[int]


@dataclass(frozen=True)
class SymbolicResult:
    permutation: List[int]
    parent: List[int]
    supernodes: List[List[int]]
    node_ranges: List[NodeRange]
    front_indices: List[List[int]]
