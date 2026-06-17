from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar, List, Tuple


ABI_VERSION = 1
ROOT_PARENT_ID = 0xFFFF

_FMT_RANGE = {"H": 0xFFFF, "I": 0xFFFFFFFF, "Q": 0xFFFFFFFFFFFFFFFF}

_NODE_TASK_FIELDS = [
    ("node_id",              "H"),
    ("flags",                "H"),
    ("parent_id",            "H"),
    ("children_count",       "H"),
    ("total_dim",            "I"),
    ("pivot_dim",            "I"),
    ("nums_sub_matrix",      "I"),
    ("last_sub_matrix_size", "I"),
    ("data_addr",            "Q"),
    ("parent_address",       "Q"),
    ("map_table_addr",       "Q"),
    ("l_factor_addr",        "Q"),
    ("u_factor_addr",        "Q"),
    ("p_vector_addr",        "Q"),
    ("reserved",              "H"),
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
    task_desc: MemoryRegion


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
    nums_sub_matrix: int
    last_sub_matrix_size: int
    data_addr: int
    parent_address: int
    map_table_addr: int
    l_factor_addr: int
    u_factor_addr: int
    p_vector_addr: int
    reserved: int = 0

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
            self.nums_sub_matrix,
            self.last_sub_matrix_size,
            self.data_addr,
            self.parent_address,
            self.map_table_addr,
            self.l_factor_addr,
            self.u_factor_addr,
            self.p_vector_addr,
            self.reserved,
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
