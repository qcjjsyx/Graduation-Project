from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class MemoryRegion:
    offset: int
    size: int


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


@dataclass(frozen=True)
class NodeCompileRecord:
    """Current compiler IR; it is not a device-visible ABI record."""

    node_id: int
    parent_id: int
    node_range: NodeRange
    front_indices: Tuple[int, ...]

    @property
    def pivot_dim(self) -> int:
        return self.node_range.size

    @property
    def total_dim(self) -> int:
        return len(self.front_indices)

    @property
    def update_dim(self) -> int:
        return self.total_dim - self.pivot_dim
