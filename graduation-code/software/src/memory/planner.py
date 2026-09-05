from __future__ import annotations

from collections.abc import Iterable

from src.dataStruct import MemoryRegion


UINT64_MAX = (1 << 64) - 1


class AddressOverflowError(ValueError):
    pass


def checked_add_u64(lhs: int, rhs: int, label: str) -> int:
    if lhs < 0 or rhs < 0:
        raise ValueError(f"{label}: sizes and addresses must be non-negative")
    if lhs > UINT64_MAX or rhs > UINT64_MAX or rhs > UINT64_MAX - lhs:
        raise AddressOverflowError(f"{label}: address range overflows u64")
    return lhs + rhs


def align_up_u64(value: int, alignment: int, label: str) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    if value < 0 or value > UINT64_MAX:
        raise AddressOverflowError(f"{label}: address is outside u64")
    padding = (-value) & (alignment - 1)
    return checked_add_u64(value, padding, label)


def plan_regions(
    specs: Iterable[tuple[str, int]], *, alignment: int = 64
) -> tuple[dict[str, MemoryRegion], int]:
    """Place named regions deterministically and reject u64 overflow."""

    regions: dict[str, MemoryRegion] = {}
    cursor = 0
    for name, size in specs:
        if not name:
            raise ValueError("memory region name must not be empty")
        if name in regions:
            raise ValueError(f"duplicate memory region name: {name}")
        if size < 0:
            raise ValueError(f"{name}: region size must be non-negative")
        offset = align_up_u64(cursor, alignment, name)
        cursor = checked_add_u64(offset, size, name)
        regions[name] = MemoryRegion(offset=offset, size=size)
    validate_region_layout(regions, cursor, alignment)
    return regions, cursor


def validate_region_layout(
    regions: dict[str, MemoryRegion], total_bytes: int, alignment: int = 64
) -> None:
    if total_bytes < 0 or total_bytes > UINT64_MAX:
        raise AddressOverflowError("memory image size is outside u64")
    occupied: list[tuple[int, int, str]] = []
    for name, region in regions.items():
        if region.offset % alignment:
            raise ValueError(f"{name}: region offset is not {alignment}-byte aligned")
        end = checked_add_u64(region.offset, region.size, name)
        if end > total_bytes:
            raise ValueError(f"{name}: region exceeds memory image")
        if region.size:
            occupied.append((region.offset, end, name))
    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if previous[1] > current[0]:
            raise ValueError(
                f"memory regions overlap: {previous[2]} and {current[2]}"
            )
