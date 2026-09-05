import pytest

from src.dataStruct import MemoryRegion
from src.memory.planner import (
    UINT64_MAX,
    AddressOverflowError,
    plan_regions,
    validate_region_layout,
)


def test_command_memory_planner_aligns_without_overlap():
    regions, total = plan_regions(
        [("commands", 96), ("descriptors", 128), ("empty", 0), ("data", 17)],
        alignment=64,
    )
    assert [regions[name].offset for name in regions] == [0, 128, 256, 256]
    assert total == 273
    validate_region_layout(regions, total, 64)


def test_command_memory_planner_rejects_overflow_and_overlap():
    with pytest.raises(AddressOverflowError, match="overflows u64"):
        plan_regions([("too_large", UINT64_MAX), ("next", 1)], alignment=64)

    with pytest.raises(ValueError, match="overlap"):
        validate_region_layout(
            {"a": MemoryRegion(0, 65), "b": MemoryRegion(64, 1)}, 128, 64
        )
