from __future__ import annotations

import struct
from typing import Iterable, List


def pack_exponents(exps: Iterable[int]) -> int:
    """Pack four int8 exponents into a uint32 (00,01,10,11 order)."""
    e = list(exps)
    if len(e) != 4:
        raise ValueError("expected 4 exponents")
    return struct.unpack("<I", struct.pack("<bbbb", e[0], e[1], e[2], e[3]))[0]


def unpack_exponents(packed: int) -> List[int]:
    b = struct.pack("<I", packed)
    return list(struct.unpack("<bbbb", b))