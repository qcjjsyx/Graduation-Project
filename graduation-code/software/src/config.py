from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class MatrixInputConfig:
    path: str | None = None
    n: int = 64
    density: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if not 0 < self.density <= 1:
            raise ValueError(f"density must be in (0, 1], got {self.density}")


@dataclass(frozen=True)
class OrderingConfig:
    method: str = "amd"
    max_supernode_size: int = 256

    _VALID_METHODS: ClassVar[frozenset[str]] = frozenset({"amd", "rcm", "identity"})

    def __post_init__(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(self._VALID_METHODS)}, got {self.method!r}"
            )


@dataclass(frozen=True)
class QuantConfig:
    effective_bits: int = 27
    clip_percentile: float = 100.0

    def __post_init__(self) -> None:
        if not 1 <= self.effective_bits <= 30:
            raise ValueError(
                f"effective_bits must be in [1, 30], got {self.effective_bits}"
            )
        if not 0 < self.clip_percentile <= 100:
            raise ValueError(f"clip_percentile must be in (0, 100], got {self.clip_percentile}")


@dataclass(frozen=True)
class MemoryConfig:
    alignment: int = 64

    def __post_init__(self) -> None:
        if self.alignment <= 0:
            raise ValueError(f"alignment must be positive, got {self.alignment}")
        if self.alignment & (self.alignment - 1) != 0:
            raise ValueError(f"alignment must be a power of 2, got {self.alignment}")


@dataclass(frozen=True)
class PipelineConfig:
    matrix: MatrixInputConfig = MatrixInputConfig()
    ordering: OrderingConfig = OrderingConfig()
    quant: QuantConfig = QuantConfig()
    memory: MemoryConfig = MemoryConfig()
    out_dir: Path = Path("out")
