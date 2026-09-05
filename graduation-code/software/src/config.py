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
        if self.max_supernode_size <= 0:
            raise ValueError("max_supernode_size must be positive")


@dataclass(frozen=True)
class EquilibrationConfig:
    mode: str = "pow2-row"
    max_scale_exponent: int = 60

    _VALID_MODES: ClassVar[frozenset[str]] = frozenset({"none", "pow2-row"})

    def __post_init__(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"equilibration mode must be one of "
                f"{sorted(self._VALID_MODES)}, got {self.mode!r}"
            )
        if not 0 <= self.max_scale_exponent <= 1023:
            raise ValueError(
                "max_scale_exponent must be in [0, 1023], got "
                f"{self.max_scale_exponent}"
            )


@dataclass(frozen=True)
class CommandCompilerConfig:
    alignment: int = 64
    tile_size: int = 16
    max_front_size: int = 256
    max_wait_tokens: int = 256

    def __post_init__(self) -> None:
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.max_front_size <= 0:
            raise ValueError("max_front_size must be positive")
        if self.max_wait_tokens <= 0:
            raise ValueError("max_wait_tokens must be positive")


@dataclass(frozen=True)
class SolveInputConfig:
    rhs_path: str | None = None
    seed: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    matrix: MatrixInputConfig = MatrixInputConfig()
    ordering: OrderingConfig = OrderingConfig()
    equilibration: EquilibrationConfig = EquilibrationConfig()
    command: CommandCompilerConfig = CommandCompilerConfig()
    solve: SolveInputConfig = SolveInputConfig()
    out_dir: Path = Path("out")
