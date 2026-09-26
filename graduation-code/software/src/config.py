from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.command_codec import normalize_pivot_threshold


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
class MatchingConfig:
    """Host-side numeric preprocessing: global one-to-one row matching.

    ``weighted_matching`` computes a maximum-product perfect matching on the
    nonzero pattern and converts it into a row permutation that strengthens the
    diagonal.  ``none`` disables the stage (baseline and ablation runs).
    """

    method: str = "weighted_matching"
    objective: str = "max_product"

    _VALID_METHODS: ClassVar[frozenset[str]] = frozenset({"none", "weighted_matching"})
    _VALID_OBJECTIVES: ClassVar[frozenset[str]] = frozenset({"max_product"})

    def __post_init__(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"matching method must be one of "
                f"{sorted(self._VALID_METHODS)}, got {self.method!r}"
            )
        if self.objective not in self._VALID_OBJECTIVES:
            raise ValueError(
                f"matching objective must be one of "
                f"{sorted(self._VALID_OBJECTIVES)}, got {self.objective!r}"
            )


@dataclass(frozen=True)
class PivotPolicyConfig:
    """Device pivot policy declared at compile time (batch-level, uniform).

    ``STATIC``: fixed pivot order = post-ordering diagonal order; no candidate
    search and no row interchange on the device; the HPU/ATU dynamic mapping
    path is bypassed.  ``DYNAMIC_PANEL``: pivot search restricted to the
    current pivot block.  Threshold rejections are relative to
    ``front_pre_elimination_max_abs`` (max |A_front| of the assembled front
    before elimination of that front).

    ``threshold_ratio`` is normalized once to its IEEE-754 binary32 value (the
    device storage format), so configuration, manifest, host reference and the
    descriptor share one effective threshold.  Values that underflow to zero or
    round to 1.0 in binary32 are rejected at configuration time.
    """

    policy: str = "STATIC"
    threshold_ratio: float = 0.0

    _VALID_POLICIES: ClassVar[frozenset[str]] = frozenset({"STATIC", "DYNAMIC_PANEL"})

    def __post_init__(self) -> None:
        if self.policy not in self._VALID_POLICIES:
            raise ValueError(
                f"pivot policy must be one of "
                f"{sorted(self._VALID_POLICIES)}, got {self.policy!r}"
            )
        object.__setattr__(
            self, "threshold_ratio", normalize_pivot_threshold(self.threshold_ratio)
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
    matching: MatchingConfig = MatchingConfig()
    pivot: PivotPolicyConfig = PivotPolicyConfig()
    command: CommandCompilerConfig = CommandCompilerConfig()
    solve: SolveInputConfig = SolveInputConfig()
    # Explicit opt-in: run the host FP64 front-tree reference (preflight +
    # residual metrics) around compilation.  Normal compilation never depends
    # on the numeric solve succeeding, so inputs that must fail on the device
    # can still be compiled.  Independent entry: src.verify.front_reference.
    reference_check: bool = False
    out_dir: Path = Path("out")
