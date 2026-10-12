from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.command_codec import (
    ScaleDesc,
    ScaleSaturationPolicy,
    normalize_pivot_threshold,
)


@dataclass(frozen=True)
class GemmConfig:
    """Top-level GEMM_SCHUR backend and local precision policy (T10).

    ``FP32_SYSTEMC`` (default) keeps every kernel on the formal FP32 device
    path: the GEMM ``KERNEL_DESC`` carries ``scale_desc_id = NONE``.

    ``FP32_SYSTEMC_INT32_GEMM`` switches ONLY the top-level ``GEMM_SCHUR``
    kernel to the local INT32 behavior model and emits one ``SCALE_DESC`` per
    node GEMM kernel.  Panel/TRSM and the panel-internal updates stay FP32, and
    the device front/DDR image keeps FP32 storage.

    The supported precision range is INT32 storage / 30 effective quantization
    bits / INT64 accumulator: ``effective_bits`` in ``[1, 30]`` with
    ``k_block_max * (2^(bits-1)-1)^2 < 2^(accumulator_bits-2)``.  Any other
    configuration is rejected here instead of being ignored at execution time.
    """

    backend: str = "FP32_SYSTEMC"
    effective_bits: int = 30
    accumulator_bits: int = 64
    k_block_max: int = 16
    scale_desc: ScaleDesc = ScaleDesc()
    conversion_rate: int = 1

    _VALID_BACKENDS: ClassVar[frozenset[str]] = frozenset(
        {"FP32_SYSTEMC", "FP32_SYSTEMC_INT32_GEMM"}
    )

    def __post_init__(self) -> None:
        if self.backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"gemm backend must be one of {sorted(self._VALID_BACKENDS)}, "
                f"got {self.backend!r}"
            )
        if not 1 <= self.effective_bits <= 30:
            raise ValueError(
                "effective_bits must be in [1, 30]: INT32 storage with 30 "
                "effective quantization bits is the largest range whose "
                "16-term dot product is provably < 2^62"
            )
        if self.accumulator_bits != 64:
            raise ValueError("only the INT64 accumulator (64) is supported")
        if not 1 <= self.k_block_max <= 16:
            raise ValueError("k_block_max must be in [1, 16]")
        qmax = (1 << (self.effective_bits - 1)) - 1
        if self.k_block_max * qmax * qmax >= 1 << (self.accumulator_bits - 2):
            raise ValueError(
                "unsafe quantization range: k_block_max * qmax^2 must stay "
                "below 2^(accumulator_bits-2)"
            )
        if self.conversion_rate < 1:
            raise ValueError("conversion_rate must be >= 1 element per cycle")
        ScaleDesc(
            exponent_a=self.scale_desc.exponent_a,
            exponent_b=self.scale_desc.exponent_b,
            rounding=self.scale_desc.rounding,
            saturation=self.scale_desc.saturation,
            overflow=self.scale_desc.overflow,
        )
        if self.backend == "FP32_SYSTEMC":
            if (
                self.scale_desc.exponent_a is not None
                or self.scale_desc.exponent_b is not None
                or self.scale_desc.rounding
                != ScaleDesc().rounding
                or self.scale_desc.saturation
                != ScaleSaturationPolicy.FAIL
            ):
                raise ValueError(
                    "FP32_SYSTEMC must not carry INT32 scale settings; the "
                    "FP32 GEMM kernel keeps scale_desc_id = NONE"
                )

    @property
    def int32_gemm_enabled(self) -> bool:
        return self.backend == "FP32_SYSTEMC_INT32_GEMM"


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
    # True: full schema (factorization + SOLVE_FORWARD/SOLVE_BACKWARD chain).
    # The device executes the chain when ControlPlaneConfig.enable_real_solve is
    # on (tree_factor_device); the library default stays True because the
    # compiler/T04/T05/T10 tests check the full command schema, and the T12 CLI
    # defaults to factorization_and_solve as well (--factorization-only remains
    # for factorization-only artifacts).
    emit_solve_commands: bool = True
    # Target device scratchpad slot size.  A front is held in one slot as a
    # compact FP32 n*n block, so every front must satisfy 4*n*n <= this value
    # (64 KiB -> n <= 128).  Recorded in the manifest; tree_factor_device sizes
    # its slots from it.  Checked at compile time so an oversize front fails
    # here instead of as BUFFER_FULL after a long device run.
    device_buffer_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if (
            isinstance(self.device_buffer_bytes, bool)
            or not isinstance(self.device_buffer_bytes, int)
            or self.device_buffer_bytes <= 0
        ):
            raise ValueError("device_buffer_bytes must be a positive integer")
        if not isinstance(self.emit_solve_commands, bool):
            raise ValueError("emit_solve_commands must be a boolean")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 < self.max_front_size <= 512:
            # The SystemC Panel engine supports front dimensions 1..512.
            raise ValueError("max_front_size must be in [1, 512]")
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
    gemm: GemmConfig = GemmConfig()
    solve: SolveInputConfig = SolveInputConfig()
    # Explicit opt-in: run the host FP64 front-tree reference (preflight +
    # residual metrics) around compilation.  Normal compilation never depends
    # on the numeric solve succeeding, so inputs that must fail on the device
    # can still be compiled.  Independent entry: src.verify.front_reference.
    reference_check: bool = False
    out_dir: Path = Path("out")
