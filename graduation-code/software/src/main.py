from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.command_codec import (
    ScaleDesc,
    ScaleSaturationPolicy,
)
from src.config import (
    CommandCompilerConfig,
    EquilibrationConfig,
    GemmConfig,
    MatchingConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
    PivotPolicyConfig,
    SolveInputConfig,
)
from src.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-solver Command v2 compiler")
    parser.add_argument("-mtx", "--mtx", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument(
        "--equilibrate", choices=["none", "pow2-row"], default="pow2-row"
    )
    parser.add_argument("--max-scale-exponent", type=int, default=60)
    parser.add_argument(
        "--matching",
        choices=["none", "weighted_matching"],
        default="weighted_matching",
    )
    parser.add_argument(
        "--pivot-policy", choices=["STATIC", "DYNAMIC_PANEL"], default="STATIC"
    )
    parser.add_argument("--pivot-threshold", type=float, default=0.0)
    parser.add_argument("--max-supernode-size", type=int, default=256)
    parser.add_argument("--max-front-size", type=int, default=256)
    parser.add_argument(
        "--device-buffer-bytes",
        type=int,
        default=64 * 1024,
        help="target device scratchpad slot size in bytes; every FP32 front "
        "must fit (default 65536 -> front <= 128)",
    )
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-wait-tokens", type=int, default=256)
    parser.add_argument(
        "--reference-check",
        action="store_true",
        help="run the host FP64 front-tree reference around compilation "
        "(preflight + residual metrics); off by default",
    )
    parser.add_argument("--rhs", type=str, default=None)
    parser.add_argument("--rhs-seed", type=int, default=1)
    parser.add_argument(
        "--ordering", choices=["amd", "rcm", "identity"], default="amd"
    )
    parser.add_argument(
        "--gemm-backend",
        choices=["FP32_SYSTEMC", "FP32_SYSTEMC_INT32_GEMM"],
        default="FP32_SYSTEMC",
        help="top-level GEMM_SCHUR backend; INT32 switches only that kernel "
        "and emits a SCALE_DESC per node (device front/DDR stays FP32)",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--factorization-only",
        dest="emit_solve_commands",
        action="store_false",
        help="emit factorization commands only; the device entry "
        "tree_factor_device accepts both scopes, but a factorization-only "
        "artifact stops before the solve chain",
    )
    scope.add_argument(
        "--emit-solve-commands",
        dest="emit_solve_commands",
        action="store_true",
        help="compatibility alias for the default scope (T12: the device now "
        "executes SOLVE_FORWARD/SOLVE_BACKWARD)",
    )
    parser.set_defaults(emit_solve_commands=True)
    parser.add_argument(
        "--gemm-effective-bits",
        type=int,
        default=30,
        help="INT32 effective quantization bits (supported range [1, 30])",
    )
    parser.add_argument(
        "--gemm-saturation",
        choices=["FAIL", "CLAMP_AND_REPORT"],
        default="FAIL",
        help="SCALE_DESC saturation policy (default FAIL)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(
                path=args.mtx,
                n=args.n,
                density=args.density,
                seed=args.seed,
            ),
            ordering=OrderingConfig(
                method=args.ordering,
                max_supernode_size=args.max_supernode_size,
            ),
            equilibration=EquilibrationConfig(
                mode=args.equilibrate,
                max_scale_exponent=args.max_scale_exponent,
            ),
            matching=MatchingConfig(method=args.matching),
            pivot=PivotPolicyConfig(
                policy=args.pivot_policy,
                threshold_ratio=args.pivot_threshold,
            ),
            command=CommandCompilerConfig(
                tile_size=args.tile_size,
                max_front_size=args.max_front_size,
                device_buffer_bytes=args.device_buffer_bytes,
                max_wait_tokens=args.max_wait_tokens,
                emit_solve_commands=args.emit_solve_commands,
            ),
            gemm=GemmConfig(
                backend=args.gemm_backend,
                effective_bits=args.gemm_effective_bits,
                scale_desc=ScaleDesc(
                    saturation=ScaleSaturationPolicy[
                        args.gemm_saturation
                    ],
                ),
            ),
            solve=SolveInputConfig(rhs_path=args.rhs, seed=args.rhs_seed),
            reference_check=args.reference_check,
            out_dir=args.out,
        )
    )
    if outputs.residual_norm is not None:
        print(f"scaled_residual_norm: {outputs.residual_norm:.3e}")
    else:
        print("golden_residual_norm: not available (no reference solve)")
    if outputs.reference_check_performed:
        assert outputs.original_residual_norm is not None
        assert outputs.componentwise_backward_error is not None
        print(f"original_residual_norm: {outputs.original_residual_norm:.3e}")
        print(f"original_backward_error: {outputs.componentwise_backward_error:.3e}")
    else:
        print("host_reference: not run (use --reference-check)")
    print(
        f"nodes: {outputs.node_count}, commands: {outputs.command_count}, "
        f"descriptors: {outputs.descriptor_count}, "
        f"max_front: {outputs.max_front_size}"
    )
    print(
        f"gemm_backend: {'SYSTEMC_INT32_GEMM_MODEL' if args.gemm_backend != 'FP32_SYSTEMC' else 'SYSTEMC_FP32_DEVICE_MODEL'}"
        f" (effective_bits={args.gemm_effective_bits}, "
        f"saturation={args.gemm_saturation})"
    )
    print(
        "operation_scope: "
        + ("factorization_and_solve" if args.emit_solve_commands else "factorization_only")
    )
    print(f"out_dir: {outputs.out_dir}")


def cli() -> int:
    """Console entry: report compile/validation errors without a traceback."""
    try:
        main()
    except (ValueError, OSError) as exc:  # CommandCompileError etc. are ValueErrors
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(cli())
