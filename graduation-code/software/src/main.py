from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CommandCompilerConfig,
    EquilibrationConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
    SolveInputConfig,
)
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-solver Command v1 compiler")
    parser.add_argument("-mtx", "--mtx", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument(
        "--equilibrate", choices=["none", "pow2-row"], default="pow2-row"
    )
    parser.add_argument("--max-scale-exponent", type=int, default=60)
    parser.add_argument("--max-supernode-size", type=int, default=256)
    parser.add_argument("--max-front-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-wait-tokens", type=int, default=256)
    parser.add_argument("--rhs", type=str, default=None)
    parser.add_argument("--rhs-seed", type=int, default=1)
    parser.add_argument(
        "--ordering", choices=["amd", "rcm", "identity"], default="amd"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            command=CommandCompilerConfig(
                tile_size=args.tile_size,
                max_front_size=args.max_front_size,
                max_wait_tokens=args.max_wait_tokens,
            ),
            solve=SolveInputConfig(rhs_path=args.rhs, seed=args.rhs_seed),
            out_dir=args.out,
        )
    )
    print(f"scaled_residual_norm: {outputs.residual_norm:.3e}")
    print(f"original_residual_norm: {outputs.original_residual_norm:.3e}")
    print(
        f"nodes: {outputs.node_count}, commands: {outputs.command_count}, "
        f"descriptors: {outputs.descriptor_count}"
    )
    print(f"out_dir: {outputs.out_dir}")


if __name__ == "__main__":
    main()
