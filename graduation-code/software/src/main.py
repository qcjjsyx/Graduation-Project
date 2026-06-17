from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MatrixInputConfig, OrderingConfig, PipelineConfig, QuantConfig
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multifrontal LU software-side pipeline")
    parser.add_argument("-mtx", "--mtx", type=str, default=None, help="input .mat or MatrixMarket file")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--effective-bits", type=int, default=27)
    parser.add_argument("--clip-percentile", type=float, default=100.0)
    parser.add_argument("--max-supernode-size", type=int, default=256)
    parser.add_argument(
        "--ordering",
        choices=["amd", "rcm", "identity"],
        default="amd",
        help="symbolic ordering method",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
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
        quant=QuantConfig(
            effective_bits=args.effective_bits,
            clip_percentile=args.clip_percentile,
        ),
        out_dir=args.out,
    )
    outputs = run_pipeline(config)
    print(f"residual_norm: {outputs.residual_norm:.3e}")
    print(f"nodes: {outputs.node_count}, tasks: {outputs.task_count}")
    print(f"out_dir: {outputs.out_dir}")


if __name__ == "__main__":
    main()
