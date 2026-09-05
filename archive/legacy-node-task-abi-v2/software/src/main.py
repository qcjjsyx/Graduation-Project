from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    EquilibrationConfig,
    MatrixInputConfig,
    OrderingConfig,
    PipelineConfig,
    QuantConfig,
    SolveInputConfig,
)
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multifrontal LU software-side pipeline")
    # 输入矩阵文件：.mat 或 MatrixMarket 格式，未指定时使用生成的随机矩阵
    parser.add_argument("-mtx", "--mtx", type=str, default=None, help="input .mat or MatrixMarket file")
    # 输出目录：保存流水线产生的结果文件
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    # 随机数种子：用于生成随机测试矩阵
    parser.add_argument("--seed", type=int, default=0)
    # 矩阵规模：随机矩阵的维度 n x n
    parser.add_argument("--n", type=int, default=64)
    # 稀疏度：随机矩阵中非零元素所占比例
    parser.add_argument("--density", type=float, default=0.1)
    # 有效位数：量化时保留的有效比特数
    parser.add_argument("--effective-bits", type=int, default=30)
    # 裁剪百分位：量化前对数值进行裁剪的百分位阈值
    parser.add_argument("--clip-percentile", type=float, default=100.0)
    parser.add_argument(
        "--equilibrate",
        choices=["none", "pow2-row"],
        default="pow2-row",
        help="sparsity-preserving numerical preprocessing",  # 数值预处理方式：保持稀疏性的均衡化方法
    )
    parser.add_argument(
        "--max-scale-exponent",
        type=int,
        default=60,
        help="absolute clamp for power-of-two row scale exponents",  # 行缩放幂指数的最大绝对值限制
    )
    # 最大超节点大小：符号分解时超节点的规模上限
    parser.add_argument("--max-supernode-size", type=int, default=256)
    # 右端项文件：可选的 RHS .mat 或 .npy 文件
    parser.add_argument("--rhs", type=str, default=None, help="optional RHS .mat or .npy file")
    parser.add_argument(
        "--rhs-seed",
        type=int,
        default=1,
        help="seed used to generate x_true and RHS when --rhs is omitted",  # 未指定 --rhs 时生成 x_true 和 RHS 所用的随机数种子
    )
    parser.add_argument(
        "--ordering",
        choices=["amd", "rcm", "identity"],
        default="amd",
        help="symbolic ordering method",  # 符号排序方法：amd（近似最小度）/ rcm（反向 Cuthill-McKee）/ identity（不排序）
    )
    return parser.parse_args()

#  main function
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
        equilibration=EquilibrationConfig(
            mode=args.equilibrate,
            max_scale_exponent=args.max_scale_exponent,
        ),
        solve=SolveInputConfig(rhs_path=args.rhs, seed=args.rhs_seed),
        out_dir=args.out,
    )
    outputs = run_pipeline(config)
    print(f"scaled_residual_norm: {outputs.residual_norm:.3e}")
    print(
        "original_residual_norm: "
        f"{outputs.original_residual_norm:.3e}"
    )
    print(f"nodes: {outputs.node_count}, tasks: {outputs.task_count}")
    print(f"out_dir: {outputs.out_dir}")


if __name__ == "__main__":
    main()
