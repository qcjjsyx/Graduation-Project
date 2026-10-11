# SystemC 定点优化实验副本

本目录是从当前基线复制出的独立优化版本。所有优化代码、构建产物和实验接口均位于
`graduation-code/optimization-v1/`，不会修改：

- `graduation-code/software/`
- `graduation-code/systemc/`

## 目录

```text
optimization-v1/
├── software/        独立软件产物生成器
├── systemc/         独立 SystemC 仿真系统
├── scripts/         B0/B1/B2 自动对比入口
├── results/         已复测的结果表
└── docs/            优化设计、实验和结论
```

## 当前优化变量

| 版本 | 软件均衡方法 | 方程 |
|---|---|---|
| B0 | `pow2-row` | `D_r A x = D_r b` |
| B1 | `pow2-row-column` | `D_r A D_c y = D_r b` |
| B2 | `pow2-ruiz` | 多轮温和行列均衡 |

B1/B2 的 SystemC 求解变量为 `y`，最终恢复：

```text
x = D_c y
```

行列缩放均限制为 2 的整数次幂，因此硬件只需 exponent 调整，不需要通用浮点乘法器。

## 构建和测试

```bash
cd graduation-code/optimization-v1/software
python -m pytest -q

cd ../systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## 自动 A/B 实验

```bash
cd graduation-code/optimization-v1
python scripts/compare_scaling.py \
  --out /tmp/graduation-scaling-comparison
```

脚本会为 256/576/1024 分别生成 B0/B1/B2 产物、运行 fixed SystemC，并输出
`comparison.json` 和 `comparison.csv`。

## 当前结论

- B1 将 576 residual 从 `2.346e-2` 降至 `4.593e-3`，改善约 80.4%；
- B2 将 1024 求精从 40 轮降至 1 轮，总周期下降约 8.0%，但前向解误差变大；
- B2 对 256 的 residual 和解误差均优于 B0；
- 不存在对三个矩阵所有指标都最优的固定缩放方案；
- B1 是当前面向最坏矩阵稳定性的推荐候选，但仍未使 576 达到 `1e-3`。

完整分析见 [docs/OPTIMIZATION_REPORT.md](docs/OPTIMIZATION_REPORT.md)。
