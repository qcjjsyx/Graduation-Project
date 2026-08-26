# v3：16×16 Tile BFP 精度优化

本目录是从 `optimization-v2-tile-bfp` 复制出的独立优化版本。冻结的 v2、根目录
软件/SystemC 基线和 RTL 均未修改。

v3 针对 v2 的主要问题继续优化：L 乘子使用 QF20 时，量化步长约为
`9.54e-7`，大于局部因子检查阈值 `2e-7`，导致大量节点进入 FP64 rescue。定向扫描
表明，统一使用 QF26 是当前更稳定、代价更低的选择。

## 推荐配置

- 矩阵格式：int32 mantissa + 每 16×16 tile 一个 int16 exponent；
- L 格式：统一 QF26；
- 工作区：int64 + 20 guard bits；
- 输出有效位：30 bit；
- 局部因子检查：`2e-7`；
- 主元/溢出/因子检查仍失败时：显式 FP64 rescue；
- 原方程求解：保留带下降保护的迭代求精。

实验性 F20→F26 节点重算仍保留在
[`systemc/config/adaptive-f20-f26.json`](systemc/config/adaptive-f20-f26.json)。
它用于证明“按失败节点混用两种 QF 精度”并不优于统一 F26，不作为推荐硬件配置。

## 目录

```text
optimization-v3-adaptive-precision/
├── software/        与 v2 制品格式兼容的软件快照
├── systemc/
│   ├── config/      推荐配置与自适应对照配置
│   ├── include/     数值核、控制、存储、求解和报告
│   ├── results/      compact 真实矩阵对照表
│   └── tests/
└── docs/
    ├── ADAPTIVE_PRECISION_OPTIMIZATION.md
    ├── TILE_BFP_ABI.md
    └── TILE_BFP_IMPLEMENTATION.md
```

## 构建与测试

```bash
cd graduation-code/optimization-v3-adaptive-precision/software
/opt/anaconda3/bin/python3 -m pytest -q

cd ../systemc
cmake -S . -B build \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2 \
  -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## 运行

推荐的统一 QF26：

```bash
./build/system_sim \
  --artifact /tmp/mf-v2-tile-576/manifest.json \
  --config config/default.json \
  --mode fixed \
  --out /tmp/v3-uniform-f26-576 \
  --seed 1
```

自适应反例：

```bash
./build/system_sim \
  --artifact /tmp/mf-v2-tile-576/manifest.json \
  --config config/adaptive-f20-f26.json \
  --mode fixed \
  --out /tmp/v3-adaptive-f20-f26-576 \
  --seed 1
```

注意：576JJ 的独立 FP64 黄金路径会在默认 `pivot_rel_tol=1e-12` 下明确报告末列小
主元，因此精度策略的 A/B 使用 `--mode fixed`。这不代表定点通道失败；定点 rescue
路径使用单独的 `rescue_pivot_rel_tol=1e-16`。

## 主要结果

统一 QF26 相比冻结的 v2 QF20：

| 矩阵 | residual：v2 → v3 | 解误差：v2 → v3 | FP64 rescue：v2 → v3 | 周期变化 |
|---|---:|---:|---:|---:|
| 256 | `5.964e-4 → 7.096e-5` | `1.514 → 0.174` | `68 → 9` | `+0.69%` |
| 576 | `3.252e-4 → 3.024e-4` | `6.934 → 0.554` | `150 → 37` | `-0.43%` |
| 1024 | `1.012e-4 → 2.816e-5` | `3.000e-8 → 1.293e-6` | `122 → 49` | `-17.02%` |

三组 residual 均低于 `1e-3`。1024 的解误差虽相对 v2 增大，绝对值仍为
`1.29e-6`；其 rescue 和求精成本显著下降。完整数据见
[`precision_comparison.csv`](systemc/results/precision_comparison.csv)，实现、失败方案和
硬件建议见
[`ADAPTIVE_PRECISION_OPTIMIZATION.md`](docs/ADAPTIVE_PRECISION_OPTIMIZATION.md)。
