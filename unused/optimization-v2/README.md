# 16×16 Tile BFP SystemC 优化版本

本目录是从 `optimization-v1` 复制出的独立实验版本。它没有修改：

- `graduation-code/software/` 与 `graduation-code/systemc/` 基线；
- `graduation-code/optimization-v1/` 行列均衡版本。

当前版本把原来的“每个 front/U/update 一个公共 exponent”扩展为
“每个 16×16 tile 一个 int16 exponent”，并将该格式贯穿软件制品、DDR、QAU、
定点 LU、L/U/update 写回、树形求解与报告。

## 目录

```text
optimization-v2-tile-bfp/
├── software/        生成 tile-BFP front 和可变长 exponent 区域
├── systemc/         tile-aware 装配、分解、写回、求解与周期模型
├── scripts/         批量实验入口
├── results/         基线与 tile-BFP 真实矩阵结果
└── docs/
    ├── TILE_BFP_ABI.md
    └── TILE_BFP_IMPLEMENTATION.md
```

## 默认数值策略

- tile：`16×16`；
- local/front/U/update mantissa：int32；
- 软件输入有效位：30 bit；
- SystemC 输出有效位：30 bit；
- L：QF，默认 `F=20`；
- 节点工作区：int64，20 guard bits；
- 写回前局部因子检查阈值：`2e-7`；
- 不满足主元、溢出或局部因子误差门槛时显式进入 FP64 precision rescue；
- 最终仍在原始 `A,b` 上执行带下降保护的迭代求精。

严格的局部误差门槛会显著增加 rescue 数量。它保证当前系统研究模型不把不可靠的
tile 定点因子继续传播，但不代表这些 rescue 已有 RTL 实现。具体原因、公式、实验和
硬件含义见 [实现与实验报告](docs/TILE_BFP_IMPLEMENTATION.md)。

## 构建与验证

```bash
cd graduation-code/optimization-v2-tile-bfp/software
/opt/anaconda3/bin/python3 -m pytest -q

cd ../systemc
cmake -S . -B build \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2 \
  -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

生成 576JJ 制品：

```bash
cd graduation-code/optimization-v2-tile-bfp/software
/opt/anaconda3/bin/python3 -m src.main \
  -mtx example/576X576JJ.mat \
  --rhs example/576fuv.mat \
  --ordering amd \
  --equilibrate pow2-row-column \
  --bfp-tile-size 16 \
  --out /tmp/mf-tile-576
```

运行定点仿真：

```bash
cd ../systemc
./build/system_sim \
  --artifact /tmp/mf-tile-576/manifest.json \
  --config config/default.json \
  --mode fixed \
  --out /tmp/mf-tile-576-result \
  --seed 1
```

`NodeTask` 仍为 ABI v2 的 128 字节记录；tile exponent 的可变长度由 manifest 中的
区域尺寸确定。详细内存布局见 [Tile BFP ABI](docs/TILE_BFP_ABI.md)。
