# SystemC 16×16 Tile BFP 多前沿系统

本目录是 `optimization-v2-tile-bfp` 的独立 SystemC 模型。它消费软件生成的真实
manifest 与 DDR 镜像，并把 16×16 tile exponent 贯穿：

```text
Artifact/DDR
→ Task Fetch / Scoreboard / Buffer
→ tile-aware QAU
→ exponent-aware HPU/ATU
→ tile BFP Panel/TRSM/GEMM-Schur
→ L/U/P/update 写回
→ 树形前代/回代
→ 原方程 residual 与迭代求精
```

## 与 v1 的差异

- `front_e` 与 `update_e` 从单个 exponent 变为二维 tile exponent 表；
- U tile exponent 表写入扩展后的 `node_meta`；
- 主元比较、L 除法和 Schur MAC 显式处理不同 tile exponent；
- 默认 `q_use_bits=30`、`workspace_guard_bits=20`；
- 写回前检查局部 `PA=LU+S` 相对误差，默认门槛 `2e-7`；
- `nodes.csv` 输出 assembled/U/update exponent 的数量、最小值和最大值。

NodeTask 仍使用小端 ABI v2、128 字节记录。布局详见
[Tile BFP ABI](../docs/TILE_BFP_ABI.md)，设计与实验详见
[实现报告](../docs/TILE_BFP_IMPLEMENTATION.md)。

## 构建

```bash
cd graduation-code/optimization-v2-tile-bfp/systemc
cmake -S . -B build \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2 \
  -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

ASan/UBSan：

```bash
cmake -S . -B /tmp/graduation-v2-tile-asan \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2 \
  -DENABLE_SANITIZERS=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/graduation-v2-tile-asan --parallel
ctest --test-dir /tmp/graduation-v2-tile-asan --output-on-failure
```

## 运行

```bash
./build/system_sim \
  --artifact /tmp/mf-tile/manifest.json \
  --config config/default.json \
  --mode fp64|fixed|both \
  --out /tmp/mf-tile-result \
  --seed 1
```

manifest 和配置中的 `bfp_tile_size` 必须一致。`--vcd` 可选。

## 默认配置

- 硬件调度 tile 与 BFP tile：16；
- int32 mantissa、int16 exponent、int64 matrix accumulator；
- QAU/因子输出有效位：30；
- L QF 小数位：20；
- workspace guard bits：20；
- fixed pivot threshold：`1e-5`；
- local factor check threshold：`2e-7`；
- rescue：FP64 功能/周期模型；
- 原方程迭代求精目标：`1e-3`。

precision rescue 尚无对应 RTL。当前严格门槛在 256/576 上触发大量 rescue，因此结果
应解释为“稳定的系统研究模型”和“RTL 风险定位”，而不是纯 INT32 核已经足够。

## 输出

- `summary.json`：配置、状态、residual、解误差、周期、rescue 和 tile 总量；
- `nodes.csv`：每 node 的阶段周期、pivot、量化风险和 exponent range；
- `operations.csv`：Panel/TRSM/GEMM、rescue 和 solve 操作；
- `memory.csv`：DDR 字节、burst、周期和利用率；
- `timeline.csv`：依赖、buffer、kernel 和写回事件；
- `solution.csv`：参考解、FP64 解和 fixed 解；
- `final_memory_image.bin`：写回后的 DDR 快照。

当前验证结果：Python 33 项、CTest 3/3、ASan/UBSan 3/3 全部通过。
