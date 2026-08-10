# SystemC 多前沿矩阵求解系统

这里实现的是由真实软件产物驱动的完整系统级原型，不再使用 C++ 内置三节点 fixture，
也不再使用 `ComputeStub`。FP64 与 int32+BFP 定点后端共享任务、依赖、缓冲、装配、
主元、算子调度和写回控制流。

## 已实现链路

```text
manifest + memory_image.bin
  -> Artifact Loader / ABI 二次校验
  -> 字节地址 DDR 与事务延迟模型
  -> Task Fetch
  -> Dependency Scoreboard（支持多根森林与任意任务文件顺序）
  -> Buffer Manager / Front Loader
  -> QAU extend-add、指数对齐、int64 累加和再量化
  -> HPU/ATU 逐周期主元握手
  -> Panel LU / TRSM / GEMM-Schur 功能核
  -> Micro Scheduler（serial 或 resource_aware）
  -> L/U/P/update DDR 写回
  -> 写回完成后释放 parent 依赖和 front buffer
  -> 树形前代、回代和 solution DDR 写回
  -> 原方程 FP64 residual + 混合精度迭代求精
  -> 正确性、量化稳定性、访存和周期报告
```

HPU、ATU、FIFO、任务依赖和缓冲生命周期按时钟推进。DDR/DMA 与计算核采用可配置
事务/吞吐模型，不宣称达到 RTL 或 AXI 信号级精度。

## 定点稳定性实现

默认稳定性链路不是简单的“int32 原位消元”，而是：

1. 软件先执行稀疏结构不变的 2 的整数次幂行均衡；
2. DDR 输入保留 30 个有效位，QAU/计算输出默认使用 26 位，预留符号、累加和增长空间；
3. front 装配后将 int32 mantissa 左移 `workspace_guard_bits=8`，在可配置的 int64
   accumulator 中完成节点内消元；
4. 每列送入 32-bit HPU 前，只对该列候选统一归一化，不改变候选大小关系；
5. `U` 和 child `update` 独立选择 BFP exponent，避免二者动态范围互相牵制；
6. 小主元、量化后零对角、L 乘子越界或工作区溢出会产生显式 precision-rescue
   事件。默认救援后端对“已经由定点 DDR 数据装配出的 front”执行 FP64 分解，再量化
   L/U/update；它不会读取黄金 FP64 front 绕过输入量化；
7. 向量求解使用 128-bit 中间除法和每节点动态 V-format exponent；
8. 最后在原始 `A,b` 上计算 FP64 residual，并执行带下降方向和最小改进保护的混合精度
   迭代求精。

precision rescue 是 SystemC 中的高精度功能/周期模型，不代表已实现对应 RTL。完整设计、
问题复现、公式、取舍和真实矩阵结果见
[定点稳定性改进说明](docs/FIXED_POINT_STABILITY_DESIGN.md)。

## 构建与测试

```bash
export SYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake -S graduation-code/systemc \
  -B graduation-code/systemc/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build graduation-code/systemc/build --parallel
ctest --test-dir graduation-code/systemc/build --output-on-failure
```

测试包括：

- ATU/HPU 边界和握手自检；
- ABI v2 与定点舍入/除法/最小负数单元测试；
- Python 生成产物到 SystemC 分解/求解的端到端测试；
- 错误版本、截断镜像、区域重叠和损坏 map 拒绝测试；
- 20 个固定反压 seed、同 seed 可复现性和性能单调性测试。

ASan/UBSan：

```bash
cmake -S graduation-code/systemc \
  -B /tmp/graduation-systemc-asan \
  -DENABLE_SANITIZERS=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/graduation-systemc-asan --parallel
ctest --test-dir /tmp/graduation-systemc-asan --output-on-failure
```

## 生成并运行产物

随机稳定算例：

```bash
cd graduation-code/software
python -m src.main \
  --n 32 --density 0.1 --seed 3 --rhs-seed 11 \
  --out /tmp/mf-artifact
```

真实 256JJ + 256fuv：

```bash
cd graduation-code/software
python -m src.main \
  -mtx example/256X256JJ.mat \
  --rhs example/256fuv.mat \
  --ordering amd \
  --out /tmp/mf-256
```

运行仿真：

```bash
graduation-code/systemc/build/system_sim \
  --artifact /tmp/mf-256/manifest.json \
  --config graduation-code/systemc/config/default.json \
  --mode both \
  --out /tmp/mf-256-result \
  --seed 1
```

加入 `--vcd` 会在结果目录写出 `system_sim.vcd`。

## 配置

`config/default.json` 覆盖：

- 时钟、tile、buffer 数量/容量和 FIFO 深度；
- DDR 基础延迟、带宽、burst、outstanding、抖动和反压；
- Panel/TRSM/GEMM 单元数量、启动延迟和吞吐率；
- `q_use_bits`、`frac_bits`、`accumulator_bits`、`workspace_guard_bits` 和
  `vector_use_bits`；
- 定点小主元阈值、高精度救援模式/吞吐率；
- 迭代求精开关、最大轮数、目标 residual、最小改进率和 residual MAC 吞吐率；
- FP64 主元阈值、调度策略、seed 和超时。

`q_use_bits` 是 QAU/计算输出的有效位宽，可以与软件源产物的初始有效位宽不同，用于量化
敏感性实验。输入源仍保留 manifest 中记录的原始 exponent 和精度。

默认数值参数为 `q_use_bits=26`、`frac_bits=20`、64-bit accumulator、8 个工作区
guard bits、55 个向量有效位；软件输入源默认保留 30 位。默认允许 FP64 precision
rescue，并启用最多 50 轮、原方程 residual 目标 `1e-3` 的迭代求精。

## 结果文件

每次运行输出：

| 文件 | 内容 |
|---|---|
| `summary.json` | 原/均衡方程残差、后向误差、求精历史、救援计数、配置和周期 |
| `nodes.csv` | 每节点 U/update 指数、主元比、增长率、救援、量化风险和阶段周期 |
| `operations.csv` | 七类分解算子、救援、前代/回代和 residual 的执行时间 |
| `memory.csv` | DDR 字节、burst、等待周期和带宽利用率 |
| `timeline.csv` | buffer、kernel、依赖和写回事件 |
| `solution.csv` | 排序前后索引、参考解、FP64 解和定点解 |
| `final_memory_image.bin` | 写回 L/U/P/update/solution 后的 DDR 快照 |

## 批量实验

默认采用“单因素变化”，避免意外启动巨大的笛卡尔积：

```bash
python graduation-code/systemc/scripts/run_sweep.py \
  --system-sim graduation-code/systemc/build/system_sim \
  --artifact /tmp/mf-artifact/manifest.json \
  --base-config graduation-code/systemc/config/default.json \
  --out /tmp/mf-sweep \
  --mode both
```

`--quick` 运行 8 个代表配置；只有显式指定 `--cartesian` 才运行完整参数组合。

## 当前真实数据结果

以下结果由 2026-07-25 的默认配置、`seed=1` 得到；relative residual 均以未均衡的原始
方程 `||b-Ax||₂/||b||₂` 计算。

| 矩阵 | 模式 | 状态 | 原方程 residual | 解相对误差 | 救援节点 | 求精 |
|---|---|---|---:|---:|---:|---|
| 256JJ | FP64 | ok | `1.349e-12` | `3.825e-8` | — | — |
| 256JJ | fixed | ok/达标 | `5.377e-4` | `2.904` | 2 | 2 轮 |
| 576JJ | FP64 | numeric failure | — | — | — | node 156 小主元 |
| 576JJ | fixed | ok/未达标 | `2.346e-2` | `2.362` | 13 | 修正非下降 |
| 1024JJ | FP64 | ok | `2.519e-12` | `4.361e-10` | — | — |
| 1024JJ | fixed | ok/达标 | `9.434e-4` | `1.394e-3` | 1 | 40 轮 |

这里的“ok”表示控制流、分解和求解正常结束，不等于满足精度目标；`summary.json` 中另有
`accuracy_target_met`。576 定点已经从原先的分解/求解失败变为可完成，但 `2.346e-2`
尚未达到 `1e-3`，仍是 delayed pivot、动态 front、tile 级缩放或更强预处理的后续研究
对象。256 的 residual 达标但解误差仍大，说明该问题病态：小残差不能单独证明解准确。

FP64 576 的严格基准使用 `pivot_rel_tol=1e-12`，最终主元相对 front norm 约
`3.664e-14`，因此有意报告数值失败；定点救援路径使用更低的 `1e-16` 阈值以继续完成
压力实验。所有失败都通过 `NodeStatus` 结束，没有以超时或死锁表现。

## 代码结构

```text
systemc/
├── config/default.json
├── docs/ABI_v2.md
├── include/
│   ├── artifact.hpp          manifest、DDR、配置与严格校验
│   ├── full_system.hpp       Task/Scoreboard/Buffer/Assembly/Kernel/Writer
│   ├── numeric_kernels.hpp   FP64/定点分解与 Micro Scheduler
│   ├── solve_controller.hpp  树形前代/回代与误差
│   ├── report.hpp            JSON/CSV/solution 输出
│   ├── system_memory.hpp     front、child map、QAU 和因子语义存储
│   ├── atu.hpp / hpu.hpp     逐周期关键硬件模型
│   └── node_task_codec.hpp   128 字节 ABI v2
├── scripts/run_sweep.py
├── tests/run_e2e.py
└── src/
    ├── system_sim.cpp
    ├── atu_hpu_demo.cpp
    └── model_unit_tests.cpp
```

ABI 细节见 [docs/ABI_v2.md](docs/ABI_v2.md)。

本目录固定 vendoring `nlohmann/json v3.12.0`，官方 single-header SHA-256 为
`aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`，
许可证保存在 `third_party/nlohmann/LICENSE.MIT`。
