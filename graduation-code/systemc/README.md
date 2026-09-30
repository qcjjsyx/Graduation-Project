# SystemC 当前主线

当前目录保留与 Command/Descriptor schema v2 主线一致的模块：

- `include/command_codec.hpp`：schema v2 固定记录 codec（含主元策略字段）；
- `include/dataplane/`、`src/dataplane/`：T04 Command 数据面（ArtifactLoader、
  DeviceMemory/RegionRegistry、MemoryService、DescriptorReader、BufferManager、
  DmaEngine、Trace、DataPlane 编排、selftest）；
- `include/controlplane/`、`src/controlplane/`：T05 Command 控制面
  （CommandFetch、TokenScoreboard、双 front 调度、共享单元池、TestStubUnit、
  DmaArbiter、CompletionWriter、CommandDevice 顶层）；
- `include/panel_lu/`、`src/panel_lu/`：Panel-LU 子系统。
  T06 部分为合成架构原型（`panel_lu_prototype`，标记
  `architecture_prototype=true`、`numerical_validation=not_performed`）；
  T07 部分为正式 FP32 STATIC 引擎（`panel_engine`、`kernels` 原语、
  `TileDispatcher`/`MultiModePe` 复用），通过
  `controlplane/panel_exec_unit` 接入 `CommandDevice` 真实 front lease，
  设计见 `docs/systemc/T07_Panel_LU数值引擎设计.md`；
- `reference/`：独立 FP64 数学参考（`PivotPolicy`/`PivotFailure`、
  `panel_lu_policy` 与通用 `panel_lu` 分离；`tools/fp64_reference_smoke`
  为数值冒烟入口）；
- `include/atu.hpp`、`include/hpu.hpp`：DYNAMIC_PANEL 用的独立模块
  （静态路径旁路）；
- `src/atu_hpu_demo.cpp`：独立模块冒烟测试；
- `src/dataplane_smoke.cpp`：T04 加载—搬运—写回冒烟入口；
- `tests/command_codec_tests.cpp`：Python/C++ 共用 fixture 的 codec 测试；
- `tests/dataplane_core_tests.cpp`：普通 C++ 数据/校验层测试；
- `tests/dataplane_systemc_tests.cpp`：SystemC 时序测试（sc_clock 驱动）；
- `tests/dataplane_artifact_tests.cpp`：真实 artifact 集成测试；
- `tests/controlplane_core_tests.cpp`：控制面核心/负面测试（含独立解释器对照）；
- `tests/controlplane_systemc_tests.cpp`：sc_clock 驱动的控制面时序测试；
- `tests/controlplane_artifact_tests.cpp`：真实 artifact 控制流集成测试；
- `tests/panel_lu_prototype_tests.cpp`：T06 原型核心测试（tile/尾块/非法配置、
  PE 排他、队列/端口/写回背压、同目标顺序、定向 abort/reset、手算时延、
  trace 可复现）；
- `tests/panel_lu_systemc_tests.cpp`：sc_clock 驱动的 T06 时序测试
  （与纯 C++ tick 模型逐周期对照）；
- `tests/panel_lu_real_tests.cpp`：T07 正式 Panel 引擎集成测试（真实
  lease 数据路径、FP64 STATIC 参考对照、失败/abort/reset/背压）；
- `tests/panel_lu_real_systemc_tests.cpp`：T07 的 sc_clock 集成测试；
- `src/command_device_smoke.cpp`：T05 控制流冒烟入口；
- `src/panel_lu_prototype_smoke.cpp`：T06 尾块/混合负载 trace 生成入口；
- `tools/t04_fixture_gen.py`：用当前软件编译器生成设备运行目录。

原先由 `NodeTask/ABI v2` 驱动的 `system_sim`、artifact loader、memory
container、fixed/global-BFP kernel 和回归脚本已经退役并归档到：

```text
archive/legacy-node-task-abi-v2/systemc/
```

## 构建与测试

SystemC 安装位置通过 `SYSTEMC_HOME` 指定。构建目录建议放在 `/tmp`：

```bash
export SYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake -S graduation-code/systemc -B /tmp/t04-build -DBUILD_TESTING=ON
cmake --build /tmp/t04-build -j8
ctest --test-dir /tmp/t04-build --output-on-failure
```

sanitizer 构建：

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build-sanitize \
  -DBUILD_TESTING=ON -DENABLE_SANITIZERS=ON
cmake --build /tmp/t04-build-sanitize -j8
ctest --test-dir /tmp/t04-build-sanitize --output-on-failure
```

CTest 会先运行 `t04_generate_fixtures`（默认使用仓库根 `.venv/bin/python`，
可用 `-DT04_PYTHON=...` 覆盖），用当前软件编译器生成单节点/多节点/多根/
DYNAMIC_PANEL/256 阶设备运行目录，然后运行：

- `dataplane_core_tests`：schema v2 加载、非法 artifact、u64 溢出/重叠/对齐、
  P-vector 负面用例、DeviceMemory 访问规则、buffer 生命周期、内存服务时序；
- `dataplane_systemc_tests`：连续/二维 DMA 逐字节正确性、位模式保持、
  固定配置周期断言、请求/响应背压、双 buffer ping-pong、reset 中断、
  非法请求无部分写入、guard 字节不变；
- `dataplane_artifact_tests`：真实 artifact 的
  descriptor → DMA → buffer → 消费者拷贝 → DMA 写回闭环；
- `dataplane_smoke`：可复现冒烟入口，输出 `transactions.csv`、
  `buffers.csv`、`summary.json`、`final_memory_image.bin`；
- `controlplane_core_tests` / `controlplane_systemc_tests` /
  `controlplane_artifact_tests` / `command_device_smoke`：T05 控制面测试与冒烟
  （输出 `commands.csv`、`units.csv`、`fronts.csv`、`summary.json`）；
- `panel_lu_prototype_tests` / `panel_lu_systemc_tests` /
  `panel_lu_smoke_tail` / `panel_lu_smoke_mixed`：T06 架构原型测试与 trace
  （合成负载、`numerical_validation=not_performed`）；
- `panel_lu_real_tests` / `panel_lu_real_systemc_tests`：T07 正式 FP32
  Panel 数值/资源/异常测试（CTest 合计 18 项）；含显式 FP64 模块诊断模式
  （`panel.numeric_mode=fp64_diagnostic`，默认 FP32，summary 标记）。

单独复现加载—搬运—写回：

```bash
/tmp/t04-build/dataplane_smoke \
  --device-dir /tmp/t04-build/t04_fixtures/single_static/device \
  --out /tmp/t04-output/single_static
```

设备运行目录只需要 `manifest.json` 与 `memory_image.bin`；golden/reference
文件不是设备数据来源，缺失不影响加载与数据面运行。

T06 原型 trace（合成负载，非数值 backend）：

```bash
/tmp/t04-build/panel_lu_prototype_smoke --mode tail --front-dim 23 --pivot-dim 23 \
  --pe-count 1 --trace-dir /tmp/t06-output/tail_p23_pe1
/tmp/t04-build/panel_lu_prototype_smoke --mode mixed --front-dim 40 --pivot-dim 40 \
  --pe-count 2 --trace-dir /tmp/t06-output/mixed_p40_pe2
```

输出 `panel_tasks.csv`、`panel_commands.csv`、`panel_events.csv`、
`panel_summary.json`（含 `architecture_prototype=true`、
`numerical_validation=not_performed`）。诊断用可选参数：
`--div-latency N`（角除法/倒数延迟，验证 TRSM_RIGHT 敏感性）、
`--task-id-base N`（uint64 task id 基址，验证跨 UINT32_MAX 不截断）。

T05 在 `dataplane` 之上实现了 Command Fetch、token scoreboard、双 front
调度与 completion 写回；数值 kernel（Panel-LU/TRSM/GEMM/solve）按后续任务
实现，当前仅以显式启用的测试执行单元（test stub）验证控制流，结果标记
`execution_backend=test_stub`、`numerical_validation=not_performed`。
控制流冒烟：

```bash
/tmp/t04-build/command_device_smoke \
  --device-dir /tmp/t04-build/t04_fixtures/multiroot_static/device \
  --out /tmp/t05-output/smoke
```
