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
  `TileDispatcher`/`MultiModePe` 复用），T09 起该引擎同时服务顶层
  TRSM（`trsm_controller`、PE 分组 `compute_group_count × pes_per_group`、
  设备级 `OnChipArbiter` 统一读/写预算、完整 L/U 保存），T10 起同时服务
  顶层 GEMM_Schur（`gemm_controller`、`gemm_types` 统一
  `GemmReq/GemmRsp`、`precision_adapter` 局部 INT32 量化/INT64 累加），通过
  `controlplane/panel_exec_unit` 接入 `CommandDevice` 真实 front lease，
  设计见 `docs/systemc/T07_Panel_LU数值引擎设计.md`、
  `docs/architecture/TRSM架构设计.md` 与
  `docs/architecture/T10_GEMM_Schur与局部INT32设计.md`；
- `controlplane/assembly_exec_unit`：T11 真实 FP32 extend-add，child update
  DDR→DMA temp→运行时 map→parent front 累加，12 B 数值 staging，共用统一
  读/写预算；双 front/树依赖/abort/reset 沿用 CommandDevice；
- `src/tree_factor_device.cpp` / `reference/tools/tree_factor_check.cpp`：
  严格真实后端 artifact 分解入口和独立 FP64 因子/update checker；
- `tools/t11_fixture_gen.py` / `tests/assembly_real_tests.cpp` /
  `tests/tree_real_tests.cpp` / `tests/tree_real_systemc_tests.cpp`：两种设备后端
  的单节点/树/forest/映射/尾块/失败与 sc_clock 回归；
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
- `tests/trsm_real_tests.cpp`：T09 顶层左右 TRSM 测试（手算案例、n=42
  对 FP64 三角 oracle、分组、带宽仲裁、完整因子、失败/abort）；
- `tests/gemm_real_tests.cpp`：T10 顶层 GEMM_Schur 测试（FP32 手算/多 tile/
  多 K 步/偏移/guard、INT32 原始与 tile 内核、scale/舍入/饱和/溢出、
  失败/abort/reset/背压、分组、artifact 集成、跨语言 INT32 设备目录、
  STATIC 主元传播）；
- `tools/t10_fixture_gen.py`：生成跨语言 INT32 GEMM 设备运行目录（生产
  Command Compiler，手写 compile IR），包括默认配置与 4 位/K=8/转换速率=8
  的配置传递回归；模型参数从 manifest 在 boot 时校验并应用，单命令 SCALE_DESC
  仍通过设备事务读取，显式非默认设备参数与 manifest 冲突时拒绝运行；
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
PYTHONDONTWRITEBYTECODE=1 ctest --test-dir /tmp/t04-build --output-on-failure
```

sanitizer 构建：

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build-sanitize \
  -DBUILD_TESTING=ON -DENABLE_SANITIZERS=ON
cmake --build /tmp/t04-build-sanitize -j8
PYTHONDONTWRITEBYTECODE=1 ASAN_OPTIONS=detect_leaks=0 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
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
  Panel 数值/资源/异常测试；`trsm_real_tests`：T09 顶层左右 TRSM；
  `gemm_real_tests`：T10 顶层 GEMM_Schur（含局部 INT32）（T10 阶段 21 项）；
  含显式 FP64 模块诊断模式
  （`panel.numeric_mode=fp64_diagnostic`，默认 FP32，summary 标记；
  与 INT32 GEMM 组合显式拒绝）。
- T11 新增 fixture、装配、树、时钟、两模式 CLI/checker 与 CLI 失败退出
  测试。当前完整 CTest **32 项**，Release 与 ASan+UBSan 均通过。macOS
  sanitizer 关闭平台不支持的 leak detector，不影响 Address/Undefined 检查。

## T11 真实树形分解

固定实验使用 **2 组×每组 2 个 16×16 PE 模块**。整条 Panel/TRSM/GEMM 命令
独占一组，组内并行处理 tile，复用同一个 PE 池。配置可通过
`ControlPlaneConfig.panel.compute_group_count/pes_per_group/pe_count` 修改，
其中 `pe_count` 必须与前两项乘积一致。本阶段不做规模扫描。

设备入口要求 `factorization_only` artifact。运行时 `--backend` 选择必须与
编译后端和实际 GEMM descriptor 一致，不能通过运行时切换修改二进制。
从仓库根目录运行以下命令（构建完成后）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  graduation-code/systemc/tools/t11_fixture_gen.py --out /tmp/t11-fixtures
/tmp/t04-build/tree_factor_device \
  --device-dir /tmp/t11-fixtures/three_level/fp32/device \
  --out /tmp/t11-output/fp32 --backend FP32_SYSTEMC
/tmp/t04-build/tree_factor_check \
  /tmp/t11-fixtures/three_level/fp32/device /tmp/t11-output/fp32
/tmp/t04-build/tree_factor_device \
  --device-dir /tmp/t11-fixtures/three_level/int32/device \
  --out /tmp/t11-output/int32 --backend FP32_SYSTEMC_INT32_GEMM
/tmp/t04-build/tree_factor_check \
  /tmp/t11-fixtures/three_level/int32/device /tmp/t11-output/int32
```

完整软件编译 CLI 也支持分解模式。从 `graduation-code/software` 目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ../../.venv/bin/python -m src.main \
  --n 24 --density 0.16 --max-supernode-size 4 --factorization-only \
  --gemm-backend FP32_SYSTEMC --out /tmp/t11-compiled
/tmp/t04-build/tree_factor_device --device-dir /tmp/t11-compiled \
  --out /tmp/t11-compiled-report --backend FP32_SYSTEMC
/tmp/t04-build/tree_factor_check /tmp/t11-compiled /tmp/t11-compiled-report
```

真实矩阵输入可用 `--mtx PATH` 代替 `--n/--density`。局部 INT32 则将编译与
运行两个 `--backend` 对应参数都选为 `FP32_SYSTEMC_INT32_GEMM`。编译默认
不执行 host reference；默认省略 `--factorization-only` 时仍会生成 solve 链，
真实 solve 待 T12，T11 入口显式拒绝该 scope。

输出包括 `commands.csv`、`nodes.csv`、`units.csv`、`fronts.csv`、
`panel_tasks.csv`、`transactions.csv`、`buffers.csv`、`summary.json` 和
`final_memory_image.bin`。设备 `control_flow_pass` 表示控制完成，
`numeric_status_pass` 表示真实后端所有命令状态 OK，设备仍报告
`numerical_validation=not_performed`。独立 checker 才写
`validation.json.numeric_pass`，检查完整 L/U/update/P，容限 `1e-4`。
checker 使用初始 FP32 image/map，设备与 checker 均不需要 golden 文件。

T11 共 24 个树形 artifact：23 个数值通过，1 个 INT32 STATIC 主元传播
按预期失败。设备遇到失败、超时或 backend 不匹配时返回非零退出码；checker
拒绝失败批次，不把预期失败记为数值通过。求解/原方程误差属于 T12。
详细方案与本轮证据见
[T11 真实装配与树形分解集成](../../docs/architecture/T11_真实装配与树形分解集成.md)。

## 既有阶段冒烟入口

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
调度与 completion 写回。下面的旧控制流冒烟仍显式使用 test stub，结果标记
`execution_backend=test_stub`、`numerical_validation=not_performed`。
控制流冒烟：

```bash
/tmp/t04-build/command_device_smoke \
  --device-dir /tmp/t04-build/t04_fixtures/multiroot_static/device \
  --out /tmp/t05-output/smoke
```
