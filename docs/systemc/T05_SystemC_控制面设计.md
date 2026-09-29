# T05：SystemC Command 控制面（任务记录）

状态：**已完成并通过三轮审查修复**。设计语义的权威说明见
[ControlPlane_设计](ControlPlane_设计.md)（分层与相位、命令生命周期、双 front
调度与窗口进度保证、执行接口契约、completion/token、失败/abort/reset、统计口径、
不变量↔测试对照表、kernel 接入清单）。本文只保留任务视角：范围确认、实现清单、
测试证据、审查修复史与任务边界。

## 1. 任务范围确认

实现控制面：固定宽度 Command Fetch、token scoreboard、双 front 驻留与共享单元池
调度、按 descriptor 派发、completion 写回与失败传播、abort/reset 与统计。

**不做**（也不得宣称）：Panel-LU/TRSM/GEMM/assemble/solve 的**正式数值 kernel**、
DYNAMIC_PANEL 设备执行、FP32 数值闭环、RTL。允许且仅允许**显式启用的测试执行单元**
验证调度与错误传播，报告必须带
`execution_backend=test_stub / control_flow_pass / numerical_validation=not_performed`。
数据面复用 T04，不绕过其访存与 buffer 校验。

## 2. 实现清单

| 模块 | 文件（`graduation-code/systemc/`） | 权威设计节 |
| --- | --- | --- |
| CommandFetch（取指 + 窗口门控） | `include/controlplane/command_fetch.hpp`、`src/controlplane/command_fetch.cpp` | ControlPlane §2/§3/§6 |
| TokenScoreboard | `token_scoreboard.hpp/.cpp` | §3（CP-I-8/9） |
| 执行接口 + TestStubUnit | `exec_types.hpp`、`test_stub_unit.hpp/.cpp` | §7 |
| DmaArbiter（公平共享 + 重试） | `dma_arbiter.hpp/.cpp` | §4（CP-I-13） |
| CompletionWriter | `completion_writer.hpp/.cpp` | §8 |
| CommandDevice（窗口/资源/FrontContext/统计） | `control_plane.hpp/.cpp` | §2/§5/§6/§9 |
| CLI 冒烟 | `src/command_device_smoke.cpp` | §16 |
| 独立控制流参考 | `tests/control_reference.hpp` | §10 |
| 合成 artifact 构造器 | `tests/control_test_artifact.hpp` | — |
| 测试 | `tests/controlplane_core_tests.cpp`、`tests/controlplane_systemc_tests.cpp`、`tests/controlplane_artifact_tests.cpp` | §12 表 |

软件侧改动（`graduation-code/software/`）：

- `src/scheduler/command_compiler.py`：命令级拓扑发射改为**交错宽度 2**、
  `NODE_BEGIN` 增加子节点 `NODE_COMMIT` 准入依赖、`command_batch` 增补
  `ordering/admission_dependencies` 字段。
- `src/verify/manifest.py`：新增 `_validate_admission_dependencies`。
- C++ loader 同步同一准入契约：`src/dataplane/artifact_loader.cpp::validate_nodes`。
- `tests/test_command_compiler.py`：+3 用例（成对交错、准入依赖、顺序不变量）。

schema v2 与数值语义未变；无冻结接口冲突（交错重排 ID 由任务书明确授权）。

## 3. 测试与证据（2026-09）

```bash
cmake --build /tmp/t04-build -j8 && ctest --test-dir /tmp/t04-build --output-on-failure
cmake --build /tmp/t04-build-sanitize -j8 && ctest --test-dir /tmp/t04-build-sanitize --output-on-failure
cd graduation-code/software && ../../.venv/bin/python -m pytest
```

| 项 | 结果 |
| --- | --- |
| `controlplane_core_tests` | 203 checks，0 failures |
| `controlplane_systemc_tests` | 31 checks，0 failures（sc_clock 每拍一次 `tick()`） |
| `controlplane_artifact_tests` | 354 checks，0 failures |
| CTest（Release / ASan+UBSan） | 12/12 passed（含 T04 全部回归） |
| Python | 110 passed；mypy（关键模块）无错误 |
| 真实 artifact | 单节点 7/243、multi 259/8059、multiroot 110/3224、DYNAMIC_PANEL 259/6510（PANEL_LU 全拒绝）、large_256 818/227830（commands/cycles） |
| 独立解释器 | 全部命令终态、`pivot_count`、completion 槽与次数、token 终态、失败传播逐条一致（abort 场景改用不变量断言） |

双 front 重叠证据（`units.csv` 区间，test stub 延迟 300）：

```text
panel unit=0 command=4  node=0 [457,757)   ∩  panel unit=1 command=5  node=1 [537,837)
gemm  unit=0 command=10 node=0 [1369,1669) ∩  gemm  unit=1 command=11 node=1 [1449,1749)
```

控制流周期对比（**标注 test kernel，不作加速比宣称**）：

| 配置 | 合成双根（latency 20） | 真实多根 artifact（latency 100） |
| --- | --- | --- |
| 1 front / 1 unit | 1048 | 7595 |
| 2 front / 1 unit | 1021 | — |
| 2 front / 2 unit | 1021 | 6107（-19.6%） |

单链 artifact（multi_static）在共享 DDR 端口模型下周期相同，已如实报告
（无独立并行度时驻留数不影响墙钟；并行收益由区间交叠 + 计算密集配置对比证明）。

报告输出：`commands.csv`、`units.csv`、`fronts.csv`、`summary.json`
（含三字段后端标注）+ 数据面 `transactions.csv/buffers.csv`。

## 4. 审查修复史

**第一轮（6 项）**

| 问题 | 修复 | 权威不变量 |
| --- | --- | --- |
| abort 后永久等待已丢弃的 DMA job | job 提交时计数 + `dma_jobs_cancelled`，只等在途事务 | `CP-I-19` |
| 端口竞争被当终态失败 | `retryable` 安全重入队（同一逻辑 job，不重复计数） | `CP-I-6`/`DP-I-24` |
| solve 静默截断只搬前缀 | 全量分块遍历 `load→unit→store` | §7/§4 表 |
| `start()` 未校验资源 | 校验驻留/队列/槽位/单元池/容量 | `CP-I-14` |
| 成功 completion 的 `pivot_count=0` | 成功返回 `pivot_dim`，解释器同步 | `CP-I-16` |
| completion stall 重复累计 | 互斥逐拍分类相加；`stall_window` 单独报告 | `CP-I-20` |

**第二轮（3 项）**

| 问题 | 修复 | 权威不变量 |
| --- | --- | --- |
| 单端口配置遇装配永久 stall | `start()` 拒绝 `buffer_port_count<2`（永久不可满足需求不伪装成背压） | `CP-I-6`/`CP-I-14` |
| P-vector 写回在途 abort 仍发成功 token | 区分“输出在途”的 PostExec：标记 aborted → 写落定后 `ABORTED`/FAILED/释放资源 | `CP-I-9` |
| 默认配置与校验冲突（4 槽 vs 需 5） | 默认 `buffer_slot_count=5`，新增默认配置可运行测试 | `CP-I-12`/`CP-I-14` |

**第三轮（文档）**：设计语义拆为按层权威文档（`DataPlane_设计.md` /
`ControlPlane_设计.md`），本文件与 T04 文件收敛为任务视角；`commands.csv`
补 `stall_cycles/stall_dep` 列以与统计口径一致。

## 5. 保留的回归要求与任务边界

- 修改调度/abort/窗口语义必须保持：三套控制面测试全绿、T04 全绿、sanitizer 全绿、
  与 `control_reference.hpp` 的逐命令一致比较不变、双 front 区间交叠断言不变。
- 后续正式 kernel 的接入步骤与契约见 ControlPlane §15（含主元检查落点、
  `pivot_count` 语义、`RequireRealKernels` 拒绝路径、需同步的解释器规则）。
- 不得据本阶段成果宣称：数值正确性、真实主元检查已实现、硬件加速比、
  Command executor 的 kernel 侧完成度。
