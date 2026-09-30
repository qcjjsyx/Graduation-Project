# ControlPlane 设计说明

本文按“架构 → 命令交互 → 模块接口 → 实现机制”的顺序解释当前控制面，供对照代码阅读。代码位于 `graduation-code/systemc/include/controlplane/` 与 `src/controlplane/`，命名空间为 `hw::controlplane`。

任务范围与历史验收见 [T05 任务记录](T05_SystemC_控制面设计.md)；搬运与片上访问见 [DataPlane 设计](DataPlane_设计.md)；外部格式见 [Command schema v2](../interface/Command_Schema_v2.md)。正文的当前执行路径是 T05 test stub；T06 独立原型与 T07 迁移点单列在第 8 节。

## 1. 控制面解决什么问题

控制面从设备内存取 Command，读取其 descriptor，检查 token 依赖，为命令分配 front、执行单元、DMA 和 buffer 访问权限，最后写 completion 并发布 token。它回答“哪条命令现在能开始、占哪些资源、何时对后继可见”。

输入为已校验的 `LoadedArtifact` 和 `ControlPlaneConfig`；输出为设备 completion 区、token 终态、batch 状态及执行 trace。FP64 reference 不参与派发。依赖只来自命令 wait token，包括 NODE_BEGIN 对子节点 NODE_COMMIT 的显式等待，设备不根据 Host 消除树补推依赖。

当前 `CommandDevice` 必须显式启用 `test_stub`。它验证调度和搬运，不提供正式 LU/TRSM/GEMM/solve 数值结果。报告始终标记 `execution_backend=test_stub`、`numerical_validation=not_performed`；`RequireRealKernels` 在启动时拒绝。T06 原型存在不改变这一后端边界。

## 2. 总体结构、所有权与调用关系

```text
SystemC 顶层 / 有界测试驱动
                │ start / 每拍 tick / abort_node / reset
                ▼
          CommandDevice（统一编排者，持有以下对象和状态）
                │
  CommandFetch ──已解码命令──> 调度窗口 / Tracker
                                 │    ▲
                                 │    └── TokenScoreboard.check / signal
                                 │
                         FrontSlot + 资源分配逻辑
                           │                    │
                           ▼                    ▼
                      DmaArbiter          TestStubUnit 池
                           │                    │ ExecResponse
                           │ Completion         ▼
                           └──────────> 输出完成 / PostExec
                                              │ completion record
                                              ▼
                                       CompletionWriter
                                              │ 写确认
                                              ▼
                                      signal READY / FAILED

CommandFetch / CompletionWriter / P-vector 写 ──> DataPlane.MemoryService
描述符解析 ──> DataPlane.DescriptorReader
DmaArbiter ──> DataPlane.DmaEngine
资源分配/执行单元 ──> DataPlane.BufferManager（handle + lease）
```

“调度窗口”“front 管理”“资源分配”是 `CommandDevice` 内部的数据结构和方法，当前没有对应的独立 Scheduler/FrontManager 类。`IExecutionUnit` 是接口；现有池的具体实例与管理代码仍绑定 `TestStubUnit`，因此后续共享 PE 接入不是替换一个虚函数实现就结束。

双 front 共用所有 kind 的单元池、一个 DMA 引擎和一个 MemoryService。每个 front 持有一个数据槽、一个暂存槽；另有一个 solve 暂存槽，默认合计五槽。一个 front 的冲突命令保守串行，不同 front 的执行可以重叠。

## 3. 沿一条 LOAD_FRONT 看命令如何完成

假设 NODE_BEGIN 已准入该节点并分配 front/temp 槽：

1. `CommandFetch` 经 MemoryService 读取一条 32B command，响应先进入 inbox，再由 `update` 解码。`CommandDevice::dispatch_phase` 从 fetch FIFO 抽取命令，建立 Tracker。
2. Tracker 从 InWindow 转 Resolving；控制面向 DescriptorReader 请求主 descriptor 与依赖 payload。解析结果返回后进入 Waiting。
3. TokenScoreboard 检查依赖。PENDING 继续等；FAILED 生成 DEPENDENCY_FAILED；全 READY 才尝试资源获取。节点已 abort 时先按 ABORTED 处理。
4. 控制面确认本 front 不 busy，构造 `DmaArbiter::Job`。有限 DMA 队列接收后命令进入 Running；队列满则保留状态，下拍重试，不把它报告为容量错误。
5. DmaArbiter 从不同 front 的队列工作中轮转选择，驱动单活动 DmaEngine。实际“读 DDR → 写片上 buffer”见数据面文档第 3 节。
6. DMA job 返回终态后，控制面释放本命令资源，记录输出完成拍，进入 PostExec，并尝试把 completion 入队。completion 队列满时保留结果，不重跑 LOAD。
7. CompletionWriter 把 64B 记录写到 `completion_base + command_id×64`。只有内存写确认后，控制面才将该命令 token 置 READY 或 FAILED，并使 Tracker 进入 Done。

对 PANEL_LU，步骤 4–5 改为向执行单元提交 ExecRequest；成功响应之后还要写完整 identity P-vector 并等待确认。对 STORE_FACTOR，需要 L/U 两个 DMA job 都落定。无论具体 opcode 是什么，都遵循：

```text
输出数据写入完成 → completion 写入确认 → token READY/FAILED
```

这三个时刻不能合并。否则后继命令可能先看到 READY，再读到尚未写完的输出。

## 4. 模块职责与公开接口

| 模块与代码入口 | 职责/持有状态 | 主要接口和调用者 | 接受/完成/背压 |
| --- | --- | --- | --- |
| [CommandDevice](../../graduation-code/systemc/include/controlplane/control_plane.hpp) · [实现](../../graduation-code/systemc/src/controlplane/control_plane.cpp) | 顶层、Tracker/front 槽、单元池、命令统计 | 外部 `start/tick/abort_node/reset`；测试 `run`；查询 `state/finished/write_reports` | start 校验配置；tick 非阻塞；run 是有界驱动；命令状态与 batch 状态分别查询 |
| [CommandFetch](../../graduation-code/systemc/include/controlplane/command_fetch.hpp) · [实现](../../graduation-code/systemc/src/controlplane/command_fetch.cpp) | 单个未决读取、有界已解码 FIFO、取指故障 | 顶层 `update/tick/at/remove_at`；内存路由 `deliver_memory_response` | 满时停止新取指；可按索引抽取已解码项；finished 需同时考虑读取和 FIFO |
| [TokenScoreboard](../../graduation-code/systemc/include/controlplane/token_scoreboard.hpp) · [实现](../../graduation-code/systemc/src/controlplane/token_scoreboard.cpp) | token 状态与唯一 producer | 顶层 `init/register_producer/check/signal` | check 返回 Ready/Pending/Failed；signal 检查 producer 和重复终态；无访存延迟 |
| [DmaArbiter](../../graduation-code/systemc/include/controlplane/dma_arbiter.hpp) · [实现](../../graduation-code/systemc/src/controlplane/dma_arbiter.cpp) | 有界 job 队列、一个 active job、front 轮转 | 顶层 `can_enqueue/enqueue/update/tick/take_completion` | enqueue 返回 job ID，0 为背压；queued 可 drop，active 必须排空 |
| [IExecutionUnit / ExecRequest](../../graduation-code/systemc/include/controlplane/exec_types.hpp) | command 级请求/响应契约 | 顶层 `submit/tick/poll_response/busy/abort_epoch/reset` | submit=false 为忙；poll 成功才取走终态；txn/epoch 用于识别响应 |
| [TestStubUnit](../../graduation-code/systemc/include/controlplane/test_stub_unit.hpp) · [实现](../../graduation-code/systemc/src/controlplane/test_stub_unit.cpp) | 延迟、指定数据变换、失败注入 | 顶层按 kind 选空闲实例，并使用上述执行接口 | 单实例一次一命令；显式测试模式，不完成正式数值运算 |
| [CompletionWriter](../../graduation-code/systemc/include/controlplane/completion_writer.hpp) · [实现](../../graduation-code/systemc/src/controlplane/completion_writer.cpp) | 有界写队列、未决内存写、确认结果 | 顶层 `enqueue/update/tick/take_result`；内存 sink 投递 | enqueue=false 为队列满；take_result 给 slot 与确认位；写失败成为 batch fault |

### 4.1 顶层入口

```cpp
CommandDevice(LoadedArtifact, ControlPlaneConfig);
bool start(std::string* error);            // backend 与资源校验，失败即不跑
void tick();                               // 一拍，非阻塞（SystemC 每 posedge 调用）
void run(uint64_t max_cycles);             // 仅测试驱动；超时→Faulted("batch timeout")
bool finished() const;  BatchState state() const;
bool control_flow_pass() const;            // state == Completed
void abort_node(uint32_t node_id);
void reset();                              // 之后必须重新 start()
void inject_failure(uint32_t command_id, ExecFailure);
// MemoryResponseSink（ExecUnit 客户端：P-vector 写回响应）
void deliver_memory_response(const MemoryResponse&) override;
// 统计/报告
const std::vector<CommandStats>& command_stats() const;
const std::vector<UnitBusyRecord>& unit_records() const;
const std::vector<FrontBusyRecord>& front_records() const;
const std::map<uint32_t, StatusCode>& completion_status() const;
uint32_t completion_written() const;  uint64_t cycle() const;
uint64_t dma_retry_count() const;
bool command_dma_progress(cmd, uint32_t* issued, uint32_t* done, uint32_t* cancelled) const;
uint32_t pending_pvector_command() const;  std::string debug_dump() const;
void write_reports(const std::string& output_dir) const;
```
### 4.2 执行请求与响应传什么

```cpp
ExecRequest { txn_id, epoch, command_id, node_id, opcode, unit_id,
              FrontView/FactorView/ContributionView/KernelView/SolveView,
              front_slot/temp_slot + front_lease/temp_lease(+has_*),
              total_dim/update_dim/front_offset/temp_offset/range_bytes,
              policy, threshold_bits }
ExecResponse { txn_id, epoch, command_id, node_id, unit_id,
               status, reason, pivot_count, overflow_count,
               bytes_read, bytes_written, start_cycle, finish_cycle,
               needs_pvector_write, pvector_dim }
class IExecutionUnit { kind(); unit_id(); submit(req,cycle)->bool; tick(cycle);
                       poll_response(out)->bool; busy(); abort_epoch(epoch,cycle);
                       reset(); };
```

- `submit` 返回 `false` = 单元忙（stall），不是错误。
- `poll_response` 只在响应“可见”后取走（`response_hold_cycles` 建模响应背压），
  响应在被取走前内容稳定；每个已接受事务恰好一个终态响应。
- `abort_epoch(epoch)` 丢弃该 epoch 的活动事务与其响应 ⇒ reset/abort 后旧响应失效。
- `StubConfig`：`latency_by_opcode`/`default_latency`、
  `transform ∈ {None, Pattern}`（`None` = 数据透传，使 STORE 写回内容可被测试
  从 image 独立预测；`Pattern` = 以 `command_id` 填充声明区间，用于多实例区分）、
  `pattern_bytes`、`response_hold_cycles`；`inject_failure(command_id → {status,
  pivot_count, overflow_count})`。


ExecRequest 的 typed view 说明矩阵布局，handle 标识存储，lease 授予本次访问权限。单元不能把这些参数换成 Host 指针，也不能自行跳过 BufferManager 的访问时延。请求被接受后由单元持有事务，响应中的 status 才表示执行结果。

## 5. 内部实现：从一拍推进到命令调度

### 5.1 固定相位与 batch 完成

`CommandDevice::tick()`：

| # | 相位 | 做什么（每模块每拍至多推进一次 ⇒ 无零时间依赖链） |
| --- | --- | --- |
| 1 | `plane_.begin_tick()` | `++cycle`（cycle 约定同数据面第 5.1 节） |
| 2 | `plane_.tick_memory_and_route()` | 内存完成 + 按 client 投递响应到各模块 inbox（只入队） |
| 3 | `update_phase()` | 响应→状态迁移：fetch 解码入窗口、completion 结果→token、DMA job 落定、单元响应、descriptor 解析完成、P-vector 写落定、依赖判定与 abort/等待分类 |
| 4 | `dispatch_phase()` | 从 fetch FIFO 按序抽取入窗口（准入门控 + parked 不计数）、启动 descriptor/dependency 读取、按流序尝试派发 |
| 5 | `issue_phase()` | 新请求发出：`fetch.tick`、`completion_writer.tick`、`dma_arbiter.tick` |
| 6 | `plane_.tick_clients()` | `DescriptorReader::tick` 与 `DmaEngine::tick` |

批次完成判定：`fetch_.finished() && completed_commands_ == total_commands_`
且所有 tracker `Done` 且 `dma_arbiter_.idle()` 且无未决 exec 响应
⇒ `BatchState::Completed`。

- `CP-I-4` 相位次序固定并写入本文；审查时对照 `tick()` 实现逐条核对。

### 5.2 Tracker 生命周期

`Tracker::Phase`：

```text
InWindow ──start_resolution──▶ Resolving ──(主 descriptor + dependency payload 完成)──▶ Waiting
   Waiting ──(先查 aborted，再查依赖)──▶ Failed(DEPENDENCY_FAILED/ABORTED) 或 try_execute
   try_execute: 资源原子获取 + 回滚 ──▶ Running ──▶ PostExec ──▶ Completing ──▶ Done
                                    │             │
                        单元/DMA/P-vector 在途   输出写完、completion 入队
```

关键规则

- `CP-I-5` **原子资源获取 + 回滚**（`acquire_exec_resources`）：单元与 buffer
  lease 必须一起成功；任一失败即整体回滚并在后续拍重试，计 `stall_resource`；
  命令**不得持有一部分资源去等另一部分**。
  验证：`test_dma_port_contention_retry`、`test_single_unit_contention`。
- `CP-I-6` 暂时 vs 永久严格区分：端口/冲突竞争、DMA 队列满、completion 队列满
  = stall（`stall_resource/stall_dma`），**不**报终态 `BUFFER_FULL`，不计数值 retry
  （`retry_count` 恒为 0）；只有硬容量不足或句柄/状态非法才终态。
  验证：`test_completion_backpressure`、`test_buffer_capacity_failure`、
  `test_concurrent_assembly_dma_retry`。
- `CP-I-7` 每个已接受命令恰好一个终态 completion；`finish_command` 幂等
  （以 `finish_cycle != 0` 判重），completion 入队失败时留在 `PostExec` 逐拍重试。
  验证：`test_completion_backpressure`、`controlplane_artifact_tests`
  （`completions == commands`）。
- `CP-I-8` 依赖失败即 `DEPENDENCY_FAILED`，不占用单元、不做数据访问；其 signal
  token 仍进 `FAILED`，失败沿 token 持续传播（后继不会永久停留）。
  验证：`test_failure_propagation`、`test_dynamic_panel_refusal`。
- `CP-I-9` 被 abort 节点的未终结命令一律 `ABORTED`（先于依赖判定），token `FAILED`；
  已完成命令保持原状态 ⇒ abort 语义与时机无关、可被独立解释器复算。
  验证：`test_abort_and_reset`、`test_abort_with_inflight_dma`、
  `test_abort_during_solve_staging`、`test_abort_during_pvector_write`。

### 5.3 双 front 与命令资源

```cpp
ControlPlaneConfig {
  max_resident_fronts{2},                       // 1 = 对照, 2 = 双 front
  panel_unit_count{1}, trsm_unit_count{2}, gemm_unit_count{1},
  solve_unit_count{1}, assemble_unit_count{1}, // 两 front 共享同一池
  buffer_slot_count{5}, buffer_capacity_bytes{64KiB}, buffer_port_count{4},
  command_window_depth{16}, completion_queue_depth{16}, dma_queue_depth{8},
  backend{TestStub}, enable_test_stub{false}, StubConfig stub,
  MemoryServiceConfig memory, DmaConfig dma,
}
```

- `CP-I-10` 执行单元是**共享池**：`units_` 按 kind 匹配，选择遍历到的第一个空闲单元，
  不存在“单元 A 只服务 front A”的硬编码。
  验证：`test_two_roots_dual_front`（panel/gemm 各 2 实例同时活动）、
  `test_single_unit_contention`（单实例时同 kind 区间不交叠 = 排队）。
- `CP-I-11` 首版保守串行化**同一 front** 的 front-touching 命令
  （`FrontSlot.busy`，覆盖 LOAD/ASSEMBLE/PANEL/TRSM/GEMM/STORE_*），
  写冲突交由 T04 lease 检查兜底；**不同 front 必须可并行**。
  验证：`test_ping_pong`（数据面）+ `test_dual_front_overlap`（控制面区间交叠）。
- `CP-I-12` 槽位需求 = `2 × max_resident_fronts + 1`
  （每驻留 front 一个数据槽 + 一个装配/搬运暂存槽，外加一个共享 solve 暂存槽）。
  最初“4 槽”方案不足：solve 与驻留 front 并发时会耗尽槽位而死锁，故按实际布局
  上调（默认 5 槽对应默认 2 驻留）。每命令数据/临时需求见第 5.3 节表。
- `CP-I-13` 单 DMA 通道由 `DmaArbiter` 按 front 标签轮转公平共享；
  未为“双 front”强行实现双 DMA。
  验证：`dataplane_systemc_tests.cpp::test_no_lost_or_duplicated_transfers` +
  `controlplane_systemc_tests.cpp::test_dual_front_overlap`（DMA 与对端计算交叠）。

#### 每命令数据/临时空间（当前测试后端）

| 命令 | front 槽 | temp 槽 | DDR 流量 | 单元池 |
| --- | --- | --- | --- | --- |
| NODE_BEGIN | 分配（`total²×4`） | 分配（同尺寸） | 无 | — |
| LOAD_FRONT | 写（1 个 DMA 装载） | — | 读 front 区 | DMA |
| ASSEMBLE_EXTEND_ADD | 写（映射列 scatter） | 读（子 update 暂存） | 读子 update（1 DMA） | assemble |
| PANEL_LU | 整 front 写 lease（stub） | — | P-vector 写（pivot×4，ExecUnit） | panel |
| TRSM_LEFT/RIGHT | 整 front 写 lease（stub） | — | — | trsm |
| GEMM_SCHUR | 整 front 写 lease（stub） | — | — | gemm |
| STORE_FACTOR | 读 | — | 写 L(total×pivot)、U(pivot×total) 两段 2D | DMA |
| STORE_UPDATE | 读 | — | 写 update(update×update) | DMA |
| NODE_COMMIT | 释放 | 释放 | — | — |
| SOLVE_FORWARD/BACKWARD | — | 分块 load+store | rhs↔solution 全量分块 | solve |
| ABORT_NODE | 清理 | 清理 | — | — |

当前 `acquire_exec_resources` 对计算命令申请整 front 写 lease，`range_bytes=total_dim²×4`，并未落实真实 kernel 的子块访问边界。T07 起必须在这一授权范围内明确实际 tile 读写；目标 Panel 只写 A11，不能沿用 stub 的整 front pattern 写法。

### 5.4 启动校验

只检查 backend 不够：必然无法推进的配置必须在启动时拒绝（不留下“运行后永久
stall”的形态）。当前校验集合，任一不满足即返回 `false` 并给出原因：

1. `backend == RequireRealKernels` → 拒绝（无真实 kernel，绝不偷偷用 stub）。
2. `backend == TestStub && !enable_test_stub` → 拒绝（stub 必须显式启用）。
3. `max_resident_fronts ∉ {1,2}` → 拒绝（首版上限 2）。
4. `command_window_depth < 1`、`completion_queue_depth < 1` → 拒绝。
5. `dma_queue_depth < 2` → 拒绝（`STORE_FACTOR` 的 L/U 两个 job 需原子入队）。
6. `buffer_capacity_bytes == 0` → 拒绝；
   `buffer_port_count < 2` → 拒绝（装配同时持 front 写 + temp 读两条 lease，
   单端口是**永久**不可满足需求，不得当作暂时竞争无限重试）。
7. `buffer_slot_count < 2×max_resident_fronts + 1` → 拒绝。
8. 任一单元池（panel/trsm/gemm/solve/assemble）为 0 → 拒绝。

- `CP-I-14` 默认配置自洽（默认 5 槽 ↔ 默认 2 驻留）。
  验证：`test_default_config_runs`、`test_start_validation`。

### 5.5 有限窗口与软件命令排列

#### 软件侧（`command_compiler.py`）

- **交错宽度 2**（`INTERLEAVE_WIDTH`）：拓扑发射时最多两个节点轮转交替，
  两个独立 front 的命令在流中 O(1) 间距；更宽的森林按“对”依次展开。
- `command_id == signal_token == 稠密下标`；`completion 槽 = command_id`；
  所有 wait token 严格小于消费者 ID（Python `manifest.py` 与 C++ loader 双侧校验）。
- **NODE_BEGIN 携带子节点 `NODE_COMMIT` token 依赖** ⇒ 节点准入完全由 token 表达，
  设备不从 host 树信息推断调度关系。
- solve 段保持尾部链式（依赖链天然串行，交错无收益）。

#### 设备侧窗口规则（`dispatch_phase`）

fetch FIFO 深度 = `command_window_depth + 2`；每拍扫描 FIFO 取**第一个可抽取项**
（乱序抽取，被门控的项不堵后面的命令）：

1. front-touching 且节点未准入 且非 `NODE_BEGIN` ⇒ **留在 FIFO**（准入门控，不占窗口）。
2. 未准入节点的 `NODE_BEGIN` ⇒ 入窗口但 **parked**（不计 `command_window_depth`）。
3. 其余（已准入节点的命令、NODE_COMMIT、SOLVE_*）⇒ 受 `command_window_depth` 限制。
4. 窗口内按流序选**最早**满足“依赖 READY + 资源可原子获取”的命令；
   获取失败整体回滚、下拍重试。
5. “调度窗口”只计未派发项（`InWindow/Resolving/Waiting`），已派发命令不再占用
   窗口名额。

#### 进度保证论证

- 全部解锁者都在**更小的流位置**：依赖 producer ID 更小；资源释放由
  已派发/在途命令完成触发（`NODE_COMMIT` 或失败终结）。
- 命令不持资源等待（`CP-I-5`），单元/DMA/内存/stub 延迟有界。
- 未准入节点的命令不占窗口（parked + 门控），因此窗口不可能被“永远无法派发”
  的项填满。
- ⇒ **`command_window_depth ≥ 1` 即可推进**；`W` 只影响重叠度。
  实测 W=1/2/4/8/16/32 均完成（`test_small_window`、
  `controlplane_systemc_tests.cpp::test_window_and_completion_backpressure`）。
- 释放槽位的三条件（`maybe_cleanup_node`）：节点已见 `NODE_COMMIT`（或被 abort）、
  该节点全部 tracker 终态、**且 fetch FIFO 内无该节点的 front-touching 命令**
  ——乱序抽取下必须检查 FIFO，否则会提前释放仍被后续 LOAD 引用的槽。
- 排列约束与最小窗口已作为契约写入
  `manifest.command_batch.ordering/admission_dependencies` 并由双侧校验器把关。
- `CP-I-15` 不用无限队列、不一次性读入全部 runtime 命令：parked 项上界 =
  未准入节点数 × 该节点命令块大小（有界），fetch FIFO 与窗口均有界；
  超出即记 `window_full_cycles` 反压。

### 5.6 当前测试执行单元与主元契约

主元契约（`PANEL_LU`）

- 策略与阈值只取 `FRONT_DESC.word8/9`，绝不从 P-vector 内容推断
  （`CP-I-16`）。
- `DYNAMIC_PANEL` → 终态 `POLICY_NOT_SUPPORTED`，不占单元、不发 READY token。
- `STATIC` 成功 → 请求写**完整 identity P-vector**（`p[i]=i`，INT32，
  经 `ExecUnit` 客户端内存写）并**等写完成**才结束命令；
  `pivot_count = pivot_dim`（成功 = 全部主元处理完）。
- `STATIC` 数值失败经注入验证 `STATIC_PIVOT_FAILURE / PIVOT_UNSTABLE /
  NON_FINITE_VALUE` 传播，`pivot_count` = 首个失败列。
- 不宣称实现了“消元后实际主元检查”；不扰动主元、不中途切换策略、
  不重跑已有副作用的命令（`retry_count` 恒 0）。
  验证：`test_single_node_flow`、`test_pivot_count_and_stall_accounting`、
  `test_dynamic_panel_refusal`、`test_failure_propagation`。
- `CP-I-17` `RequireRealKernels` 是**启动级拒绝**，绝不“默默用 stub 报成功”。
  验证：`test_backend_gating`。
当前 ASSEMBLE 路径以映射搬运验证控制流，stub 不执行正式 extend-add；solve 按完整范围分块搬运，不执行三角求解。表中的 L/U/update 表示 STORE 的目标布局，不表示 stub 已算出这些数值。

## 6. completion、失败、abort 与 reset

### completion 与 token 可见性

```text
执行结束 → 输出写入完成（lease 释放 / DMA 全部落定 / P-vector 写确认）
        → completion 写入确认（固定槽 slot = command_id）
        → token READY / FAILED
```

- 完成顺序可不同于取指顺序；槽位映射不变（`CP-I-18`）。
- completion 模板的 `OK` 不算“执行完成”：batch 另有
  `completion_written()/completed_commands_` 计数。
- 写失败：单次尝试（不重复写同一坏地址）→ 顶层 `BatchState::Faulted` +
  `fault_reason()` 可见 → 该命令 token 不发 READY。
  验证：`test_completion_backpressure`（成功路径）+ `test_fetch_faults`
  （故障路径的可观察性模式）。
- 记录字段：`command_id/node_id/status/pivot_count/start_cycle/finish_cycle/
  read_bytes/write_bytes/stall_cycles/overflow_count/retry_count`。

### 失败与 abort

| 触发 | 命令终态 | token | 资源 |
| --- | --- | --- | --- |
| 依赖含 FAILED | `DEPENDENCY_FAILED` | FAILED | 不申请 |
| `PANEL_LU` + DYNAMIC_PANEL | `POLICY_NOT_SUPPORTED` | FAILED | 不申请 |
| 注入/真实数值失败 | 注入状态（含失败列） | FAILED | 正常释放 |
| 硬容量不足 | `BUFFER_FULL` | FAILED | 正常释放 |
| `abort_node(N)` / `ABORT_NODE` | `ABORTED` | FAILED | 释放（见下） |
| completion 写失败 | 命令终态保留 + batch Faulted | 不发 READY | 已释放 |

abort 细则

- 排队 DMA job 丢弃并计 `dma_jobs_cancelled`；tracker 只等**实际在途**事务
  （`done + cancelled == issued` 才终结）（`CP-I-19`）。
  未提交的分阶段工作（solve 后续 chunk、assemble stage2、待重试 job）直接取消。
- 在途 DMA 不可取消：自然完成后终结为 `ABORTED`，其写入**不声称可回滚**。
- 输出仍在途的 `PostExec`（等 P-vector 写）必须标记 aborted 并终结为 `ABORTED`
  （发 FAILED），只有“输出已完成、仅等 completion”的命令保留原状态。
- 有 DMA/计算仍引用 slot 时不得释放或复用；单节点 abort 不 reset 共享数据面。
- 一个 front 失败不影响另一独立 front（其 solve 可能因全局 solve 链依赖失败，
  如实报告）。
  验证：`test_abort_with_inflight_dma`、`test_abort_during_solve_staging`、
  `test_abort_during_pvector_write`、`test_abort_and_reset`。

### 全局 reset

`CommandDevice::reset()`：`DataPlane::reset()`（buffer/memory/reader/DMA + epoch）
+ fetch/completion/arbiter/unit 复位 + scoreboard 重建 + trackers/FIFO/上下文/
统计清空 + `epoch_++` ⇒ 旧响应与旧句柄一律失效；状态回 `Idle`，
**必须重新 `start()` 才跑批次**，不会重放旧请求。
验证：`test_abort_and_reset`（reset 后重启完成）。
## 7. 观测、测试与当前限制

### 7.1 时间与字节统计

| 指标 | 定义 |
| --- | --- |
| `fetch_cycle` | 记录解码进窗口当拍 |
| `admit_cycle` | tracker 建立当拍 |
| `start_cycle` | 资源获取成功 / 单元提交或 DMA 入队当拍 |
| `finish_cycle` | **输出写入完成**拍（lease 释放 + 全部 DMA + P-vector 写确认） |
| `completion_cycle` | completion 写入确认拍 |
| `stall_dep / stall_resource / stall_dma` | **互斥逐拍**等待分类（每拍最多计一类） |
| `stall_cycles`（completion 记录） | `stall_dep + stall_resource + stall_dma` |
| `stall_window` | `start_cycle - fetch_cycle` 的整体驻留时间，**单独报告、不与细分相加** |
| `bytes_read / bytes_written` | 该命令的 DMA/内存/片上数据字节，控制流量单列（`memory_service`、`descriptor_reader`、`completion_writer` 计数） |
| 单元占用 | `units.csv` 的 `[start, finish)` 区间（含 kind/unit_id/command_id/node_id） |
| front 占用 | `fronts.csv` 的驻留区间 |

- `CP-I-20` 不重复计时/重复计字节；不同事务不得沿用旧计数（数据面
  `DP-I-*` 与 `accept_request` 计数复位保证）。
  验证：`test_pivot_count_and_stall_accounting`、
  `dataplane_systemc_tests.cpp::test_zero_size_after_transfer`。
- `CP-I-21` 统计仅代表架构模型，不构成芯片实测性能宣称。

报告契约（`write_reports(dir)`，另含数据面 `transactions.csv/buffers.csv`）：

```text
commands.csv : command_id,node_id,opcode,unit_kind,unit_id,fetch_cycle,
               admit_cycle,start_cycle,finish_cycle,completion_cycle,status,reason,
               bytes_read,bytes_written,stall_window,stall_cycles,stall_dep,
               stall_resource,stall_dma,pivot_count,overflow_count,dma_jobs
units.csv    : kind,unit_id,command_id,node_id,start_cycle,finish_cycle
fronts.csv   : slot,node_id,occupy_start,occupy_end
summary.json : tool, execution_backend, control_flow_pass, numerical_validation,
               batch_state, fault_reason, cycles, commands, completions_written, config{...}
```

（`opcode` 列为 opcode 的十进制数值，便于与 codec 枚举对照。）

### 7.2 独立控制流参考

`tests/control_reference.hpp::interpret()` 以解码后的命令/描述符记录 + 注入表为输入，
**不复用 GCU 状态机**，按流序单遍即可（producer 恒在更早位置）：

1. 被 abort 节点的命令 → `ABORTED`；
2. 依赖含 FAILED → `DEPENDENCY_FAILED`；
3. 命中注入 → 该状态与失败列；
4. `PANEL_LU` + `DYNAMIC_PANEL` → `POLICY_NOT_SUPPORTED`；
5. 否则 `OK`；`STATIC` 成功 `pivot_count = pivot_dim`；
6. signal token：`OK → READY`，其余 `FAILED`。

比较项：各命令终态、pivot_count、completion 槽与次数、token 终态、节点生命周期、
失败传播。独立命令的**完成顺序不作比较**。
abort 场景因“abort 前已完成命令保留原状态”，测试改用不变量断言
（全部终态 + 集合 ⊆ {ABORTED, OK, DEPENDENCY_FAILED} + 独立 front 正常）。

### 7.3 实现边界

- 无正式数值 kernel：PANEL/TRSM/GEMM/ASSEMBLE/SOLVE 全部由 test stub 承接，
  数据只透传或按 `command_id` 填充；STORE_* 写回的是**测试数据**，
  不是求出的因子或解。
- 单 DMA 通道（公平共享）；同 front 一次只跑一个 front-touching 命令（保守）。
- solve 为全局串行链、共享一个 solve 暂存槽；`rhs_count>1` 时按行分块遍历。
- 单共享 DDR 端口模型：内存密集负载下双 front 的周期收益可能不显著
  （已实测单链 artifact 周期相同，如实在测试输出中报告）；并行收益由
  单元区间交叠 + 计算密集配置的周期对比共同证明。
- parked/门控项随未准入节点数增长（每节点上界为其命令块大小），非全量读入。
- 未实现：多 batch/多线程控制面、真实 DDR/AXI PHY、HPU/ATU 动态主元执行、
  INT32/BFP GEMM、RTL closure。

### 7.4 运行与验证

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build -DBUILD_TESTING=ON
cmake --build /tmp/t04-build -j8
ctest --test-dir /tmp/t04-build --output-on-failure      # 运行当前注册的全部测试，数量随任务增加
/tmp/t04-build/command_device_smoke --device-dir <DIR> --out <DIR> \
    [--max-cycles N] [--max-resident-fronts 1|2] [--window W]
```

- 时序/重叠/周期对比必须在 `controlplane_systemc_tests.cpp`（sc_clock 每拍一次
  `tick()`，所有等待有 `max_cycles` 上界）。
- 真实 artifact 由 `tools/t04_fixture_gen.py` 生成（设备目录仅 manifest + image），
  期望数据由 `control_reference.hpp` + 快照对比独立算得。

## 8. T06 原型与 T07 接入位置

T06 代码位于 `include/panel_lu/` 和 `src/panel_lu/`，包含 PanelController、TileDispatcher、MultiModePe、TileStore 和 PanelLuPrototype。T06 原型独立接收合成 PanelCommandRequest，其通过原型测试不代表真实 Panel；T07 已在其上交付正式引擎并接入 CommandDevice：

- `controlplane/panel_exec_unit.hpp/.cpp`：`PanelExecutionUnit`（共享
  `PanelEngine` 的 IExecutionUnit 适配器，`tick()` 为 no-op）与
  `FactorStoreUnit`（kind=Factor，STORE_FACTOR 真实 L/U 转换）。
- `panel_lu/panel_engine.hpp/.cpp`：共享 PE 池 + dispatcher + 每 command
  controller；真实 front 通过 BufferManager lease 读写；扫描与装载共享读
  预算（有界轮转），写回有独立写预算与 reservation；每拍由 CommandDevice
  统一 `tick()` 一次。
- `IExecutionUnit` 新增 `abort_txn`（定向取消）与 `deliver_memory_response`
  （默认空）；`abort_node` 改为定向 `abort_txn`，不再用共享 epoch 取消。
- `ControlPlaneConfig` 新增 `enable_real_panel` 与 `panel=PanelLuConfig`；
  `start()` 在 `RequireRealKernels` 下必须显式开启 real panel，两个后端都
  未开启则拒绝。未绑定 unit 的 kind（TRSM/GEMM/assemble/solve）在派发前
  返回 `POLICY_NOT_SUPPORTED`，不落 stub、不永久等待。
- P-vector 仍由 PostExec 写完整 identity 并等确认；真实引擎只返回
  `needs_pvector_write`。
- `summary.json` 增加 `panel_backend_real_fp32`、`mixed_backend_explicit`、
  `numerical_scope` 与 `panel_engine` 统计；真实引擎运行时额外输出
  `panel_tasks.csv`（task/PE/mode/周期/字节/提交状态）。
- `reset()` 额外复位共享引擎；失败/abort 后在途任务安全排空，旧响应由
  `handle_exec_response` 的终态守卫丢弃。
- 定向 abort：`abort_node` 调用 `unit->abort_txn` 后**保留 tracker 与 lease**，
  等执行单元产出唯一终态响应（Factor 单元还会等已接受的 DDR 写确认）才释放
  资源、结束命令；`tick()` 的 Completed 判定额外要求所有执行单元 `!busy()`、
  DMA 空闲且 `exec_inbox` 为空，避免“取消已请求”被当成“执行已排空”。
- `STORE_FACTOR`：绑定正式 Factor 单元时忙则等待（`stall_resource`），旧
  DMA 矩形复制只在显式 test-stub 后端（无 Factor 单元）运行。
- 内存响应按 request id 精确路由：`IExecutionUnit::handles_request` 决定属主，
  单实例只看到自己的未决写；未决集合为空且无新写时才返回终态。
- T07 测试：`panel_lu_real_tests`（206 checks，含数值/失败/双 front/abort、定向 drain、Factor 路由/背压/FP64 诊断/
  reset/背压）与 `panel_lu_real_systemc_tests`（sc_clock 集成）。

目标路径如下，T07 已完成其中 Panel 分支的迁移：

```text
CommandDevice：command 依赖/准入/lease/失败传播/completion
    └── ExecRequest ──> Panel/TRSM/GEMM controller（command 级）
                            └── TileTask ──> 共享 TileDispatcher
                                                 └── pe_count 个 MultiModePe
                            <── task 写回确认 / 失败 ────────────┘
        <── 一个 command 终态响应（所有所属任务安全结束）
    └── 输出确认 → CompletionWriter → token
```

Panel 内部的 TRSM/GEMM task 与外部 TRSM/GEMM command 是两层概念：内部 task 只服务 A11 的分块 LU；外部命令继续处理 A12/A21/A22。全 front 扫描在消元前进行，固定 maxabs 和阈值；STATIC 不搜索或交换行，HPU/ATU 留到可选的 T14。

| 当前代码接缝 | 接入要做的工作 | 保留的上层契约 |
| --- | --- | --- |
| `units_`、kind 数量配置、实例选择 | 从按 kind 的 TestStubUnit 池迁移为 controller + 统一物理 PE 池 | 双 front 共享资源，同一 PE 不被不同模式重复分配 |
| `acquire_exec_resources` 与 ExecRequest | 授予真实 front 布局和 lease；tile 访问映射到数据面端口 | 原子获取/回滚、owner 与生命周期 |
| 执行响应处理 | 多个 task 汇聚为一个 command 终态 | 输出写完才进入 completion；一命令一终态 |
| `abort_node` / 清理路径 | 按 command/node 定向取消排队 task，排空在途访问 | 不 reset 共享数据面，不误伤另一 front |
| P-vector PostExec | 真实 Panel 成功后完整 identity 写入并等确认 | 失败不发布 READY，不静默改策略 |
| `units.csv` 与统计 | 增加 task/PE/mode 维度，保留 command 层时刻 | 不重复计时，不把原型周期当硬件加速比 |

### 8.1 接入验收清单

1. 保留 command 请求/终态响应契约；Panel/TRSM/GEMM controller 生成内部 tile task，共用 pe_count 个多模式 PE。不得仅按 kind 替换 stub 后继续宣称共享物理计算池。
2. 显式迁移 TestStubUnit 类型绑定、按 kind 池、单响应处理和统计到 command/task 两级结构；多个 task 完成后汇聚一个 command 响应。统一 PE 不得被重复分配，node abort 定向取消所属任务，不能按共享 epoch 误伤另一 front。
3. 输入参数一律取自 `ExecRequest` 的 typed view + `BufferLease`（范围/布局已校验）。
4. 主元与失败：`STATIC` 在 kernel 内做消元后实际主元检查，
   返回 `STATIC_PIVOT_FAILURE/PIVOT_UNSTABLE/NON_FINITE_VALUE` 与首个失败列；
   `DYNAMIC_PANEL` 未实现前继续返回 `POLICY_NOT_SUPPORTED`。
5. 输出契约：PANEL 成功必须 `needs_pvector_write`（或自行写并等确认）+
   `pivot_count`；STORE_* 的写回范围/布局由第 5.3 节表决定，不得整 front 复制。
6. 统计：`bytes_read/bytes_written/start_cycle/finish_cycle` 语义保持；
   不得引入 `retry_count`（首版无重跑）。
7. 同步 `tests/control_reference.hpp`（新增真实语义对应的期望规则）与
   `SystemC_模型规范.md` §4“计算模块的真实性标准”。
8. 每接入一个 kernel：重跑 controlplane 三套测试 + 数据面回归 + sanitizer，
   并重新验证 `STATIC` 的实际主元检查、数值结果、资源需求与时序；INT32/BFP 不属于当前 FP32 接入范围。

### 8.2 与数据面的接口对照

| 控制面需求 | 数据面提供 | 不变量 |
| --- | --- | --- |
| 取指/写回的内存事务 | `MemoryService` + `MemoryClient::{CommandFetch,CompletionWriter}` + sink | `DP-I-11/12/13` |
| descriptor/dependency 解析 | `DescriptorReader`（异步、无 cache） | `DP-I-7/14` |
| 片上 buffer 与冲突 | `BufferManager`（lease/transient/transfer_owner/rearm） | `DP-I-17~20` |
| 数据搬运 | `DmaEngine` + `DmaArbiter`（owner 可配、retryable） | `DP-I-21~25` |
| 追踪 | `TraceRecorder` + `write_reports` | `DP-I-27/28`、`CP-I-20` |
## 附录 A：不变量与测试索引

| ID | 不变量（摘要） | 验证 |
| --- | --- | --- |
| CP-I-1 | 只经数据面事务接口 | `controlplane_core_tests.cpp` 各 `check_data` |
| CP-I-2 | 新客户端必须注册 sink | `test_fetch_faults`、`test_completion_backpressure` |
| CP-I-3 | 运行时禁 `run_until` | 代码走查 + `test_timeout`（有界） |
| CP-I-5 | 原子获取 + 回滚 | `test_single_unit_contention`、`test_dma_port_contention_retry` |
| CP-I-6 | stall ≠ 终态 BUFFER_FULL | `test_completion_backpressure`、`test_buffer_capacity_failure` |
| CP-I-7 | 一命令一终态 completion | `controlplane_artifact_tests`（completions==commands） |
| CP-I-8 | 依赖失败不占单元 + 持续传播 | `test_failure_propagation` |
| CP-I-9 | abort 命令一律 ABORTED、token FAILED | `test_abort_during_pvector_write` |
| CP-I-10 | 单元池共享 | `test_two_roots_dual_front`、`test_single_unit_contention` |
| CP-I-11 | 同 front 串行、异 front 并行 | `test_dual_front_overlap` |
| CP-I-12 | 槽位 = 2r+1 | `test_start_validation`、`test_abort_during_solve_staging` |
| CP-I-14 | 默认配置自洽可运行 | `test_default_config_runs` |
| CP-I-15 | 无无限队列 | `test_small_window`、`test_window_and_completion_backpressure` |
| CP-I-16 | 策略只来自 FRONT_DESC | `test_dynamic_panel_refusal` |
| CP-I-17 | 缺 kernel 启动拒绝 | `test_backend_gating` |
| CP-I-18 | 槽位 = command_id、完成序自由 | `controlplane_artifact_tests` |
| CP-I-19 | 只等在途事务 | `test_abort_with_inflight_dma` |
| CP-I-20 | 统计不重复 | `test_pivot_count_and_stall_accounting` |

## 附录 B：跨层访问约束

- `CP-I-1` 控制面**不直接**访问 `DeviceMemory::image()`、`BufferManager` 存储或
  预解码的 manifest 内容；取指、descriptor 解析、数据搬运、completion 写回
  全部经数据面事务接口。
  验证：`controlplane_core_tests.cpp::test_single_node_flow`（数据契约由测试侧
  独立计算）、`dataplane` 侧 `descriptor_reads>0` 类断言。
- `CP-I-2` 三类新内存客户端 `CommandFetch / CompletionWriter / ExecUnit`
  经 `DataPlane::set_memory_sink` 注册路由；**不得**落入被丢弃的 `Test/Other` 分支。
  验证：`test_fetch_faults`、`test_completion_backpressure`、
  `test_abort_during_pvector_write`（ExecUnit 写回路径）。
- `CP-I-3` 运行时状态机从不调用会内部推进时间的
  `DataPlane::run_until()/run_dma_request()`；这些只是测试驱动便捷接口。
  `CommandDevice::run()` 是**有界**测试驱动器（超时→batch `TIMEOUT` 故障）。
