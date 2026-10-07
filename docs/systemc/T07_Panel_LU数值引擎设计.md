# T07 Panel-LU SystemC 数值引擎设计（正式 FP32 STATIC）

状态：**T07 Panel 子系统已实现并完成数值验证**。T09 已在此基础上把
`PanelEngine` 扩展为共享计算引擎（Panel/TRSM 共用 PE 池、PE 分组、
统一片上带宽仲裁、完整 L/U 保存），见
[TRSM架构设计](../architecture/TRSM架构设计.md)。T10 已接入顶层 GEMM，T11
已接入真实 assemble/STATIC 树形分解，见
[T11 集成设计](../architecture/T11_真实装配与树形分解集成.md)；solve/Ax=b
属于 T12。本文保留 T07 的权威设计说明（架构组成
→ 模块交互 → 接口 → 实现机制 → 验证）。T06 的合成架构原型
（`panel_lu_prototype`）与其测试保持独立，本文描述的是接入 T05 真实
command/completion/token 流程的正式引擎。

---

## 1. 架构组成

```text
CommandDevice（T05 顶层，每拍一次）
  ├─ 共享计算子系统（唯一实例，engine_）
  │    PanelEngine
  │      ├─ MultiModePe[pe_count]           16×16 多模式 PE（复用 T06）
  │      ├─ TileDispatcher + TileVersionStore  有界队列/依赖/版本/公平
  │      ├─ read_budget（扫描+PE 装载共享） / write_budget / writeback reservation
  │      └─ PanelController[≤2]             每 command 一个上下文
  │           ├─ 全 front 扫描（FrontReader→lease_read，共享读预算）
  │           ├─ A11 16×16 任务图（步屏障）
  │           └─ 结果可见性/失败/abort 收敛
  ├─ PanelExecutionUnit × panel_unit_count  IExecutionUnit 适配器（共享同一 engine）
  ├─ FactorStoreUnit × factor_unit_count    STORE_FACTOR 真实 L/U 转换
  └─ TestStubUnit（仅 enable_test_stub 显式开启；未实现的 kind 不自动回退）
```

要点：

- **计算池唯一**：Panel/TRSM/GEMM 是任务模式而非物理池；`pe_count` 决定物理
  PE 数，两个 front 的 controller 共用同一 dispatcher 与 PE 数组。
- **每拍只推进一次**：`CommandDevice::update_phase` 调用
  `panel_engine_->tick(cycle)` 一次；`PanelExecutionUnit::tick()` 是
  no-op，避免多 adapter 重复推进。
- **command 上下文与物理 pe_id 分离统计**：`ExecResponse.unit_id` 记录
  adapter 槽位；`panel_tasks.csv` 记录实际 `pe_id`。
- **能力显式声明**：未绑定 unit 的 opcode（真实后端的 TRSM/GEMM/assemble/
  solve）在派发前返回 `POLICY_NOT_SUPPORTED`，不落 stub、不永久等待。

## 2. 模块交互（每拍相位）

```text
tick(cycle):
  1. read_budget/write_budget 重置（每拍字节预算）
  2. MultiModePe[*].tick：装载/执行/写回阶段推进；装载与扫描共享 read_budget
  3. route_pe_responses：写回完成→版本提交→controller.on_task_response
  4. TileDispatcher.tick：依赖/版本/同目标写者检查后派发
  5. handle_cancel_requests：controller 的失败/abort 取消排队任务并 abort PE 任务
  6. service_contexts（轮转）：扫描推进、步内任务生成、P-vector 不在此处（T05 负责）
  7. finalize_contexts：controller 终结且无在途任务→产出唯一终态响应
```

- 扫描与 PE 装载共享同一读带宽预算，PE 装载在相位 2 先消耗，扫描在相位 6
  使用余量；多上下文轮转并通过 `stall_scan_port` 计数观察竞争，保证有界推进。
- 写回使用独立写预算与有界 reservation；P-vector 写由 T05 既有 PostExec 路径
  走 MemoryService（见 §5）。
- reset/abort 语义：定向 abort 在 `abort_txn` 内同步取消排队任务、标记运行中
  任务跳过写回；返回后引擎不再对该事务发起任何 lease 访问。

## 3. 接口

### 3.1 内部任务契约（复用 T06，去合成化）

- `TileOperand` 增加 `version_key`：T06 合成布局 key=slot；真实布局
  key=(槽位<<32)|(tile 索引+1)。
- `TileVersionStore`：dispatcher 的依赖/顺序检查；`TileAccess`：PE 操作数读取
  与结果写回；`FrontReader`：扫描字节源。
- `PanelFrontLayout`：每 command 的 front 物理布局
  （`PerTileSlots` 合成原型 / `SingleBufferStrided` 真实 strided buffer）。
- 操作数校验按“有效行宽”检查 stride（`>= 有效列数×4`），不再是固定 64B，
  以适配真实 `row_stride = total_dim×4`。

### 3.2 引擎/适配器接口

- `PanelEngineRequest/Response`（panel_lu 内部，不含 controlplane 类型）：
  txn/epoch/command/node、total/pivot、policy/threshold、front 句柄+lease、
  终态 status/reason/pivot_count/failed_column、bytes、周期、P-vector 需求。
- `PanelExecutionUnit : IExecutionUnit`：把 `ExecRequest` 转成引擎请求；
  引擎忙返回 false（stall）；不支持请求返回 true+立即终态失败响应。
- `IExecutionUnit` 新增 `abort_txn`（定向取消）与
  `deliver_memory_response`（默认空实现）。
- `FactorStoreUnit : IExecutionUnit`（kind=Factor）：STORE_FACTOR 真实转换。

### 3.3 T05 配置

```cpp
ControlPlaneConfig {
  backend = RequireRealKernels; enable_real_panel = true; enable_test_stub = false;
  panel = PanelLuConfig{ pe_count, task_slots_per_pe, dispatcher_queue_depth, ... };
  factor_unit_count = 1;
}
```

- `start()`：`RequireRealKernels` 且未开启 real panel → 拒绝；
  两个后端都未开启 → 拒绝；real panel 开启时校验并 `configure()` 引擎。
- 混合（两个开关都开）是显式测试配置，报告标记
  `execution_backend=real_panel_fp32_plus_test_stub`，不得作为完整数值成功证据。

## 4. 实现机制

### 4.1 真实 scratchpad 访问

- command 的 front 生命周期由 T05 管理；PANEL 派发时 T05 已持有
  `begin_write` lease（BufferManager 语义允许就地读写）。引擎通过
  `lease_read` 读取扫描/操作数、通过 `lease_write` 写回结果，**不重复申请
  与自身冲突的整 front lease**，也不直接访问 BufferManager 底层容器。
- 地址计算使用真实 offset/row stride/有效行宽/tile 坐标；
  handle generation、owner、范围、`lease.ready_cycle` 全部生效。
- 端口占用与字节预算是两回事：BufferManager 端口/lease 负责“谁能访问、
  访问是否越界/就绪”；每拍字节预算（引擎 `PortBudget`）负责“带宽”。二者不
  重复计费：lease 不再叠加额外延迟，引擎也不绕过 lease 直接读数据。
- PE 本地只保留有限 task slot 对应的操作数与结果（每 slot ≤3 个 16×16
  staging 操作数 + 1 个累加/结果寄存器组，容量有界）。**操作数不在 accept
  时读取**：Loading 阶段按共享读预算分段（行内可再分段）通过
  `read_operand_segment` 装载，字节进度跨拍保留；全部操作数字节到齐后才基于
  快照执行功能计算，结果在写回事件前隔离。即使某拍读预算为 0，派发也无法
  越过预算提前读到操作数。

### 4.2 数值原语（正式）

`hw::panel_lu::kernels`（T09/T10 复用）：

| 原语 | 语义要点 |
| --- | --- |
| `run_lu_diag` | STATIC packed LU：逐列检查实际消元主元（非有限→NON_FINITE、精确零→STATIC_PIVOT_FAILURE、≤阈值→PIVOT_UNSTABLE）；倒数溢出（subnormal 输入）在第 j 列报 NON_FINITE；FMA 更新；单位下三角隐含 |
| `run_trsm_left` | 单位下三角前向代入 `L X = A`，无除法通路 |
| `run_trsm_right` | 对角倒数预计算（逐列有限性检查，溢出在首列失败）+ 纯乘法前向代入 |
| `run_gemm_update` | `D -= A·B`，FMA；有效 m/n/k 与 mask |

所有 MAC 使用 `std::fma`；不做 fast-math/flush-to-zero；失败列与设计时延
规则一致（LU stage 反馈链、TRSM_RIGHT 倒数延迟）。TRSM_LEFT/RIGHT 的共享
语义即为 T08 讨论基线：左解单位对角、右解非单位对角，本实现先行固定。

### 4.3 扫描与固定尺度

- 扫描整个 n×n front（含 A12/A21/A22）：`scan_bytes = n²×4`，经
  `FrontReader` 按读预算分块读取；扫描归约延迟 `scan_reduce_latency_cycles`。
- 输出精确 FP32 maxabs 与有限值标志；NaN/Inf → NON_FINITE_VALUE，不生成
  任何 tile 任务。
- 所有对角 tile 共用 `threshold_product = threshold_ratio × maxabs`
  （binary64 精确乘积，见 Panel-LU 架构文档 §4/6）。

### 4.4 A11 任务图与步屏障

- `p = 16k + r`：共 `k+1` 个 pivot tile；每步
  `LU_DIAG(s,s) → TRSM_LEFT/TRSM_RIGHT → GEMM_UPDATE`，步内独立任务可并行，
  同目标 tile 更新按版本顺序提交，下一步只读已提交版本。
- 尾块用有效 m/n/k 与 mask；无效元素不参与计算/检查/归约/写回；
  `descriptor.tile_size != 16` 明确 `BAD_DESCRIPTOR`。
- 面板只写 A11；A12/A21/A22 在 front 中保持不变（n>p 测试验证字节不变）。

### 4.5 因子转换（STORE_FACTOR）

- `FactorStoreUnit`：按行分块（4 行/块）从 front lease 读 packed A11，
  生成 L 行（严格下三角取 packed、对角补 1、上三角补 0）与 U 行
  （上三角含对角取 packed、下三角补 0），通过 MemoryService 写入
  `factor_l/factor_u` 区域；有界分块，无整 front 副本。
- 只写 p×p panel 块：L21/U12 属于顶层 TRSM（T09 之后），T07 不写、不宣称。
- **资源语义**：只要绑定了正式 Factor 单元，STORE_FACTOR 在单元忙时等待
  （`stall_resource`），不会落入旧矩形 DMA 复制；该 DMA 路径仅在显式
  test-stub 后端（没有 Factor 单元绑定）时存在。
- **精确响应路由**：每个单元维护本事务未决请求 id 集合；T05 按 request id
  只投递给属主单元（不再广播），未知/迟到响应被丢弃。终态响应只在全部已接受
  写确认后产生，`finish_cycle ≥ 最后一笔 write 的完成拍`。
- **背压与错误分离**：内存队列满返回可重试背压（保留行内部分提交进度，
  下一拍继续，不重复提交已接受的写）；只有 lease/范围等永久错误才是终态
  `ADDRESS_FAULT`。
- **事务内状态生命周期**：`l_submitted_/u_submitted_` 等半行提交标志在新
  `submit()`、终态 `finish_terminal()` 与 `reset()` 时一律清零，防止
  “L 已接受、U 背压时 abort 排空”后标志泄漏到同一单元的下一条命令
  （回归：双 front、1 个 Factor 单元、`max_inflight=1`，被取消命令排空后
  另一 front 的 L/U 全量与字节数正确）。

### 4.6 P-vector、失败与取消

- 成功：Panel 引擎返回 `needs_pvector_write=true`（不复用 T06 原型的
  P-vector 模拟）；T05 PostExec 写完整 identity INT32 并等确认后才发布
  READY。失败命令不写 P-vector（哨兵保持）。
- 失败：首个失败列进入 completion 的 `pivot_count`；后续任务取消，已提交
  写不回滚；无 READY token。
- 定向 abort：`CommandDevice::abort_node` → `unit->abort_txn(txn)`；引擎同步
  停止生成/取消排队任务并让运行任务在安全点结束且跳过写回。**取消 ≠ 排空**：
  T05 保留取消中的 tracker 与 lease，等待执行单元的唯一终态响应（Factor
  单元还会先等已接受的 DDR 写全部确认）后才释放资源并结束命令；
  `finished()`/Completed 判定额外要求所有执行单元 `!busy()`、DMA 空闲且
  `exec_inbox` 为空。旧响应由 `handle_exec_response` 的终态守卫丢弃。
  test-stub 的 `abort_txn` 同样产生恰好一个 ABORTED 终态响应。
- reset：`CommandDevice::reset` → `panel_engine_->reset()`（epoch++、
  版本表/队列/PE/上下文清空）。旧 epoch/generation/响应不得生效。

## 5. 与 T05 的分工

| 事项 | T05 | T07 引擎 |
| --- | --- | --- |
| command 取指/依赖/token/completion | ✔ | — |
| front/temp buffer 分配与 lease 生命周期 | ✔ | 只使用授权 lease |
| P-vector identity 写 | ✔（PostExec） | 只返回 `needs_pvector_write` |
| 失败传播/READY 门槛 | ✔ | 返回终态状态与失败列 |
| 单元 busy/定向取消 | ✔ 调度 | `abort_txn` 同步安全化 |
| A11 分解/因子转换 | — | ✔ |

## 6. 验证

### 6.1 数值

`panel_lu_real_tests`（215 checks，Release+ASan/UBSan）：

- 资源/路由回归（对应审查修复）：绑定但忙的 Factor 单元必须等待（双
  front、1 个 Factor 单元时第二个 STORE_FACTOR 不落 DMA、L 仍为单位
  对角/mask 并与参考一致）；两个 Factor 实例交叠时按 request id 精确路由，
  `command finish ≥ 该命令最后一笔 DDR 写完成拍`（独立 trace 证据）；实例
  reset 复用后因子正确重写；`memory.max_inflight=1` 的合法背压下
  STORE_FACTOR 成功而不是 `ADDRESS_FAULT`；L 已接受/U 背压窗口内 abort
  排空后同一单元服务另一 front，L/U 全量匹配且字节数完整（215 checks）；
  `load_bytes_per_cycle=8` 时
  diag 任务 `bytes_read=1024` 且 `load_done-dispatch ≥ 128`、执行在装载
  之后；abort 后 lease 不立即释放、batch 完成时引擎全部 adapter 空闲；
  FP64 诊断模式 subnormal 用例与 16×16 对照。
- n=p ∈ {4,15,16,17,23}：PANEL+STORE_FACTOR 成功；L/U 与独立 FP64
  `reference_fp64::factor_front(STATIC)` 对照（max|diff| < 5e-3，实测约
  1e-7 量级）；L 上三角 0/单位对角、U 下三角 0；L·U 重构 A11；P-vector
  identity。
- n>p（17,16）：A11 packed 与参考一致；A12/A21/A22 字节不变；顶层 TRSM/GEMM
  显式 `POLICY_NOT_SUPPORTED`（不落 stub）；不宣称完整分解。
- 失败：消元后零主元（列 3/列 6）→ `STATIC_PIVOT_FAILURE` 且列号正确；
  `0.9×maxabs` → `PIVOT_UNSTABLE`；A12 区域 NaN → `NON_FINITE_VALUE` 且零任务；
  失败不写 P-vector、不写 L。
- 控制/资源：双 front（pe=2）跨 mode 排他；依赖任务只在 producer 写回提交后
  派发；扫描期间 abort → ABORTED；双 front 定向 abort 不影响另一 front；
  reset 后重启成功；共享读带宽 16B/cycle + 队列/写回背压（depth=1、
  wb 4B/cycle）下两 front 均完成；`DYNAMIC_PANEL`、`tile_size=8` 显式拒绝。

### 6.2 时序

`panel_lu_real_systemc_tests`（sc_clock，10 checks）：每 posedge 一次
`device.tick()`；真实 FP32 panel 数值与 FP64 对照通过；时钟驱动与纯 C++
tick 模型逐周期一致；运行时模型不自行推进 SystemC 时间。

### 6.3 报告标记

- `summary.json`：`execution_backend=real_panel_fp32`（混合时
  `real_panel_fp32_plus_test_stub` 且 `mixed_backend_explicit=true`）、
  `numerical_scope=panel_fp32_external_reference_only`、
  `numerical_validation=not_performed`（设备内部不做 golden 校验）、
  `panel_engine.task_records/control_fault`。
- `panel_tasks.csv`：任务级 mode/pe_id/周期/字节/提交状态。
- 周期是未综合校准的架构模型周期；不得写成芯片实测或加速比。

## 6b. T09 扩展（共享引擎、分组与仲裁）

- `pe_count` 由 `compute_group_count × pes_per_group` 唯一推导（冲突配置
  拒绝）；一条被接受的计算命令独占一个 PE 组（含扫描阶段），dispatcher 只
  向该组派发，组在终态响应被 T05 接收后释放。
- `IComputeController` 统一 PanelController/TrsmController；
  `TrsmController` 的左右解、步屏障与失败映射见 T08/T09 设计文档。
- 读/写预算由设备级 `OnChipArbiter` 统一仲裁（每域每拍单请求方 grant、
  确定性轮转、剩余不共享），覆盖扫描、PE 装载/写回、Factor 读取与 DMA
  片上读写；本文 §4.1 中“共享读预算”由该仲裁器实现。
- `FactorStoreUnit` 升级为完整 L/U 保存（L11 单位对角 + L21、U11 mask +
  U12），成功字节数 8·n·p。

## 7. FP64 模块诊断模式（显式）

T07 提供 `PanelNumericMode::Fp64Diagnostic` 显式模块诊断模式（默认
`Fp32Device`）：

- 同一批 `kernels` 算法在 binary64 快照上执行（`kernels::fp64_diagnostic::*`），
  结果按 FP32 舍入回 buffer；调度、command/token、P-vector、失败列语义完全
  不变，时延模型也不变。
- 只由 `PanelLuConfig::numeric_mode` 声明启用；**没有自动回退**：默认 FP32
  路径失败不会尝试 FP64。运行报告 `summary.json` 标记
  `panel_numeric_mode=fp32_device|fp64_diagnostic`。
- 与“独立 FP64 reference 对照”是两回事：reference 只在测试中计算期望值，
  不进入设备；诊断模式是设备内可执行的显式模式，用于区分“输入/舍入语义”
  与“实现缺陷”。
- 验证：1×1 subnormal 用例（输入 `1e-39f` 在两种格式都可精确表示）——
  FP32 设备倒数溢出报 `NON_FINITE_VALUE`，显式 FP64 诊断模式成功并保留
  subnormal 因子；另有 16×16 诊断模式与 FP64 reference 对照。
- 舍入回 FP32 的溢出检查沿用各模式既有失败坐标约定：`LU_DIAG` 报全局
  主元列 `pivot_base + local`（与 FP32/FP64 kernel 的失败列语义一致，
  `pivot_count` 随之为已完成主元数）；`TRSM_LEFT` 报局部行、其余模式报
  局部列，由 controller 映射到 front 坐标。回归：n=p=18、第二个对角块
  `[[0.5, FLT_MAX],[FLT_MAX, FLT_MAX]]` 在诊断模式舍入溢出报告列 16、
  `pivot_count=16`（而非局部列 0）。

## 8. 尚未完成 / T10 复用点

- T09 已完成：顶层 TRSM_LEFT/TRSM_RIGHT 数值执行（TrsmController）、
  PE 分组调度、统一片上带宽仲裁、完整 L/U（L11/L21/U11/U12）持久化。
- T10 已实现顶层 GEMM_SCHUR（FP32/局部 INT32）；T11 已实现真实 assemble
  与 STATIC 树形分解。未实现：树形 solve/完整 Ax=b；DYNAMIC_PANEL；
  INT32 RTL；bank/NoC 细化；跨步 lookahead。
- T10 复用点：`kernels` 原语、`TileDispatcher`/`MultiModePe`、PE 分组与
  `IComputeController` 契约、`OnChipArbiter` 预算接口、
  `PanelFrontLayout`/`TileAccess`；只需新增 GEMM/Schur controller 与任务
  生成，继续共享同批 PE/dispatcher，不得各建计算池。
