# 共享计算引擎与 PE 设计

实现基准：2026-10-11 当前工作区。命名空间 `hw::panel_lu`。
本文集中定义 Panel/TRSM/GEMM 共用的执行机制；特定任务图见各模块设计。

## 1. 职责与模块组成

```text
CommandDevice（每拍唯一推进者）
  ├─ PanelExecutionUnit / TrsmExecutionUnit / GemmExecutionUnit
  │       └─ 同一个 PanelEngine
  │            ├─ Context + IComputeController
  │            ├─ TileDispatcher + TileVersionStore
  │            ├─ MultiModePe[计算组数 × 每组 PE 数]
  │            ├─ LeaseTileAccess / LeaseFrontReader
  │            └─ WritebackReservation
  └─ OnChipArbiter（设备所有片上通路共用）
```

计算命令独占一个 PE 组直到终态被适配器消费。默认两个组，每组两个 PE。
组可依次服务不同节点与操作；同组不同 tile 可并行，单 tile 不拆分给多个 PE。
装配和单 RHS 求解采用独立通路，只共享片上预算，不使用本 PE 池。

## 2. 接口与实现对象

| 对象 | 代码 | 作用 |
|---|---|---|
| `PanelEngine/PanelEngineRequest/Response/Context` | [头文件](../../graduation-code/systemc/include/panel_lu/panel_engine.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/panel_engine.cpp) | 组分配、front binding、controller、响应与记录 |
| `IComputeController` | [compute_controller.hpp](../../graduation-code/systemc/include/panel_lu/compute_controller.hpp) | start/tick、任务响应、取消、终态接口 |
| `TileTask/TileOperand/TileDims/TileMask/TileTaskResponse` | [task_types.hpp](../../graduation-code/systemc/include/panel_lu/task_types.hpp) | 模式、有效维度、依赖、版本、地址和身份 |
| `TileDispatcher` | [头文件](../../graduation-code/systemc/include/panel_lu/tile_dispatcher.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/tile_dispatcher.cpp) | 有界入队、版本/依赖与同目标检查、派发和 commit |
| `MultiModePe` | [头文件](../../graduation-code/systemc/include/panel_lu/pe_model.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/pe_model.cpp) | task slot、FIFO、装载、执行、写回与响应 |
| `TileAccess/TileVersionStore/WritebackReservation` | [tile_store.hpp](../../graduation-code/systemc/include/panel_lu/tile_store.hpp) | 数据访问抽象、提交版本和有界写回预约 |
| `OnChipArbiter` | [头文件](../../graduation-code/systemc/include/dataplane/on_chip_arbiter.hpp)、[实现](../../graduation-code/systemc/src/dataplane/on_chip_arbiter.cpp) | 稳定客户端身份、读写预算与轮转 |

`begin_command(req,adapter_key,cycle)` 校验请求、绑定空闲组并建立 controller。
`adapter_key=(kind<<32)|unit_id` 区分不同类型中的重复单元编号。
`poll_response()` 消费终态后释放 context/组；忙时保留请求，非法请求提供明确失败状态。

## 3. 内部任务与 front 访问

一个 command 是完整 Panel/TRSM/GEMM 操作，controller 增量生成内部 `TileTask`。
任务携带 txn/epoch/command/node/task 身份、M/N/K、mask、源/目标操作数、依赖和输出版本。
内部任务不增加外部 opcode 或改变 schema。

正式 front 是紧凑 FP32 strided buffer，`PanelFrontLayout::SingleBufferStrided`
表达基址、stride、网格和组。`LeaseTileAccess` 使用控制面授予的 lease 访问真实字节，
检查 generation、owner、范围和 ready cycle，不重新申请会与自己冲突的整 front lease。

有效 M/N/K 在 `1…16`，尾块 mask 与结果形状一致。
stride 为 4 字节倍数且至少覆盖有效行宽；不能把所有尾块都要求为 64 字节行宽。
LU_DIAG 必须方阵，左右 TRSM 有相应 M/K、N/K 关系，任务依赖数量最多 8。

## 4. 版本、派发与可见性

版本键统一为：

```text
tile_version_key(slot,view,index) = (slot<<48) | (view<<32) | (index+1)
view: FrontGrid / A12 / A21 / A22
```

front 网格与不对齐 pivot 后的子块网格分别标识，不能让槽位或数据视图互相别名。
引擎建立新计算 context 时清除该 slot 的旧版本，当前版本序列服务该命令的内部任务。
slot 复用、reset 与命令收尾还要清理绑定及依赖元数据。

派发必须同时满足：依赖已 commit、源版本可读、目标版本匹配、同目标没有未提交写者、
所属组有可接收的 PE。任务声明目标当前版本与下一输出版本；commit 冲突升级为控制故障。
步屏障和同目标版本顺序共同防止跨步读改写竞争。

```text
controller 入队 → dispatcher 派发 → PE 装载 → 功能计算/执行等待
→ 真实写回 → dispatcher commit → controller 消费任务响应
→ 全部任务收尾 → engine 终态 → adapter 消费
```

数值函数返回不表示输出可见。失败/取消任务不能发布成功版本。
队列满时 controller 保留生成位置并逐拍重试；运行队列容量与报告用的 host 记录向量分开。

## 5. PE 数据通路与有限存储

一个 `MultiModePe` 表示 16×16 output-stationary 阵列、角倒数/除法通路、操作数 FIFO、
task slot 与写回通路。LU_DIAG、TRSM_LEFT、TRSM_RIGHT、GEMM_UPDATE/GEMM_SCHUR 使用相同资源。

每 PE 默认四个 task slot，阵列同一时刻只执行一个任务；其他 slot 可装载或排空结果。
默认 FIFO 4096 字节，至少容纳满 tile A+B 的 2048 字节。
目标累加/结果存储按 slot 单列，不能重复计入操作数 FIFO。
INT32 模式的 INT64 累加器和转换存储见 [GEMM 设计](GEMM_Schur与局部INT32设计.md)。

`try_accept()` 原子占 slot 与 FIFO 额度，不读取操作数。
Loading 按预算分段读取，数据全部到齐后在有限快照上计算；结果保持隔离至真实写回事件。
写回 reservation 满时等待，结束后释放预约、FIFO 与 task slot。

## 6. 周期模型

周期函数位于 [task_types.cpp](../../graduation-code/systemc/src/panel_lu/task_types.cpp)。
以下为 FP32 task 的阵列阶段，装载、写回、仲裁和响应保持另计：

| 模式 | 周期 |
|---|---|
| LU_DIAG | `(k-1)*(div_latency+lu_update)+lu_pivot_check` |
| TRSM_LEFT | `tsolve_fill+k+n` |
| TRSM_RIGHT | `tsolve_fill+(div_latency+k-1)+k+m` |
| GEMM_UPDATE / FP32 GEMM_SCHUR | `fill+k*gemm_k_ii+m*drain_row_cycles` |

LU 的相邻 stage 经倒数和 rank-1 更新反馈，启动间隔是 `div_latency+lu_update`。
右 TRSM 先流水计算各对角倒数，再进入乘法代入；左 TRSM 单位对角，不使用除法。
失败时延按实际已完成阶段与失败位置计算。INT32 另外建模尺度、量化与反量化阶段。

## 7. 统一带宽仲裁与每拍相位

读、写两个域默认各 64 B/cycle。每域每拍只选一个请求方，未用完的 grant 不再转授。
客户端使用稳定 kind/id，确定性轮转覆盖 PanelScan、PeLoad/PeStore、DMA、FactorRead、
Assemble 和 Solve。所有通路必须在被授予的预算内保留分段进度。

全局预算由 CommandDevice 初始化一次。engine 随后推进 PE、路由写回响应、dispatcher、
取消处理、controller 服务与 context 收尾。独立模块测试可使用 fallback arbiter。
扫描和 PE 装载共享读域，扫描没有额外免费端口；P-vector 经 DDR MemoryService 写确认。

lease 的授权/就绪时间与预算的字节限制分别生效。DMA 在 DDR 队列背压期间保留已读取 beat，
不能重复读取、清零 staging 或绕过预算。仲裁参数是模型假设，没有显式 bank/NoC 冲突仿真。

## 8. 配置与能力校验

| `PanelLuConfig` 参数 | 默认 | 含义 |
|---|---|---|
| `compute_group_count / pes_per_group` | 2 / 2 | 唯一物理 PE 数为乘积 |
| `pe_count` | 0 | 0 表示派生；非零须与乘积一致 |
| `task_slots_per_pe` | 4 | 每 PE 在途 slot 上限 |
| `dispatcher_queue_depth` | 32 | 有界等待队列 |
| `max_resident_commands` | 2 | 计算 context 数量 |
| `max_front_dim / tile_dim` | 256 / 16 | front 上限可配置到 512；物理 tile 固定 16 |
| `pe_input_fifo_bytes` | 4096 | 至少 2048 字节 |
| `load/writeback_bytes_per_cycle` | 64 / 64 | 与全局片上仲裁配置一致 |
| `load/writeback_latency_cycles` | 2 / 2 | 装载和写回尾延迟 |
| `load/writeback_queue_depth` | 8 / 8 | 请求/预约容量 |
| `scan_reduce_latency_cycles` | 8 | 全 front 扫描归约尾延迟 |
| `response_hold_cycles` | 0 | 终态保持模型 |

`validate_panel_config()` 拒绝非法资源组合；请求还要满足 front 形状、固定 tile、策略、backend、
scale 和有效 lease。模块默认值与正式入口的显式配置分别记录，不能混用。

## 9. 失败、取消与复位

controller 失败后停止生成，取消排队任务，定向标记在运行 PE 的任务跳过后续写回。
已写数据不回滚，成功提交和取消结果分别记录。context 在任务排空、终态被消费后回收。
错误响应尚未 poll 时，abort 保留同 txn 的唯一终态；不匹配 txn 不影响另一 context。

reset 清理 PE、队列、版本、binding、组和 epoch；旧响应或 generation 不得影响复用后的槽。
`PanelLuPrototype/SyntheticTileAccess` 是独立原型路径，正式 engine 使用真实 front lease。

## 10. 测试与观测入口

相关目标：`panel_lu_prototype_tests`、`panel_lu_real_tests`、`trsm_real_tests`、`gemm_real_tests`
及对应 SystemC 驱动。跨槽版本隔离在 `trsm_real_tests::test_engine_slot_version_isolation`
和真实 artifact 测试中覆盖。
主要检查任务数据、依赖/版本、尾块、有限队列、组/PE 排他、端口背压、取消和槽复用。
`panel_tasks.csv`、engine task records 与 PE events 用于核对装载、执行、写回和提交时刻。
