# TRSM 架构设计（T08）

状态：**T08 设计已落地；T09 按本文实现完成**（顶层左右 TRSM 数值执行、
PE 分组调度、统一片上带宽仲裁、完整 L/U 因子保存）；**T10 已按本文 §3.9
的复用接口接入顶层 GEMM_Schur**（设计见
[T10_GEMM_Schur与局部INT32设计](T10_GEMM_Schur与局部INT32设计.md)）。
完整 solve、树形前代回代、RTL、DYNAMIC_PANEL 仍属后续任务。

本文按“整体架构与交互 → 模块职责/请求响应/数据布局 → 实现机制/状态/时序 →
验收与边界”组织，是 T09 与后续 SolveController 的设计权威。

---

## 1. 整体架构、模块组成与交互

```text
CommandDevice（T05，每拍一次 tick）
  ├─ 共享计算引擎（唯一物理 PE 池，Section 5 分组）
  │    PanelEngine
  │      ├─ MultiModePe[total_pe_count = group_count × pes_per_group]
  │      ├─ TileDispatcher + TileVersionStore（有界队列/依赖/版本/组内派发）
  │      ├─ OnChipArbiter（设备级统一读/写预算，Section 6）
  │      └─ 每命令一个 controller 上下文（PanelController / TrsmController）
  ├─ PanelExecutionUnit × panel_unit_count     ─┐ 共享同一 engine
  ├─ TrsmExecutionUnit  × trsm_unit_count       │ 每拍只由 CommandDevice
  ├─ FactorStoreUnit    × factor_unit_count     │ 推进一次共享引擎
  └─ TestStubUnit（仅显式 stub 后端）          ─┘
```

- **命令独占 PE 组**：一条被接受的 `PANEL_LU`/`TRSM_LEFT`/`TRSM_RIGHT`
  （以及后续 `GEMM_SCHUR`/solve）命令在开始执行（含 Panel 的扫描阶段）时
  绑定一个 PE 组，直到该命令的全部 tile、队列记录与访问排空、终态响应被
  T05 接收后才释放。另一条命令即使在本组有 PE 空闲时也不能使用该组。
- **组内 tile 并行**：一个 16×16 tile 仍由一个 PE 执行；同一命令的不同 tile
  可并行使用本组内的多个 PE，不得把单个 tile 拆给多个阵列。
- **组不永久绑定操作或 front**：同一组可以先执行 Panel，释放后再执行 TRSM、
  后续 GEMM/solve；整个设备只有一套物理 PE 池。
- 顶层 `TRSM_LEFT`/`TRSM_RIGHT` 命令经 T05 依赖（PANEL 的 READY token）派发，
  使用 T07 已实现的内部 tile 原语与多模式 PE；`STORE_FACTOR` 不再经过 DMA
  矩形复制，而是由 `FactorStoreUnit` 读 packed front、按三角 mask/单位对角
  转换后写入持久化 L/U。

模块与文件：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| TrsmController | `panel_lu/trsm_controller.{hpp,cpp}` | 一条 TRSM 命令的 tile 任务生成、步屏障、失败/取消、响应 |
| PanelEngine（共享引擎） | `panel_lu/panel_engine.{hpp,cpp}` | PE 组分配/释放、context、dispatcher、版本表、响应汇聚 |
| IComputeController | `panel_lu/compute_controller.hpp` | Panel/TRSM（后续 GEMM/solve）controller 的统一接口 |
| TrsmExecutionUnit | `controlplane/panel_exec_unit.{hpp,cpp}` | T05 `IExecutionUnit` 适配器（kind=Trsm），内核形状校验 |
| FactorStoreUnit | 同上 | 完整 L/U 保存（有界 staging + 统一读预算） |
| OnChipArbiter | `dataplane/on_chip_arbiter.{hpp,cpp}` | 读/写预算域的单请求方轮转仲裁 |

## 2. 模块职责、请求/响应与数据布局

### 2.1 请求（T05 → 共享引擎）

`ExecRequest`（T05 内部）→ `PanelEngineRequest`：

- 身份：`txn_id/epoch/command_id/node_id` + 适配器身份
  `adapter_key = (kind << 32) | unit_id`（不同 kind 的 unit_id 可以相同，
  响应路由必须带 kind）。
- `compute_kind ∈ {PanelLu, TrsmLeft, TrsmRight}`（可扩展 Gemm/solve）。
- 维度：`total_dim = n`、`pivot_dim = p`；TRSM 额外携带 `KERNEL_DESC` 的
  `kernel_m/n/k`，严格核对 `TRSM_LEFT M/N/K = p/u/p`、
  `TRSM_RIGHT M/N/K = u/p/p`（`u = n - p`），并校验
  `tile_size = 16`、backend 为 FP32 设备模型。
- **操作数身份核对**：TRSM 的 `KERNEL_DESC` 三个操作数引用必须解析为
  当前驻留 front 的同一 REGION_DESC（descriptor id、base、byte_size 与
  驻留 `FrontView.front` 完全一致）。派发计算前核对；不一致返回
  `BAD_DESCRIPTOR`，不建立执行上下文、不修改任何数据、不发布 READY。
- **偏移口径**：任务的 `operand.offset` 一律相对 front 起点；front 的
  绝对起点（`front_offset`）只由 lease/访问层叠加一次。三种 operand
  构造与扫描字节偏移共享同一相对口径，guard/padding 不变。
- 数值上下文：`front_scale`（Panel 锁存的
  `front_pre_elimination_max_abs`）与 `threshold_product`（binary64 精确
  乘积），由 PANEL 响应 → T05 front slot → 后续 TRSM 请求传递；TRSM 的
  非单位 U 对角检查**不得**从已消元的 front 重新推导尺度。
- 授权与布局：`front` 句柄 + `front_lease`（由 T05 在派发时原子获取的
  write/read lease）。front 已按紧凑 row-major 装载，片上 `row_stride =
  total_dim × 4`；DDR 侧 stride 由各 REGION_DESC 独立表达（本引擎不读 DDR）。

### 2.2 数据布局与 tile 视图

设 front 起点偏移 0，行 stride `S = n×4` 字节：

```text
        col 0            col p                 col n
row 0   [      A11 (p×p)      ][   A12 (p×u)   ]
row p   [      A21 (u×p)      ][   A22 (u×u)   ]
row n
```

- **A11**：PANEL 原地 packed L/U（严格下三角 L 单位对角隐含、上三角含对角
  U）。TRSM 只读其中的对角块与次对角块，不修改 A11。
- **A12**：`TRSM_LEFT` 原地求解 `X = L11^{-1} A12`；tile 从**块起点**
  （列 `p`）划分：行块 `i` 对应 front 行 `i×16`，列块 `j` 对应 front 列
  `p + j×16`（`p=23` 时首列是 23，不是 16/32）。
- **A21**：`TRSM_RIGHT` 原地求解 `L21 = A21 U11^{-1}`；行块从行 `p` 起。
- tail：`m/n/k` 取有效维度（≤16），mask 等于有效维度；无效元素不读、不算、
  不写，guard/padding 不变。
- **版本 key**：`(slot << 48) | (view_tag << 32) | (index+1)`，
  `view_tag ∈ {FrontGrid, A12, A21, A22}`。同一 slot 内 A12/A21 子块视图
  与全 front 网格索引不会错误别名；slot 复用时由 context 释放/ reset
  清空对应版本。

### 2.3 响应（共享引擎 → T05）

`PanelEngineResponse` → `ExecResponse`：状态码沿用 `StatusCode`；
`failed_column` 为**front 坐标**（右解/对角失败为全局主元列，左解失败为
全局主元行）；TRSM 成功不写 P-vector（`needs_pvector_write=false`）、
`pivot_count=0`；PANEL 成功携带 `front_scale/threshold_product` 供后续
TRSM 使用。字节数由引擎按任务响应汇聚（读=操作数装载，写=结果写回）。

## 3. 实现机制、状态流转、依赖与关键时序

### 3.1 数学方向与对角语义

- `TRSM_LEFT`：单位下三角前向代入
  `x_i = b_i − Σ_{t<i} L[i,t]·x_t`（单位对角隐含 1，不读取 packed A11 的
  U 对角）。
- `TRSM_RIGHT`：非单位上三角右解
  `x_j = (b_j − Σ_{t<j} x_t·U[t,j]) / U[j,j]`；对角倒数预计算（流水
  II=1），逐列检查：非有限 → `NON_FINITE_VALUE`；精确零 →
  `STATIC_PIVOT_FAILURE`；`|U[j,j]| ≤ threshold_product` →
  `PIVOT_UNSTABLE`；倒数溢出到 Inf → `NON_FINITE_VALUE`。

### 3.2 对角 tile solve 与非对角 GEMM 更新的划分

- 对角块 `(s,s)` 的 solve 由单个 tile 任务完成（左解 `Lss Xsj = Bsj`，
  右解 `Xis Uss = Bis`）；右解使用非单位对角检查，左解为单位对角。
- 非对角更新由统一 `GEMM_UPDATE` tile 任务完成
  （左解 `B(q,j) -= L(q,s)·X(s,j)`；右解 `B(i,q) -= X(i,s)·U(s,q)`），
  通过 PE 的多模式 GEMM/MAC 数据通路，不直接修改全局 C++ 矩阵。

### 3.3 步屏障、增量入队与依赖

```text
LEFT  step s: solve X(s,j) for every update column tile j
              for each committed solve: update B(q,j) -= L(q,s)·X(s,j), q>s
              barrier: every task of step s committed -> s+1
RIGHT step s: solve X(i,s) for every update row tile i
              for each committed solve: update B(i,q) -= X(i,s)·U(s,q), q>s
```

- 增量入队：solve 提交后立即生成对应更新；队列满时保留生成进度
  （solve/update 生成标志），下一拍继续，不展开无界任务图。
- 依赖：每个更新依赖其 solve 任务；同目标 tile 的读写在版本表 + 步屏障下
  串行；依赖只引用最近生产者与对应 solve，不堆积历史依赖。
- 释放顺序：controller 终态且无在途任务 → 引擎产出唯一终态响应 →
  T05 接收（poll）后释放组与 lease；abort 时先排空再释放。

### 3.4 时序模型（沿用 T07 阵列推导，无整条命令总延迟）

| 模式 | 阵列占用周期 |
| --- | --- |
| `TRSM_LEFT` | `tsolve_fill + k + n` |
| `TRSM_RIGHT` | `tsolve_fill + div_latency + (k−1) + k + m` |
| `GEMM_UPDATE` | `fill + k×II + m×drain_row_cycles` |

装载/写回按统一预算分拍计；失败时延按实际检测阶段（对角失败在倒数预计算
列，求解失败计到失败行/列）。装载在 Loading 阶段按预算分段进行，数据到齐后
才对快照执行功能计算。

### 3.5 统一片上带宽仲裁

- 读、写两个预算域（默认 64 B/cycle），**每拍每个域最多服务一个请求方**，
  授予最多完整预算；本拍剩余额度不再分给第二个请求方。
- 请求方使用稳定身份（kind + id：PanelScan(command)、PeLoad(pe)、
  PeStore(pe)、FactorRead(unit)、Dma、后续 Assemble/Solve）；按固定顺序
  轮转，避免 tick 顺序造成永久优先权；未获 grant 的请求保留进度等待。
- 覆盖全部真实 scratchpad 访问：Panel 扫描、PE 操作数装载、PE 写回、
  FactorStore 读取、DMA 片上读写。DDR 延迟/带宽仍由单端口 `MemoryService`
  与单活动 DMA 建模。
- **DMA beat staging 规则**：`BufferToMemory` 的整个 beat 只在开始新 beat
  时初始化 staging，并严格按 grant 分段读取恰好一次；预算等待与 DDR 提交
  背压重试保留已装载字节与进度，不整 beat 重读、不重新清零、不绕过预算。
  成功写出的字节与读取字节一致（测试用 lease 级累计访问量与仲裁器字节数
  双重独立核对）。
- lease 管授权与 ready_cycle，预算管带宽，二者不重复计延迟。

### 3.6 完整 L/U 因子保存

`FactorStoreUnit` 按 front 行分块（有界 staging + 统一读预算分段）：

- L（n×p）：行 `r<p` 取 packed A11 的严格下三角、对角补 1、上三角补 0；
  行 `r≥p` 直接取已求解的 A21 值（L21）。
- U（p×n）：列 `<p` 取 packed A11 的上三角含对角、下三角补 0（U11）；
  列 `≥p` 直接取已求解的 A12 值（U12）。
- 按 FACTOR/REGION 的地址与 stride 写入；MemoryService 队列满为可重试
  背压，保留行内提交进度，不重复提交已接受的写；按 request_id 精确归属；
  终态前等待自身全部已接受写确认；成功字节数为 `8·n·p`。
- 只由 PANEL/TRSM 计算出的真实值填充，不使用未计算的 A12/A21 冒充。

### 3.7 失败位置、abort/reset

- 失败位置映射：右解/对角 → 全局主元列；左解 solve → 全局主元行；更新
  非有限结果 → 该更新目标 tile 的全局列。completion 与扩展 trace
  （`panel_tasks.csv` 的 mode/`failed_column`/`group_id`）按此口径。
- 失败不发布 READY、不扰动对角、不切换策略；`STATIC` 失败不是奇异性结论。
- 定向 abort：只取消该事务，运行任务在安全点结束并跳过写回；取消中保留
  tracker、组与 lease，全部排空后终态 `ABORTED`；reset 使旧组绑定、旧响应、
  旧句柄与旧版本失效。
- 数值模式：默认 FP32（FMA、渐进下溢、无 fast-math）；显式
  `Fp64Diagnostic` 模块诊断模式在 tile 内用 binary64 计算并舍入回 FP32。
  **回存储格式时逐有效元素检查有限性**：舍入溢出到 Inf/NaN 时返回
  `NON_FINITE_VALUE`、报告正确失败列/行，tile 不提交（不把 Inf 写入
  front）；失败坐标沿用各模式约定——`LU_DIAG` 为全局主元列
  `pivot_base + local`（其余模式局部坐标，由 controller 映射）；
  无自动回退，报告标记。

### 3.8 手算案例

**A. 左解** `L=[[1,0],[2,1]]`，`B=[[3,1],[10,5]]`：
`x0=[3,1]`；`x1=[10,5]−2·[3,1]=[4,3]` → `X=[[3,1],[4,3]]`。

**B. 右解** `U=[[2,3],[0,4]]`，`B=[[4,14],[2,11]]`：
第 1 行 `x0=4/2=2`，`x1=(14−2·3)/4=2`；第 2 行 `x0=2/2=1`，
`x1=(11−1·3)/4=2` → `X=[[2,2],[1,2]]`。

**C. 4×4 左解**（单位下双对角，次对角 2、3、4；`b=[1,4,9,16]`）：
`x0=1`；`x1=4−2·1=2`；`x2=9−3·2=3`；`x3=16−4·3=4` → `x=[1,2,3,4]`。

### 3.9 SolveController 复用接口

后续前代/回代复用同一 PE 池与同一 controller 模式，不实现完整 solve：

- 左单位下三角前代 `Ly = Pb` 可直接复用 `TRSM_LEFT` 的能力（单位对角、
  递增方向）；
- 回代 `Ux = y` 是**上三角左解、递减方向**，需要一个新的 solve 任务模式
  （递减的行方向 + 非单位对角检查），**不能**把当前右解 `XU=B`
  （递增方向）冒充回代；
- `IComputeController` 已固定 `start/tick/on_task_response/abort/
  poll_response` 契约，SolveController 只需生成自己的 tile 任务；
- 版本 key 的 view tag 空间与步屏障/失败映射规则可原样复用。

## 4. 验收、报告口径与未完成边界

### 4.1 验收

- 手算案例 A/B/C（kernel 级独立期望）；
- 真实数据路径 `LOAD → PANEL → TRSM_LEFT/TRSM_RIGHT → STORE_FACTOR`
  （n=42、p=23、u=19）与独立 FP64 三角 oracle 对照：L/U 最大差 < 5e-3、
  `L11·U12−A12` 与 `L21·U11−A21` 归一化残差 < 5e-3、mask/单位对角、
  A11/A22/guard 不变、完整因子字节 `8·n·p`；
- 左右引擎可独立调用（直接 engine 级用例）；
- 分组：2 组×2 PE 实验配置、命令独占组、组内多 PE 并行、跨操作类型/跨
  front 复用、每 PE 阵列排他、adapter 数不等于 PE 数；
- 带宽：仲裁器单元测试（单请求方 grant、轮转、剩余不共享、过期请求清理）
  与低带宽集成（扫描/装载/写回/DMA 竞争，无饥饿）；
- 失败：零/过小对角、NaN/Inf、倒数溢出、非有限结果；abort/reset/复用；
- 非法输入：kernel 形状、tile_size、backend、lease、容量。

### 4.2 报告口径

- 设备执行报告（`summary.json`/`panel_tasks.csv`/`commands.csv`）与外部
  参考校验报告分别表达；局部 TRSM 残差不能冒充原始 `Ax=b` 残差；没有
  solution 时 residual/backward error 标记未执行，不填 0。
- 周期为未综合校准的架构模型周期，不声称实测性能/面积/加速比。
- 2 组×2 PE 是本次实验配置；组数与每组模块数可配置。

### 4.3 未完成边界

顶层 `GEMM_SCHUR` 数值 controller 已在 T10 实现（含局部 INT32），真实
assemble/STATIC 树形分解已在 T11 实现，见
[T11 集成设计](T11_真实装配与树形分解集成.md)。树形前代/回代 solve、
DYNAMIC_PANEL、RTL、lookahead、带宽拆分与额外 PE 规模实验仍未完成；
solve 与动态策略在真实后端显式 `POLICY_NOT_SUPPORTED` 或依赖失败。
