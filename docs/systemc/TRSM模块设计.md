# TRSM 模块设计

实现基准：2026-10-11 当前工作区。TRSM 在 Panel 之后计算 U12/L21，为 Schur 更新与持久因子提供数据。
数学背景见 [矩阵求解理论](../theory/矩阵求解理论基础.md)，公共执行机制见
[共享计算引擎与 PE](共享计算引擎与PE设计.md)。

## 1. 架构与执行路径

```text
CommandDevice → TrsmExecutionUnit → PanelEngine → TrsmController
→ TileDispatcher → MultiModePe → front A12/A21 写回 → 唯一终态
```

TrsmExecutionUnit 校验 opcode、形状、操作数身份、lease 和 Panel 锁存的数值上下文。
TrsmController 增量生成对角 solve 与后续更新任务；PE 使用 FP32 TRSM/GEMM 内核。
左右解共享组/PE，单条 TRSM 独占一个组至终态消费；同一 front 保守串行。

## 2. 模块职责、请求/响应与数据布局

### 2.1 请求（控制面 → 共享引擎）

`ExecRequest`（控制面 内部）→ `PanelEngineRequest`：

- 身份：`txn_id/epoch/command_id/node_id` + 适配器身份
  `adapter_key = (kind << 32) | unit_id`（不同 kind 的 unit_id 可以相同，
  响应路由必须带 kind）。
- 本模块 `compute_kind` 为 TrsmLeft 或 TrsmRight；GemmSchur 使用同一引擎，树形求解使用独立单元。
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
  乘积），由 PANEL 响应 → 控制面 front slot → 后续 TRSM 请求传递；TRSM 的
  非单位 U 对角检查**不得**从已消元的 front 重新推导尺度。
- 授权与布局：`front` 句柄 + `front_lease`（由 控制面 在派发时原子获取的
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
  与全 front 网格索引不会错误别名；新 compute context 建立时清除该 slot 的旧版本，reset 也清除版本。

### 2.3 响应（共享引擎 → 控制面）

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

- 增量入队：solve 结果提交后生成对应更新；队列满时保留生成进度
  （solve/update 生成标志），下一拍继续，不展开无界任务图。
- 依赖：每个更新依赖其 solve 任务；同目标 tile 的读写在版本表 + 步屏障下
  串行；依赖只引用最近生产者与对应 solve，不堆积历史依赖。
- 释放顺序：controller 终态且无在途任务 → 引擎产出唯一终态响应 →
  适配器接收（poll）后释放组，控制面收尾并释放 lease；abort 时先排空再释放。

### 3.4 时序模型

| 模式 | 阵列占用周期 |
| --- | --- |
| `TRSM_LEFT` | `tsolve_fill + k + n` |
| `TRSM_RIGHT` | `tsolve_fill + div_latency + (k−1) + k + m` |
| `GEMM_UPDATE` | `fill + k×II + m×drain_row_cycles` |

装载/写回按统一预算分拍计；失败时延按实际检测阶段（对角失败在倒数预计算
列，求解失败计到失败行/列）。装载在 Loading 阶段按预算分段进行，数据到齐后
才对快照执行功能计算。

### 3.5 存储、带宽与因子保存

访问服从 [共享引擎的预算与版本契约](共享计算引擎与PE设计.md)。
A12/A21 写回后保存完整 L/U 的机制见 [装配与因子保存设计](装配与因子保存设计.md)。
单 RHS 树形前代/回代由 [SolveExecutionUnit](树形求解模块设计.md) 执行，不经过 TrsmController。

### 3.7 失败位置、abort/reset

- 失败位置映射：右解/对角 → 全局主元列；左解 solve → 全局主元行；更新
  非有限结果 → 该更新目标 tile 的全局列。内部响应与 `panel_tasks.csv` 保留该坐标；外部 Completion 没有 failed_column 字段，TRSM 的 pivot_count 为 0。
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

## 4. 实现与验证入口

| 对象 | 代码 | 阅读重点 |
|---|---|---|
| `TrsmExecutionUnit` | [panel_exec_unit.hpp](../../graduation-code/systemc/include/controlplane/panel_exec_unit.hpp)、[实现](../../graduation-code/systemc/src/controlplane/panel_exec_unit.cpp) | 形状、同一 front 身份、阈值传递和响应适配 |
| `TrsmController` | [trsm_controller.hpp](../../graduation-code/systemc/include/panel_lu/trsm_controller.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/trsm_controller.cpp) | 左右任务图、尾块、增量生成与步屏障 |
| FP32 内核 | [tile_kernels.cpp](../../graduation-code/systemc/src/panel_lu/tile_kernels.cpp) | 单位下三角左解、非单位上三角右解、非有限检查 |
| 模块测试 | [trsm_real_tests.cpp](../../graduation-code/systemc/tests/trsm_real_tests.cpp) | 手算值、不对齐 pivot、guard、背压、失败坐标、取消和复用 |

数值对照使用独立参考和手算值；同时核查非目标块字节、任务真实写回、版本顺序和终态唯一性。
整体分解的因子误差由 tree_factor_check 检查，命令 OK 本身不能证明精度。
