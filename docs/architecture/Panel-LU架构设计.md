# Panel-LU 硬件架构与多模式 PE 接口/时序原型（T06）

状态：**T06 架构设计 + 可执行原型已交付；T07 正式 FP32 STATIC Panel 引擎已接入
T05 真实数据面/控制面并完成数值验证；T09 将该引擎扩展为共享计算引擎**
（Panel/TRSM 共用 PE 池、`compute_group_count × pes_per_group` 分组、统一
片上带宽仲裁、完整 L/U 保存），详见
[T07_Panel_LU数值引擎设计](../systemc/T07_Panel_LU数值引擎设计.md) 与
[TRSM架构设计](TRSM架构设计.md)。本文是
Panel-LU 子系统与共享多模式 PE 池的设计权威文档；T06 合成原型配套可编译接口
（`graduation-code/systemc/include/panel_lu/`）、可执行时序/资源原型
（`src/panel_lu/`）与测试（`tests/panel_lu_prototype_tests.cpp`、
`tests/panel_lu_systemc_tests.cpp`），T07 正式引擎与测试见
`src/panel_lu/panel_engine.cpp`、`tests/panel_lu_real_tests.cpp`。

边界声明（必须随结果一起引用）：

- 本原型是**架构原型**：`execution_backend=architecture_prototype`、
  `numerical_validation=not_performed`、`architecture_prototype=true`；
- 原型内部对显式合成负载做的最小 FP32 tile 运算（`tile_kernels.hpp`）只用于产生
  数据相关的主元失败列与依赖版本，**不是**正式设备数值后端，不接入
  `CommandDevice`，不给真实 command 输出数值成功；
- 周期数是架构模型周期，不是芯片实测；所有参数是设计假设，没有综合/流片校准；
- 不宣称 FP32 全链路数值通过，不宣称加速比；DYNAMIC_PANEL、INT32、RTL 不属于
  T06。

---

# 一、总体结构

## 1.1 模块图

```text
                        ┌────────────────────────────────────────────────┐
 PANEL_LU command ─────▶│ PanelController（每 command 一个）              │
 （T05 ExecRequest        │  Scan → ScanReduce → 逐步 Diag/Trsm/Gemm         │
   的同构替换接口）       │  → PVector → Done/Failed/Aborted                │
                        └───────────────┬────────────────────────────────┘
                                        │ TileTask（mode/task_id/command_id/
                                        │ node_id/epoch/buffer/offset/stride/
                                        │ eff M/N/K/mask/deps/output_version）
                                        ▼
                        ┌────────────────────────────────────────────────┐
                        │ TileDispatcher + TileScoreboard（有界队列）      │
                        │  · 依赖就绪（producer 已 commit）                │
                        │  · 版本就绪（store.version ≥ 操作数版本）        │
                        │  · 同目标 tile 写者顺序                          │
                        │  · 双 front 轮转公平                             │
                        └───────────────┬────────────────────────────────┘
                                        │ 每拍轮转选择空闲 PE
                                        ▼
      ┌──────────────┬──────────────┬──────────────┬──────────────┐
      │ MultiModePe0 │ MultiModePe1 │     ...      │ MultiModePeN │  pe_count 个
      │ 16×16 FMAC   │              │              │              │  同构多模式
      │ 4 task slot  │              │              │              │
      │ input FIFO   │              │              │              │
      │ cor. div ALU │              │              │              │
      └──────┬───────┴──────┬───────┴──────────────┴──────────────┘
             │ 装载（共享读预算）        │ 写回（共享写预算 + 有界 reserv.）
             ▼                          ▼
      ┌────────────────────────────────────────────────────────────┐
      │ TileStore：每 command 一个 16×16 tile 网格，slot+generation  │
      │ + 每 tile 版本号（commit 可见性）                            │
      └────────────────────────────────────────────────────────────┘
```

- command controller → tile task generator/scoreboard → shared dispatcher →
  PE pool → tile writeback → command completion：每一层的输入/输出、背压与可见性
  时刻都在本文定义，并由原型测试覆盖。
- 复用已有类型：状态码一律 `hw::command::StatusCode`，buffer 句柄一律
  `hw::dataplane::BufferHandle`（slot+generation）；不新增状态码，不修改外部
  Command/Descriptor/Completion 二进制布局（32B/64B/64B、schema v2）。
- `PANEL_LU` 对外只写 A11；A12/A21/A22 仍由顶层 `TRSM_LEFT`/`TRSM_RIGHT`/
  `GEMM_SCHUR` command 负责，Panel 内部任务不得重复处理。首版采用 tile 步屏障，
  不做跨步 lookahead。

## 1.2 command 生命周期（首版 step barrier）

```text
SCAN         全 front（n×n）扫描：有限值检查 + 精确 maxabs 归约（共享读预算，轮转仲裁）
SCAN_REDUCE  归约延迟结束后锁存 front_pre_elimination_max_abs 与阈值乘积
DIAG(s)      生成 LU_DIAG(s,s)                     deps: 无（版本由 store 保证）
TRSM(s)      生成 TRSM_LEFT(s,j>s) 与 TRSM_RIGHT(i>s,s)   deps: 本次 diag
GEMM(s)      生成 GEMM_UPDATE(i,j>s)               deps: 对应左右 TRSM
→ s+1，直到全部 pivot tile 完成
PVECTOR      写完整 identity P-vector（pivot_dim×4B），等写确认
DONE         发布 OK（pivot_count=pivot_dim）
```

失败：任一任务返回数值错误 → 命令终态为该状态、`pivot_count` = 首个失败列、
不写 P-vector；依赖失败传播沿用 T05 语义。abort：停止生成，定向取消本 command
队列任务与 PE 上任务，等本 command 未终结任务安全结束后发布 `ABORTED`。

---

# 二、PE 微架构

## 2.1 16×16 阵列与模式数据流

PE = 16×16 output-stationary FMAC 阵列（256 FMAC，T=16 为已确认尺寸），所有模式
共用同一阵列、同一输入 FIFO、同一写回通路。数据流：

| 模式 | 计算 | 阵列数据流 | 关键资源 |
| --- | --- | --- | --- |
| `GEMM_UPDATE` | `D -= A(m×k)·B(k×n)` | 每拍喂一列 A、一行 B，256 MAC/拍，结果驻留累加器 | FMAC 阵列 |
| `LU_DIAG` | packed LU（k×k） | 逐 stage：角 ALU 做除法/倒数，输出列/行经对角链路回灌，尾块用阵列做秩 1 更新 | 角除法 ALU + 对角反馈链路 |
| `TRSM_LEFT` | `X = L11⁻¹·A12` | 三角行流式求解（Kung 型）：A 沿边流下，已解行回灌 | FMAC 阵列 + 行回灌 |
| `TRSM_RIGHT` | `X = A21·U11⁻¹` | 先用角 ALU 预计算全部对角倒数（k 个独立倒数，II=1），再逐行前向代入 `x_j = (a_j − Σ_{t<j} x_t·U[t][j])·r_j`，已解元素回灌 | 角除法 ALU + FMAC 阵列 + 行回灌 |

- **特殊除法/倒数通路**：阵列角 ALU 支持 FP32 除法/倒数，`div_latency_cycles`
  固定延迟、II=1。`LU_DIAG` 每 stage 使用一次（在反馈环内）；`TRSM_RIGHT`
  的对角 tile 非单位，任务开始时**先经同一通路预计算 k 个独立倒数**
  （`div_latency + k-1` 拍），再进入纯乘法前向代入，因此后续列依赖的是
  倒数结果而非除法器 II。`TRSM_LEFT` 单位对角，不使用除法通路。
- **反馈连接**：对角 tile 的列/行结果通过阵列对角链路回灌到后续元素，使
  LU/TRSM 不需要额外的串行行循环；`GEMM_UPDATE` 不使用反馈链路。
- **输入 FIFO**：每个 PE 一个片上操作数 FIFO（默认 4096B），驻留任务的
  A/B 操作数（满 tile 最大 2048B = A+B）。目标 tile 不占 FIFO 容量（见下）。
  任务操作数超过 FIFO 余量时 `try_accept` 返回 false（背压，不是错误）。
- **目标累加缓冲**：每个 task slot 持有独立的 16×16 累加寄存器组
  （1KB/slot，固定存在，不计入操作数 FIFO）。GEMM 的 `D` 先装入累加器再
  累加，LU_DIAG 的对角 tile 也直接读入累加器。**容量规则**：操作数 FIFO
  最低要求 = 最大单任务操作数集 `A+B = 2048B`（满 16×16 GEMM），低于该值
  或把累加器计入 FIFO 的配置会在 `validate_panel_config` 直接拒绝；满足该
  下界的配置一定可推进。结果只在**用写回通路搬完**后才对后继任务可见
  （原型内部可提前计算，但可见性由写回事件决定）。
- **task slot 与双缓冲**：默认 4 个 task slot/PE（论文同量级，仅作设计参考）。
  阵列一次最多执行一个 slot；另一 slot 可以并行装载，写回可与执行重叠，
  受共享读/写预算与有界写回 reservation 约束。

## 2.2 流水规则、延迟与 initiation interval

所有时延来自数据流参数，可按闭式手算；**没有任意常数延迟**：

| 事件 | 规则（`PeTimingConfig` / `PanelLuConfig`） |
| --- | --- |
| 装载 | `load_bytes = task_load_bytes(task)`；每拍消耗共享读预算 `load_bytes_per_cycle`；最后一拍后 `+load_latency_cycles` 进入 READY |
| 执行（GEMM） | `fill_cycles + k_eff·gemm_k_ii + m_eff·drain_row_cycles`（默认 `2 + k + m`） |
| 执行（LU_DIAG） | `(k_eff-1)·(div_latency + lu_update_cycles) + lu_pivot_check`（默认 `(k-1)·9 + 1`） |
| 执行（TRSM_LEFT） | `tsolve_fill + k_eff + n_eff`（默认 `2 + k + n`） |
| 执行（TRSM_RIGHT） | `tsolve_fill + (div_latency + k_eff-1) + k_eff + m_eff`（默认 `2 + (8+k-1) + k + m`） |
| 写回 | `task_writeback_bytes`；每拍消耗共享写预算；最后一拍后 `+writeback_latency_cycles` 结果可见 |
| 失败进度 | LU_DIAG 第 f 列失败：`f·(div_latency + lu_update) + lu_pivot_check`；TRSM_RIGHT 倒数预计算失败（第 f 列为零/非有限/倒数溢出为 Inf）：`div_latency + f`，求解阶段失败：`(div+k-1) + tsolve_fill + (f+1) + m`；TRSM_LEFT/GEMM 只计到失败行/列 |

- **LU_DIAG stage 反馈依赖（关键时序关系）**：
  1. stage j 的对角主元 `pivot_j` 在上一 stage 的第一个秩 1 更新拍写入
     累加器（`lu_update`）；
  2. 角 ALU 在累加器写回路径上做组合比较（零/阈值/非有限），失败位注册
     一拍（`lu_pivot_check`）；
  3. 检查通过后下一拍发射倒数/除法（`div_latency`），乘数列/行再回灌阵列；
  4. 乘数列与 `U(j,:)` 的第一个更新拍产生 `pivot_{j+1}`（`lu_update`）。
  因此**相邻 stage 的启动间隔 = `div_latency + lu_update`（默认 9 拍），
  不是除法器 II=1**：除法结果经反馈链路回到下一个主元，无法流水重叠。
  16×16 tile 上 15 次相互依赖的除法至少 15×9 = 135 拍。
  成功关键路径 = `(k-1)·(div_latency+lu_update) + lu_pivot_check`；
  第 f 列失败 = `f·(div_latency+lu_update) + lu_pivot_check`。
- 16×16 满 tile GEMM = 2+16+16 = 34 周期；LU_DIAG = 15·9+1 = 136 周期；
  TRSM = 2+16+16 = 34 周期。尾块按有效维度缩短；`k_eff,m_eff,n_eff ∈ [1,16]`。
- II：阵列对连续 GEMM 任务可以背靠背启动（双缓冲累加器），模型不插入气泡；
  **LU_DIAG 内部 stage 的 II 受反馈依赖限制为 `div+update`**；装载/写回与
  执行重叠受端口预算限制。
- 论文只提供参考（16×16 tile、每 PE 多 task slot、任务粒度仿真、综合校准的
  时延模型）。本项目参数**不是**论文综合结果，也没有做频率/工艺换算。

## 2.3 端口、bank 与容量

| 资源 | 默认 | 说明 |
| --- | --- | --- |
| PE 数量 | `pe_count`（默认 2） | Panel/TRSM/GEMM 共享；不存在按模式独立的计算池 |
| task slot/PE | 4 | 有限；slot 满 = 背压 |
| PE 操作数 FIFO | 4096 B（最低 2048B） | 驻留 A+B 操作数；最低容量 = 满 tile GEMM 的 A+B；累加器 1KB/slot 单列，不计入 |
| tile store 读预算 | 64 B/cycle | PE 装载与 full-front 扫描共享；扫描在每拍控制器相位请求，PE 优先 |
| tile store 写预算 | 64 B/cycle | tile 写回与 P-vector 写共享 |
| 装载队列 | 8 | 在途装载任务上界 |
| 写回队列/reservation | 8 | 在途写回任务上界；满则 ExecDone 停等 |
| dispatcher 队列 | 32（有界） | 满时 controller 计 `stall_dispatcher_full` 并下拍重试 |
| 扫描 | 共享读预算 + 8 周期归约 | 与 PE 装载共用 `load_bytes_per_cycle`（有界轮转仲裁）；不占 PE；T06 原型的独立扫描预算仅作历史近似 |
| 响应 hold | 0 | 响应背压可配 |

- 首版不做显式 bank 冲突模型：tile store 是每 tile 一个 slot 的抽象，带宽由
  共享读写预算表达；bank 级冲突列为 T07 的可选细化项（见 §7）。
- 容量规则：`tile_dim` 固定 16；`max_front_dim` 默认 256（可配到 512）；
  slot 数 = `max_resident_commands × ceil(max_front_dim/16)²`。front 或
  `pivot_dim` 超限、`tile_size ≠ 16`、队列/PE 为 0 等配置在 `configure()`
  或 `submit_command()` 明确拒绝（`BAD_DESCRIPTOR`/`POLICY_NOT_SUPPORTED`），
  **不静默截断**。

## 2.4 同 PE 排他、资源获取/释放顺序

- 一个 PE 同时最多一个任务占用阵列（测试逐对检查 `[exec_start, exec_done)`
  不重叠，pe_count=1/2 都验证）。
- 资源顺序：dispatcher 先做依赖/版本/写者检查 → 选空闲 PE → `try_accept`
  原子占用 slot+FIFO 额度 → 装载 → 执行 → 申请写回 reservation → 写回 →
  释放 reservation 与 FIFO，响应入队。任务不持资源等待（除自身流水阶段）。
- 失败/abort：未执行任务立即取消；执行中任务在安全点（当前阵列 pass 结束）
  终止且不写回；已进入写回的任务按“已写数据不回滚”语义完成字节搬运，但
  结果不提交版本、响应为 `ABORTED`。定向 abort 不影响另一个 front 的任务。

---

# 三、任务契约与局部调度

## 3.1 内部类型（`panel_lu/task_types.hpp`）

`TileTask`：`task_id/epoch/command_id/node_id/step/tile_row/tile_col/mode`、
`dst/lhs/rhs` 三个 `TileOperand`（buffer handle、byte offset、row stride、
要求版本）、有效 `dims{m,n,k}` 与 `mask{rows,cols}`、`deps`、`output_version`、
`pivot_base`、`sequence`、`front_pre_elimination_max_abs`、`threshold_product`。

`TileTaskResponse`：同一标识 + `pe_id/slot/status/reason/failed_column/
result_visible/enqueue/dispatch/load_done/exec_start/exec_done/writeback_done/
bytes/stall 分类计数`。

`TaskRecord`（原型汇总，trace 行）额外有 `committed/cancelled`。

校验（`validate_tile_task`）：M/N/K ∈ [1,16]；mask == 有效结果维度；
LU_DIAG 方阵；TRSM_LEFT `m==k`；TRSM_RIGHT `n==k`；stride 为 4B 倍数且 ≥64B；
deps ≤ 8。非法任务显式拒绝（`BAD_DESCRIPTOR`）。

## 3.2 依赖与版本

- 依赖只在 producer **写回完成并 commit** 后满足：dispatcher 维护已提交集合，
  `dispatch_cycle > producer.writeback_done_cycle`。
- 版本：每个 tile 有单调版本号。任务声明 `dst.version`（必须等于当前 store
  版本）与 `output_version = dst.version+1`；输入操作数要求
  `store.version ≥ operand.version`。同目标 tile 未提交写者存在时，后继任务
  不可发射；版本不匹配在 commit 时被拒绝并升级为控制故障（不静默通过）。
- 步屏障保证同目标更新顺序：step s 的 GEMM 必须全部提交后 step s+1 才生成；
  step s+1 的 LU_DIAG/GEMM 读取新版本。
- **同 front 合法并行**：同一 step 内 `TRSM_LEFT(s,j)` 之间、
  `TRSM_RIGHT(i,s)` 之间、不同 `GEMM_UPDATE(i,j)` 之间互不冲突，可分散到
  多个 PE 并行；同一 `(i,j)` 的跨步更新被版本与屏障串行化。

## 3.3 有界推进与公平性

- dispatcher 队列有界；满时 controller 观察到 `submit=false`，计
  `stall_dispatcher_full` 并在后续拍重试，不丢任务。
- **簿记回收**：`known_`（在队任务）在派发时释放；`records_`（在途调度元
  数据）在 commit/terminal 时释放；`committed_`（依赖可见性）只保留到命令
  终态，由 `release_command` 统一释放。task id 由原型**单调分配**，驻留槽
  释放后被下一条命令复用时不会与旧依赖/记录别名；命令结束后 dispatcher 的
  known/committed/record 三项计数必须归零（测试断言）。
- 双 command（双 front）按 command 轮转选择可发射任务；控制器服务顺序也轮转，
  使共享扫描端口公平共享。测试断言两个 command 都有提交发生在对方全部完成前
  （无饥饿），且单 PE 上仍逐任务排他。
- 每一步的任务数有上界（tile 网格 ≤ 16×16 时每 step ≤ 289），配合屏障，
  在途任务/图规模有界，不依赖无界任务图避免死锁。

---

# 四、全 front 扫描与 PANEL_LU 的依赖

- 装配完成后的 front 为 n×n FP32（`front_dim=n`），Panel 只分解 A11（p×p，
  `pivot_dim=p`）。扫描覆盖整个 front（n×n），包括 A12/A21/A22 区域：
  `scan_bytes = n²×4`，带宽 `scan_bytes_per_cycle`，归约延迟
  `scan_reduce_latency_cycles`。
- 扫描输出：`front_pre_elimination_max_abs`（精确 max）与有限值标志；
  NaN/Inf → `NON_FINITE_VALUE`，**不生成任何 tile 任务**、不写 P-vector。
- 所有对角 tile 共用同一尺度：`threshold_product = threshold_ratio ×
  front_pre_elimination_max_abs` 在 binary64 中精确计算（两个 binary32 的乘积
  在 binary64 可精确表示，避免二次舍入）。任务携带该值，LU_DIAG 逐消元列检查
  **实际消元后**的主元。
- 扫描必须完成且尺度有效后才申请计算 PE；扫描与 PE 装载共享同一读预算
  （有界轮转仲裁，见 §2.3），可与其他 front 的 PE 计算重叠，但与本 front 的
  Panel 计算是前置依赖。
- 扫描不融合装配、不做历史最大值估计、不从 Host 注入尺度（与计算基线一致）。

---

# 五、主元、packed LU 与写回格式

- STATIC：不搜索、不交换；LU_DIAG 逐列消元后检查：
  零 → `STATIC_PIVOT_FAILURE`；`|pivot| ≤ threshold_product`（非零）→
  `PIVOT_UNSTABLE`；NaN/Inf → `NON_FINITE_VALUE`。失败 `pivot_count` =
  首个失败列；不发 READY（T05 接入时），不中途换策略。
- P-vector：成功时写完整 identity（`p[i]=i`，INT32，`pivot_dim×4` 字节），
  写确认后才算 command 完成；写入经共享写预算，可观察争用。
- packed LU：A11 就地存放 packed L/U：严格下三角为 L（单位对角隐式），
  对角及上三角为 U。LU_DIAG 对角 tile 完成 packed 分解；TRSM_LEFT 读取其
  严格下三角（单位对角），TRSM_RIGHT 读取其非单位上三角（含对角）。
- 与 STORE_FACTOR 的关系：`STORE_FACTOR` 由顶层 command 读取 packed A11，
  按 `FACTOR_DESC` 拆成 L（`total×pivot`，单位对角可隐式或显式补 1）与
  U（`pivot×total`，三角 mask）。Panel 不直接写 L/U 区域，因此不需要在
  Panel 内复制因子；拆分规则、三角 mask、容量上限（`L ≥ total×pivot`、
  `U ≥ pivot×total`）由 T07/T11 的 STORE_FACTOR 实现负责，本文冻结其语义：
  L 对角必须单位化、U 必须含对角、尾块按 mask 截断。**未决项**：显式补 1
  的字节来源与 `FACTOR_DESC` 是否有 workspace 复用，留给 T07 与 T11 评审。
- P-vector 与 packed LU 都不引入设备可见的新格式；P-vector 仍是 INT32
  region，front/factor 仍是 FP32。

---

# 六、FP32 正式路径的舍入/FMA/阈值规则（T07 约束）

T06 原型不做数值验收，但正式路径的语义在此冻结，避免 T07 自行选择：

1. 设备路径为 IEEE-754 binary32；乘法-累加统一使用**融合乘加 FMA**
   （`a*b+c` 一次舍入），LU 消元、TRSM 内积、GEMM Schur 更新一致。
2. 除法：真除法（正确舍入）或倒数+Newton 修正，必须在 T07 文档中明确；
   `LU_DIAG` 的 pivot 倒数与 `TRSM_RIGHT` 的对角倒数预计算使用**同一条**
   角 ALU 通路，不允许静默切换近似除法。TRSM_RIGHT 首版为
   “倒数预计算 + 纯乘法代入”，倒数流水化（II=1）但不与其他任务共享，
   资源与延迟按 §2.2 计。**倒数结果必须逐列检查有限性**：有限非零的
   subnormal 对角（如 `1e-39f`）其倒数可溢出为 Inf，必须在预计算第 j 列
   立即报 `NON_FINITE_VALUE`（该列早于后续任何坏对角），不得推迟到求解
   阶段或用后续列替代；subnormal 输入本身合法，不要求 flush-to-zero。
3. subnormal：按 IEEE-754 渐进下溢处理，**不 flush-to-zero**；产生
   subnormal 不是错误，产生 NaN/Inf 才是数值错误。
4. 阈值乘积：`threshold_ratio` 是配置一次性规范化后的 binary32 值；
   `front_pre_elimination_max_abs` 是精确 FP32 max；乘积在 binary64 中计算
   （精确），比较 `|pivot| ≤ product`。原型已按此实现。
5. 非有限结果：输入或消元产生 NaN/Inf → `NON_FINITE_VALUE`；真实溢出到 Inf
   在 T07 可细化区分 `NUMERIC_OVERFLOW`（本原型统一报非有限）。
6. 失败进度：每个数值失败返回首个失败列/行，已完成工作与响应时间一致；
   不得全列套用成功时延。

---

# 七、16×16 物理 tile 与 `descriptor.tile_size`

- `descriptor.tile_size` 只接受 16（= 物理 tile）。其他值（含 0）在
  `check_front_capability` 显式拒绝：`BAD_DESCRIPTOR`，reason 说明物理 tile
  固定为 16、未静默忽略。codec/loader 仍按 schema v2 接受该字段的任意 u32，
  因此旧 stub fixture 不会被判为 schema 非法。
- 有效维度：`p = 16·k + r`（0 ≤ r < 16）时共 `k+1` 个 pivot tile，最后一个
  对角 tile 为 `r×r`（r>0）；相应 TRSM/GEMM 任务带有效 M/N/K 与 mask。
  `p < 16`、`p = 16`、`p > 16` 非整倍数（如 17/23/33）都有测试。
- tile 边界不改变未来 DYNAMIC_PANEL 的 pivot 候选范围：候选仍为装配后
  `k..pivot_dim-1` 的逻辑行，不缩小到当前 16 行；本版 STATIC 无候选搜索。
- 不支持的配置一律显式失败，不永久等待：非法配置在提交时拒绝，运行中所有
  等待都有资源上界与测试超时。

---

# 八、参数依据与限制

- 16×16：已确认架构决策（论文同尺寸仅作参考）。
- 4 task slot/PE、双缓冲累加器：论文设计思想参考；本项目容量、FIFO 字节数、
  端口预算、队列深度均为设计假设，未做综合校准。
- 时延闭式由数据流推导（§2.2），但：没有 RTL 综合、没有频率/工艺换算、
  没有 HBM/DDR 时序标定；周期数只用于相对比较与背压可见性验证。
- 不支持：RTL、真实 DDR/AXI、DYNAMIC_PANEL、INT32、lookahead、bank 冲突
  细化、多 RHS。**不得**把原型周期写成芯片周期或硬件加速比。

---

# 九、T07 接入清单

T06 不重构 T05 生产调度器；T07 按下列顺序迁移：

1. **backend 类型绑定**：`CommandDevice` 的 `IExecutionUnit` 由
   `TestStubUnit` 扩为可绑定 Panel 引擎（新增 `PanelExecutionUnit`，仍实现
   `kind()==UnitKind::Panel`、`submit/tick/poll_response/abort_epoch`）。
   `RequireRealKernels` 改为：Panel 槽绑定真实引擎，其余 kind 仍可显式 stub
   或拒绝。
2. **独立 kind 池 → 共享 PE 池**：删除按 kind 的 `units_` 池语义，改为
   `pe_count` 个 `MultiModePe` + 一个 `TileDispatcher`；TRSM/GEMM controller
   后续复用同一 dispatcher。T05 的 `panel_unit_count/trsm_unit_count/...` 仅
   保留为 legacy stub 配置并标注 deprecated。
3. **任务汇聚**：一个 `PANEL_LU` command 对应多个 tile task；T07 在
   `PanelExecutionUnit::submit` 内启动 `PanelController`，在
   `poll_response` 汇聚为单条 `ExecResponse`（含 `pivot_count`、
   `needs_pvector_write`）。T05 的 command/token/completion 语义不变。
4. **buffer ownership**：`ExecRequest` 的 `front_lease/temp_lease` 传给
   Panel 引擎；tile store 改为 T04 `BufferManager` 上的 tile 视图（slot+
   generation 已对齐），任务提交前用 lease 校验 range，写回经 `end_write`/
   `begin_writeback`。P-vector 写继续走 `ExecUnit` 内存写并等确认。
5. **定向取消**：`abort_epoch` 只用于全局 reset；node abort 走
   `abort_command(command_id/node_id)` → dispatcher `cancel_command` + PE
   `abort_command`，不得按 epoch 误伤另一 front。
6. **统计**：command 级 `CommandStats` 保持；新增 task 级 trace
   （`units.csv` 语义扩展或在 T07 增加 `tiles.csv`），`bytes_read/written`
   仍不得重复计费；`retry_count` 保持 0。
7. **数值原语**：T07 实现最小真实 tile LU_DIAG/TRSM_LEFT/TRSM_RIGHT/
   GEMM_UPDATE 原语，复用本接口；T09/T10 直接复用，不允许“Panel 等后续
   GEMM、后续 GEMM 等 Panel”的依赖倒置。
8. **主元/阈值**：`FRONT_DESC.word8/9` 解码后进入
   `PanelCommandRequest.policy/threshold_bits`；DYNAMIC_PANEL 仍返回
   `POLICY_NOT_SUPPORTED`。

**未决项（不伪装为已冻结）**：

- `FACTOR_DESC` 单位对角显式补 1 的写入方式与 workspace 复用；
- bank 冲突与 NoC 细化是否必要；
- STORE_FACTOR 的三角 mask 容量上限在最大 front（256/512）下的实测；
- 失败任务的部分结果可见性策略在真实 command 中的对外状态（当前原型
  不提交失败结果）。

---

# 十、原型验证证据

## 10.1 测试

| 测试 | 覆盖 |
| --- | --- |
| `panel_lu_prototype_tests`（plain C++，871 checks） | 配置/能力/阈值位拒绝（tile_size、DYNAMIC_PANEL、非法策略、NaN/Inf/负值/≥1 阈值）；完整/尾块/非法维度任务图；PE 跨模式排他；双 PE 并行；同目标版本顺序；依赖可见性；队列满、装载端口与写回背压与字节记账；手算周期与 LU stage 反馈链；扫描依赖与 NaN；数据相关失败（零主元、消元后零主元、低于阈值）；定向 abort；reset 失效；双 command 公平；驻留槽连续复用与簿记回收；2048B 操作数 FIFO 推进；TRSM_RIGHT 前向代入手算与除法敏感性（div=1000）；subnormal 对角倒数溢出在预计算列报 NON_FINITE_VALUE；task id 跨 UINT32_MAX 不截断；trace 可复现；报告标记 |
| `panel_lu_systemc_tests`（sc_clock，48 checks） | 每 posedge 一次 tick；与 plain 模型逐周期一致；PE 排他；依赖顺序；abort/reset；模块自身不推进 SystemC 时间 |
| `panel_lu_smoke_tail` | p=23 > 16 且带尾块的跨 tile trace（`panel_tasks.csv` 5 任务） |
| `panel_lu_smoke_mixed` | 双 front、双 PE 混合模式负载 trace |

## 10.2 手算例（配置：pe=1、load/writeback 64B/cycle(+2)、scan 64B/cycle(+8)、16×16、fill=2、div=8）

- p=16 LU_DIAG：scan 1024B/64=16 → reduce 8 → 生成 25；dispatch 26；
  load 16 拍 → 42；+2 → exec 44；`15×(8+1)+1=136` 拍 → 180；
  写回 16 拍 +2 → 199。
- p=32 step0 GEMM_UPDATE(1,1)：step0 diag 提交 247；TRSM_LEFT 提交 335；
  TRSM_RIGHT（m=16,k=16）`2+(8+15)+16+16=57` 拍，316→373，提交 392
  （共享装载端口，右侧操作数在左侧装载完成后装载，且阵列被左侧占用到
  316）→ GEMM enqueue 392、dispatch 393；load 3072B/64=48 → 441；
  +2 → 443；34 拍 → 477；写回 16 拍 +2 → 496。
- 失败进度：零主元列 1 → `1×(8+1)+1=10` 拍；消元后零主元列 1 → 10 拍；
  低于阈值列 0 → `0+1=1` 拍。以上均为测试硬编码期望，由公式独立手算。

## 10.3 运行

```bash
cmake -S graduation-code/systemc -B /tmp/t06-build -DBUILD_TESTING=ON
cmake --build /tmp/t06-build -j8
ctest --test-dir /tmp/t06-build --output-on-failure     # 16/16
/tmp/t06-build/panel_lu_prototype_smoke --mode tail --front-dim 23 --pivot-dim 23 \
    --pe-count 1 --trace-dir /tmp/t06-out/tail
/tmp/t06-build/panel_lu_prototype_smoke --mode mixed --front-dim 40 --pivot-dim 40 \
    --pe-count 2 --trace-dir /tmp/t06-out/mixed
```

trace：`panel_tasks.csv`（任务级事件时间与字节）、`panel_commands.csv`、
`panel_events.csv`、`panel_summary.json`（含
`architecture_prototype=true`、`numerical_validation=not_performed`）。

## 10.4 不变量↔测试

| ID | 不变量 | 验证 |
| --- | --- | --- |
| PL-I-1 | 一 PE 同时最多一个阵列任务 | `test_pe_exclusion_and_cross_mode`、SystemC |
| PL-I-2 | 依赖只在写回提交后满足 | `test_dependency_not_ready`、SystemC |
| PL-I-3 | 同目标 tile 版本顺序、无重叠写者 | `test_same_target_ordering` |
| PL-I-4 | 有界队列不丢任务、背压可观察 | `test_queue_full_backpressure`、`test_port_and_writeback_backpressure` |
| PL-I-5 | 结果只在写回完成后可见/提交 | `test_dependency_not_ready`、`test_reset_invalidates_old_work` |
| PL-I-6 | 定向 abort 不伤另一 command | `test_directed_abort`、SystemC |
| PL-I-7 | reset 后旧 epoch 不写回、不产生响应 | `test_reset_invalidates_old_work` |
| PL-I-8 | 不支持配置显式拒绝、不永久等待 | `test_config_validation`、`test_front_capability_checks` |
| PL-I-9 | 不支持的 tile_size/配置不静默忽略 | `test_front_capability_checks`（tile_size=8/0） |
| PL-I-10 | 失败进度与首个失败列一致 | `test_data_dependent_failure` |
| PL-I-11 | trace 可复现 | `test_reproducible_trace` |
| PL-I-12 | 原型不冒充数值后端 | `summary_json` 标记 + 文档边界 |
| PL-I-13 | task id 在有效生命周期内唯一；命令结束后依赖/记录表回收 | `test_slot_reuse_without_reset`（known/committed/record 计数归零） |
| PL-I-14 | 阈值位按本次请求校验并显式拒绝 | `test_front_capability_checks`（NaN/Inf/负值/1.0/2.0） |
| PL-I-15 | 操作数 FIFO 最低容量保证推进（累加器独立计数） | `test_fifo_holds_largest_operand_set`（2048B + p=32） |
| PL-I-16 | TRSM_RIGHT 计入倒数预计算延迟且随 div_latency 敏感 | `test_trsm_right_division_sensitivity`（div=1000，右侧 1034 拍、左侧 19 拍） |
| PL-I-16b | 倒数预计算逐列检查有限性，溢出在首列报告 | `test_trsm_right_forward_substitution`（diag(1e-39,0) → 第 0 列 NON_FINITE_VALUE） |
| PL-I-17 | task id 全链路 uint64，不截断 | `test_task_ids_beyond_uint32`（base=2^32+1，TRSM dep 不截断） |
