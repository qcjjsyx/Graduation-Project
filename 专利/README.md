# 文档 1：项目总体介绍（Project Overview）

## 1. 项目目标与约束

### 1.1 总体目标

构建一个面向 **稀疏线性方程组求解** 的软硬件协同系统，参考 **MUMPS / multifrontal** 思路，在硬件端完成节点级（node-level）的 **frontal matrix** 分解与更新，输出全局 LU 的一部分与对父节点的更新矩阵，实现可扩展的加速框架。

核心展示点：

* **节点级波前矩阵处理** ：主元块分解 + TRSM + Schur 更新（GEMM）+ update 写回父节点（extend-add/scatter）
* **全局调度** ：依赖感知的预取与双缓冲，保证计算核心不断供
* **低精度/整数路径** ：在缺乏浮点矩阵核的条件下，通过 BFP/定点化与（可选）迭代求精实现可控误差

### 1.2 关键硬件约束（已明确）

* **SFU** ：支持 *有符号* `+ - * /`（可做定点除法、缩放、对齐、TRSM 等）
* **矩阵计算核心（GEMM/TPU 核）** ：支持 有*符号* `+ - *`（适合高吞吐 MAC，但需处理“有符号乘法语义”问题）
* **HPU/ATU** ：围绕 pivot 选择、阈值策略、索引/追踪等控制与辅助

### 1.3 数值与数据格式决策（阶段性结论）

* 软件侧/系统侧采用  **块浮点（BFP）/定点** ：`x ≈ q · 2^e`
* 采用  **32×32 tile** ，并进一步分成  **16×16 子块** ，每子块  **1 个 exponent（int8）** ，mantissa **int32**
* 定标采用  **分位数（p99~p99.5）+ 裁剪** ，避免 outlier 使大量元素量化成 0/1
* LU 中“除法”的精度问题采用  **定点乘子** ：通过左移（F 或 t）形成带小数除法，并将 t 吸收到乘子 exponent 中（不要求软件计算 F）

---

## 2. 算法与执行流

### 2.1 Multifrontal 节点处理（单 node）

每个 node 的 frontal matrix FF**F** 包含：

* node 本身变量对应的条目（来自原矩阵 A）
* 子节点上传的更新矩阵（contribution/update），通过 **extend-add** 累加装配到 FF**F**

处理流程：

1. **装配（assembly）** ：F←FA+∑EcTUcEcF \leftarrow F_A + \sum E_c^T U_c E_c**F**←**F**A+**∑**E**c**TU**cE**c
2. **主元块分解（pivot / factorization）** ：对 F11F_{11}**F**11 做（块）LU，产生 L/U 因子（全局 LU 的一部分）
3. **TRSM** ：求解 F21←F21U11−1F_{21}\leftarrow F_{21}U_{11}^{-1}**F**21←**F**21U**11**−**1** 等三角求解步骤
4. **Schur 更新（GEMM）** ：F22←F22−F21F12F_{22}\leftarrow F_{22} - F_{21}F_{12}**F**22←**F**22−**F**21F**12**
5. **输出** ：

* 写回 L/U 因子到 DDR（factor_writer）
* 生成 update/contribution 写回父节点 frontal（scatter_engine + map_table）

> 重要：在 in-place LU 中，“消成 0”不是必须通过计算得到；通常直接覆盖存储 L 的乘子，避免依赖整数除法产生精确 0。

### 2.2 依赖与预取风险（父子数据冒险）

已识别风险：若父节点任务被预取到 Buffer B，而子节点 update 尚未写回父节点区域，则父节点使用过期数据。

体系化解决：

* **软件侧** ：任务队列排序（sibling scheduling）尽量插入无依赖节点填充空隙
* **硬件侧** ：dep_scoreboard 维护 `pending_children`，仅当父节点 `front_ready` 才允许 task_fetch 发射与 buffer_mgr 预取

---

## 3. 硬件架构总览

### 3.1 主要模块

* **GCU（Global Control Unit）**
  * `task_fetch`：从 DDR 读取 `Node_Task` 描述符，做 decode
  * `dep_scoreboard`：依赖记分牌（pending_children / front_ready）
  * `buffer_mgr`：双缓冲管理（Buffer A/B），状态机（IDLE/LOAD/READY/PROC/WB）
  * `micro_scheduler`：节点内调度（panel/tile 级），驱动 SFU/HPU/ATU/矩阵核协作（lookahead）
  * `dma_front_loader`：frontal 数据预取到 buffer
  * `scatter_engine`：extend-add 写父节点 frontal（指数对齐+累加）
  * `dma_factor_writer`：写回 L/U
  * （可选）`phase_ctrl`：阶段切换（pivot+update vs 大规模外部更新）
* **ATU（辅助追踪/表结构）**
  * 负责索引/追踪类元数据（你已完成初版并做了 tb）
* **HPU（pivot 选择与控制）**
  * 支持 THRESH / CALU（锦标赛树比较）等模式
  * 建议：HPU 顶层收集批量数据再送 core（更能发挥并行特性）
* **SFU（Signed Functional Unit）**
  * signed `+ - * /`：定点乘子、缩放/对齐、TRSM 相关运算、控制路径
* **矩阵计算核心（TPU/GEMM core）**
  * signed `+ - *`：用于 Schur 更新 GEMM 等高吞吐部分

### 3.2 三层调度模型（已对齐）

1. **Macro-scheduling（节点级生命周期）** ：task 获取、双缓冲预取、切换、DDR 写回
2. **Micro-scheduling（节点内流水）** ：panel/tile 依赖、scoreboarding、lookahead、冲突仲裁
3. **Phase/Dispatch（阶段切换与结果分发）** ：停用/重配模块、factor 写回、scatter 写父节点

---

## 4. 软件架构与工作内容

### 4.1 软件侧必须完成的链路（建议清单）

1. **I/O 与置换管理** ：读取矩阵（CSR/CSC/mtx），应用重排映射（P/Q）
2. **符号分析（Symbolic）** ：

* ordering（AMD/METIS/SCOTCH 或替代）
* elimination tree（etree）构建
* supernode 切分（可先简化）

1. **任务生成（Node_Task）** ：

* node_id、parent_id、children_count
* total_dim、pivot_dim、各类 DDR 地址

1. **map_table 生成（extend-add 映射）**
2. **任务队列排序（sibling scheduling）** ：避免父子相邻导致预取风险
3. **内存规划与序列化（ABI）** ：

* tasks.bin / map_table.bin / front_q.bin / front_e.bin / manifest.json

1. **初始化数值装配** ：

* 将原矩阵 A 的本地贡献填入各 node 的 frontal 初值
* 软件侧完成初始化量化（tile 32×32，subblock 16×16 exponent int8，q int32）

1. **验证与指标** ：

* residual、相对误差、sat_count/裁剪统计
* 与 SciPy 参考解对照（小规模可 dense 对照）

1. **迭代求精（可选但建议占位）**
   * iterative refinement 框架（残差、停止条件、更新）
   * 可作为“低精度 LU 的可靠性兜底”展示点

### 4.2 Python vs C++（你当前选择）

* 你对 C++ 不熟，建议软件侧  **优先 Python** ：更快构建、便于验证与数据生成
* 与硬件对接关键在 ABI（struct.pack、小端对齐、地址规划一致），语言本身不是瓶颈

---

## 5. 我认为需要补充/明确的关键点（工程上不可回避）

### 5.1 extend-add 的指数对齐规则与饱和策略

采用 subblock exponent 后，scatter_engine 写父节点 frontal 时必须做：

* exponent 对齐（以父为基准或重新选 exponent）
* 右移舍入规则
* 饱和/裁剪统计（sat_count）与必要时的“重标定”策略（可后续做）

### 5.2 定点乘子（F/t）的运行时选择

* 不建议软件计算每列 F（依赖 pivot 与运行时值）
* 推荐 SFU 动态选择 `t`，并把 `t` 吸收到乘子 exponent 中，统一为 `(mantissa, exponent)` 表示

### 5.3 验收指标与实验矩阵集

你提到验证矩阵规模约 2142^{14}**2**14 且稀疏。建议明确：

* 数据来源（随机SPD/非对称、SuiteSparse 等）
* 条件数/缩放策略
* 评价指标：残差、迭代次数、饱和率、性能（吞吐/延迟）

---

## 6. 里程碑建议（按风险优先）

1. **软件侧闭环** ：symbolic → tasks/map_table → 初始化量化 → 生成二进制 → Python 参考验证
2. **硬件调度闭环** ：dep_scoreboard + buffer_mgr + task_fetch 联动跑通（先不追求数值正确）
3. **extend-add 正确性** ：scatter 写回父节点与依赖释放逻辑正确
4. **数值路径** ：signed GEMM 语义补齐 + SFU 定点乘子 + 基础残差验证
5. **迭代求精（可选）** ：展示低精度可用性增强
