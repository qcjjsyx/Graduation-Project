# 项目总体介绍（Project Overview）

## 1. 项目目标与约束

### 1.1 总体目标

构建一个面向 **稀疏线性方程组求解** 的软硬件协同系统，参考 **MUMPS / multifrontal** 思路，在硬件端完成节点级（node-level）的 **frontal matrix** 分解与更新，输出全局 LU 的一部分与对父节点的更新矩阵，最后生成全局矩阵的LU分解，

核心展示点：

* **节点级波前矩阵处理** ：主元块分解 + TRSM + Schur 更新（GEMM）+ update 写回父节点（extend-add/scatter）
* **全局调度** ：依赖感知的预取与双缓冲，保证计算核心不断供
* **低精度/整数路径** ：在缺乏浮点矩阵核的条件下，通过 量化与（可选）迭代求精实现可控误差

### 1.2 硬件约束

* **矩阵计算核心（GEMM/TPU 核）** ：支持int32 的 + - * 。能够支持一般的矩阵与向量的运算

---

## 2. 算法与执行流

### 2.1 Multifrontal 节点处理（单 node）

每个 node 的 frontal matrix $F$ 包含：

* node 本身变量对应的条目（来自原矩阵 A）
* 子节点上传的更新矩阵（contribution/update），通过 **extend-add** 累加装配到 $F$

处理流程：

1. **装配（assembly）** ：$F\leftarrow extend\_add(F_A, Update)$
2. **主元块分解（pivot / factorization）** ：对 $F_{pivot}(F_{11})$ 做（块）LU，产生 L/U 因子（全局 LU 的一部分）
3. **TRSM** ：求解 $F_{21}←F_{21}U_{11}^{−1}$   $F_{12} \leftarrow L_{11}^{-1}F_{12}$三角求解步骤
4. **Schur 更新（GEMM）** ：$F_{22}\leftarrow F_{22} - F_{21}F_{12}$,生成对父节点的更新矩阵
5. **输出** ：

* 写回 L/U 因子到 DDR（factor_writer）
* 当前节点执行完成后，生成面向父节点的 update payload，并写入独立 update 区；父节点进入执行前，再由 front_loader / assembly 路径结合本地贡献与所有子节点 update 完成装配，并确定该父节点的 node-scale

### 2.1.1

父节点进入执行前，不直接使用软件侧预装配好的统一 front，而是由硬件侧根据本地贡献 `A_local` 与所有子节点 update payload 完成装配，并在装配完成后决定该父节点的统一 `node-scale`。

为此，区分三类数值格式：

1. **S_format（assembly 输入格式）**

   $$
   x\approx q_x \cdot 2^{e_s}
   $$

   其中：
   \- `q_x` 为 int32 mantissa
   \- `e_s` 为 source 自带指数
   \- source 可以是：
   \- 当前节点本地贡献 `A_local`
   \- 某个子节点 update payload
2. **M_format（node 内矩阵值格式）**
   $x \approx q_x \cdot 2^{e_n}$
   其中：
   \- `e_n` 为当前 node 的统一指数
   \- node 执行期间 `e_n` 固定
3. **QF_format（乘子格式）**
   $l \approx \frac{m}{2^F}$
   用于 node 内 LU / TRSM 中的乘子表示。

父节点装配流程如下：

1. `dep_scoreboard` 判断所有子节点 update 已就绪，允许父节点进入装配阶段；
2. `front_loader` 先读取本地贡献和各 child update 的 exponent metadata；
3. 确定统一装配参考指数 `e_asm`；
4. 各 source 按 `e_asm` 做指数对齐后，通过 `map_table` 在线装配到 assembly buffer；
5. 在 assembly buffer 中统计 `maxabs_acc`；
6. 装配完成后根据 `maxabs_acc` 决定最终 node-scale `e_n`；
7. 将 assembly buffer 重新量化写入 front SRAM，得到 node 内执行所需的 `M_format` 数据。

当前原型版建议使用：
$e_{asm} = \max(e_{local}, e_{child,1}, e_{child,2}, \dots)$

该方案实现简单，但可能导致较小 source 在右移对齐时丢失。因此测试使用的矩阵会性质比较良好

### 2.2 依赖与预取风险（父子数据冒险）

若父节点任务被预取到 Buffer B，而子节点 update 尚未写回父节点区域，则父节点使用过期数据。

体系化解决：

* **软件侧** ：任务队列排序（sibling scheduling）尽量插入无依赖节点填充空隙
* **硬件侧** ：dep_scoreboard 维护 `pending_children`，仅当父节点 `front_ready` 才允许 task_fetch 发射与 buffer_mgr 预取

### 2.3 轻量级 ready-task 选择（机会发现）

当前系统中，`dep_scoreboard` 首先保证任务发射的正确性：只有当 `pending_children = 0` 且 `front_ready = 1` 时，节点才进入 ready 集合。

在此基础上，为了提高双缓冲下预取与计算的重叠概率，引入轻量级 ready-task selector。其目标不是做全局最优调度，而是在多个合法可发节点中，优先选择当前更适合预取的节点。

当前原型版采用简单启发式：

1. 优先选择 `front_size_class` 与当前计算窗口更匹配的节点；
2. 若接近，则优先选择 `critical_level` 更高的节点；
3. 若仍接近，则优先选择 `child_update_count` 更小、assembly 成本更低的节点。

其中，以下字段由软件侧在 `Node_Task` 中预先给出：

- `front_size_class`
- `critical_level`
- `child_update_count`

该机制的定位是“在正确性 gating 之上增加轻量机会发现”，不宣称全局最优，也不直接承诺整体性能一定提升。

---

## 3. 硬件架构总览

### 3.1 主要模块

* **GCU（Global Control Unit）**

  * `task_fetch`：从 DDR 读取 `Node_Task` 描述符，做 decode
  * `dep_scoreboard`：
    - 维护 `pending_children / front_ready`
    - 仅当父节点依赖满足时，允许其进入 ready 集合
    - 向 `task_fetch / buffer_mgr` 输出 ready 节点集合，供轻量级任务选择器进一步决定“先发谁”
  * `buffer_mgr`：双缓冲管理，状态机（IDLE/LOAD/READY/PROC/WB），逻辑上对节点进行一个宏观的管理，包括加载数据，可以开始计算，确定更新矩阵写回等任务。
  * `micro_scheduler`：节点内调度，驱动 SFU/HPU/ATU/矩阵核协作，完成节点对应的frontal matrix的全部计算
* **ATU（地址变化单元）**

  * 该模块负责逻辑行与物理行的映射。LU分解涉及除法，我们希望主元尽可能大，需要进行行置换等操作。为了避免大规模搬移数据，设计了该模块。
* **HPU（层级主元单元）**

  * 进行LU分解时的主元选择
  * 当前版本采用了锦标赛树的形式来筛选出主元
* **SFU（Signed Functional Unit）**

  * signed `+ - * /`：定点乘子、缩放/对齐、TRSM 相关运算、控制路径、简单LU相关运算、控制路径
  * 实际上为一简单的CPU核
* **矩阵计算核心（TPU/GEMM core）**

  * signed `+ - *`：用于 Schur 更新 GEMM 等高吞吐部分
* **辅助模块**

  * `dma_front_loader`：
    - 读取 parent node 的本地贡献与 child update payload
    - 预读 exponent metadata
    - 配合 assembly 单元完成多 source 在线装配
    - 统计 assembly buffer 的 `maxabs`
    - 根据 `node-scale` 对装配结果重新量化并写入 front SRAM
  * `update_writer`：update 写更新区；
  * `dma_factor_writer`：写回 L/U
  * 其他一些必要的辅助模块，目前还没有确定

---

## 4. 软件架构与工作内容

### 4.1 软件侧

1. **I/O 与置换管理** ：读取矩阵（CSR/CSC/mtx）
2. **符号分析（Symbolic）** ：

* ordering（AMD/METIS/SCOTCH 或替代）
* elimination tree构建
* supernode (超节点，目前限制最大为256*256)形成

3. **任务生成（Node_Task）** ：`Node_Task` 除 front 尺寸、地址、映射等必要字段外，还需要补充以下元数据：

   - `front_size_class`：当前 node 前沿矩阵规模类别，用于轻量调度
   - `critical_level`：节点关键级别，用于 ready-task 选择
   - `child_update_count`：子节点 update 数量，用于粗略估计 assembly 成本
   - `e_local` 或本地贡献 exponent metadata 地址
   - child update descriptor / exponent metadata 地址
4. **map_table 生成（extend-add 映射）**

   确定子节点的更新矩阵应该加到父节点的哪一行/列中
5. **任务队列排序（sibling scheduling）** ：避免父子相邻导致预取风险
6. **内存规划与序列化（ABI）** ：
7. **初始化本地贡献准备** ：

* 根据 symbolic 结果提取每个 node 对应的 `A_local`
* 对 `A_local` 进行预量化，得到 `(q_local, e_local)`，作为 assembly 输入格式 `S_format`
* 将其与 `map_table`、`Node_Task` 一起下发到板端 DDR

8. **验证与指标** ：

* residual、相对误差、sat_count/裁剪统计

9. **迭代求精**：在硬件侧完成低精度/整数化 LU 分解后，利用硬件中已有的量化 `L/U` 作为近似因子，在软件侧执行迭代求精闭环：

   - 将量化后的右端项 `b` 送入硬件，利用已有 `L/U` 做前代/回代，得到初始近似解 `x_0`；
   - 软件侧使用真实高精度 `A` 与 `b` 计算残差：$r_k = b - A x_k$  ;
   - 对残差单独确定 residual-scale `e_{r,k}`，量化后重新送入硬件；
   - 硬件利用已有 `L/U` 解修正方程，得到 correction `d_k`；
   - 软件侧以较高精度更新：$x_{k+1} = x_k + d_k$
   - 根据 residual 范数、更新量大小和最大迭代次数决定是否停止。

   该模块的定位是：

   - 作为低精度 LU 的可靠性兜底；
   - 展示在无浮点矩阵核条件下，低精度分解结果仍可通过软件辅助提高最终解精度

---

## 5. 后续工作

1. **软件侧闭环** ：symbolic → tasks/map_table → `A_local` 预量化 → 生成二进制 → Python 参考验证 → 板端数据下发
2. **硬件调度闭环** ：dep_scoreboard + ready-task selector + buffer_mgr + task_fetch + micro_scheduler 联动跑通
3. **assembly 正确性** ：验证多指数 source 的在线对齐装配、`e_asm` 决定、`node-scale` 重定标和 front SRAM 写入流程
4. **迭代求精** ：验证在已有量化 `L/U` 基础上的 software-assisted refinement 是否能改善 residual 与最终解精度
5. **风险评估** ：统计 `align_drop_count`、`asm_overflow_count`、`requant_sat_count`、迭代求精中的 residual stagnation 情况
