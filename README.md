# 项目总体介绍（Project Overview）

## 1. 项目目标与约束

### 1.1 总体目标

构建一个面向 **稀疏线性方程组求解** 的软硬件协同系统，参考 **MUMPS / multifrontal** 思路，在硬件端完成节点级（node-level）的 **frontal matrix** 分解与更新，输出全局 LU 的一部分与对父节点的更新矩阵，最后生成全局矩阵的LU分解，

核心展示点：

* **节点级波前矩阵处理** ：主元块分解 + TRSM + Schur 更新（GEMM）+ update 写回父节点（extend-add/scatter）
* **全局调度** ：依赖感知的预取与双缓冲，保证计算核心不断供
* **低精度/整数路径** ：在缺乏浮点矩阵核的条件下，通过 BFP/定点化与（可选）迭代求精实现可控误差

### 1.2 硬件约束

* **矩阵计算核心（GEMM/TPU 核）** ：支持int32 的 + - * 。能够支持一般的矩阵与向量的运算

### 1.3 数值与数据格式决策（阶段性结论）

为了使得例子中的浮点格式能够被矩阵核心计算，采用了这个阶段性的量化方式

* 软件侧/系统侧采用  **块浮点（BFP）/定点** ：`x ≈ q · 2^e`
* **16×16 子块** ，每子块  **1 个 exponent（int8）** ，mantissa **int32**
* 定标采用  **分位数（p99~p99.5）+ 裁剪** ，避免 outlier 使大量元素量化成 0/1

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
* 生成 update/contribution 写回父节点 frontal（scatter_engine + map_table）

### 2.2 依赖与预取风险（父子数据冒险）

若父节点任务被预取到 Buffer B，而子节点 update 尚未写回父节点区域，则父节点使用过期数据。

体系化解决：

* **软件侧** ：任务队列排序（sibling scheduling）尽量插入无依赖节点填充空隙
* **硬件侧** ：dep_scoreboard 维护 `pending_children`，仅当父节点 `front_ready` 才允许 task_fetch 发射与 buffer_mgr 预取

---

## 3. 硬件架构总览

### 3.1 主要模块

* **GCU（Global Control Unit）**
  
  * `task_fetch`：从 DDR 读取 `Node_Task` 描述符，做 decode
  
  * `dep_scoreboard`：依赖记分牌（pending_children / front_ready）
  
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

  * `dma_front_loader`：frontal 数据预取到buffer
  * `scatter_engine`：extend-add 写父节点 frontal（指数对齐+累加）
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

3. **任务生成（Node_Task）** ：

 		目前有一个node_task结构体作为我任务结构的参考

```c++
struct Node_Task {
    // --- 1. 身份与控制信息 ---
    uint32_t node_id;         // 节点ID，用于调试
    uint32_t flags;           // 标志位 (例如: Is_Leaf, Is_Root, Finish_Interrupt)
    uint32_t parent_id;       // 父节点id
    uint32_t children_count;   // 该节点有多少个子节点，用于初始化 pending_children[node_id]

    
    // --- 2. 几何尺寸信息 (用于配置循环边界) ---
    uint16_t total_dim;            // 当前波前矩阵的总维数 (N, e.g., 256)
    uint16_t pivot_dim;            // 需要分解的主元块大小 (K, e.g., 32, 64 或 256)
                                   // 注意: Update块大小 = total_dim - pivot_dim
    uint16_t nums_sub_matrix;      // 主元块中以16*16为单位的子矩阵数量 (pivot_dim / 16)，同时也确定了指数保存区域的大小
    uint16_t last_sub_matrix_size; // 最后一个子矩阵的实际大小 (如果 pivot_dim 不是16的倍数)
                                   // 注: 这两个字段主要用于配置循环边界和指数保存区域的大小

    // --- 3. 内存地址信息 (DMA搬运指针) ---
    uint64_t data_addr;       // 当前节点波前矩阵数据在DDR中的起始地址
                              // 软件需预先将 Original A 和 子节点的 Update 累加到这里
    uint64_t parent_address;  // 父节点波前矩阵在DDR中的基地址 (用于写回Update)
    // --- 4. 关键：父节点映射表 (Inter-Node Mapping) ---
    uint64_t map_table_addr;  // 指向一个数组的指针。
                              // 数组内容: 当前节点的第 i 行/列，对应父节点的第 j 行/列 (相对索引)


    // 下面两项为主元矩阵的分解结果
    uint64_t l_factor_addr;   // L 因子写到哪里 (DDR Base Address for L)
    uint64_t u_factor_addr;   // U 因子写到哪里 (DDR Base Address for U)

    uint64_t p_vector_addr;   // [NEW] 用于存储行置换历史 (Permutation Vector)
                              // 大小通常为: total_dim * sizeof(int32)
    uint16_t flag;            //预留字段                          
};
```

4. **map_table 生成（extend-add 映射）**

   确定子节点的更新矩阵应该加到父节点的哪一行/列中

5. **任务队列排序（sibling scheduling）** ：避免父子相邻导致预取风险

6. **内存规划与序列化（ABI）** ：

7. **初始化数值装配** ：

* 将原矩阵 A 的本地贡献填入各 node 的 frontal 初值
* 软件侧完成初始化量化（tile 32×32，subblock 16×16 exponent int8，q int32）

8. **验证与指标** ：

* residual、相对误差、sat_count/裁剪统计

9. **迭代求精**

* iterative refinement 框架（残差、停止条件、更新）
* 可作为“低精度 LU 的可靠性兜底”展示点



---

## 5. 后续工作

1. **软件侧闭环** ：symbolic → tasks/map_table → 初始化量化 → 生成二进制 → Python 参考验证 但目前缺少例子，等待物理院回应
2. **硬件调度闭环** ：dep_scoreboard + buffer_mgr + task_fetch + micro_scheduler 联动跑通，暂时不需要具体计算
3. **extend-add 正确性** ：scatter 写回父节点与依赖释放逻辑正确
4. **迭代求精** ：展示低精度可用性增强
5. **量化算法**：如果采用目前的量化算法，在具体进行节点矩阵计算和更新矩阵更新到父节点时的指数该如何变化。具体的计算如何正确部署到硬件上
