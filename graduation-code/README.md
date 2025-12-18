# Graduation Code Repository (System Implementation)

本目录包含 **"基于张量计算的大型稀疏矩阵求解系统"** 的核心软硬件实现代码。系统采用软硬协同架构，Host 端负责预处理与调度，硬件端负责高性能数值计算。

## 📂 目录结构索引

```text
├── hardware/              # 硬件加速器 RTL/HLS 源码
│   ├── ATU/               # 地址转换单元 (Address Translation Unit)
│   ├── GCU/               # 全局控制单元 (Global Control Unit / Scheduler)
│   ├── HPU/               # 冒险与主元单元 (Hazard/Pivot Unit)
│   └── Matrix_Engine/     # 矩阵计算引擎 (包含 SFU 与 Tensor Core)
├── software/              # Host 端驱动与预处理代码
│   └── dataStruct.cpp     # 核心数据结构定义与符号分析算法
└── README.md              # 本文档

```

---

## 🏗️ 硬件模块详解 (Hardware)

硬件架构设计遵循 **Right-Looking Blocked LU** 算法，针对 256 \times 256 超节点进行优化。

### 1. `GCU` (Global Control Unit) - 系统的“大脑”

负责宏观与微观的双层调度，确保计算流水线满载。

* **Macro-Scheduling**: 维护 **Ping-Pong Buffer** 状态机。负责任务预取（Prefetching），并执行**硬件依赖检查**，防止预取未完成写回的父节点数据。
* **Micro-Scheduling (Lookahead)**: 维护超节点内部的 **Dependency Scoreboard (记分牌)**。
* 当 Tensor Core 正在更新 Panel K 时，GCU 提前触发 HPU/SFU 处理 Panel K+1。


* **Phase Switching**: 控制系统在 *Kernel Factorization* (分解模式) 和 *Large Update* (更新模式) 之间切换。

### 2. `ATU` (Address Translation Unit) - 零拷贝存储路由

解决 LU 分解中主元交换（Pivoting）导致的大量数据搬运问题。

* **Local Indirection**: 维护一个局部映射表 `Logical_Row -> Physical_BRAM_Addr`。
* **Zero-Copy Swap**: 当 HPU 决定交换两行时，ATU 仅交换寄存器索引，无需物理搬运 BRAM 数据。

### 3. `HPU` (Hazard/Pivot Unit) - 数值稳定性卫士

* **Tournament Tree**: 采用流水线化的竞标赛树结构，快速在当前列中搜索全局最大值（Partial Pivoting）。
* **Parallel Search**: 在 GCU 的调度下，利用计算间隙并行执行搜索，掩盖比较延迟。

### 4. `Matrix_Engine` (Compute Complex) - 算力核心

这是系统的异构计算复合物，包含两类核心单元：

* **SFU (Special Function Unit)**: 处理标量瓶颈。负责 32 \times 32 主元块的除法、倒数计算及 Panel 的行列更新 (TRSM)。
* **Tensor Core**: 处理矩阵吞吐。采用 32 \times 32 脉动阵列 (Systolic Array)，负责：
* 超节点内部的子块更新。
* 外部大规模 Schur Complement 的生成 (GEMM)。


* *(注：Scatter Engine 逻辑通常集成于此模块的输出端口，负责将计算结果 Scatter-Add 到父节点)*

---

## 💻 软件核心逻辑 (Software)

### `software/dataStruct.cpp`

该文件是软硬件交互的契约核心，主要负责以下任务：

1. **任务描述符定义 (`Node_Task`)**:
定义了硬件执行所需的全部元数据。硬件通过 DMA 读取此结构体来启动一个节点的计算。
```cpp
struct Node_Task {
    uint32_t total_dim;        // 波前矩阵总维数 (N)
    uint32_t pivot_dim;        // 主元块大小 (K)
    uint64_t data_addr;        // 当前节点 DDR 基地址 (包含预填的 A)
    uint64_t parent_base_addr; // 父节点基地址 (用于 Scatter Add)
    uint64_t map_table_addr;   // 映射表指针 (用于隐式装配)
    uint64_t l_factor_addr;    // L 因子写回地址
    uint64_t u_factor_addr;    // U 因子写回地址
};

```


2. **符号分析与消解树构建**:
* 输入稀疏矩阵，进行 AMD 重排和符号分解。
* 构建 Elimination Tree，识别并合并 Supernodes (Max size 256)。


3. **Sibling Scheduling (兄弟节点优先调度)**:
* 在生成任务队列时，优化节点顺序。
* **策略**：在“子节点”和“父节点”任务之间插入无依赖的“兄弟节点”，利用计算时间掩盖子节点 Scatter 写回 DDR 的延迟，避免数据冒险。


4. **隐式装配预处理**:
* 负责在 DDR 中预分配内存，并将原始矩阵 A 的数值填入对应的节点区域，作为 Scatter Add 的“底板”。



---

## 🚀 系统数据流 (System Workflow)

1. **Pre-process**: Host 运行 `dataStruct.cpp`，生成 `Node_Task` 队列并填入 DDR。
2. **Fetch**: `GCU` 预取任务，DMA 将 Task N 加载至 Buffer A，Task N+1 加载至 Buffer B。
3. **Compute**:
* `GCU` 指挥 `HPU` 搜索主元，`ATU` 映射地址。
* `Matrix_Engine` (SFU + Tensor Core) 执行 Blocked LU 分解。


4. **Write-Back**:
* L/U 因子 Burst 写回 DDR。
* Update Matrix 通过 Scatter 逻辑累加到父节点地址。



---

*Last Updated: 2025*