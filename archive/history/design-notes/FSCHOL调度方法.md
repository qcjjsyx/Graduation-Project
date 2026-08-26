# FSCHOL 主机端调度算法

> 来源: Bank-Tavakoli et al., "FSCHOL: An OpenCL-based HPC Framework for Accelerating Sparse Cholesky Factorization on FPGAs", SBAC-PAD 2021.

## 问题背景

稀疏 Cholesky 分解的超节点多波前（Supernodal Multifrontal）方法将整体分解组织为消除树上一系列超节点的 **update**（扩展加，用于聚合子节点贡献）和 **factorize**（局部稠密 Cholesky）。子节点 `factorize` 产生的 update matrix 必须传递给父节点用于 `update`。

调度目标：**安排超节点的计算顺序，消除 update matrix 的片外内存存取，并最小化片上缓冲需求**。

## 数据结构

| 符号 | 含义 |
|---|---|
| `T` | 无子节点的超节点列表（叶子节点） |
| `P[S]` | 超节点 `S` 的父节点 |
| `NC[S]` | 超节点 `S` 的子节点总数 |
| `nc[S]` | `S` 已被多少个子节点完成 update 的计数器（初始化为 0） |
| `Q1`, `Q2` | PE1、PE2 各自维护的单元素任务队列 |

## 调度算法伪代码

```
初始化: nc 全零, Q1 和 Q2 置空

while 根节点尚未 factorize:
    if Q1 为空:
        S = T[0]                    // 取一个叶子超节点
        指派 update(S, -) 到 PE1    // 无子节点，无 update 依赖
        从 T 移除 S
        p = P[S]
        push(p, Q1)                  // 父节点入队，等待后续更新

    else:
        S = pop(Q1)                  // S 已准备好被更新
        标记 S 正在被更新
        标记 V_U 应从 Q_U 读取      // 指示硬件读 update matrix

        if nc[S] == 0:
            if S 在当前 PE 之前被更新过:
                V_F 从 Q_F,PE 读取  // 更新中间值在 PE 内部 FIFO
            else:
                V_F 从 Q_F 读取     // 更新中间值在跨 PE FIFO

        通知另一 PE: 应把 V_F 写入 Q_F

    nc[S]++                          // S 又完成了一个子节点的 update

    if nc[S] == NC[S]:               // S 已被所有子节点更新完毕
        factorize(S)                 // 在此 PE 上执行局部分解
        push(S, Q1)                  // 推回队列，让父节点可以被更新

// PE2 执行对称逻辑
```

## 关键设计：父子流水线

**`update(S, C)` 紧跟 `factorize(C)` 在同一 PE 上执行。**

子节点 `C` factorize 完成后，其 update matrix 不写回片外内存，而是：
1. 经 PE 内部 FIFO（`Q_U`）直接流向同一 PE 的 update 模块
2. 父节点 `S` 的 `update(S, C)` 在该 PE 上立即消费该数据

这样消除了 update matrix 的片外内存存取，**整个消除树遍历过程中中间结果只在片上流动**。

## 调度示例

对应论文 Figure 5 的超节点消除树：

```
    时间轴（自底向上）
    ┌─────────────────────────────────────────┐
    │                                         │
    │  update(9,8)          factorize(9)       │ ← 根节点
    │  update(8,3)  factorize(8)  update(9,6) send(F9)
    │  update(3,1)  factorize(3)  update(6,5) factorize(6)
    │  update(1,-)  factorize(1)  update(5,2) factorize(5)
    │  update(9,7)  store(F9)    update(2,-)  factorize(2)
    │  update(7,0)  factorize(7) update(5,4)  store(F5)
    │  update(0,-)  factorize(0) update(4,-)  factorize(4)
    │                                         │
    │       PE1                         PE2    │
    └─────────────────────────────────────────┘
```

本质是**消除树的自底向上深度优先遍历**，兄弟子树可以分配到不同 PE 并行。

## 与我方工作的关联

| FSCHOL | 我方 Block LU |
|---|---|
| Cholesky 分解 (SPD矩阵) | LU 分解 (通用矩阵) |
| 消除树依赖 | Tile 级 DAG 依赖 |
| 超节点 update/factorize | Panel-Fact / TRSM / GEMM |
| 父子流水消除 update matrix 片外访存 | ATU 地址映射消除物理行搬移 |
| 双 PE + FIFO 片内通信 | 脉动阵列 + SRAM 模块 |

核心共性：**通过任务调度编排 + 硬件架构配合，最大程度减少数据搬移，实现片内数据复用**。
