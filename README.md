# 基于张量计算的大型稀疏矩阵求解系统

本项目是研究生毕业设计代码与文档仓库，目标是构建一个面向大型稀疏矩阵 LU 分解的
软硬件协同原型系统。整体思路参考 multifrontal 方法：软件侧完成矩阵前处理和硬件输入
生成，硬件侧完成节点级 front 装配、整数化 LU/TRSM/GEMM 和 child update 生成。

当前项目重点在硬件侧；软件侧的定位是 **硬件执行前的数据准备工具链**，不是完整的软件求解器。

## 仓库结构

```text
Graduation-Project/
  graduation-code/
    hardware/                 硬件 RTL/HLS 相关代码
    sim/                      硬件行为仿真与实验展示代码
    software/                 软件侧前处理与硬件输入生成代码
    software_legacy_before_refactor/
                              软件侧重构前代码快照
  graduation-project/
    src/                      设计文档、方案说明
    参考文献/                 论文与参考资料
    中期/                     中期答辩材料
```

## 系统执行流

```text
稀疏矩阵
  -> 软件侧符号分析
  -> 任务与映射生成
  -> A_local 预量化
  -> DDR 输入产物
  -> 硬件侧装配与整数 LU
  -> L/U 因子与 child update
```

## 软件侧当前实现

软件侧代码位于：

```text
graduation-code/software/
```

当前已完成的主要功能：

- 支持 `.mat`、MatrixMarket 文件输入，也支持随机 SPD 矩阵生成
- 支持 `amd`、`rcm`、`identity` 三种排序入口
- 构建 elimination tree
- 根据排序后矩阵图的闭邻接集合合并连续超节点
- 生成每个 node 的 `front_indices`
- 生成硬件任务描述符 `NodeTask`
- 生成 child update 到 parent front 的 `map_table`
- 根据 front 维度规划 DDR 区域
- 提取每个 node 的本地矩阵贡献 `A_local`
- 将每个 `A_local` 独立量化为 `S_format`
- 输出 `tasks.bin`、`map_table.bin`、`front_q.bin`、`front_e.bin` 和 `manifest.json`
- 对 manifest、ABI、文件大小、内存对齐、map table 等进行自动校验

软件侧运行示例：

```bash
python graduation-code/software/src/main.py \
  -mtx graduation-code/software/example/1024X1024JJ.mat \
  --out out
```

或：

```bash
cd graduation-code/software
python -m src.main --mtx example/1024X1024JJ.mat --out out
```

常用参数：

```text
--ordering amd|rcm|identity
--max-supernode-size 256
--effective-bits 27
--clip-percentile 100.0
```

更详细的软件侧说明见：

```text
graduation-code/software/README.md
```

## 软件侧量化职责

本项目的量化方案主要是硬件侧方案。软件侧只负责生成硬件 DDR 中的初始量化输入。

软件侧负责：

- 根据符号分析结果确定每个 node 的 `front_indices`
- 从原始矩阵中提取每个 node 的本地贡献 `A_local`
- 对每个 `A_local` 独立量化：

```text
A_local ~= q_local * 2^e_local
q_local: int32 mantissa
e_local: int16 source exponent
```

- 将 mantissa、exponent、任务描述符和 map table 写入输出产物

硬件侧负责：

- 读取软件准备的 `A_local` source 和 child update source
- 在父节点装配阶段完成 exponent 对齐和累加
- 根据装配结果选择最终 node-scale
- 执行整数 panel LU、TRSM、GEMM/Schur update
- 生成 child update payload 并写回 DDR

因此，软件侧不会生成整数 LU 数值结果，也不会生成 child update 数值 payload。

## 符号分析与超节点说明

当前 `amd` 入口是简化的最小度排序启发式算法，用显式对称图模拟消元和 fill-in。
它不是论文中的完整 quotient-graph AMD，也不等价于 SuiteSparse AMD。

当前超节点合并作为排序后的后处理执行：

```text
ordering -> elimination tree -> supernode grouping -> front_indices
```

超节点合并规则：

- 只合并当前排序中连续的列
- 若两列的闭邻接集合相同，则可合并：

```text
Adj(i) U {i} == Adj(j) U {j}
```

- 合并列数受 `--max-supernode-size` 限制，默认 256

这个规则参考 AMD 论文中 indistinguishable variables / supervariables 的描述，但没有实现
完整 AMD 中的动态 quotient graph supervariable 检测。这样做的原因是当前软件侧更关注为硬件
生成稳定、紧凑、易解码的 node 任务。

## 输出产物

软件侧输出目录中包含：

| 文件 | 说明 |
|---|---|
| `tasks.bin` | 硬件节点任务描述符 |
| `map_table.bin` | child update 到 parent front 的映射表 |
| `front_q.bin` | 每个 node 的 `A_local` 量化 mantissa |
| `front_e.bin` | 每个 node 的 source exponent |
| `manifest.json` | 输出元数据、内存布局、量化统计和校验信息 |

`manifest.json` 中会记录软件和硬件的量化职责边界，以及哪些区域由硬件运行时写入。

## 硬件侧目标

硬件侧围绕 node-level multifrontal 执行展开，核心模块包括：

- `GCU`：任务获取、依赖检查、buffer 管理和节点内调度
- `ATU`：逻辑行到物理存储位置的地址映射，支持低成本行交换
- `HPU`：主元选择，当前设计采用层级/锦标赛式比较结构
- `SFU`：标量和控制路径计算，包括除法、缩放、TRSM 相关操作
- `TPU/GEMM core`：高吞吐整数矩阵乘加，用于 Schur update
- `front_loader / assembly`：读取本地贡献和 child update，完成多 source 在线装配
- `update_writer / factor_writer`：写回 child update 和 L/U 因子

硬件行为仿真和实验展示代码位于：

```text
graduation-code/sim/
```

## 测试

软件侧测试：

```bash
cd graduation-code/software
python -m pytest
```

当前测试覆盖：

- `NodeTask` ABI 编解码
- map table 编解码
- 内存规划对齐和不重叠
- 排序输出合法性
- 超节点多列合并与上限控制
- `A_local` 量化与硬件装配参考模型
- 端到端 pipeline 与 manifest 一致性

## 当前状态

软件侧已经形成从输入矩阵到硬件输入产物的完整链路。后续工作的重点应放在：

- 硬件侧 front assembly 与 node-scale 选择
- 整数 LU/TRSM/GEMM 数据通路
- child update 写回和父节点装配联动
- 软硬件联合仿真与实验结果展示

最后更新：2026-06-17
