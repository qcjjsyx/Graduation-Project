# 软件侧算法说明

状态：当前主线算法文档。记录本轮软件侧实际使用的算法——**原理、推导、边界与代码锚点**，
不逐行复述实现。

## 范围

本文档只覆盖 host（Python）软件链的算法：

```text
A、b
  → 行均衡 D_r（01）
  → 最大乘积匹配 P_match（02）
  → 符号分析：包络/排序/填充/森林/supernode/front/map（03）
  → 变换语义与置换约定（04）
  → 多波前装配与树形求解（05）
  → 主元策略与阈值（06）
  → 精度指标与参考校验（07）
  → 命令流编译与内存布局（08）
```

设备执行器（T04/T05）、HPU/ATU 动态硬件、INT32/BFP GEMM 属于设备侧，
不在本目录展开；相应契约见 [Command_Schema_v2.md](../interface/Command_Schema_v2.md)。

## 索引

| 文档 | 内容 | 主要代码 |
| --- | --- | --- |
| [01 行均衡](01-row-equilibration.md) | 2 的幂行缩放、无损性与不变量 | `src/equilibration.py` |
| [02 最大乘积匹配](02-max-product-matching.md) | 完美匹配、log 等价、求解器与方向推导 | `src/matching.py` |
| [03 符号分析](03-symbolic-analysis.md) | 包络、最小度排序、精确填充、消去树、supernode、front、map | `src/symbolic/*` |
| [04 变换语义](04-transform-semantics.md) | gather 约定、四个变换公式、解恢复 | `src/pipeline.py` |
| [05 多波前装配与求解](05-multifrontal-assembly-and-solve.md) | Schur 补、LCA 存储规则、树形分解与代入 | `src/verify/front_reference.py`、`systemc/reference/` |
| [06 主元策略与阈值](06-pivot-policy-and-threshold.md) | STATIC / DYNAMIC_PANEL、相对阈值、失败分类、P-vector | `src/config.py`、`src/verify/front_reference.py` |
| [07 精度指标与参考](07-accuracy-metrics-and-reference.md) | 残差、分量后向误差、golden 解来源、交叉验证 | `src/verify/metrics.py` |
| [08 命令流编译与内存布局](08-command-compilation-and-memory-layout.md) | token 依赖 DAG、确定性调度、两阶段布局 | `src/scheduler/*`、`src/memory/planner.py` |

## 统一记号

- 矩阵 `A ∈ R^{n×n}`，右端 `b`；`nnz` 表示非零个数。
- 置换一律用 **gather 约定**：`y[i] = x[p[i]]`；矩阵 `B = P A` 表示 `B[i,:] = A[p[i],:]`。
- `D_r`：对角行缩放；`P_match`：匹配行置换；`Q`：结构排序。
- `front`：多波前法中的稠密前沿矩阵；`pivot block`：front 内允许消元的前若干行/列。
- 设备数值格式 FP32；索引与 P-vector 为 INT32；所有参考数据 FP64 且只在 host。

## 与其它文档的关系

- 契约与失败语义：[`docs/interface/Command_Schema_v2.md`](../interface/Command_Schema_v2.md)
- 模块职责与边界：[`docs/architecture/系统架构与模块职责.md`](../architecture/系统架构与模块职责.md)
- 验证口径与证据：[`docs/verification/验证计划与结果口径.md`](../verification/验证计划与结果口径.md)
