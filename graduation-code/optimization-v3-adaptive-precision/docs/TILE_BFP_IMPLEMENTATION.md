# 16×16 Tile BFP 实现、问题与实验报告

> 本文保留 v2 Tile BFP 基线的实现与实验记录。v3 的 QF26、F20→F26 重算实验、
> 新报告字段和最新真实矩阵结论见
> [v3 定点精度优化报告](ADAPTIVE_PRECISION_OPTIMIZATION.md)。

## 1. 为什么从标量 BFP 继续优化

v1 的每个 local front、assembled front、U 和 update 各只有一个公共 exponent。该设计
控制简单，但一个大元素会决定整个矩阵块的量化步长。远小于最大值的元素可能在输入
量化、child-to-parent 装配或 Schur 写回时变成零。

行列 2 的幂均衡已经使 576JJ 的原方程 residual 从基线 `2.346e-2` 降到
`4.593e-3`，但仍未达到 `1e-3`。继续扫描全局 `q_use_bits`、`frac_bits` 和 guard bits
没有解决这一问题，说明需要改变缩放粒度，而不是只调同一个全局 exponent。

16×16 tile 与现有 Panel/TRSM/GEMM 调度 tile 一致，指数 SRAM、地址生成和 RTL
控制的代价也比逐行、逐列或逐元素 exponent 更可控，因此选择它作为下一版。

## 2. 独立实现范围

所有代码位于：

```text
graduation-code/optimization-v2-tile-bfp/
```

基线目录和 `optimization-v1` 均未修改。v2 保留 FP64 与 fixed 共用控制流、任务依赖、
DDR、buffer、HPU/ATU、七类微操作、writer 和 solve controller，只替换或扩展数值
payload 与 fixed 数值路径。

## 3. 软件制品生成

`QuantConfig` 增加 `bfp_tile_size`，允许：

- `0`：v1 标量 exponent 回归模式；
- `16`：本版本 tile BFP。

软件对每个 local `A_local` 的 tile 独立计算：

```text
q_limit = 2^effective_bits - 1
e_tile = ceil(log2(maxabs(tile) / q_limit))
q_ij = round(A_ij / 2^e_tile)
```

随后写出完整 row-major int32 mantissa 和 row-major int16 tile exponent。RHS 仍保持一个
全局 exponent，避免在本阶段同时改变向量 ABI。

内存规划器根据矩阵形状计算：

- local front exponent 数量；
- update exponent 数量；
- U exponent 数量；
- 可容纳 U exponent 表的 `node_meta` 大小。

manifest 记录 tile size、顺序、每个 local source 的 exponent 数量。Python 验证器会
重新计算全部数量，拒绝尺寸不一致的制品。

## 4. Tile-aware QAU

父 front 的一个目标 tile 可能同时接收 local 数据和多个 child update。实现采用两遍
装配：

1. 遍历所有落入该目标 tile 的非零源，选择最大的源 exponent 作为
   `e_assembly[tile]`；
2. 每个源按
   `round_shift_signed(q_source, e_assembly-e_source)` 对齐后进入 int64 accumulator；
3. 对每个目标 tile 独立执行 maxabs、scale 选择和 int32 饱和再量化。

这样不会让 front 中一个 tile 的大元素直接决定所有其他 tile 的分辨率。所有
align drop、overflow、saturation 仍累计到 node 统计。

child update 的 tile 坐标在 map 前确定，目标 exponent 的 tile 坐标在 map 后确定，
避免把父坐标误用于 child exponent 查找。

## 5. Tile BFP 定点 LU

### 5.1 工作区

assembled mantissa 左移 `workspace_guard_bits` 后进入 int64 工作区。每个物理
16×16 tile 保存自己的工作 exponent：

```text
e_workspace[tile] = e_assembled[tile] - guard_bits
```

L multiplier 单独保存在 QF 缓冲中，不覆盖 tile 工作区。这一点很重要：如果 tile
后续重新归一化，已经生成的 L 不能被一起移位。

### 5.2 主元

主元仍只在 `k..pivot_dim-1` 搜索。比较值为：

```text
abs(mantissa) × 2^tile_exponent
```

选定结果再归一化成 int32 候选流送入现有逐周期 HPU，ATU 继续维护 P-vector。相同
绝对值仍选择最先出现的逻辑行。

### 5.3 L 除法

不同 tile 的 value 和 pivot 具有不同 exponent：

```text
L_q = round(
  value_m / pivot_m ×
  2^(F + value_e - pivot_e)
)
```

实现使用 `__int128` 中间分子/分母和有符号最近舍入；范围不足、零除或 int32 L
越界会触发显式 precision rescue。

### 5.4 Schur 更新

乘积表示为：

```text
product_m = L_q × U_m
product_e = U_e - F
```

更新前优先把 product mantissa 左移到目标 tile exponent，只在 int64 工作区放不下时
才提高目标 tile exponent 并右移整个 tile。这比看到 `product_e>destination_e` 就立即
降低整个目标 tile 精度更保守。

### 5.5 U/update 写回

由于逻辑行经过 P-vector 映射，U 输出 tile 与物理工作 tile 不一定重合。写回阶段会
重新按逻辑 `pivot_dim×total_dim` 的 16×16 tile 选择 exponent；update 也在去掉 pivot
前缀后按自己的坐标重新分 tile。

U exponent 写入 `node_meta`，update exponent 写入 `update_e`。FP64 rescue 生成的
U/update 也执行同样的 tile 量化，保证父装配看到的格式一致。

## 6. 为什么增加局部因子检查

最初实验只使用主元阈值和 overflow 触发 rescue。576JJ 上出现：

- 局部 fixed factorization error 约 `1e-6`；
- 但原方程初始 residual 被病态缩放放大到约 `1.99e2`；
- 迭代求精无法稳定恢复。

这说明“主元非零且 accumulator 未溢出”不足以证明 tile 定点因子可以安全传播。写回
前现在计算：

```text
eta_node = ||P A_assembled - L U - S||_F
           / ||P A_assembled||_F
```

默认门槛为 `2e-7`。超过门槛的 node 进入已有 FP64 precision rescue，rescue 只消费
定点 QAU 已装配的 front，不读取黄金 local FP64 front。

选择该门槛的理由是：`5e-7` 虽能让 576 达到约 `8.03e-4`，但 256 仍约
`2.38e-3`；`2e-7` 使三个真实矩阵均达到 `1e-3` residual 目标。代价是 rescue 数量
很高，因此它是当前系统模型的稳定性护栏，也是后续 RTL 设计必须继续优化的证据。

## 7. 求解路径

普通 tile node 的 L 使用 QF，U 每次访问根据 `(row,col)` 查找 tile exponent。rescue
node 使用同一 assembled front 产生的高精度因子。前代、回代、ordering 恢复、列缩放
恢复和原方程迭代求精保持不变。

当前 tile 模式使用 exponent-aware 的 double 向量控制路径消费定点因子，因为一个
node 的 U 已不再只有一个 exponent，v1 的“每 node 一个 solution exponent”整数
V-format 不能直接表达所有乘积。这是明确边界：矩阵存储、装配、分解和写回已经是
tile BFP；tile-aware 纯整数向量 accumulator 仍是后续可优化项。

## 8. 真实矩阵结果

基线为 v1 的 B1 `pow2-row-column`，配置为 26-bit 输出、8 guard bits、标量 exponent。
v2 是完整组合优化：16×16 tile、30-bit 输出、20 guard bits 和局部因子检查。因此表格
表示“可工作的 v2 系统方案”对比，不应被解释为只隔离 tile size 的单因素实验。

| 矩阵 | 方案 | residual | 初始 residual | 解误差 | 后向误差 | Rescue | IR | 周期 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | v1 B1 | `6.408e-4` | `1.309e-2` | `1.929` | `2.819e-7` | 2 | 1 | 175098 |
| 256 | v2 tile | `5.964e-4` | `5.964e-4` | `1.514` | `3.774e-7` | 68 | 0 | 178625 |
| 576 | v1 B1 | `4.593e-3` | `1.766e-2` | `0.958` | `2.599e-6` | 13 | 2 | 1700662 |
| 576 | v2 tile | `3.252e-4` | `1.229e-3` | `6.934` | `9.370e-8` | 150 | 1 | 1693321 |
| 1024 | v1 B1 | `9.730e-4` | `3.071e1` | `1.436e-3` | `1.418e-4` | 1 | 39 | 7378043 |
| 1024 | v2 tile | `1.012e-4` | `2.182` | `3.000e-8` | `4.622e-5` | 122 | 2 | 8722424 |

结论：

- 三个矩阵的 residual 都达到 `1e-3`；
- 576 residual 比 v1 B1 下降约 92.9%，但解误差从 `0.958` 增至 `6.934`；
- 576 的 residual、后向误差和前向解误差并不等价，必须在论文中说明其病态性；
- 1024 的求精轮数从 39 降至 2，解误差显著改善，但总周期增加约 18.2%；
- 256/576 的绝大多数 node 被严格门槛 rescue，说明当前 tile 整数核还不能直接作为
  “无需高精度后端”的 RTL 结论。

原始摘要字段整理在 `results/real_matrix_comparison.csv`。

## 9. 报告扩展

`summary.json.stability` 新增：

- `assembled_tile_count`；
- `factor_tile_count`；
- `max_tile_exponent_span`。

`nodes.csv` 新增 assembled/U/update 的 tile 数量和 exponent 最小/最大值。真实算例
最大 tile exponent span 分别为 42、39 和 74，可直接用于估计 exponent SRAM 位宽和
对齐移位器需求。

## 10. 验证

- Python：33 项测试通过；
- CTest：ATU/HPU、数值单元、软件到 SystemC 端到端 3/3 通过；
- ASan/UBSan：3/3 通过；
- 新增 17×17 跨 tile 动态范围测试，确认小尺度 tile 对角不会因大尺度 tile 丢失；
- manifest/config tile size 不匹配时启动失败；
- 256/576/1024 相同配置和 seed 均正常完成，无死锁。

## 11. 对 RTL 的直接建议

1. QAU exponent SRAM 按 front tile 网格寻址，并支持 child source tile 到 parent
   destination tile 的双地址查找。
2. MAC 路径不要只比较 exponent 决定整体右移；应优先利用 int64 mantissa 余量左移
   product，减少无谓 tile renormalization。
3. L multiplier 必须与可重标度矩阵工作区分离。
4. writer 为 U 和 update 分别维护 exponent 网格；不能复用一个 node exponent。
5. 增加局部因子误差或可实现的代理风险指标。当前 `2e-7` 检查依赖高精度重构，
   适合 SystemC 研究模型，不适合原样实现成 RTL。
6. 在大规模 RTL 开发前，优先研究降低 rescue 的方案：tile 内额外 guard、分阶段
   accumulation、condition-aware threshold、选择性更高精度 Panel，或 delayed pivot。
