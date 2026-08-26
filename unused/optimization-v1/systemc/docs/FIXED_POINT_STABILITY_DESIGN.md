# 定点多前沿分解稳定性：问题、改进与当前实现

最后更新：2026-07-25

## 1. 文档目的

本文记录当前 SystemC 多前沿矩阵求解系统为解决定点分解稳定性所完成的设计与实现，
包括：

- 原始定点方案及其在 576JJ、1024JJ 上暴露的问题；
- 已验证无效或不足的直接参数调整；
- 软件预处理、节点内分解、精度救援、向量求解和迭代求精的完整改进；
- 每项改进的原因、硬件含义和报告接口；
- 256JJ、576JJ、1024JJ 的可复现实验结果；
- 当前仍未解决的边界和下一步 RTL/算法研究方向。

本文描述的是当前仓库代码，而不是仅停留在建议层面的方案。对应实现主要位于：

- 软件行均衡与产物生成：`software/src/equilibration.py`、`software/src/pipeline.py`；
- QAU 装配与系统存储：`systemc/include/system_memory.hpp`；
- 定点/FP64 分解核：`systemc/include/numeric_kernels.hpp`；
- 救援调度与写回：`systemc/include/full_system.hpp`；
- 定点树形求解与求精：`systemc/include/solve_controller.hpp`；
- 配置和报告：`systemc/config/default.json`、`systemc/include/report.hpp`。

## 2. 原设计

### 2.1 数据格式

原设计采用三类主要定点格式：

- S/M-format：int32 mantissa 加共享的 int16 exponent，
  `real_value = mantissa × 2^exponent`；
- QF-format：L 乘子使用 int32 定点小数，默认 `F=20`，
  `real_value = mantissa × 2^-20`；
- V-format：向量 mantissa 加 node exponent。

软件把每个节点唯一的 `A_local` 量化为 S-format。QAU 将 local contribution 与所有
child update 对齐到共同的 assembly exponent，在 int64 中累加后重新量化为 int32。
节点计算随后执行主元选择、Panel LU、TRSM 和 Schur update，并把 L、U、P-vector 和
update 写回 DDR。父节点只有在子节点 update 写回完成后才能被释放。

### 2.2 原始定点节点内流程

原实现把装配后的 int32 front 直接作为节点内消元工作区。每一列：

1. HPU 在 `k..pivot_dim-1` 中选最大绝对值；
2. ATU 更新逻辑行到物理行的 P-vector；
3. 计算 `L(i,k)=round(F(i,k)/F(k,k) × 2^F)`；
4. 执行 `F(i,j)-=L(i,k)×F(k,j)`；
5. 将结果继续限制在较窄的定点表示中。

该设计在小型稳定矩阵上能够逐元素复现 Python 定点模型，但它隐含了三个过强假设：

- 一个 node exponent 足以同时表示 U、Schur update 和所有中间结果；
- 每列消元后的微小非零主元仍能保留在 int32 网格上；
- 输入量级差异不会在消元前把有效位耗尽。

真实高条件数矩阵违反了这些假设。

## 3. 观察到的问题

### 3.1 576JJ：极小主元与病态性

在 AMD ordering 和当前 supernode 划分下，576JJ 有 157 个节点。FP64 在根节点
`node 156` 的最后一列 `column 155` 检测到：

- front infinity norm 约 `1.1380817e7`；
- 候选主元绝对值约 `4.1699e-7`；
- `|pivot| / ||front||∞ ≈ 3.664e-14`；
- 默认 FP64 阈值为 `pivot_rel_tol × ||front||∞`，其中
  `pivot_rel_tol=1e-12`。

因此该主元低于阈值，FP64 基准有意报告 numeric failure。旧定点流程虽然有时能继续，
但最终原方程 relative residual 约为 `8.58`，解相对误差超过 `1e4`，并出现大量向量
drop。这不是控制流死锁，而是因子和向量表示已失去足够精度。

### 3.2 1024JJ：消元中产生量化零主元

1024JJ 在当前规则下有 274 个节点，最大 pivot 为 256、最大 front 为 272。旧定点流程
在 `node 269` 的 `column 15` 失败。该列在节点刚装配时并非全零；经过前 15 列整数消元
后，剩余候选在定点网格上全部变成 0，HPU 因而得到 zero pivot。相同输入的 FP64 分解
能够完成。

这说明问题不是原始稀疏结构缺失，也不是 HPU 候选范围错误，而是节点内连续舍入和共享
尺度造成的小量丢失。

### 3.3 小 residual 与解准确度不是同一指标

256JJ 的新定点结果能够达到 `5.377e-4` relative residual，但解相对误差仍约
`2.904`。FP64 residual 约 `1.349e-12` 时，解误差也仍为 `3.825e-8`，明显大于 residual。

这是病态线性系统的典型表现：很小的后向误差不保证很小的前向误差。因此系统现在同时
报告：

- 原方程 relative residual；
- 均衡方程 relative residual；
- componentwise backward error；
- 存在 `x_true` 时的 relative solution error。

毕业设计中不能只用 residual 宣称“解完全准确”。

## 4. 验证过但不足的简单调整

在形成当前方案前，先验证了若干低成本调整：

### 4.1 单纯提高 `q_use_bits`

把计算有效位从 27 提高到 30 并未解决未均衡的 1024JJ。原因是 int32 总位宽固定：
有效位越接近 31 位，留给符号、增长和加减法的余量越少。行均衡后直接使用 30 位计算还
观察到 17 次溢出，residual 约 `29.9`。更多输入有效位不等于更稳定的中间计算。

因此当前把“源保存精度”和“计算/写回余量”分开：

- 软件 DDR 源默认保存 30 个有效位；
- SystemC QAU/因子输出默认使用 26 位。

### 4.2 单纯修改 QF 小数位

分别测试 `F=16/20/24` 不能消除未均衡 1024JJ 的零主元。QF 位数主要控制 L 乘子的
表示误差；当 U/Schur 工作区本身已经把小主元舍入为零时，只调整 L 格式不能恢复信息。

### 4.3 更换 ordering 或强制拆分 supernode

RCM、identity ordering 以及更小的 supernode 能改变失败位置，却不能系统性消除动态
范围和 cancellation 问题。ordering 对稳定性和 fill 均有影响，但它不能代替数值缩放
和宽工作区。

### 4.4 让救援路径读取黄金 FP64 front

这种方案能给出漂亮结果，但会绕过 DDR 定点源、QAU 装配误差甚至内存损坏，无法用于验证
硬件原型。当前实现明确禁止这样做：precision rescue 的输入只能是
`assembled_q × 2^assembled_exp`。

## 5. 改进一：2 的整数次幂行均衡

### 5.1 变换

软件在 symbolic analysis 和量化前构造：

```text
D_r A x = D_r b
D_r[i,i] = 2^s_i
s_i = clamp(-round(log2(max_j |A[i,j]|)), -60, 60)
```

零行使用 `s_i=0`。变换不改变稀疏结构，也不缩放未知量，所以最终 `x` 不需要反变换。

### 5.2 为什么使用 2 的整数次幂

- 对二进制 BFP，乘以 `2^s` 只需修改 exponent；
- 不需要在软件和硬件之间引入额外的实数缩放误差；
- etree、fill、supernode 和 map table 完全不变；
- RTL QAU 可以复用已有 exponent 对齐移位器。

### 5.3 产物与验证边界

软件新增：

- `row_scale_e.bin`：原始行编号顺序的 int16 exponent；
- `original_matrix_f64.bin`：未均衡原始矩阵；
- `original_rhs_f64.bin`：未均衡 RHS。

`reference_front_f64.bin` 和 `rhs_f64.bin` 表示均衡、排序后的黄金数据。SystemC 的
最终 residual 始终在 `original_matrix_f64.bin` 和 `original_rhs_f64.bin` 上计算。
这样可以防止缩放后的指标掩盖原问题。

## 6. 改进二：源精度与计算余量分离

软件 `effective_bits=30` 尽量保留 DDR 输入信息。SystemC 默认
`q_use_bits=26`，令：

```text
q_limit = 2^26 - 1
```

QAU 在装配后根据 maxabs 选择 node-scale，把结果重新量化到该范围。这留下约 5 个
int32 顶部余量，用于符号和局部增长。该取舍的理由是：

- 源量化只发生一次，应该尽量保真；
- 节点计算会反复执行乘加和相减，需要增长余量；
- 两个参数分离后可以通过 sweep 定量研究精度与溢出的折中。

## 7. 改进三：带 guard bits 的宽节点工作区

### 7.1 表示

装配结果进入节点内分解时，不再直接在 int32 网格原位更新。对 guard bits `g=8`：

```text
workspace_q = assembled_q << g
workspace_e = assembled_e - g
real_value  = workspace_q × 2^workspace_e
```

左移和 exponent 抵消，所以实数值不变，但节点内多出 8 个低位用于保存 cancellation 后
的小量。工作区存放在可配置 accumulator 中，默认 int64。

### 7.2 消元

每个 L 乘子计算为：

```text
L_q = round_signed_nearest(workspace(i,k) × 2^F / pivot)
```

乘法和除法的中间值使用 `__int128`，避免“分子超出 int64”被误判为数值失败。更新为：

```text
delta = round_signed_nearest(L_q × workspace(k,j) / 2^F)
workspace(i,j) = workspace(i,j) - delta
```

每次更新检查配置的 accumulator 位宽。若发生不可恢复的工作区溢出，不做静默饱和并继续，
而是触发 precision rescue。

### 7.3 HPU 仍保持 32-bit 接口

HPU RTL 计划仍使用 int32 候选。SystemC 对同一列的宽候选执行共同的右移归一化，使所有
候选适配 int32，然后送入逐周期 HPU。共同移位保持绝对值顺序；相同绝对值仍选择最先
出现的逻辑行。原始宽值用于零判断和 pivot ratio 统计。

因此硬件实现可以保留 32-bit HPU 比较树，而宽 accumulator 和列归一化属于矩阵计算核/
接口适配逻辑。

## 8. 改进四：U 与 child update 独立 exponent

一个 front 分解后，U 和 Schur update 的动态范围可能相差很多。原设计让二者共享
node exponent，会导致较小块损失有效位。

当前实现分别对两块做 maxabs/requantize：

```text
U_real      = U_q      × 2^u_exponent
Update_real = Update_q × 2^update_exponent
```

- `u_exponent` 写入 `node_meta[2:4]`；
- `update_exponent` 写入原有 `update_e`，并镜像到 `node_meta[4:6]`；
- 父节点 QAU 只读取 child 的 `update_exponent`。

NodeTask 仍保持 ABI v2 的 128 字节布局，不需要增加描述符字段。详细布局见
[ABI v2](ABI_v2.md)。

## 9. 改进五：显式 precision rescue

### 9.1 触发条件

定点节点内分解遇到以下情况会抛出 `PrecisionRescueRequired`：

- 当前 pivot 候选在宽工作区中全部为零；
- `|pivot| / initial_front_max <= fixed_pivot_rel_tol`；
- L 乘子不能由 int32 QF 表示；
- guard-bit 工作区发生 accumulator 溢出；
- U 在最终重新量化后对角变为零。

地址错误、非有限值或其它不可恢复错误仍作为真正 numeric/address failure，不会被救援
掩盖。

### 9.2 救援输入

救援 front 按下式构造：

```text
F_rescue = assembled_q × 2^assembled_exp
```

也就是说，它只能恢复“当前定点 DDR 数据已经表达出的实数”，不能恢复源量化或装配阶段
已经丢失的信息。它不读取 `reference_front_f64.bin`。

### 9.3 救援计算与存储

默认 `fixed_rescue_mode=fp64`。救援后端：

1. 在当前 node 的 `pivot_dim` 范围内执行 FP64 partial pivoting；
2. 使用独立的 `rescue_pivot_rel_tol=1e-16`；
3. 将 L 量化回 QF，U/update 分别量化回 M-format；
4. 保存一份该救援节点的 FP64 factor，用于 SystemC 的 hybrid solve；
5. 记录 `PRECISION_RESCUE` 操作和配置化启动/吞吐周期；
6. 在 `node_meta[1]` 写 rescue 标志。

保存 FP64 factor 是 SystemC 研究模型行为，不是当前 DDR/RTL factor 格式。它使求解阶段
能判断“如果少数危险节点交给高精度单元，系统能否稳定”，同时定点写回仍可检查量化风险。

### 9.4 为什么不用静默饱和

静默饱和会让任务表面完成，却把数值错误推迟到最终 residual，难以定位。显式救援/失败
状态使报告能够回答：

- 哪个节点首先失去定点安全性；
- 触发原因是小主元、L 越界还是 accumulator；
- 高精度后端的使用次数与周期代价；
- 即使完成，最终是否达到准确度目标。

## 10. 改进六：宽向量求解与动态 V-format

树形前代和回代仍遵循多前沿依赖顺序：

- 前代叶到根：对 pivot RHS 应用 P-vector，解 `LPP*yP=bP`，
  更新 `bU-=LUP*yP`；
- 回代根到叶逆序：计算 `rP-=UPU*xU`，解 `UPP*xP=rP`；
- 最后恢复 ordering permutation。

定点模式的主要变化为：

- 除法使用 `__int128` 分子和有符号最近舍入；
- 中间 scratch 使用 int64；
- `vector_use_bits=55` 控制向量安全范围；
- 节点解接近范围上限时自动调整 solution exponent；
- 不同 node 的 `U*x` 在 int64 中对齐；
- 对 rescue 节点使用保存的 FP64 factor，其余节点使用反量化的定点 factor。

shift、drop、overflow、solution renormalization 和 divide-by-zero 风险均进入
`summary.json`。

## 11. 改进七：混合精度迭代求精

### 11.1 流程

初始定点/混合精度解为 `x_k`。每轮在原始坐标 FP64 中计算：

```text
r_k = b - A x_k
```

把 residual 按行均衡并恢复到排序坐标：

```text
r_scaled_perm[p] = 2^row_scale_e[original(p)] × r_k[original(p)]
```

然后使用已经得到的低精度/hybrid 因子求解修正：

```text
M delta_x ≈ D_r r_k
```

最终解在 FP64 中累加，不要求把所有修正压回单一 V-format。

### 11.2 下降保护

病态问题上，低精度修正不一定改善 residual。当前实现先计算
`z=A*delta_x`，再选择：

```text
alpha = clamp((r^T z)/(z^T z), 0, 1)
x_candidate = x_k + alpha*delta_x
```

出现以下情况会停止且保留上一个解：

- `r^T z <= 0`，修正不是下降方向；
- residual 非有限；
- 新 residual 未达到 `ir_min_improvement` 指定的最小改善；
- 修正求解失败；
- 达到最大轮数。

默认目标为原方程 relative residual `1e-3`，最多 50 轮，最小改善率 `1e-3`。每轮
residual MAC 和 correction solve 都计入周期与 `operations.csv`。

### 11.3 输出语义

若至少接受一轮修正：

- `solution.csv` 输出 FP64 累加后的最终解；
- `final_memory_image.bin` 的 `solution_q` 区域写 FP64；
- `solution_e` 清零；
- `summary.json.solve.fixed.refined_solution=true`。

该写回是 SystemC 研究扩展，不是当前 RTL 协议。若没有接受修正，仍写 int64 V-format
mantissa 和每节点 exponent。

## 12. 正确性与稳定性指标

### 12.1 原方程 relative residual

```text
rho = ||b-Ax||_2 / max(||b||_2, 1e-300)
```

这是 fixed 模式 `accuracy_target_met` 的判断指标。

### 12.2 均衡方程 residual

同时报告由定点 local contributions 重构的均衡矩阵和定点 RHS 上的 residual，用于区分：

- 因子/求解本身对均衡系统的误差；
- 从均衡坐标回到原方程后被行尺度放大的误差。

### 12.3 分量后向误差

```text
eta = max_i |r_i| /
      (|b_i| + sum_j |A_ij| |x_j|)
```

该指标比单一 L2 residual 更能观察某些行是否异常。

### 12.4 前向解误差

当软件根据固定 seed 生成 `x_true` 时：

```text
epsilon_x = ||x-x_true||_2 / ||x_true||_2
```

用户提供 RHS 且没有真实解时，不应凭 residual 推断前向误差。

## 13. 配置接口

默认稳定性相关配置为：

```json
{
  "q_use_bits": 26,
  "frac_bits": 20,
  "accumulator_bits": 64,
  "workspace_guard_bits": 8,
  "vector_use_bits": 55,
  "adaptive_factor_scaling": true,
  "fixed_pivot_rel_tol": 1e-5,
  "fixed_rescue_mode": "fp64",
  "rescue_pivot_rel_tol": 1e-16,
  "precision_rescue_startup": 32,
  "precision_rescue_macs_per_cycle": 32,
  "iterative_refinement": true,
  "ir_max_iters": 50,
  "ir_tolerance": 1e-3,
  "ir_min_improvement": 1e-3,
  "ir_residual_macs_per_cycle": 256
}
```

软件侧默认：

```text
effective_bits = 30
equilibrate = pow2-row
max_scale_exponent = 60
```

批量入口已经支持扫描 `q_use_bits`、`frac_bits`、accumulator、guard bits、定点主元阈值
和求精轮数，见 `systemc/scripts/run_sweep.py`。

## 14. 报告接口

`summary.json` 新增或扩展：

- 原/均衡 relative residual；
- componentwise backward error；
- initial residual、residual history、接受的求精轮数和停止原因；
- `accuracy_target` 与 `accuracy_target_met`；
- precision-rescue 节点数；
- 正常矩阵溢出与救援重新量化饱和分别计数；
- 配置快照和 factor/solve/total cycles。

`nodes.csv` 新增：

- U exponent、update exponent；
- precision-rescue 标志/次数；
- small pivot、workspace renormalization；
- minimum pivot ratio、maximum growth ratio。

`operations.csv` 除七类 Panel/TRSM/GEMM 操作外，还记录：

- `PRECISION_RESCUE`；
- `SOLVE_FORWARD`、`SOLVE_BACKWARD`；
- `SOLVE_RESIDUAL`。

## 15. 当前实验结果

### 15.1 实验条件

- 日期：2026-07-25；
- ordering：仓库内确定性 AMD 启发式；
- software source：30 effective bits、power-of-two row equilibration；
- SystemC：`config/default.json`；
- seed：1；
- residual：未均衡原方程。

### 15.2 FP64 基准

| 矩阵 | 状态 | factor 相对误差 | 原方程 residual | 解相对误差 | 周期 |
|---|---|---:|---:|---:|---:|
| 256JJ | ok | `2.526e-16` | `1.349e-12` | `3.825e-8` | 152858 |
| 576JJ | numeric failure | — | — | — | 336656 |
| 1024JJ | ok | `4.801e-17` | `2.519e-12` | `4.361e-10` | 6229110 |

576JJ 的失败为根节点最后一列主元低于严格 `1e-12 × front_norm` 阈值。它说明矩阵/
ordering 本身接近当前无 delayed pivot 设计的稳定边界，而不是 SystemC 卡死。

### 15.3 定点/hybrid 模式

| 矩阵 | 状态 | factor 相对误差 | 原 residual | 均衡 residual | 后向误差 | 解误差 | 救援节点 | 求精 | 周期 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 256JJ | ok/达标 | `1.100e-7` | `5.377e-4` | `1.068e-6` | `1.544e-7` | `2.904` | 2 | 2 轮，达标 | 176774 |
| 576JJ | ok/未达标 | `1.450e-7` | `2.346e-2` | `3.199e-5` | `8.346e-6` | `2.362` | 13 | 0 轮，修正非下降 | 1687907 |
| 1024JJ | ok/达标 | `2.438e-7` | `9.434e-4` | `3.539e-7` | `1.378e-4` | `1.394e-3` | 1 | 40 轮，达标 | 7393107 |

三个定点实验的正常 `matrix_overflow_count` 均为 0。

### 15.4 如何解释结果

- 1024 的原始 zero pivot 已被解决，分解、求解和求精正常结束；
- 576 已从“错误结果或数值失败”改善为可完成，但精度仍未达 `1e-3`；
- 256、576 的解误差仍大，反映问题条件数高，不能把 residual 达标解释为前向解精确；
- 576 的均衡 residual 明显小于原 residual，说明原方程的行尺度会放大剩余误差；
- rescue 次数本身是架构信号：576 有 13 个危险节点，不适合仅靠一个偶发慢路径掩盖。

## 16. 验证状态

当前已通过：

- 软件 pytest：30/30；
- SystemC CTest：3/3；
- ATU/HPU 边界、握手和最小负数测试；
- 宽除法、guard-bit cancellation、救援触发单元测试；
- Python 产物到 SystemC 的完整端到端测试；
- ABI 版本、截断、越界、重叠和 map 损坏拒绝测试；
- 20 个反压 seed、可复现性和性能单调性测试；
- AddressSanitizer + UndefinedBehaviorSanitizer 全部 CTest。

端到端损坏测试也验证 precision rescue 不能利用黄金 FP64 front 绕过被破坏的定点 DDR。

## 17. 对 RTL 的影响

### 17.1 QAU

QAU RTL 应优先实现：

- power-of-two row exponent 的纳入或软件已缩放数据的 exponent 透传；
- `e_asm=max(e_sources)`；
- 对称有符号最近舍入右移；
- int64 align accumulator；
- maxabs 和 26-bit node scale selector；
- U/update 独立 requantize；
- drop、overflow、saturation 计数。

其中 U/update 独立 exponent 是当前稳定性方案的必要接口，不应再退回单一 node exponent。

### 17.2 HPU

- 候选只允许 `k..pivot_dim-1`；
- 最大候选数 256；
- 行索引 9 bit；
- 支持上游统一列归一化；
- 覆盖 int32 最小负数、同绝对值、全零和 backpressure。

HPU 不需要自己实现 64-bit 比较；但必须明确接收的候选已经使用共同 shift，且报告全零。

### 17.3 ATU

- 初始化实际 `pivot_dim` 行；
- update 行保持 identity/bypass；
- 对 normal fixed path 重放 HPU 逐列 pivot；
- precision-rescue 若未来硬件化，需要接收高精度后端给出的 selected row 序列。

### 17.4 GCU

GCU/Scoreboard 必须把 precision rescue 当作节点 compute 的一种延迟可变状态。只有救援完成、
U/update 写回 DDR 后，才能提交 NodeCommit、释放父依赖和 buffer。救援失败必须通过
NodeStatus 传播，不能表现为依赖永不满足。

### 17.5 计算核与救援后端

当前项目不要求实现 Panel/TRSM/GEMM 或 FP64 rescue RTL。论文应明确：

- 正常核的数值规则和周期模型由 SystemC 实现；
- rescue 是架构假设与性能模型；
- 若将来实现硬件，可选 CPU/Host 回退、共享 FP64 单元或更宽定点 microcode；
- SystemC 中保存 rescue FP64 factor 供 hybrid solve，不属于现有 DDR ABI。

## 18. 当前限制

当前实现仍有以下明确边界：

1. 不支持 delayed pivot；
2. 不支持运行时扩大/合并 front；
3. 不支持列主元和一般非结构对称稀疏结构；
4. 不支持 tile/block 级 exponent，只支持 local、assembled、U、update 和 node-vector 层级；
5. precision rescue 是 FP64 SystemC 后端，不是 RTL；
6. 迭代求精的 residual 使用稠密黄金 `A`，属于 Host Checker/研究模型；
7. `mode=both` 共用一次节点执行，严格 FP64 路径若先失败会终止该次运行；576 的 fixed
   压力结果需使用 `--mode fixed` 单独测量；
8. 576 的 `2.346e-2` 尚未满足默认精度目标；
9. 对病态问题，当前 partial pivot 范围受 `pivot_dim` 限制，update 行不允许成为主元；
10. SystemC 的 DDR/计算延迟是事务级模型，不是门级或真实 AXI/PHY 时序。

这些限制应作为论文结论和后续工作，而不应在结果中隐藏。

## 19. 下一步研究建议

按收益与项目风险排序：

1. 先完成 QAU RTL 的 exponent align、int64 accumulate、maxabs 和 U/update 双 exponent，
   用当前 DDR/CSV 做 Python-SystemC-RTL 三方逐元素验证；
2. 对 256/576/1024 扫描 `q_use_bits=24/26/28`、guard bits、QF、pivot threshold 和
   accumulator 位宽，建立误差—救援次数—周期曲线；
3. 对 576 评估 tile/block exponent，确认误差来自少数 front 的内部动态范围还是树级
   update 对齐；
4. 比较 static row+column equilibration。列缩放会要求最终解反缩放，ABI 和 Checker
   需要同步扩展；
5. 若 576 仍不达标，再评估 delayed pivot 或动态 front 扩张。这会改变任务尺寸、map、
   buffer 容量和依赖，工程代价明显高于缩放改进；
6. 迭代求精可研究更高精度 residual、GMRES-based correction 或条件数估计，但不应把
   求精作为掩盖不稳定分解的唯一措施。

## 20. 可复现实验命令

### 20.1 生成真实矩阵产物

```bash
cd graduation-code/software

python -m src.main \
  -mtx example/256X256JJ.mat \
  --rhs example/256fuv.mat \
  --ordering amd \
  --out /tmp/mf-256

python -m src.main \
  -mtx example/576X576JJ.mat \
  --rhs example/576fuv.mat \
  --ordering amd \
  --out /tmp/mf-576

python -m src.main \
  -mtx example/1024X1024JJ.mat \
  --rhs example/1024fuv.mat \
  --ordering amd \
  --out /tmp/mf-1024
```

默认命令已经启用 `pow2-row` 和 30-bit 源量化。复现旧行为可显式加入
`--equilibrate none`。

### 20.2 构建与测试

```bash
cd graduation-code/systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### 20.3 运行

```bash
./build/system_sim \
  --artifact /tmp/mf-1024/manifest.json \
  --config config/default.json \
  --mode fixed \
  --out /tmp/mf-1024-fixed \
  --seed 1
```

FP64 与 fixed 对比可用 `--mode both`；对严格 FP64 会失败的 576，应分别执行
`--mode fp64` 和 `--mode fixed`。

### 20.4 批量扫描

```bash
python scripts/run_sweep.py \
  --system-sim build/system_sim \
  --artifact /tmp/mf-256/manifest.json \
  --base-config config/default.json \
  --out /tmp/mf-256-sweep \
  --mode fixed
```

默认是单因素扫描；`--quick` 使用代表配置，只有显式 `--cartesian` 才运行完整笛卡尔积。

## 21. 结论

当前改造解决了“1024 定点分解在中间消元产生零主元”这一硬失败，并使 576 定点压力用例
能够明确完成；同时把原设计中模糊的数值失败转化为可定位、可计数、可估算周期的系统事件。

但结果也表明，仅靠定点位宽和迭代求精不能保证所有病态矩阵达到目标。576 仍未满足
`1e-3`，256 虽 residual 达标但前向解误差较大。正确的项目结论应是：当前 SystemC
原型已经具备研究和指导 RTL 的完整观测能力，并给出了可实现的稳定性基线；下一阶段应以
QAU 双 exponent/宽累加 RTL 和 576 的 block scaling/delayed-pivot 研究为重点。
