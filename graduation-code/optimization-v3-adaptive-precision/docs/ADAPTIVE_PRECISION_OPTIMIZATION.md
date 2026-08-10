# v3 定点精度优化：设计、实验与 RTL 建议

## 1. 优化目标与版本边界

v2 已实现完整的 16×16 Tile BFP SystemC 路径，但真实矩阵上仍有大量节点需要
FP64 precision rescue：

| 矩阵 | 节点数 | v2 FP64 rescue |
|---|---:|---:|
| 256 | 73 | 68 |
| 576 | 157 | 150 |
| 1024 | 274 | 122 |

这意味着 v2 能给出稳定结果，却不能支持“绝大多数计算可由整数 RTL 独立完成”的
硬件结论。v3 的目标是在不改变 16×16 Tile BFP 矩阵格式、不修改冻结 v2 和基线代码
的前提下，降低 rescue、改善求解精度，并保留可复现的失败方案作为论文对照。

全部改动位于：

```text
graduation-code/optimization-v3-adaptive-precision/
```

## 2. 原设计的问题

### 2.1 QF20 与误差门槛不匹配

v2 的 L 乘子使用 int32 QF20：

```text
L_real = L_q × 2^-20
量化步长 = 2^-20 ≈ 9.54e-7
```

写回前的局部因子检查门槛是：

```text
eta_node =
  ||P A_assembled - L U - S||_F / ||P A_assembled||_F
  <= 2e-7
```

单个 L 的量化步长已经大于检查门槛。虽然误差不会简单按一个量化步长直接等于
`eta_node`，但这两个量级的矛盾使大量节点即使没有零主元、溢出或饱和，也会因为
局部重构误差进入 FP64 rescue。

### 2.2 guard bits 不能替代 L 精度

`workspace_guard_bits` 保护 Panel/Schur 更新工作区中的低位，主要解决消减和对齐时的
信息丢失。L 一旦由除法器写入独立 QF 缓冲，工作区 guard 不会改变 L 的 QF 步长。

576JJ 定向实验中，QF20 下把 guard 从 20 改为 16 或 24，residual 和 150 个 rescue
基本不变。这说明当时的主导因素是 L 格式，而不是工作区低位。

### 2.3 只看 residual 会误判

病态矩阵上，小 residual 不保证小前向解误差。v2 的 576JJ：

```text
relative residual       = 3.252e-4
relative solution error = 6.934
```

因此 v3 同时使用 residual、componentwise backward error、solution error、局部因子
误差、rescue 数量和周期评价方案，不用单一指标宣布成功。

## 3. 定向参数扫描

在 576JJ、16×16 Tile BFP、30-bit mantissa、int64 工作区、局部因子门槛 `2e-7`
条件下，只改变 L 的 QF 小数位：

| L 格式 | residual | 解误差 | FP64 rescue | IR |
|---|---:|---:|---:|---:|
| QF20 | `3.252e-4` | `6.934` | 150 | 1 |
| QF22 | `3.581e-3` | `1.238` | 58 | 1 |
| QF24 | `9.178e-4` | `2.270` | 37 | 1 |
| QF26 | `3.024e-4` | `0.554` | 37 | 0 |
| QF28 | `5.254e-4` | `3.295` | 37 | 0 |

结果不是“小数位越多越好”的单调关系。原因包括：

- 更高 F 减小 L 量化步长；
- 同时缩小 int32 L 可表示的实数范围；
- 节点 update 的微小变化会改变父 front 的舍入、主元和求精轨迹；
- 病态系统会放大这些非单调变化。

QF26 在本组矩阵上取得最好的综合结果，因此作为 v3 默认值。它的步长为：

```text
2^-26 ≈ 1.49e-8
```

比 QF20 小 64 倍，也明显低于 `2e-7` 局部因子门槛。

## 4. 实现的两种精度策略

### 4.1 统一 QF26（推荐）

所有节点直接使用 QF26：

```text
QAU assembled front
→ QF26 Panel/TRSM/GEMM-Schur
→ local factor check
→ 通过：写回 fixed factor
→ 失败：FP64 rescue
```

它只执行一次定点分解，避免不同节点精度和额外重算周期。

### 4.2 F20→F26→FP64（保留的实验策略）

实验配置先使用 QF20。只有局部因子重构误差超限时，才完整重算 QF26：

```text
QF20 factor
  ├─ pass → commit QF20
  └─ FactorCheckFailure
       → QF26 factor
          ├─ pass → precision-assisted fixed factor
          └─ fail → FP64 rescue
```

零主元、小主元、工作区溢出、U 对角再量化为零等错误不会盲目先重试 QF26，而是直接
进入原有 rescue；因为提高 L 小数位并不能可靠修复这些类别。

实现增加了 `FactorCheckFailure`，它继承原有 `PrecisionRescueRequired`。这样旧的
失败处理语义仍成立，同时 dispatcher 可以只对“局部因子检查失败”触发精度重算。

## 5. 每节点精度状态

`FixedFactor` 新增：

```text
l_frac_bits
fraction_retry_attempted
precision_assisted
precision_rescued
```

含义：

- `l_frac_bits`：该节点 L 的真实 QF 小数位；
- `fraction_retry_attempted`：是否执行过第二次定点分解；
- `precision_assisted`：第二次 QF26 分解成功，最终仍是整数因子；
- `precision_rescued`：最终使用 FP64 rescue 因子。

前代和局部因子误差计算均按每节点 `l_frac_bits` 解码 L，不再使用一个全局 F 假设。
完整 rescue 节点的前代/回代继续读取对应的高精度 shadow factor。

## 6. 周期模型

初始定点尝试始终生成一套现有七类操作：

```text
FACT
TRSM_U
TRSM_L
GEMM_PIVOT
TRSM_F12
TRSM_F21
GEMM_SCHUR
```

发生 F26 重算时，SystemC 从当前周期重新生成并调度第二套七类操作，timeline 写入
`fixed_frac_retry`。若重算仍失败，再追加 `PrecisionRescue` 操作。

因此三条路径的周期关系为：

```text
普通 fixed       = 1 × fixed pass
precision assist = 2 × fixed pass
full rescue      = 1 × fixed pass + rescue
retry + rescue   = 2 × fixed pass + rescue
```

数值函数在主机上执行的实际墙钟时间不作为硬件周期；报告周期来自事务/核吞吐模型。

## 7. DDR node_meta 扩展

NodeTask 仍是小端、128 字节 ABI v2。v3 只扩展 `node_meta` payload：

```text
byte 0: factor valid
byte 1: flags
        bit0 full FP64 rescue
        bit1 QF26 precision assist
        bit2 fraction retry attempted
byte 2..3: first U exponent
byte 4..5: first update exponent
byte 6: tile size
byte 7: actual L frac bits
byte 8..11: U tile exponent count
byte 12..15: update tile exponent count
byte 16..: U tile exponents
```

旧 reader 不能区分 byte 1 中的新 flags，因此 v3 写回镜像必须配合 v3 reader 使用。

## 8. 报告接口

`summary.json.stability`：

- `fraction_retry_nodes`；
- `precision_assisted_nodes`；
- `precision_rescue_nodes`。

`nodes.csv`：

- `l_frac_bits`；
- `fraction_retry_attempted`；
- `precision_assisted`；
- `precision_rescued`；
- 三类累计计数。

`operations.csv` 和 `timeline.csv` 可以验证重算是否真实计入周期，而不只是改变数值
函数参数。

## 9. 三组真实矩阵对比

所有结果使用相同 v2 Tile BFP 制品、seed 1 和 `--mode fixed`。

### 9.1 推荐的统一 QF26 对比冻结 v2

| 矩阵 | 指标 | v2 QF20 | v3 QF26 |
|---|---|---:|---:|
| 256 | final residual | `5.964e-4` | `7.096e-5` |
| | solution error | `1.514` | `0.174` |
| | FP64 rescue | 68 | 9 |
| | cycles | 178625 | 179858 |
| 576 | final residual | `3.252e-4` | `3.024e-4` |
| | solution error | `6.934` | `0.554` |
| | FP64 rescue | 150 | 37 |
| | cycles | 1693321 | 1686076 |
| 1024 | final residual | `1.012e-4` | `2.816e-5` |
| | solution error | `3.000e-8` | `1.293e-6` |
| | FP64 rescue | 122 | 49 |
| | cycles | 8722424 | 7237984 |

主要变化：

- rescue 分别减少 `86.8%`、`75.3%`、`59.8%`；
- 256 周期增加 `0.69%`，原因是多执行一轮迭代求精；
- 576 周期减少 `0.43%`；
- 1024 周期减少 `17.02%`，主要来自 rescue 和求精次数下降；
- 三组 final residual 均满足 `1e-3`；
- 1024 solution error 比 v2 大，但仍为 `1.29e-6`，同时 residual 和周期改善。

### 9.2 自适应策略为什么不推荐

| 矩阵 | QF 重算 | 定点辅助成功 | FP64 rescue | residual | 解误差 | cycles |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 66 | 59 | 9 | `5.517e-4` | `1.216` | 182462 |
| 576 | 137 | 113 | 37 | `4.489e-4` | `2.836` | 1694797 |
| 1024 | 122 | 77 | 46 | `9.056e-5` | `2.654e-6` | 8291797 |

与统一 QF26 相比，自适应策略：

- 需要 66～137 次完整定点重算；
- 只在 1024 上额外减少 3 个 rescue；
- 三组 residual 和 solution error 都更差；
- 周期分别增加约 `1.45%`、`0.52%`、`14.56%`。

因此“低精度先试一次”并没有节省硬件工作，反而重复计算。对于本项目已知的
16×16 Tile BFP 工作负载，统一 QF26 是更简单、更快、更稳定的实现。

完整数值在 `systemc/results/precision_comparison.csv`。

## 10. `both` 模式与 576JJ 的说明

`both` 模式会同时执行独立 FP64 黄金分解和 fixed 分解。576JJ 的 FP64 黄金路径在
默认：

```text
pivot_rel_tol = 1e-12
```

下于根节点末列明确报告小主元。fixed rescue 使用单独的：

```text
rescue_pivot_rel_tol = 1e-16
```

所以 `fixed` 模式可以完成。这是两个通道使用不同数值失败门槛的预期结果，不是
QF26 回归，也不是死锁。论文中应把“FP64 压力测试 numeric failure”和“定点策略
是否达到 residual 目标”分开陈述。

## 11. 对 RTL 的直接影响

### 11.1 L 存储容量不变

QF20 和 QF26 都使用 int32，L SRAM 的元素宽度和容量不变。变化是二进制小数点位置，
可在全局配置寄存器固定为 26；若未来支持每节点精度，才需要随 node 保存
`l_frac_bits`。

### 11.2 除法器和乘法器

L 除法的缩放左移从 20 增至 26，宽中间分子必须保留至少额外 6 bit。SystemC 使用
`__int128` 规避主机溢出；RTL 不能直接照搬类型，应根据：

```text
F + value_exponent - pivot_exponent
```

推导除法器分子宽度，并加入溢出检测。

Schur 乘法仍是 `int32 × int32`，建议至少 64-bit product/accumulator。乘积 exponent
由 `U_e-F` 计算，F=26 必须进入控制路径或固定常量。

### 11.3 表示范围的代价

int32 QF26 的实数范围约为 `[-32, 32)`，小于 QF20 的 `[-2048, 2048)`。对 pivot
候选行内的标准部分主元，LPP 通常不超过 1；但 update 行不参与主元候选，LUP 可能
更大。因此 RTL 必须保留 L 越界检测，不能因为本次真实矩阵通过就删除 rescue/failure
接口。

### 11.4 推荐的硬件实现顺序

1. 第一版 RTL 固定 QF26，不实现运行时 F20/F26 重算；
2. 保留 `L_overflow`、`zero_pivot`、`small_pivot` 和 accumulator overflow 状态；
3. 用 SystemC 的每节点 P-vector、L/U/update 和 exponent 表做三方对照；
4. 将 SystemC 的高精度局部因子检查替换为可硬件实现的风险代理，再评估是否需要
   高精度 Panel、软件回退或 delayed pivot；
5. 不建议为当前自适应策略增加双模式控制，因为实验没有显示性能收益。

## 12. 已知边界与下一步

- precision rescue 仍是 SystemC 高精度功能/周期模型，没有对应 RTL；
- tile 模式的向量求解仍使用 exponent-aware double 控制路径消费定点因子，尚不是
  纯整数 V-format；
- 当前结论来自 256/576/1024 三组矩阵，固定 QF26 仍需更多矩阵验证其 L 范围风险；
- 下一项最有价值的工作是实现 tile-aware 纯整数向量格式，以及为局部因子误差设计
  可综合的代理指标。

## 13. 验证状态

- v3 软件测试：33 项全部通过；
- SystemC CTest：3/3；
- ASan/UBSan：3/3；
- 新增单元测试验证 factor 保留实际 `l_frac_bits`；
- 新增单元测试验证局部误差门槛抛出可重试的 `FactorCheckFailure`；
- 256/576/1024 统一 QF26 与自适应策略均完成 fixed 模式，无死锁；
- 相同配置与 seed 的结果可复现。
