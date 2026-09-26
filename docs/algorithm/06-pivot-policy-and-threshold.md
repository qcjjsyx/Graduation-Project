# 06 主元策略与阈值

代码：`software/src/config.py`（`PivotPolicyConfig`）、
`software/src/verify/front_reference.py`、`systemc/reference/`（`panel_lu_policy`）
契约：`docs/interface/Command_Schema_v2.md` §4–§5

## 1. 策略分层

```text
Host 数值预处理： none | weighted_matching      （默认 weighted_matching）
Device 主元策略： STATIC | DYNAMIC_PANEL        （默认 STATIC）
```

两者独立：默认组合 `weighted_matching + STATIC`；允许显式关闭匹配做消融；动态模式
也可以使用同样的匹配预处理，保证对比公平。策略是 **batch 级统一** 的，且只能来自
descriptor 字段（`FRONT_DESC.body_words[8]`），**绝不从 P-vector 内容反推**。

## 2. STATIC（默认，固定主元顺序）

- 按最终排序后的对角固定顺序消元（第 k 步消 pivot block 第 k 列）；
- **不搜索候选行、不交换行**，不依赖 HPU/ATU 动态映射事务（静态路径运行时旁路；
  "运行时旁路"与"综合时裁剪"是两种不同能力）；
- 仍然检查**实际消元后**的主元，而不是只检查输入对角线：

  ```text
  零主元                                  → STATIC_PIVOT_FAILURE
  0 < |pivot| <= ratio · scale            → PIVOT_UNSTABLE
  NaN/Inf（输入或消元产生）                → NON_FINITE_VALUE
  ```

- 失败语义：
  - 固定顺序失败**不是**矩阵奇异性结论，不静默替换微小主元；
  - 不在执行途中切换策略；
  - 失败命令不发布 READY token；Host 重试必须从原始输入或完整重建状态开始。

## 3. DYNAMIC_PANEL（可选，块内选主元）

```text
第 k 步候选行： k .. pivot_dim-1（只在本 front 的 pivot block 内）
选择： 最大绝对值；tie-break = 最小逻辑行号（严格大于比较，确定性）
```

- 不支持：跨 front 换行、update 行提升为 pivot、delayed pivot、动态 front 扩张；
  必须在更大的范围内才能继续时明确失败（`PIVOT_NOT_FOUND`）；
- 实际的主元搜索由 HPU 完成、逻辑→物理行映射由 ATU 维护，但这些都是 Panel 控制器
  的**内部微操作**，不展开为顶层 command；
- 与 STATIC 一样受同一阈值检查：块内最佳候选仍不达标 ⇒ `PIVOT_NOT_FOUND`；
  无有限候选 ⇒ `NON_FINITE_VALUE`。

## 4. 相对阈值：尺度的选择

### 4.1 定义

```text
scale = front_pre_elimination_max_abs
      = 该 front 装配完成、消元开始前的 max |A_front|
threshold = threshold_ratio · scale        （ratio ∈ [0, 1)）
```

- **选相对阈值而非绝对阈值**：均衡后（01）各数量级仍可能不同，绝对常数没有跨矩阵
  意义；相对 front 自身尺度才可解释。
- **尺度来源必须单次归约**：只对整个 front 取一次最大值。静态检查**不得**隐含
  "逐列候选最大搜索"——那会把 STATIC 重新拖回 HPU 式开销，违背固定顺序的初衷。
- 零主元永远拒绝（任何 ratio 下 `0 <= ratio·scale` 恒成立）。

### 4.2 binary32 规范化（host 与设备共用同一有效值）

设备把阈值存为 `FRONT_DESC.body_words[9]` 的 IEEE-754 binary32 位型；若 host 配置
直接用 float64，会出现"0.1 的 float64 值 ≠ binary32(0.1)"导致两侧判决不一致。
因此阈值在**配置时一次性规范化**为其 binary32 值：

```text
normalize(ratio) = float(binary32(ratio))
```

所有消费者（配置、manifest、host reference、C++ reference、descriptor）共用该值。
拒绝两类不可表示输入：

```text
ratio > 0 但 binary32 下溢为 0     → 拒绝（设备会静默丢阈值）
ratio 舍入到 1.0                    → 拒绝（离开合法区间 [0,1)）
```

## 5. P-vector 语义（设备局部置换）

| 属性 | 值 |
| --- | --- |
| 类型/长度 | INT32，`pivot_dim` |
| 方向 | gather：`p[i]` = 第 i 个主元位置在 pivot block 内的源行号 |
| 初值 | INT32 哨兵 `-1`（"未写"）；全零数组**永不**解释为 identity |
| STATIC | 显式写 identity `p[i] = i`（不是隐式、不是省略） |
| DYNAMIC_PANEL | 写块内交换结果 |
| 引用 | FRONT/FACTOR/SOLVE 引用**不允许 NONE**（不存在"省略"分支） |
| 校验 | 沿实际 descriptor 引用检查 INT32、容量恰为 `pivot_dim`、内容全为哨兵；同一节点的 FRONT/FACTOR/SOLVE 必须指向**同一个** region |

区分：P-vector 是**设备局部**运行期置换；`P_match`/`Q` 是 **host 全局**编译期
坐标变换（见 04）。两者不是同一个东西。

## 6. 失败分类与状态码映射

| 情形 | 参考实现失败类型 | 状态码 |
| --- | --- | --- |
| STATIC 固定顺序零主元 | `zero_pivot` | `STATIC_PIVOT_FAILURE` (0x0105) |
| 非零但低于相对阈值 | `below_threshold` | `PIVOT_UNSTABLE` (0x0101) |
| NaN/Inf | `non_finite` | `NON_FINITE_VALUE` (0x0106) |
| 动态块内无可容许主元 | `no_candidate` | `PIVOT_NOT_FOUND` (0x0100) |
| executor 未实现/不支持声明策略 | 拒绝执行 | `POLICY_NOT_SUPPORTED` (0x0006) |
| 非法策略编码、NONE P 引用、v1/v2 混合 | 验证期拒绝 | `BAD_COMMAND`/`BAD_DESCRIPTOR` |

## 7. 为什么固定顺序也能用：与预处理的配合

- 匹配（02）把较大的非零搬到对角线；
- 行均衡（01）把各行量级归一；
- 阈值以 front 自身尺度衡量；
- 三者组合使 STATIC 在多数良态矩阵上成功，从而不需要 HPU/ATU 动态路径。

但它**不保证**成功：`[[1,1,0],[1,1,1],[0,1,1]]` 对角非零、矩阵非奇异，固定顺序
第 2 步仍出现零主元，此时必须报告数值失败而不是称矩阵奇异——这也是
DYNAMIC_PANEL 存在的意义（同一矩阵块内动态可成功）。

实测参考（256 阶真实矩阵）：对**未做行预处理**的原始矩阵直接做通用稠密部分主元
动态消元，pivot growth ≈ 1.0；固定顺序直解则达 ≈ 403——说明预处理链路是静态
策略可用的前提（见 07 的 smoke 记录）。

## 8. 成本模型

- STATIC：每步只做零/阈值/有限性检查（`O(1)`），无候选搜索；
- DYNAMIC_PANEL：每步在 pivot block 列内做一次最大值扫描（`O(pivot_dim)`），
  代价被限制在块内，不随 front 的 update 行数增长；
- 无主元扰动、无自动迭代求精、无运行中策略切换（首版明确不做）。

## 9. 验证锚点

- 单 front 与整树失败分类：`tests/test_front_reference.py`、
  `systemc/reference/tests/fp64_reference_tests.cpp`（policy 用例）；
- 同策略整树交叉验证 fixture：`fixtures/reference_tree/`；
- 端到端策略落盘与 manifest 一致性：`tests/test_pipeline_e2e.py`、
  `tests/test_command_compiler.py`、`tests/test_command_codec.py`。
