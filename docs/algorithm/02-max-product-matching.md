# 02 最大乘积完美匹配（加权行置换）

代码：`graduation-code/software/src/matching.py` · 配置：`MatchingConfig`

## 1. 问题形式化

给定 `n×n` 稀疏矩阵 `A` 的**非零结构**，构造二分图：左部为行 `{r_j}`，右部为列
`{c_i}`，边 `(r_j, c_i)` 存在当且仅当 `A[j, i] ≠ 0`，边权 `w(j, i) = log|A[j, i]|`。

在所有**完美匹配**（每行每列恰好用一次的双射 `σ`，且 `A[j, σ(j)] ≠ 0`）中最大化

```text
∏_j |A[j, σ(j)]|
```

结果转成行置换 `P_match` 后用于把较大的非零"搬运"到对角线上，改善固定主元顺序
（STATIC）的消元成功率。

## 2. 为什么是"乘积"，为什么不是贪心

- **乘积（几何量级）而非求和**：求和会奖励"多而平庸"的配对，忽略量级短板。
  反例 `[[5,4],[4,1]]`：对角乘积 5、非对角乘积 16（正确选非对角），但求和
  对角 6、非对角 5（会选错方向）。
- **必须全局匹配，不能逐列/逐行独立取最大**：反例 `[[10,9],[9,1]]`：
  逐行贪心得 `10×1=10`，全局最优为 `9×9=81`。因此实现里没有"逐列 argmax"。

## 3. log 等价与数值技巧

`log` 严格单调 ⇒

```text
argmax ∏|A[j, σ(j)]|  ≡  argmax Σ log|A[j, σ(j)]|
```

这是**精确等价**（不是近似），把乘积目标化成标准线性分配问题。实现细节：

1. `log|a|` 可能为负、也可能为 `0.0`（当 `|a| = 1`），而 SciPy 的稀疏分配求解器把
   稀疏矩阵中**值为 0 的项当作"无边"**。因此做常数平移：

   ```text
   w = log|a| - min(log|a|) + 1.0     # 最小权恰为 1.0，全部 > 0
   ```

2. 平移是常数：任何完美匹配都有 `n` 条边，`Σ` 同增 `n·c`，`argmax` 不变。
3. 用 `scipy.sparse.csgraph.min_weight_full_bipartite_matching(..., maximize=True)`
   求解——稀疏 LAPJVsp（Jonker–Volgenant 最短增广路），**不构造稠密 `n×n` 代价矩阵**，
   复用现有依赖；它只接受完整匹配，不存在时抛 `ValueError`，这正好作为"无完整匹配"
   的判定信号。

## 4. 从匹配到行置换（方向推导）

匹配返回 `col_ind[j] = σ(j)`（原行 `j` 匹配到列 `σ(j)`）。要求：新对角位置 `i` 上的
元素是该列的匹配项。由同一 gather 约定 `B[i,:] = A[p_match[i],:]`：

```text
B[i, i] = A[p_match[i], i] 应等于 A[j, σ(j)]（其中 σ(j) = i）
⇒ p_match[i] = σ^{-1}(i)
```

实现：`row_perm[col_ind] = arange(n)`。`p_match` 是 **σ 的逆**，方向错误会整体
破坏变换语义——`tests/test_transform_semantics.py` 专门验证方向。

## 5. 校验链（全部硬校验，不满足即 `MatchingError`）

1. 输入规范化：CSR、float64、去重、去显式零、排序、非有限值检查；空矩阵拒绝。
2. 求解器返回：行满分配（`row_ind == arange(n)`）、列双射。
3. 每个匹配位置在 CSR 结构中**确为存有的非零**，值非零且有限。
4. 回代检查：`matched_value[p_match[i]] == A[p_match[i], i]`——置换确实把匹配项
   搬到对角线。

## 6. 失败与诊断

无完整匹配时：先用 `maximum_bipartite_matching`（最大基数匹配）找出**未匹配的
行/列**列表（截断记录），拼成可诊断的错误信息。pipeline 收到后直接终止编译，
不产出看似合法的静态产物——因为缺匹配必然导致对角线存在结构性零。

## 7. 性质与边界

- **复杂度**：稀疏分配约为 `O(n · nnz · log n)` 量级（LAPJVsp），对 256–1024 阶
  实测秒级以内；具体参见 07/验证文档的 smoke。
- **确定性**：规范化输入顺序 + 确定性算法 ⇒ 同输入同环境输出确定；SciPy 文档声明
  并列最优解可能随 SciPy/Python 版本变化，属于已声明边界。
- **不改变变量（列）**：只做行置换，解 `x` 不变。
- **与行缩放交换**：行缩放对匹配最优解集不变（常数因子）。
- **不是 MC64**：没有对偶势、没有缩放 pass、没有 partial matching/秩一更新；
  文档只声称"最大乘积完美匹配"这一阶段。
- **不保证静态消元成功**：反例 `[[1,1,0],[1,1,1],[0,1,1]]` 存在完美匹配
  （对角全非零，乘积 1），但第二主元为零，STATIC 必须报告失败。匹配是提高
  成功率的启发式预处理，不是正确性保证。

## 8. 记录与测试锚点

manifest `preprocess.matching`：method、objective、algorithm、`full_matching`、
`log_product`、`row_perm`、`matched_column_of_row`、`matched_value`。

测试：`tests/test_matching.py`（贪心/求和陷阱、log 等价（暴力枚举对照）、对角搬运、
零对角救回、`none` 基线、无匹配诊断、规范 CSR 确定性、gather 方向）。

`method = "none"` 时返回恒等置换，`full_matching=false`、`log_product=null`——
这是消融基线，不构成匹配结论。
