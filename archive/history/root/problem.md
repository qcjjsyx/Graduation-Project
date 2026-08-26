# 待统一处理的问题

最后更新：2026-08-10

## 1. 软件错误限制原始矩阵必须具有对称非零结构

### 状态

- 已于 2026-08-10 修复并通过软件与 SystemC 端到端验证。
- ABI v2 的二进制布局未改变；manifest 新增并强制校验符号包络语义。

### 修复结果

- 流水线保留原始数值矩阵 `A`，不再拒绝非零结构不对称的方阵；
- 新增公共符号模式构造，先丢弃数值，再计算
  `pattern(A) union pattern(A.T)`，AMD、RCM 与 fill 共用同一语义；
- `matrix.structurally_symmetric` 记录原始输入的真实属性，`symbolic` 段明确记录包络来源；
- Python 与 C++ 验证器均检查上述元数据；
- 非对称算例验证了 `A_local` 对排序后原始矩阵的唯一、完整重构，并通过 SystemC
  FP64/定点求解与原方程 residual 检查。

### 问题位置（修复前）

修复前，软件流水线在读取矩阵后调用：

```python
require_structurally_symmetric(original_matrix)
```

`symbolic_fill_pattern()` 内部也会再次调用同一检查。因此，只要原始矩阵满足
`pattern(A) != pattern(A.T)`，流水线就会在符号分析前拒绝输入。

此外，manifest 此前将：

```json
"structurally_symmetric": true
```

硬编码为真，README 和现有测试也把“结构不对称必须失败”描述为当前约束。

### 逻辑结论

该限制不是当前多前沿 LU 数值逻辑的必要条件。对于一般方阵，应区分：

```text
数值矩阵：A
符号包络：S = pattern(A) union pattern(A.T)
```

ordering、fill、elimination forest、supernode 和 front 可以基于对称符号包络 `S`
构造；节点 `A_local`、FP64/定点 LU、RHS 和最终 residual 仍必须使用原始数值矩阵
`A`。构造符号包络不会令数值矩阵对称，也不会丢失 `A_ij != A_ji` 的方向性。

应使用非零模式的并集，而不是直接进行数值 `A + A.T`，避免相反数发生抵消后错误地
删除符号边。

对固定消去顺序，非对称 LU 在消去 `k` 时由 `(i,k)` 和 `(k,j)` 产生的 `(i,j)` fill，
会被对称包络中消去 `k` 时形成的邻居 clique 覆盖。因此该方案是保守符号分析：可能
高估 fill、生成较大的 front，但不应仅因原始结构不对称而漏掉潜在 fill。

### 修复时核对的影响范围

1. `software/src/pipeline.py`
   - 移除或替换入口处对原始矩阵的结构对称检查；
   - 明确分离 numeric matrix 与 symbolic pattern；
   - 不再在 manifest 中硬编码结构对称。
2. `software/src/symbolic/fill.py`
   - 允许一般方阵输入，内部构造非零模式对称包络；或只接收已显式构造的包络；
   - 避免入口和 fill 函数重复执行互相矛盾的检查。
3. `software/src/symbolic/ordering.py`
   - 所有 ordering 统一使用对称符号包络；
   - 特别检查 RCM 的 `symmetric_mode=True`，不能把原始非对称 pattern 当作已对称输入。
4. manifest、README、ABI 说明与验证器
   - 记录原始矩阵是否结构对称；
   - 记录符号分析采用 `pattern(A) union pattern(A.T)`；
   - 删除“结构不对称必须拒绝”的错误约束。
5. 测试
   - 将原有结构非对称拒绝测试改为支持性测试；
   - 增加结构不对称且非奇异矩阵的端到端测试；
   - 验证所有 `A_local` 贡献能够唯一、完整地重构排序后的原始数值矩阵；
   - 验证 Python 参考解、SystemC FP64、定点通道和原方程 residual。

### 与本问题不同但相关的数值边界

解除结构对称限制不等于已经支持所有一般非对称稀疏矩阵。当前主元搜索只覆盖当前
supernode 的 pivot 行，不允许 update 行成为主元，也没有 delayed pivot、动态 front
扩张、结构匹配或独立行列置换。某些整体非奇异的非对称矩阵仍可能因为当前 pivot block
内没有可用主元而失败。

后续需要分别评估：

- 行/列 matching 或 maximum transversal；
- 独立行列 ordering；
- threshold partial pivoting；
- delayed pivot 与动态 front 扩张及其 ABI/buffer 代价。

这些属于主元稳定性与通用非对称 LU 支持问题，不应与“符号分析是否可以使用
`pattern(A) union pattern(A.T)`”混为一谈。

### 验收结果

- [x] 原始非零结构不对称的方阵能够完成软件制品生成；
- [x] 符号结构明确来自非零模式对称包络；
- [x] 数值方向性和原始矩阵元素不丢失、不重复；
- [x] ABI 地址、map table、消去森林及多根情况继续通过验证；
- [x] 结构不对称、非奇异的端到端算例通过 FP64 与定点求解及原方程 residual 检查；
- [x] 数值不可解情形继续报告 `numeric_failure`，不误报结构不支持或死锁。
