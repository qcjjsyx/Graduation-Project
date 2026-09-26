# 05 多波前装配与树形求解

代码：`software/src/verify/front_reference.py`（host 参考）、
`systemc/reference/` 的 `factor_front_tree` / `solve_front_tree`（独立 C++ 参考）、
`software/src/pipeline.py` 的 `extract_local_contribution`

## 1. 多波前法要解决什么

直接对稀疏矩阵做 LU 会在填充位置上产生大量零散运算。多波前法把消元组织成
**消去树上的稠密块运算**：

- 每个节点（超节点）对应一个 **front（前沿矩阵）**：它的 pivot block 是要消去的
  变量，update 部分是与后代共享的变量；
- 子节点消元后留下的**Schur 补**向上装配进父节点 front；
- 每个 front 内部按稠密矩阵处理（Panel-LU、TRSM、GEMM），天然映射到设备的宏命令。

## 2. Schur 补与局部消元

设 front 分块为

```text
F = [ A11  A12 ]      A11: pivot block (p×p)
    [ A21  A22 ]      A22: update block (u×u)
```

消去 pivot block：

```text
A11 = L11 · U11                      （按主元策略分解）
U12 = L11^{-1} · A12                 （TRSM-left）
L21 = A21 · U11^{-1}                 （TRSM-right）
S   = A22 - L21 · U12                （Schur 补，GEMM：C = C - A×B）
```

- `S` 就是子节点传给父节点的 **update contribution**；
- 因式保存 `L = [[L11],[L21 I]]`、`U = [[U11 U12],[0 S]]`；
- 在设备命令流中分别对应 `PANEL_LU`、`TRSM_LEFT`、`TRSM_RIGHT`、`GEMM_SCHUR`、
  `STORE_FACTOR`、`STORE_UPDATE`（见 08）。

## 3. 局部贡献的提取与 LCA 存储规则

**规则**：原始矩阵元素 `A[i, j]` 存放在"变量 i 与变量 j 的**最低共同祖先**节点"
的局部 front 中（"属于"指该变量是该节点的 pivot；对超节点树而言即列级消去树的
推广）。

因此节点的局部贡献为

```text
local[node] = A_hat[front_indices, front_indices]
local[node][pivot:, pivot:] = 0        # update-update 块清零
```

- pivot–pivot、pivot–update 元素：两端点的 LCA 就是本节点 ⇒ 保留；
- update–update 元素：两端点属于本节点后代的不同子树 ⇒ 其 LCA 是更上层的祖先，
  会由子节点的 Schur 更新装配得到 ⇒ 本节点清零。

### 为什么清零不会丢信息（3 变量链上的验证）

设稠密矩阵 `A`，front 链 `node0{0,1,2} → node1{1,2} → node2{2}`，局部贡献按规则
清零后：

```text
node0 local = [ A00 A01 A02 ]     node1 local = [ A11 A12 ]    node2 local = [A22]
              [ A10  0   0  ]                   [ A21  0  ]
              [ A20  0   0  ]
```

node0 消去变量 0 得 `S0 = -[A10;A20][A01 A02]/A00`；装配后
`node1 assembled = local1 + S0`，其 `(2,2)` 元素是 `-A20A02/A00`（缺 `A22`——
它在 node2 的 local 里）。node1 消去变量 1 得到标量 Schur，node2 装配：

```text
node2 assembled = A22 - A20A02/A00
                  - (A21 - A20A01/A00)(A12 - A10A02/A00)/(A11 - A10A01/A00)
```

而前 2×2 块的块消元 Schur 补正是

```text
S = A22 - [A20 A21] · inv([[A00,A01],[A10,A11]]) · [A02;A12]
  = A22 - A20A02/A00 - (A21 - A20A01/A00)(A12 - A10A02/A00)/(A11 - A10A01/A00)
```

两者恒等（块 Gauss 消元恒等式）⇒ 清零 + 子 Schur 装配的组合是精确的，
`A22` 由最后拥有它的节点提供，恰好只出现一次。

## 4. 装配语义与 map 表

对每个子节点的贡献：

```text
parent_front[col_map[i], col_map[j]] += child_schur[row_map[i], row_map[j]]
```

当前主线 `row_map` 恒为 `range(update_dim)`（子 update 块内的行索引），
`col_map[i]` 是该 update 变量在父 front 中的局部位置（见 03 §7）。
C++ 参考接口中对应 `ContributionMap{child, row_map, col_map}`。

- 装配是**加法**：父 front 的局部贡献已经包含属于它的原始元素，子 Schur 是修正量；
- 同一父节点的多个子贡献互不重叠（各子 update 集合按树结构划分）。

## 5. 分解顺序与数据依赖

处理顺序必须是"**子先于父**"的拓扑序：

```text
order = sibling_friendly_order(parent)
for node in order:
    装配所有子贡献（此时子 Schur 已算出）
    factor_front(node)  → 得到本节点 Schur，供父节点使用
```

- 顺序确定性：同层按父节点分组、按索引稳定排序（设备端对应 token 依赖，
  见 08）；
- C++ 参考 `factor_front_tree` 内部按 pending-children 计数得到同样的拓扑序；
- 多根森林（多个 parent = -1）同时支持。

## 6. 树形求解

分解完成后按同一棵树做前代/回代：

```text
前代（order，子先父后）：
    按局部置换 p_local 从 RHS 取值
    L11 z = rhs                      （pivot block 内前代）
    update 行： rhs_update -= L21 · z （写回工作向量，供祖先使用）
回代（逆序，父先子后）：
    U11 x_pivot = z - U12 · x_update
```

- 每个变量恰由**一个** front 的 pivot block 消去（所有权唯一），否则拒绝；
- 置换范围：`p_local` 只覆盖 pivot block，且要求 `p_local[row] < pivot_dim`；
  update 行永不移动——这是与设备契约一致的约束（见 06）；
- 静态策略下 `p_local` 为恒等；动态策略下为块内交换结果；
- 不一致的 factor（跨块换行、update 行被移动、变量有多个所有者、变量未被任何
  front 消去）会被 host 与 C++ 两侧同时拒绝。

## 7. 复杂度

单个 front 的代价：`O(p³)`（Panel-LU）+ `O(p²u)`（TRSM/GEMM）；整棵树的总代价约
为 `Σ_node O(front_size³)` 量级。front 大小与 supernode 大小受
`max_front_size`/`max_supernode_size`（默认 256）约束，超出即编译失败、如实报告。

## 8. 验证锚点

- host 参考：`factor_front`、`factor_front_tree`、`solve_front_tree`
  （`src/verify/front_reference.py`）；
- C++ 独立参考：`panel_lu_policy`、`factor_front`、`factor_front_tree`、
  `solve_front_tree`（`systemc/reference/`）；
- 两侧**同策略整树交叉验证**：共享 fixture
  `graduation-code/fixtures/reference_tree/tree_policy_cases.json`
  （9 个用例：两策略成功、失败分类、跨块拒绝、多根森林、block 内换行且 update
  行不动），由 `tests/test_reference_cross.py` 与 `fp64_reference_tests` 分别复现
  同一组期望值。
