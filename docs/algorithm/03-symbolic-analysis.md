# 03 符号分析（包络 / 排序 / 填充 / supernode / front）

代码：`graduation-code/software/src/symbolic/`、`src/scheduler/map_gen.py`

## 1. 为什么可以"只算一次结构"

设备端两种主元策略都被约束为：

- STATIC：固定顺序，不换行；
- DYNAMIC_PANEL：只在当前 **pivot block 内**换行，且不提升 update 行、不扩张 front。

在这两种策略下，换行始终发生在同一个 front 的行集合内部，**front 覆盖的行/列集合
不变**，因此 LU 的填充结构（哪些位置会变为非零）与数值无关，只由变换后矩阵
`B = P_match D_r A` 的非零结构决定。这就是"结构一次编译、数值多次执行"的基础。

## 2. 对称包络

```text
envelope(E) = pattern(B) ∪ pattern(B^T)     （先丢弃数值，再取并，布尔语义）
```

- `pattern(B) ∪ pattern(B^T)` 保证结构对称，供排序/填充使用；
- 先丢弃数值再取并：对项不会因数值相消而消失（保守、安全）；
- 对本来就结构对称的输入，包络即原结构。
- 因为匹配只重排行，`pattern(B)` 是 `pattern(D_r A)` 的行置换，但**转置并集不是
  简单的行列置换**，所以必须在 `B` 上重新计算包络与后续全部结构。

## 3. 排序（ordering）

支持 `amd / rcm / identity`，默认 `amd`。

**本项目的 AMD 风格最小度排序**（自实现，非 SuiteSparse AMD）：

```text
剩余图 G = 包络的非对角邻接
while 还有节点:
    取当前度最小的节点（度相同取最小索引，确定性 tie-break）
    把该节点的所有剩余邻居两两连边（消元产生的填充团）
    移除该节点，记录顺序
```

- 输出为 gather 置换 `q`，后续一切索引都建立在 `q` 重排后的坐标上；
- RCM 用 `scipy.sparse.csgraph.reverse_cuthill_mckee`（对称模式）；
- `identity` 用于基线/复现实验。
- 复杂度：朴素实现约 `O(n · d²)`（`d` 为平均度），对本项目矩阵规模足够；
  不声称与生产级 AMD 等价。

## 4. 精确填充与消去树

填充在**已排序**的包络上进行（消除图法）：

```text
for k = 0..n-1:
    higher = 当前图中 k 的高号邻居（排序）
    columns[k] = [k, *higher]                 # L 的第 k 列结构
    parent[k] = higher[0]（若存在）            # 消去树父节点
    把 higher 连成完全团（新增边即 fill edge，计数）
    从图中移除 k
```

- 得到精确的符号分解结构 `columns`（用于 supernode/front）与消去树 `parent`；
- `src/symbolic/etree.py` 另提供独立的标准消去树算法（CSC 扫描 + 路径压缩），
  用于对照/测试；主链使用填充过程给出的 `parent`。

## 5. supernode（超节点）

将**结构不可区分的连续列**合并成超节点：

```text
合并条件： parent[k] == k + 1 且 filled_columns[k][1:] == filled_columns[k+1]
合并上限： max_supernode_size（默认 256）
```

- 条件含义：第 k 列的非对角结构恰好是"下一个主元 + 第 k+1 列的非对角结构"，
  这是稠密前沿块所需的标准结构条件；
- 合并成连续块后，消元可以走稠密 BLAS3 形态（局部 GEMM/TRSM），而不是逐列操作；
- `build_supernodes`（基于原图闭邻域）保留为替代实现，主线使用
  `build_supernodes_from_filled`（基于填充结构，才能捕获填充产生的一致性）。
- 上限触发时的行为：编译报错（容量限制如实暴露，不静默放宽）。

## 6. front 与前向索引

每个超节点对应一个 front（前沿矩阵）：

```text
pivot block   = 超节点自身列（连续）
update set    = ∪_{k ∈ pivot} (filled_columns[k] \ {k}) \ pivot
front_indices = pivot block + sorted(update set)        # 全局列索引（Q 空间）
```

- front 的稠密矩阵即 `A_hat[front_indices][:, front_indices]`（数值来自变换后矩阵）；
- pivot block 位于 front 的前 `pivot_dim` 行/列；
- **update-update 块在局部贡献中清零**，由子节点的 Schur 更新装配填充（原理见 05）；
- 节点范围（`node_range`）是 pivot block 在全局排序中的连续区间。

## 7. 节点树与 map 表

- 超节点树：`build_supernode_parent` 把列级消去树映射到超节点级；
- 处理顺序：`sibling_friendly_order` 给出"子先于父"的确定性拓扑序（同层按父节点
  分组、按索引稳定排序）；
- **map 表**（`map_gen.generate_map_tables`）：对每个子节点，其 update 变量是
  `front_indices[child][pivot_dim:]`；在父 front 中查出这些变量各自的局部位置：

  ```text
  row_map[i] = i                      # 子 update 块内的行索引（恒等）
  col_map[i] = 父 front 中该变量的局部列索引
  ```

  装配语义：`parent_front[col_map[i], col_map[j]] += child_schur[row_map[i], row_map[j]]`。

## 8. 输出与验证锚点

`run_symbolic_analysis` 产出：`permutation(q)`、`parent`、`supernodes`、
`node_ranges`、`front_indices`；`map_tables` 由 map_gen 生成；全部进入编译器与
manifest。

测试：`tests/test_ordering.py`、`tests/test_fill_symbolic.py`、
`tests/test_supernode.py`、`tests/test_map_table.py`；
变换后重算结构的端到端验证见 `tests/test_transform_semantics.py`
（同一输入不同匹配 ⇒ 不同符号结构，且必须重算）。

## 9. 边界

- 排序/填充是**保守**的：包络取并集，可能比真实数值填充多，但不遗漏；
- 结构只对"换行限制在 pivot block 内"的策略有效；若未来引入跨 front 换行或
  动态 front 扩张，符号结构必须重算（当前明确不支持）；
- supernode 合并受 `max_supernode_size` 与 front 容量上限约束。
