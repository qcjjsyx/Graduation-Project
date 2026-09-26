# 04 变换语义与置换约定

代码：`graduation-code/software/src/pipeline.py`（主链）· 验证：`tests/test_transform_semantics.py`

## 1. 为什么要单独定义"约定"

本轮出现过"把行置换错误地应用到解上"这类方向性缺陷的审查项，因此把置换方向
提升为显式契约：所有数组使用同一个 **gather 约定**，全链路一致。

```text
gather： y[i] = x[p[i]]      "新位置 i 取自旧位置 p[i]"
矩阵行置换： B = P A  ⇔  B[i, :] = A[p[i], :]
矩阵双侧置换：A_hat = Q^T B Q  ⇔  A_hat[i, j] = B[q[i], q[j]]
```

实现接口：`gather_rows(matrix, p)`、`apply_permutation(matrix, q)`（`matrix[q][:, q]`）。

## 2. 变换公式

保留行缩放 `D_r`（01）与匹配行置换 `P_match`（02），结构排序为 `Q`（03）：

```text
B        = P_match · D_r · A
A_hat    = Q^T · B · Q
b_hat    = Q^T · P_match · D_r · b
x        = Q · x_hat
```

在 gather 约定下逐元素展开：

```text
B[i, :]       = (D_r A)[p_match[i], :]
A_hat[i, j]   = B[q[i], q[j]]
b_hat[i]      = (D_r b)[p_match[q[i]]]      ← 组合映射 rhs_row_map
x_hat[i]      = x[q[i]]
x[q[i]]       = x_hat[i]                    ← 解的恢复（scatter）
```

验证一次：`A_hat x_hat` 的第 i 个分量 `= Σ_j B[q[i], q[j]] x_hat[j]`
`= (B x)[q[i]] = (P_match D_r b)[q[i]] = b_hat[i]`，即 `A_hat x_hat = b_hat`。

## 3. 解为什么不需要反缩放

- `D_r` 只缩放**行**（左乘对角阵）⇒ 等价变形，`x` 不变；
- `P_match`、`Q` 都是置换 ⇒ 只改坐标，解在原始变量顺序下不变；
- 因此恢复公式只有 `x[q[i]] = x_hat[i]`，**没有** `D_r^{-1}` 因子
  （manifest 明确 `solution_requires_unscale: false`）。

## 4. 三类置换的职责区分

| 置换 | 作用对象 | 空间 | 存放位置 |
| --- | --- | --- | --- |
| `P_match`（全局） | 矩阵行、RHS | host 编译期 | manifest `preprocess.transform.row_perm_match` |
| `Q`（全局） | 矩阵行列、RHS、解 | host 编译期 | manifest `preprocess.transform.ordering_perm` |
| 设备局部 P-vector | 单个 front 的 pivot block 内换行 | 设备运行期 | 设备 image 的 INT32 region（初值 `-1`） |

**关键区别**：前两者是"编译期全局坐标变换"，后者是"运行期设备局部行交换"。
manifest 里额外记录组合映射 `rhs_row_map[i] = p_match[q[i]]`（从缩放后 RHS 直接
取数），并显式声明三者不是同一个东西。

## 5. 残差与后向误差的目标

所有精度指标最终对**原始系统** `A x = b` 计算：

- 原始 `A`、`b` 写入 reference 文件（`original_matrix_f64.bin`、`original_rhs_f64.bin`）；
- 参考解从 `x_hat` 经 `x[q[i]] = x_hat[i]` 恢复后再代入原始系统；
- 前树参考只改运行数据，**不**把参考 pivot/factor/solution 注入设备 image。

## 6. 方向错误的可检测性

`tests/test_transform_semantics.py` 用"非对称矩阵 + 非恒等匹配 + 非恒等且非对合的
排序"构造测试，验证：

- 恒等式成立：`b_hat == (D_r b)[p_match][q]`、`x_hat ≈ x_true[q]`、前后端 front
  数值等于 `A_hat` 的对应块（逐元素）；
- 错误方向会失败：`(D_r b)[q]`（漏掉匹配）、`(D_r b)[argsort(q)]`（方向反了）、
  `x_true[p_match]`（把行置换用到解上）都会产生明显不同的值/大残差；
- 匹配改变时符号结构必须重算：`q` 等于在 `B` 的包络上重算的结果，而不等于在
  `A` 的包络上算的结果。

## 7. 边界

- 约定是**全项目统一**的：新增代码若使用置换，必须沿用 `y[i] = x[p[i]]`；
- 设备 P-vector 的方向是"pivot block 内的 gather"（第 i 个主元位置取源行
  `p[i]`），语义见 06；
- 不实现列缩放/列置换；若未来引入，需同步扩展本节的公式、manifest 与 checker。
