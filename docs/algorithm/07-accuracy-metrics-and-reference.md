# 07 精度指标与参考校验

代码：`software/src/verify/metrics.py`、`software/src/verify/front_reference.py`、
`systemc/reference/`（独立 C++ FP64 参考）

## 1. 指标定义

对原始系统 `A x = b`、近似解 `x̂`：

```text
残差            r = b - A x̂
残差范数        ‖r‖₂
分量后向误差    max_i |r_i| / ( |b_i| + (|A| |x̂|)_i )
相对解误差      ‖x̂ - x_true‖ / ‖x_true‖        （有 golden 解时）
相对残差        ‖r‖ / (‖A‖·‖x̂‖ + ‖b‖)          （C++ 参考口径）
```

- **为什么用分量后向误差**：它回答"是否解了一个与 `A` 每个元素逐分量接近的系统"，
  对行尺度差异不敏感，是 LAPACK 风格的可信口径；纯残差会被矩阵范数/条件数误导。
- 分母为零时：分子也为零记 0；分子非零记 `inf`。
- 所有指标最终对**原始** `A x = b` 计算——变换（04）只用于构造设备数据，不改变
  "问题是什么"。

## 2. golden 解的两种来源

| RHS 来源 | golden 解 | 是否求解 | 条件 |
| --- | --- | --- | --- |
| 生成 RHS（默认，`rhs = A·x_true`） | `x_true` 直接已知 | 不需要 | 始终可写出 |
| 外部 RHS 文件 | `spsolve(A, b)` | **是** | 仅当显式 `reference_check=true` |

**设计原则**：正常编译不以数值求解成功为前提。

- 整树 FP64 前树参考是**可选**预检（`PipelineConfig.reference_check` / CLI
  `--reference-check`），默认关闭；
- 外部 RHS 的 golden 参考解（一次完整 `spsolve`）同样只在 `reference_check` 下
  执行，否则参考解文件与指标**缺省**（manifest 显式记录 `performed=false`、
  文件为 `null`）；
- 这样可以编译"设备端应当主元失败"的合法输入，也不会因参考求解失败阻塞编译。

## 3. 两级参考实现

```text
host Python FP64 前树参考  —— 编译期预检（可选）+ checker
C++ FP64 独立参考          —— 独立实现，可脱离 Python 复现数值
```

- host 参考（`front_reference.py`）：与设备契约同语义（STATIC/DYNAMIC_PANEL、
  相对阈值、块内置换范围），但**从不**向设备 image 注入 pivot/factor/solution；
- C++ 参考（`systemc/reference/`）：
  - `panel_lu`：通用稠密部分主元 LU（update 行可提升）——完整数学能力，
    仅作**独立求解对照**，不是设备契约；
  - `panel_lu_policy` + `factor_front_tree` + `solve_front_tree`：设备契约语义；
  - `fp64_reference_smoke` 支持 `generic | static | dynamic_panel` 三种模式。

## 4. 交叉验证（同策略整树）

共享 fixture `graduation-code/fixtures/reference_tree/tree_policy_cases.json`
（schema `fp64_reference_tree_v1`）固定以下内容的期望值：

- 装配（子 Schur → 父 front）、主元策略与阈值行为、局部 P-vector 方向、
  树形求解结果 `x_hat`、失败分类（kind/column）。

Python（`tests/test_reference_cross.py`）与 C++（`fp64_reference_tests`）必须
**各自独立**复现同一组期望值；任一侧语义漂移都会失败。生成器
`generate_fixture.py` 只负责产出期望值，重新生成后必须重跑两侧套件。

## 5. 数值冒烟（历史记录，非硬件结论）

256/576/1024 阶真实矩阵（`example/*JJ.mat`，默认 matching+STATIC）：

| 阶数 | 节点/最大 front | 命令/描述符 | image | host 参考残差 / 后向误差 |
| --- | --- | --- | --- | --- |
| 256 | 69 / 75 | 818 / 1582 | 1,472,984 B | 1.56e-10 / 6.65e-14 |
| 576 | 134 / 166 | 1598 / 3077 | 10,356,400 B | 1.00e-7 / 3.77e-12 |
| 1024 | 触发容量上限（front 264 > 256），如实报告；显式 300 后 212 / 298 | 2534 / 4871 | 21,957,072 B | 1.23e-5 / 8.3e-15 |

独立 C++ FP64 参考对原始 `A,b`（`fp64_reference_smoke generic`）：n=256 相对残差
3.1e-17 / 后向 1.0e-12 / 解误差 1.6e-7；n=576 5.0e-17 / 1.7e-11 / 5.6e-5；
n=1024 2.8e-17 / 1.6e-6 / 2.4e-6。

**限制声明**：以上全部是 **FP64 参考路径**的验证；FP32 设备路径的数值稳定性
尚未完整验证（等待 T04/T05 执行器闭环）。不依据算法操作数或局部模块测试宣称
任何硬件加速比。

## 6. 误差传播视角

- `D_r`/`P_match`/`Q` 都是精确或条件数中性的操作（01/02/04）；
- 真正的误差源是：FP64→FP32 转换（相对 `2^-24`）、front 消元的舍入累积
  （受 pivot growth 影响）、以及 thresold 拒绝后的报告路径；
- 因此 STATIC 路径的正确性论证分两层：**结构正确**（符号分析精确、装配恒等，
  见 03/05）与**数值健康**（主元检查 + 相对阈值，见 06）。本次只完成了 FP64
  层的数值验证。

## 7. 验证锚点

- 指标定义与用法：`tests/` 中 e2e/transform 用例的残差断言；C++ 参考测试的
  `compute_accuracy_metrics`；
- 确定性：同输入同配置下 manifest / image / reference 文件字节级一致
  （`test_command_artifacts_are_reproducible` 等）。
