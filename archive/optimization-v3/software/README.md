# 软件侧符号分析与 SystemC 产物生成

> 本文件属于 `optimization-v3-adaptive-precision` 独立副本。软件制品格式继承
> v2，默认使用 `pow2-row-column` 与 16×16 tile BFP；基线、v1 和冻结 v2
> 均不受影响。

软件侧负责把结构对称的稀疏矩阵转换为 ABI v2 DDR 镜像。数值可以非对称；若稀疏结构
不对称会明确拒绝。SystemC 只依赖 `manifest.json` 和 `memory_image.bin`，其余独立
二进制文件用于调试、黄金验证和跨语言测试。

## 当前流程

```text
矩阵与 RHS
  -> 2 的整数次幂行均衡 D_r A x = D_r b
  -> ordering
  -> 显式消元图 fill-in
  -> elimination forest / supernode / front
  -> 唯一 A_local 归属
  -> child update map
  -> 16×16 tile BFP 量化
  -> ABI v2 地址规划
  -> memory_image.bin + manifest
```

每个原始元素只在最早消去它的节点中出现。`A_local` 的 `F11/F12/F21` 来自原矩阵，
`F22` 初始化为零，避免跨 front 重复装配。多根消去森林会原样保留。

默认使用稀疏结构不变的 2 的整数次幂行均衡。第 `i` 行的缩放为
`2^row_scale_e[i]`，因此硬件只需要调整 exponent，不需要增加浮点乘法器。该变换只缩放
方程、不缩放未知量，所以求得的 `x` 无需反变换。软件同时保留原始 `A` 和 `b`，最终
正确性始终按原方程 `A*x=b` 计算，而不是只检查均衡后的方程。

## 运行

```bash
cd graduation-code/optimization-v3-adaptive-precision/software
python -m src.main \
  -mtx example/256X256JJ.mat \
  --rhs example/256fuv.mat \
  --ordering amd \
  --bfp-tile-size 16 \
  --out /tmp/mf-256
```

未提供 `--rhs` 时，程序使用固定 `--rhs-seed` 生成 `x_true`，再计算 `b=A*x_true`：

```bash
python -m src.main \
  --n 32 --density 0.1 --seed 3 --rhs-seed 11 \
  --out /tmp/mf-random
```

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `-mtx`, `--mtx` | 无 | `.mat` 或 MatrixMarket 输入 |
| `--rhs` | 无 | 可选 `.mat`/`.npy` RHS |
| `--rhs-seed` | 0 | 生成参考解/RHS 的 seed |
| `--ordering` | `amd` | `amd`、`rcm` 或 `identity` |
| `--max-supernode-size` | 256 | 最大 pivot supernode |
| `--effective-bits` | 30 | 软件输入源量化有效位 |
| `--clip-percentile` | 100 | 输入源裁剪分位数 |
| `--bfp-tile-size` | 16 | `16` 为 tile exponent，`0` 为标量回归模式 |
| `--equilibrate` | `pow2-row-column` | `none`、`pow2-row`、`pow2-row-column` 或 `pow2-ruiz` |
| `--max-scale-exponent` | 60 | 行缩放 exponent 的绝对值上限 |
| `--equilibration-iterations` | 4 | 行列均衡迭代轮数 |

这里的 `amd` 是仓库内的确定性最小度启发式实现，不等同于 SuiteSparse AMD。

## 输出

硬件/SystemC 主入口：

| 文件 | 内容 |
|---|---|
| `manifest.json` | ABI、符号结构、量化信息、地址和文件索引 |
| `memory_image.bin` | 与地址规划完全一致的完整 DDR 镜像 |

调试镜像：

| 文件 | 内容 |
|---|---|
| `tasks.bin` | 128 字节 ABI v2 NodeTask 队列 |
| `map_table.bin` | child update 到 parent front 的映射 |
| `front_q.bin` / `front_e.bin` | 软件量化的唯一 A_local |
| `rhs_q.bin` / `rhs_e.bin` | 定点 RHS |

黄金验证旁路：

| 文件 | 内容 |
|---|---|
| `reference_front_f64.bin` | 均衡后 FP64 A_local |
| `rhs_f64.bin` | 均衡且排序后的 FP64 RHS |
| `x_reference_f64.bin` | 排序后的参考解 |
| `original_matrix_f64.bin` | 原始坐标、未均衡的稠密 FP64 矩阵 |
| `original_rhs_f64.bin` | 原始坐标、未均衡的 FP64 RHS |
| `original_solution_f64.bin` | 原始坐标参考解 |
| `row_scale_e.bin` | 原始行编号顺序的 int16 行缩放 exponent |
| `column_scale_e.bin` | 原始列编号顺序的 int16 列缩放 exponent |

行列模式求解 `D_r A D_c y=D_r b`。`x_reference_f64.bin` 保存排序后的 `y`，Host
Checker 使用 `column_scale_e.bin` 恢复 `x=D_c y`。

黄金文件不属于硬件 DDR 输入。

## ABI 与校验

ABI v2 使用小端 128 字节 `NodeTask`，不兼容 v1。软件生成结束后会校验：

- 实际矩阵维度、任务数和文件大小；
- task order、parent/children/flags 和 tile metadata；
- 所有 DDR 地址的对齐、边界和不重叠；
- map table 的父子关系和 child update 完整覆盖；
- `memory_image.bin` 与独立调试文件逐区域一致；
- RHS、参考 front、参考解、原始矩阵/RHS 和行缩放 exponent 长度；
- 行均衡方程、模式及“解不需要反缩放”语义。

完整布局见 [../systemc/docs/ABI_v2.md](../systemc/docs/ABI_v2.md)。

## 测试

```bash
cd graduation-code/software
python -m pytest -q
```

当前 32 个测试覆盖 ABI、map、地址规划、ordering、fill/supernode、量化、三种
power-of-two 均衡、解反缩放、原方程验证、产物可复现性和端到端产物。
SystemC 的 CTest 会再次调用此工具链生成真实产物，并进行 C++ 侧独立校验。

## 当前规模

当前 ordering 与 supernode 规则下：

| 矩阵 | 节点数 | 最大 pivot | 最大 front |
|---|---:|---:|---:|
| 256JJ | 73 | 72 | 72 |
| 576JJ | 157 | 156 | 156 |
| 1024JJ | 274 | 256 | 272 |

这也是 SystemC HPU 保留 256 个候选、ATU 使用 9 bit 行索引的依据。
