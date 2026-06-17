# 软件侧代码说明

本目录是毕业设计中软件侧的当前实现。它的定位不是完整的软件 LU 求解器，而是
**面向硬件加速器的稀疏矩阵前处理与硬件输入生成工具链**。

软件侧完成从输入稀疏矩阵到硬件 DDR 输入数据的生成流程，包括排序、消去树、超节点、
任务描述符、map table、本地贡献量化和 manifest 校验。

## 当前流程

```text
稀疏矩阵
  -> 排序
  -> 消去树
  -> 超节点合并
  -> Front Indices / A_local
  -> S_format 量化
  -> 产物
  -> 硬件所需数据
```

其中 `A_local` 表示每个消去树节点在原始矩阵中的本地前沿贡献。软件侧只负责把
`A_local` 量化并写入 DDR 输入文件；节点装配、node-scale 选择、整数 LU/TRSM/GEMM
和 child update 生成均由硬件侧完成。

## 运行方式

推荐在 `graduation-code/software` 目录下以包方式运行：

```bash
cd graduation-code/software
python -m src.main --mtx example/1024X1024JJ.mat --out out
```

也支持从项目根目录直接运行脚本：

```bash
python graduation-code/software/src/main.py \
  -mtx graduation-code/software/example/1024X1024JJ.mat \
  --out out
```

如果不提供 `--mtx`，程序会根据 `--n`、`--density` 和 `--seed` 生成随机 SPD 矩阵：

```bash
python -m src.main --n 128 --density 0.05 --seed 0 --out out_random
```

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-mtx`, `--mtx` | `None` | 输入 `.mat` 或 MatrixMarket 文件 |
| `--out` | `out` | 输出目录 |
| `--ordering` | `amd` | 排序方法，可选 `amd`、`rcm`、`identity` |
| `--max-supernode-size` | `256` | 超节点最大合并列数 |
| `--effective-bits` | `27` | 量化 mantissa 有效位数，`Q_use = 2^bits - 1` |
| `--clip-percentile` | `100.0` | `A_local` 量化裁剪分位数，默认使用最大绝对值 |
| `--n` | `64` | 随机矩阵维度 |
| `--density` | `0.1` | 随机矩阵稀疏密度 |
| `--seed` | `0` | 随机种子 |

说明：当前 `amd` 是 dependency-free 的简化最小度排序启发式算法，并不是 SuiteSparse AMD
或论文中完整的 quotient-graph AMD 实现。它用于当前软件 pipeline 的功能验证和硬件输入生成。

## 输出产物

运行后会在 `--out` 目录生成：

| 文件 | 作用 |
|---|---|
| `tasks.bin` | 硬件节点任务描述符 `NodeTask` |
| `map_table.bin` | child update 到 parent front 的映射表 |
| `front_q.bin` | 每个 node 的 `A_local` 量化 mantissa，`int32` |
| `front_e.bin` | 每个 node 的 source exponent，`int16` |
| `manifest.json` | 输出元数据、内存布局、量化信息和校验信息 |

`manifest.json` 会在生成后自动校验。当前校验内容包括：

- 输出文件大小是否一致
- `NodeTask` ABI 大小是否一致
- node/task 数量是否一致
- `front_q` / `front_e` 与 `A_local` 维度是否匹配
- DDR 内存区域是否按配置对齐且不重叠
- `map_table.bin` 是否可解码
- task order 是否满足 child-before-parent 的依赖顺序

## 量化职责边界

本项目的量化方案主要面向硬件侧。软件侧只生成硬件执行所需的初始 DDR 输入。

软件侧负责：

- 根据排序和消去树结果确定每个 node 的 `front_indices`
- 从原始矩阵中提取每个 node 的本地贡献 `A_local`
- 将每个 `A_local` 独立量化为 `S_format`
- 写出 `front_q.bin`、`front_e.bin`、`tasks.bin`、`map_table.bin` 和 `manifest.json`

硬件侧负责：

- 读取软件准备的 `A_local` source 和 child update source
- 在父节点装配阶段完成 exponent 对齐和累加
- 根据装配结果选择最终 node-scale
- 执行整数 panel LU、TRSM、GEMM/Schur update
- 生成 child update payload 并写回 DDR

当前软件侧的 `S_format` 为：

```text
x ~= q_x * 2^e_s
q_x: int32 mantissa
e_s: int16 source exponent
```

注意：软件侧不会生成整数 LU 数值结果，也不会生成 child update 数值 payload。

## 符号分析与超节点

当前符号分析流程为：

```text
ordering -> elimination tree -> supernode grouping -> front indices
```

排序方法：

- `amd`：简化最小度排序启发式算法
- `rcm`：Reverse Cuthill-McKee
- `identity`：不重排

超节点合并规则：

- 在当前排序后的矩阵图上，只合并连续列
- 若相邻列满足闭邻接集合相同，则可以合并：

```text
Adj(i) U {i} == Adj(j) U {j}
```

- 合并大小受 `--max-supernode-size` 限制，默认 256

该规则参考 AMD 论文中 indistinguishable variables / supervariables 的描述，但这里作为排序后的
后处理使用，不是完整 AMD 过程中的动态 quotient graph supervariable 检测。

## 代码结构

```text
src/
  config.py                 配置对象
  dataStruct.py             NodeTask、MapTableEntry、内存区域和 ABI 定义
  io.py                     二进制 tasks/map_table/front 数据读写
  main.py                   命令行入口
  matrix_io.py              矩阵加载和随机矩阵生成
  pipeline.py               端到端生成流程
  matrix_compress/
    compress.py             .mat 读取和稀疏格式转换
  memory/
    planner.py              DDR 区域规划
  quant/
    bfp_quant.py            A_local 的 S_format 量化和硬件装配参考模型
  scheduler/
    map_gen.py              child update 到 parent front 的映射表生成
    task_queue.py           child-before-parent 任务顺序生成
  symbolic/
    ordering.py             排序
    etree.py                消去树
    supernode.py            超节点合并和 front_indices 生成
  verify/
    manifest.py             manifest 校验
    metrics.py              基础 residual 指标
```

## 测试

在 `graduation-code/software` 目录运行：

```bash
python -m pytest
```

当前测试覆盖：

- `NodeTask` ABI 编解码
- `map_table` 编解码
- 内存规划不重叠
- 排序输出合法性
- 超节点多列合并与上限控制
- `A_local` 量化与装配参考模型
- 端到端 pipeline 与 manifest 一致性

## 示例结果

在项目根目录运行：

```bash
python graduation-code/software/src/main.py \
  -mtx graduation-code/software/example/1024X1024JJ.mat \
  --out /tmp/software_demo
```

示例输出：

```text
residual_norm: 9.259e-06
nodes: 784, tasks: 784
out_dir: /tmp/software_demo
```

`residual_norm` 当前是对输入矩阵进行高精度参考求解得到的 sanity check，不代表硬件整数 LU
执行结果。
