# 软件侧：稀疏 LU 命令编译器

把稀疏矩阵和 RHS 编译成设备可读的 Command/Descriptor schema v2 artifact。
软件**不执行**设备侧的 Panel-LU、TRSM、GEMM，也不向设备镜像写任何参考结果。

完整职责、数据契约、配置、产物与故障处理见
[软件侧技术文档](../../docs/software/软件侧技术文档.md)。

```text
A、b
  → 可选二的幂行均衡 D_r                               equilibration.py
  → 加权最大乘积匹配行置换 P_match（默认开启）            matching.py
  → B = P_match D_r A；pattern(B) ∪ pattern(Bᵀ)          symbolic/pattern.py
  → ordering Q（amd / rcm / identity）→ fill / 消除森林    symbolic/ordering.py、symbolic/fill.py
  → supernode → front_indices → child→parent map          symbolic/supernode.py、scheduler/map_gen.py
  → Command + Descriptor + payload + FP32 数据 + Completion  scheduler/command_compiler.py、command_codec.py
  → manifest.json + memory_image.bin（并由 verify/manifest.py 复核）
```

总流程在 `src/pipeline.py`，命令行入口在 `src/main.py`。变换约定统一为 gather：`y[i] = x[p[i]]`，
`B = P_match D_r A`、`A_hat = Qᵀ B Q`、`b_hat = Qᵀ P_match D_r b`、`x = Q x_hat`（见 [软件侧技术文档 §3.2](../../docs/software/软件侧技术文档.md#32-坐标变换与置换约定)）。

## 运行

```bash
cd graduation-code/software
python -m src.main --mtx example/256X256JJ.mat --out /tmp/mf-art       # 真实矩阵
python -m src.main --n 128 --density 0.03 --out /tmp/mf-art             # 随机对角占优矩阵
```

常用选项：

| 选项 | 默认 | 说明 |
|---|---|---|
| `--mtx PATH` / `--n --density --seed` | 随机矩阵 | `.mat` 或 Matrix Market 方阵 |
| `--rhs PATH` | 由随机 `x_true` 生成 | 外部 RHS（`.mat` / `.npy`） |
| `--ordering` | `amd` | `amd` / `rcm` / `identity` |
| `--matching` | `weighted_matching` | `none` 用于消融 |
| `--pivot-policy` | `STATIC` | `DYNAMIC_PANEL` 可编译，但设备会拒绝执行 |
| `--pivot-threshold` | `0.0` | 相对 front 最大元的主元阈值；0 只拦截零主元（真实矩阵的数值问题见 [真实矩阵数值分析](../../experiments/real_matrix_diagnosis/实验报告.md)） |
| `--gemm-backend` | `FP32_SYSTEMC` | `FP32_SYSTEMC_INT32_GEMM` 只把顶层 GEMM 换成局部 INT32 |
| `--device-buffer-bytes` | `65536` | 目标设备 scratchpad 槽大小；每个 FP32 front 必须满足 `4·n² ≤` 该值（默认即 front ≤ 128），solve 暂存 `4·(2n+2n+p) ≤` 该值，否则编译报错 |
| `--max-front-size` / `--max-supernode-size` | 256 / 256 | 结构上限 |
| `--factorization-only` / `--emit-solve-commands` | 默认 `factorization_and_solve` | CLI 默认生成完整分解+求解链（T12 设备可直接执行）；`--factorization-only` 生成纯分解 artifact，`--emit-solve-commands` 保留为兼容别名 |
| `--reference-check` | 关闭 | 编译时跑 host FP64 前树参考，输出原方程残差/后向误差，失败则拒绝产出 |

编译错误（例如 front 超过设备容量）以 `error: ...` 输出并返回码 2。

## 产物

```text
manifest.json            布局、命令/描述符元数据、变换、配置（严格 JSON）
memory_image.bin         设备唯一输入：命令、描述符、completion 槽、FP32 front/RHS、INT32 P-vector
front_f64.bin            变换后各节点 local front（FP64，仅 checker 使用）
rhs_hat_f64.bin          b_hat（Q 序变量空间）
original_matrix_f64.bin  原始 A（稠密 FP64，仅验证用）
original_rhs_f64.bin     原始 b
row_scale_e.bin          行均衡指数（int16）
x_true_f64.bin           golden 解（原始变量顺序；外部 RHS 且未开 --reference-check 时缺省）
x_hat_f64.bin            host 参考解（仅 --reference-check）
```

设备运行只需要 `manifest.json` + `memory_image.bin`；FP64 文件只给验证程序使用。
`factorization_and_solve` artifact 的 manifest 额外声明
`abi.solve_desc_payload = "front_variable_indices_u32"` 与 `solve` 段
（累加器语义、`rhs_data` 只读、workspace 行、拓扑序链、每节点求解暂存上界）。

## 测试

```bash
python -m pytest -q
```

最新测试规模和结果统一见 [当前进度](../../当前进度.md)。

与 C++ 参考的同策略整树交叉验证使用共享 fixture `../fixtures/reference_tree/`
（`tests/test_reference_cross.py` 与 C++ `fp64_reference_tests`）。
T12 追加的测试面：`SOLVE_DESC` payload 的 codec/负面用例、求解链拓扑序与
有界 wait list（每条求解命令只等本节点 commit + 上一条，2026-10-11 去掉全局求解门；
节点数 > 256 的回归用例）、求解暂存容量检查、CLI 默认 scope、§1.3 累加器式 FP64 参考
（`verify/front_reference.solve_front_tree_accumulator`）与数组式参考逐步一致。

## 源码一览

| 文件 | 作用 |
|---|---|
| `main.py` / `pipeline.py` / `config.py` | CLI、总流程、全部配置与校验 |
| `matrix_io.py` | 读取 `.mat` / Matrix Market / 随机矩阵与 RHS |
| `equilibration.py`、`matching.py` | 数值预处理 |
| `symbolic/` | 对称包络、ordering、fill 与消除森林、supernode / front |
| `scheduler/map_gen.py` | child update → parent front 位置映射 |
| `scheduler/task_queue.py` | 子先父后的拓扑序（编译器、参考、校验共用） |
| `scheduler/command_compiler.py` | 命令/描述符/内存布局/manifest 生成 |
| `command_codec.py` | schema v2 二进制编解码（与 C++ `command_codec.hpp` 对应） |
| `memory/planner.py` | 区域对齐与 u64 溢出检查 |
| `verify/front_reference.py`、`verify/metrics.py` | FP64 前树参考与误差指标 |
| `verify/manifest.py` | artifact 独立复核 |
| `io.py`、`dataStruct.py` | manifest 写出、编译 IR 数据结构 |
| `tools/numeric_diagnosis.py` | host 侧数值诊断：条件数、逐节点后向误差、FP32 因子求解、设备 `solution_data` 与 host 用设备因子求解的对比、FP64 迭代求精 |

旧 NodeTask ABI v2 软件位于 `archive/legacy-node-task-abi-v2/software/`，不属于当前构建与测试。
