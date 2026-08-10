# 软件—SystemC 产物 ABI v2

> 本文保留 128 字节 NodeTask 和基础 v2 说明。`optimization-v2-tile-bfp`
> 对 `front_e/update_e/node_meta` payload 的扩展以
> [../../docs/TILE_BFP_ABI.md](../../docs/TILE_BFP_ABI.md) 为准。

ABI v2 是当前软件工具链与 SystemC 模型之间唯一受支持的接口。v1 不兼容，也不继续读取。

## 基本约定

- 版本：`2`
- 字节序：little-endian
- DDR 地址：从 `memory_image.bin` 首字节开始的 64 位字节地址
- 区域起始地址：默认按 64 字节对齐，以 manifest 的 `config.memory.alignment` 为准
- `NodeTask`：固定 128 字节
- 整数：二进制补码
- 矩阵：行优先连续存储

SystemC 启动时会同时校验 manifest、镜像大小、地址范围、区域重叠、任务字段、消去森林和
map table。任何一项不一致都会在执行任务前终止。

## NodeTask

| 偏移 | 类型 | 字段 | 含义 |
|---:|---|---|---|
| 0 | `u16` | `node_id` | 稠密节点编号 |
| 2 | `u16` | `flags` | bit0=leaf，bit1=root |
| 4 | `u16` | `parent_id` | 根节点使用 `0xFFFF` |
| 6 | `u16` | `children_count` | 直接子节点数 |
| 8 | `u32` | `total_dim` | front 总维度 |
| 12 | `u32` | `pivot_dim` | 本节点消去维度 |
| 16 | `u32` | `tile_count` | `ceil(pivot_dim/16)` |
| 20 | `u32` | `tail_dim` | 最后一个 16×16 tile 的有效维度 |
| 24 | `u32` | `map_table_bytes` | 本节点 map table 字节数 |
| 28 | `u32` | `reserved` | 必须为 0 |
| 32 | `u64` | `front_q_addr` | 软件输入 mantissa |
| 40 | `u64` | `front_e_addr` | 软件输入 exponent |
| 48 | `u64` | `update_q_addr` | 硬件写回 child update mantissa |
| 56 | `u64` | `update_e_addr` | 硬件写回 child update exponent |
| 64 | `u64` | `map_table_addr` | child update scatter map |
| 72 | `u64` | `l_factor_addr` | `total_dim × pivot_dim` int32 QF |
| 80 | `u64` | `u_factor_addr` | `pivot_dim × total_dim` int32 M |
| 88 | `u64` | `p_vector_addr` | `pivot_dim` 个 u16 |
| 96 | `u64` | `node_meta_addr` | 节点状态与 exponent |
| 104 | `u64` | `solve_workspace_addr` | 节点求解工作区 |
| 112 | `u64` | `reserved_addr0` | 必须为 0 |
| 120 | `u64` | `reserved_addr1` | 必须为 0 |

Python 定义位于 `software/src/dataStruct.py`，C++ 编解码位于
`systemc/include/node_task_codec.hpp`。

## Map table

每个节点的 map table 独立存储：

```text
u32 entry_count
repeat entry_count:
  u32 child_id
  u32 row_count
  u32 col_count
  u32 row_map[row_count]
  u32 col_map[col_count]
```

当前 v2 要求 `row_count == col_count == child_update_dim`。`row_map` 必须完整覆盖
`0..child_update_dim-1`；`col_map` 给出对应变量在 parent front 中的局部位置。

## 数值格式

- `front_q/update_q/U`：int32 M-format mantissa
- `front_e/update_e/U exponent`：int16，实际值为 `q × 2^e`
- `L`：int32 QF-format，默认 `F=20`，实际值为 `q × 2^-F`
- RHS 输入：全局 int32 mantissa 与一个 int16 exponent
- solution DDR 输出：排序坐标下每变量 int64 mantissa；每节点一个 int16 exponent
- `U` 与 child `update` 允许使用不同 exponent：U exponent 在 `node_meta[2:4]`，
  update exponent写在 `update_e`，同时镜像到 `node_meta[4:6]`
- FP64 front、RHS、原始矩阵/RHS 和参考解是黄金验证旁路文件，不属于硬件 DDR 输入

软件输入源默认保存 30 个有效位；SystemC 的 `q_use_bits` 是 QAU/因子写回的目标有效位，
默认 26。二者不需要相同，具体值分别记录在 manifest 和仿真配置中。

### node metadata

当前每节点预留 64 字节，定点写回使用前 6 字节：

| 偏移 | 类型 | 含义 |
|---:|---|---|
| 0 | `u8` | factor valid，成功写回时为 1 |
| 1 | `u8` | precision-rescue 标志 |
| 2 | `i16` | U exponent，小端 |
| 4 | `i16` | child update exponent，小端 |
| 6..63 | bytes | 保留，写 0 |

`update_e` 是父节点装配时读取的权威 update exponent；metadata 镜像用于结果检查和调试。

### 行均衡旁路

manifest 的 `equilibration` 段描述
`D_r A x = D_r b`，其中 `D_r[i,i]=2^row_scale_e[i]`。
`row_scale_e.bin` 是原始行编号顺序的 int16 数组。当前只有行缩放，因此未知量 `x`
不需要反缩放。下列文件仅供 Host Checker 使用：

- `original_matrix_f64.bin`：原始坐标、未均衡的稠密矩阵；
- `original_rhs_f64.bin`：原始坐标、未均衡 RHS；
- `row_scale_e.bin`：行缩放 exponent；
- `reference_front_f64.bin`、`rhs_f64.bin`：均衡且排序后的黄金数据；
- `x_reference_f64.bin`：排序后的参考解。

precision rescue 也不允许读取 `reference_front_f64.bin`。它只能把已经由
`front_q/front_e` 与 child `update_q/update_e` 装配出的定点 front 还原为 double，
因此无法绕过 DDR 损坏、装配错误或源量化误差。

### solution 区域的 SystemC 扩展

未执行迭代求精时，`solution_q` 仍是 int64 mantissa，`solution_e` 是每节点 V-format
exponent。若固定点求解实际接受了至少一轮 FP64 累加的混合精度修正，SystemC 会将
最终排序坐标解以 little-endian FP64 写入同样的 8-byte/element `solution_q` 区域，并
把全部 `solution_e` 写 0；`summary.json.solve.fixed.refined_solution=true` 是类型判别
依据。这一扩展用于系统级研究和快照调试，并不是当前 RTL 输出协议。

## 区域所有权

软件预装：

- task queue、permutation
- RHS mantissa/exponent
- 每节点 `front_q/front_e`
- 每节点 map table

SystemC/硬件写回：

- `L/U/P-vector`
- `update_q/update_e`
- node metadata、solve workspace
- solution mantissa/exponent

`final_memory_image.bin` 是一次 SystemC 运行结束后的完整 DDR 快照，便于跨语言逐字节调试。
硬件 DDR 保持排序坐标；Host Checker 会恢复 ordering，`solution.csv` 按原始变量编号输出。

ABI 的 128 字节 `NodeTask` 布局仍为 v2，没有因为稳定性改造发生字段级变更；新增的行均衡
文件属于 manifest 黄金验证扩展，分离的 U/update exponent 使用原有 `node_meta` 和
`update_e` 区域。
