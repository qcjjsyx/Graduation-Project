# 软件—SystemC Artifact ABI v2 与接口迁移说明

状态：当前实现接口；迁移阶段保留

## 1. 重要定位

ABI v2 是当前软件工具链和 SystemC `system_sim` 实际使用的 artifact 接口。它不是最终的 command stream 接口，也不应继续通过增加字段来承载新的硬件语义。

目标接口见：

[`docs/interface/Command_Stream_and_Descriptor.md`](../../../docs/interface/Command_Stream_and_Descriptor.md)

T01 冻结的固定二进制布局见：

[`docs/interface/Command_Schema_v1.md`](../../../docs/interface/Command_Schema_v1.md)

迁移原则：

```text
现有 NodeTask/manifest v2
          ↓ adapter
command ring + descriptor table
          ↓
SystemC 或 RTL executor
```

在 command executor 完成前，ABI v2 仍然是可运行回归路径；command stream 通过后，停止扩张 `NodeTask`。

## 2. 基本约定

- artifact 版本：`2`；
- 字节序：little-endian；
- DDR 地址：相对于 `memory_image.bin` 首字节的 64 位字节地址；
- 区域起始地址：按 manifest 的 `config.memory.alignment` 对齐，默认 64 字节；
- `NodeTask`：固定 128 字节；
- 整数：二进制补码；
- 矩阵：行优先连续存储。

## 3. 当前 NodeTask 布局

| 偏移 | 类型 | 字段 | 含义 |
|---:|---|---|---|
| 0 | `u16` | `node_id` | 稠密节点编号 |
| 2 | `u16` | `flags` | bit0=leaf，bit1=root |
| 4 | `u16` | `parent_id` | 根节点为 `0xFFFF` |
| 6 | `u16` | `children_count` | 直接子节点数 |
| 8 | `u32` | `total_dim` | front 总维度 |
| 12 | `u32` | `pivot_dim` | 本节点消去维度 |
| 16 | `u32` | `tile_count` | `ceil(pivot_dim/16)` |
| 20 | `u32` | `tail_dim` | 最后 tile 的有效维度 |
| 24 | `u32` | `map_table_bytes` | map table 字节数 |
| 28 | `u32` | `reserved` | 必须为 0 |
| 32 | `u64` | `front_q_addr` | 软件输入 mantissa |
| 40 | `u64` | `front_e_addr` | 软件输入 exponent |
| 48 | `u64` | `update_q_addr` | child update mantissa |
| 56 | `u64` | `update_e_addr` | child update exponent |
| 64 | `u64` | `map_table_addr` | child update scatter map |
| 72 | `u64` | `l_factor_addr` | L 因子区域 |
| 80 | `u64` | `u_factor_addr` | U 因子区域 |
| 88 | `u64` | `p_vector_addr` | P-vector 区域 |
| 96 | `u64` | `node_meta_addr` | 节点状态和旧 exponent |
| 104 | `u64` | `solve_workspace_addr` | 求解工作区 |
| 112 | `u64` | `reserved_addr0` | 必须为 0 |
| 120 | `u64` | `reserved_addr1` | 必须为 0 |

Python 定义位于 `software/src/dataStruct.py`，C++ 编解码位于 `systemc/include/node_task_codec.hpp`。

## 4. 符号和数值语义

原始数值矩阵可以非对称。软件保留原始 `A`，符号分析使用：

```text
pattern(A) ∪ pattern(Aᵀ)
```

manifest 至少记录：

- `matrix.structurally_symmetric`：原始输入模式是否对称；
- `symbolic.pattern_source`：必须是 `union_of_A_and_transpose_nonzero_patterns`；
- `symbolic.pattern_structurally_symmetric`：符号包络是否对称。

并集在丢弃数值之后进行，不是数值 `A+Aᵀ`。SystemC 启动时校验 manifest、镜像大小、地址范围、区域重叠、任务、森林和 map table。

## 5. 当前 v2 Map table

```text
u32 entry_count
repeat entry_count:
  u32 child_id
  u32 row_count
  u32 col_count
  u32 row_map[row_count]
  u32 col_map[col_count]
```

当前 v2 要求 `row_count == col_count == child_update_dim`。`row_map` 必须覆盖 child update 的局部行，`col_map` 给出对应变量在 parent front 中的局部位置。

## 6. 当前 v2 数值区域

当前 artifact 仍包含历史定点区域：

- `front_q/update_q/U`：int32 mantissa；
- `front_e/update_e/U exponent`：int16；
- `L`：int32 QF-format；
- RHS：int32 mantissa + exponent；
- solution：int64 mantissa + exponent，或 SystemC 求精后的 FP64 扩展写回。

这些字段只描述现有 v2 运行路径，不定义新架构的全局数值格式。

新架构的主路径是：

```text
FP32/FP64 front、Panel-LU、TRSM、solve
INT32 仅作为 GEMM tile 的局部输入格式
```

GEMM 的 `scale_a/scale_b` 应进入目标 `ScaleDesc`，不再扩展 `NodeTask` 的全局 exponent 字段。

## 7. 当前黄金文件边界

以下文件只供 Host checker、调试和结果对照：

- `original_matrix_f64.bin`；
- `original_rhs_f64.bin`；
- `row_scale_e.bin`；
- `reference_front_f64.bin`；
- `rhs_f64.bin`；
- `x_reference_f64.bin`。

它们不是设备计算输入。precision rescue 或 FP64 checker 不得通过这些文件绕过设备的 DDR、assembly、量化和写回路径。

## 8. 当前区域所有权

软件预装：

- task queue；
- permutation；
- RHS；
- 每节点当前 v2 front 数据；
- map table。

SystemC/硬件写回：

- L/U/P-vector；
- child update；
- node metadata；
- solve workspace；
- solution。

`final_memory_image.bin` 是 SystemC 运行结束后的 DDR 快照，供跨语言调试。Host checker 可以恢复 ordering，但不能向设备路径回写修正值。

## 9. 迁移时的目标映射

| v2 字段/对象 | 目标对象 | 处理 |
|---|---|---|
| `NodeTask` | `NODE_BEGIN`、`FrontDesc` | 由 compiler 转换 |
| `front_q/front_e_addr` | `FrontDesc` + data region | 保留地址语义，改为 descriptor 引用 |
| `map_table_addr` | `ContributionDesc` | 独立描述 child contribution |
| `l_factor_addr/u_factor_addr` | `FactorDesc` | 描述因子布局和写回 |
| `solve_workspace_addr` | `KernelDesc`/solve descriptor | 不继续扩张 NodeTask |
| `children_count/parent_id` | wait/signal token | 由 GCU scoreboard 管理 |
| `front_e/update_e` | 旧 v2 compatibility 或 `ScaleDesc` | 新 GEMM scale 不传播为全局 front 格式 |
| reserved 字段 | command/descriptor version | 不能直接重解释旧字段 |

## 10. 迁移验收

- [ ] 同一内部 front graph 能生成 v2 artifact 和 command stream；
- [ ] 两条路径的节点顺序、front/map 语义一致；
- [ ] command executor 能拒绝损坏 descriptor；
- [ ] command executor 不读取 Host 指针；
- [ ] SystemC device path 不读取黄金 front；
- [ ] v2 regression 继续通过；
- [ ] command path 通过单节点、多节点、多根和 numeric failure 测试；
- [ ] 通过后停止增加 v2 `NodeTask` 字段。
