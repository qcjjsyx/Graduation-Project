# Command / Descriptor 固定格式 v1

状态：T01 已冻结  
适用范围：当前主线软件 Command Compiler、SystemC Command Executor 和未来 RTL GCU  
字节序：little-endian  
地址单位：相对于 `memory_image.bin` 首地址的 64 位字节偏移

## 1. 冻结结论

本版本采用以下固定规格：

| 对象 | 固定大小 | 说明 |
|---|---:|---|
| Command record | 32B | 28B 语义字段 + 4B 保留字段 |
| Descriptor record | 64B | 固定外层记录，变长 map/token 放 payload 区 |
| Completion record | 64B | 固定状态和性能结果 |
| ID/token | u32 | `command_id`、`node_id`、`descriptor_id`、token 统一使用 u32 |
| 地址 | u64 | 字节地址，必须经过范围校验 |
| 默认对齐 | 64B | 当前 Command v1 memory image 的统一约定 |

本版本不包含：

- command record 内的 `schema_version`；
- descriptor record 内的 `version`；
- command record 内的 `record_bytes`；
- 独立的 `WAIT_DEP` 正式执行指令；
- 动态 producer/consumer ring；
- 全局 INT32/BFP scale；
- FP64 golden device backend。

固定大小由 manifest 中的区域信息和本规范共同确定。将来如果改变二进制布局或字段语义，应升级顶层 artifact/command 格式，而不是重新解释当前 v1 的 reserved 字段。

## 2. ABI v2 退役说明

ABI v2 已完成阶段性验证并退出当前主线。历史实现只保存在
`archive/legacy-node-task-abi-v2/`，不再参与构建、测试或产物生成：

```text
symbolic/front/map compiler IR
          ↓ Command Compiler
Command record + Descriptor table + payload + FP32 data
          ↓
SystemC Command Executor / future RTL Executor
```

当前接口直接固定以下约定：

- little-endian；
- 64 位字节地址；
- memory region 起始地址按 64B 对齐；
- 矩阵默认 row-major；
- stride 使用字节数；
- 消除树、front 和 map 只存在于编译器 IR 与 manifest；
- 设备只能通过 command、descriptor、payload 和 memory region 访问这些信息。

## 3. Command record

每条 command 固定 32B，字段如下：

| 偏移 | 大小 | 类型 | 字段 | 语义 |
|---:|---:|---|---|---|
| 0 | 4 | u32 | `opcode` | 宏指令编号 |
| 4 | 4 | u32 | `flags` | 当前 command 行为标志 |
| 8 | 4 | u32 | `command_id` | 本次运行内唯一编号 |
| 12 | 4 | u32 | `node_id` | 所属节点；全局 solve 可使用 `NONE` |
| 16 | 4 | u32 | `descriptor_id` | 主 descriptor；无 descriptor 时为 `NONE` |
| 20 | 4 | u32 | `wait_list_id` | `DependencyDesc` 编号；无依赖时为 `NONE` |
| 24 | 4 | u32 | `signal_token` | 完成后更新的 token；不产生 token 时为 `NONE` |
| 28 | 4 | u32 | `arg0` | v1 保留，必须为 0 |

```text
NONE = 0xFFFFFFFF
COMMAND_RECORD_BYTES = 32
```

### 3.1 Command flags

当前只冻结两个标志：

| bit | 名称 | 语义 |
|---:|---|---|
| 0 | `TRACE_ENABLE` | 要求记录详细 command trace |
| 1 | `ALLOW_RETRY` | 允许 executor 将数值失败报告为可重试状态 |
| 2–31 | reserved | 必须为 0 |

`IN_PLACE`、scale、backend、tile size 和 buffer 语义属于 descriptor，不放在 command flags 中。未知 flags 必须返回 `BAD_COMMAND`，不能静默忽略。

### 3.2 Opcode

| 数值 | 名称 | 完成语义 |
|---:|---|---|
| 0x01 | `NODE_BEGIN` | 注册节点并申请所需资源 |
| 0x02 | `LOAD_FRONT` | 从 DDR region 读取 front 到片上 buffer |
| 0x03 | `ASSEMBLE_EXTEND_ADD` | 读取 child contribution 并装配当前 front |
| 0x04 | `PANEL_LU` | 执行当前 front 的 Panel-LU |
| 0x05 | `TRSM_LEFT` | 执行 `U12 = L11^-1 · A12` |
| 0x06 | `TRSM_RIGHT` | 执行 `L21 = A21 · U11^-1` |
| 0x07 | `GEMM_SCHUR` | 执行 `C = C - A × B` |
| 0x08 | `STORE_FACTOR` | 写回 L/U/P 和 factor metadata |
| 0x09 | `STORE_UPDATE` | 写回 child update 和 update metadata |
| 0x0A | `SOLVE_FORWARD` | 执行 `L y = P b` |
| 0x0B | `SOLVE_BACKWARD` | 执行 `U x = y` |
| 0x0C | `NODE_COMMIT` | 原子提交当前节点并发布结果 token |
| 0x0D | `ABORT_NODE` | 终止节点并将相关 token 标记为失败 |
| 其它 | — | 返回 `BAD_COMMAND` |

不使用独立 `WAIT_DEP`。command 发射前，GCU 自动检查 `wait_list_id` 指定的全部 token。

### 3.3 Command 生命周期

```text
FETCH
  → DECODE
  → CHECK_DESCRIPTOR
  → WAIT_ALL_TOKENS_READY
  → ISSUE
  → RUN
  → COMPLETE
  → SIGNAL_TOKEN
```

如果 wait list 中出现 `FAILED` token，command 不得进入 `RUN`，应直接完成为 `DEPENDENCY_FAILED`，并将自己的 `signal_token` 标记为 `FAILED`。

## 4. Dependency token

token 是本次 command batch 内的 u32 编号，具有三种状态：

```text
UNSIGNALED
READY
FAILED
```

规则：

1. `wait_list_id == NONE` 表示无依赖。
2. 非空 wait list 必须等待其中全部 token 为 `READY`。
3. 任意 token 为 `FAILED` 时，当前 command 完成为 `DEPENDENCY_FAILED`。
4. 一个 token 只能被一个 command signal。
5. command 成功完成后 signal 为 `READY`。
6. command 数值或控制失败后 signal 为 `FAILED`。
7. 未知 token、重复 signal 和 token list 越界必须报错。
8. 失败 token 必须传播，不能让父节点永久等待。

典型父子关系：

```text
child STORE_UPDATE → child_update_token = READY
                  → parent ASSEMBLE_EXTEND_ADD 可以发射

child failure      → child_update_token = FAILED
                  → parent command = DEPENDENCY_FAILED
```

`NODE_COMMIT` 是节点的原子提交点。只有当前节点所有必需 command 成功、factor/update 写回完成后，才允许发布节点完成 token 并释放 buffer。

## 5. Descriptor record

descriptor table 中每项固定 64B。descriptor ID 等于 table index，因此 record 内不重复保存 `descriptor_id`。

| 偏移 | 大小 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 2 | u16 | `descriptor_type` |
| 2 | 2 | u16 | `flags` |
| 4 | 4 | u32 | `reserved`，必须为 0 |
| 8 | 8 | u64 | `payload_offset` |
| 16 | 8 | u64 | `payload_bytes` |
| 24 | 40 | bytes | type-specific body |

```text
DESCRIPTOR_RECORD_BYTES = 64
```

`payload_offset/payload_bytes` 用于存储变长数组，例如 row map、column map 和 token list。payload 仍然是 memory image 内的字节区域，不能使用 Host 指针。

### 5.1 Descriptor 类型

| 数值 | 类型 | 主要语义 |
|---:|---|---|
| 0x01 | `REGION_DESC` | DDR/片上数据区域、尺寸、stride、布局和数据格式 |
| 0x02 | `FRONT_DESC` | front 维度、pivot 维度、front region 和相关 descriptor 引用 |
| 0x03 | `CONTRIBUTION_DESC` | child update 到 parent front 的源/目标及 map |
| 0x04 | `FACTOR_DESC` | L/U/P factor 区域及其维度/格式 |
| 0x05 | `KERNEL_DESC` | Panel-LU、TRSM 或 GEMM 的输入输出和 M/N/K |
| 0x06 | `SOLVE_DESC` | RHS、factor、P-vector、solution 和 solve workspace |
| 0x07 | `SCALE_DESC` | 仅供局部 GEMM quantize/dequantize 使用 |
| 0x08 | `DEPENDENCY_DESC` | wait token 列表 |
| 其它 | — | 返回 `BAD_DESCRIPTOR` |

descriptor body 使用固定 little-endian 字段；变长内容只能放在 payload。descriptor 不携带 version 字段。未知 flags、reserved 非零、payload 越界或 type/body 不匹配必须拒绝。

descriptor 的 type-specific body 占用 offset `24..63` 的 10 个 u32 word，按 `body_word[0]..body_word[9]` 解释。各类型不得改变这 10 个 word 的位置；row map、column map 和 token 数组使用 payload。

### 5.2 REGION_DESC

`REGION_DESC` 的语义字段为：

```text
base_addr       u64
byte_size       u64
row_stride      u32       // 字节
rows            u32
cols            u32
format          u32       // FP64 / FP32 / INT32 / LEGACY_INT32_EXP
layout          u32       // v1 仅支持 ROW_MAJOR
```

区域必须满足：

- `base_addr + byte_size` 不溢出且不超过 `memory_image`；
- 默认 base 按 64B 对齐；
- `row_stride >= cols × element_size`；
- v1 的 layout 为 row-major；
- 未声明 `IN_PLACE` 时，相关输入输出区域不得重叠。

对应 body word：`base_addr` 使用 word 0–1，`byte_size` 使用 word 2–3，`row_stride/rows/cols/format/layout` 使用 word 4–8，word 9 必须为 0。

### 5.3 FRONT_DESC

至少描述：

```text
front_region_id
total_dim
pivot_dim
tile_size
contribution_desc_id / list reference
factor_desc_id
p_vector_desc_id
solve_workspace_desc_id
```

front 的数值格式由其 region descriptor 给出，不由 front descriptor 隐含指定。

对应 body word：`front_region_id/total_dim/pivot_dim/tile_size` 使用 word 0–3，`contribution_desc_id/factor_desc_id/p_vector_desc_id/solve_workspace_desc_id` 使用 word 4–7，word 8–9 必须为 0。

### 5.4 CONTRIBUTION_DESC

至少描述：

```text
source_region_id
target_region_id
child_id
parent_id
row_count
col_count
source_stride
target_stride
row_map / column_map payload
```

row map 和 column map 以 u32 数组放在 payload 区。v1 允许 source/target 为连续 row-major region，但不允许硬件自行推测 map。

对应 body word：`source_region_id/target_region_id/child_id/parent_id` 使用 word 0–3，`row_count/col_count/source_stride/target_stride` 使用 word 4–7，word 8 为数据格式，word 9 必须为 0。

### 5.5 FACTOR_DESC

至少描述：

```text
L_region_id
U_region_id
P_vector_region_id
total_dim
pivot_dim
factor_format
```

L/U/P 写回地址、容量、stride 和布局由对应 `REGION_DESC` 提供。

对应 body word：`L_region_id/U_region_id/P_vector_region_id` 使用 word 0–2，`total_dim/pivot_dim/factor_format` 使用 word 3–5，`solve_workspace_region_id` 使用 word 6，word 7–9 必须为 0。

### 5.6 KERNEL_DESC

至少描述：

```text
A_descriptor_id
B_descriptor_id
C_descriptor_id
M, N, K
backend
format
scale_desc_id
tile_size
```

Panel-LU、TRSM 和 GEMM 的内部微操作、lane、divider、MAC 数量和实际状态机不在 T01 冻结。

对应 body word：`A_descriptor_id/B_descriptor_id/C_descriptor_id` 使用 word 0–2，`M/N/K` 使用 word 3–5，`backend/format/scale_desc_id/tile_size` 使用 word 6–9。

### 5.7 SOLVE_DESC

至少描述：

```text
factor_desc_id
rhs_region_id
solution_region_id
p_vector_region_id
solve_workspace_region_id
rhs_count
direction
format
```

v1 首先支持单 RHS；`rhs_count` 为 1。多 RHS 只保留接口扩展位置。

对应 body word：`factor_desc_id/rhs_region_id/solution_region_id/p_vector_region_id` 使用 word 0–3，`solve_workspace_region_id/rhs_count/direction/format` 使用 word 4–7，word 8–9 必须为 0。

### 5.8 SCALE_DESC

`SCALE_DESC` 只服务于单次 GEMM 调用，不能传播为整棵树的全局 exponent。

至少描述：

```text
scale_a
scale_b
output_scale
rounding_mode
saturation_policy
overflow_policy
```

INT32 量化是否采用二的幂 scale、具体累加宽度和 tile 级策略在 T11 冻结；T01 只冻结引用边界和“不扩散到整棵树”的原则。

`SCALE_DESC` 的核心参数使用 word 0–5，具体布局为 `scale_a/scale_b/output_scale/rounding_mode/saturation_policy/overflow_policy`；word 6–9 必须为 0。具体数值编码和计算策略在 T11 冻结。

### 5.9 DEPENDENCY_DESC

payload 为 u32 token 数组：

```text
token_count
token_ids[token_count]
```

token list 不得越过 descriptor payload 区域，重复 token、未知 token 和超出最大 wait 数必须返回 `BAD_DESCRIPTOR` 或 `DEPENDENCY_ERROR`。

对应 body word：word 0 为 `token_count`，word 1–9 必须为 0；token ID 数组位于 payload。

## 6. 静态 Command Batch 和 memory region

v1 使用静态 command batch。manifest 或启动参数必须提供：

```text
command_region_offset
command_region_bytes
command_count
command_record_bytes = 32

descriptor_region_offset
descriptor_region_bytes
descriptor_count
descriptor_record_bytes = 64
```

`command_count` 是 command 结束条件，不使用动态 producer/consumer 指针，也不需要 `BATCH_END` 指令。

推荐的全局区域：

```text
COMMAND_BUFFER
DESCRIPTOR_TABLE
DESCRIPTOR_PAYLOAD
FRONT_DATA
CONTRIBUTION_DATA
FACTOR_DATA
UPDATE_DATA
RHS_DATA
SOLUTION_DATA
STATUS_DATA
TRACE_DATA
```

command、descriptor、payload、数据和 completion 区域不能重叠。软件负责生成，设备负责校验和消费。

## 7. Completion record

每条 command 产生一条 64B completion record：

| 偏移 | 大小 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 4 | u32 | `command_id` |
| 4 | 4 | u32 | `node_id` |
| 8 | 2 | u16 | `status_code` |
| 10 | 2 | u16 | reserved，必须为 0 |
| 12 | 4 | u32 | `pivot_count` |
| 16 | 8 | u64 | `start_cycle` |
| 24 | 8 | u64 | `finish_cycle` |
| 32 | 8 | u64 | `read_bytes` |
| 40 | 8 | u64 | `write_bytes` |
| 48 | 8 | u64 | `stall_cycles` |
| 56 | 4 | u32 | `overflow_count` |
| 60 | 4 | u32 | `retry_count` |

`status_code=OK` 只表示 command 控制流程成功，不代表 residual 达标。

## 8. Status code

```text
0x0000 OK

0x0001 BAD_COMMAND
0x0002 BAD_DESCRIPTOR
0x0003 ADDRESS_FAULT
0x0004 BUFFER_FULL
0x0005 DEPENDENCY_FAILED

0x0100 PIVOT_NOT_FOUND
0x0101 PIVOT_UNSTABLE
0x0102 NUMERIC_OVERFLOW
0x0103 QUANTIZATION_SATURATION
0x0104 PRECISION_RETRY

0x0200 TIMEOUT
0x0201 ABORTED
```

状态区间含义：

```text
0x00xx 控制/协议/依赖
0x01xx 数值
0x02xx 生命周期/超时
```

控制失败和数值失败必须分别记录。错误 command 不得 signal `READY` token，只能 signal `FAILED` 或执行 abort 语义。

## 9. 精度后端边界

command schema 只为设备路径定义以下数据格式：

```text
FP64
FP32
INT32
LEGACY_INT32_EXP
```

固定数值编码如下：

| 类别 | 名称 | 数值 |
|---|---|---:|
| DataFormat | `FP64` | 0x01 |
| DataFormat | `FP32` | 0x02 |
| DataFormat | `INT32` | 0x03 |
| DataFormat | `LEGACY_INT32_EXP` | 0x04 |
| DataLayout | `ROW_MAJOR` | 0x01 |
| KernelBackend | `SYSTEMC_FP32_DEVICE_MODEL` | 0x01 |
| KernelBackend | `SYSTEMC_INT32_GEMM_MODEL` | 0x02 |
| SolveDirection | `FORWARD` | 0x01 |
| SolveDirection | `BACKWARD` | 0x02 |

`LEGACY_INT32_EXP` 只用于读取历史归档，不允许 T03 当前主线生成该格式。

后端关系为：

```text
HOST_FP64_REFERENCE
    独立 Python/C++ 参考程序，不属于 device command backend

SYSTEMC_FP32_DEVICE_MODEL
    Panel-LU/TRSM/solve 使用 FP32 行为计算

SYSTEMC_INT32_GEMM_MODEL
    只有 GEMM 调用边界执行 FP tile → INT32 GEMM → FP tile
```

FP8、全局 Tile BFP、QF20/QF26 和整树 exponent 传播不属于 T01 契约。

## 10. T01 验收标准

- Python/C++ 都能编码和解码 32B command record；
- Python/C++ 都能编码和解码 64B descriptor record；
- completion record 为固定 64B；
- command/descriptor 的字节序、offset、size 和 reserved 规则有 golden fixture；
- `WAIT_DEP` 不出现在正式 command 流；
- 静态 batch 能通过 `command_count` 完成；
- wait token 支持 `UNSIGNALED/READY/FAILED`；
- `NODE_COMMIT` 的原子提交语义已测试；
- 非法 opcode、flags、descriptor、payload、地址和 token 能返回明确错误；
- 当前 Command v1 compiler 产物可由 codec 重新读取；
- schema 实现不依赖 Panel-LU/TRSM 的最终微架构。
