# Command / Descriptor Schema v2

状态：**当前主线规范**。Command Compiler 只生成 schema v2 产物。

命名说明：本格式是 **Command/Descriptor schema v2**，与已归档的
**NodeTask ABI v2**（`archive/legacy-node-task-abi-v2/`）没有任何关系。
NodeTask ABI v2 是退役的旧执行链；本文描述的是 Command/Descriptor 记录
格式的第二次修订。历史版本见 [Command_Schema_v1.md](Command_Schema_v1.md)
（仅作追溯，不再是有效产物格式）。

## 0. 相对 v1 的变化

| 项目 | v1 | v2 |
| --- | --- | --- |
| 记录大小 | command 32B / descriptor 64B / completion 64B | 不变 |
| 顶层 schema 版本 | manifest `abi.version = 1` | manifest `abi.version = 2` |
| 主元策略 | 未定义 | `FRONT_DESC.body_words[8]` 显式声明 |
| 主元阈值 | 未定义 | `FRONT_DESC.body_words[9]`（binary32 位型） |
| `PANEL_LU` 的 descriptor | `KERNEL_DESC` | `FRONT_DESC`（policy 所在记录） |
| P-vector 初始化 | 未定义 | INT32 哨兵 `-1`，由 `PANEL_LU` 改写 |
| 状态码 | 无 policy/静态失败专用码 | 新增 `POLICY_NOT_SUPPORTED`、`STATIC_PIVOT_FAILURE`、`NON_FINITE_VALUE` |

v1 的 `FRONT_DESC` 将 word 8–9 置 0；v2 把 0 解释为**非法策略**并拒绝
该记录。因此 v1 产物不会被 v2 悄悄接受，也不存在“根据 P-vector 是否为
identity 猜策略”的路径：策略只来自 descriptor 字段。

命令流中的 pivot 搜索、行交换、更新是 Panel 控制器的内部微操作，不展开为
顶层 command。动态模式下实际 pivot 行由更新后的数据决定。

## 1. 顶层 artifact

manifest（JSON）声明：

```json
"abi": {
  "name": "command_descriptor",
  "version": 2,
  "endianness": "little",
  "address_unit": "byte_offset_from_memory_image_base",
  "command_record_bytes": 32,
  "descriptor_record_bytes": 64,
  "completion_record_bytes": 64,
  "pivot_policy_field": "FRONT_DESC.body_words[8..9]"
}
```

## 2. Command record（32B，布局与 v1 相同）

| 偏移 | 大小 | 类型 | 字段 | 语义 |
|---:|---:|---|---|---|
| 0 | 4 | u32 | `opcode` | 宏指令编号 |
| 4 | 4 | u32 | `flags` | 行为标志（`TRACE_ENABLE`、`ALLOW_RETRY`） |
| 8 | 4 | u32 | `command_id` | batch 内唯一编号 |
| 12 | 4 | u32 | `node_id` | 所属节点；全局 solve 可为 `NONE` |
| 16 | 4 | u32 | `descriptor_id` | 主 descriptor；无则 `NONE` |
| 20 | 4 | u32 | `wait_list_id` | `DEPENDENCY_DESC` 编号；无则 `NONE` |
| 24 | 4 | u32 | `signal_token` | 完成后发布的 token；无则 `NONE` |
| 28 | 4 | u32 | `arg0` | 保留，必须为 0 |

`NONE = 0xFFFFFFFF`。flags bit 2–31 保留必须为 0。未知 opcode/flags 返回
`BAD_COMMAND`。

### 2.1 Opcode

与 v1 相同（`NODE_BEGIN` 0x01 … `ABORT_NODE` 0x0D）。语义补充：

- `PANEL_LU` 的关联 descriptor 是该节点的 `FRONT_DESC`；执行语义由
  `FRONT_DESC` 的 `pivot_policy`、`pivot_threshold`、`pivot_dim` 与
  P-vector 引用共同确定，数据格式由对应 `REGION_DESC`/`FACTOR_DESC` 给出。
- `PANEL_LU` 成功时必须把完整局部置换写入 P-vector region（`STATIC` 写
  显式 identity），并发布 `READY` token。
- `PANEL_LU` 失败时**不**发布 `READY` token；`signal_token` 进入 `FAILED`。

## 3. Descriptor record（64B）

头部布局与 v1 相同：`descriptor_type:u16`、`flags:u16`、`reserved:u32`、
`payload_offset:u64`、`payload_bytes:u64`、body 10×u32（offset 24..63）。
descriptor ID = table index。未知 type/flags、reserved 非零、payload 越界
必须拒绝。

### 3.1 FRONT_DESC（type 0x02）——v2 变化点

```text
word0  front_region_id        REGION_DESC 引用
word1  total_dim              u32
word2  pivot_dim              u32（= 局部 P-vector 长度）
word3  tile_size              u32
word4  contribution_desc_id   CONTRIBUTION_DESC 引用或 NONE
word5  factor_desc_id         FACTOR_DESC 引用（必须有效）
word6  p_vector_region_id     REGION_DESC 引用（必须有效，不允许 NONE）
word7  solve_workspace_id     REGION_DESC 引用（必须有效）
word8  pivot_policy           0x01 STATIC / 0x02 DYNAMIC_PANEL
word9  pivot_threshold        IEEE-754 binary32 位型，[0, 1)
```

- word 8 = 0x00 表示未定义（v1 记录），必须拒绝；executor 也不得根据
  P-vector 内容推断策略。
- word 9 解码为 binary32 浮点数后必须落在 `[0, 1)`；阈值的尺度来源固定为
  `front_pre_elimination_max_abs`（该 front 装配完成、消元开始前的
  `max |A_front|`，每个 front 一次归约，不做逐列候选搜索）。
- **阈值唯一有效值**：配置、manifest、host reference 与 descriptor 共用同一个
  有效阈值——比值在配置时一次性规范化为其 binary32（设备存储格式）值；
  正数下溢为 0 或舍入到 1.0 的输入直接拒绝。manifest 的 `threshold_ratio`
  必须与 `threshold_bits_fp32` 解码值严格相等。
- `PANEL_LU` command 的 `descriptor_id` 指向本记录；`NODE_BEGIN`、
  `LOAD_FRONT`、`NODE_COMMIT` 同样引用本记录。
- v2 不再为 panel factorization 生成 `KERNEL_DESC`；`KERNEL_DESC` 只用于
  TRSM/GEMM。

### 3.2 其它类型

`REGION_DESC`、`CONTRIBUTION_DESC`、`FACTOR_DESC`、`KERNEL_DESC`、
`SOLVE_DESC`、`SCALE_DESC`、`DEPENDENCY_DESC` 的 body 布局与 v1 相同。
差异与约束：

- `FACTOR_DESC.word2`（P-vector region）与 `SOLVE_DESC.word3`（P-vector
  region）必须有效，`NONE` 非法。
- `CONTRIBUTION_DESC` 的 payload 为 `row_map || col_map`（各
  `row_count`/`col_count` 个 u32）。当前主线 `row_map` 恒为
  `0..update_dim-1`，装配语义为
  `target[col_map[i], col_map[j]] += source[i, j]`。

## 4. 主元策略语义

### 4.1 STATIC（默认）

- 按最终排序后的对角固定顺序消元（第 k 步消第 k 个 pivot block 列）。
- 不搜索候选行、不交换行、不使用 HPU/ATU 动态映射事务；该路径**旁路**
  HPU/ATU（运行时旁路与综合时裁剪是两种不同能力，见 SystemC 模型规范）。
- 仍然检查**实际消元后**的主元（而不是只检查输入对角）：
  - 零主元 → `STATIC_PIVOT_FAILURE`；
  - `|pivot| <= threshold_ratio * front_pre_elimination_max_abs`（非零）→
    `PIVOT_UNSTABLE`；
  - NaN/Inf（输入或消元产生）→ `NON_FINITE_VALUE`。
- 失败语义：固定顺序失败**不**等于矩阵奇异；不静默替换微小主元；不在
  运行途中切换策略；失败 command 不发布 `READY` token。Host 重试必须从
  原始输入或完整重建的状态开始。
- P-vector：设备写显式 identity（`p[i] = i`），供 solve 统一使用；初始化
  哨兵为 INT32 `-1`，全零数组永远不解释为 identity。

### 4.2 DYNAMIC_PANEL（可选）

- 仅在当前 pivot block 内选主元：第 k 步候选行为 `k .. pivot_dim-1`。
- 选择最大绝对值，tie-break 取**最小逻辑行号**（确定性）。
- 不支持跨 front 换行、update 行提升为 pivot、delayed pivot、动态 front
  扩张。需要跨 pivot block 才能继续时明确失败（`PIVOT_NOT_FOUND`）。
- 动态选择与行映射由 HPU/ATU 完成；实际 pivot 行是 Panel 控制器内部微操作。
- P-vector 记录 pivot block 内的实际行交换（见 §5）。

两种策略是 batch 级统一策略（本版本不支持逐命令混用）；descriptor 必须
显式携带策略，executor 语义以 descriptor 为准。静态与动态可能产生不同的
L/U/P；因子对照只能在同一策略内进行。

## 5. P-vector 语义

- 位置：每节点一个 INT32 region（`*.p_vector`），长度 `pivot_dim`。
- 方向：gather，`p[i]` = 第 i 个 pivot 位置的源行（pivot block 内行号）。
  solve 侧 RHS 按 `p` gather，solution 按变量列顺序恢复。
- 初始化：image 中填充 INT32 `-1`（未写）；`PANEL_LU` 结束后必须全部改写。
- `STATIC` 写显式 identity；`DYNAMIC_PANEL` 写 pivot block 内交换结果。
- 该数组是**设备局部**置换，与全局预处理映射（`P_match`、`Q`）无关；后者
  只存在于 host 侧 manifest/verification，不进入设备执行语义。
- v2 中 FRONT/FACTOR/SOLVE 的 P-vector 引用不允许 `NONE`（不存在“省略”
  情形，因此无需定义 NONE 的合法条件）。
- 校验者必须**沿实际 descriptor 引用**检查 P-vector region：目标必须是
  `REGION_DESC`、格式 INT32、容量恰为 `pivot_dim`、内容全为哨兵 `-1`；
  并要求同一节点的 `FRONT_DESC`/`FACTOR_DESC`/`SOLVE_DESC` 指向**同一个**
  P-vector region。含 `-1` 未写或越界值的 P-vector 必须拒绝；不得把全零
  数组当作 identity（`pivot_dim == 1` 时 `p[0] == 0` 本身是合法 identity）。

## 6. Completion record（64B，布局与 v1 相同）

`pivot_count` 语义补充：成功时为已消元主元数；数值失败时为**首个失败列号**。
其余字段（cycle/bytes/stall/overflow/retry）与 v1 相同。completion 初始化
模板由 host 生成（status = `OK`、计数清零），executor 每槽覆写。

## 7. 状态码

| 数值 | 名称 | 语义 |
|---:|---|---|
| 0x0000 | `OK` | 成功 |
| 0x0001 | `BAD_COMMAND` | 非法命令（未知 opcode/flags/arg0） |
| 0x0002 | `BAD_DESCRIPTOR` | 非法 descriptor（含非法策略编码、非法 P-vector 引用、混合格式记录） |
| 0x0003 | `ADDRESS_FAULT` | 越界/未对齐/region 重叠访问 |
| 0x0004 | `BUFFER_FULL` | 缓冲不足 |
| 0x0005 | `DEPENDENCY_FAILED` | 依赖 token 失败 |
| 0x0006 | `POLICY_NOT_SUPPORTED` | executor 未实现/不支持声明的主元策略，拒绝执行 |
| 0x0100 | `PIVOT_NOT_FOUND` | DYNAMIC_PANEL 在 pivot block 内找不到可容许主元 |
| 0x0101 | `PIVOT_UNSTABLE` | 实际主元低于相对阈值（非零但过小） |
| 0x0102 | `NUMERIC_OVERFLOW` | 数值溢出 |
| 0x0103 | `QUANTIZATION_SATURATION` | 局部量化饱和（保留） |
| 0x0104 | `PRECISION_RETRY` | 可重试的精度失败（保留） |
| 0x0105 | `STATIC_PIVOT_FAILURE` | STATIC 固定顺序产生零主元（非奇异性结论） |
| 0x0106 | `NON_FINITE_VALUE` | 输入或消元出现 NaN/Inf |
| 0x0200 | `TIMEOUT` | 超时 |
| 0x0201 | `ABORTED` | 中止 |

失败分类：

- **未实现 / 未支持**：executor 缺少所声明策略的能力 → `POLICY_NOT_SUPPORTED`，
  命令不执行、不发布 `READY` token；依赖它的命令完成为 `DEPENDENCY_FAILED`。
- **契约违规**（非法策略值、P-vector 引用 `NONE`、v1/v2 混合记录）→
  `BAD_COMMAND` / `BAD_DESCRIPTOR`，在验证阶段拒绝。
- **主元失败**：见 §4 的数值状态码；命令不发布 `READY` token。

## 8. 与 manifest 的一致性要求

manifest 除 `abi` 外还必须给出：预处理（行缩放、匹配、变换公式与各置换
数组）、`pivot` 块（策略、阈值、阈值尺度、静态失败策略、P-vector 语义）、
`required_device_capabilities`（`STATIC` 要求 `hpu_atu_dynamic_mapping =
bypassed`，`DYNAMIC_PANEL` 要求 `required`），以及独立验证用的原始
`A`、`b`、golden `x`。验证器交叉检查所有 `FRONT_DESC` 的 word 8/9 与
manifest 一致、P-vector region 为哨兵初值、置换数组互为一致组合。

## 9. Fixture

共享 golden fixture 见 `graduation-code/fixtures/command_schema_v2/`
（Python 与 C++ codec 测试共同消费；含 `static_front_descriptor` 与
`dynamic_front_descriptor`）。历史 v1 fixture 保留在
`graduation-code/fixtures/command_schema_v1/`，仅用于追溯与负面测试
（v2 codec 必须拒绝 v1 `FRONT_DESC`）。
