# T10：顶层 GEMM_Schur 与局部 INT32 GEMM 设计

状态：**T10 已实现并接入设备闭环**（FP32 顶层 Schur 数值执行、局部
PrecisionAdapter、INT32 GEMM 行为模型、SCALE_DESC 运行时读取、控制面接入与
独立验证）。T11 已补齐真实 assemble/STATIC 树形分解，见
[T11 集成设计](T11_真实装配与树形分解集成.md)。树形 solve（T12）、
DYNAMIC_PANEL、INT32 RTL、FP8 与全局 BFP 仍属后续任务。

实现名称核对（2026-10-07）：本文的 `PrecisionAdapter` 和 `Int32GemmBehavioral`
表示职责，不是现存同名类。实际接口位于
[`precision_adapter.hpp`](../../graduation-code/systemc/include/panel_lu/precision_adapter.hpp)，
算法位于 [`precision_adapter.cpp`](../../graduation-code/systemc/src/panel_lu/precision_adapter.cpp)；
[`MultiModePe`](../../graduation-code/systemc/src/panel_lu/pe_model.cpp) 调用
`run_int32_gemm_update`，并由共享 engine/dispatcher 管理时序及写回。

本文是 T10 的设计与实现权威，按“架构与交互 → 数据/接口契约 → 实现机制与
时序 → 数值策略 → 失败/abort/reset → 验收与边界”组织。

---

## 1. 顶层数学与块地址

设 front 阶数 `n`、主元维 `p`、更新维 `u = n - p`。顶层 Schur 更新为：

```text
A22 := A22 - L21 * U12
```

- `A11`：PANEL_LU 原地 packed LU（严格下三角 L 单位对角隐含 + 上三角 U），
  T10 只读、不改；
- `A12`：TRSM_LEFT 求解后的 `U12`（p × u），T10 只读、不改；
- `A21`：TRSM_RIGHT 求解后的 `L21`（u × p），T10 只读、不改；
- `A22`：唯一被 GEMM_SCHUR 修改的块（u × u），原地读改写。

`KERNEL_DESC`：`M = u`、`N = u`、`K = p`、`format = FP32`、
`tile_size = 16`（其他值显式拒绝）。

块起点（front 基址偏移只由 lease 访问层叠加一次）：

| 块 | 行起点 | 列起点 |
| --- | --- | --- |
| A21 (u×p) | `p` | `0` |
| A12 (p×u) | `0` | `p` |
| A22 (u×u) | `p` | `p` |

任务按“块起点为原点”划分。令 K 块大小为 `b`：FP32 固定为 16，INT32
采用 `precision.k_block_max`（可配置 1–16，默认 16）。有效维度为
`m = min(16, u - 16i)`、`n_tile = min(16, u - 16j)`、`k = min(b, p - bs)`，
任务数为 `ceil(u/16)^2 · ceil(p/b)`，A21 的 K 列与 U12 的 K 行都从 `bs` 开始。
`p = 23`、`u = 19` 时不是 16 的整数倍，仍得到 `ceil(19/16)^2 · ceil(23/16)
= 8` 个任务；不会按完整 front 的 16 网格错误定位子块。

`p = n`（`u = 0`）时编译器不生成 GEMM_SCHUR/STORE_UPDATE 命令；运行期收到
空更新维的 GEMM 请求返回 `BAD_DESCRIPTOR`，不用零维 descriptor 伪造成功。

## 2. 共享计算引擎与调度

```text
CommandDevice（T05，每拍一次 tick）
  ├─ PanelEngine（唯一物理 PE 池）
  │    ├─ compute_group_count × pes_per_group 个 MultiModePe
  │    ├─ TileDispatcher + TileVersionStore（有界队列/依赖/版本/组内派发）
  │    ├─ OnChipArbiter（设备级统一读/写预算）
  │    └─ 每命令一个 controller（PanelController / TrsmController / GemmController）
  ├─ PanelExecutionUnit / TrsmExecutionUnit / GemmExecutionUnit ─┐ 共享同一 engine
  └─ FactorStoreUnit / TestStubUnit（显式后端）                  ─┘
```

- 默认实验配置 `compute_group_count = 2`、`pes_per_group = 2`（共 4 个物理
  PE），组数与每组模块数保留可配置；`pe_count` 只是与乘积一致的校验字段。
- 一条被接受的 GEMM_SCHUR 命令独占一个 PE 组，直到全部 tile、队列记录与访问
  排空、终态响应被 T05 接收后才释放；其他命令不得借用组内空闲 PE。
- 组内不同 tile 可并行使用多个 PE；单个 16×16 tile 不拆分到多个阵列。
- 组不永久绑定操作或 front：同一组可先做 Panel、再 TRSM、再 GEMM，整个设备
  只有一套物理 PE 池。GEMM/INT32 不新增独立 PE 池。
- 继续沿用最多两个 resident front/compute command、有限 task slots、FIFO、
  写回队列、默认 64 B/cycle 读/写预算与“每域每拍单请求方 grant”。

### 2.1 K 步屏障与 A22 版本语义

按上述 K 块大小 `b`，`s = 0 .. ceil(p/b)-1`，同一步内按下标递增生成
`C(i,j) -= A(i,s) * B(s,j)`：

- 同一个 `s` 中不同 `C(i,j)` 可并行；
- **本步全部任务真实写回并提交 tile 版本后**才生成下一步；
- 同一个 C tile 的不同 K 步不会并发读改写；
- 任务生成受 dispatcher 队列容量与背压约束，保留生成进度，不会一次性把整个
  front 的数值复制进任务列表。

A22 使用独立版本 key 空间 `TileViewTag::A22`（与 A11/A12/A21 的视图 key 不
别名）。第 `s` 步任务的 `dst.version = s`，提交后版本递增；第 `s+1` 步只消费
上一已提交版本。A21/U12 只读，使用各自视图的当前已提交版本。

### 2.2 统一 GemmReq / GemmRsp

共享 PE 池内部仍以 `TileTask` 表达任务（一个 dispatcher 服务所有模式），
GEMM controller 在边界上把任务映射为统一记录（`gemm_types.hpp`）：

```text
GemmReq {
  txn_id, task_id, epoch, command_id, node_id, descriptor_id( KERNEL_DESC ),
  front_dim, pivot_dim, step, tile_row, tile_col, m, n, k,
  a/b/c: { buffer_id, offset, stride_bytes, version, version_key, valid },
  backend (Fp32Device | Int32Gemm), scale (GemmScalePolicy),
  accumulate_mode (SubtractProduct), front_row, front_col
}
GemmRsp {
  txn_id, task_id, epoch, command_id, node_id, status, reason,
  output_visible, output_version,
  saturation_count, overflow_count, saturated, overflowed,
  read_bytes, write_bytes, local_conversion_bytes,
  scale_reduction_cycles, quantize_cycles, integer_compute_cycles,
  dequantize_cycles, exponent_a, exponent_b,
  failed_phase, failed_tile_row/col, failed_step, failed_front_row/col
}
```

- 这些字段全部是内部契约，不扩展外部 Command/Descriptor/Completion ABI
  （32B/64B/64B 不变）。
- 上层调度不依赖具体 FP32/INT32 数值实现：后端选择、scale 策略与统计都在
  记录内表达；未来替换 RTL 后端不需要修改 GCU/front 流程。
- `PanelEngine::gemm_log()` 在命令结束后保留该命令的 `GemmReq/GemmRsp` 序列，
  供报告与测试核对。

### 2.3 数值完成 ≠ 响应完成

- 功能数值计算在写回事件之前不可见：只有真实写回并提交版本后
  `GemmRsp.output_visible` 才为真。
- GEMM 命令只有在全部任务完成、controller 排空、组与 lease 释放后才发布终态
  响应；成功 completion 的 DDR 写入确认后 token 才 READY。
- 失败不发布 READY；`pivot_count` 对 GEMM 恒为 0（失败列号绝不填入
  `pivot_count`）。

## 3. 设备执行路径

### 3.1 GemmExecutionUnit（控制面适配器）

`GemmExecutionUnit` 在建立 engine context 之前完成全部契约检查，失败时立即
终态（无数据访问、无 READY）：

1. opcode 必须是 `GEMM_SCHUR`；
2. `KERNEL_DESC` 的 A/B/C 三个操作数必须解析为当前驻留 front 的同一 region
   （descriptor id、base、byte_size 完全一致），外来 front 引用 →
   `BAD_DESCRIPTOR`；
3. `M/N/K = u/u/p`，`tile_size = 16`；
4. 后端与 SCALE_DESC 组合：
   - `SYSTEMC_FP32_DEVICE_MODEL`（0x01）→ `scale_desc_id` 必须为 `NONE`，
     否则 `BAD_DESCRIPTOR`；
   - `SYSTEMC_INT32_GEMM_MODEL`（0x02）→ 必须引用有效 `SCALE_DESC`，否则
     `BAD_DESCRIPTOR`；策略解码失败按第 4 节的映射返回；
   - 未知 backend code → `BAD_DESCRIPTOR`。

### 3.2 运行时 SCALE_DESC 读取

`DescriptorReader` 通过 `MemoryService` 实际读取 `KERNEL_DESC` 与它引用的
`SCALE_DESC` 记录（每次访问都是设备事务并计数），解析 word 0–5 并验证保留
字，把解码后的 `GemmScalePolicy` 放进 `KernelView`；畸形编码 → 描述符读终态
`BAD_DESCRIPTOR`，已定义但不支持 → `POLICY_NOT_SUPPORTED`。单命令的 scale
策略始终来自运行期 descriptor 读取；位宽、K 上限与转换速率属于 boot 时模型
参数，按第 6 节从 manifest 校验并应用，两类配置分别处理。

### 3.3 数据路径

- 操作数经真实 `BufferManager` lease 装载到 PE 本地 staging（有界预算分段），
  结果写回同一 front 的 A22 原地槽位；写回完成后才提交版本。
- FP32 后端复用既有 `kernels::run_gemm_update`，保持有效 K 元素递增的 FP32
  FMA 顺序。
- INT32 后端使用 `precision_adapter::run_int32_gemm_update`（第 4 节），与
  FP32 共享同一 dispatcher/PE/时序/写回/背压路径，不存在绕过资源约束的第二
  套“行为后端”。

### 3.4 有限 staging 与资源

- PE 本地 staging：A/B 的 FP32 装载缓冲在 maxabs 之后复用为 INT32 staging，
  不保留两份完整操作数。容量：FP32/INT32 staging 每 16×16 操作数 tile
  1024 B，A+B 共 2048 B（与 `pe_input_fifo_bytes` 同量级但单独计；FIFO 只
  计装载，不把复用缓冲重复计容）。
- INT64 累加器是每输出元素的架构累加存储：8 B/元素，每 slot 最大
  16×16×8 = 2048 B，位于每 slot 的累加寄存器组，不占 operand FIFO；一个 PE
  同时只推进一个计算/转换任务（排他关系）。两者都计入 `summary.json` 的
  `gemm_execution` 资源字段。
- 一个 PE 同时只推进一个计算/转换任务；转换阶段按顺序建模（本版不做转换与
  另一任务计算的并行优化）。
- 片上 front stride 为紧凑 `n×4`；DDR region stride 由各自 descriptor 独立
  表达（`STORE_UPDATE` 使用 region 的 row_stride，支持 padded stride）。

## 4. 局部 INT32 数值策略

只把顶层 `GEMM_SCHUR` 切换到局部 INT32。Panel 与 TRSM 内部使用的
`GemmUpdate` 继续 FP32；外部 front、DDR image、L/U/update 继续存储 FP32。
后端是命令/tile 的显式属性，绝不根据 `TileMode::GemmUpdate` 或存储格式推断。

### 4.1 数据模型（报告口径）

```text
INT32 storage / 30 effective quantization bits / INT64 accumulator
```

- INT32 存储；有符号 30 位有效量化范围，对称范围
  `q ∈ [-(2^29-1), +(2^29-1)]`；`qmax = 2^29 - 1`。
- INT64 累加；默认 K 分块最大 16。
- 安全界：`16 · (2^29-1)^2 < 2^62`，因此默认量化范围的 16 项点积安全。
  **不得**宣称任意完整 INT32 范围的 16 项累加安全：完整 INT32 极值会溢出
  （见 4.6 的显式溢出检查与负面测试）。

### 4.2 每个 K 分块的数据流

```text
已装载 FP32 A/B
  → 分别计算有效 tile 的 maxabs（含有限性检查）
  → 分别选择二的幂 scale eA/eB（AUTO）或使用固定 scale
  → 量化成 INT32（round-to-nearest, ties-to-even，先检查有限性与范围）
  → INT32×INT32、INT64 累加（每步显式范围检查）
  → 反量化为 FP32 部分乘积（正确舍入的 binary32）
  → FP32 C 减去该部分乘积（C := C - partial）
  → 按现有写回路径提交
```

每个 K 分块的 `eA/eB` 可以不同；各块独立反量化后更新 FP32 C，不直接累加不同
exponent 的整数部分和。

### 4.3 AUTO scale 规则

`e` 是满足 `maxabs / 2^e <= qmax` 的最小整数（使用 binary64 精确比较两个
候选指数，正确处理幂边界、subnormal、极小值与接近 FP32 最大值的输入）：

- `maxabs = 0` → 确定性的 `e = 0` 与全零整数；
- `maxabs = f · 2^m`（`f ∈ [0.5,1)`）：候选 `e0 = m - 29`，若
  `qmax · 2^e0 >= maxabs` 取 `e0`，否则取 `e0+1`（两个候选的检查在 binary64
  中精确）。

固定 exponent 使用 i32 二进制补码编码，支持范围 `[-256, 256]`；越界 →
`POLICY_NOT_SUPPORTED`。

### 4.4 舍入与转换

- 量化：`y = x / 2^e` 用 binary64 精确表示（x 为 24 位尾数），
  round-to-nearest/ties-to-even 通过 `floor` 与小数部分比较实现，不依赖进程
  偶然设置的浮点舍入模式。
- 有限性与范围检查发生在浮点转整数之前；`|y| >= qmax + 0.5` 即饱和
  （`qmax` 为奇数，恰好半格时远离零取偶）。
- 反量化：`fp32_from_int64_scaled(S, eA+eB)` 返回精确值 `S · 2^(eA+eB)` 的
  正确舍入 binary32（渐进下溢；溢出返回 ±Inf 并由调用方映射为
  `NON_FINITE_VALUE`）。反量化结果与最终 FP32 C 都检查有限性。
- 舍入顺序明确为：每个输出元素一次整数点积 → 一次正确舍入的反量化 → 一次
  FP32 减法；`round-to-nearest, ties-to-even` 语义由独立 oracle/手算边界用例
  验证。

### 4.5 定时模型

每个有效元素处理速率可配置，默认每周期 1 个有效元素；阶段周期由有效元素数、
处理速率与明确的流水参数推导，不使用任意固定总延迟：

```text
scale_reduction = ceil((M·K + K·N) / maxabs_elements_per_cycle)
quantize        = ceil((M·K + K·N) / quantize_elements_per_cycle)
integer_compute = fill + k · k_ii + m · drain_row_cycles     （复用 PE 阵列参数）
dequantize      = ceil(M·N / dequantize_elements_per_cycle)
exec_cycles     = 上述四项之和
```

失败时只保留到失败点为止的阶段周期（例如饱和发生在量化阶段时
`scale_reduction` 为完整值、`quantize` 为部分值，后续阶段为 0）。
功能内核在装载完成时预计算结果与阶段成本，但统计只在实际进入执行后发布。
`Loading/ReadyWait` 取消时，转换/计算周期、本地转换 bytes、饱和/溢出计数均为
0；真实 operand 装载流量照常保留。`Running` 取消沿用既有排空模型：完成已启动
任务的剩余执行周期，再返回 ABORTED，不写回结果；命令统计包含这些排空成本。

### 4.6 PE 本地流量与统计

- 本地转换 bytes（单独统计，绝不重复记作 DDR/scratchpad 流量）：
  `12 · (M·K + K·N) + 16 · M·N`
  （maxabs 读 FP32 + 量化读 FP32/写 INT32 + 反量化读 INT64/读 C/写 C）。
  该公式适用于成功 tile；失败路径按实际访问递增，包括失败元素的读取，不包含
  未发生的 staging/C 写入。例如 1×1×1 在首个 A 元素量化饱和时为
  `8 B maxabs + 4 B quantize read = 12 B`；FAIL 与 CLAMP 均累计检测到的饱和。
- 分别报告：DDR bytes、scratchpad 读/写 bytes、PE 本地转换 bytes、
  scale reduction / quantize / integer compute / dequantize 周期、队列/端口/
  写回/响应背压、PE/group 占用与配置、saturation/overflow 计数。
- 溢出检查使用扩展宽度累加 + 每步显式范围检查，禁止依赖 signed overflow 或
  事后推断；完整 INT32 极值用例显式返回 `NUMERIC_OVERFLOW`。

### 4.7 scale 不扩散

- scale 不传播为父 front 或整棵树的数值格式；不修改 Panel 已冻结的
  front maxabs/threshold context；不重新量化 Panel/TRSM 的内部更新。
- INT32 模式与现有 `Fp64Diagnostic` 组合显式拒绝（`POLICY_NOT_SUPPORTED`）；
  保留原 FP64 诊断路径与回归，不增加隐式精度切换。

## 5. SCALE_DESC 编码（T10 冻结）

`SCALE_DESC` 只服务单次 GEMM 调用，word 6–9 必须为 0：

| word | 语义 | 取值 |
| ---: | --- | --- |
| 0 | A 的有符号二的幂 exponent | `AUTO = 0x80000000`，否则 i32 补码 `[-256, 256]` |
| 1 | B 的有符号二的幂 exponent | 同上 |
| 2 | output scale | 本版仅 `AUTO`（由实际 `eA+eB` 推导）；固定值 → `POLICY_NOT_SUPPORTED` |
| 3 | rounding mode | `1 = nearest, ties-to-even` |
| 4 | saturation policy | `1 = FAIL`，`2 = CLAMP_AND_REPORT` |
| 5 | overflow policy | `1 = FAIL` |
| 6–9 | 保留 | 必须为 0（codec 强制） |

默认 SCALE_DESC：`AUTO, AUTO, AUTO, 1, 1, 1`。

状态映射：

| 情况 | 状态 |
| --- | --- |
| 未定义枚举 0 / 保留字非零 / 畸形编码 | `BAD_DESCRIPTOR`（描述符读终态） |
| 已定义但不支持（固定 output scale、exponent 越界、未知 rounding/overflow） | `POLICY_NOT_SUPPORTED` |
| FP32 后端引用 SCALE_DESC 或 INT32 后端缺失 SCALE_DESC | `BAD_DESCRIPTOR` |
| 默认策略量化饱和 | `QUANTIZATION_SATURATION`（不提交失败 tile） |
| 累加超出配置范围 | `NUMERIC_OVERFLOW` |
| 非有限输入/输出 | `NON_FINITE_VALUE` |

Python 与 C++ 共享同一编码、验证与跨语言 fixture
（`fixtures/command_schema_v2/scale_desc_default.bin` /
`scale_desc_fixed_exponents.bin`）。

## 6. 软件编译配置

`PipelineConfig.gemm: GemmConfig` 显式选择：

- `FP32_SYSTEMC`（默认）：所有 kernel 走 FP32 设备路径，GEMM 的
  `scale_desc_id = NONE`；
- `FP32_SYSTEMC_INT32_GEMM`：只把顶层 `GEMM_SCHUR` 的 `KERNEL_DESC` 切到
  `SYSTEMC_INT32_GEMM_MODEL` 并生成/引用每节点一个 `SCALE_DESC`；Panel/TRSM
  与内部更新保持 FP32，front/DDR image 保持 FP32。

`GemmConfig` 还携带 `effective_bits`、`accumulator_bits`、`k_block_max`、
scale policy、`conversion_rate`，并校验安全界；不支持或不安全的配置在编译期
拒绝，配置不会被静默忽略。artifact manifest 记录实际模式、scale 策略、有效
位宽、累加宽度与模型参数；设备报告 `summary.json` 的 `gemm_execution` 块记录
实际执行的命令数、INT32 命令数、saturation、本地转换 bytes 与各阶段周期。

manifest 的 `gemm.effective_bits/accumulator_bits/k_block_max/
conversion_rate_per_cycle` 经 `ArtifactLoader` 在 boot 时解析、验证，再由
`CommandDevice` 在创建共享 PE 池前应用。转换速率统一用于 maxabs、量化与
反量化三个阶段；默认设备配置继承 manifest。显式非默认的设备 precision 配置
必须与声明完全一致，冲突在 `start()` 拒绝，避免报告参数与执行参数分离。
旧 artifact 若没有 `gemm` 对象，保留设备侧 precision 配置；存在该对象时必须
完整声明四个参数，畸形类型、零速率或不支持的范围在加载时拒绝。
报告同时记录实际位宽、K 上限、三个转换速率及 `precision_config_source`。

## 7. 失败、abort/reset 与资源复用

- 失败位置在内部响应（`TileTaskResponse.gemm_failed_phase/row/col`、
  `GemmRsp.failed_*`）与 trace 中保留 tile 行/列、K 步及 front 坐标；不扩展
  Completion ABI。命令级 `failed_column` 为 A22 tile 列 + 局部失败列（始终在
  失败输出 tile 内）。
- 失败 tile 不写回部分结果；其他已成功提交的 tile 保留（本版不承诺整条 GEMM
  回滚）。
- 失败/abort 后：停止新任务与新访问；定向取消未执行任务；正确处理已接受事务
  与背压；排空后释放组、task slot、FIFO、version/lease 资源；后续 front 可
  重新使用。reset 后不接收旧 epoch 的迟到成功响应，旧句柄/版本失效。
- 转换阶段 abort（INT32）同样不发布部分结果，之后同一引擎可继续执行新命令。
- FP64 diagnostic 的 GEMM_SCHUR 回存 FP32 时检查完整有效 M×N 区域，任一
  元素溢出为 Inf 都返回 NON_FINITE_VALUE，不提交该 tile。

## 8. 验证与验收

### 8.1 独立 oracle 与误差口径

- FP32：`n = 42, p = 23, u = 19` 与独立 FP64 递推
  `C = A22 - Σ L21·U12` 对照（相对误差 < 5e-5），另在 T09 数据路径上对
  设备实际提交的 L21/U12 做 FP64 复核；
- INT32：与独立 INT64（`__int128`）oracle 精确一致（良态构造）；
  最终 Schur 输出与“设备已提交 L21/U12 的 FP64 复核”相对误差 < 1e-4；
  输入量化误差由独立 checker 验证不超过本 tile scale 步长的一半；
- 不把设备执行链用作自己的 golden；误差计算不在设备链中另算 golden。

### 8.2 STATIC 主元影响（数值传播夹具）

受控用例：

```text
A11 = I2, A12 = [1, 1]^T, A21 = [1, 2^-35], A22 = [1]
```

按既定 FP32 FMA 顺序，Schur = `-2^-35`；默认局部 INT32 路径得到 `0`。
测试通过真实设备链（PANEL → TRSM → GEMM → STORE_UPDATE）生成 update，再把它
作为下一单主元 front 的输入运行真实 STATIC Panel：

- FP32 update：`-2^-35` 非零 → STATIC 成功（pivot_count = 1）；
- INT32 update：`0` → `STATIC_PIVOT_FAILURE`（pivot_count = 0）。

该夹具是**数值传播测试**，不是正式 ASSEMBLE_EXTEND_ADD，也不是 T11 树执行
器；没有注入 FP64 pivot，没有使用 FP64 覆盖实际主元。

### 8.3 测试规模

- `gemm_real_tests` 277 checks（含默认与非默认配置的跨语言回归）（FP32 手算/多 tile/多 K 步/偏移/guard、
  INT32 原始与 tile 内核、scale/舍入边界、饱和/溢出、失败/abort/reset/背压、
  分组、artifact 集成、跨语言 INT32 设备目录、STATIC 主元传播）；
- `trsm_real_tests` 162 checks（T09 链路在 T10 下改为断言 GEMM 真实执行、
  A22 实际更新且与设备因子 FP64 复核一致）；
- `panel_lu_real_tests` 215 checks（n>p 用例改为断言 GEMM 执行）；
- CTest Release 21/21、ASan+UBSan 21/21；
- Python 软件测试 115 passed（含 SCALE_DESC fixture、INT32 编译配置）；
- 跨语言：`tools/t10_fixture_gen.py` 用生产编译器生成 INT32 设备目录，
  C++ 设备加载并执行，`node.0.update = -0.5`（精确）、
  `node.1.update ≈ 2/3`（量化步内）。
- 配置传递：4 位有效量化/K 上限 8/转换速率 8 的生产 artifact 得到手算
  Schur `[-1,-1;0,0]`，阶段周期为 `1/1/5/1`，本地转换 112 B；加载检查
  畸形/缺失/越界参数，设备检查显式配置冲突，summary 与实际 PE 配置一致。
- K 分块：在同一 2×2 PE 配置下检查 K 上限 1、8、16；`p=9,u=17` 的任务数
  分别为 36、8、4，涵盖 K=8+1 与输出尾块，地址/输出对独立 oracle 一致。
- 失败统计：FAIL 饱和计数与失败点的实际本地读取计入命令报告；ReadyWait
  取消无转换成本，Running 取消包含已启动执行的排空成本且无结果提交；
  FP64 diagnostic 在 M/N 大于 K 时仍检查完整 Schur 输出的 FP32 有限性。

### 8.4 周期与字节（n=42, p=23, u=19）

- GEMM 任务数 8；operand read 9880 B；tile writeback 2888 B；
  STORE_UPDATE DDR 数据 1444 B = `u·u·4`；
- 周期为未综合校准的架构模型周期，不声称硬件实测性能/面积/加速比。

## 9. 未完成边界（不得对外宣称）

- T11 已交付真实多节点 ASSEMBLE_EXTEND_ADD 与 STATIC 消除树分解执行器；
  求解扩展属于 T12；
- SOLVE_FORWARD/SOLVE_BACKWARD、整树分解—求解闭环与 `Ax=b` 残差；
- DYNAMIC_PANEL 设备执行与 T14 HPU/ATU 数值路径；
- INT32 GEMM RTL、FP8、全局 BFP front 装配、跨 K 步流水、lookahead、
  缓存复用优化、多 K 步并行归约；
- 新的 PE 规模实验（本版只用 2 组 × 2 PE 默认配置）；
- 实际矩阵总体 STATIC 主元失败率统计；T10 局部传播测试与 T11 受控真实树
  传播均不代表总体失败率；
- 模型参数未经综合校准，所有周期/带宽/面积数字都不是硬件实测。
