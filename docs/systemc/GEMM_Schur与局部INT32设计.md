# GEMM Schur 与局部 INT32 设计

实现基准：2026-10-11 当前工作区。
本文定义顶层 Schur 更新及局部 INT32 路径。Panel/TRSM 内部更新仍使用 FP32，设备存储格式均为 FP32。
公共队列、PE、版本、带宽和取消机制见 [共享计算引擎与 PE](共享计算引擎与PE设计.md)。

## 1. 顶层数学与块地址

设 front 阶数 `n`、主元维 `p`、更新维 `u = n - p`。顶层 Schur 更新为：

```text
A22 := A22 - L21 * U12
```

- `A11`：PANEL_LU 原地 packed LU（严格下三角 L 单位对角隐含 + 上三角 U），
  GEMM 只读、不改；
- `A12`：TRSM_LEFT 求解后的 `U12`（p × u），GEMM 只读、不改；
- `A21`：TRSM_RIGHT 求解后的 `L21`（u × p），GEMM 只读、不改；
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

## 2. 架构与任务调度

```text
CommandDevice → GemmExecutionUnit → PanelEngine → GemmController
→ TileDispatcher → MultiModePe（FP32 或局部转换/整数计算）
→ A22 写回、版本提交 → adapter 终态 → completion/token
```

两种后端共用 controller、dispatcher、PE 存储、装载/写回和背压机制。
适配器绑定同一共享引擎；每条命令占一个计算组，不额外复制整块 front。

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

共享 PE 池内部仍以 `TileTask` 表达任务（一个 dispatcher 服务公共计算模式），
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
  记录内表达；适配器与公共调度通过这套内部记录交互。
- `PanelEngine::gemm_log()` 在命令结束后保留该命令的 `GemmReq/GemmRsp` 序列，
  供报告与测试核对。

### 2.3 数值完成 ≠ 响应完成

- 功能数值计算在写回事件之前不可见：只有真实写回并提交版本后
  `GemmRsp.output_visible` 才为真。
- 全部任务及 controller 排空后引擎提供终态；适配器 poll 消费后释放组，控制面释放 lease 并收尾。
  成功 completion 的 DDR 写入确认后 token 才 READY。
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
参数，按第 5 节从 manifest 校验并应用，两类配置分别处理。

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

- INT32 存储；默认有符号 30 位有效量化范围（可配置 1–30 位），对称范围
  `q ∈ [-(2^29-1), +(2^29-1)]`；`qmax = 2^29 - 1`。
- INT64 累加；默认 K 分块最大 16。
- 安全界：`16 · (2^29-1)^2 < 2^62`，因此默认量化范围的 16 项点积安全。
  **不得**宣称任意完整 INT32 范围的 16 项累加安全：完整 INT32 极值会溢出
  （见 4.6 的显式溢出检查）。

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
- `maxabs = f · 2^m`（`f ∈ [0.5,1)`）：候选 `e0 = m - (effective_bits - 1)`，若
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

量化越界时 FAIL 策略返回 QUANTIZATION_SATURATION；CLAMP 策略夹紧到有效量化范围并保留饱和计数。

### 4.7 scale 不扩散

- scale 不传播为父 front 或整棵树的数值格式；不修改 Panel 已冻结的
  front maxabs/threshold context；不重新量化 Panel/TRSM 的内部更新。
- INT32 模式与现有 `Fp64Diagnostic` 组合显式拒绝（`POLICY_NOT_SUPPORTED`）；
  保留原 FP64 诊断路径与回归，不增加隐式精度切换。

## 5. 接口与启动精度参数

SCALE_DESC 的字节编码以 [Command schema v2](../interface/Command_Schema_v2.md) 为准。
运行时 descriptor 选择 AUTO 或固定指数、舍入和饱和处理；boot 精度配置提供有效位数、
INT64 累加位数、K 分块上限，以及 maxabs/quantize/dequantize 每周期元素数。
manifest 参数先校验再应用到 PanelLuConfig；显式模型配置与镜像冲突时启动拒绝。

软件 CLI 配置及 manifest 字段见 [软件侧技术文档](../software/软件侧技术文档.md)。
设备后端必须与编译时 GEMM backend 一致。FP32 不带 SCALE_DESC；INT32 必须带有效 SCALE_DESC。

## 6. 失败、abort/reset 与资源复用

- 失败位置在内部响应（`TileTaskResponse.gemm_failed_phase/row/col`、
  `GemmRsp.failed_*`）与 trace 中保留 tile 行/列、K 步及 front 坐标；不扩展
  Completion ABI。命令级 `failed_column` 为 A22 tile 列 + 局部失败列（始终在
  失败输出 tile 内）。
- 失败 tile 不写回部分结果；其他已成功提交的 tile 保留（当前实现不承诺整条 GEMM
  回滚）。
- 失败/abort 后：停止新任务与新访问；定向取消未执行任务；正确处理已接受事务
  与背压；排空后释放组、task slot、FIFO、version/lease 资源；后续 front 可
  重新使用。reset 后不接收旧 epoch 的迟到成功响应，旧句柄/版本失效。
- 转换阶段 abort（INT32）同样不发布部分结果，之后同一引擎可继续执行新命令。
- FP64 diagnostic 的 GEMM_SCHUR 回存 FP32 时检查完整有效 M×N 区域，任一
  元素溢出为 Inf 都返回 NON_FINITE_VALUE，不提交该 tile。

## 7. 代码与验证入口

| 对象 | 代码 | 作用 |
|---|---|---|
| `GemmExecutionUnit` | [panel_exec_unit.hpp](../../graduation-code/systemc/include/controlplane/panel_exec_unit.hpp)、[实现](../../graduation-code/systemc/src/controlplane/panel_exec_unit.cpp) | 同一 front 与后端/scale 检查、请求响应适配 |
| `GemmController` | [gemm_controller.hpp](../../graduation-code/systemc/include/panel_lu/gemm_controller.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/gemm_controller.cpp) | K 屏障、A22 tile、内部 GemmReq/Rsp |
| `GemmReq/GemmRsp` | [gemm_types.hpp](../../graduation-code/systemc/include/panel_lu/gemm_types.hpp) | 内部记录和诊断 |
| 局部精度适配器 | [precision_adapter.hpp](../../graduation-code/systemc/include/panel_lu/precision_adapter.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/precision_adapter.cpp) | scale、ties-to-even、显式溢出与反量化 |
| 模块测试 | [gemm_real_tests.cpp](../../graduation-code/systemc/tests/gemm_real_tests.cpp) | FP32/INT32 数值、尾块、非法 scale、范围、取消、背压与复用 |

FP32 对照独立 FP64 乘积；INT32 对照独立整数/二进制舍入 oracle。
验证同时核查 A11/A12/A21 未修改、失败 tile 未写回、局部流量不重复计为 DDR 流量，
以及每个 K 步的部分乘积舍入顺序。整树误差仍由独立因子/求解 checker 评估。
