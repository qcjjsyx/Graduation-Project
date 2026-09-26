# Command Stream 与 Descriptor 接口

状态：当前主线高层语义；固定二进制契约见 [Command_Schema_v2.md](Command_Schema_v2.md)
（历史格式 [Command_Schema_v1.md](Command_Schema_v1.md) 仅作追溯）

## 1. 接口定位

已归档的 **NodeTask ABI v2**（`archive/legacy-node-task-abi-v2/`）是退役的旧执行链，
与本文的 **Command/Descriptor schema v2** 无关：后者是同一命令/描述符记录格式的第二次
修订，记录大小不变（command 32B / descriptor 64B / completion 64B）。当前软件只生成
schema v2 产物（manifest `abi.version = 2`），后续 SystemC executor 也只消费该接口。

目标接口由四部分组成；当前第一版使用静态 command batch，`command ring` 这一名称只保留为未来动态提交的概念：

```text
command ring
descriptor table
data image / DDR regions
completion queue
```

软件内部使用消除树、front、map 和 node compiler IR；`Command Compiler` 将它们转换为硬件可见的命令和描述符，不再生成 `NodeTask`。

## 2. 两级指令语义

### 2.1 宏指令

软件下发稳定的高层语义：

```text
NODE_BEGIN
LOAD_FRONT
ASSEMBLE_EXTEND_ADD
PANEL_LU
TRSM_LEFT
TRSM_RIGHT
GEMM_SCHUR
STORE_FACTOR
STORE_UPDATE
SOLVE_FORWARD
SOLVE_BACKWARD
NODE_COMMIT
ABORT_NODE
```

### 2.2 内部微操作

GCU 可以将宏指令展开为：

```text
LOAD_TILE
PIVOT_CANDIDATE
PIVOT_SELECT
ATU_SWAP
RECIPROCAL_OR_DIVIDE
MAC_UPDATE
TRIANGULAR_STEP
QUANTIZE
DEQUANTIZE
STORE_TILE
```

pivot 搜索、行交换和更新是未来 Panel 控制器的内部微操作，不展开为顶层 command；其中
`PIVOT_CANDIDATE`、`PIVOT_SELECT`、`ATU_SWAP` 只在 DYNAMIC_PANEL 策略下出现。软件不依赖
这些内部微操作，因此改变 lane 数、流水级数或 buffer 组织不需要修改软件命令语义。

## 3. Command Header（高层语义）

以下字段冻结为 32B 固定记录（布局与 v1 相同）；具体偏移见 [Command_Schema_v2.md](Command_Schema_v2.md)：

```text
CommandHeader {
    opcode
    flags
    command_id
    node_id
    descriptor_id
    wait_list_id
    signal_token
}
```

本版本不在每条 command 中携带 `schema_version` 或 `record_length`。固定大小、字节序和字段语义由 command 格式规范及 manifest 区域信息确定。

## 4. Descriptor 类型

### 4.1 FrontDesc

描述当前 front：

- DDR 或片上 buffer base；
- `total_dim`、`pivot_dim`、有效 tail；
- tile/panel 尺寸；
- leading dimension 和布局；
- pivot 区、update 区和 map descriptor 引用；
- P-vector region 引用（不允许 `NONE`）；
- `pivot_policy`（0x01 STATIC / 0x02 DYNAMIC_PANEL）与相对主元阈值；
- 允许的 front 上限。

`NODE_BEGIN`、`LOAD_FRONT`、`PANEL_LU`、`NODE_COMMIT` 都引用本记录；panel factorization
不再生成 `KERNEL_DESC`，`PANEL_LU` 的关联 descriptor 是该节点的 `FRONT_DESC`。

### 4.2 ContributionDesc

描述 child update 到父 front 的贡献：

- source region；
- target region；
- 相对 row/column map；
- 元素数量和 stride；
- child token；
- 数据格式。

### 4.3 KernelDesc

描述 TRSM 或 GEMM（schema v2 中 `KERNEL_DESC` 只用于 TRSM/GEMM）：

- 算子方向；
- A/B/C descriptor 引用；
- `M/N/K` 和 tile size；
- stride；
- lane、accumulator 和延迟模式；
- precision backend；
- 溢出、retry 和完成策略。

### 4.4 ScaleDesc

只为局部 INT32 GEMM 适配服务：

- `scale_a`、`scale_b`；
- 舍入模式；
- saturate/overflow policy；
- 输出反量化 scale；
- quantize/dequantize 计数区域。

它不定义整个 front、factor、pivot 或消除树的全局数据格式。

## 5. 主元策略、P-vector 与失败语义

### 5.1 策略声明

主元策略是 batch 级统一策略，由每个 `FRONT_DESC` 显式声明：

- `body_words[8]`：`0x01 STATIC`（默认）/ `0x02 DYNAMIC_PANEL`；`0` 非法并标记 v1 记录，
  必须拒绝。executor 不得根据 P-vector 内容猜测策略。
- `body_words[9]`：相对主元阈值，IEEE-754 binary32 位型，必须落在 `[0, 1)`；阈值尺度来源
  固定为 `front_pre_elimination_max_abs`（该 front 装配完成、消元开始前的 `max |A_front|`，
  每 front 一次归约，不做逐列候选搜索）。

### 5.2 STATIC（默认）

- 按最终排序后的对角固定顺序消元；不搜索候选行、不交换行；HPU/ATU 动态映射路径在
  运行时被旁路（运行时旁路与综合时裁剪是两种不同能力）。
- 仍检查**实际消元后**的主元：零主元 → `STATIC_PIVOT_FAILURE`（明确**不是**奇异性结论）；
  非零但 `|pivot| <= threshold_ratio * front_pre_elimination_max_abs` → `PIVOT_UNSTABLE`；
  NaN/Inf → `NON_FINITE_VALUE`。
- 失败语义：不发布 READY token；不静默替换微小主元；不在运行途中切换策略；host 重试
  从原始输入或完整重建的状态开始。

### 5.3 DYNAMIC_PANEL（可选）

- 主元搜索限制在当前 pivot block 内的 `k .. pivot_dim-1` 行；取最大绝对值，确定性
  tie-break 为最小逻辑行号。
- 跨 front 行交换、update 行提升为 pivot、delayed pivot、动态 front 扩张均不支持，
  需要时显式失败（`PIVOT_NOT_FOUND`）。
- 实际 pivot 行是未来 Panel 控制器的内部微操作（HPU 做搜索，ATU 维护逻辑→物理行映射），
  不展开为顶层命令。
- 当前设备执行未实现，仅交付契约表达与参考行为。

### 5.4 P-vector

- 每节点一个 INT32 region，长度 = `pivot_dim`；gather 方向：`p[i]` = 第 i 个 pivot 位置的
  源行（pivot block 内行号）。
- image 中初始化为 INT32 哨兵 `-1`；`PANEL_LU` 必须改写每一个条目（STATIC 写显式
  identity）；全零数组永远不解释为 identity。
- 它是**设备局部**置换，与全局 host 预处理映射 `P_match`/`Q` 不同。
- `FACTOR_DESC`/`SOLVE_DESC` 的 P-vector 引用不允许 `NONE`；非法策略编码、`NONE` P-vector
  引用、v1/v2 混合记录 → `BAD_COMMAND`/`BAD_DESCRIPTOR`。

## 6. 命令示例

```text
NODE_BEGIN          node=17 desc=front17 signal=t17_begin
LOAD_FRONT          desc=front17 signal=t17_loaded
ASSEMBLE_EXTEND_ADD desc=map17 wait=t17_loaded signal=t17_assembled
PANEL_LU            desc=front17 wait=t17_assembled signal=t17_panel
TRSM_LEFT           desc=f12_17 wait=t17_panel signal=t17_u12
TRSM_RIGHT          desc=f21_17 wait=t17_panel signal=t17_l21
GEMM_SCHUR          desc=schur17 wait=(t17_u12,t17_l21) signal=t17_update
STORE_FACTOR        desc=factor17 wait=t17_panel
STORE_UPDATE        desc=update17 wait=t17_update signal=t17_commit
NODE_COMMIT         node=17 wait=t17_commit
```

## 7. 内存区域

在 manifest 中明确区域所有权：

```text
COMMAND_RING
DESCRIPTOR_TABLE
FRONT_DATA
CONTRIBUTION_DATA
FACTOR_DATA
UPDATE_DATA
RHS_DATA
SOLUTION_DATA
STATUS_DATA
TRACE_DATA
```

所有地址都是相对于 `memory_image.bin` 的字节地址。硬件不能使用 Host 指针，也不能因为 SystemC 使用 C++ 容器而绕过这些区域。

## 8. Completion 与错误

每条命令至少产生：

```text
command_id
node_id
status
start_cycle
finish_cycle
read_bytes
write_bytes
pivot_count
overflow_count
retry_count
```

状态至少包括：

```text
OK
BAD_COMMAND
BAD_DESCRIPTOR
ADDRESS_FAULT
BUFFER_FULL
DEPENDENCY_FAILED
POLICY_NOT_SUPPORTED
PIVOT_NOT_FOUND
PIVOT_UNSTABLE
NUMERIC_OVERFLOW
QUANTIZATION_SATURATION
PRECISION_RETRY
STATIC_PIVOT_FAILURE
NON_FINITE_VALUE
TIMEOUT
ABORTED
```

控制失败和数值失败必须分离。`status=OK` 只表示命令执行完成，不表示 residual 达标。
数值失败时 `pivot_count` 记录首个失败列号；失败命令不发布 READY token。失败分类：

- **未实现 / 未支持**：executor 不实现所声明策略 → `POLICY_NOT_SUPPORTED`，命令不执行、
  不发布 READY token；依赖它的命令完成为 `DEPENDENCY_FAILED`。
- **契约违规**（非法策略编码、P-vector 引用 `NONE`、v1/v2 混合记录）→
  `BAD_COMMAND`/`BAD_DESCRIPTOR`。
- **主元失败**：`STATIC_PIVOT_FAILURE`、`PIVOT_UNSTABLE`、`NON_FINITE_VALUE`、
  `PIVOT_NOT_FOUND`，见第 5 节。

## 9. 精度后端

同一套高层命令语义可由不同设备后端执行；FP64 golden 是独立 checker，不是设备 command backend：

```text
HOST_FP64_REFERENCE
    独立 Python/C++ 参考程序，不属于 device command backend；
    host 侧 Python front-tree 参考仅用于显式开启的预检（`reference_check`，默认关闭）与检查，不向设备数据供数
SYSTEMC_FP32_DEVICE_MODEL
SYSTEMC_INT32_GEMM_MODEL
```

在 `FP32_SYSTEMC_INT32_GEMM` 中，`PANEL_LU`、`TRSM`、pivot 和树形依赖仍使用浮点；只有 `GEMM_SCHUR` 通过 `ScaleDesc` 进行局部 INT32 适配。

## 10. 当前实现策略

```text
symbolic/front/map IR
          ↓ Command Compiler
command batch + descriptor table + FP32 data image
          ↓
SystemC executor
          ↓ future
RTL executor
```

当前主线只生成 Command/Descriptor schema v2，不再双写 NodeTask ABI v2；旧实现位于
`archive/legacy-node-task-abi-v2/`（与 schema v2 无关）。固定二进制布局和错误码见
[Command_Schema_v2.md](Command_Schema_v2.md)，[Command_Schema_v1.md](Command_Schema_v1.md)
仅作追溯。T04/T05 的 SystemC executor（数据面/控制面、DMA、descriptor reader、token
scoreboard 运行时、FP32 数值闭环）尚未实现。
