# Command Stream 与 Descriptor 接口

状态：目标接口草案，M0 冻结后成为当前主线接口

## 1. 接口定位

当前软件和 SystemC 使用 ABI v2 的固定 128 字节 `NodeTask`。该接口保留为迁移阶段的当前运行接口，但不作为长期软硬件交互物。

目标接口由四部分组成：

```text
command ring
descriptor table
data image / DDR regions
completion queue
```

软件内部仍可以使用 `NodeTask`、消除树和 front 对象；`Command Compiler` 将它们转换为硬件可见的命令和描述符。

## 2. 两级指令语义

### 2.1 宏指令

软件下发稳定的高层语义：

```text
NODE_BEGIN
WAIT_DEP
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

软件不依赖这些内部微操作，因此改变 lane 数、流水级数或 buffer 组织不需要修改软件命令语义。

## 3. Command Header

以下是语义字段，不立即冻结具体 bit layout：

```text
CommandHeader {
    opcode
    schema_version
    flags
    record_length
    command_id
    node_id
    descriptor_id
    wait_token
    signal_token
}
```

`schema_version` 属于 command schema；它与当前 artifact `abi_version=2` 分开，避免把迁移接口和已有 memory image ABI 混为一个版本号。

## 4. Descriptor 类型

### 4.1 FrontDesc

描述当前 front：

- DDR 或片上 buffer base；
- `total_dim`、`pivot_dim`、有效 tail；
- tile/panel 尺寸；
- leading dimension 和布局；
- pivot 区、update 区和 map descriptor 引用；
- 允许的 front 上限。

### 4.2 ContributionDesc

描述 child update 到父 front 的贡献：

- source region；
- target region；
- 相对 row/column map；
- 元素数量和 stride；
- child token；
- 数据格式。

### 4.3 KernelDesc

描述 Panel-LU、TRSM 或 GEMM：

- 算子方向；
- A/B/C descriptor 引用；
- tile 坐标和 `M/N/K`；
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

## 5. 命令示例

```text
NODE_BEGIN          node=17 desc=front17 signal=t17_begin
WAIT_DEP            wait=child_tokens(4,9,12)
LOAD_FRONT          desc=front17 signal=t17_loaded
ASSEMBLE_EXTEND_ADD desc=map17 wait=t17_loaded signal=t17_assembled
PANEL_LU            desc=panel17 wait=t17_assembled signal=t17_panel
TRSM_LEFT           desc=f12_17 wait=t17_panel signal=t17_u12
TRSM_RIGHT          desc=f21_17 wait=t17_panel signal=t17_l21
GEMM_SCHUR          desc=schur17 wait=(t17_u12,t17_l21) signal=t17_update
STORE_FACTOR        desc=factor17 wait=t17_panel
STORE_UPDATE        desc=update17 wait=t17_update signal=t17_commit
NODE_COMMIT         node=17 wait=t17_commit
```

## 6. 内存区域

建议在 manifest 中明确区域所有权：

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

## 7. Completion 与错误

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
DEPENDENCY_ERROR
PIVOT_NOT_FOUND
PIVOT_UNSTABLE
NUMERIC_OVERFLOW
QUANTIZATION_SATURATION
PRECISION_RETRY
TIMEOUT
ABORTED
```

控制失败和数值失败必须分离。`status=OK` 只表示命令执行完成，不表示 residual 达标。

## 8. 精度后端

同一命令语义支持：

```text
FP64_GOLDEN
FP32_SYSTEMC
FP32_SYSTEMC_INT32_GEMM
```

在 `FP32_SYSTEMC_INT32_GEMM` 中，`PANEL_LU`、`TRSM`、pivot 和树形依赖仍使用浮点；只有 `GEMM_SCHUR` 通过 `ScaleDesc` 进行局部 INT32 适配。

## 9. 迁移策略

```text
NodeTask + manifest v2
          ↓ adapter
command ring + descriptor table
          ↓
SystemC executor
          ↓ future
RTL executor
```

迁移期间允许同时生成旧 ABI v2 和 command stream。等 SystemC command executor 通过单节点、多节点和损坏输入测试后，才停止扩展旧 `NodeTask` 字段。

