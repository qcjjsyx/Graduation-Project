# SystemC 模型规范

状态：当前主线文档

## 1. 模型定位

SystemC 是本项目的主系统级硬件架构模型。它需要模拟设备的执行过程，但不等同于完整 Verilog RTL、门级仿真或综合后的芯片周期。

SystemC 的目标是同时回答三类问题：

1. 命令和数据是否能完成多前沿分解—求解；
2. GCU、buffer、HPU、ATU 和计算引擎之间的时序/依赖是否正确；
3. 在不同精度、带宽、buffer 和资源配置下，周期和数值误差如何变化。

## 2. 三种后端

### 2.1 `FP64_GOLDEN`

普通 C++/FP64 参考计算，不作为设备仿真。用于：

- 生成或检查 pivot、L/U/P 和 solution；
- 计算 residual、backward error 和解误差；
- 对比 SystemC 输出。

### 2.2 `FP32_SYSTEMC`

主硬件架构路径。所有算子都通过 SystemC 模块、buffer 和状态机执行，数值计算使用 FP32 或 FP64 行为运算，具体精度由配置决定。

必须具备：

- command fetch；
- descriptor 读取；
- explicit front/scratchpad buffer；
- 算子启动、运行、完成和异常状态；
- load/compute/writeback 延迟；
- ready/valid 和反压；
- cycle、bytes、资源占用和状态 trace。

### 2.3 `FP32_SYSTEMC_INT32_GEMM`

在 `FP32_SYSTEMC` 基础上，只将 GEMM tile 经过：

```text
FP tile → quantize → INT32 GEMM behavior → dequantize → FP tile
```

该模式用于模拟现有 INT32 GEMM 约束。没有 Verilog GEMM 时，使用 `Int32GemmBehavioral`；将来有 RTL 后，使用相同 transaction 接口替换。

## 3. 模块层次

```text
SystemC top
 ├─ CommandFetch
 ├─ GCU
 │   ├─ DependencyScoreboard
 │   ├─ BufferManager
 │   └─ ResourceScheduler
 ├─ DDR/MemoryModel
 ├─ AssemblyEngine
 ├─ HPU
 ├─ ATU
 ├─ PanelLuEngine
 ├─ TrsmEngine
 ├─ GemmFpEngine / Int32GemmBehavioral
 ├─ PrecisionAdapter
 ├─ ResultWriter
 ├─ SolveController
 └─ GoldenChecker
```

## 4. 计算模块的真实性标准

### 4.1 不允许直接访问全局黄金对象

Panel-LU、TRSM、GEMM 和 solve engine 只能消费 descriptor、scratchpad、stream 或 memory interface。不能在模块内部直接读取预先装入的 `reference_front_f64` 或完整 C++ front 对象。

### 4.2 必须显式建模片上存储

矩阵 tile 必须经过 load 进入 buffer。buffer 应体现：

- 容量；
- 读写端口；
- bank 或端口冲突；
- 满/空；
- 读写延迟；
- 生命周期和所有权。

### 4.3 必须显式建模算子状态

Panel-LU 至少拆分为：

```text
IDLE
LOAD
PIVOT_CANDIDATE
PIVOT_SELECT
ROW_SWAP
RECIPROCAL_OR_DIVIDE
UPDATE
WRITEBACK
DONE
ERROR
```

TRSM 至少体现对角处理、三角 mask、行/列依赖和完成条件。GEMM 至少体现 tile load、K 维累加和 writeback。

### 4.4 必须显式建模延迟和反压

乘法、累加、比较归约、除法/倒数、行映射、DMA 和写回都应有可配置延迟。资源不足、buffer 满或下游未 ready 时，模块必须停顿而不是瞬间完成。

## 5. INT32 GEMM 行为模型

`Int32GemmBehavioral` 必须具有与未来 RTL 相同的请求和响应语义：

```text
GemmReq {
    command_id
    descriptor_id
    M, N, K
    A/B/C buffer reference
    stride
    scale_a
    scale_b
    accumulate_mode
}

GemmRsp {
    command_id
    status
    output reference
    cycles
    read_bytes
    write_bytes
    overflow_count
    saturation_count
}
```

该模块执行 INT32 乘法，优先使用 INT64 或等价 guard bits 累加。如果实际目标单元只有 INT32 累加，必须采用 K 分块或缩小支持范围，并在报告中说明。

## 6. 量化边界

量化只发生在一次 GEMM 调用内部：

```text
eA = choose_scale(A_tile)
eB = choose_scale(B_tile)
qA = round_and_saturate(A_tile / 2^eA)
qB = round_and_saturate(B_tile / 2^eB)
qC = Int32Gemm(qA, qB)
C  = qC × 2^(eA+eB)
```

`C` 反量化回 FP32/FP64 后再参与 Schur update。不得把 scale 传播为整棵消除树的统一 front 格式。

## 7. 设备执行闭环

单节点必须能够通过：

```text
LOAD
→ ASSEMBLE
→ PANEL_LU
→ TRSM_LEFT / TRSM_RIGHT
→ GEMM_SCHUR
→ STORE_FACTOR / STORE_UPDATE
→ COMMIT
```

整树必须体现 child update 写回完成后父节点才能装配，不能只依赖初始化时预装的 C++ 数据。

solve 阶段在 SystemC 中通过 `SOLVE_FORWARD` 和 `SOLVE_BACKWARD` 完成，并读取已经提交的因子和 RHS。

## 8. 输出和 trace

每次运行应输出：

- command start/finish；
- 节点就绪、装配、pivot、swap、kernel、写回和 commit 事件；
- DDR read/write bytes；
- buffer occupancy 和 stall；
- pivot、overflow、saturation、retry；
- L/U/P、update 和 solution 的数据快照；
- residual、backward error 和解误差。

## 9. 不能作出的表述

即使 SystemC 运行成功，也不能直接表述为：

- 完整 Verilog 硬件已经实现；
- SystemC 周期数等于真实芯片周期；
- INT32 GEMM 已经完成 RTL 验证；
- FP32/INT32 对所有矩阵都达到 FP64 精度。

正确表述是：

> 建立了命令驱动、buffer 驱动、周期级近似的 SystemC 硬件架构模型，并对 INT32 GEMM 局部混合精度路径进行了行为建模。

