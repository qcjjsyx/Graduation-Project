# SystemC 模型规范

状态：当前主线文档

## 1. 模型定位

SystemC 是本项目的主系统级硬件架构模型。它需要模拟设备的执行过程，但不等同于完整 Verilog RTL、门级仿真或综合后的芯片周期。

SystemC 的目标是同时回答三类问题：

1. 命令和数据是否能完成多前沿分解—求解；
2. GCU、buffer、HPU、ATU 和计算引擎之间的时序/依赖是否正确；
3. 在不同精度、带宽、buffer 和资源配置下，周期和数值误差如何变化。

当前状态：**T04 SystemC Command 数据面已实现**（ArtifactLoader、byte-addressable
DeviceMemory、MemoryService、DescriptorReader、BufferManager、DMA Engine、Trace；
设计权威见 [DataPlane_设计](DataPlane_设计.md)，任务记录见
[T04_SystemC_数据面设计](T04_SystemC_数据面设计.md)）；T05 双 front 控制面已实现
（设计权威见 [ControlPlane_设计](ControlPlane_设计.md)，任务记录见
[T05_SystemC_控制面设计](T05_SystemC_控制面设计.md)），保留显式 test stub
控制流回归并已接入正式数值后端。T06 交付了共享多模式 PE 的硬件设计、内部 task 契约与可执行时序/资源原型
（`include/panel_lu/`、`src/panel_lu/`）；T07 完成正式 FP32 STATIC Panel
数值引擎并接入设备执行闭环；T09 把该引擎扩展为共享计算引擎（顶层左右
TRSM、`compute_group_count × pes_per_group` 分组、设备级 `OnChipArbiter`
统一片上带宽仲裁、完整 L/U 因子保存），设计权威见
[Panel-LU架构设计](../architecture/Panel-LU架构设计.md)、
[T07_Panel_LU数值引擎设计](T07_Panel_LU数值引擎设计.md) 与
[TRSM架构设计](../architecture/TRSM架构设计.md)。T06 合成原型仍显式标记
`architecture_prototype=true`、`numerical_validation=not_performed`。
T10 完成顶层 GEMM_SCHUR 数值执行（FP32 与局部 INT32 行为模型）、统一
`GemmReq/GemmRsp`、运行时 SCALE_DESC 读取与受控 STATIC 主元传播验证，
设计权威见
[../architecture/T10_GEMM_Schur与局部INT32设计.md](../architecture/T10_GEMM_Schur与局部INT32设计.md)。
T11 已完成真实 FP32 assemble、STATIC 单节点/树/forest 分解与两种后端的
独立 FP64 L/U/update/P 验收，见
[T11 集成设计](../architecture/T11_真实装配与树形分解集成.md)。树形 solve 与
DYNAMIC_PANEL 尚未实现，真实后端显式 `POLICY_NOT_SUPPORTED`/依赖失败。
下一步为 T12 完整 Ax=b；T14 HPU/ATU 契约统一后移至 STATIC 数值闭环之后，不阻塞
STATIC 阶段验收。本文其余部分描述目标模型规范。

## 2. 三种后端

### 2.1 `FP64_GOLDEN`

普通 C++/FP64 参考计算，不作为设备仿真。用于：

- 生成或检查 pivot、L/U/P 和 solution；
- 计算 residual、backward error 和解误差（对原始 `A x = b`）；
- 对比 SystemC 输出。

独立 FP64 参考（`graduation-code/systemc/reference/`）具有显式 `PivotPolicy`
（`Static` / `DynamicPanel`）与 `PivotFailure` 语义：`panel_lu_policy()` 表达设备契约
语义（置换范围限于 pivot block）；经典通用稠密部分主元 `panel_lu()` 保留完整数学能力
（update 行可提升为 pivot），它**不是**设备契约。front-tree solve 强制同样的
pivot-block 置换范围。host 侧 Python front-tree 参考只用于显式开启的预检（`PipelineConfig.reference_check`，默认关闭）与检查，
不向设备数据供数。

### 2.2 `FP32_SYSTEMC`

主硬件架构路径。当前已接入的分解算子通过 buffer、状态机和真实 FP32 数据运算执行；普通 C++ tick 与 sc_clock 驱动同一模型。独立 FP64 reference 与局部 FP64 diagnostic 不属于该正式后端；真实设备 solve 尚未接入。

必须具备（T04 已实现数据面部分，命令执行部分属于 T05）：

- command fetch（消费 Command/Descriptor schema v2，manifest `abi.version = 2`）；
- descriptor 读取（T04：运行时经设备事务，无 cache）；
- explicit front/scratchpad buffer（T04：slot + generation、所有权、容量、
  有效范围、端口/延迟）；
- 算子启动、运行、完成和异常状态（T05 及后续 kernel）；
- load/compute/writeback 延迟（T04 已含 DMA 装载/写回延迟与 DDR 服务延迟）；
- ready/valid 和反压（T04：DMA 请求/响应、内存服务队列、buffer 端口）；
- cycle、bytes、资源占用和状态 trace（T04：transactions.csv、buffers.csv、
  summary.json、final_memory_image.bin）。

### 2.3 `FP32_SYSTEMC_INT32_GEMM`

在 `FP32_SYSTEMC` 基础上，只将顶层 `GEMM_SCHUR` 的 tile/K 分块经过（Panel/TRSM 内部 GEMM_UPDATE 仍为 FP32）：

```text
FP tile → quantize → INT32 GEMM behavior → dequantize → FP tile
```

该模式用于模拟 INT32 GEMM 约束。`Int32GemmBehavioral` 与 `PrecisionAdapter` 为设计职责名称，当前由
[`precision_adapter.cpp`](../../graduation-code/systemc/src/panel_lu/precision_adapter.cpp)
函数集合实现，`MultiModePe` 调用 `run_int32_gemm_update`，并非两个独立 C++ 类。
将来有 RTL 后，应保持相同 transaction 契约。

## 3. 目标模块层次与当前实现映射

以下是职责层次示意，不是当前 C++ 类的逐项清单。GoldenChecker 为设备外部参考；HPU/ATU 动态路径和 SolveController 属于后续接入。

```text
SystemC top
 ├─ CommandFetch
 ├─ GCU
 │   ├─ DependencyScoreboard
 │   ├─ BufferManager
 │   └─ ResourceScheduler
 ├─ DDR/MemoryModel
 ├─ AssemblyEngine
 ├─ HPU（DYNAMIC_PANEL 单元）
 ├─ ATU（DYNAMIC_PANEL 单元）
 ├─ 全 front 扫描/尺度归约
 ├─ Panel / TRSM / GEMM Controllers
 ├─ Tile Task Queue / Scoreboard / Dispatcher
 ├─ 多模式 PE 池（16×16 tile，LU/TRSM/GEMM 模式）
 ├─ 局部 INT32 GEMM / Precision Adapter（函数实现，T10 已接入）
 ├─ ResultWriter
 └─ SolveController（后续 T12）
设备外：GoldenChecker（独立 FP64 reference/checker）
```

STATIC 路径不需要 HPU/ATU，运行时旁路其动态映射事务（旁路与综合时裁剪是两种不同
能力）；后续 DYNAMIC_PANEL 路径要求在当前 pivot block 内使用 HPU 搜索主元、ATU 维护逻辑→物理
行映射，当前正式设备尚未实现该路径。

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

目标资源架构以 [多模式 PE 计算基线](../architecture/多模式PE与16x16分块计算基线.md) 为准。Panel 先完成全 front 扫描，再生成 A11 内部 tile 任务。以下为逻辑阶段，不要求逐拍模拟每个 ALU；tile 粒度的执行必须保持这些阶段的数值依赖和外部可观察事件。

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

其中 `PIVOT_CANDIDATE`、`PIVOT_SELECT`、`ROW_SWAP` 只在 DYNAMIC_PANEL 下使用；STATIC
路径不搜索候选、不换行，但仍检查消元后的实际主元。TRSM 至少体现对角处理、三角 mask、
行/列依赖和完成条件。GEMM 至少体现 tile load、K 维累加和 writeback。

### 4.4 必须显式建模延迟和反压

乘法、累加、比较归约、除法/倒数、行映射、DMA 和写回都应有可配置延迟。资源不足、buffer 满或下游未 ready 时，模块必须停顿而不是瞬间完成。

### 4.5 任务粒度 PE 模拟

允许数值功能计算与时序推进分离，但任务时延须由阵列数据流、流水 latency/II、有效维度与装载/排空推导。真实建模输入到齐、共享 PE 排他、有限队列、bank/端口、写回与背压；结果在写回完成前不可见。数据相关失败按实际失败进度返回。不得使用任意固定延迟或提前发布结果，也不得把未综合校准的参数宣称为实测硬件性能。

## 5. INT32 GEMM 行为模型

T10 状态：`PrecisionAdapter` 与 `Int32GemmBehavioral` 语义已实现，只作用于
顶层 `GEMM_SCHUR`（`FP32_SYSTEMC_INT32_GEMM` 编译模式），经真实
`DescriptorReader`/`MemoryService` 读取 `SCALE_DESC`，与 FP32 后端共享同一
PE/dispatcher/时序/写回路径。数据模型表述为 **INT32 storage / 30 effective
quantization bits / INT64 accumulator**；默认对称范围 `[-(2^29-1),+(2^29-1)]`
且 `16·(2^29-1)^2 < 2^62`。设计见
[../architecture/T10_GEMM_Schur与局部INT32设计.md](../architecture/T10_GEMM_Schur与局部INT32设计.md)。

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

## 7. 主元策略契约（Command/Descriptor schema v2）

- 设备执行链消费 Command/Descriptor schema v2（manifest `abi.version = 2`，见
  [../interface/Command_Schema_v2.md](../interface/Command_Schema_v2.md)）；主元策略由
  `FRONT_DESC` 显式声明（STATIC 默认 / DYNAMIC_PANEL 可选），executor 不得根据
  P-vector 内容猜测策略。
- 未来 executor 对未实现的策略必须返回 `POLICY_NOT_SUPPORTED` 并拒绝执行，且不得发布
  READY token；依赖它的命令完成为 `DEPENDENCY_FAILED`。契约违规（非法策略编码、
  P-vector 引用 `NONE`、v1/v2 混合记录）返回 `BAD_COMMAND`/`BAD_DESCRIPTOR`。
- STATIC：固定顺序消元，不搜索候选、不换行，运行时旁路 HPU/ATU；仍检查消元后的实际
  主元（`STATIC_PIVOT_FAILURE` / `PIVOT_UNSTABLE` / `NON_FINITE_VALUE`）。失败命令不
  发布 READY token。
- DYNAMIC_PANEL：主元搜索与行映射限于当前 pivot block 内 `k .. pivot_dim-1` 行（HPU
  搜索、ATU 行映射），tie-break 取最小逻辑行号；跨 pivot block 才能继续时显式失败
  （`PIVOT_NOT_FOUND`）。
- 当前状态：DYNAMIC_PANEL 仅契约表达 + 参考行为，设备执行未实现；T04 数据面
  对 STATIC/DYNAMIC_PANEL 均可加载、解析和搬运数据；T05 控制面已实现并对
  DYNAMIC_PANEL 返回 `POLICY_NOT_SUPPORTED`，STATIC 正式分解路径已实现。
  后移的 T14 及动态数值路径完成前，STATIC 阶段验收不包含动态数值能力。

## 8. 设备执行闭环

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
当前 T04 只交付“descriptor → DMA → scratchpad → 消费者拷贝 → DMA 写回”的
数据面闭环（写回内容是测试数据，不是已求出的因子或解）；上面的节点闭环属于
T05 控制面与后续数值 kernel。

solve 阶段在 SystemC 中通过 `SOLVE_FORWARD` 和 `SOLVE_BACKWARD` 完成，并读取已经提交的因子和 RHS。

## 9. 输出和 trace

每次运行应输出：

- command start/finish；
- 节点就绪、装配、pivot、swap、kernel、写回和 commit 事件；
- DDR read/write bytes；
- buffer occupancy 和 stall；
- pivot、overflow、saturation、retry；
- L/U/P、update 和 solution 的数据快照；
- residual、backward error 和解误差（对原始 `A x = b`）。

## 10. 不能作出的表述

即使 SystemC 运行成功，也不能直接表述为：

- 完整 Verilog 硬件已经实现；
- SystemC 周期数等于真实芯片周期；
- INT32 GEMM 已经完成 RTL 验证；
- FP32/INT32 对所有矩阵都达到 FP64 精度；
- FP32 静态设备路径的数值稳定性已完全验证（当前仅存在 FP64 验证）；
- 已实现任何硬件加速（不得作硬件加速结论）。

正确表述是：

> 建立了命令驱动、buffer 驱动、周期级近似的 SystemC 硬件架构模型，并对 INT32 GEMM 局部混合精度路径进行了行为建模。
