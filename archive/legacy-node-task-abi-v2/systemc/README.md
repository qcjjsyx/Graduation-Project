# SystemC 多前沿稀疏 LU 系统模型

状态：当前主线说明

## 1. 模型定位

SystemC 是本项目的主硬件架构模型，目标是模拟软件产物驱动下的设备执行过程，包括任务流动、依赖、buffer、DDR 事务、主元、局部计算、写回和求解。

它不是：

- 完整 Verilog RTL 的自动翻译；
- 门级仿真；
- 真实 DDR/AXI PHY 仿真；
- 综合后芯片周期的直接预测。

## 2. 当前代码已经完成的部分

当前 `system_sim` 已经具备：

- artifact loader 和 ABI 二次校验；
- 字节地址 DDR 与事务延迟模型；
- Task Fetch；
- Dependency Scoreboard；
- Buffer Manager；
- FrontAssembler；
- HPU/ATU SystemC 模型；
- Panel-LU、TRSM、GEMM-Schur 的 C++ 行为计算和操作周期估计；
- L/U/P/update 写回；
- 树形前代、回代和 residual 检查；
- VCD、timeline、operation、memory 和 summary 报告。

当前模型仍然存在一个重要边界：部分 front 装配和数值算子直接消费 `SystemMemory` 中的 C++ 容器，尚未完全转换为每个算子都经过显式片上 buffer 和状态机。因此当前结果应称为“系统级功能/事务模型”，不能称为完整 RTL 等价模型。

## 3. 目标三种后端

### `FP64_GOLDEN`

普通 C++ FP64 计算，用于数学参考和 checker，不作为硬件时序结果。

### `FP32_SYSTEMC`

主硬件模型。Panel-LU、TRSM、GEMM、solve 等核心算子使用浮点行为运算，但必须经过：

- command/descriptor；
- front/scratchpad buffer；
- 显式状态机；
- 计算资源和延迟；
- ready/valid 和反压。

### `FP32_SYSTEMC_INT32_GEMM`

只有 GEMM 调用边界进行：

```text
FP tile → quantize → INT32 GEMM behavior → dequantize → FP tile
```

当前没有 INT32 GEMM RTL，因此使用 `Int32GemmBehavioral`。未来有 RTL 时，用相同 transaction 接口替换，不修改上层 GCU 和 front 流程。

## 4. 目标 SystemC 数据流

```text
Command Fetch
  → GCU / Dependency Scoreboard
  → DDR/DMA / Buffer Manager
  → Front Assembly
  → Panel-LU ↔ HPU/ATU
  → TRSM
  → GEMM-Schur / Precision Adapter
  → Factor/Update Writer
  → Solve Controller
  → Golden Checker
```

## 5. 计算模块改造要求

Panel-LU、TRSM、GEMM 和 solve engine 不得直接读取全局黄金对象。必须使用显式 buffer 和接口，并至少体现：

- load；
- pivot/select；
- row swap；
- reciprocal/divide；
- triangular step 或 MAC update；
- writeback；
- done/error。

普通 C++ numeric kernel 只能作为独立参考实现。它可以与 SystemC 共享数学辅助函数，但不能替设备路径写入结果、提前提供 pivot 或绕过 DDR。

## 6. INT32 GEMM 行为模型

`Int32GemmBehavioral` 应模拟未来 GEMM RTL 的请求和响应：

```text
request: command_id, descriptor_id, M/N/K, buffer, stride, scale
response: status, output, cycles, bytes, overflow, saturation
```

模块需要体现 INT32 乘法、至少 INT64/guard-bit 累加、K 维分块、tile 延迟、buffer 反压和写回。若目标 GEMM 只有 INT32 累加，必须缩小 K/tile 支持范围或明确报告溢出风险。

## 7. 求解阶段

SystemC 必须完成最终前代和回代闭环，并读取已经写回、提交的 L/U/P 和 RHS。单 RHS 树形求解首版不要求 RTL；它可以作为 SystemC 的 `SOLVE_FORWARD` 和 `SOLVE_BACKWARD` 命令执行。

## 8. 当前运行命令

当前代码的实际模式仍是 `fp64|fixed|both`：

```bash
cd graduation-code/systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

目标后端名称和 command executor 完成后，再更新命令行参数；在此之前不能把目标模式写成已经可运行。

## 9. 输出口径

每次运行应区分：

- backend；
- matrix/seed/config；
- command、node、operation、timeline、memory；
- pivot、overflow、saturation、retry；
- cycles、bytes、stall；
- residual、backward error 和解误差。

`status=OK` 只表示控制流完成，不表示数值结果达标。

