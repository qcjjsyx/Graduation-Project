# 毕业设计当前任务状态

状态：**T00–T03 已完成，下一阶段从 T04 开始。**

> 本文档只记录当前主线已经完成的工作、验证证据和后续边界，不再按月份或周数安排进度。

## 1. 当前主线

- 软件与 SystemC 的主接口已切换到 **Command / Descriptor v1**。
- 旧 NodeTask / ABI v2 软件、SystemC 执行器及历史优化代码已整体归档到：
  - archive/legacy-node-task-abi-v2/
- 主线软件当前负责：
  - 从 Matrix Market 输入构建统一符号分析结果；
  - 生成 Command、Descriptor、Completion、内存镜像和 manifest；
  - 生成供验证使用的 FP64 reference，reference 不进入 device memory image。
- 主线 SystemC 当前保留：
  - Command / Descriptor / Completion C++ codec；
  - 独立 FP64 reference；
  - 独立 ATU / HPU 模块与测试。
- **SystemC Command 执行器尚未实现**，属于后续 T04/T05。

## 2. 已完成任务

### T00：旧基线归档与主线边界整理 [x]

- [x] 保留旧 ABI v2 的历史实现、文档语义和构建入口，作为可追溯基线。
- [x] 将旧软件流水线、旧 SystemC NodeTask 执行器和历史 optimization 整体迁出主目录。
- [x] 为归档代码提供独立 README 和 SystemC 构建入口。
- [x] 确认主线代码不再依赖归档目录。
- [x] 明确当前主线只使用 Command / Descriptor v1，不再为 ABI v2 增加兼容适配层。

归档位置：

- archive/legacy-node-task-abi-v2/software/
- archive/legacy-node-task-abi-v2/systemc/
- archive/legacy-node-task-abi-v2/optimization/

### T01：Command / Descriptor v1 codec [x]

- [x] 实现 32B little-endian Command record 的 Python/C++ 编解码。
- [x] 实现 64B little-endian Descriptor record 的 Python/C++ 编解码。
- [x] 实现 64B little-endian Completion record 的 Python/C++ 编解码。
- [x] 固定并校验 opcode、status code、NONE、flags 和 reserved 字段。
- [x] 实现 typed DataFormat、Layout、Backend、Direction 字段及区域范围校验。
- [x] 提供 READY / FAILED / UNSIGNALED token fixture。
- [x] 提供 Python/C++ 共用 golden binary fixture 和正反向单元测试。

主要产物：

- graduation-code/software/src/abi/command_schema_v1.py
- graduation-code/systemc/include/command_schema_v1.hpp
- graduation-code/systemc/src/command_schema_v1.cpp
- graduation-code/fixtures/command_schema_v1/

### T02：SystemC FP64 reference [x]

- [x] 实现独立、确定性的 C++ FP64 数值 reference。
- [x] 实现 panel LU。
- [x] 实现左下三角和右上三角 TRSM。
- [x] 实现 GEMM Schur update。
- [x] 实现基于 LU 的线性求解。
- [x] 实现 front-tree factor/solve reference。
- [x] 实现相对残差、分量后向误差和相对解误差统计。
- [x] 提供 JSON fixture、单元测试和 sanitizer 测试。
- [x] 保证 reference 不调用 sc_start()，也不生成设备侧 FP64 镜像。

主要产物：

- graduation-code/systemc/include/fp64_reference.hpp
- graduation-code/systemc/src/fp64_reference.cpp
- graduation-code/systemc/tests/test_fp64_reference.cpp
- graduation-code/systemc/tests/test_fp64_reference_fixture.cpp
- graduation-code/fixtures/fp64_reference/

### T03：Command Compiler [x]

#### 统一前处理

- [x] 读取 Matrix Market 稀疏矩阵并完成输入合法性检查。
- [x] 对非对称输入按 A union A-transpose 构建统一结构图。
- [x] 完成 AMD 风格排序、精确填充图、消去树、supernode、front 和 front forest 构建。
- [x] 删除主线全局 BFP/指数选择/量化前处理。
- [x] 使用 NodeCompileRecord 作为编译 IR，不再生成 NodeTask。

#### Command / Descriptor 生成

- [x] 为每个节点生成显式 assemble、panel factor、TRSM、Schur update、store-update 命令。
- [x] 为根节点生成 solve 命令序列。
- [x] 通过 token 表达子节点 store-update 到父节点 assemble 的依赖。
- [x] 生成 operand/output/completion descriptor，并对描述符类型和 payload 进行校验。
- [x] 生成 Completion 初始化模板；实际运行时状态写回留给后续 SystemC 执行器。

#### 内存规划和产物

- [x] 实现两遍内存规划，先确定节点局部尺寸，再分配全局地址。
- [x] 实现 u64 地址溢出、alignment、区间越界和 region overlap 检查。
- [x] 生成唯一的 FP32 device memory image。
- [x] 将 P 向量固定为 INT32；reference/metrics 保留在 host-side JSON。
- [x] 生成 Command binary、Descriptor binary、Completion binary、memory image 和 manifest。
- [x] manifest 包含 schema/version、区域表、命令统计、token producer/consumer 和节点摘要。
- [x] 对 opcode、flags、descriptor index、token、reserved、region、data format、layout、backend、direction 实施编译期验证。
- [x] 确保相同输入和配置生成字节级一致的 manifest 与 memory image。

主要产物：

- graduation-code/software/src/scheduler/command_compiler.py
- graduation-code/software/src/pipeline/pipeline.py
- graduation-code/software/src/scheduler/planner.py
- graduation-code/software/src/io/manifest.py
- graduation-code/software/src/main.py
- graduation-code/software/tests/

## 3. 当前验证结果

| 验证项 | 结果 |
| --- | --- |
| 软件单元测试 | 49 passed |
| T03 关键模块 mypy | Success: no issues found in 5 source files |
| 主线 SystemC CTest | 3/3 passed |
| 主线 SystemC sanitizer CTest | 3/3 passed |
| 归档 SystemC 独立构建 | 通过 |
| 256 阶真实矩阵编译 | 73 nodes、866 commands、1747 descriptors |
| 256 阶 reference 原始残差 | 5.256e-11 |
| 256 阶 device memory image | 1,400,576 bytes |
| 重复编译确定性 | manifest 和 memory image 字节级一致 |

## 4. 当前未完成内容

以下内容不属于 T00–T03 的已完成范围，不能作为当前能力对外宣称：

- [ ] SystemC 对 Command artifact 和 manifest 的加载。
- [ ] SystemC byte-addressable device memory。
- [ ] Descriptor Reader、Buffer Handle、DMA Engine。
- [ ] 基于 token scoreboard 的 GCU 调度与运行时 Completion 写回。
- [ ] SystemC Panel-LU、TRSM、GEMM、assemble、update 和 solve 执行 kernel。
- [ ] FP32 Command 全链路数值闭环。
- [ ] 局部 INT32 / BFP GEMM。
- [ ] RTL Command Processor、GCU、DMA、HPU/PE 和完整 RTL 仿真闭环。
- [ ] 性能、带宽和面积证据。

## 5. 下一阶段

### T04：SystemC Command 数据面 [ ]

- [ ] 加载 manifest、Command、Descriptor、Completion 和 memory image。
- [ ] 实现统一 byte-addressable memory 与严格 region/bounds 检查。
- [ ] 实现 Descriptor Reader 和轻量 Buffer Handle。
- [ ] 实现带请求/响应、延迟和背压语义的 DMA Engine。
- [ ] 覆盖非法 descriptor、越界访问、错误 data format/layout 和背压场景。

### T05：SystemC Command 控制面 [ ]

- [ ] 实现固定宽度 Command Fetch。
- [ ] 实现 GCU token scoreboard 与 token 状态机。
- [ ] 按 descriptor 解释命令并派发到执行 kernel。
- [ ] 实现 Completion 成功/失败写回、错误传播和完成顺序验证。
- [ ] 建立 Command 级 golden interpreter/reference，对照 SystemC 执行结果。
- [ ] 不重新引入 NodeTask 或 ABI v2 adapter。

后续 T06 及 RTL 阶段只有在 T04/T05 完成并形成可验证闭环后再展开。

## 6. 固定边界

- 主线不恢复全局量化前处理。
- 主线不生成多份不同精度的全局矩阵副本。
- FP64 仅作为 host/SystemC reference，不进入 device memory image。
- 归档目录只用于历史追溯和独立复现，不参与主线构建。
- 新 SystemC 执行器必须直接消费 Command / Descriptor v1，不以旧 ABI v2 为中间层。
- 任何“已完成”项必须同时具备代码、测试和可复现验证证据。
