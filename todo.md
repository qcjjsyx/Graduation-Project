# 毕业设计当前任务状态

状态：**T00–T07 已完成（含主元策略、schema v2、SystemC 数据面与控制面、Panel-LU 硬件架构/多模式 PE 原型、正式 FP32 STATIC Panel 数值引擎与 T05 接入）；下一步为 T09/T10 顶层 TRSM/GEMM 数值核并复用共享 PE 池；T14 后移至 STATIC 数值闭环之后。**

> 本文档只记录当前主线已经完成的工作、验证证据和后续边界，不再按月份或周数安排进度。
> T06 完成的是硬件设计、可编译内部契约与可执行时序/资源原型；T07 完成的是
> Panel（A11）子系统的正式 FP32 STATIC 数值执行与 T05 接入，**尚未**包含顶层
> TRSM/GEMM/solve 与完整分解/求解闭环，也不能据此宣称 FP32 全链路数值通过、
> 芯片周期或加速比。

## 1. 当前主线

- 软件与 SystemC 的主接口为 **Command/Descriptor schema v2**（记录布局
  32B/64B/64B 不变，manifest `abi.version = 2`；`FRONT_DESC` 显式携带主元
  策略与阈值）。历史 v1 规范与 fixture 保留用于追溯与负面测试。
  schema v2 与已归档的 **NodeTask ABI v2** 无关。
- 主元策略：**默认 STATIC**（软件数值预处理 + 硬件固定主元顺序），
  **DYNAMIC_PANEL 为显式可选能力**（仅 pivot block 内选主元）。
  静态模式旁路 HPU/ATU 动态映射；运行时旁路与综合时裁剪是两种不同能力。
- 主线软件负责：
  - Matrix Market 输入与合法性检查；
  - pow2 行均衡 `D_r` + 全局加权最大乘积匹配行置换 `P_match`
    （等价对数目标 + SciPy 稀疏 LAPJVsp；非 MC64；不完整匹配直接诊断失败）；
  - 在 `B = P_match D_r A` 上重算符号包络、ordering、fill、森林、
    supernode、front 与 map（gather 约定统一：`y[i] = x[p[i]]`）；
  - 生成 Command、Descriptor、Completion、memory image 与 manifest；
  - 生成仅供验证的 FP64 数据（原始 `A`、`b`、golden 解），不进入设备 image；
  - 软件侧 FP64 前树参考（`src/verify/front_reference.py`）是独立验证入口；
    编译期预检与外部 RHS 的 golden 参考解均为**显式选项**（
    `PipelineConfig.reference_check` / CLI `--reference-check`，默认关闭），
    正常编译不以数值求解成功为前提、不为外部 RHS 承担分解成本，可以为
    验证设备主元失败生成合法输入；未执行时参考解文件与指标缺省；参考不向
    设备产物注入结果。
- 主线 SystemC 保留：
  - schema v2 的 Command/Descriptor/Completion C++ codec（含策略字段校验）；
  - 独立 FP64 reference（`PivotPolicy`/`PivotFailure`、`panel_lu_policy`，
    与通用 dense partial-pivot `panel_lu` 明确分离）；
  - 独立 ATU / HPU 模块（DYNAMIC_PANEL 用）与测试；
  - `tools/fp64_reference_smoke` 数值冒烟入口。
- **T04 SystemC Command 数据面已实现**：schema v2 artifact/manifest 加载与交叉
  校验、byte-addressable device memory、运行时 descriptor reader、片上
  buffer（slot + generation、所有权与生命周期）、单通道 DMA（1D/2D、延迟、
  背压、reset）、trace 与最终 snapshot；设计权威见
  `docs/systemc/DataPlane_设计.md`，任务记录见
  `docs/systemc/T04_SystemC_数据面设计.md`。
- **T05 SystemC Command 控制面已实现**：固定宽度取指、token scoreboard、
  双 front 驻留与共享单元池调度、执行派发、completion/token 写回、
  abort/reset 与统计；设计权威见 `docs/systemc/ControlPlane_设计.md`，
  任务记录见 `docs/systemc/T05_SystemC_控制面设计.md`。
- **正式数值 kernel 尚未实现**：Panel-LU/TRSM/GEMM/solve 数值核按后续任务
  实现；T05 以显式 test stub 验证控制流，结果标记
  `execution_backend=test_stub`，不得宣称数值正确性或加速比。
- **T06 共享多模式 PE 架构/原型已交付**：详细设计见
  `docs/architecture/Panel-LU架构设计.md`；内部 task/config/response 契约
  （`include/panel_lu/task_types.hpp`）与可执行时序/资源原型
  （`src/panel_lu/`）覆盖 16×16 tile/尾块、PE 跨模式排他、有界队列与端口/
  写回背压、同目标版本顺序、定向 abort 与 reset 失效、可复现 trace。
  原型标记 `architecture_prototype=true`、`numerical_validation=not_performed`，
  不是数值后端，未接入 T05 `CommandDevice`。

## 2. 已完成任务

### T00：旧基线归档与主线边界整理 [x]

- [x] 保留旧 ABI v2 的历史实现、文档语义和构建入口，作为可追溯基线。
- [x] 将旧软件流水线、旧 SystemC NodeTask 执行器和历史 optimization 整体迁出主目录。
- [x] 为归档代码提供独立 README 和 SystemC 构建入口。
- [x] 确认主线代码不再依赖归档目录。
- [x] 明确当前主线不为 NodeTask ABI v2 增加兼容适配层。

归档位置：

- archive/legacy-node-task-abi-v2/software/
- archive/legacy-node-task-abi-v2/systemc/
- archive/legacy-node-task-abi-v2/optimization/

### T01：Command / Descriptor codec [x]

- [x] 32B Command、64B Descriptor、64B Completion 的 Python/C++ 编解码（little-endian）。
- [x] opcode、status code、NONE、flags、reserved 字段的冻结与校验。
- [x] typed DataFormat、Layout、Backend、Direction 与区域范围校验。
- [x] READY / FAILED / UNSIGNALED token fixture。
- [x] Python/C++ 共用 golden binary fixture 与正反向单元测试。

主要产物（v2 命名，v1 为历史）：

- graduation-code/software/src/command_codec.py
- graduation-code/systemc/include/command_codec.hpp
- graduation-code/systemc/tests/command_codec_tests.cpp
- graduation-code/fixtures/command_schema_v2/（主线）
- graduation-code/fixtures/command_schema_v1/（历史追溯 + 负面测试）

### T02：SystemC FP64 reference [x]

- [x] 独立、确定性的 C++ FP64 数值 reference；panel LU、TRSM、GEMM Schur、LU 求解。
- [x] front-tree factor/solve reference（置换范围限于 pivot block）。
- [x] 相对残差、分量后向误差、相对解误差统计。
- [x] JSON fixture、单元测试与 sanitizer 测试；不调用 sc_start()，不生成设备 FP64 镜像。

主要产物：

- graduation-code/systemc/reference/include/fp64_reference.hpp
- graduation-code/systemc/reference/src/fp64_reference.cpp
- graduation-code/systemc/reference/tests/fp64_reference_tests.cpp
- graduation-code/systemc/reference/fixtures/fp64_reference_v1.json

### T03：Command Compiler [x]

#### 统一前处理

- [x] Matrix Market 稀疏矩阵读取与输入合法性检查。
- [x] 对非对称输入按 A ∪ Aᵀ 构建统一结构图。
- [x] AMD 风格排序、精确填充图、消去树、supernode、front 与 front forest。
- [x] 删除主线全局 BFP/指数选择/量化前处理。
- [x] NodeCompileRecord 作为编译 IR（不再生成 NodeTask）。

#### Command / Descriptor 生成

- [x] 每节点显式 assemble、panel factor、TRSM、Schur update、store-update 命令；根节点 solve 序列。
- [x] token 表达子节点 store-update 到父节点 assemble 的依赖。
- [x] operand/output/completion descriptor 与 payload 校验。
- [x] Completion 初始化模板（运行时写回留给 T04/T05）。

#### 内存规划和产物

- [x] 两遍内存规划 + u64 地址溢出、alignment、越界、region overlap 检查。
- [x] 唯一 FP32 device memory image；P-vector INT32；reference/metrics 留 host 侧。
- [x] Command/Descriptor/Completion binary、memory image、manifest。
- [x] 编译期验证与同输入同配置字节级可复现。

主要产物：

- graduation-code/software/src/scheduler/command_compiler.py
- graduation-code/software/src/pipeline.py
- graduation-code/software/src/memory/planner.py
- graduation-code/software/src/verify/manifest.py
- graduation-code/software/src/main.py

### T04：SystemC Command 数据面 [x]

- [x] 独立 schema v2 ArtifactLoader / DeviceBootConfig：manifest 与 memory_image
  校验（版本、记录大小、区域对齐/重叠/溢出、command/descriptor/completion
  解码、descriptor 引用、FRONT_DESC 策略/阈值、P-vector 哨兵与跨 descriptor
  一致性、静态 batch 元数据）；设备运行目录可只有 manifest + image。
- [x] 统一 byte-addressable DeviceMemory 与 RegionRegistry：u64 偏移、跨
  region/padding 检查、按访问类型对齐、只读控制区域、非法写不改内存、
  零长度成功空操作（不得绕过区域检查）。
- [x] MemoryService：DescriptorReader 与 DMA 共享的单端口 FIFO 服务、固定
  启动/beat 延迟、队列背压、非法访问单终态 AddressFault、reset epoch。
- [x] DescriptorReader 与类型化视图（Region/Front/Factor/Contribution/
  Kernel/Solve/Dependency）：全部经设备事务读取、无 cache；校验引用、形状、
  stride、format、layout、payload；STATIC/DYNAMIC_PANEL 原样保留。
- [x] BufferManager / Scratchpad：slot + generation、owner、容量、有效范围、
  FREE→RESERVED→LOADING→READY→COMPUTING→DIRTY→WRITING_BACK→READY→FREE
  生命周期、端口/占用冲突、错误 owner/过期 handle/重复释放拒绝。
- [x] 单通道 DmaEngine：明确 request/response + valid/ready、一个活动传输、
  顺序完成、1D/2D（offset/stride/尾块）、非法请求无部分写入、响应稳定、
  reset 中断语义。
- [x] 双 buffer ping-pong：DMA 装载 B 时消费者可读 A，显式所有权移交。
- [x] Trace/统计：transactions.csv、buffers.csv、summary.json、
  final_memory_image.bin；请求等待、内存服务、响应背压分开计数。
- [x] 测试：普通 C++ 加载/内存/buffer/服务测试；SystemC 时序测试；
  真实 artifact 集成（STATIC/DYNAMIC_PANEL、单节点/多节点/多根/256 阶、
  外部 RHS 且 `reference_check=False`）；Release 与 sanitizer CTest 8/8。

主要产物：

- graduation-code/systemc/include/dataplane/
- graduation-code/systemc/src/dataplane/
- graduation-code/systemc/src/dataplane_smoke.cpp
- graduation-code/systemc/tests/dataplane_core_tests.cpp
- graduation-code/systemc/tests/dataplane_systemc_tests.cpp
- graduation-code/systemc/tests/dataplane_artifact_tests.cpp
- graduation-code/systemc/tools/t04_fixture_gen.py
- docs/systemc/DataPlane_设计.md
- docs/systemc/T04_SystemC_数据面设计.md（任务记录）

### T05：SystemC Command 控制面（双 front 并行）[x]

- [x] CommandFetch：32B 固定宽度取指、有界 FIFO/窗口、valid/ready、
  运行时故障（截断/非法 opcode）与 loader 校验分离测试。
- [x] TokenScoreboard：PENDING/READY/FAILED、唯一生产者、单次终态、
  NONE 不入表、失败持续传播。
- [x] 双 front：`max_resident_fronts=1/2`、共享单元池
  （panel/trsm/gemm/solve/assemble 数量可配）、FrontContext 生命周期、
  窗口准入门控（未准入节点不堵窗）、原子资源获取与回滚。
- [x] 软件命令交错（交错宽度 2）+ NODE_BEGIN 准入依赖 + manifest/验证器
  同步（producer<consumer、token 稠密、completion 槽=command_id 不变）。
- [x] 执行接口 ExecRequest/ExecResponse 与可替换单元；TestStubUnit 显式
  启用（确定性延迟、busy、响应背压、测试变换、错误注入、多实例、epoch）；
  `RequireRealKernels` 启动即拒绝。
- [x] 主元契约：策略只取 FRONT_DESC；DYNAMIC_PANEL → POLICY_NOT_SUPPORTED；
  STATIC 成功写 identity P-vector 并等写完成；注入失败传播三类数值失败码。
- [x] 全 opcode 控制路径（NODE_BEGIN/LOAD/ASSEMBLE/PANEL/TRSM/GEMM/
  STORE_FACTOR(L/U 布局)/STORE_UPDATE/COMMIT/SOLVE/ABORT_NODE）。
- [x] Completion/token 可见性、失败与独立清理、abort/reset、统计与
  commands/units/fronts.csv + summary.json。
- [x] 测试：core 203 checks、systemc 31 checks（sc_clock、重叠证据、周期
  对比）、artifact 354 checks（真实 artifact + 独立解释器对照 + 注入 +
  周期对比）；CTest 12/12（Release 与 sanitizer）。

主要产物：

- graduation-code/systemc/include/controlplane/
- graduation-code/systemc/src/controlplane/
- graduation-code/systemc/src/command_device_smoke.cpp
- graduation-code/systemc/tests/controlplane_core_tests.cpp
- graduation-code/systemc/tests/controlplane_systemc_tests.cpp
- graduation-code/systemc/tests/controlplane_artifact_tests.cpp
- graduation-code/systemc/tests/control_reference.hpp
- graduation-code/systemc/tests/control_test_artifact.hpp
- docs/systemc/ControlPlane_设计.md
- docs/systemc/T05_SystemC_控制面设计.md（任务记录）

### T06：Panel-LU 硬件架构与多模式 PE 接口/时序原型 [x]

- [x] 详细设计 `docs/architecture/Panel-LU架构设计.md`：
  controller→tile task generator/scoreboard→shared dispatcher→PE pool→
  tile writeback→command completion；16×16 阵列、角除法/倒数通路、反馈
  链路、输入 FIFO、目标累加缓冲、task slot/双缓冲；四种模式数据流；
  端口/带宽/容量/latency/II；依赖与并行条件；有界推进与公平性；同目标
  tile 顺序、输出版本与可见性；全 front 扫描依赖；packed LU/单位对角/
  STORE_FACTOR 关系；FP32 舍入/FMA/subnormal/阈值乘积规则；
  `descriptor.tile_size` 能力检查；参数依据与限制（不挪用论文综合结果）。
- [x] 内部 task/config/response 契约（复用 `StatusCode`/`BufferHandle`，
  不改外部 schema v2 二进制布局）：`include/panel_lu/task_types.hpp`、
  `tile_kernels.hpp`、`tile_store.hpp`、`pe_model.hpp`、
  `tile_dispatcher.hpp`、`panel_controller.hpp`、`panel_lu_prototype.hpp`。
- [x] 可执行时序/资源原型（`src/panel_lu/`、
  `src/panel_lu_prototype_smoke.cpp`）：有界 dispatcher 队列、`pe_count`
  共享多模式 PE、一 PE 一任务、装载/执行/写回/完成事件、端口与写回背压、
  同目标版本提交、定向 abort、reset/epoch 旧响应失效、CSV/JSON trace。
- [x] 测试：`panel_lu_prototype_tests` 871 checks、`panel_lu_systemc_tests`
  48 checks（sc_clock 逐周期与纯 C++ tick 模型对照）、p=23 尾块与双 front
  混合 trace smoke；期望来自独立手算与资源不变量，不调用被测时延函数自证。
- [x] T07 接入清单与未决项见设计文档 §9（TestStubUnit 类型绑定、独立 kind
  池→共享 PE 池、任务汇聚、buffer ownership、定向取消、统计与最小真实
  tile 原语复用；STORE_FACTOR 单位对角补 1、bank 细化、容量上限、失败结果
  可见性列为未决项）。

主要产物：

- graduation-code/systemc/include/panel_lu/
- graduation-code/systemc/src/panel_lu/
- graduation-code/systemc/src/panel_lu_prototype_smoke.cpp
- graduation-code/systemc/tests/panel_lu_prototype_tests.cpp
- graduation-code/systemc/tests/panel_lu_systemc_tests.cpp
- docs/architecture/Panel-LU架构设计.md
- docs/systemc/T05_SystemC_控制面设计.md（任务记录）

### 本轮：主元策略调整与 schema v2 演进 [x]

#### 数值预处理与置换语义

- [x] 符号分析之前的全局一对一加权匹配行置换（最大乘积目标，等价对数目标求解；仅选有效非零；校验双射/有限值/零元；无完整匹配时诊断失败、不产出产物）。
- [x] 明确变换语义：`B = P_match D_r A`、`A_hat = Q^T B Q`、
  `b_hat = Q^T P_match D_r b`、`x = Q x_hat`；统一 gather 约定并全链路一致。
- [x] 在 B 上重建 `pattern(B) ∪ pattern(B^T)` 并重算 ordering/fill/forest/supernode/front/map。
- [x] 分别记录 `P_match`、结构排序 `Q` 与组合 RHS 映射；残差与后向误差针对原始 `Ax=b`。
- [x] 非对称 + 非恒等匹配 + 非恒等排序的测试（能发现置换方向错误与错误地把行置换用于解）。

主要产物：

- graduation-code/software/src/matching.py
- graduation-code/software/tests/test_matching.py
- graduation-code/software/tests/test_transform_semantics.py

#### 主元策略契约

- [x] 配置分层：Host 预处理 `none / weighted_matching`（默认 weighted_matching）；
  设备策略 `STATIC / DYNAMIC_PANEL`（默认 STATIC）；消融可显式关闭匹配。
- [x] schema v2：`FRONT_DESC.body_words[8]` 策略、`[9]` 阈值（binary32 位）；
  `PANEL_LU` 关联 `FRONT_DESC`；不再生成 panel `KERNEL_DESC`。
- [x] P-vector：INT32、长度 pivot_dim、gather 语义、初值哨兵 -1、STATIC 写显式
  identity、引用不允许 NONE；全零数组永不解释为 identity；设备局部置换与
  全局预处理映射明确区分。
- [x] 静态失败策略：检查实际消元后主元（零→`STATIC_PIVOT_FAILURE`、过小→
  `PIVOT_UNSTABLE`、NaN/Inf→`NON_FINITE_VALUE`），阈值尺度固定为
  `front_pre_elimination_max_abs`；失败不发 READY、不静默替换、不中途切换策略。
- [x] 阈值唯一有效值：配置/manifest/reference/descriptor 共用一次性规范化
  的 binary32 阈值；拒绝下溢为 0 或舍入到 1.0 的输入。
- [x] 新状态码 `POLICY_NOT_SUPPORTED`（executor 未实现/不支持策略时拒绝执行）。
- [x] manifest 记录：schema 版本、预处理方法与参数、主元策略、缩放与各类置换
  语义、静态失败策略、所需设备能力、独立验证用原始问题信息。
- [x] manifest 严格 JSON（禁 NaN/Inf 常量）；P-vector 校验沿实际 descriptor
  引用检查 INT32/容量/哨兵内容与同节点 FRONT/FACTOR/SOLVE 一致性。

#### FP64 reference 与测试

- [x] C++ reference 增加 `PivotPolicy`/`PivotFailure`/`panel_lu_policy`/`factor_front`；
  保留通用 dense partial-pivot `panel_lu`（能力不缩水，非设备契约）。
- [x] 软件侧 FP64 前树参考（STATIC/DYNAMIC_PANEL）用于预检与校验。
- [x] 覆盖：无需交换成功；需预处理行交换才能成功；`[[1,1,0],[1,1,1],[0,1,1]]`
  （对角非零但消元中零主元，非奇异；静态失败、panel 内动态成功）；跨 pivot
  block 明确拒绝；近零主元、NaN/Inf、确定性 tie-break；多节点与多根 forest
  静态求解与原方程误差。
- [x] **同策略整树 Python/C++ 交叉验证**：共享 fixture
  `graduation-code/fixtures/reference_tree/`（9 个用例，覆盖两策略、失败
  分类、多根 forest、block 内换行），两侧复现同一 `x_hat`、局部 P-vector
  与失败分类；C++ reference 新增 `factor_front_tree`（装配+策略分解，
  置换范围与 `solve_front_tree` 一致）；`fp64_reference_smoke` 增加
  `static`/`dynamic_panel` 设备契约模式（`generic` 仍为独立求解对照）。
- [x] schema v2 fixture（两种策略）+ 负面测试（非法策略、错误版本、非法
  P-vector 引用、混合格式/v1 记录拒绝）。

主要产物：

- graduation-code/software/src/command_codec.py（v2）
- graduation-code/software/src/config.py（MatchingConfig、PivotPolicyConfig）
- graduation-code/software/src/verify/front_reference.py
- graduation-code/systemc/reference/include/fp64_reference.hpp（policy API）
- graduation-code/systemc/reference/tools/fp64_reference_smoke.cpp
- docs/interface/Command_Schema_v2.md

## 3. 当前验证结果

| 验证项 | 结果 |
| --- | --- |
| 软件单元测试 | 107 passed |
| 关键模块 mypy | 无错误（command_codec/pipeline/matching/command_compiler/manifest/front_reference/config/io） |
| 主线 SystemC CTest | 3/3 passed |
| 主线 SystemC sanitizer CTest | 3/3 passed |
| 256 阶真实矩阵编译（默认 matching+STATIC） | 69 nodes、max front 75、818 commands、1582 descriptors、image 1,472,984 B |
| 256 阶 host 参考残差 / 后向误差 | 1.56e-10 / 6.65e-14 |
| 576 阶真实矩阵编译 | 134 nodes、max front 166、1598 commands、3077 descriptors、image 10,356,400 B |
| 576 阶 host 参考残差 / 后向误差 | 1.00e-7 / 3.77e-12 |
| 1024 阶真实矩阵编译 | **触发既有容量上限**：node 190 front 264 > max_front_size 256，如实报告、不放宽；显式 `--max-front-size 300` 后 212 nodes、max front 298、2534 commands、4871 descriptors、image 21,957,072 B，残差 1.23e-5 / 后向误差 8.3e-15 |
| 独立 C++ FP64 reference 数值冒烟（原始 A、b） | n=256 相对残差 3.1e-17 / 后向 1.0e-12 / 解误差 1.6e-7；n=576 5.0e-17 / 1.7e-11 / 5.6e-5；n=1024 2.8e-17 / 1.6e-6 / 2.4e-6 |
| 重复编译确定性 | manifest 与 memory image 字节级一致（复核） |
| 同策略整树 Python/C++ 交叉验证 | 9 个共享 fixture 用例两侧全部一致（x_hat、局部 P-vector、失败分类） |
| T04 主线 SystemC CTest | 8/8 passed（3 个既有 + t04_generate_fixtures + core + systemc + artifact + smoke） |
| T04 sanitizer CTest | 8/8 passed（ASan+UBSan） |
| T04 数据面测试规模 | core 144 checks、systemc 187 checks、artifact 90 checks |
| T04 真实 artifact 集成 | 单节点/22 节点/2 根森林/DYNAMIC_PANEL/256 阶；256 阶 image 1,472,984 B、最大搬运 tile 67×67 = 17,956 B、scratchpad 容量 65,536 B |
| T05 主线 SystemC CTest | 12/12 passed（含 T04 回归），sanitizer 12/12 |
| T05 控制面测试规模 | core 203 checks、systemc 31 checks、artifact 354 checks |
| T05 双 front 重叠证据 | panel/gemm 单元区间交叠（units.csv）：node0 [457,757) ∩ node1 [537,837) 等 |
| T05 周期对比（test_stub） | 两根 artifact：1front/1unit=7595 → 2front/2unit=6107；小例子 1048 → 1021 |
| T05 真实 artifact 控制流 | 7/259/110/259(DYNAMIC 全拒绝)/818 commands 全部终态并与独立解释器一致 |
| T06 主线 SystemC CTest | 16/16 passed（12 个既有 + panel_lu 4 项） |
| T06 sanitizer CTest | 16/16 passed（ASan+UBSan） |
| T06 原型测试规模 | prototype 871 checks、SystemC 48 checks |
| T06 尾块/混合 trace | p=23 pe=1 尾块 5 任务；p=40 双 front 双 PE 混合模式全部提交 |
| T06 手算周期证据 | p=16 LU_DIAG exec 136、wb_done 199；TRSM_RIGHT 57 拍（div 敏感）；p=32 GEMM 16³ exec 34、wb_done 496；失败列 1/1/0 分别 10/10/1 拍 |
| T07 主线 CTest | 18/18 passed（16 既有 + panel_lu_real_tests / panel_lu_real_systemc_tests） |
| T07 测试规模 | real 206 checks、real SystemC 10 checks |
| T07 数值证据 | n=p ∈ {4,15,16,17,23} L/U 对 FP64 STATIC 参考 max diff < 5e-3（实测 ~1e-7），L·U 重构 A11，P-vector identity |
| T07 边界证据 | n=17,p=16：A11 与参考一致、A12/A21/A22 字节不变、顶层 TRSM/GEMM/solve `POLICY_NOT_SUPPORTED`（无 stub 回退） |

说明：

- `example/*fuv.mat` 不是方阵，不能作为矩阵输入（历史 RHS/向量文件）。
- 当前数值验证仅覆盖 FP64 参考路径；**FP32 静态设备路径的稳定性尚未被完整验证**。
- 本次不依据算法操作数或局部模块测试宣称任何硬件加速比。

## 4. 当前未完成内容

以下内容不属于已完成范围，不能作为当前能力对外宣称：

- [ ] SystemC Panel-LU（STATIC 固定主元 + 数值检查）、TRSM、GEMM、assemble、update 和 solve 正式执行 kernel（后续任务）。
- [ ] DYNAMIC_PANEL 的设备执行（HPU/ATU 动态选择与映射），未实现前必须 `POLICY_NOT_SUPPORTED` 拒绝（后续）。
- [ ] FP32 Command 全链路数值闭环（当前控制流正确性已闭环，数值由 test stub 透传/测试变换）。
- [ ] 局部 INT32 / BFP GEMM（需重新评估对静态主元的影响）。
- [ ] RTL Command Processor、GCU、DMA、HPU/PE 和完整 RTL 仿真闭环。
- [ ] 性能、带宽和面积证据。

## 5. 下一阶段

### T07：Panel-LU 数值引擎（STATIC FP32 + T05 接入）[x]

- [x] 正式 FP32 tile 原语（`panel_lu/tile_kernels.hpp`，命名空间
  `hw::panel_lu::kernels`）：STATIC packed LU_DIAG（消元后实际主元检查、
  倒数溢出列报告）、单位下三角 TRSM_LEFT、非单位上三角 TRSM_RIGHT
  （倒数预计算+前向代入）、FMA GEMM_UPDATE；供 T09/T10 复用。
- [x] 全 front 扫描（n×n 含 A12/A21/A22，精确 FP32 maxabs + 有限值检查，
  共享读预算轮转仲裁）与 binary64 阈值乘积。
- [x] 共享多模式 PE 池上的 A11 16×16 步屏障 PanelController（LU_DIAG /
  左右 TRSM / GEMM_UPDATE、尾块有效维度与 mask），真实 front 经
  BufferManager lease 读写（`PanelEngine`；T06 的队列/task slot/FIFO/
  累加器/写回背压全部保留）。
- [x] T05 接入：`PanelExecutionUnit`（共享引擎、每拍只 tick 一次）、
  `FactorStoreUnit`（STORE_FACTOR 真实 L 单位对角/U mask 分块转换）、
  `IExecutionUnit::abort_txn` 定向取消、`enable_real_panel` 能力显式声明、
  未绑定 kind 返回 `POLICY_NOT_SUPPORTED` 不落 stub；P-vector 仍走 PostExec。
- [x] 测试：`panel_lu_real_tests` 206 checks（n=p ∈ {4,15,16,17,23}
  与 FP64 STATIC 参考对照、n>p 外围块字节不变、零/消元零/阈值/NaN 失败、
  双 front 排他与定向 abort、扫描期 abort、reset、共享读带宽与队列/写回
  背压）、`panel_lu_real_systemc_tests` 10 checks；CTest 18/18
  （Release 与 ASan+UBSan），既有 T04/T05/T06 回归无回退。
- [ ] 顶层 TRSM_LEFT/TRSM_RIGHT/GEMM_SCHUR 数值执行（T09/T10，复用
  `kernels` 与同一 dispatcher/PE 池；当前显式拒绝）。
- [ ] 完整分解/求解闭环、L21/U12 持久化、DYNAMIC_PANEL、INT32、RTL。

主要产物：

- graduation-code/systemc/include/panel_lu/panel_engine.hpp
- graduation-code/systemc/src/panel_lu/panel_engine.cpp
- graduation-code/systemc/include/controlplane/panel_exec_unit.hpp
- graduation-code/systemc/src/controlplane/panel_exec_unit.cpp
- graduation-code/systemc/tests/panel_lu_real_tests.cpp
- graduation-code/systemc/tests/panel_lu_real_systemc_tests.cpp
- docs/systemc/T07_Panel_LU数值引擎设计.md

### T06/T08：硬件设计与共享 PE 接口/时序原型 [x/T06 完成]

已确认目标：[共享多模式 PE、16×16 tile 与任务粒度模拟](docs/architecture/多模式PE与16x16分块计算基线.md)；保留全 front 扫描和主元冻结语义。

- [x] T06 完成 Panel-LU 硬件架构、共享 PE/task 契约、可编译接口及可执行时序/资源原型，验证多模式争用、tail、背压与定向 abort；不宣称正式数值完成。设计见 `docs/architecture/Panel-LU架构设计.md`，原型与测试见 `include/panel_lu/`、`src/panel_lu/`、`tests/panel_lu_*`。
- [x] 不替换 T05 生产 backend；T07 接入清单（类型绑定、共享 PE 池、任务汇聚、buffer ownership、定向取消、统计）写入设计文档 §9。
- [ ] 协调 T08 的 Panel/TRSM/GEMM/solve 职责边界；设计评审后再进入数值核编码。
- [ ] STATIC 设计和实现不等待 T14；DYNAMIC_PANEL 保留为后续显式扩展，未实现时继续返回 `POLICY_NOT_SUPPORTED`。

### 正式数值 kernel（T07/T09/T10 → T11/T12）[ ]

- [ ] 实现 STATIC 固定顺序 Panel-LU 数值核（含消元后实际主元检查）并接入
      `IExecutionUnit`（替换 test stub 的对应槽位）。
- [ ] T07 提供 Panel 所需的最小 tile LU/TRSM/GEMM 原语；T09/T10 复用并接入顶层 TRSM/GEMM controller，共享同一批物理 PE；assemble/solve 后续接入。
- [ ] 每接入一个 kernel 重新评估其对控制流统计、错误语义与（INT32 路径）
      `STATIC` 主元触发率的影响。
- [ ] FP32 全链路数值闭环与 FP64 residual/backward 对照（沿用验证口径）。

### 后移的 T14、DYNAMIC_PANEL 扩展及 RTL 阶段 [ ]

- [ ] T11/T12 先形成 STATIC FP32 分解—求解闭环，再展开 T14 HPU/ATU SystemC 行为与 RTL 契约统一。
- [ ] T14 完成后接入动态数值路径，补充 T07/T09/T11/T12 的 DYNAMIC_PANEL 验收；独立 HPU/ATU 测试不代表设备动态路径已完成。
- [ ] T13/T15 仍依赖 T14；任务编号已重排，原有交付要求不变，T14 保持未完成状态。
- [ ] T10 可先接入 FP32，再完成局部 INT32；各任务的阶段验收与完整完成状态分开记录。完整依赖以 `docs/项目实现任务说明书.md` 为准。

### 后续 kernel/RTL 可直接使用的 T04/T05 接口

- 设备运行目录加载：`hw::dataplane::load_artifact(device_dir)` →
  `LoadedArtifact { DeviceBootConfig config; image; }`（校验 schema v2、
  region、descriptor、P-vector、completion 模板；不要求 reference 文件）。
- 内存与事务：`DeviceMemory`（region/权限/对齐/零长度语义）、
  `MemoryService`（`submit(request, issue_cycle)`、`tick(cycle)`、
  `try_pop_response`、`reset`、epoch/统计）。
- 运行时 descriptor：`DescriptorReader::request(ReadKind, descriptor_id, cycle)`
  + `poll`，产出 `RegionView/FrontView/FactorView/ContributionView/
  KernelView/SolveView/DependencyView`；无 cache，全部计入访存。
- 片上 buffer：`BufferManager`（allocate/begin_load/finish_load/
  begin_write/end_write/begin_read/release_lease/begin_writeback/
  finish_writeback/transfer_owner/release/abort_lease/reset，
  slot+generation、owner、容量、有效范围、端口冲突）。
- 数据搬运：`DmaEngine`（单通道、一个活动传输、DMA request/response
  valid/ready、1D/2D、背压、reset），通过 `DataPlane::run_dma_request`
  或自行驱动握手。
- 编排与 trace：`DataPlane`（固定 tick 顺序、内存响应路由）、
  `TraceRecorder`（transactions.csv/buffers.csv）、`run_dataplane_selftest`。
- 契约与 fixture：`docs/interface/Command_Schema_v2.md`；
  `fixtures/command_schema_v2/`；`tools/t04_fixture_gen.py` 生成设备运行目录。
- 设备数值数据类型：front/factor/update/RHS/solution FP32，P-vector INT32
  （初值 -1，STATIC 写显式 identity）。
- 数值对照：C++ `hw::reference_fp64::panel_lu_policy` / `factor_front_tree` /
  `solve_front_tree`；host 侧 `graduation-code/software/src/verify/front_reference.py`；
  同策略交叉验证 fixture：`graduation-code/fixtures/reference_tree/`。
- 控制面（T05）：`CommandDevice`（`start/tick/run/abort_node/reset/
  inject_failure/write_reports`）、`IExecutionUnit`/`ExecRequest`/`ExecResponse`
  （kernel 接入点，替换 `TestStubUnit`）、`TokenScoreboard`、`DmaArbiter`、
  `CompletionWriter`；控制流期望可由 `tests/control_reference.hpp` 复算。

## 6. 固定边界

- 主线不恢复全局量化前处理；不生成多份不同精度的全局矩阵副本。
- FP64 仅作为 host/SystemC reference，不进入 device memory image；reference
  不向设备执行数据注入 pivot/factor/solution。
- 归档目录只用于历史追溯和独立复现，不参与主线构建。
- 新 SystemC 执行器必须直接消费 Command/Descriptor schema v2，不以 NodeTask
  ABI v2 为中间层；不新增旧 NodeTask 兼容层。
- 首版不实现主元扰动、自动迭代求精、运行途中自动切换主元策略。
- 静态消元失败不是奇异性结论；不做微小主元的静默替换。
- 任何“已完成”项必须同时具备代码、测试和可复现验证证据。
