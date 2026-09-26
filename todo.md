# 毕业设计当前任务状态

状态：**T00–T03 已完成（含主元策略调整与 Command/Descriptor schema v2 演进），下一阶段从 T04 开始。**

> 本文档只记录当前主线已经完成的工作、验证证据和后续边界，不再按月份或周数安排进度。

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
- **SystemC Command 执行器尚未实现**，属于后续 T04/T05。

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

说明：

- `example/*fuv.mat` 不是方阵，不能作为矩阵输入（历史 RHS/向量文件）。
- 当前数值验证仅覆盖 FP64 参考路径；**FP32 静态设备路径的稳定性尚未被完整验证**。
- 本次不依据算法操作数或局部模块测试宣称任何硬件加速比。

## 4. 当前未完成内容

以下内容不属于已完成范围，不能作为当前能力对外宣称：

- [ ] SystemC 对 Command schema v2 artifact 和 manifest 的加载。
- [ ] SystemC byte-addressable device memory。
- [ ] Descriptor Reader、Buffer Handle、DMA Engine。
- [ ] 基于 token scoreboard 的 GCU 调度与运行时 Completion 写回。
- [ ] SystemC Panel-LU（STATIC 固定主元 + 数值检查）、TRSM、GEMM、assemble、update 和 solve 执行 kernel。
- [ ] DYNAMIC_PANEL 的设备执行（HPU/ATU 动态选择与映射），未实现前必须 `POLICY_NOT_SUPPORTED` 拒绝。
- [ ] FP32 Command 全链路数值闭环。
- [ ] 局部 INT32 / BFP GEMM（需重新评估对静态主元的影响）。
- [ ] RTL Command Processor、GCU、DMA、HPU/PE 和完整 RTL 仿真闭环。
- [ ] 性能、带宽和面积证据。

## 5. 下一阶段

### T04：SystemC Command 数据面 [ ]

- [ ] 加载 manifest（schema v2）、Command、Descriptor、Completion 和 memory image。
- [ ] 实现统一 byte-addressable memory 与严格 region/bounds 检查。
- [ ] 实现 Descriptor Reader 和轻量 Buffer Handle。
- [ ] 实现带请求/响应、延迟和背压语义的 DMA Engine。
- [ ] 覆盖非法 descriptor、越界访问、错误 data format/layout 和背压场景。

### T05：SystemC Command 控制面与执行 kernel [ ]

- [ ] 实现固定宽度 Command Fetch。
- [ ] 实现 GCU token scoreboard 与 token 状态机（失败命令不发 READY token）。
- [ ] 按 descriptor 解释命令并派发到执行 kernel；先实现 STATIC 固定主元
      Panel-LU 及其零主元/阈值/非有限检查。
- [ ] DYNAMIC_PANEL 经 HPU/ATU 作为可选能力；未实现时以 `POLICY_NOT_SUPPORTED`
      拒绝执行，不得猜测策略。
- [ ] Completion 成功/失败写回（`pivot_count` 失败时为首个失败列）、错误传播和完成顺序验证。
- [ ] 建立 Command 级 golden interpreter/reference，对照 SystemC 执行结果。
- [ ] 不重新引入 NodeTask 或 NodeTask ABI v2 adapter。

后续 T06 及 RTL 阶段只有在 T04/T05 完成并形成可验证闭环后再展开。

### T04 开始时使用的明确接口

- 二进制契约：`docs/interface/Command_Schema_v2.md`（manifest `abi.version=2`；
  记录 32/64/64B；`FRONT_DESC.body_words[8..9]` = 策略 + 阈值）。
- Python codec：`graduation-code/software/src/command_codec.py`；C++ codec：
  `graduation-code/systemc/include/command_codec.hpp`（`hw::command`）。
- 共用 fixture：`graduation-code/fixtures/command_schema_v2/`。
- 主元语义与失败口径：schema v2 文档 §4–§7；manifest 的 `pivot` 与
  `required_device_capabilities` 块为执行语义来源。
- 数值对照：C++ `hw::reference_fp64::panel_lu_policy` / `factor_front_tree` /
  `solve_front_tree`；host 侧 `graduation-code/software/src/verify/front_reference.py`；
  同策略交叉验证 fixture：`graduation-code/fixtures/reference_tree/`。
- 设备数值数据类型：front/factor/update/RHS/solution FP32，P-vector INT32
  （初值 -1，STATIC 写显式 identity）。

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
