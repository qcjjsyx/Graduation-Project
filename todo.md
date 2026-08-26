# 毕业设计当前推进计划

最后更新：2026-08-26

## 1. 当前目标

具体代码任务以[项目实现任务说明书](docs/项目实现任务说明书.md)为准；本文件保留阶段性推进概览。

六个月内完成一个边界明确、可验证的多前沿稀疏 LU 硬件架构原型：

- 软件完成符号分析、front/任务产物和 command compiler；
- SystemC 完成所有核心模块的显式数据通路、周期模型和分解—求解闭环；
- SystemC 提供 FP32/FP64 浮点主设备路径；
- INT32 只用于 GEMM 调用边界，使用局部 quantize/dequantize；
- 没有 INT32 GEMM RTL 时，使用可替换的 `Int32GemmBehavioral`；
- RTL 优先完成 GCU、HPU、ATU、片上存储接口和关键控制路径；
- 不把全局 INT32/BFP、真实 DDR/PCIe 或完整浮点 LU RTL 作为首版前提。

## 2. 当前事实

### 已有基础

- [x] 软件可以生成稀疏多前沿相关产物；
- [x] 支持数值非对称矩阵和 `pattern(A) ∪ pattern(Aᵀ)` 符号包络；
- [x] 已有消除森林、supernode、front、map 和 ABI v2 产物；
- [x] SystemC 已有任务获取、依赖管理、buffer、DDR 事务、HPU/ATU、写回和树形求解框架；
- [x] 已有 FP64、定点和实验性精度策略代码；
- [x] 已有 GCU、HPU、ATU 等 RTL 原型和测试入口；
- [x] 软件和 SystemC 已有可运行测试与历史实验结果。

### 尚未完成且不能误写为已完成

- [ ] 当前主线尚未完成 command stream executor；
- [ ] 当前 SystemC 数值后端仍有 C++ 行为函数直接参与，尚未全部转换为显式 buffer-driven 算子；
- [ ] 当前没有可用于联调的 INT32 GEMM RTL；
- [ ] 当前 RTL 尚未形成完整 GCU 顶层设备；
- [ ] Panel-LU、TRSM、GEMM 的 SystemC 周期级数据通路仍需按新规范重构；
- [ ] 最终结果尚不能表述为完整浮点求解器 RTL。

## 3. M0：架构冻结（第 1—2 周）

- [ ] 确认正式问题为 `Ax=b`，显式逆矩阵只作为多 RHS 扩展；
- [ ] 确认数值范围为方形、数值可非对称、符号使用对称包络；
- [ ] 确认 `FP64_GOLDEN`、`FP32_SYSTEMC`、`FP32_SYSTEMC_INT32_GEMM` 三种后端；
- [ ] 确认 INT32 量化只发生在 GEMM 调用边界；
- [ ] 确认 front、panel、tile、RHS 数量和 HPU/ATU 最大容量；
- [ ] 确认 ABI v2 是迁移阶段接口，command stream 是目标接口；
- [ ] 确认 SystemC 是完整主模型，浮点 Panel-LU/TRSM 不要求首版 RTL；
- [ ] 确认 GCU/HPU/ATU/存储接口是 RTL 优先模块；
- [ ] 确认论文暂不进入本阶段实现任务。

## 4. M1：命令流和软件适配（第 3—5 周）

- [ ] 定义 command header、schema version、command ID 和 token；
- [ ] 定义 `FrontDesc`、`ContributionDesc`、`KernelDesc`、`ScaleDesc`；
- [ ] 定义 command ring、descriptor table、data image 和 completion queue；
- [ ] 实现 `NodeTask → command stream` 的适配器；
- [ ] 保留 ABI v2 生成路径用于回归，不继续扩张 `NodeTask` 字段；
- [ ] 实现命令长度、版本、地址、依赖和 descriptor 合法性检查；
- [ ] 实现命令级黄金解释器。

## 5. M2：SystemC 硬件模型重构（第 6—10 周）

- [ ] Command Fetch 从 command buffer 读取命令；
- [ ] GCU 通过 token 管理依赖和资源；
- [ ] Buffer/DMA/Assembly 只从 DDR/descriptor/buffer 读取数据；
- [ ] Panel-LU 使用显式 load、pivot、swap、divide、update、writeback 状态；
- [ ] TRSM 使用显式三角依赖、mask、对角处理和完成状态；
- [ ] GEMM 使用显式 tile buffer、K 维累加和写回；
- [ ] Solve Controller 从已提交的因子和 RHS 完成 SystemC 前代/回代；
- [ ] 原有 C++ numeric kernel 降为独立 checker，不再作为设备旁路；
- [ ] 输出 command、node、operation、timeline、memory 和 status trace。

## 6. M3：INT32 GEMM 行为后端（第 11—13 周）

- [ ] 实现 `GemmFp32Reference`；
- [ ] 实现 `PrecisionAdapter`；
- [ ] 实现 `Int32GemmBehavioral`；
- [ ] 支持 tile 内二的幂 scale、舍入、饱和和反量化；
- [ ] 确认 INT32 GEMM 的累加宽度；必要时实现 K 维分块；
- [ ] 统计量化误差、overflow、saturation、cycles 和 bytes；
- [ ] 验证 `FP32_SYSTEMC_INT32_GEMM` 的单节点和小树路径；
- [ ] 保留未来 `Int32GemmRtlWrapper` 的接口。

## 7. M4：控制和地址路径 RTL（第 14—17 周）

- [ ] 将 command schema 接入 GCU parser；
- [ ] 完成 GCU token scoreboard 和完成队列；
- [ ] 完成 buffer/SRAM 风格接口；
- [ ] 补齐 HPU reset、候选边界、tie-break 和 backpressure 测试；
- [ ] 补齐 ATU 初始化、query、swap 和非法访问测试；
- [ ] 统一 RTL/SystemC 的 row width、pivot capacity 和状态语义；
- [ ] 使用同一条短 command stream 做 SystemC/RTL trace 对照。

## 8. M5：端到端验证和实验（第 18—23 周）

- [ ] 单节点闭环：`LOAD → ASSEMBLE → PANEL → TRSM → SCHUR → STORE → COMMIT`；
- [ ] 多节点和多根消除森林闭环；
- [ ] `FP64_GOLDEN` 与 `FP32_SYSTEMC` 数值对照；
- [ ] `FP32_SYSTEMC_INT32_GEMM` 的局部量化误差实验；
- [ ] 量化导致 pivot 变化时记录实际 pivot，不使用黄金 pivot 覆盖；
- [ ] 验证非法 command、损坏 descriptor、buffer 满/空、timeout 和 numeric failure；
- [ ] 扫描 tile、lane、DDR 带宽、延迟、buffer 和调度策略；
- [ ] 分离报告 Host、SystemC、行为模型和 RTL 结果。

## 9. M6：收口材料（第 24—26 周）

- [ ] 固定最终矩阵集合和实验 seed；
- [ ] 复现所有主结果；
- [ ] 生成 residual、backward error、pivot、overflow、量化开销、周期和 bytes 报告；
- [ ] 整理 SystemC/行为模型/RTL 的边界说明；
- [ ] 检查 README、接口文档、验证文档和代码路径一致；
- [ ] 论文在架构和实验冻结后再开始编写。

## 10. 明确不做

- [ ] 不实现任意规模的通用非对称 multifrontal LU；
- [ ] 不将 INT32/BFP scale 传播到整棵消除树；
- [ ] 不把 FP8 作为主路径；
- [ ] 不实现真实 DDR PHY、PCIe 和完整 FPGA 驱动；
- [ ] 不为了证明完整性强行实现单 RHS 树形求解 RTL；
- [ ] 不用黄金数据、预选 pivot 或静默饱和掩盖设备路径失败；
- [ ] 不在论文开始前继续堆叠没有验证的架构功能。

## 11. 当前验收口径

`status=OK` 只表示控制流完成。数值是否可接受，必须同时查看：

```text
relative_residual
componentwise_backward_error
relative_solution_error
pivot_growth / minimum_pivot_ratio
overflow / saturation / retry
cycles / bytes / stall
```
