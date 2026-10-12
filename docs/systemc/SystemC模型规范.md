# SystemC 模型规范

实现基准：2026-10-11 当前工作区。本文规定执行驱动、事务、数值模式、资源计费和报告语义。
系统连接关系见 [总体架构](../architecture/项目总体架构设计.md)，字段布局见
[Command Schema v2](../interface/Command_Schema_v2.md)。

## 1. 执行对象与时间推进

正式执行器为 `hw::controlplane::CommandDevice`，数据面和计算引擎是由它组合的 C++ 对象。
它们不在内部调用 `wait()`，不自行推进 SystemC 时间。
纯 C++ 有界驱动与 `sc_clock` 测试顶层都调用同一 `CommandDevice::tick()`。
独立 HPU/ATU 使用 `sc_module` 端口模型，详见其模块设计。

一次设备 tick 的固定顺序如下：

1. `DataPlane::begin_tick()` 更新周期，`OnChipArbiter::begin_cycle()` 初始化本拍预算。
2. `tick_memory_and_route()` 完成到期内存事务并将响应投递到客户端 inbox。
3. `update_phase()` 消费响应；推进共享计算引擎一次及各执行单元；更新依赖与完成状态。
4. `dispatch_phase()` 做准入、描述符请求及资源检查，派发可执行命令。
5. `issue_phase()` 推进取指、completion 和 DMA job 发出。
6. `tick_clients()` 推进 DescriptorReader 与 DmaEngine。

共享 `PanelEngine` 每拍只推进一次，Panel/TRSM/GEMM 的多个适配器不会再次 tick 引擎。
单元的 `tick/poll` 非阻塞；模型内状态迁移使用 cycle 判定，外部驱动负责何时进入下一拍。

## 2. 后端与数值模式

| 模式 | 选择位置 | 数据与职责 |
|---|---|---|
| `FP32_SYSTEMC` | 正式设备 CLI | FP32 分解、装配、因子写出和树形求解 |
| `FP32_SYSTEMC_INT32_GEMM` | 正式设备 CLI 与 artifact | 顶层 Schur GEMM 使用局部 INT32，其他计算及存储仍为 FP32 |
| `PanelNumericMode::Fp64Diagnostic` | 显式模块配置 | tile 内用 FP64 诊断计算，结果检查后舍入回 FP32；不自动启用 |
| `TestStub` | 显式测试配置 | 调度、搬运、延迟和错误注入测试，不计算正式数值因子 |
| 独立 FP64 reference/checker | 设备外可执行程序 | 重算参考或验收设备结果，不属于设备 CLI 后端 |

`RequireRealKernels` 要求配置真实计算路径，未支持的请求以失败终态结束，不能自动转为 stub。
FP64 diagnostic 与 INT32 GEMM 组合被拒绝。各运行模式必须记录在报告中。

Panel/TRSM/GEMM 内核使用显式 std::fma 与有限性检查，保持 subnormal 的渐进下溢语义。
树形求解使用 C++ float 乘加/减表达式，融合行为受编译器影响，不保证与 tile 内核逐位相同。
Panel/TRSM 使用对角倒数时检查倒数有限性，求解回代检查 U 对角与除法结果。
主元阈值由 binary32 ratio 与装配后全 front 尺度生成，具体比较精度见 Panel 文档。

## 3. 设备内存与参考隔离

设备只消费 manifest、schema v2 记录和镜像中的 FP32/整数数据。
运行时的取指、描述符、DDR 读写经 MemoryService；片上操作经 handle/lease 或声明的单元 staging。
设备计算不能读取 `front_f64.bin/x_true_f64.bin/original_*_f64.bin` 等参考文件。

启动加载、报告落盘和测试允许检查镜像快照；这些工具访问不替代运行时事务。
引擎的 trace/task record 是 host 观测数据，不能当作额外片上矩阵存储。

## 4. 请求、背压与唯一终态

| 事件 | 语义 |
|---|---|
| `submit=false` | 请求未被接受，调用方保留原请求逐拍重试 |
| `submit=true` | 单元持有该 txn，后续提供唯一可消费终态；非法请求也可被接受后立即失败 |
| 执行结束 | 算术或内部计算阶段结束，结果可能仍未写回 |
| 输出可见 | 真实写入完成；tile 场景还需要版本 commit |
| 终态响应消费 | 上层获得状态并安全释放对应资源 |
| completion 写确认 | 完成记录已写入设备内存，随后才可发布 token |

资源队列满、端口竞争、执行单元忙属于可重试背压；硬容量不足、非法句柄或错误描述符属于终态错误。
响应保持期间 busy、身份和数据稳定。定向 abort 不删除已接受事务的唯一响应，等待在途访问收尾；
全局 reset 则使旧 epoch 的事务与响应失效。

## 5. 存储容量、时延与带宽计费

每个 FP32 front 占 `4*n*n` 字节；默认槽容量 64 KiB，结构上限与槽字节上限分别检查。
求解上界为 `4*(depth*n+2*n+p)`，编译器当前使用 `depth=2`，设备改变深度须重新检查。
PE FIFO、task slot、写回 reservation 和单元 staging 分别声明容量，不使用无界数据副本隐藏资源需求。

DDR 服务时间由 MemoryService 的 startup/beat 参数计费；DMA 另有自己的启动/beat 间隔，
片上授权等待与带宽等待分别建模。装载、执行和写回阶段的周期由有效数据量、流水参数与竞争决定。
这些参数描述本模型的数据通路和服务假设，未经过芯片综合或真实 DDR 时序校准。

片上使用统一读/写两个预算域，每域每拍一个请求方获得 grant，剩余额度不转给另一方。
lease 控制权限、范围与 ready cycle，OnChipArbiter 控制每拍字节数。
DDR、scratch、PE 本地转换流量分别报告，不能直接相加作为 DDR 带宽。

## 6. 失败、取消与复位

数值失败不发布 READY token，不扰动主元、不自动换策略。
依赖失败由 token 传播，未执行后继以 `DEPENDENCY_FAILED` 终结。
STATIC 主元失败是该消元顺序的失败，不能据此推断矩阵奇异。

`abort_node` 定向停止生成与排队工作，等待已接受 DMA/DDR/单元任务排空，再释放 slot/lease。
已成功写入的字节保留。立即错误响应与 abort 交叉时也必须保留一个终态。
`reset()` 清空队列、epoch/generation、版本与上下文，不回滚 DDR；重新执行须使用正确初始镜像。

## 7. 报告与观测

| 文件 | 内容 |
|---|---|
| `final_memory_image.bin` | 执行后设备 DDR 快照 |
| `commands.csv` | 逐命令阶段周期、状态、流量、等待、失败进度与 GEMM 转换统计 |
| `units.csv` / `fronts.csv` | 单元和驻留 front 的区间；求解 finish 字段的统计限制见求解模块 §4.2 |
| `nodes.csv` | 节点 commit 及数值状态 |
| `transactions.csv` / `buffers.csv` | DDR/descriptor/DMA 等事务和 buffer 生命周期 |
| `panel_tasks.csv` | 内部 tile 模式、PE/group、事件周期、版本提交与字节 |
| `summary.json` | batch、后端、资源、DMA/引擎/求解汇总与实际参数 |

设备输出 `control_flow_pass` 与 `numeric_status_pass`，数值校验标记为外部执行。
`numeric_status_pass` 反映命令状态，原方程解是否准确仍需独立 checker。
`stall_window` 是整体驻留时间，细分等待互斥计数；completion 写确认周期另行记录。
原始统计定义见 [控制面设计 §7](控制面设计.md#7-观测与测试入口)。

## 8. 独立 FP64 参考与验收代码入口

| 对象 | 代码入口 | 职责 |
|---|---|---|
| `DenseMatrix/PanelLuResult/FrontInput/FrontFactor/ContributionMap` | [fp64_reference.hpp](../../graduation-code/systemc/reference/include/fp64_reference.hpp) | 独立稠密矩阵、节点因子与映射数据 |
| `panel_lu_policy/trsm_left_lower/trsm_right_upper/gemm_schur/factor_front_tree` | [fp64_reference.cpp](../../graduation-code/systemc/reference/src/fp64_reference.cpp) | 从局部贡献重算 FP64 树分解 |
| `TreeValidation/check_tree_artifact` | [tree_artifact_check.hpp](../../graduation-code/systemc/reference/include/tree_artifact_check.hpp)、[实现](../../graduation-code/systemc/reference/src/tree_artifact_check.cpp) | 初始 artifact 与最终镜像的因子/update/P 验收 |
| `SolveValidation/check_tree_solve_artifact` | [tree_solve_check.hpp](../../graduation-code/systemc/reference/include/tree_solve_check.hpp)、[实现](../../graduation-code/systemc/reference/src/tree_solve_check.cpp) | 从设备因子、workspace 和解计算三角及原方程误差 |

`tree_factor_check` 输出因子验证报告；`tree_solve_check` 输出 `solve_validation.json`。
后者的通过条件包含全部求解命令 OK 与 η_f/η_b 门限，原方程误差和解误差另行报告。
外部参考的数值期望独立于被测函数，工具说明见 [reference README](../../graduation-code/systemc/reference/README.md)。

## 9. 构建与验证入口

构建命令见 [SystemC README](../../graduation-code/systemc/README.md) 与根目录 AGENTS.md。
普通 C++、SystemC 时钟、真实 artifact、失败/取消/复位分别有测试入口；各模块文档列出相关目标。
具体测试断言与运行记录见对应测试源码和 verification 目录。
