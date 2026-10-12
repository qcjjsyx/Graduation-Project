# HPU / ATU 模块设计

实现基准：2026-10-11 当前工作区。本文说明独立主元选择与地址映射模块的接口、状态和行为。
数学背景见 [矩阵求解理论 §6](../theory/矩阵求解理论基础.md)。

## 1. 当前模块与正式路径的关系

| 对象 | 代码入口 | 当前用途 |
|---|---|---|
| SystemC `hw::HPU` | [hpu.hpp](../../graduation-code/systemc/include/hpu.hpp) | 独立候选收集、绝对值归约、结果握手 |
| SystemC `hw::ATU` | [atu.hpp](../../graduation-code/systemc/include/atu.hpp) | 独立 identity 初始化、逻辑行到物理行查询与交换 |
| 独立连线演示 | [atu_hpu_demo.cpp](../../graduation-code/systemc/src/atu_hpu_demo.cpp) | 展示模块事务；不执行多前沿动态 LU |
| HPU RTL | [hpu_top.sv](../../graduation-code/hardware/HPU/hpu_top.sv)、[比较树](../../graduation-code/hardware/HPU/hpu_core_tree.sv) | 候选缓冲、比较归约与输出状态机 |
| ATU RTL | [ATU.sv](../../graduation-code/hardware/ATU/ATU.sv) | 映射表、初始化、cfg 写、swap 与查询 |

正式 `PanelEngine::build_context` 遇到 DYNAMIC_PANEL 仍返回 POLICY_NOT_SUPPORTED。
正式设备尚未调用这些模块执行动态分解/求解。
STATIC 路径写显式 identity P-vector，不调用这些独立模块。

## 2. HPU 的现有事务

状态按 `IDLE → LOAD → SELECT → OUT → IDLE` 推进。
输入只在 `in_valid && in_ready` 时接受，记录数值与逻辑行号；`in_last` 或候选容量结束收集。
选择阶段比较有符号整数的绝对值，相同幅值保留较早输入候选。
SystemC 使用 64 位整数计算幅值，避免 INT32_MIN 取绝对值溢出；RTL 的比较树按候选输入顺序组织。
SystemC 候选容量为 256，逻辑行号宽 9 位；较早候选优先不等于数值上最小逻辑行号优先。

结果通过 `pivot_valid && pivot_ready` 消费。ready 为 0 时保持数据和 valid；
成功握手的时钟沿清 valid，不能在下一拍重复接受同一结果。
失败搜索也需要有效终态结果并带 `pivot_fail`，避免等待一个永远不会产生的成功候选。

现有候选接口为 signed INT32，并非 FP32 比较。
FP32 的符号、指数、尾数及 NaN/Inf 不能直接按有符号整数编码的幅值比较。
当前原型没有定义正式 FP32 候选排序、NaN/Inf/阈值检查或带 txn/epoch 的取消接口，
不能直接作为 PanelEngine 的动态主元执行后端。

## 3. ATU 的现有事务

ATU 保存逻辑行到物理行的 P-vector：identity 初始化写入自身行号，swap 交换两个映射值，
query 返回指定逻辑行的物理行。它改变索引关系，不能单独完成矩阵数值行交换。
后续 Panel/TRSM/因子保存和求解必须使用一致映射。

SystemC 原型支持有限行数初始化、查询与交换；RTL 还支持 cfg 写口。
这些端口、行号宽度和初始化/查询时序并不完全相同，当前没有共用的逐拍事务契约。

RTL 当前采用 pivot 写优先：同拍 `pivot_req_valid` 时 cfg_ready 拉低，
cfg 只有 `cfg_we && cfg_ready` 才算接受。调用方保持尚未接受的 cfg 请求，稍后重试。
查询还要遵守初始化与写操作的互斥条件。cfg 请求未接受时保持输入并重试。

## 4. 实现结构与验证入口

SystemC HPU 在 SELECT 阶段遍历已接受候选；RTL HPU 使用一维 heap wire 比较树，
候选缓冲、比较树和输出状态机分别实现。二者按输入顺序决定相同幅值的优先级。
SystemC ATU 保存最多 256 项映射；identity 初始化每拍写一项，初始化期间查询和交换停止接受。
query/pivot 的有效、就绪和结果脉冲以各自源码定义，调用方不能将未握手请求当作已执行。

| 验证入口 | 核查内容 |
|---|---|
| [atu_hpu_demo.cpp](../../graduation-code/systemc/src/atu_hpu_demo.cpp) | 独立初始化、候选选择、映射查询与交换 |
| [tb_hpu_handshake.sv](../../graduation-code/hardware/HPU/tb_code/tb_hpu_handshake.sv) | 请求/结果一一对应，输出反压、失败结果和握手沿清 valid |
| [tb_ATU_collision.sv](../../graduation-code/hardware/ATU/tb_ATU_collision.sv) | 不相交 cfg 与 swap 的同拍竞争、接受优先级和重试 |
| [run_rtl_tests.sh](../../graduation-code/hardware/run_rtl_tests.sh) | 独立 RTL 仿真入口 |

这些测试检查现有独立协议，不覆盖 schema v2 GCU 集成或动态整树数值求解。
正式 Panel 及 P-vector 输出关系见 [Panel-LU 模块设计](Panel-LU模块设计.md)；
树形 RHS gather 见 [树形求解模块设计](树形求解模块设计.md)。
