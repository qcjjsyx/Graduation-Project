# RTL 硬件模块说明

状态：当前主线说明（2026-10-07 按实际实现同步；区分当前能力与后续接口目标）

硬件目录保存关键模块的 SystemVerilog 原型。当前项目不以完整流片级硬件为目标，而是以“SystemC 全系统架构模型 + 关键 RTL 模块”为交付边界。

## 1. RTL 范围

| 模块 | 主要职责 | 当前定位 |
|---|---|---|
| `GCU/` | 旧 task 取指、child 计数、buffer 与 micro-scheduler 原型 | `gcu_top` 空壳；尚未实现 schema v2 command/token/completion 闭环 |
| `ATU/` | logical row 到 physical row 的 P-vector 映射和 swap | RTL 原型 |
| `HPU/` | pivot candidate 接收、最大绝对值归约和 tie-break | RTL 原型 |
| `QAU/` | 精度适配或量化边界的候选模块 | 重新定义，不再默认承担全局 front BFP 装配 |
| `Matrix_Engine/` | 预留计算核接口 | 暂不假设已有可联调 GEMM RTL |

## 2. 与 SystemC 的关系

```text
当前 SystemC/C++ 设备
  ├─ CommandDevice / DataPlane / 真实 Assembly / Panel-LU / TRSM / GEMM
  ├─ FP32_SYSTEMC 或 FP32_SYSTEMC_INT32_GEMM
  └─ L/U/P/update 写回；真实 Solve 尚未接入（T12）

独立 FP64 reference / checker
  └─ 分解、参考 solve 与设备输出检查，不向设备注入黄金数据

当前 RTL
  └─ GCU/HPU/ATU 独立原型，未接入 schema v2 全链
后续目标
  └─ 契约统一、RTL 联调及可选 INT32 GEMM RTL 替换
```

以下是后续 RTL 接入 SystemC 设备契约时必须对齐的要求，当前旧 GCU 原型尚未全部满足：

- command schema；
- descriptor 语义；
- ready/valid 协议；
- row/pivot 边界；
- status/error 编码；
- tile transaction 边界。

## 3. INT32 GEMM 处理原则

当前没有 INT32 GEMM RTL。T10 的局部 INT32 行为计算已由
[`precision_adapter.cpp`](../systemc/src/panel_lu/precision_adapter.cpp) 实现，
`MultiModePe` 调用 `run_int32_gemm_update`。`PrecisionAdapter` / `Int32GemmBehavioral`
是设计职责名称，不是现存同名 C++ 类。仅顶层 `GEMM_SCHUR` 使用该路径：

```text
FP tile → quantize → INT32 multiply/accumulate → dequantize → FP tile
```

量化只在一次 GEMM 调用边界发生，不把 scale 传播为整棵消除树的全局格式。未来得到真实 GEMM RTL 时，使用相同的 request/response 接口替换行为模型。

## 4. GCU

GCU 的长期输入是 command ring 和 descriptor table，而不是软件内部的 `NodeTask` 字段。最低职责包括：

- 读取和校验 command header；
- 管理 wait/signal token；
- 管理 buffer 和资源状态；
- 发射 Panel、TRSM、GEMM、DMA 命令；
- 产生 completion 和 error；
- 记录 cycle、bytes、stall、pivot 和 retry。

当前代码事实：

- [`gcu_top.sv`](GCU/gcu_top.sv) 只有 clk/rstn 端口，无执行逻辑。
- [`gcu_task_fetch.sv`](GCU/gcu_task_fetch.sv) 默认读取 128bit/16B 旧 task，不是 32B schema v2 Command。
- [`gcu_dep_scoreboard.sv`](GCU/gcu_dep_scoreboard.sv) 维护 child 计数，不是控制面 TokenScoreboard。
- [`gcu_micro_scheduler.sv`](GCU/gcu_micro_scheduler.sv) 默认 TILE_SIZE=32，F12/F21/Schur 阶段仍占位；不能当作主线 16×16 PE 调度器。

因此这里的 GCU 是独立 RTL 原型，不能视为 `CommandDevice` 的已完成 RTL 对应实现。

## 5. HPU

以下为 HPU 首版的目标契约；现有比较树/top 为独立原型，尚未接入主线动态执行：

- 输入 `(value, logical_row, valid)`；
- 支持 last 或显式 candidate count；
- 相同绝对值使用固定 tie-break；
- 输出 pivot row、pivot value、valid 和 failure；
- 支持 backpressure、reset 和空候选测试。

当前比较节点相等时选输入 a；top 的重复输出握手缺陷 B02 尚未修复，不能把背压等目标要求视为已完整验证。证据见[当前审查报告](../../docs/当前实现审查与验证报告.md)。CALU、threshold、rook 和复杂 tournament 不作为首版必需模式。

## 6. ATU

ATU 维护 P-vector：

```text
logical_row → physical_row
```

pivot swap 只修改映射，不搬移整个 front。Panel、TRSM 和写回必须使用同一映射语义。当前最大 front/pivot 的位宽必须由 descriptor/config 显式限制，不能由不同 RTL 文件各自硬编码。

当前 ATU 是独立原型，尚未接入动态整树；cfg/pivot 同拍接受后丢 cfg 写的 B03 尚未修复，见[审查报告](../../docs/当前实现审查与验证报告.md)。

## 7. QAU / Precision Adapter

旧 QAU 方案中的全局 exponent 对齐、全局 front 再量化和 U/update 双 scale 不再是当前主线硬件要求。新的候选职责是：

- 为 GEMM 输入 tile 选择二的幂 scale；
- 执行 round/shift/saturate；
- 记录 overflow、saturation 和 scale；
- 接收 INT32 GEMM 输出并执行反量化；
- 向 SystemC/未来 RTL GEMM 提供统一接口。

在确认 tile 误差和累加位宽之前，不继续扩大 QAU 的 RTL 范围。

## 8. 验证规则

每个 RTL 模块必须有独立 testbench，并与 SystemC 进行：

- reset 对照；
- valid/ready 和 backpressure 对照；
- 边界尺寸对照；
- 状态和完成顺序对照；
- error/status 对照；
- 数据输出对照。

没有真实 GEMM RTL 时，不能生成或宣称 `RTL_GEMM_PROTOTYPE` 结果；只能报告 `SYSTEMC_INT32_GEMM_MODEL`。

