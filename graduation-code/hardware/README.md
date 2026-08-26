# RTL 硬件模块说明

状态：当前主线说明

硬件目录保存关键模块的 SystemVerilog 原型。当前项目不以完整流片级硬件为目标，而是以“SystemC 全系统架构模型 + 关键 RTL 模块”为交付边界。

## 1. RTL 范围

| 模块 | 主要职责 | 当前定位 |
|---|---|---|
| `GCU/` | command fetch、依赖 token、资源和完成控制 | 优先完成 RTL 原型 |
| `ATU/` | logical row 到 physical row 的 P-vector 映射和 swap | RTL 原型 |
| `HPU/` | pivot candidate 接收、最大绝对值归约和 tie-break | RTL 原型 |
| `QAU/` | 精度适配或量化边界的候选模块 | 重新定义，不再默认承担全局 front BFP 装配 |
| `Matrix_Engine/` | 预留计算核接口 | 暂不假设已有可联调 GEMM RTL |

## 2. 与 SystemC 的关系

```text
SystemC
  ├─ 完整 GCU/Buffer/Assembly/Panel-LU/TRSM/GEMM/Solve 模型
  ├─ FP32/FP64 浮点主设备路径
  └─ Int32GemmBehavioral

RTL
  ├─ GCU/HPU/ATU/片上存储接口原型
  └─ 未来可选 Int32GemmRtlWrapper
```

RTL 不需要与 SystemC 共享实现代码，但必须共享：

- command schema；
- descriptor 语义；
- ready/valid 协议；
- row/pivot 边界；
- status/error 编码；
- tile transaction 边界。

## 3. INT32 GEMM 处理原则

当前没有 INT32 GEMM RTL。SystemC 先实现 `Int32GemmBehavioral`，模拟：

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

当前 `gcu_top.sv` 和部分 micro-scheduler 仍是原型或占位逻辑，不能在文档中表述为完整设备。

## 5. HPU

首版只实现确定性的最大绝对值归约：

- 输入 `(value, logical_row, valid)`；
- 支持 last 或显式 candidate count；
- 相同绝对值使用固定 tie-break；
- 输出 pivot row、pivot value、valid 和 failure；
- 支持 backpressure、reset 和空候选测试。

CALU、threshold、rook 和复杂 tournament 不作为首版必需模式。

## 6. ATU

ATU 维护 P-vector：

```text
logical_row → physical_row
```

pivot swap 只修改映射，不搬移整个 front。Panel、TRSM 和写回必须使用同一映射语义。当前最大 front/pivot 的位宽必须由 descriptor/config 显式限制，不能由不同 RTL 文件各自硬编码。

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

