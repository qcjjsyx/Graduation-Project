# 代码目录说明

状态：当前主线代码入口

本目录包含软件产物生成器、SystemC 系统级硬件模型、Python 实验模型和关键 RTL 原型。

## 目录结构

```text
graduation-code/
├── software/                         软件预处理和 artifact 生成
├── systemc/                          SystemC 系统级硬件模型
├── hardware/                         GCU/ATU/HPU 等 RTL 原型
├── sim/                              Python 数值行为和历史实验
├── optimization-v1/                  历史行列均衡实验
├── optimization-v2-tile-bfp/         历史 Tile BFP 实验
└── optimization-v3-adaptive-precision/历史全局 BFP 精度实验
```

`optimization-v1/v2/v3` 是独立实验副本，不是当前主线执行路径。它们的代码和结果用于复现实验，不用于继续定义当前 ABI 或硬件架构。

## 当前三层边界

### 软件

负责：

- 读取稀疏方阵和 RHS；
- 行均衡、符号包络、ordering、fill、消除森林；
- supernode、front、local map 和数据布局；
- 当前 ABI v2 artifact 生成；
- 目标 command stream/descriptor 生成；
- 软件侧合法性检查。

### SystemC

负责：

- command fetch、GCU、依赖和资源调度；
- DDR/DMA、buffer、front assembly 和写回；
- HPU、ATU、Panel-LU、TRSM、GEMM 和 solve；
- ready/valid、反压、周期、bytes 和 stall；
- FP32/FP64 设备模型；
- 局部 INT32 GEMM 行为模型。

### RTL

优先验证：

- GCU 命令和依赖控制；
- HPU；
- ATU；
- 片上 buffer/SRAM 风格接口；
- 未来可插拔的 INT32 GEMM 接口。

没有 INT32 GEMM RTL 时，不阻塞 SystemC。使用 `Int32GemmBehavioral` 完成行为/周期模型，未来通过相同 transaction 接口替换为 RTL wrapper。

## 当前接口状态

当前可运行路径仍是：

```text
software artifact
  → manifest.json + memory_image.bin
  → ABI v2 NodeTask
  → SystemC executor
```

目标路径是：

```text
NodeTask / front graph
  → Command Compiler
  → command ring + descriptor table + data image
  → SystemC 或 RTL executor
```

两条路径在迁移期间并存，不能将目标 command stream 误写为已经完成的功能。

## 推荐阅读顺序

1. [`../docs/architecture/项目范围与交付边界.md`](../docs/architecture/项目范围与交付边界.md)
2. [`../docs/architecture/系统架构与模块职责.md`](../docs/architecture/系统架构与模块职责.md)
3. [`../docs/interface/Command_Stream_and_Descriptor.md`](../docs/interface/Command_Stream_and_Descriptor.md)
4. [`../docs/systemc/SystemC_模型规范.md`](../docs/systemc/SystemC_模型规范.md)
5. [`../docs/verification/验证计划与结果口径.md`](../docs/verification/验证计划与结果口径.md)
6. 本目录下对应模块 README。

## 注意事项

- FP64 C++ 结果是黄金参考，不是硬件时序结果；
- FP32 SystemC 是主硬件架构模型；
- INT32 只在 GEMM 调用边界进行局部量化/反量化；
- SystemC 行为模型成功不等于 Verilog RTL 已实现；
- 历史优化目录的 README 和 ABI 不得覆盖当前主线定义。

