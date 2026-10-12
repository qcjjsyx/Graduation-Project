# 代码目录说明

代码入口说明；进度入口更新：2026-10-11。项目全貌见
[项目主线梳理](../项目主线梳理.md)，完成证据见 [当前进度](../当前进度.md)。

## 目录结构

```text
graduation-code/
├── software/      Python 编译器：矩阵 + RHS → Command/Descriptor schema v2 artifact；FP64 host 参考
├── systemc/       SystemC/C++ 设备模型（数据面、控制面、Panel/TRSM/GEMM、装配、树形求解）、FP64 checker、CTest
├── hardware/      RTL 独立原型：HPU、ATU（有自检 testbench）、旧 GCU、Precision Adapter 规范
└── fixtures/      Python 与 C++ 共用的 golden fixture（schema v2、schema v1 历史、同策略参考树）
```

旧的 Python 全局定点行为模型（原 `sim/`）以及 optimization-v1/v2/v3 实验已移到仓库根目录
`archive/`，不参与当前构建和测试。

## 唯一的运行路径

```text
software:  python -m src.main ...            → manifest.json + memory_image.bin（schema v2）
systemc:   tree_factor_device --device-dir … → final_memory_image.bin + 报告
systemc:   tree_factor_check DEVICE REPORT    → validation.json（numeric_pass）
systemc:   tree_solve_check DEVICE REPORT     → solve_validation.json（求解与原方程指标）
```

软件与设备之间只通过 schema v2 二进制 artifact 交互（32B Command、64B Descriptor、64B
Completion，数据在同一 `memory_image.bin` 的显式区域中）。旧的 `NodeTask` ABI v2 已整体归档到
`archive/legacy-node-task-abi-v2/`，主线代码不依赖、也不兼容它。

## 三层职责

| 层 | 负责 | 不负责 |
|---|---|---|
| 软件 | 输入检查、行均衡、匹配行置换、ordering/fill/森林、supernode/front/map、内存布局与命令编译、manifest 校验；可选 FP64 预检 | 数值分解（不向设备镜像写任何参考结果） |
| SystemC 设备 | 取指、token 依赖、双 front 驻留、DDR/DMA/片上 buffer、extend-add、Panel-LU、TRSM、GEMM-Schur（FP32 或局部 INT32）、L/U/update 写回、周期与访存统计 | 树形 solve（T12 未实现）、动态主元（T14 未实现） |
| RTL | HPU、ATU 独立原型与自检 testbench | 与 SystemC 设备的联调；GCU 仍是旧 NodeTask 设计 |

INT32 只在一次 `GEMM_SCHUR` 调用边界内量化/反量化（`systemc/src/panel_lu/precision_adapter.cpp`，
由 `MultiModePe` 调用）。“`PrecisionAdapter` / `Int32GemmBehavioral`” 是设计职责名，不是同名 C++ 类。

## 各子目录说明

1. [software/README.md](software/README.md)：CLI、产物文件、测试
2. [systemc/README.md](systemc/README.md)：构建、CTest、设备与 checker 用法
3. [hardware/README.md](hardware/README.md)：RTL 范围与已知限制
4. [fixtures/command_schema_v2/README.md](fixtures/command_schema_v2/README.md)、[fixtures/reference_tree/README.md](fixtures/reference_tree/README.md)

## 注意事项

- FP64 C++/Python 结果是参考，不是硬件时序结果；FP32 SystemC 是主硬件架构模型。
- 设备 `numeric_status_pass=1` 只说明命令状态成功；数值是否正确只看 checker。
- SystemC 行为模型通过不等于 RTL 已实现。
