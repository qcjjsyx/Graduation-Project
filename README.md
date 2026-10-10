# 多前沿稀疏 LU 求解器软硬件协同架构

状态：当前项目入口（2026-10-07 按当前代码同步；STATIC 分解已推进至 T11）

本项目研究使用多前沿方法组织稀疏方阵求解，并通过软件编译器、SystemC 硬件架构模型和关键 RTL 原型验证软硬件协同方案。

正式问题是：

```text
Ax = b
```

显式逆矩阵不作为主工作负载；需要求逆时，将单位矩阵作为多个 RHS，复用同一套 LU 因子和求解流程。

## 当前架构结论

```text
软件编译器 → schema v2 artifact → FP32 SystemC/C++ 设备模型 → 最终 DDR image + reports
                                     （可选局部 INT32 GEMM）              ↓
初始 artifact ─────────────────────────────────────────────→ 独立 FP64 checker
```

- FP64 Python/C++：独立数学参考与外部 checker，不向设备注入参考 front、因子、预选主元或解；
- 正式设备后端：`FP32_SYSTEMC` 和 `FP32_SYSTEMC_INT32_GEMM`，模拟命令、buffer、依赖、真实数据运算与周期；FP64 局部诊断不等同于正式 FP64 设备后端；
- 局部 INT32：仅用于顶层 `GEMM_SCHUR` 的 tile/K 分块，front、因子、update、装配、Panel 和 TRSM 保持 FP32；`Int32GemmBehavioral` 是设计职责名称，当前由 `precision_adapter.cpp` 函数集合实现，并由 `MultiModePe` 调用；
- RTL：GCU/HPU/ATU 等独立原型尚未形成 schema v2 设备闭环，不能用 SystemC 通过记录宣称 RTL 已完成。

## 软件、SystemC 与硬件的职责

```text
稀疏矩阵 A、RHS b
  → 均衡、加权匹配行置换、ordering、fill、消除森林
  → supernode、front、map
  → command buffer、descriptor table、memory image
  → SystemC/C++ STATIC 分解设备
  → 持久 L/U/P、update、status 和周期/访存报告
  → 独立 FP64 因子/update/P 验收

后续 T12：真实树形前代/回代 → solution 写回 → 原方程 Ax=b 验收
```

软件负责全局结构分析和命令编译；设备执行已编译命令。当前分解入口是
[`tree_factor_device`](graduation-code/systemc/src/tree_factor_device.cpp)，
外部验收入口是 [`tree_factor_check`](graduation-code/systemc/reference/tools/tree_factor_check.cpp)。
分配 RHS/solution 区域或存在参考 solve 不表示设备已完成求解。

当前 artifact 是 Command/Descriptor schema v2：32B Command、64B Descriptor、64B Completion，数据位于同一 `memory_image.bin` 的显式区域。`FRONT_DESC` 携带主元策略；正式设备支持 `STATIC`，局部 P-vector 成功后写 identity。`DYNAMIC_PANEL` 已有接口与独立参考语义，但真实设备尚未实现，显式拒绝。旧 `NodeTask` ABI v2 已归档，与 schema v2 无关。

当前进度：T00–T11 的 STATIC 分解链已实现，包含真实 Panel-LU、左右 TRSM、GEMM_Schur、extend-add、完整因子与 update 写回、单节点/树/森林执行。T05 test stub 仅用于显式控制流测试；T06 合成原型不替代正式数值验收。T12 树形 solve 与完整 Ax=b 尚未完成；T14 HPU/ATU 契约统一后移，动态路径依赖该工作。详见[任务说明书](docs/项目实现任务说明书.md)和[当前任务状态](todo.md)。

共享计算架构已落地：Panel/TRSM/GEMM 命令共用一个 PE 池，实验配置为 **2 组×每组 2 个完整 16×16 PE**；每条计算命令独占一组，组内按依赖处理 tile。统一片上读/写预算各方向每拍只授予一个 requester，余量不转授。`*_unit_count` 是命令执行单元实例数，PE 数量由 `compute_group_count × pes_per_group` 决定。见[计算架构基线](docs/architecture/多模式PE与16x16分块计算基线.md)。

新人入口：[当前实现技术导读](docs/当前实现技术导读.md)。当前缺陷与复现见[审查报告](docs/当前实现审查与验证报告.md)：B01 abort 响应丢失已修复；独立 RTL 的 B02/B03 尚未修复。

## 当前代码目录

```text
Graduation-Project/
├── README.md
├── todo.md
├── docs/                         当前架构、接口、SystemC 和验证规范
├── graduation-code/
│   ├── software/                 稀疏结构分析和软件产物生成
│   ├── systemc/                  多前沿 SystemC 系统模型
│   ├── hardware/                 GCU、ATU、HPU 等 RTL 原型
│   ├── sim/                      Python 行为模型和实验脚本
│   └── fixtures/                 Python/C++ 共用 golden fixture（schema v1/v2）
├── graduation-project/           论文、开题、汇报和参考资料
├── archive/                      归档：NodeTask ABI v2、历史优化与精度实验
└── unused/                       早期设计与历史 optimization-v1/v2/v3 副本
```

## 快速验证

软件：

```bash
cd graduation-code/software
python -m pytest -q
```

SystemC：

```bash
cd graduation-code/systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Python 历史行为模型：

```bash
python graduation-code/sim/hardware.py --n 32 --mode stable --seed 42 --ir-iters 5
```

## 当前主线文档

- [项目范围与交付边界](docs/architecture/项目范围与交付边界.md)
- [系统架构与模块职责](docs/architecture/系统架构与模块职责.md)
- [软件侧算法说明](docs/algorithm/README.md)
- [Command Stream 与 Descriptor 接口](docs/interface/Command_Stream_and_Descriptor.md)
- [Command / Descriptor Schema v2](docs/interface/Command_Schema_v2.md)
- [Command / Descriptor 固定格式 v1（历史）](docs/interface/Command_Schema_v1.md)
- [SystemC 模型规范](docs/systemc/SystemC_模型规范.md)
- [验证计划与结果口径](docs/verification/验证计划与结果口径.md)
- [T00 可复现基线](docs/verification/T00_baseline.md)
- [项目实现任务说明书](docs/项目实现任务说明书.md)
- [总体架构重新定型方案](项目总体架构重新定型方案.md)
- [当前推进计划](todo.md)

## 文档使用规则

上述主线文档定义当前项目范围。`unused/optimization-v1/v2/v3`、`archive/`、旧开题方案、旧量化方案和早期 prompt 只用于追溯历史实验，不能继续修改为当前架构依据。

FP64 参考结果、FP32 SystemC 结果、INT32 GEMM 行为模型结果和未来 RTL 结果必须在实验中分开标记。SystemC 运行成功不等于 Verilog 已经实现，也不等于真实芯片周期已经得到验证。
