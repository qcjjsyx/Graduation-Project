# 多前沿稀疏 LU 求解器软硬件协同架构

状态：当前项目入口

本项目研究使用多前沿方法组织稀疏方阵求解，并通过软件编译器、SystemC 硬件架构模型和关键 RTL 原型验证软硬件协同方案。

正式问题是：

```text
Ax = b
```

显式逆矩阵不作为主工作负载；需要求逆时，将单位矩阵作为多个 RHS，复用同一套 LU 因子和求解流程。

## 当前架构结论

```text
FP64 C++ reference
        ↓
FP32/FP64 SystemC hardware architecture model
        ↓
selected RTL prototypes
```

- FP64 C++：数学黄金参考，不进行硬件时序仿真；
- FP32/FP64 SystemC：主硬件架构模型，模拟命令、buffer、依赖、算子状态和周期；
- INT32 GEMM：只在 GEMM 调用边界做局部量化/反量化；当前没有 RTL 时，使用 SystemC `Int32GemmBehavioral`；
- RTL：优先实现 GCU、HPU、ATU、片上存储接口和关键控制原型，不强行把整个求解器改成全局 INT32/BFP。

## 软件、SystemC 与硬件的职责

```text
稀疏矩阵 A、RHS b
  → 均衡、ordering、fill、消除森林
  → supernode、front、map
  → command ring、descriptor table、data image
  → SystemC 或 RTL device executor
  → L/U/P、update、solution、status
  → residual、backward error、周期和访存报告
```

软件负责全局结构分析和命令编译。设备侧不负责 ordering、fill 或 supernode 构造，只执行已经编译的命令。

当前 artifact ABI 是 Command/Descriptor v1：32B command、64B descriptor、64B completion 和同一 `memory_image.bin` 中的显式数据区域。旧 `NodeTask` ABI v2 已归档，不参与当前构建。

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
│   └── optimization-v1/v2/v3/   历史精度实验副本，暂不作为主线
├── graduation-project/           论文、开题、汇报和参考资料
└── unused/                      后续归档的废弃设计和旧提示词
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
- [Command Stream 与 Descriptor 接口](docs/interface/Command_Stream_and_Descriptor.md)
- [Command / Descriptor 固定格式 v1](docs/interface/Command_Schema_v1.md)
- [SystemC 模型规范](docs/systemc/SystemC_模型规范.md)
- [验证计划与结果口径](docs/verification/验证计划与结果口径.md)
- [T00 可复现基线](docs/verification/T00_baseline.md)
- [项目实现任务说明书](docs/项目实现任务说明书.md)
- [总体架构重新定型方案](项目总体架构重新定型方案.md)
- [当前推进计划](todo.md)

## 文档使用规则

上述主线文档定义当前项目范围。`graduation-code/optimization-v1/v2/v3`、旧开题方案、旧量化方案和早期 prompt 只用于追溯历史实验，不能继续修改为当前架构依据。

FP64 参考结果、FP32 SystemC 结果、INT32 GEMM 行为模型结果和未来 RTL 结果必须在实验中分开标记。SystemC 运行成功不等于 Verilog 已经实现，也不等于真实芯片周期已经得到验证。
