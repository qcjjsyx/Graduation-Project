# 代码目录说明

本目录保存毕业设计的软件工具链、SystemC 系统模型、Python 数值行为模型和关键 RTL 模块。

## 当前目录

```text
graduation-code/
├── software/       稀疏矩阵预处理、任务生成、量化输入与二进制产物
├── systemc/        完整多前沿分解、树形求解和架构性能模型
├── sim/            Python 数值行为模型、批量实验与图表
├── hardware/       可综合 RTL 与对应 testbench
│   ├── GCU/        任务、依赖、缓冲和节点内调度
│   ├── ATU/        逻辑行到物理行的地址映射
│   ├── HPU/        主元选择
│   ├── QAU/        量化装配单元，当前为接口与实现计划
│   └── Matrix_Engine/ 预留的矩阵计算核接口目录
├── README_code.md  早期架构设计记录，部分内容已与当前实现不一致
└── 进度.md         早期模块进度记录
```

## 三层验证边界

- `software/`：生成真实的 `tasks.bin`、`map_table.bin`、`front_q.bin`、`front_e.bin` 和 `manifest.json`。
- `systemc/`：由 ABI v2 DDR 镜像驱动完整系统数据流；计算核采用功能正确、延迟可配置的模型。
- `hardware/`：实现并验证 GCU、ATU、HPU，以及建议实现的 QAU，不要求实现 TPU/SFU 计算核 RTL。
- `sim/`：作为数值黄金模型和实验数据生成工具，与 SystemC、RTL 使用相同测试向量对照。

当前 SystemC 已加入 2 的整数次幂行均衡、guard-bit 宽工作区、U/update 双 exponent、
precision rescue 和混合精度迭代求精。原始失败、改进理由、实现细节、真实矩阵结果与
RTL 约束见
[systemc/docs/FIXED_POINT_STABILITY_DESIGN.md](systemc/docs/FIXED_POINT_STABILITY_DESIGN.md)。

项目执行计划和验收条件见仓库根目录的 `todo.md`。
