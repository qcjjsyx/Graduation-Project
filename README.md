# 基于张量计算的大型稀疏矩阵求解系统

本仓库是面向多前沿稀疏 LU 的软硬件协同毕业设计。当前实现边界是：

- 软件完成符号分析、唯一 local contribution、量化、任务/map 和 DDR 镜像生成；
- SystemC 完成多节点装配、主元、Panel/TRSM/GEMM-Schur、写回、树形求解和性能建模；
- RTL 聚焦 GCU、ATU、HPU，以及后续需要补强的 QAU；
- 不要求实现矩阵计算核、向量计算核或真实 DDR 控制器 RTL。

因此，项目成果应表述为“关键模块 RTL + 完整 SystemC 系统级原型”，而不是完整计算核 RTL。

## 目录

```text
Graduation-Project/
├── README.md
├── todo.md
├── out/                ABI v2 的 256JJ + 256fuv 可运行样例
├── graduation-code/
│   ├── software/       符号分析、量化和 ABI v2 DDR 产物
│   ├── systemc/        完整分解/求解与架构性能模型
│   ├── sim/            Python 数值行为与历史实验
│   └── hardware/       GCU、ATU、HPU、QAU 等 RTL
└── graduation-project/ 论文、答辩、图片和参考资料
```

## 完整执行流

```text
结构对称稀疏矩阵 + RHS
  -> 2 的整数次幂行均衡
  -> fill-aware 符号分析与 supernode/front
  -> ABI v2 memory_image.bin
  -> DDR/Task/Scoreboard/Buffer
  -> QAU front 装配
  -> HPU/ATU pivot
  -> Panel LU / TRSM / GEMM-Schur
  -> L/U/P/update 写回
  -> 树形前代/回代
  -> precision rescue / 混合精度迭代求精
  -> 原方程 residual、解误差、量化风险与周期报告
```

## 快速验证

```bash
cd graduation-code/software
python -m pytest -q

cd ../systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

完整运行命令、配置和结果文件见
[SystemC README](graduation-code/systemc/README.md)，软件产物说明见
[software README](graduation-code/software/README.md)，ABI 见
[ABI v2](graduation-code/systemc/docs/ABI_v2.md)。定点稳定性问题、改进方案、实现细节、
实验结果和当前边界见
[定点分解稳定性设计与实现](graduation-code/systemc/docs/FIXED_POINT_STABILITY_DESIGN.md)。

当前推进状态和剩余工作见 [todo.md](todo.md)。
