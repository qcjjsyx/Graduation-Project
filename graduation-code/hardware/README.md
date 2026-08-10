# RTL 模块说明

硬件侧采用“关键控制与边缘数据通路 RTL + SystemC 系统级模型”的实现范围。

## 计划保留为 RTL 的模块

| 目录 | 职责 | 当前状态 |
|---|---|---|
| `GCU/` | Task Fetch、依赖记分牌、双缓冲和节点内调度 | 子模块原型已存在，顶层尚未集成 |
| `ATU/` | 逻辑行到物理行映射、零拷贝 pivot | RTL 原型已存在 |
| `HPU/` | 流式候选输入和树形主元选择 | RTL 原型已存在 |
| `QAU/` | 多指数对齐、宽装配累加、U/update 双 scale 和再量化 | SystemC 规则已锁定，RTL 待补强 |

## 不要求实现为 RTL 的模块

- Panel LU / SFU
- TRSM / 向量计算核
- TPU / GEMM / Schur 矩阵计算核
- 完整 DDR 控制器
- Factor Writer 和 Update Writer 的真实总线实现

上述模块仍须在 SystemC 中保留功能、接口、延迟和反压模型，从而跑通完整节点执行闭环。

当前 SystemC 已输出每节点 pivot、P-vector、量化风险、算子时间线和最终 DDR 镜像，可作为
GCU/ATU/HPU/QAU RTL testbench 的黄金数据来源。ATU 行索引按最大 272 front 配置为 9 bit，
HPU 主元候选只覆盖最大 256 pivot 行，update 行不得成为主元。

稳定性相关的 RTL 约束已经锁定为：

- 软件源默认 30 effective bits，QAU/因子输出默认 26 bits；
- 装配与计算接口需要为 int64 accumulator/guard bits 保留语义；
- U exponent 写入 node metadata，child update exponent 仍写 `update_e`；
- HPU 接收同一列统一 shift 后的 int32 候选；
- precision rescue 是 SystemC 中的高精度后端假设，GCU 只需支持其可变延迟和
  success/failure 状态，不要求本阶段实现 FP64 rescue RTL；
- 混合精度迭代求精属于 Host/SystemC Checker，不要求向量核 RTL。

完整原因、公式和真实矩阵结果见
[定点稳定性设计说明](../systemc/docs/FIXED_POINT_STABILITY_DESIGN.md)。
