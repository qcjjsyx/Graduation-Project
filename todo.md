# 毕业设计推进计划

最后更新：2026-07-26

## 1. 项目边界

目标是完成“关键模块 RTL + 完整 SystemC 系统级原型”：

- 软件：符号分析、量化、任务/map、ABI v2 DDR 镜像；
- SystemC：完整 front 装配、分解、update、写回、树形求解和架构实验；
- RTL：GCU、ATU、HPU，以及建议完成的 QAU；
- Panel/TRSM/GEMM/向量核不要求 RTL，但必须在 SystemC 中功能正确且延迟可配置；
- 不实现 delayed pivot、动态 front 扩张和真实 AXI/DDR PHY；
- SystemC 已实现混合精度迭代求精，但它属于 Host/系统研究模型，不要求 RTL。

## 2. 已完成：软件数值基础与 ABI v2

- [x] 基于显式消元 fill 结构生成 etree、supernode 和 front。
- [x] 输入允许数值非对称，但要求稀疏结构对称。
- [x] 保留多根消去森林。
- [x] 每个原始元素只归属最早消去节点，`A_local.F22=0`。
- [x] manifest 使用文件实际矩阵维度。
- [x] 支持 `--rhs`；未提供时由固定 seed 生成 `x_true` 和 `b=A*x_true`。
- [x] 生成地址一致的 `memory_image.bin`。
- [x] 生成 FP64 local front、RHS 和参考解黄金旁路。
- [x] 默认执行稀疏结构不变的 2 的整数次幂行均衡 `D_r A x=D_r b`。
- [x] 生成原始矩阵/RHS 和行缩放 exponent，最终按原方程验证。
- [x] 软件源量化默认提升至 30 effective bits。
- [x] ABI 升级为不兼容 v2：小端、128 字节 `NodeTask`。
- [x] 固定 map table 格式和区域所有权。
- [x] Python/C++ 分别校验版本、尺寸、地址、重叠、任务、森林和 map 完整性。
- [x] 记录 ABI 文档：`graduation-code/systemc/docs/ABI_v2.md`。

当前符号规模：

| 矩阵 | 节点数 | 最大 pivot | 最大 front |
|---|---:|---:|---:|
| 256JJ | 73 | 72 | 72 |
| 576JJ | 157 | 156 | 156 |
| 1024JJ | 274 | 256 | 272 |

## 3. 已完成：SystemC 完整系统模型

### 控制与存储

- [x] Artifact Loader 和严格 ABI 二次校验。
- [x] 字节地址 DDR、burst/outstanding、延迟、带宽、抖动和可复现反压。
- [x] Task Fetch 读取真实 DDR task queue。
- [x] Dependency Scoreboard 支持任意任务顺序、多根树、重复完成和下溢检测。
- [x] 配置化 Buffer Manager、FIFO、容量和生命周期统计。
- [x] Writer 在 DDR 写回后才释放 parent 依赖和 buffer。
- [x] 超时与 numeric failure 分离，不用死锁表示数值失败。

### 装配与计算

- [x] QAU：exponent 预取、`e_asm=max`、有符号舍入移位、int64 累加、node-scale 和再量化。
- [x] FP64 与 int32+BFP 共用控制流。
- [x] HPU 只搜索 `k..pivot_dim-1`，相同绝对值保留首个候选。
- [x] ATU 使用 P-vector；update 行 identity/bypass；行索引 9 bit。
- [x] 定点 L 使用 QF，默认 `F=20`；矩阵/U/update 使用 M-format。
- [x] accumulator 位宽、int32 饱和、drop/overflow/saturation 风险计数可配置。
- [x] 默认采用 26-bit 计算输出、8 guard-bit int64 节点工作区。
- [x] 宽候选统一归一化后复用 32-bit HPU。
- [x] U 与 child update 独立选择 exponent 并写回。
- [x] 小主元、L 越界、工作区溢出和 U 零对角触发显式 precision rescue。
- [x] rescue 只消费定点 assembled front，不允许读取黄金 FP64 front。
- [x] rescue 使用可配置 FP64 功能/周期后端，并保留节点级统计。
- [x] 实现七类算子：
  `FACT、TRSM_U、TRSM_L、GEMM_PIVOT、TRSM_F12、TRSM_F21、GEMM_SCHUR`。
- [x] 实现 `serial` 和 `resource_aware` 调度。
- [x] Panel/TRSM/GEMM 数量、启动延迟和吞吐率可配置。

### 树形求解

- [x] 前代：P-vector、`LPP`、`LUP`。
- [x] 回代：`UPU`、`UPP`。
- [x] FP64 double 向量。
- [x] 定点 V-format：全局 RHS exponent、int64 scratch、每节点 solution exponent。
- [x] 使用 128-bit 中间除法，消除 int64 分子越界。
- [x] 每节点 solution exponent 自适应调整。
- [x] rescue 节点使用高精度 factor、普通节点使用定点 factor 的 hybrid solve。
- [x] 在原始 `A,b` 上执行带下降保护的混合精度迭代求精。
- [x] solution 写回 DDR，并恢复/记录 ordering 映射。
- [x] 输出原/均衡 residual、分量后向误差、解误差、求精历史和 solve 周期。

## 4. 已完成：运行接口与研究输出

- [x] 固定命令：

```text
system_sim
  --artifact <manifest.json>
  --config <sim_config.json>
  --mode fp64|fixed|both
  --out <result_dir>
  [--vcd]
  [--seed N]
```

- [x] 输出 `summary.json`、`nodes.csv`、`operations.csv`、`memory.csv`、
  `timeline.csv`、`solution.csv` 和最终 DDR 镜像。
- [x] 报告 U/update exponent、pivot ratio、growth、rescue 和求精停止原因。
- [x] 可选 VCD。
- [x] 批量实验入口支持 buffer、DDR、tile、kernel、位宽、guard bits、主元阈值、
  求精轮数和调度策略比较。
- [x] 默认单因素实验；显式 `--cartesian` 才启动完整笛卡尔积。

## 5. 已完成：验证

- [x] 软件测试：30/30。
- [x] SystemC CTest：ATU/HPU、数值单元、完整端到端全部通过。
- [x] Python/C++ ABI v2 字段一致。
- [x] 定点 L/U/P/update 使用 Python 推导的 bit-exact 小 front 向量。
- [x] 错误版本、截断镜像、重叠区域和损坏 map 必须失败。
- [x] 20 个固定反压 seed 无丢任务、重复执行或依赖错误。
- [x] 同配置同 seed 的 summary 完全一致。
- [x] DDR 带宽和 kernel 延迟单调性测试。
- [x] ASan/UBSan 全部测试通过。
- [x] 稳定小算例满足 FP64 `1e-10`、定点 `1e-3` residual 目标。
- [x] 256JJ：FP64 原 residual `1.349e-12`；定点经 2 轮求精为 `5.377e-4`。
- [x] 576JJ：FP64 明确报告 node 156 小主元；定点可完成但 residual
  `2.346e-2`，未达标。
- [x] 1024JJ：原定点 zero-pivot 已消除；经 40 轮求精 residual
  `9.434e-4`。
- [x] 定点真实矩阵正常 `matrix_overflow_count=0`。
- [x] 完整问题、改进和限制文档：
  `graduation-code/systemc/docs/FIXED_POINT_STABILITY_DESIGN.md`。

说明：`status=ok` 只表示系统正常结束，不代表精度达标；以
`summary.json.solve.fixed.accuracy_target_met` 为准。病态问题还必须同时观察解误差和
componentwise backward error。

## 6. 下一阶段：RTL 与三方交叉验证

### GCU

- [ ] 将正式 ABI v2 解码层接入 `gcu_top`。
- [ ] 覆盖 F12/F21/Schur 调度和 writer 完成事件。
- [ ] 增加依赖重复完成、underflow、buffer 生命周期自检。
- [ ] 用 `timeline.csv` 事件序列与 SystemC 对照。

### ATU

- [ ] RTL 行索引统一为 9 bit。
- [ ] 初始化行数改为实际 `pivot_dim`，update 行 bypass。
- [ ] 覆盖随机交换、边界、初始化期间访问和 query/pivot 仲裁。
- [ ] 与 SystemC 输出 P-vector 逐节点比较。

### HPU

- [ ] 候选容量固定为 256。
- [ ] 覆盖 int32 最小负数、相同绝对值、非 2 次幂、全零和 backpressure。
- [ ] 与 SystemC `nodes.csv` pivot 结果比较。

### QAU（建议作为下一项主任务）

- [ ] `round_shift_signed`。
- [ ] exponent align + int64 `align_accumulate`。
- [ ] `maxabs_tracker`。
- [ ] `node_scale_selector`。
- [ ] U/update 双 exponent `requantize_sat`。
- [ ] drop、overflow、saturation 计数器。
- [ ] 使用 SystemC 最终 DDR 镜像建立 Python/SystemC/RTL 逐元素比较。

## 7. 下一阶段：研究实验

- [ ] 对真实矩阵扫描 `q_use_bits/frac_bits/accumulator_bits/guard_bits`，定位 residual
  与 rescue 主导因素。
- [ ] 比较 single/double/multi buffer 的等待周期和收益。
- [ ] 扫描 DDR 带宽/延迟，绘制性能敏感性曲线。
- [ ] 扫描 Panel/TRSM/GEMM 单元数量与吞吐率。
- [ ] 比较 `serial` 与 `resource_aware`。
- [x] 分析 576/1024 的原始失败节点、pivot、zero-pivot 和 exponent/overflow。
- [x] 实现并验证 2 的整数次幂行均衡、宽工作区和 precision rescue 基线。
- [x] 在独立 `optimization-v2-tile-bfp` 中实现 16×16 tile exponent。
- [x] tile exponent 贯穿 local/front、QAU、LU、U/update 写回和求解读取。
- [x] 保持 128 字节 NodeTask ABI v2，扩展可变长 `front_e/update_e/node_meta`。
- [x] 增加 tile-aware 主元比较、跨 exponent L 除法和 Schur 对齐。
- [x] 增加写回前局部因子误差门槛与显式 precision rescue。
- [x] 针对 576 比较 tile exponent、工作区位宽和 rescue threshold。
- [x] v2 真实矩阵 residual：256 `5.964e-4`、576 `3.252e-4`、
  1024 `1.012e-4`。
- [x] v2 Python 33 项、CTest 3/3、ASan/UBSan 3/3 通过。
- [x] 在独立 v3 中将 L 从统一 QF20 优化为统一 QF26；rescue 从
  68/150/122 降到 9/37/49。
- [x] 实现并计费 F20→F26→FP64 分层重算，实验确认它不如统一 QF26，保留为反例
  配置而不作为默认方案。
- [ ] 扩展真实矩阵集合，继续验证 QF26 的 L 实数范围和 rescue 比例。
- [ ] 实现 tile-aware 纯整数向量 V-format；当前 tile 路径使用 exponent-aware
  double 向量控制模型。
- [ ] 仅当静态缩放仍不足时评估 delayed pivot/动态 front；明确其 ABI 和 buffer 代价。
- [ ] 保存每批实验的配置、commit、原始 CSV 和绘图脚本。

## 8. 已完成：v3 QF26 精度优化

- [x] 从冻结 v2 复制独立 `optimization-v3-adaptive-precision`，未修改基线与 v2。
- [x] 完成 576 定向扫描：QF20/22/24/26/28、guard 和局部误差门槛。
- [x] 确认 QF20 步长 `2^-20` 大于 `2e-7` 局部因子门槛，是大量 rescue 的主要原因。
- [x] 每个 factor 保存实际 `l_frac_bits`，求解和因子检查按节点解码。
- [x] 区分并报告 fraction retry、precision assist 和完整 FP64 rescue。
- [x] 重算节点在周期模型中追加第二套七类定点操作。
- [x] `node_meta` 保存精度路径 flags 与实际 L frac bits。
- [x] 真实矩阵统一 QF26 residual：
  256 `7.096e-5`、576 `3.024e-4`、1024 `2.816e-5`。
- [x] 统一 QF26 周期：
  256 `179858`、576 `1686076`、1024 `7237984`。
- [x] v3 软件 33 项、CTest 3/3、ASan/UBSan 3/3 通过。
- [x] 完整设计与实验文档：
  `graduation-code/optimization-v3-adaptive-precision/docs/ADAPTIVE_PRECISION_OPTIMIZATION.md`。

## 9. 论文与答辩

- [ ] 第三章：S/M/QF/V-format、行均衡、guard bits、QAU、误差来源。
- [ ] 第四章：ABI v2、DDR、任务依赖、buffer 和 SystemC 架构。
- [ ] 第五章：GCU/ATU/HPU/QAU RTL 与三方验证。
- [ ] 第六章：256/576/1024 的 v2-v3 精度与周期对比、576 病态性、架构扫描和 RTL
  综合。
- [ ] 明确区分 RTL 实现、SystemC 功能模型和事务级性能估算。
- [ ] 更新答辩 PPT 和架构图，使实现范围与仓库一致。

## 10. 当前最优先的三件事

1. 按 v3 推荐值实现 tile-aware QAU/LU RTL 最小闭环：16×16 exponent、QF26 L、
   int64 工作区和 U/update exponent SRAM。
2. 为 SystemC 的高精度局部因子检查设计可综合的风险代理，并用 v3 的 rescue 节点
   作为标注数据验证误报/漏报。
3. 实现 tile-aware 纯整数向量 V-format，并完成 Python/SystemC/RTL 三方逐元素验证。
