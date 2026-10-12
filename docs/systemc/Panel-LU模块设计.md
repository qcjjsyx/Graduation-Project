# Panel-LU 模块设计

实现基准：2026-10-11 当前工作区。Panel-LU 对已装配 front 的 A11 做 STATIC packed LU，
为后续 TRSM、Schur 更新和因子保存提供结果。数学背景见
[矩阵求解理论](../theory/矩阵求解理论基础.md)，公共计算机制见
[共享计算引擎与 PE](共享计算引擎与PE设计.md)。

## 1. 架构与执行路径

```text
CommandDevice：解析 FrontView、获取 front 写 lease
  → PanelExecutionUnit：校验并转换 command 级请求
  → PanelEngine：分配计算组、绑定 front、建立 PanelController
  → 全 front 扫描：有限性、maxabs、固定阈值
  → PanelController：增量生成 A11 tile 任务
  → TileDispatcher / MultiModePe：装载、计算、写回、版本提交
  → engine 终态 / adapter 响应
  → 控制面写 identity P-vector、等待 DDR 确认
  → completion 写确认、token 发布
```

引擎不直接操作控制面的 token 或 completion。front 生命周期与 lease 由控制面管理，
引擎使用已授予的访问权，不重复申请会与自己冲突的整 front lease。
同一 front 的计算命令串行；不同 front 可占不同计算组并行。

## 2. 请求、布局与响应

PanelExecutionUnit 从 ExecRequest 构造 PanelEngineRequest，携带事务身份、
节点/命令号、`total_dim=n`、`pivot_dim=p`、tile 大小、策略、FP32 阈值位型及 front binding。
合法维度为 `1 <= p <= n`，tile 固定 16，front 为紧凑 FP32 row-major，stride 为 `4*n`。
维度、容量、handle generation、owner、范围和 ready cycle 都需要校验。

```text
front = [ A11(p×p)  A12(p×u) ]     u=n-p
        [ A21(u×p)  A22(u×u) ]
```

Panel 只修改 A11。其严格下三角保存 L11 的非对角元素，对角及上三角保存 U11；
L11 的单位对角隐含，不能把 U 对角当作 L 对角。A12/A21/A22 保持原字节。

成功响应提供 pivot_count=p、扫描得到的 front_scale 和 threshold_product，
并标记需要写 P-vector。后续 TRSM 使用这份锁存上下文，不从已消元 front 重算阈值。
内部 failed_column 记录失败主元坐标；外部 Completion 没有同名字段，错误详情写入 trace/reason。

## 3. 主元扫描与数值语义

### 3.1 固定尺度

扫描整个 `n*n` front，包含 A12/A21/A22，而非只扫描 A11。
FrontReader 经 lease 和统一读预算分段访问，实际读字节为 `4*n*n`，扫描归约延迟另计。
任何 NaN/Inf 都返回 NON_FINITE_VALUE，不生成 tile 任务。

front maxabs 保持输入 FP32 的精确值；threshold_ratio 由 descriptor 的 binary32 位型确定。
阈值乘积 `threshold_ratio * front_maxabs` 用 binary64 表示两个 binary32 值的精确乘积，
仅用于判断，不改变设备矩阵精度，也不读取 FP64 参考数据。整个 Panel 使用同一乘积。

### 3.2 STATIC 检查与算术

`run_lu_diag` 按递增主元列执行 packed LU，每列检查实际消元后的主元：

| 条件 | 结果 |
|---|---|
| 主元或中间结果非有限 | NON_FINITE_VALUE |
| 主元精确为零 | STATIC_PIVOT_FAILURE |
| 主元绝对值不大于固定阈值 | PIVOT_UNSTABLE |
| 对角倒数溢出为 Inf | NON_FINITE_VALUE，定位当前主元列 |
| 检查通过 | 计算下三角乘子，以 FP32 FMA 更新尾部 |

乘加使用 `std::fma`，保留渐进下溢，不启用 fast-math/flush-to-zero。
不扰动主元、不静默替换零或小主元、不自动切换策略或精度。
STATIC 失败是当前消元策略失败，不能据此判定原矩阵奇异。

### 3.3 动态策略与诊断模式

当前 DYNAMIC_PANEL 明确返回 POLICY_NOT_SUPPORTED，独立 HPU/ATU 原型不参与这条路径。
模块诊断模式 Fp64Diagnostic 在有限 tile 内用 binary64 计算并舍入回 FP32，
回存每个有效元素前检查有限性；舍入溢出时失败 tile 不提交。
该模式须显式选择并在报告标记，正式 FP32 CLI 不会自动开启。

## 4. A11 任务图

主元 tile 数为 `ceil(p/16)`。写成 `p=16*k+r` 时，只有 `r>0` 才有第 `k+1` 块。
每个主元步 s 的任务关系为：

```text
LU_DIAG(s,s)
  ├─ TRSM_LEFT：求解该步右侧 A11 子块
  └─ TRSM_RIGHT：求解该步下侧 A11 子块
        → GEMM_UPDATE：更新 A11 尾部子块
        → 本步所有任务真实写回并 commit → s+1
```

步内独立目标可以并行；同目标任务通过版本与依赖串行提交。
controller 受有限队列容量约束，保留生成位置并重试，不一次展开无界任务图。
尾块只读、算、检查和写有效 M/N/K 与 mask，guard/padding 不变。
这里的内部 GEMM_UPDATE 始终 FP32；局部 INT32 只作用于顶层 GEMM_SCHUR。

任务存储、派发、PE 周期推导、写回预算与 tile 版本键集中定义在
[共享计算引擎与 PE](共享计算引擎与PE设计.md)，本模块不另建一套资源模型。

## 5. P-vector、因子与输出可见性

设备镜像中的 P-vector 初值为 -1。STATIC Panel 成功时产生 identity gather 映射，
由控制面在 PostExec 阶段经 MemoryService 写入完整 INT32 P-vector，等待所有写确认后收尾。
只有数值任务结束并不表示 PANEL_LU 的输出已经对后继可见。

```text
A11 写回、任务提交 → engine/adapter 终态 → identity P-vector DDR 写确认
→ completion DDR 写确认 → token READY
```

Panel 数值失败时不请求成功 P-vector 写。控制面在 P-vector 写在途期间收到 abort 时，
等待已接受写排空，命令终态为 ABORTED；已发生的 DDR 写不回滚。
部分或失败的因子必须结合 completion 解释，不能仅从 P-vector 字节判断成功。

完整 `L(n×p)`、`U(p×n)` 由 STORE_FACTOR 转换并持久化；L21/U12 来自顶层 TRSM。
详细布局和按行写回见 [装配与因子保存设计](装配与因子保存设计.md)。

## 6. 失败、取消与复用

数值失败后停止新任务生成，定向取消尚未执行的任务；已运行任务在安全点排空并跳过结果写回。
其他已成功写回的 tile 保留，当前实现不提供整 front 回滚。
controller 与在途任务全部收尾后生成唯一终态；适配器消费后释放组，控制面释放 lease。
响应在 poll 前保持稳定，重复 abort 幂等，不能丢失已接受事务的终态。

全局 reset 使旧 epoch、绑定、句柄和版本失效，丢弃旧响应；它与定向取消不同，
也不重载或恢复 DDR 初始镜像。重新执行初始输入应加载原始 artifact。

## 7. 实现与验证入口

| 对象 | 代码 | 阅读重点 |
|---|---|---|
| `PanelExecutionUnit` | [panel_exec_unit.hpp](../../graduation-code/systemc/include/controlplane/panel_exec_unit.hpp)、[实现](../../graduation-code/systemc/src/controlplane/panel_exec_unit.cpp) | 请求校验、front binding、阈值和 P-vector 标记 |
| `PanelController` | [panel_controller.hpp](../../graduation-code/systemc/include/panel_lu/panel_controller.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/panel_controller.cpp) | 扫描、任务生成、步屏障与失败坐标 |
| FP32/诊断内核 | [tile_kernels.hpp](../../graduation-code/systemc/include/panel_lu/tile_kernels.hpp)、[实现](../../graduation-code/systemc/src/panel_lu/tile_kernels.cpp) | 逐列检查、FMA、尾块与回存有限性 |
| 设备模块测试 | [panel_lu_real_tests.cpp](../../graduation-code/systemc/tests/panel_lu_real_tests.cpp) | 数值、阈值、n>p、取消、资源与背压 |

模块验证使用手算矩阵与独立 FP64 参考，同时核查 A12/A21/A22 未修改、失败位置、
P-vector 完整性、真实写回时刻和终态唯一性。整树因子验收入口为 tree_factor_check；
后向误差和解误差的解释见 [数值误差分析理论](../theory/数值误差分析理论基础.md)。
