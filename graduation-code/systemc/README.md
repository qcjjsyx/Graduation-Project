# SystemC 当前主线

当前目录只保留与 Command/Descriptor schema v2 主线一致的基础模块：

- `include/command_codec.hpp`：schema v2 固定记录 codec（含主元策略字段）；
- `reference/`：独立 FP64 数学参考（`PivotPolicy`/`PivotFailure`、
  `panel_lu_policy` 与通用 `panel_lu` 分离；`tools/fp64_reference_smoke`
  为数值冒烟入口）；
- `include/atu.hpp`、`include/hpu.hpp`：DYNAMIC_PANEL 用的独立模块
  （静态路径旁路）；
- `src/atu_hpu_demo.cpp`：独立模块冒烟测试；
- `tests/command_codec_tests.cpp`：Python/C++ 共用 fixture 的 codec 测试。

原先由 `NodeTask/ABI v2` 驱动的 `system_sim`、artifact loader、memory
container、fixed/global-BFP kernel 和回归脚本已经退役并归档到：

```text
archive/legacy-node-task-abi-v2/systemc/
```

后续 T04/T05 在本目录直接建立 Command/schema v2 memory、buffer 和
executor（先实现 STATIC 固定主元路径及其数值检查；DYNAMIC_PANEL 经
HPU/ATU 为可选能力，未实现时必须以 `POLICY_NOT_SUPPORTED` 拒绝），
不得重新依赖归档中的 `NodeTask` 执行入口。
