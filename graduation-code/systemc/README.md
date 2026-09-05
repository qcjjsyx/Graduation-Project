# SystemC 当前主线

当前目录只保留与 Command/Descriptor v1 主线一致的基础模块：

- `include/command_codec.hpp`：T01 固定记录 codec；
- `reference/`：T02 独立 FP64 数学参考；
- `include/atu.hpp`、`include/hpu.hpp`：不依赖旧 ABI 的独立模块；
- `src/atu_hpu_demo.cpp`：独立模块冒烟测试。

原先由 `NodeTask/ABI v2` 驱动的 `system_sim`、artifact loader、memory
container、fixed/global-BFP kernel 和回归脚本已经退役并归档到：

```text
archive/legacy-node-task-abi-v2/systemc/
```

后续 T04/T05 在本目录直接建立 Command v1 memory、buffer 和 executor，
不得重新依赖归档中的 `NodeTask` 执行入口。
