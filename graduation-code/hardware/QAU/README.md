# QAU：Quantization and Assembly Unit

QAU 是计划实现的量化装配 RTL 原型，用来证明硬件侧在线装配与 node-scale 选择具备可实现性。

## 建议的最小功能

1. 扫描本地贡献与子节点 update 的 exponent，选择 `e_asm = max(e_src)`。
2. 对有符号 int32 mantissa 执行舍入移位和指数对齐。
3. 使用 int64 累加器完成流式装配，并统计 `maxabs_acc`。
4. 根据最高有效位计算 node-scale 调整量。
5. 将 int64 装配结果重新量化为 int32，执行舍入、裁剪和饱和。
6. 输出 `align_drop_count`、`align_shift_max`、`asm_overflow_count` 和 `requant_sat_count`。

## 范围边界

- QAU 不负责解析完整二进制 map table；SystemC 或 GCU 提供目标地址/索引。
- QAU 不内置大容量 front SRAM；采用外部 SRAM 风格的读改写接口。
- 首版只需支持小规模固定 front 的 testbench，再逐步参数化。
- RTL 结果必须逐元素对照 Python 和 SystemC 量化黄金模型。

