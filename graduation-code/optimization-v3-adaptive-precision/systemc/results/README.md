# v3 精度策略结果

`precision_comparison.csv` 保存相同制品、seed 和 `fixed` 模式下的三组对照：

- `baseline_v2_f20`：冻结的 v2，所有节点 L 使用 QF20；
- `uniform_f26`：v3 推荐方案，所有节点 L 使用 QF26；
- `adaptive_f20_f26`：先 QF20，局部因子检查失败后用 QF26 重算，仍失败才进行
  FP64 rescue。

大体积的 DDR 快照和逐操作日志不纳入本目录。可用 README 中的命令重新生成。
对比结论和字段解释见
[`../../docs/ADAPTIVE_PRECISION_OPTIMIZATION.md`](../../docs/ADAPTIVE_PRECISION_OPTIMIZATION.md)。
