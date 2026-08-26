# HPU 功能规范

状态：当前主线 RTL 规范

## 1. 目标

HPU（Pivot/Hazard Unit）在当前 Panel 的一个消元列中，从候选元素中选择 pivot，并输出逻辑行号和状态。首版优先保证确定性、边界清晰和可验证，不同时实现多种复杂主元策略。

## 2. 输入输出语义

输入候选：

```text
candidate_valid
candidate_ready
candidate_last
candidate_value
candidate_logical_row
```

输出结果：

```text
pivot_valid
pivot_ready
pivot_value
pivot_logical_row
pivot_unstable
pivot_not_found
```

一个候选段由 `candidate_last` 或 descriptor 中的 candidate count 结束。空候选段必须产生 `pivot_not_found`。

## 3. 首版选择策略

首版采用全段最大绝对值归约：

```text
argmax(abs(candidate_value))
```

tie-break 必须固定。推荐相同绝对值时保留较小 logical row，以保证 Python、SystemC 和 RTL 结果可复现。

HPU 不负责：

- 修改 P-vector；
- 执行行交换；
- 访问 DDR 或 front SRAM；
- 决定 delayed pivot；
- 把 update 行自动提升为 pivot。

这些行为分别属于 GCU、ATU 或软件错误/失败处理。

## 4. SystemC 与 RTL 一致性

SystemC HPU 和 RTL HPU 必须共享：

- 候选顺序语义；
- valid/ready；
- last/count 结束语义；
- abs 比较规则；
- signed/unsigned 解释；
- tie-break；
- zero、NaN/非法值和 empty candidate 的状态。

没有要求 SystemC 和 RTL 的内部归约树结构相同，但必须对同一候选流输出相同的 pivot 和 failure 状态。

## 5. 非首版功能

以下功能作为后续实验，不进入第一版验收门：

- threshold pivot；
- CALU/tournament；
- rook pivoting；
- 多候选每拍输入；
- 跨 front 的 pivot 协调。

只有在实际测得 HPU 扫描成为瓶颈时，才增加这些模式。

## 6. 必测场景

- 单候选；
- 最大候选容量；
- 空候选；
- 全零候选；
- 正负值混合；
- 相同绝对值 tie-break；
- pivot 在首行、末行和中间行；
- valid 间断；
- output backpressure；
- reset 和连续两段候选；
- 非法/溢出数值状态。

