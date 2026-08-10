# INT32 多前沿求解优化实验报告

最后更新：2026-07-26

## 1. 优化目标

基线已经证明 INT32 mantissa 可以完成 256JJ 和 1024JJ，但 576JJ 原方程 relative
residual 为 `2.346e-2`，且有 13 个 precision-rescue 节点。本轮不直接将矩阵存储改成
INT64/FP64，而是验证低硬件代价的 power-of-two 行列均衡能否改善共享 exponent 的动态
范围问题。

本实验全部位于 `optimization-v1/`，没有修改基线代码。

## 2. 实现内容

### 2.1 B1：交替行列满归一化

每轮先根据当前行 maxabs 选择行 exponent，再根据缩放后矩阵的列 maxabs 选择列
exponent：

```text
D_r[i] = 2^(-round(log2(row_max_i)))
D_c[j] = 2^(-round(log2(column_max_j)))
```

累计 exponent 默认限制在 `[-60, 60]`。SystemC 实际求解：

```text
D_r A D_c y = D_r b
x = D_c y
```

### 2.2 B2：幂次 Ruiz 均衡

B2 每轮只应用 maxabs exponent 的一半：

```text
delta_r = -round(0.5 log2(row_max))
delta_c = -round(0.5 log2(column_max))
```

行列 delta 使用同一轮开始时的矩阵计算，再同时更新。该方案比 B1 温和，用于避免一次
满归一化过度改变列尺度。

### 2.3 解坐标恢复

软件新增：

- `column_scale_e.bin`
- `original_solution_f64.bin`

SystemC 中的 `x_permuted` 仍表示排序后的变换变量 `y`；生成报告前先恢复 ordering，再
执行：

```text
x_original[i] = y_original[i] * 2^column_scale_e[i]
```

原方程 residual、componentwise backward error 和 solution error 全部使用恢复后的
`x_original`。迭代求精的 residual 仍在原始 `A,b` 上计算，修正 RHS 只应用 `D_r`；
低精度因子返回的修正变量会通过 `D_c` 自动恢复为原坐标修正。

## 3. 实验配置

除均衡方法外，三组实验配置相同：

- ordering：AMD；
- software source effective bits：30；
- SystemC `q_use_bits=26`；
- QF `F=20`；
- 64-bit accumulator；
- 8 workspace guard bits；
- 2 front buffers；
- tile 16；
- DDR 32 B/cycle；
- precision rescue 开启；
- 最多 50 轮迭代求精，原方程 residual 目标 `1e-3`。

## 4. 真实矩阵结果

| 矩阵 | 方案 | 原 residual | 初始 residual | 解误差 | 后向误差 | Rescue | IR | 周期 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | B0 row | `5.377e-4` | `1.473e-2` | `2.904` | `1.544e-7` | 2 | 2 | 176774 |
| 256 | B1 row-column | `6.408e-4` | `1.309e-2` | `1.929` | `2.819e-7` | 2 | 1 | 175098 |
| 256 | B2 Ruiz | `4.517e-4` | `3.728e-3` | `0.600` | `9.679e-7` | 2 | 1 | 175108 |
| 576 | B0 row | `2.346e-2` | `2.346e-2` | `2.362` | `8.346e-6` | 13 | 0 | 1687907 |
| 576 | B1 row-column | `4.593e-3` | `1.766e-2` | `0.958` | `2.599e-6` | 13 | 2 | 1700662 |
| 576 | B2 Ruiz | `1.120e-2` | `1.655e-2` | `2.914` | `1.967e-5` | 12 | 5 | 1708862 |
| 1024 | B0 row | `9.434e-4` | `3.177e1` | `1.394e-3` | `1.378e-4` | 1 | 40 | 7393107 |
| 1024 | B1 row-column | `9.730e-4` | `3.071e1` | `1.436e-3` | `1.418e-4` | 1 | 39 | 7378043 |
| 1024 | B2 Ruiz | `3.916e-4` | `1.160e-1` | `9.094e-3` | `3.462e-4` | 1 | 1 | 6805637 |

原始数据保存在 `results/real_matrix_comparison.csv`。

## 5. 结果解释

### 5.1 576验证了列缩放有价值

B1 相对 B0：

- residual 改善约 80.4%；
- solution error 改善约 59.4%；
- componentwise backward error 改善约 68.9%；
- 代价是总周期增加约 0.76%。

这说明 576 的误差不只是“INT32 位数不足”，列方向的动态范围确实是重要来源。

### 5.2 B2减少求精轮数，但不总是提高前向精度

1024 的 B2：

- residual 改善约 58.5%；
- 求精从 40 轮降到 1 轮；
- 总周期下降约 8.0%；
- 但 solution error 从 `1.394e-3` 增到 `9.094e-3`。

因此 B2 可以作为 residual/延迟优化候选，不能直接宣称解更准确。对于病态矩阵，缩放和
停止条件改变后，小 residual 仍可能对应较大的前向误差。

### 5.3 没有一个静态缩放模式支配全部指标

- 256：B2综合最好；
- 576：B1明显最好；
- 1024：B0前向误差最好，B2 residual和周期最好。

因此最终软件应保留多种模式，并在论文中分别报告 residual、后向误差、前向误差和周期，
而不是只选一个指标。

## 6. 576单因素扫描

在 B1 上继续扫描：

- `q_use_bits=24/26/28`；
- `frac_bits=16/20/24`；
- accumulator 48/64；
- guard bits 0/4/8；
- pivot threshold；
- IR 轮数。

结果中最小 residual 为 guard bits 0 的 `4.427e-3`，仍未达到 `1e-3`。其它关键结果：

| 变化 | residual | 解误差 |
|---|---:|---:|
| B1默认 | `4.593e-3` | `0.958` |
| q=24 | `5.242e-2` | `14.379` |
| q=28 | `6.741e-3` | `2.464` |
| F=16 | `1.132e-1` | `1.817` |
| F=24 | `1.568e-2` | `14.614` |
| guard=0 | `4.427e-3` | `0.697` |
| guard=4 | `6.572e-3` | `0.986` |

这说明仅继续调整全局位宽参数不能解决 576。下一项数值优化应该改变缩放粒度或主元策略，
而不是继续增加全局 mantissa 位数。

## 7. 对硬件设计的结论

### 可以直接采用

- 保留 INT32 矩阵存储；
- 软件生成 `D_r/D_c` power-of-two exponent；
- QAU继续使用 int64 accumulator；
- U/update独立 exponent；
- Host在输出端恢复 `x=D_c y`。

列缩放不需要矩阵核增加通用乘法器，只需要软件修改输入 exponent 和 solution writer/Host
调整输出 exponent。

### 暂不固定

- B2不应作为所有矩阵的唯一默认模式；
- 不建议仅为了576把全系统升级成INT64；
- guard bits不是越多越好，应保持可配置；
- 不能用 residual 单独选择缩放模式。

### 下一项优先研究

576在B1和全局参数扫描后仍停留在约 `4.4e-3`。下一步应按以下顺序推进：

1. 16×16 tile/block BFP exponent；
2. 数值匹配或对角增强 ordering；
3. risk-aware supernode 划分；
4. delayed pivot和动态front；
5. 以低精度LU为预条件器的Krylov修正。

其中 tile exponent 最贴近当前 QAU/矩阵核架构，也最适合作为下一版独立实验。

## 8. 验证状态

- 优化软件 pytest：32项通过；
- 优化 SystemC CTest：3项通过；
- B0结果与基线数值一致；
- B1/B2均完成256/576/1024 fixed分解与求解；
- B1/B2都按原坐标恢复解并计算原方程指标；
- 自动比较入口：`scripts/compare_scaling.py`。

## 9. 可复现命令

```bash
cd graduation-code/optimization-v1/systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel

cd ..
python scripts/compare_scaling.py \
  --out /tmp/graduation-scaling-comparison
```

每个运行目录均保留 artifact、`summary.json`、nodes/operations/memory/timeline CSV 和最终
DDR 镜像，可继续作为下一版 tile exponent 的对照基线。
