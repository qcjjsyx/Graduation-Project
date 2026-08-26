# 硬件行为级仿真实验汇总

生成时间：2026-06-15 15:09:17

## 实验目的

本脚本批量运行 `hardware.py` 中的硬件行为模型，用于展示在加入 ATU、HPU 与量化数据格式后，定点 LU、TRSM、Schur 更新和迭代求精的整体行为。默认覆盖多种矩阵模式、多个矩阵规模和多个随机种子，避免只展示单一特定矩阵。

## 实验配置

- 矩阵模式：stable, pivot_stress, random, large_value
- 矩阵规模：128, 256, 512
- 随机种子：1, 2, 3
- tile size：16
- q_use_bits：27
- frac_bits：20
- IR 最大迭代次数：8
- IR residual tolerance：1e-10

## 总体结果

- 成功用例：36 / 36
- 失败用例：0
- 直接求解 residual 中位数：1.334e-01
- IR 后 residual 中位数：1.068e-11
- LU 相对误差中位数：4.415e-06
- residual 改善倍数中位数：1.532e+10

## PPT 可用代表性结果

| 矩阵模式 | 规模 | seed | swap 次数 | LU 相对误差 | 直接 residual | IR 后 residual | IR 收敛 | 估算周期 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| stable | 512 | 1 | 0 | 4.419e-06 | 3.133e-01 | 3.088e-11 | 是 | 539904 |
| pivot_stress | 512 | 1 | 401 | 1.031e-05 | 1.545e-01 | 1.348e-12 | 是 | 539904 |
| random | 512 | 1 | 403 | 5.835e-05 | 8.693e-01 | 1.245e-12 | 是 | 539904 |
| large_value | 512 | 1 | 404 | 7.758e-06 | 1.719e+03 | 5.484e-09 | 否 | 539904 |

## 图表文件

- `figures/residual_convergence_by_mode.png`：不同矩阵模式下 residual 相对初始 residual 的中位收敛曲线，阴影为 25%-75% 分位区间。
- `figures/final_residual_by_mode.png`：不同矩阵模式下 IR 后最终 residual 分布。
- `figures/op_counts_by_size.png`：不同规模下硬件行为模型的操作数量。
- `figures/quantization_risk_counters.png`：量化/装配风险计数统计。

## 数据文件

- `run_summary.csv`：每个实验用例的指标汇总。
- `residual_history.csv`：每个用例每次迭代的 residual 历史，可用于重新绘图。
- `op_counts.csv`：硬件操作数量统计。
- `quant_stats.csv`：量化与装配风险指标。
