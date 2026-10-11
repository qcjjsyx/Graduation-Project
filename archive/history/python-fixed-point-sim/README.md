# Python 行为模型与历史实验

状态：辅助验证和历史实验目录

本目录不是 SystemC，也不是 RTL。它用于快速验证数学想法、生成小规模实验和检查某些硬件策略的趋势。

## 1. 文件职责

| 文件 | 职责 |
|---|---|
| `hardware.py` | 早期硬件行为模型，包含定点 LU、TRSM、Schur、pivot、ATU 和迭代求精实验 |
| `BLU_TRSM.py` | 块 LU/TRSM 数学实验 |
| `quant.py` | 量化和定点辅助实验 |
| `hardware_experiment_report.py` | 批量实验和报告生成 |
| `results/` | 历史实验结果，不作为当前主线基准 |

## 2. 与 SystemC 的关系

```text
Python sim
  → 快速数学/参数实验

SystemC
  → 完整 command、buffer、依赖、周期和求解模型

RTL
  → 关键控制、地址和未来计算模块原型
```

Python 结果可以作为算法和量化的早期参考，但不能替代 SystemC 的硬件数据流，也不能替代 RTL testbench。

## 3. 当前主线使用方式

```bash
python graduation-code/sim/hardware.py \
  --n 32 \
  --mode stable \
  --seed 42 \
  --ir-iters 5
```

建议使用它快速观察：

- pivot 和 row swap；
- LU/TRSM/GEMM 的数学输出；
- 量化误差、overflow 和 saturation；
- 迭代求精趋势；
- 不同 tile、scale 和矩阵条件的影响。

## 4. 结果口径

历史结果必须标记为 Python behavior model。不能将其写成：

- SystemC 周期结果；
- Verilog RTL 结果；
- 真实 FPGA 吞吐；
- INT32 GEMM RTL 联调结果。

正式实验应优先使用 `FP64_GOLDEN`、`FP32_SYSTEMC` 和 `FP32_SYSTEMC_INT32_GEMM` 三种后端，并将 Python 结果作为辅助对照。

