# 优化实验 ABI v2 扩展

优化实验继续使用 128 字节 ABI v2 `NodeTask`，没有修改任务描述符字段或现有 DDR 地址
含义。新增内容均为 Host Checker 黄金旁路：

| 文件 | 类型 | 含义 |
|---|---|---|
| `column_scale_e.bin` | little-endian int16[N] | 原始列编号顺序的 `D_c` exponent |
| `original_solution_f64.bin` | little-endian FP64[N] | 原始坐标参考解 |

manifest 的 `equilibration` 段新增：

```text
column_scale_exponent_file
column_scale_exponent_dtype
column_scale_exponent_count
solution_requires_unscale
```

对于 B1/B2：

```text
D_r A D_c y = D_r b
x = D_c y
solution_requires_unscale = true
```

`x_reference_f64.bin` 保存排序坐标下的变换变量参考值 `y`；
`original_solution_f64.bin` 保存原始坐标下的 `x`。SystemC 的 factor/solve 数据通路处理
`y`，Host Checker 恢复 `x` 后再计算原方程 residual 和解误差。

`column_scale_e.bin` 不进入当前 DDR 镜像，它代表软件/Host 的变量变换 metadata。若未来
要求硬件直接输出原变量，可将 exponent 数组增加为全局只读 DDR 区域，并在 solution
writer 中执行 exponent 调整。
