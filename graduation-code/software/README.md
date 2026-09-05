# 软件侧：Command v1 稀疏求解编译器

软件主线将稀疏矩阵和 RHS 编译成设备可读的 Command/Descriptor v1
artifact，不执行 Panel-LU、TRSM 或 GEMM 数值核。

```text
A、b
  → 可选 pow2 行均衡
  → pattern(A) ∪ pattern(Aᵀ)
  → ordering / fill / elimination forest
  → supernode / front / child-update map
  → command + descriptor + payload + FP32 data + completion slots
  → manifest.json + memory_image.bin
```

设备 image 中的 front、factor、update、RHS 和 solution 使用 FP32；P-vector
使用 INT32。FP64 文件只供独立 checker 使用，不进入设备运行数据。全局
BFP、front exponent 和 `NodeTask` 已从当前主线移除。

运行：

```bash
cd graduation-code/software
python -m src.main \
  -mtx example/256X256JJ.mat \
  --rhs example/256fuv.mat \
  --ordering amd \
  --out /tmp/mf-command-v1
```

主要产物：

```text
manifest.json
memory_image.bin
reference_front_f64.bin
rhs_f64.bin
original_matrix_f64.bin
original_rhs_f64.bin
row_scale_e.bin
x_reference_f64.bin
```

验证：

```bash
python -m pytest -q
```

旧 ABI v2 软件和测试位于
`archive/legacy-node-task-abi-v2/software/`，不属于当前构建和测试。
