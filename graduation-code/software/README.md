# 软件侧：Command/Descriptor v2 稀疏求解编译器

软件主线将稀疏矩阵和 RHS 编译成设备可读的 Command/Descriptor schema v2
artifact，不执行设备侧 Panel-LU、TRSM 或 GEMM 数值核。

```text
A、b
  → 可选 pow2 行均衡（D_r）
  → 加权最大乘积匹配行置换 P_match（默认开启，可显式关闭）
  → B = P_match D_r A；pattern(B) ∪ pattern(Bᵀ)
  → ordering / fill / elimination forest
  → supernode / front / child-update map
  → command + descriptor + payload + FP32 data + completion slots
  → manifest.json + memory_image.bin
```

主元策略由配置显式声明并写入每个 `FRONT_DESC`：默认 `STATIC`
（固定顺序、无交换，仍做零主元/阈值/非有限检查），可选
`DYNAMIC_PANEL`（仅 pivot block 内选主元）。阈值在配置时一次性规范化为
binary32 有效值，配置、manifest、参考与 descriptor 共用。

正常编译不执行整树数值求解，也不以数值成功为前提（可为验证设备主元失败
生成合法输入）。软件侧 FP64 前树参考是独立验证入口（
`src/verify/front_reference.py`）；加 `--reference-check` 显式开启编译期
预检与原始方程残差指标，失败则拒绝产出。外部 RHS 的 golden 参考解同样只
在 `--reference-check` 下求解（`spsolve`），未开启时参考解文件与相关指标
缺省。参考不向设备 image 注入任何结果。

与 C++ 独立参考的**同策略整树交叉验证**由共享 fixture
`graduation-code/fixtures/reference_tree/` 完成：Python 与 C++ 必须复现
同一组 `x_hat`、局部 P-vector 与主元失败分类（`tests/test_reference_cross.py`
与 `fp64_reference_tests`）。

设备 image 中的 front、factor、update、RHS 和 solution 使用 FP32；P-vector
使用 INT32。FP64 文件只供独立 checker 使用，不进入设备运行数据。全局
BFP、front exponent 和 `NodeTask` 已从当前主线移除。

运行：

```bash
cd graduation-code/software
python -m src.main \
  -mtx example/256X256JJ.mat \
  --ordering amd \
  --matching weighted_matching \
  --pivot-policy STATIC \
  --out /tmp/mf-command-v2
```

主要产物：

```text
manifest.json
memory_image.bin
front_f64.bin          # 变换后各节点 local front（FP64，checker 用）
rhs_hat_f64.bin        # b_hat（Q 序变量空间）
original_matrix_f64.bin
original_rhs_f64.bin
row_scale_e.bin
x_true_f64.bin         # golden 解（原始变量顺序）
x_hat_f64.bin          # 参考解（Q 序变量空间）
```

验证：

```bash
python -m pytest -q
```

旧 ABI v2 软件和测试位于
`archive/legacy-node-task-abi-v2/software/`，不属于当前构建和测试。
