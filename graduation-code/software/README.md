# 软件侧：稀疏求解编译器

状态：当前主线说明

软件侧不执行硬件计算，而是将输入矩阵转换为设备能够执行的结构和数据产物。

## 1. 软件职责

```text
A、b
  → 行均衡（可选）
  → pattern(A) ∪ pattern(Aᵀ)
  → ordering / fill / elimination forest
  → supernode / front / update map
  → artifact ABI v2
  → command stream（目标接口）
```

具体职责包括：

- 保留原始矩阵数值和方向性；
- 使用非零模式对称包络进行符号分析；
- 构建消除森林、supernode、front 和 child update map；
- 确保每个原始数值元素唯一归属；
- 规划 DDR 地址、区域所有权和 buffer 需求；
- 生成 manifest、memory image 和当前 ABI v2 `NodeTask`；
- 后续由 command compiler 生成 command ring 和 descriptor table。

## 2. 当前产物接口

当前 SystemC 运行路径读取：

```text
manifest.json
memory_image.bin
```

其中 `memory_image.bin` 包含当前 ABI v2 所需的任务、front、map 和数据区域。独立的 FP64 front、原始矩阵、RHS 和参考解文件只用于调试或黄金检查，不得被设备计算路径当作旁路输入。

当前 ABI v2 的固定任务记录为 128 字节 `NodeTask`。它仍然是迁移阶段的实际接口，不是目标长期接口。

## 3. 目标 command compiler

目标软件接口为：

```text
NodeTask / forest / front metadata
              ↓
       Command Compiler
              ↓
command ring + descriptor table + data image
```

command 只表达 `PANEL_LU`、`TRSM`、`GEMM_SCHUR`、`STORE`、`COMMIT` 等语义操作；front 地址、尺寸、stride、map、局部 GEMM scale 和异常策略进入 descriptor。

迁移策略是先增加适配器，不立即删除 ABI v2：

1. 现有 pipeline 继续生成 ABI v2；
2. 从同一内部 front graph 生成 command stream；
3. 用命令解释器和现有 SystemC 结果交叉检查；
4. command executor 稳定后，停止扩张 `NodeTask` 字段。

## 4. 数值数据原则

- FP64 作为软件黄金参考；
- FP32/FP64 front 作为 SystemC 主设备路径；
- INT32 只在 GEMM tile 调用边界量化；
- 不将 INT32/BFP scale 传播为整棵消除树的全局格式；
- 量化导致 pivot 或 residual 失败时，返回明确状态，不使用黄金结果覆盖。

## 5. 当前运行

```bash
cd graduation-code/software
python -m src.main \
  -mtx example/256X256JJ.mat \
  --rhs example/256fuv.mat \
  --ordering amd \
  --out /tmp/mf-256
```

验证软件：

```bash
cd graduation-code/software
python -m pytest -q
```

## 6. 软件输出检查

必须检查：

- 矩阵是否方形；
- 原始数值非对称性是否被保留；
- 符号来源是否为 `pattern(A) ∪ pattern(Aᵀ)`；
- front、forest、supernode 和 map 是否完整；
- 地址范围、对齐和区域是否重叠；
- descriptor 引用是否有效；
- front/panel/tile 是否超过当前硬件支持上限。

## 7. 不负责的内容

软件不负责：

- 在设备侧动态生成 fill；
- 在硬件中维护消除树缓存；
- 为每个消除树节点定义一套独立全局量化格式；
- 直接把 Host 指针传递给 SystemC 或 RTL；
- 把 FP64 黄金结果写入设备计算 buffer。

