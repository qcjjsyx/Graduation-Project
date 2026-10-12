# 多前沿稀疏 LU 求解器软硬件协同毕业设计

本项目研究面向稀疏线性方程 `Ax=b` 的多前沿 LU 求解器软硬件协同设计，
关注数值可靠性、计算资源组织和存储开销。工程包含 Python 编译工具、
SystemC/C++ 设备架构模型、RTL 模块及独立验证工具。

## 文档入口

| 文档 | 阅读目的 |
|---|---|
| [项目主线梳理](项目主线梳理.md) | 理解项目目标、数据流、代码分工和下一阶段方向 |
| [当前进度](当前进度.md) | 查看已交付能力、验证证据和完成边界 |
| [todo](todo.md) | 查看当前尚未完成的工作 |

理论、接口、模块设计和验证规范/记录放在 `docs/`；数值实验的报告、批次数据和图表放在
`experiments/`，由上述文档链接进入。

## 环境准备

以下命令按 macOS 编写，均从仓库根目录开始。

- Python 3.10+，使用根目录 `.venv`，依赖 NumPy、SciPy、pytest。
- CMake 3.16+、支持 C++17 的编译器、SystemC 3.0.x。
- 本机 SystemC 路径为 `/usr/local/systemc-3.0.2`；其他环境替换下方路径。
- RTL 仿真使用 Icarus Verilog 12+，需要 `iverilog` 和 `vvp`。

本机已有 `.venv`。新环境可按下列命令创建：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy pytest
```

## 构建与测试

### 分别运行 Python、C++ 和端到端样例

从仓库根目录执行对应脚本；四个脚本彼此独立，不会自动依次执行：

```bash
bash scripts/run_pytest.sh
bash scripts/run_cpp_build.sh
bash scripts/run_cpp_test.sh
bash scripts/run_sample.sh
```

其中 `run_cpp_build.sh` 会完成 CMake Release 配置和 C++/SystemC 编译；
`run_cpp_test.sh` 需要先存在已经配置和编译好的构建目录；`run_sample.sh` 需要先
完成 `run_cpp_build.sh`，然后运行 Python 编译器、SystemC 设备模型、因子 checker
和求解 checker。

脚本默认使用根目录 `.venv`、`/usr/local/systemc-3.0.2` 和
`/tmp/gp-docs-rel`。`--sample` 的输入和报告会写入自动创建的 `/tmp` 临时目录。
如需覆盖这些设置，可在执行时传入环境变量，例如：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
SYSTEMC_HOME=/usr/local/systemc-3.0.2 \
BUILD_DIR=/tmp/gp-rel JOBS=4 \
bash scripts/run_cpp_build.sh
```

运行样例时也可以固定输出目录：

```bash
SAMPLE_DIR=/tmp/gp-docs-smoke \
REPORT_DIR=/tmp/gp-docs-smoke-rep \
bash scripts/run_sample.sh
```

### Python

```bash
cd graduation-code/software
../../.venv/bin/python -m pytest -q
cd ../..
```

### C++ / SystemC Release

使用新的构建目录，避免复用仓库中残留的旧构建产物。
CMake 默认使用仓库 `.venv` 生成测试输入。

```bash
cmake -S graduation-code/systemc -B /tmp/gp-docs-rel \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake --build /tmp/gp-docs-rel -j 4
ctest --test-dir /tmp/gp-docs-rel --output-on-failure
```

### ASan / UBSan

修改 C++ 后同时执行 Release 与 sanitizer 回归。

```bash
cmake -S graduation-code/systemc -B /tmp/gp-docs-san \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DENABLE_SANITIZERS=ON \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake --build /tmp/gp-docs-san -j 4
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir /tmp/gp-docs-san --output-on-failure
```

### RTL

在仓库根目录执行：

```bash
bash graduation-code/hardware/run_rtl_tests.sh
```

RTL 仿真的覆盖范围和其他入口见 [硬件目录说明](graduation-code/hardware/README.md)。

## 运行一个完整样例

先完成上述 Release 构建，再执行：

```bash
cd graduation-code/software
../../.venv/bin/python -m src.main --n 128 --density 0.03 --seed 0 \
  --out /tmp/gp-docs-smoke
/tmp/gp-docs-rel/tree_factor_device \
  --device-dir /tmp/gp-docs-smoke --out /tmp/gp-docs-smoke-rep \
  --backend FP32_SYSTEMC
/tmp/gp-docs-rel/tree_factor_check \
  /tmp/gp-docs-smoke /tmp/gp-docs-smoke-rep
/tmp/gp-docs-rel/tree_solve_check \
  /tmp/gp-docs-smoke /tmp/gp-docs-smoke-rep
cd ../..
```

输入目录保存编译产物，报告目录保存设备执行报告、最终镜像和两份验收报告。
解读结果时同时查看因子验收、前代/回代后向误差、原方程后向误差和解误差；
具体口径见 [验证规范](docs/verification/验证规范.md)。

真实矩阵在 `graduation-code/software/example/`。在上述软件目录的编译命令中，
用 `--mtx` 替换随机矩阵参数，并按矩阵设置容量：

| 矩阵 | 编译参数 |
|---|---|
| 256JJ | `--mtx example/256X256JJ.mat` |
| 576JJ | `--mtx example/576X576JJ.mat --device-buffer-bytes 131072` |
| 1024JJ | `--mtx example/1024X1024JJ.mat --max-front-size 300 --device-buffer-bytes 524288` |

真实矩阵的数值限制见 [真实矩阵数值分析](experiments/real_matrix_diagnosis/实验报告.md)。
更多输入、后端和运行选项见 [软件说明](graduation-code/software/README.md)
和 [SystemC 说明](graduation-code/systemc/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `graduation-code/software/` | Python 编译工具、数值诊断与实验代码 |
| `graduation-code/systemc/` | C++/SystemC 模型、独立参考与验收工具 |
| `graduation-code/hardware/` | RTL 模块和仿真入口 |
| `graduation-code/fixtures/` | 跨语言共享的测试输入 |
| `docs/` | 算法、接口、设计和验证资料 |
| `参考文献/` | 文献与阅读笔记 |
| `archive/` | 历史材料，只读，不参与当前主线构建 |
