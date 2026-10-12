# AGENTS.md

多前沿（multifrontal）稀疏 LU 求解器的软硬件协同毕业设计。完整介绍见 `项目主线梳理.md`，
当前待办见 `todo.md`。开始任何任务前先读这两个文件。

## 主线（一句话）

Python 编译器把稀疏矩阵编译成 **Command/Descriptor schema v2** 设备镜像 → SystemC/C++ 设备模型
`tree_factor_device` 执行 FP32 分解和树形前代/回代 → 独立 FP64 checker（`tree_factor_check` 验因子、
`tree_solve_check` 验求解）验收。

- 已完成：T00–T12（STATIC 主元下的分解 + 求解闭环）。
- 未完成：T14 动态主元（设备遇到 `DYNAMIC_PANEL` 返回 `POLICY_NOT_SUPPORTED`）；RTL 没有接入主线
  （HPU/ATU 只是独立原型，GCU 仍读取旧的 16B NodeTask）。

## 目录

| 路径 | 内容 |
|---|---|
| `graduation-code/software/` | Python 编译器：`src/main.py`（CLI）、`src/scheduler/command_compiler.py`、`src/command_codec.py`、`src/verify/`（manifest 校验、FP64 参考）、`tools/numeric_diagnosis.py`、`example/*.mat`（真实矩阵） |
| `graduation-code/systemc/` | C++/SystemC 设备模型：`src/controlplane/`（控制面、执行单元，含 `solve_exec_unit.cpp`）、`src/dataplane/`（loader、descriptor reader、MemoryService）、`reference/`（独立 FP64 参考和 checker）、`tests/`、`tools/*_fixture_gen.py` |
| `graduation-code/hardware/` | Verilog RTL 原型（HPU/ATU/GCU/QAU），`run_rtl_tests.sh` 用 iverilog 运行 |
| `docs/` | 设计与验证文档。`architecture/项目总体架构设计.md` 是系统组成与数据流入口；`software/软件侧技术文档.md` 是软件侧统一实现说明；`systemc/树形求解模块设计.md` 是求解部分的权威设计；`interface/Command_Schema_v2.md` 是接口权威；`theory/` 是矩阵求解与数值误差理论；`systemc/` 是模块设计与代码阅读入口；`verification/` 保存验证规范和具体记录 |
| `experiments/` | 数值实验：`precision_boundary/` 为精度边界实验，`real_matrix_diagnosis/` 为真实矩阵诊断；各目录保存实验报告和按批次组织的结果、元数据与图表 |
| `archive/` | 历史材料，**只读，不要修改** |

## 构建与测试（macOS）

Python 环境是仓库根目录的 `.venv`（numpy、scipy、pytest）；CMake 默认用它生成 fixture。
SystemC 安装在 `/usr/local/systemc-3.0.2`。

```bash
# Python 测试
cd graduation-code/software && ../../.venv/bin/python -m pytest -q

# C++ Release
cmake -S graduation-code/systemc -B /tmp/gp-rel -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DSYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake --build /tmp/gp-rel -j && ctest --test-dir /tmp/gp-rel --output-on-failure

# C++ ASan+UBSan（每次改 C++ 后两种构建都要跑）
cmake -S graduation-code/systemc -B /tmp/gp-san -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
      -DENABLE_SANITIZERS=ON -DSYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake --build /tmp/gp-san -j
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir /tmp/gp-san --output-on-failure
```

当前回归基线与历史阶段结果统一见 `当前进度.md`；原始验收证据见
`docs/verification/验证记录.md`。

## 端到端流程

在 `graduation-code/software` 下运行（`BUILD` 是 Release 构建目录）：

```bash
python -m src.main --n 128 --density 0.03 --out /tmp/rand            # 随机矩阵；真实矩阵用 --mtx example/256X256JJ.mat
$BUILD/tree_factor_device --device-dir /tmp/rand --out /tmp/rand-rep --backend FP32_SYSTEMC
$BUILD/tree_factor_check /tmp/rand /tmp/rand-rep
$BUILD/tree_solve_check  /tmp/rand /tmp/rand-rep                      # 输出 eta_forward / eta_backward / 原方程误差等
python tools/numeric_diagnosis.py /tmp/rand /tmp/rand-rep             # 条件数、节点后向误差、设备解对比、迭代求精
```

- CLI 默认 scope 是 `factorization_and_solve`；`--factorization-only` 只生成分解命令。
- 设备片上容量由 `--device-buffer-bytes` 决定（默认 64 KiB，对应 front ≤ 128）。真实矩阵参数：
  256JJ 用默认值；576JJ 加 `--device-buffer-bytes 131072`；1024JJ 加
  `--max-front-size 300 --device-buffer-bytes 524288`。
- 仿真耗时：576JJ 约 600 万周期、1024JJ 约 1400 万周期，每个要跑几分钟。

## 固定边界（不能违反）

- FP64 只用于参考和验收，不进入设备镜像；设备**不得读取任何 golden 文件**（`front_f64.bin`、
  `x_true_f64.bin`、`original_*_f64.bin` 等）。参考结果不得注入设备。
- 设备只消费 schema v2 的 Command / Descriptor / Completion（32 / 64 / 64 字节）；不恢复 NodeTask，
  未经设计文档修改不改记录布局和 schema 版本。
- 不做全局量化或 BFP；INT32 只用在 GEMM 调用内部。
- 静态主元失败不等于矩阵奇异；不静默替换小主元或零主元；不吞掉 NaN/Inf。
- `status=OK` 不等于数值正确：报告结果时必须同时给出后向误差（η_f、η_b、原方程后向误差）和解误差。
- 单元测试的期望值必须手算或来自独立参考，不得用被测函数自己生成。
- C++ 不得依赖未规定的求值顺序（本仓库曾因 `f(x.size(), std::move(x))` 在 GCC 上越界写）。
- “已完成”必须同时有代码、测试和可复现证据，写进 `docs/verification/`。

## 工作约定

- 改代码时同步改文档：接口改 `docs/interface/`，设计偏离先改对应设计文档再改代码，完成后更新 `todo.md`。
- 修改既有测试的期望前先说明理由，并征得同意。
- 不要自行 git commit 或 push，除非用户要求。
- 回复用户时用中文。

## 数值背景（解读结果时必须知道）

详见 `experiments/real_matrix_diagnosis/实验报告.md` 与 `experiments/precision_boundary/实验报告.md`。
较早真实矩阵诊断的原始 JSON 未找到，目前保留报告级历史结果；精度边界实验已归档 365 例主批次与 8 例独立补跑批次。

- `256X256JJ`、`576X576JJ` 接近奇异：所测试的对角缩放后条件数仍有 1e10、1e13 量级。
  已归档精度实验中，设备求解的 η 在 1e-7 量级，但原方程解误差可达 10～1000 倍；
  三角求解通过不表示原方程解准确。条件数和静态消元增长都需单独分析，不能据此排除所有实现缺陷。
  本批输入使用的 host 求精未收敛；不能外推为所有主元策略或求精方法均失败。
- `1024X1024JJ` 缩放后条件数约 1e9；较早 host 诊断的 FP64 三角修正步能把残差降到机器精度，
  较新的 FP32 三角修正步未达到边界实验成功判据。二者均不代表设备求精流程已完成。
- 随机矩阵良态，所有指标都在 1e-7 量级。
- Linux/GCC 和 macOS/clang 的浮点末位会不同（例如 FMA 收缩），η 的量级一致即可，不要求逐位相同。

## 已知问题

- 分解阶段停滞（2026-10-11 已修复）：根因是 PANEL_LU 的 tile 版本键与 TRSM/GEMM 键空间别名，
  不是调度窗口；见 `docs/verification/验证记录.md` §10.11。乱序准入下 fetch FIFO 能否被门控
  命令填满而独立成环尚未证明（无反例）。
- 分解侧：父节点 `NODE_BEGIN` 等待全部子节点的 token，子节点数 > `max_wait_tokens`（默认 256）
  的节点无法编译（求解侧的同类限制已在 2026-10-11 去掉）。
- `graduation-code/systemc/build*` 是旧的本地构建目录，可能早于 T11/T12；验证时请用新的构建目录。
