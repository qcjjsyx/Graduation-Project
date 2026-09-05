# T00 可复现基线运行记录

状态：当前主线基线（baseline）
任务：T00 项目主线冻结与可复现基线

本文件冻结当前主线的构建、测试与 smoke test 结果，作为后续 T01、T02 和 SystemC 重构的稳定对照。所有命令可在新环境中复跑；相同配置与 seed 下结果可复现。

## 1. 记录时间

- 记录时间：2026-08-31 19:30 CST
- HEAD commit：`9824b6ec6871ee8a43a8c7adb163ffa31eb8cce4`（`9824b6e 20260826`，提交时间 2026-08-26 22:04:44 +0800）

## 2. 当前工作区状态摘要

- `git status --short`：仅有未跟踪文件 `专利/drawio重绘图/`、`专利/说明书-...docx`（与代码主线无关，本任务未触碰）。
- `git diff --stat`：空（无已跟踪文件修改）。
- 结论：代码主线工作树干净；唯一未跟踪内容在 `专利/`，不属于 T00 范围。
- 本任务未执行 `git reset/checkout/clean/rm -rf`，未覆盖或格式化任何已有修改。

## 3. 环境信息

| 项 | 值 |
|---|---|
| 操作系统 | macOS 26.6.2（BuildVersion 25G83） |
| 内核 | Darwin 25.6.0，arm64（Apple Silicon） |
| Python | 3.13.5（`/opt/anaconda3/bin/python3`） |
| pytest | 8.3.4 |
| numpy / scipy | 2.1.3 / 1.15.3 |
| CMake | 4.0.2 |
| 编译器 | Apple clang 21.0.0（clang-2100.1.1.101），target arm64-apple-darwin25.6.0 |
| GNU Make | 3.81 |
| SystemC | 3.0.2，`SYSTEMC_HOME=/usr/local/systemc-3.0.2`，`libsystemc.3.0.2.dylib`，含 `SystemCLanguage`/`SystemCTLM` CMake config |
| RTL 仿真器 | 无：`iverilog`、`verilator`、`vvp`、`gtkwave`、`yosys` 均未安装 |

## 4. 软件测试

- 命令：`cd graduation-code/software && python -m pytest -q`
- 结果：**32 passed, 0 failed**（约 0.31s）
- 失败测试名称：无
- 失败原因：无
- 是否已有问题：无失败，全部为通过项。

## 5. SystemC 测试

构建与测试命令（与 README 一致，均验证可执行）：

```bash
cd graduation-code/systemc
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

- 配置：成功（生成 `build/`）。
- 编译：成功，目标 `atu_hpu_demo`、`model_unit_tests`、`system_sim` 均构建完成。
- CTest 总测试数：3
- 通过数：3
- 失败数：0
- 失败测试名称：无

| 测试 | 结果 | 说明 |
|---|---|---|
| `atu_hpu_demo` | Passed | ATU/HPU SystemC 演示 |
| `model_unit_tests` | Passed | 数值/装配单元测试（含定点辅助） |
| `system_sim_e2e` | Passed | 端到端：软件产物 → system_sim，含复现性、反压、坏版本/截断/重叠/坏 map 拒绝、零 pivot 数值失败等断言 |

- 超时/死锁/数值失败/控制失败：本次 CTest 运行中**均无**（e2e 中的数值失败用例如零 pivot 属预期拒绝路径，断言其返回正确错误状态，非测试失败）。

## 6. RTL 测试/工具检查

- 工具：**当前环境无可用 RTL 仿真器**（iverilog/verilator/vvp/gtkwave/yosys 均未安装）。因此 RTL testbench **未运行**，仅从代码层面检查入口与明显错误。
- 运行入口：硬件目录下**没有** Makefile/运行脚本，需手工 `iverilog -g2012 ... && vvp ...`；testbench 自带 `$finish/$fatal/$display PASS` 自检。

代码层面检查结论：

| 模块 | RTL | testbench | 代码级判断 |
|---|---|---|---|
| `GCU/gcu_task_fetch.sv` | 完整 | `tb/tb_gcu_task_fetch.sv`（含简化 DDR 模型与校验） | 无明显错误 |
| `GCU/gcu_dep_scoreboard.sv` | 完整 | `tb/tb_scoreboard.sv` | 无明显错误 |
| `GCU/gcu_buffer_mgr.sv` | 完整 | `tb/tb_buffer_mgr.sv` | 无明显错误 |
| `GCU/gcu_micro_scheduler.sv` | 完整但含占位 | `tb/tb_gcu_micro_scheduler.sv` | Stage2 `S_F12/S_F21/S_SCHUR` 与 `PH_F12/PH_F21/PH_SCHUR` 显式标注为未实现占位（源码注释 TODO），与 README 一致 |
| `GCU/gcu_top.sv` | **空占位**（8 行，仅 clk/rstn，无逻辑） | 无 | 顶层未实现，与 README“原型或占位逻辑”一致 |
| `HPU/hpu_top.sv`(+`hpu_core_tree.sv`+`hpu_cmp_node.sv`) | 完整 | `tb_code/tb_hpu_top.sv`（扫 N=2..256） | 端口与 tb 对齐，无明显错误 |
| `HPU/tb_code/tb_hpu_cmp_node.sv`、`tb_hpu_core_tree.sv` | — | **空占位**（空 module） | 子模块 testbench 未实现 |
| `ATU/ATU.sv` | 完整（文件头自述 placeholder/待改，但有实际逻辑与端口） | `ATU/tb_ATU.sv`（init/query/swap） | 端口与 tb 对齐，无明显错误 |
| `ATU/ATU_asc.v` | **空文件**（0 字节） | — | 空占位 |
| `QAU/` | 无 RTL（仅 README） | — | Precision Adapter 尚无 RTL |
| `Matrix_Engine/` | **空目录** | — | 预留接口，无可联调 GEMM RTL |

- 结论：RTL 仅完成控制/接口级原型（GCU 子模块、HPU、ATU），顶层 `gcu_top` 与部分子 testbench 仍为占位；无 GEMM/Panel-LU/TRSM RTL；因无仿真器本次未运行。

## 7. 小型矩阵 smoke test

- 矩阵来源：当前主线测试夹具（`systemc/tests/run_e2e.py` 使用的确定性生成参数），非新增算法。
- 生成命令（软件）：`python -m src.main --out <artifact> --n 16 --density 0.2 --seed 3 --rhs-seed 11 --ordering amd`（运行目录 `graduation-code/software`）
- 软件输出：`nodes: 10, tasks: 10`，`original_residual_norm: 0.000e+00`
- 运行命令（SystemC）：`system_sim --artifact <artifact>/manifest.json --config config/default.json --mode both --out <result> --seed 1`
- backend/mode：`both`（含 `fp64` 与 `fixed`）；seed=1；config=`config/default.json`（seed 字段 1）
- 结果：`status=ok`，`completed_nodes=10/10`，`root_count=1`

关键指标（取自 `summary.json`）：

| 指标 | fp64 | fixed |
|---|---|---|
| relative_residual | 9.26e-17 | 4.86e-07 |
| componentwise_backward_error | 7.27e-17 | 5.47e-07 |
| relative_solution_error | 9.62e-17 | 5.04e-07 |
| factorization relative_error | 2.74e-17 | 4.58e-07 |

- cycles：total=1339（factorization=1004，solve=335）
- memory：read_bytes=3546，write_bytes=2374，read_cycles=599，write_cycles=1636，read_bursts=61，write_bursts=73
- buffer：busy_cycles=1202，wait_cycles=314
- 数值/控制状态：control_failure=false，address_failure=false，timed_out=false，failure_reason=空
- pivot/overflow/saturation/retry：stability.assembly_drop_count=0，matrix_overflow_count=0，precision_rescue_nodes=0，rescue_quantization_saturation_count=0；fixed.vector_shift_count=116（定点对齐移位，正常）
- 输出目录/关键文件：`<result>/{summary.json,nodes.csv,operations.csv,timeline.csv,memory.csv,solution.csv,final_memory_image.bin}` 均生成且非空。
- **复现性**：同 artifact、同 config、同 seed 二次运行，`summary.json` 与 `final_memory_image.bin` 字节完全一致（diff/cmp 均为 IDENTICAL）。

## 8. 真实矩阵 smoke test

- 矩阵来源：**当前主线** `graduation-code/software/example/256X256JJ.mat`（256×256 float64，键 `JJ`），RHS `example/256fuv.mat`（256×1 float64，键 `fuv`）。主线自含该矩阵，**无需**读取历史目录 `optimization-v1`（其副本仅只读对照，未使用）。
- 生成命令（软件）：`python -m src.main -mtx example/256X256JJ.mat --rhs example/256fuv.mat --ordering amd --out <artifact>`
- 软件输出：`nodes: 73, tasks: 73`，`original_residual_norm: 5.256e-11`
- 运行命令（SystemC）：`system_sim --artifact <artifact>/manifest.json --config config/default.json --mode both --out <result> --seed 1`
- backend/mode：`both`；seed=1；config=`config/default.json`
- 结果：`status=ok`，`completed_nodes=73/73`，`root_count=2`，无超时/死锁/控制失败

关键指标：

| 指标 | fp64 | fixed |
|---|---|---|
| relative_residual | 1.35e-12 | 5.38e-04 |
| componentwise_backward_error | 4.88e-16 | 1.54e-07 |
| relative_solution_error | 3.82e-08 | **2.90** |
| factorization relative_error | 2.53e-16 | 1.10e-07 |

- cycles：total=178492（factorization=171773，solve=6719）
- memory：read_bytes=1066136，write_bytes=608690，read_cycles=119901，write_cycles=515266，read_bursts=16704，write_bursts=9730
- buffer：busy_cycles=317243，wait_cycles=144799
- fixed 路径：refinement_iterations=2，used_precision_rescue=true
- pivot/overflow/saturation/retry：stability.assembly_drop_count=**367**，matrix_overflow_count=0，precision_rescue_nodes=**2**，rescue_quantization_saturation_count=0；control_failure=false，address_failure=false，timed_out=false
- 输出目录/关键文件：`<result>/{summary.json,nodes.csv,operations.csv,timeline.csv,memory.csv,solution.csv,final_memory_image.bin}` 均生成且非空。

## 9. 结果来源分类

按[验证计划与结果口径](验证计划与结果口径.md)的标记口径区分：

| 来源标记 | 本次结果 | 说明 |
|---|---|---|
| `HOST_FP64_REFERENCE` | **有结果** | 当前 CLI `fp64` 模式的 FP64 数学参考路径（上表 fp64 列） |
| `SYSTEMC_FP32_DEVICE_MODEL` | **未实现** | 目标 FP32 设备路径后端尚未作为独立命名后端存在；当前 `fixed` 是实验性定点/自适应精度路径，不等同目标 FP32 后端 |
| `SYSTEMC_INT32_GEMM_MODEL` | **未实现** | `Int32GemmBehavioral` 尚未作为独立后端实现；当前无 GEMM 边界 INT32 量化后端 |
| `RTL_CONTROL_PROTOTYPE` | **未运行** | 无 RTL 仿真器，且 `gcu_top` 仍为占位 |

补充说明：当前 `fixed` 结果（上表 fixed 列）属于迁移期的实验性定点/自适应精度路径，单独标注，不归入上述任一目标后端名称。

## 10. 已知失败与失败分类

- 软件 pytest：无失败。
- SystemC CTest：无失败。
- 运行期 smoke：两次 `status=ok`，无超时、无死锁、无控制失败、无地址错误。
- 已有问题（非本次引入，不通过降阈值/删测试隐藏）：
  1. **真实矩阵 fixed 路径解误差大**（relative_solution_error=2.90，伴随 precision_rescue_nodes=2、assembly_drop_count=367）：该矩阵条件数较高，实验性定点/自适应精度路径需精度救援；属模型既有数值行为，分类为“实验性定点精度限制”，T00 不修改。
  2. **RTL 不可运行且无仿真器**：`gcu_top.sv`、`tb_hpu_cmp_node.sv`、`tb_hpu_core_tree.sv`、`ATU_asc.v` 为占位/空，`Matrix_Engine/` 空，`QAU/` 仅文档。分类为“工具缺失 + 顶层/子模块占位”，T00 仅记录。
  3. **目标后端未实现**：`SYSTEMC_FP32_DEVICE_MODEL` 与 `SYSTEMC_INT32_GEMM_MODEL` 尚无独立后端。分类为“未实现”，等待后续任务。

## 11. 后续任务需要注意的问题

- T01（command/descriptor/status 契约）实现时，必须保留旧 CLI `--mode fp64|fixed|both` 可运行（迁移期不得删除旧入口），并维持相同输入下产物字节稳定（本基线已验证 fp64/fixed 复现性）。
- 本基线固定的小型夹具（n=16/density=0.2/seed=3/rhs-seed=11/amd）与真实矩阵（256X256JJ.mat + 256fuv.mat）可直接作为 T01/T02 的回归输入。
- fixed 路径在真实矩阵上的大解误差与精度救援是既有限制；后续精度/后端工作（T11、T13、T16）需单独报告，不得用黄金结果覆盖设备路径。
- RTL 联调（T14/T15）前需先安装仿真器（iverilog/verilator）并补齐 `gcu_top` 等占位模块；当前环境无仿真器，L4 级验证暂缓。
- 历史目录 `optimization-v1/v2/v3`、`archive/`、`unused/` 仅只读，本基线未读取其代码、未复制其逻辑回主线。
