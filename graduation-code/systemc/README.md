# SystemC 当前主线

当前目录保留与 Command/Descriptor schema v2 主线一致的模块：

- `include/command_codec.hpp`：schema v2 固定记录 codec（含主元策略字段）；
- `include/dataplane/`、`src/dataplane/`：T04 Command 数据面（ArtifactLoader、
  DeviceMemory/RegionRegistry、MemoryService、DescriptorReader、BufferManager、
  DmaEngine、Trace、DataPlane 编排、selftest）；
- `reference/`：独立 FP64 数学参考（`PivotPolicy`/`PivotFailure`、
  `panel_lu_policy` 与通用 `panel_lu` 分离；`tools/fp64_reference_smoke`
  为数值冒烟入口）；
- `include/atu.hpp`、`include/hpu.hpp`：DYNAMIC_PANEL 用的独立模块
  （静态路径旁路）；
- `src/atu_hpu_demo.cpp`：独立模块冒烟测试；
- `src/dataplane_smoke.cpp`：T04 加载—搬运—写回冒烟入口；
- `tests/command_codec_tests.cpp`：Python/C++ 共用 fixture 的 codec 测试；
- `tests/dataplane_core_tests.cpp`：普通 C++ 数据/校验层测试；
- `tests/dataplane_systemc_tests.cpp`：SystemC 时序测试（sc_clock 驱动）；
- `tests/dataplane_artifact_tests.cpp`：真实 artifact 集成测试；
- `tools/t04_fixture_gen.py`：用当前软件编译器生成 T04 设备运行目录。

原先由 `NodeTask/ABI v2` 驱动的 `system_sim`、artifact loader、memory
container、fixed/global-BFP kernel 和回归脚本已经退役并归档到：

```text
archive/legacy-node-task-abi-v2/systemc/
```

## 构建与测试

SystemC 安装位置通过 `SYSTEMC_HOME` 指定。构建目录建议放在 `/tmp`：

```bash
export SYSTEMC_HOME=/usr/local/systemc-3.0.2
cmake -S graduation-code/systemc -B /tmp/t04-build -DBUILD_TESTING=ON
cmake --build /tmp/t04-build -j8
ctest --test-dir /tmp/t04-build --output-on-failure
```

sanitizer 构建：

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build-sanitize \
  -DBUILD_TESTING=ON -DENABLE_SANITIZERS=ON
cmake --build /tmp/t04-build-sanitize -j8
ctest --test-dir /tmp/t04-build-sanitize --output-on-failure
```

CTest 会先运行 `t04_generate_fixtures`（默认使用仓库根 `.venv/bin/python`，
可用 `-DT04_PYTHON=...` 覆盖），用当前软件编译器生成单节点/多节点/多根/
DYNAMIC_PANEL/256 阶设备运行目录，然后运行：

- `dataplane_core_tests`：schema v2 加载、非法 artifact、u64 溢出/重叠/对齐、
  P-vector 负面用例、DeviceMemory 访问规则、buffer 生命周期、内存服务时序；
- `dataplane_systemc_tests`：连续/二维 DMA 逐字节正确性、位模式保持、
  固定配置周期断言、请求/响应背压、双 buffer ping-pong、reset 中断、
  非法请求无部分写入、guard 字节不变；
- `dataplane_artifact_tests`：真实 artifact 的
  descriptor → DMA → buffer → 消费者拷贝 → DMA 写回闭环；
- `dataplane_smoke`：可复现冒烟入口，输出 `transactions.csv`、
  `buffers.csv`、`summary.json`、`final_memory_image.bin`。

单独复现加载—搬运—写回：

```bash
/tmp/t04-build/dataplane_smoke \
  --device-dir /tmp/t04-build/t04_fixtures/single_static/device \
  --out /tmp/t04-output/single_static
```

设备运行目录只需要 `manifest.json` 与 `memory_image.bin`；golden/reference
文件不是设备数据来源，缺失不影响加载与数据面运行。

后续 T05 在本目录的 `dataplane` 之上实现 Command Fetch、GCU/token
scoreboard、资源调度与 completion 写回；数值 kernel（Panel-LU/TRSM/GEMM）
按后续任务实现。T04 不执行命令、不写 completion、不做数值计算。
