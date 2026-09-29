# T04：SystemC Command 数据面（任务记录）

状态：**已完成并通过审查修复**。设计语义的权威说明见
[DataPlane_设计](DataPlane_设计.md)（分层、模块、状态机、时序与背压、reset、
统计口径、接口与扩展点、不变量↔测试对照表）。本文只保留任务视角：范围确认、
实现清单、测试证据与审查修复史。

## 1. 任务范围确认

实现：从 manifest + `memory_image.bin` 加载真实编译产物，经 descriptor 定位数据，
通过有延迟、背压与容量限制的事务，把数据搬入片上 buffer 并写回。

本任务**不做**（也不得宣称）：命令 fetch/GCU/token 调度与 completion 写回（T05）、
Panel-LU/TRSM/GEMM/solve 数值核、HPU/ATU 动态主元执行、真实 DDR/AXI PHY、
多通道 DMA、descriptor cache、芯片实测性能。加载阶段可校验 command 与
completion 初始化记录，但不等于执行了命令流。

## 2. 实现清单

| 模块 | 文件（`graduation-code/systemc/`） | 权威设计节 |
| --- | --- | --- |
| ArtifactLoader / DeviceBootConfig | `include/dataplane/artifact_loader.hpp`、`src/dataplane/artifact_loader.cpp` | DataPlane §3 |
| DeviceMemory / RegionRegistry | `device_memory.hpp/.cpp` | §4 |
| MemoryService | `memory_service.hpp/.cpp` | §5 |
| DescriptorReader + 6 类 typed view | `descriptor_reader.hpp/.cpp`、`region_view.hpp` | §6 |
| BufferManager / Scratchpad / BufferHandle | `buffer_manager.hpp/.cpp` | §7 |
| DmaEngine / DmaRequest / DmaResponse | `dma_engine.hpp/.cpp` | §8 |
| Trace / DataPlane 编排 / selftest | `trace.*`、`dataplane.*`、`selftest.*` | §9–§10 |
| CLI 冒烟 | `src/dataplane_smoke.cpp` | §17 |
| 测试 | `tests/dataplane_core_tests.cpp`、`tests/dataplane_systemc_tests.cpp`、`tests/dataplane_artifact_tests.cpp` | §12 表 |
| fixture 生成 | `tools/t04_fixture_gen.py` | — |

关键契约决策（详见权威文档对应节）：

- 设备地址 = image 内 u64 偏移；运行时模块不持有整个 image 引用（§1）。
- 加载期解码只用于校验；运行时 descriptor 必须走设备事务、无 cache（§3/§6）。
- P-vector 分阶段语义：加载期哨兵 `-1`，运行期只校验引用/格式/容量（§3/§6）。
- STATIC 与 DYNAMIC_PANEL 均可加载/解析/搬运；执行能力判断留给 T05（§6）。
- `DataPlane` 相位化 tick 与延迟归属表（§2）；三类控制面客户端经 sink 路由（§16）。

## 3. 测试与证据（2026-09）

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build -DBUILD_TESTING=ON
cmake --build /tmp/t04-build -j8
ctest --test-dir /tmp/t04-build --output-on-failure
cmake -S graduation-code/systemc -B /tmp/t04-build-sanitize \
      -DBUILD_TESTING=ON -DENABLE_SANITIZERS=ON
cmake --build /tmp/t04-build-sanitize -j8
ctest --test-dir /tmp/t04-build-sanitize --output-on-failure
```

| 项 | 结果 |
| --- | --- |
| `dataplane_core_tests` | 144 checks，0 failures |
| `dataplane_systemc_tests` | 187 checks，0 failures（sc_clock 驱动，含周期断言） |
| `dataplane_artifact_tests` | 90 checks，0 failures（真实 artifact 闭环） |
| CTest（Release） | 12/12 passed（含 T05 控制面与回归项） |
| CTest（ASan+UBSan） | 12/12 passed |
| 真实 artifact 覆盖 | 单节点、22 节点、2 根森林、DYNAMIC_PANEL、256 阶 |
| 256 阶 smoke | image 1,472,984 B；最大搬运 tile 67×67 = 17,956 B；scratchpad 容量 65,536 B（容量足够，如实报告） |
| 输出 | `transactions.csv`、`buffers.csv`、`summary.json`、`final_memory_image.bin` |

数据面写回内容是**测试数据**，不是求出的 LU 因子或 solution；
`summary.json` 内含该免责声明（`transfer.writeback_note`）。

## 4. 审查修复史

**第一轮（6 项，均已修 + 回归）**

| 问题 | 修复 | 权威不变量 |
| --- | --- | --- |
| buffer lease 边界无符号下溢（`offset>end` 时 `end-offset` 回绕） | 先比 `offset` 与 `lease_end` 再算剩余长度 | `DP-I-17` |
| DMA 地址回绕可逃到别的 region | `multiply_checked/add_checked/row_range_overflow` + 逐行绝对地址 checked | `DP-I-21` |
| 2D 把实际字节数当有效范围 | 有效范围 = 布局跨度；响应 `bytes` 仍为实际字节 | `DP-I-17`/§7 |
| reset 后仍接受复位前锁存的请求 | `reset()` 清 `request_valid_` | `DP-I-22` |
| 零长度/早期失败沿用上批计数 | `accept_request` 开头统一初始化 | `CP-I-20`（`DP` 侧同源） |
| 写请求长度与 payload 不一致 | 提交即校验，不一致终态 `BadCommand`、不占端口 | `DP-I-13` |

**第二轮（控制面审查中发现、属数据面边界）**

| 问题 | 修复 | 权威不变量 |
| --- | --- | --- |
| `release` 拒绝 RESERVED → 控制面回滚路径泄漏槽位并死锁 | `release` 允许 RESERVED（无数据丢失） | `DP-I-17` |
| 端口/冲突竞争无法与永久错误区分 | `BufferResult.transient` + `DmaResponse.retryable`（仅首 beat 前） | `DP-I-24` |
| 装配复用已用槽位需重装载 | 新增 `BufferManager::rearm`（保留 generation） | §7 |
| DMA owner 固定 `"dma"` 与多所有者冲突 | `DmaRequest.owner`（默认 `"dma"`，行为不变） | §8 |
| 单端口装配永久 stall（控制面侧发现） | 端口竞争标记为 transient；配置层面的拒绝见 ControlPlane §5 | `CP-I-13`/`CP-I-6` |

## 5. 保留的回归要求

后续修改数据面必须保持：`dataplane_core/systemc/artifact` 三套测试全绿、
sanitizer 全绿、周期断言（不能只查最终数据）、非法写零副作用、
reset 隔离旧响应/旧句柄、`retryable` 仅在首 beat 前为真。
