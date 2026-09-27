# T04：SystemC Command 数据面设计说明

状态：已实现（代码、测试与集成证据见本文末尾与
[验证计划与结果口径](../verification/验证计划与结果口径.md)）。T04 只实现**数据面**，
不实现命令流执行。

## 1. 范围与边界

实现：

- 从设备运行目录加载 `manifest.json`（schema v2）与 `memory_image.bin`，完成
  结构与交叉校验；
- 统一 byte-addressable 设备内存与 region 注册表，禁止 host 指针作为设备地址；
- 运行时 descriptor 读取（走设备事务、无 cache）与类型化视图；
- 片上数据 buffer（slot + generation、所有权、容量、有效范围、生命周期状态）；
- 单通道 DMA（显式 request/response、ready/valid、延迟、背压、reset）；
- 事务 trace、统计、最终内存快照；
- 双 buffer / ping-pong 访问能力（DMA 装载 B 时消费者可读 A）。

不实现（明确不做，也不得宣称）：

- command fetch / GCU / token scoreboard / completion 写回（T05）；
- Panel-LU、TRSM、GEMM、assemble、solve 数值核（后续任务）；
- DYNAMIC_PANEL 的设备执行；T04 只加载、解析、搬运其数据契约；
- HPU/ATU 动态主元、真实 DDR/AXI PHY、多通道 DMA、descriptor cache；
- 任何数值运算或从 host reference 旁路得到结果。

T04 在加载阶段校验 command 与 completion 初始化记录，但**不执行任何命令**，
不产生 command completion。

## 2. 分层

代码分为两层：

```text
普通 C++ 数据/校验/时序语义层（include/dataplane/*.hpp、src/dataplane/*.cpp）
   ArtifactLoader / DeviceMemory / RegionRegistry / MemoryService /
   DescriptorReader / BufferManager / DmaEngine / TraceRecorder / DataPlane
SystemC 时序驱动层
   DataPlane 是普通 C++ 对象；SystemC 测试用 sc_clock + SC_THREAD 在
   每个 posedge 调用 plane.step()，因此所有周期断言都在 SystemC 调度器
   下产生，但数据面本身不依赖 SystemC 数值/容器。
```

`DataPlane` 汇总所有模块并固定 tick 顺序，保证确定性：

```text
step():  ++cycle;  memory.tick();  buffers.tick();  dma.tick();
```

cycle 约定：`cycle` 从 0 开始；`step()` 先自增再推进模块，因此第 n 次
`step()` 处理的事件记在 cycle = n。所有事件在 `valid && ready` 同时成立的
那一拍之后生效。

延迟归属（防止重复收费）：

| 延迟/开销 | 计入模块 |
| --- | --- |
| DDR 访问服务（startup + beat） | `MemoryService` |
| 片上端口访问延迟 | `BufferManager`（lease.ready_cycle） |
| DMA 启动、beat 间隔 | `DmaEngine`（不再叠加 DDR 延迟） |
| descriptor 解码 | 读取该 descriptor 的内存事务完成后同拍完成，不额外计费 |
| 消费者计算 | 测试消费者自行控制（T04 测试用固定拷贝延迟） |

## 3. ArtifactLoader / DeviceBootConfig

`load_artifact(device_dir)`：读 `manifest.json` 与 manifest 声明的
`memory_image.bin`，返回 `LoadedArtifact { DeviceBootConfig config; image bytes; }`。

校验项（复用 `hw::command` codec，不复制二进制格式定义）：

- `abi.name = command_descriptor`、`abi.version = 2`、little-endian、
  记录大小 32/64/64、`pivot_policy_field`；
- `total_bytes`、`memory_image.size`、文件实际长度三者一致；
- region 表对齐（64B 起点）、u64 加法/乘法溢出、重叠、越界、padding；
- command/descriptor/completion 记录解码、数量与 region 大小一致；
- command→descriptor 类型、descriptor 引用类型、id 范围；
- `FRONT_DESC` 策略与 manifest `pivot.policy`、阈值位型一致；拒绝
  v1（word8 = 0）与未知策略；
- P-vector 沿实际引用检查：REGION_DESC、INT32、容量 = `pivot_dim`、
  初始化内容全为哨兵 `-1`，同节点 FRONT/FACTOR/SOLVE 指向同一条；
- 静态 batch 元数据基本一致性：节点 id 稠密、命令归属、parent 合法、
  `factorization_order` 是 0..N-1 的排列；
- completion 模板与命令一一对应且未执行（status = OK、计数清零）；
- 设备数据格式仅 FP32/INT32。

Loader **不要求** golden/reference 文件存在（设备运行目录可只有
manifest 与 image），不调用 Python reference，不检查 `verification`
声明的文件是否存在；只在 manifest 声明 `device_memory_contains_fp64_reference`
时校验其必须为 `false`。

加载阶段解码的对象只用于校验与区域索引；运行时 descriptor 读取必须重新
经 `DescriptorReader` 的设备事务。

## 4. DeviceMemory / RegionRegistry

- 设备地址 = image 内 u64 字节偏移；region 由 manifest 的 name→{offset,size}
  注册，角色按用途分类。
- `RegionRole`：`Command`、`Descriptor`、`Payload`（只读）；`Completion`、
  `Data`（按用途可写）；`P_vector`（PANEL_LU 输出，T04 只读搬运）。
- 每次访问检查：`address + length` 不溢出；完整落在单一 region 内（不得跨
  region 或进入 padding）；按访问类型对齐；写权限；非法写不改内存。
- 对齐按访问类型：command 32B/记录、descriptor 64B/记录、completion 64B/记录；
  元素访问按元素宽度（FP32/INT32 = 4B）；所有子访问不要求 64B 对齐。
- 零长度访问：不产生实际访存，返回可识别的成功空操作，但仍校验地址位于
  某个 region 内（不得借零长度绕过检查）。
- 运行时模块只拿到 `MemoryService`/`BufferManager` 接口，不能直接改整个 image；
  只有 loader 初始化与最终 snapshot 持有完整字节数组。

## 5. MemoryService

共享内存服务，FIFO 单端口、可配置队列深度；DescriptorReader 与 DMA 共用：

- `submit(request)`：队列满 → 返回背压（调用方重试）；地址/权限非法 →
  接受但立即产生 `AddressFault` 终态响应（有且仅有一个终态响应）；
- 服务时序：`start = max(cycle, port_free_cycle)`；
  `finish = start + startup + ceil(len/beat_bytes) * beat_cycles`；
  写数据在 `finish` 拍可见，读数据在完成后由 `try_pop_response` 取回；
- 零长度：`finish = cycle`、0 字节、不占用端口；
- `tick()` 在 `finish_cycle <= cycle` 时落实写/产出读响应；
- 统计：接受数、字节数、端口忙周期。

## 6. DescriptorReader 与类型化视图

`request(kind, descriptor_id)` 经 `MemoryService` 读 64B descriptor；需要
payload（CONTRIBUTION/DEPENDENCY）或引用 region（FRONT/FACTOR/KERNEL/SOLVE）
时再发起后续读取。每次读取都计入访存与延迟；首版无 cache。

视图（引用与形状校验，不做数值运算）：

- `RegionView`：base/size、rows/cols、row_stride、format、layout、元素宽度；
- `FrontView`：front/p/workspace region、factor 引用、`total_dim`、`pivot_dim`、
  `tile_size`、policy、threshold；拒绝 v1/非法策略/非法阈值；P-vector 为
  INT32 且容量 = `pivot_dim`；
- `FactorView`：L/U/P/workspace region 与形状；
- `ContributionView`：source/target region、stride、payload row/col map 范围；
- `KernelView`：a/b/c front、M/N/K、backend、format、tile、scale 引用；
- `SolveView`：factor、rhs/solution、P-vector、workspace、方向与格式。

P-vector 的阶段语义：**加载阶段**要求全为哨兵 `-1`；**运行时**每次读
descriptor 只校验引用、格式与容量，不重新要求 `-1`（数值输出阶段由 T05
按有效置换校验）。

## 7. BufferManager / Scratchpad

buffer = 片上矩阵数据 buffer，不是命令队列。`BufferHandle = slot_id + generation`。

内部记录：owner、容量、有效数据范围、状态、活动 lease（读写端口占用）。

生命周期：

```text
FREE →(allocate)→ RESERVED →(begin_load)→ LOADING →(finish_load)→ READY
READY →(begin_compute)→ COMPUTING →(end_compute)→ DIRTY
READY/DIRTY →(begin_writeback)→ WRITING_BACK →(finish_writeback)→ READY
READY/DIRTY →(release)→ FREE
```

合法分支允许：READY 只读 lease 后保持 READY；READY 直接 `begin_writeback`
（搬运未修改数据）；DIRTY 释放时丢弃数据（显式 `release`，文档说明）。

拒绝：无效/过期 handle（generation 不匹配）、错误 owner、重复释放、
未 `finish_load` 前读取、DMA/计算未完成就覆盖或释放、超出容量或有效范围。

端口/带宽：`port_count` 限制同时持有的 lease 数；`access_cycles` 给出
`lease.ready_cycle`；同一 slot 冲突访问（写 vs 读/写）返回 `BufferFull`
（内部 reason = "busy"）而不是无约束并发。至少 2 个 slot，容量与 tile_size
分开配置。

## 8. DmaEngine

单通道、一个活动传输、顺序完成。请求/响应字段与语义：

```text
DmaRequest {
  request_id; has_command; command_id; node_id;
  direction;                 // MemoryToBuffer / BufferToMemory
  ddr_region (RegionView); ddr_offset; ddr_row_stride;
  buffer (BufferHandle); buffer_offset; buffer_row_stride;
  rows; cols; element_bytes;
}
DmaResponse {
  request_id; status; reason;
  issue_cycle; accept_cycle; first_beat_cycle; last_beat_cycle; accept→finish;
  read_bytes; write_bytes; memory_service_cycles; buffer_wait_cycles;
}
```

握手：

- `request_ready` 仅在 Idle 且无 pending 响应时为真；只有
  `request_valid && request_ready` 的同一拍接受请求，接受前无任何副作用；
- 接受时一次性校验整个源/目标范围；非法请求产生终态错误响应，
  保证目标无部分写入；合法请求才开始 beat；
- `response_valid` 且 `response_ready=false` 时响应内容保持稳定；
  期间 `request_ready=false`，不会重复读写；
- 每个已接受请求最终恰好一个终态响应（reset 除外，见 §9）；
- 请求背压（忙）与非法请求错误分开：忙时 `request_ready=false`，非法时
  接受并返回错误状态。

传输：按行、按 `beat_bytes` 切 beat；每 beat 一次 DDR 访问 + 一次片上
lease；尾块允许小于 beat；支持非零 offset、源/目标 stride、2D tile。
多 beat 写回**没有**全局原子性：每个 beat 在其内存事务 `finish_cycle`
可见；reset 中断时可能留下已完成的 beat（文档如实说明，测试覆盖）。

## 9. Reset 语义

- `DmaEngine::reset()`：丢弃活动传输与 pending 响应（不再交付），释放
  所有 buffer lease，回到 Idle；旧 `request_id` 的响应不会再出现；
- `MemoryService::reset()`：丢弃在途/排队访问，epoch 自增，旧响应失效；
- `BufferManager::reset()`：所有 slot 回 FREE 并自增 generation，旧 handle
  全部过期；
- `DataPlane::reset()` 依次调用三者；测试确认 reset 后不接收旧响应、
  旧 handle 被拒绝、新请求可正常执行。

## 10. Trace 与统计

- `transactions.csv`：request_id、client、command/node、region、address、
  length、direction、issue/accept/data_complete/response_accept cycle、status、
  实际字节、request_wait、memory_service、response_backpressure；
- `buffers.csv`：cycle、slot、generation、owner、state from/to、reason；
- `summary.json`：配置、artifact 信息、事务计数、字节、等待/背压分类统计；
- `final_memory_image.bin`：最终快照。

分类不重复累计：`request_wait` 只统计 valid 未被 ready 接受的周期；
`memory_service` 只统计内存事务 accept→finish；`response_backpressure`
只统计 response valid 但 `response_ready=false`。统计仅代表架构模型，
不宣称芯片实测性能。

## 11. 失败与状态码

复用 `hw::command::StatusCode`：`Ok`、`BadDescriptor`、`AddressFault`、
`BufferFull`、`Aborted`、`BadCommand`。细分诊断（busy/stale/wrong owner 等）
保留为 `reason` 字符串，不新增/修改 schema 字段。

## 12. 实现落点与运行方法

代码（`graduation-code/systemc/`）：

```text
include/dataplane/artifact_loader.hpp      设备运行目录加载与校验
include/dataplane/device_memory.hpp        DeviceMemory / RegionRegistry / AccessKind
include/dataplane/memory_service.hpp       共享 DDR 服务（FIFO、延迟、背压、epoch）
include/dataplane/descriptor_reader.hpp    DescriptorReader 与六类类型化视图
include/dataplane/buffer_manager.hpp       BufferManager / BufferHandle / lease
include/dataplane/dma_engine.hpp           DmaEngine / DmaRequest / DmaResponse
include/dataplane/region_view.hpp          RegionView
include/dataplane/trace.hpp                TransactionRow / BufferRow / CSV
include/dataplane/dataplane.hpp            DataPlane tick 编排与响应路由
include/dataplane/selftest.hpp             真实 artifact 闭环自检
src/dataplane/*.cpp                        上述模块实现
src/dataplane_smoke.cpp                    CLI 冒烟入口
tests/dataplane_core_tests.cpp             普通 C++ 测试
tests/dataplane_systemc_tests.cpp          SystemC 时序测试（sc_clock）
tests/dataplane_artifact_tests.cpp         真实 artifact 集成测试
tools/t04_fixture_gen.py                   用软件编译器生成设备运行目录
```

运行（构建目录建议放 `/tmp`）：

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build -DBUILD_TESTING=ON
cmake --build /tmp/t04-build -j8
ctest --test-dir /tmp/t04-build --output-on-failure
/tmp/t04-build/dataplane_smoke   --device-dir /tmp/t04-build/t04_fixtures/single_static/device   --out /tmp/t04-output/single_static
```

## 13. 实现补充语义（相对第 2–11 节的澄清）

- **握手消费 valid**：`DmaEngine` 在 `valid && ready` 接受请求的那一拍内部
  消费 valid；调用方在 busy 期间保持 valid 不会导致同一请求被重复接受，
  第二个请求的 `issue_cycle` 在 valid 首次拉高时记录，`request_wait_cycles`
  单独统计。
- **2D buffer 跨度**：DMA 预校验按源/目标 stride 计算几何跨度
  `(rows-1)*stride + row_bytes`；stride 大于行宽时 lease 范围用跨度而不是
  `rows*cols*element_bytes`。
- **零长度边界**：零长度访问优先落在某个 region 的严格内部；若恰好位于
  相邻 region 的边界，则按“前一个 region 的末端”解释；两者都不满足时
  返回 `AddressFault`，不产生访存。
- **所有权移交**：`BufferManager::transfer_owner` 允许 READY/DIRTY 且无活动
  lease 时显式移交，用于 DMA 装载完成 → 消费者处理 → DMA 写回的角色交换。
- **只读 lease 释放**：消费者 `begin_read` 的 lease 必须用 `release_lease`
  释放；写 lease、装载 lease、写回 lease 只能经各自完成路径结束。
- **reset 的 cycle**：`DataPlane::reset()` 按 DMA → buffer → reader → memory
  的顺序清理，并清空 trace；被丢弃的装载/写回 lease 通过 `abort_lease`
  回到原状态（不清理已写入的 slot 字节，状态回退后必须重新装载），
  旧 handle 因 generation 自增而失效，旧响应因 memory epoch 变化而被丢弃。
- **部分完成**：非法请求在首个 beat 之前整体拒绝，目标无部分写入；合法
  传输中途失败或 reset 时，已完成的 beat 保持可见（DDR）或残留在 slot
  存储中（状态回退），与“多 beat 无全局原子性”的说明一致。
- **lease 边界检查顺序**：`lease_read`/`lease_write` 先比较 `offset` 与
  `lease.offset + lease.length`，再计算剩余长度，避免 `end - offset` 无符号
  下溢；越界（含接近 `UINT64_MAX` 的偏移）返回 `AddressFault` 且不改存储。
- **DMA 全范围 checked arithmetic**：`offset + (rows-1)*stride + row_bytes`
  的乘法、加法以及绝对地址计算全部检查 u64 溢出；失败在首个 beat 之前
  返回 `AddressFault`，不得访问指定 region 之外的地址。
- **buffer 有效范围 = 布局跨度**：2D/带 stride 的传输按
  `(rows-1)*stride + row_bytes` 记录有效区间（不是实际搬运字节数），
  因此行间空隙与最后一行都能经正常 lease 接口读取和写回；DMA 响应的
  `bytes` 仍表示实际搬运字节数。
- **reset 清除锁存请求**：`reset()` 清除内部 `request_valid_`，复位前处于
  等待状态的第二个请求不会被复位后的引擎接受，也不会用旧 handle 产生
  错误响应。
- **接受即初始化事务状态**：`accept_request` 在任何校验和提前返回之前把
  字节计数、等待计数、lease 与内存在途标志清零；零长度请求和早期失败
  请求不会继承上一次传输的 `bytes`/`read_bytes`。
- **写入长度一致性**：`MemoryService` 写请求要求
  `write_data.size() == length`，不一致时返回终态 `BadCommand`，不修改内存、
  不占用数据端口。
- **DFT 验证读**：测试/自检为独立预期结果对比而直接读取 `DeviceMemory`
  snapshot；数据面自身只通过 `MemoryService`/`BufferManager` 访问数据。
