# DataPlane 设计说明

本文按“架构 → 交互 → 模块接口 → 实现机制”的顺序解释当前数据面，供对照代码阅读。代码位于 `graduation-code/systemc/include/dataplane/` 和 `src/dataplane/`，命名空间为 `hw::dataplane`。正文描述现有实现；后续共享 PE 的接入边界单列在第 8 节。

任务范围与历史验收见 [T04 任务记录](T04_SystemC_数据面设计.md)；命令调度见 [ControlPlane 设计](ControlPlane_设计.md)；二进制契约见 [Command schema v2](../interface/Command_Schema_v2.md)。本文保留 `DP-I-*` 编号以便已有审查记录定位，但这些编号放在具体机制后，不作为首次阅读入口。

## 1. 数据面解决什么问题

Host 编译器生成 `manifest.json` 和 `memory_image.bin`。数据面负责检查这两份产物、建立设备地址空间，并把矩阵在设备内存与片上 buffer 之间按事务搬运。它向控制面提供三类服务：读取描述符、申请片上访问权限、发起内存或 DMA 事务。

输入是设备镜像、区域表、时序/容量配置及运行时请求；输出是类型化描述符、buffer 数据、事务终态响应和 trace。数据面不决定命令先后、不发布 token、不生成 completion 记录，也不做 LU/TRSM/GEMM 数值计算。completion 的字节最终由本层写入内存，但“何时写哪条 completion”归控制面所有。

这里的三类存储需要分清：

| 名称 | 存放什么 | 谁负责 |
| --- | --- | --- |
| `DeviceMemory` | 整份设备 image，包括命令、描述符、矩阵、P-vector、completion | 区域与权限由本层校验；运行时经 MemoryService 访问 |
| `BufferManager` 的 slot | 已搬到片上的矩阵或暂存数据 | 控制面安排用途，本层管理 owner、状态和 lease |
| 模块内部队列 | 尚未完成的请求与响应 | 各模块自行维护；不是矩阵 buffer |

设备地址始终是 image 内的 u64 字节偏移，不是 Host 指针（`DP-I-1`）。运行时读写经 `MemoryService` / `BufferManager`；`image()/snapshot()/bytes()` 的直接内省仅供加载、自检和测试比对（`DP-I-2`）。

## 2. 总体结构与模块关系

```text
启动路径
manifest + memory_image ──load_artifact──> LoadedArtifact
                                            │ config + image
                                            ▼
                                         DataPlane
运行路径（下列对象均由 DataPlane 持有）
控制面 ──request/poll──> DescriptorReader ──MemoryRequest──┐
控制面 ──DMA job───────> DmaEngine ────────MemoryRequest──┤
控制面取指/写completion ─────────────────────────────────┤
                                                       ▼
                                                  MemoryService
                                                       │ check/read/write
                                                       ▼
                                                  DeviceMemory
                                                       │
                                                  RegionRegistry

DmaEngine <──lease读写──> BufferManager <──lease读写── 执行单元/控制面
各模块 ──事务与状态事件──> TraceRecorder
MemoryService ──MemoryResponse──> DataPlane路由 ──> 原请求客户端
```

箭头表示调用或数据传递，不表示独立物理通道。`MemoryService` 当前只有一个串行服务端口；多个客户端共用这个端口。`DmaEngine` 也只有一个活动传输，双 front 的 DMA 排队由控制面的 `DmaArbiter` 处理。

`DataPlane` 是普通 C++ 组合对象，不是 `sc_module`。SystemC 测试顶层在每个时钟上升沿推进一次；与控制面集成时，唯一时钟推进者是 `CommandDevice::tick()`。模块的 `tick` 不自行等待 SystemC 时间，也不递归推进别的周期。

## 3. 沿一条 LOAD_FRONT 看模块如何交互

以下流程与控制面文档第 3 节使用同一例子。数据面看到的是若干读取/搬运请求，不直接解释 `LOAD_FRONT` opcode。

1. 控制面向 `DescriptorReader::request(Front, id, cycle)` 请求 front 视图。reader 通过 MemoryService 读取 FRONT_DESC、引用记录及必要 payload；队列忙时逐拍重试。加载期的预解码不能替代这次运行时访问。
2. 控制面用 `poll(id, outcome)` 取得经过引用、维度、格式与范围校验的 `FrontView`。`request` 返回 ID 只表示任务建立；`poll` 返回结果后仍要检查 `outcome.status`。
3. front 槽在 `NODE_BEGIN` 时已申请。控制面准备 `DmaRequest`，填入设备 region、buffer handle、owner、offset、行列数与两侧 stride，再交给 DmaArbiter。
4. DmaArbiter 驱动 DmaEngine 的 valid/ready。DMA 接受请求后先验证完整搬运几何，再申请装载 lease，使 slot 从 RESERVED 进入 LOADING。
5. DMA 按行和 beat 向 MemoryService 请求数据。内存响应经 DataPlane 路由回 DMA；DMA 等 lease 的 `ready_cycle` 后写片上字节。DDR 延迟、片上访问延迟和 DMA 自身启动/beat 间隔各只计一次。
6. 最后一拍数据落定后，DMA 调用 `finish_load`，slot 进入 READY，并提供稳定的 `DmaResponse`。响应被接收前不会启动下一次传输。
7. 控制面取得 DMA job 终态，再写 completion；收到 completion 内存写确认后才发布 READY token。本层的“DMA 完成”不等于整条命令已经对后继可见。

反向 STORE 使用 `begin_writeback → lease_read → MemoryService 写 → finish_writeback`。多 beat 写入逐步可见，不提供整次搬运的原子回滚。

## 4. 模块职责与公开接口

接口表列出调用者最需要理解的入口；精确参数以链接中的头文件为准。`false` 的含义依接口而定，不能统一理解为执行失败。

| 模块与代码入口 | 职责/持有状态 | 主要调用者与接口 | 返回与生命周期 |
| --- | --- | --- | --- |
| [ArtifactLoader](../../graduation-code/systemc/include/dataplane/artifact_loader.hpp) · [实现](../../graduation-code/systemc/src/dataplane/artifact_loader.cpp) | 启动校验；生成 boot config 和 image | 启动代码调用 `load_artifact` / `load_artifact_from_bytes` | 同步返回 `LoadedArtifact`；非法产物抛 `ArtifactLoadError`，不计运行周期 |
| [DataPlane](../../graduation-code/systemc/include/dataplane/dataplane.hpp) · [实现](../../graduation-code/systemc/src/dataplane/dataplane.cpp) | 组合模块、维护 cycle、路由响应 | 顶层调用 `step` 或三段 tick；`set_memory_sink` 注册控制面客户端 | 每周期只选一种推进方式；sink 必须在运行前注册 |
| [DeviceMemory / RegionRegistry](../../graduation-code/systemc/include/dataplane/device_memory.hpp) · [实现](../../graduation-code/systemc/src/dataplane/device_memory.cpp) | 持有 image、区域类型/权限；安全字节访问 | MemoryService 调用 `check/read/write` | 同步校验，不建模时延；失败无数据副作用 |
| [MemoryService](../../graduation-code/systemc/include/dataplane/memory_service.hpp) · [实现](../../graduation-code/systemc/src/dataplane/memory_service.cpp) | 单端口排队、完成时落实访存 | reader/DMA/控制面调用 `submit(req, cycle)`；顶层调用 `tick/try_pop_response` | `false` 为队列背压；`true` 包括已接受的非法请求，终态看 response.status |
| [DescriptorReader](../../graduation-code/systemc/include/dataplane/descriptor_reader.hpp) · [实现](../../graduation-code/systemc/src/dataplane/descriptor_reader.cpp) | 递归读取引用与 payload，构建 typed view | 控制面调用 `request/pending/poll`；DataPlane 调用 `tick/deliver_memory_response` | 异步无 cache；响应投递只入队，poll 取走终态 |
| [BufferManager](../../graduation-code/systemc/include/dataplane/buffer_manager.hpp) · [实现](../../graduation-code/systemc/src/dataplane/buffer_manager.cpp) | 片上存储、owner/generation、有效范围、端口与 lease | 控制面/DMA/执行单元使用下表中的 API | 获取成功不表示访问延迟已结束；访问须等待 lease.ready_cycle |
| [DmaEngine](../../graduation-code/systemc/include/dataplane/dma_engine.hpp) · [实现](../../graduation-code/systemc/src/dataplane/dma_engine.cpp) | 单活动事务、2D 搬运、beat 状态机 | DmaArbiter 或测试驱动 valid/ready；DataPlane 每拍 tick | 请求接受与传输成功分开；响应保持到消费；首 beat 前的暂时竞争可 retryable |
| [TraceRecorder](../../graduation-code/systemc/include/dataplane/trace.hpp) · [实现](../../graduation-code/systemc/src/dataplane/trace.cpp) | 收集事务与 buffer 迁移，写 CSV | 各模块记事件，顶层输出报告 | 观察路径，不负责调度与推进 |

### 4.1 BufferManager 的调用配对

| 目的 | 入口 → 结束接口 | 结果/约束 |
| --- | --- | --- |
| 申请与释放槽 | `allocate → release` | 得到带 generation 的 BufferHandle；有活动 lease 时不能释放 |
| DDR 装载到槽 | `begin_load → lease_write → finish_load` | 建立有效数据范围，转 READY |
| 计算读 | `begin_read → lease_read → release_lease` | 不改变 READY/DIRTY 内容状态 |
| 计算写 | `begin_write → lease_read/lease_write → end_write` | 写结束转 DIRTY |
| 写回 DDR | `begin_writeback → lease_read → finish_writeback` | 写回结束转 READY |
| 更换使用者/重装载 | `transfer_owner` / `rearm` | 都受状态与 lease 约束；rearm 保留句柄 generation |
| 异常中止 | `abort_lease` / `reset` | 前者结束本次租约；后者使全部旧句柄失效 |

`BufferResult.transient=true` 只表示冲突或端口暂忙，可释放已获取资源后重试；错 owner、stale handle、容量不足不能靠无限重试解决。slot 是长期存储分配；lease 是一次有范围、有权限、有时延的访问授权，两者不能互相替代。

### 4.2 顶层最小使用顺序

```cpp
LoadedArtifact artifact = load_artifact(device_dir);
DataPlane plane(std::move(artifact), config);
// 与控制面集成时注册 CommandFetch / CompletionWriter / ExecUnit sink。
// 每个时钟沿：plane.step();  // 仅独立数据面驱动
// 集成顶层改用 begin_tick / tick_memory_and_route / tick_clients 三段。
```

`run_until` 与 `run_dma_request` 内部会推进周期，只用于有界测试驱动；运行时模块不能调用它们来等待另一个模块完成。

## 5. 内部实现：时序、校验与状态机

阅读顺序建议与调用链一致：先看顶层相位，再看 loader 与地址校验，最后看异步读取、buffer 与 DMA。以下保留完整契约细节供实现和审查查阅。

### 5.1 一个周期如何推进

`DataPlane` 提供可拆分的相位，供控制面按固定顺序编排；`step()` 是其组合
（T04 测试沿用，语义不变）：

```text
begin_tick()            : ++cycle
tick_memory_and_route() : MemoryService.tick(cycle) → 按 client 路由响应
   ├ DescriptorReader → reader_.deliver_memory_response
   ├ Dma              → dma_.deliver_memory_response
   ├ CommandFetch / CompletionWriter / ExecUnit → 已注册 sink（控制面）
   └ Test / Other     → 丢弃
tick_clients()          : reader_.tick(cycle); dma_.tick(cycle)
step()                  : begin_tick + tick_memory_and_route + tick_clients
```

- `DP-I-3` cycle 从 0 起；`step()`/`begin_tick()` **先自增后推进**，因此第 n 次推进
  处理的事件记为 cycle = n；`issue/accept/data_complete/response_accept` 均在
  `valid && ready` 成立的那一拍之后生效。
- `DP-I-4` 延迟归属唯一，禁止重复计费：

| 延迟 | 计入模块 | 代码 |
| --- | --- | --- |
| DDR 服务（startup + beat） | `MemoryService` | `MemoryService::submit` |
| 片上端口访问 | `BufferManager`（`lease.ready_cycle`） | `BufferManager::make_lease` |
| DMA 启动、beat 间隔 | `DmaEngine`（不再叠加内存延迟） | `DmaEngine::accept_request` / `start_next_beat` |
| descriptor 解码 | 读该 descriptor 的内存事务完成拍，不额外计费 | `DescriptorReader::on_record` |

- `DP-I-5` 共享内存服务是单端口 FIFO：`DescriptorReader` 与 `DmaEngine`（以及控制面
  的 fetch/completion/exec 客户端）按提交顺序串行化，队列满即背压。
  验证：`test_memory_service`、`test_no_lost_or_duplicated_transfers`。

### 5.2 加载期校验

`load_artifact(device_dir)` → `LoadedArtifact { DeviceBootConfig; image }`，
或 `load_artifact_from_bytes(manifest_text, image, label)`（测试注入用）。

必须校验（全部在 `load_artifact_from_bytes` 主流程 + 匿名区 helper）：

1. `abi`：name/version=2/little-endian/`address_unit`/记录 32/64/64/
   `pivot_policy_field`。
2. `total_bytes`、`memory_image.size`、文件实际长度三者一致。
3. region 表：`compiler.alignment ≥ 64` 且为对齐基数、region 起点已对齐、
   u64 加/乘无溢出、互不重叠（零长度区域不参与重叠判定）、不越 `total_bytes`。
4. command/descriptor/completion 记录解码；region 大小与记录数严格相等；
   `validate_command_batch` 复用 codec。codec 抛出的 `std::runtime_error`
   一律包装为 `ArtifactLoadError`（调用方只需处理一种异常）。
5. descriptor 引用类型图（`validate_descriptor_references`）与
   command→descriptor 类型（`allowed_descriptor_types`）。
6. `FRONT_DESC`：策略 = manifest `pivot.policy`、阈值位 = `threshold_bits_fp32`、
   `threshold_ratio` 与 binary32 位型一致、v1（word8=0）与未知策略拒绝。
7. P-vector：沿实际引用取 REGION_DESC、格式 INT32、容量 == `pivot_dim`、
   `byte_size == pivot_dim*4`、内容全为哨兵 `-1`；
   同节点 FRONT/FACTOR/SOLVE 引用同一条 P-vector。
8. 节点准入契约：`NODE_BEGIN` 的 wait list 必须包含其每个子节点的
   `NODE_COMMIT` token（`validate_nodes` 内，与 Python
   `manifest.py::_validate_admission_dependencies` 同源）。
9. 静态 batch 一致性：节点稠密、`command_ids` 与命令归属互斥且覆盖全部命令、
   每节点含必需 opcode、`parent` 在 `[-1,N)`、`factorization_order` 是排列、
   signal token 稠密。
10. completion 模板与命令一一对应、status=OK、计数器全零。
11. 设备数据精度：所有 REGION_DESC ∈ {FP32, INT32}，layout RowMajor；
    `compiler.device_format=FP32`、`global_bfp=false`。
12. `verification.device_memory_contains_fp64_reference` 若存在必须为 false。

- `DP-I-6` 设备运行目录可只有 `manifest.json + memory_image.bin`；loader 不
  `stat` 任何 golden/reference 文件，不调用 Python reference。
  验证：`test_load_valid_device_dirs`（文件数断言 == 2）。
- `DP-I-7` 加载期解码结果只用于校验与区域索引；运行时 descriptor 必须由
  `DescriptorReader` 重新经设备事务读取。
  验证：`controlplane_core_tests.cpp::test_single_node_flow` 等用例的
  `descriptor_reads > 0`（见 `dataplane_artifact_tests.cpp`）。

### 5.3 设备地址与权限

`RegionRole ∈ {Command, Descriptor, Payload, Completion, Data, DataInput, PVector}`
（`classify_region_role`：`rhs_data`/`permutation_data` 为只读输入，
`*.p_vector` 为 PVector 角色）。`AccessKind ∈ {Command, Descriptor, Payload,
Completion, Fp32Data, Int32Data, ByteData}`。

`DeviceMemory::check(address, length, kind, write)` 的判定次序（每拍每次访问都跑）：

1. `length == 0`：不产生任何访存、返回可识别的成功空操作，但**必须**先证明地址合法：
   第一趟取“严格落在某 region 内部”（`offset ≤ addr < offset+size`），
   无则第二趟取“恰好等于某 region 末端”。两趟都不成立即 `AddressFault`。
   角色与写权限同样要检查（零长度不得成为绕过检查的后门）。
2. `address + length` u64 溢出 → `AddressFault`。
3. 整个区间必须完整落在**单一** region 内：不得跨 region，也不得落入 padding。
4. 按访问类型对齐：command 32B/记录、descriptor 64B 且恰好一条记录、
   completion 64B 倍数、payload/FP32/INT32 按 4B 元素；`ByteData` 只要求区间合法。
   **region 的 64B 起点对齐不适用于任何子访问。**
5. 角色允许该访问类型；写还需 `write_allowed(role)`（Command/Descriptor/Payload
   永远只读，DMA 数据搬运不可能误写控制区）。

`read/write` 先 `check`，失败即不改内存、out 清空；`write` 校验通过才 `copy`。

- `DP-I-8` 非法写不改变内存；非法读不返回数据。
  验证：`test_device_memory`（写 command 区失败后 snapshot 相等）。
- `DP-I-9` 零长度语义如上，且不绕过区域检查。
  验证：`test_device_memory`、`test_zero_size_region_memory`。
- `DP-I-10` 子访问对齐按元素宽度，不按 region 起点。
  验证：`test_device_memory`（`ByteData` 非 64B 对齐合法、`Fp32Data` 非 4B 对齐非法）。

### 5.4 内存排队与完成

`submit(request, issue_cycle)` 的三类出口（互斥）：

| 情形 | 行为 | 副作用 |
| --- | --- | --- |
| 地址/权限/对齐非法 | 接受并**立即**产出一条终态 `AddressFault` | 不占端口、不改内存 |
| 写请求 `write_data.size() != length` | 接受并立即产出终态 `BadCommand` | 不占端口、不改内存 |
| `length == 0`（合法） | 立即产出成功空操作，`bytes=0`、`service_cycles=0` | 不占端口 |
| 在途数 ≥ `max_inflight` | 返回 `false`（正常背压） | 无（不得有半副作用） |
| 正常请求 | 返回 `true`，`start = max(cycle, port_free)`、`finish = start + startup + ceil(len/beat_bytes)*beat_cycles` | 占端口 |

`tick(cycle)` 在 `finish_cycle <= cycle` 时落实写、捕获读数据并产出响应；
读数据只在完成后经 `try_pop_response` 取回 ⇒ **写入可见时刻 = 该事务 finish_cycle**。
`reset()` 清空在途与响应、`++epoch`：被丢弃的请求不再有响应，携带旧 epoch 的响应
在路由处被丢弃。

- `DP-I-11` 每个被接受的请求恰好一个终态响应；reset 例外（丢弃）。
  验证：`test_memory_service`、`dataplane_systemc_tests.cpp::test_no_lost_or_duplicated_transfers`。
- `DP-I-12` 背压与非法请求错误分离：`false`=重试，终态错误=响应。
  验证：`test_memory_service`（队列满返回 false + 非法写立即终态）。
- `DP-I-13` 声明写长度与 payload 必须严格相等。
  验证：`test_memory_service`（`BadCommand`，端口占用不变）。

### 5.5 描述符解析与 typed view

异步、无 cache；`request(kind, descriptor_id, cycle)` → `poll(request_id, outcome)`，
`tick()` 负责提交背压重试，`deliver_memory_response()` 只入队。每个 descriptor 读取
与其 payload 读取都是真实内存事务并计入 trace。

视图与校验（`DescriptorReader::finalize`）：

| 视图 | 引用解析 | 关键校验 |
| --- | --- | --- |
| `RegionView` | REGION_DESC 本体 | 设备格式、RowMajor、`MemoryService::check` 合法、必须能在 manifest region 表按 offset+size 命中（不得指向控制区） |
| `FrontView` | front/p/workspace region + factor 记录（+contribution 记录） | `0<pivot_dim≤total_dim`、`tile_size>0`、front FP32 且 ≥total_dim、p INT32 且容量==pivot_dim 且 `byte_size==pivot_dim*4`、workspace ≥2×pivot_dim、factor 的 p 引用与维度一致 |
| `FactorView` | l/u/p(+workspace) | L ≥total×pivot、U ≥pivot×total、p 容量==pivot_dim、FP32/INT32 |
| `ContributionView` | source/target + payload map | payload 长度==4×(row+col)、map 值域、stride 落在 region row_stride 与行宽之间、格式 FP32 |
| `KernelView` | a/b/c 的 front 记录 + 各自 front region（(+scale)） | m/n/k>0、tile>0、backend 已知、format FP32、a≥M×K、b≥K×N、c≥M×N、scale 为 SCALE_DESC |
| `SolveView` | factor 记录 + rhs/solution/p/workspace | `rhs_count>0`、方向已知、format FP32、p == factor.p、p 容量==factor.pivot_dim、rhs/solution ≥rhs_count×total_dim、workspace ≥2×pivot_dim |
| `DependencyView` | payload token | 数量不超过 `max_wait_tokens` 上限、无重复、token 在 `token_count` 内 |

拒绝项：v1 `FRONT_DESC`、未知策略、非法阈值、用 region 名字猜类型（类型只来自
descriptor 字段）、以及“运行时重读 descriptor 时再次要求 P-vector 全为 `-1`”。

- `DP-I-14` P-vector 阶段语义：加载期哨兵 `-1`（第 5.2 节第 7 条），运行期只校验引用/
  格式/容量。验证：`controlplane_core_tests.cpp::test_single_node_flow` 在
  PANEL_LU 之后仍可读同批 descriptor；`test_abort_during_pvector_write` 中
  identity 写入后解析仍成功。
- `DP-I-15` STATIC 与 DYNAMIC_PANEL 的原样保留：reader 只搬运语义，
  `DYNAMIC_PANEL` 的执行拒绝属于控制面。
  验证：`controlplane_artifact_tests.cpp`（dynamic_panel 用例）。
- `DP-I-16` 不做数值运算，只解析与校验。

### 5.6 Buffer 状态与访问

这里的 buffer 是**片上矩阵数据 buffer**，不是命令队列。句柄 =
`BufferHandle{slot_id, generation}`。每 slot 记录 owner、容量、有效数据范围
（`valid_offset/valid_bytes`）、状态、活动 lease。

```text
FREE --allocate--> RESERVED --begin_load--> LOADING --finish_load--> READY
READY --begin_write--> COMPUTING --end_write--> DIRTY
READY|DIRTY --begin_writeback--> WRITING_BACK --finish_writeback--> READY
READY|DIRTY|RESERVED --release--> FREE      READY|DIRTY --rearm--> RESERVED
任意持有 lease 期间：冲突访问 → BufferFull(reason)；reset() → 全部 FREE 且 generation++
```

合法分支不强制经过全部状态：只读计算（`begin_read`/`release_lease`）保持
READY/DIRTY；`begin_writeback` 允许 READY（搬运未修改数据）；`release` 允许
DIRTY（显式丢弃数据）。

关键语义

- **原子获取 + 回滚**：`make_lease` 依次校验 handle 有效性（slot 界、generation、
  非 Free、owner）、容量（`offset > end || len > end - offset` 的**两比较式**，
  不做可能下溢的减法）、同 slot 冲突、全局端口数。
- **有效范围 = 布局跨度**：2D/带 stride 的搬运以 `(rows-1)*stride + row_bytes`
  记为 `valid_bytes`，因此行间空隙与最后一行都可用正常 lease 接口读写；
  实际搬运字节数由调用方/DMA 响应单独记录。
- **lease 数据面操作**：`lease_write/lease_read` 要求 `cycle >= lease.ready_cycle`
  （片上端口延迟唯一计入点），越界返回 `AddressFault` 且不改存储。
- **所有权移交**：`transfer_owner` 仅在 READY/DIRTY 且无活动 lease 时允许；
  DMA 装载 → 消费者处理 → DMA 写回的角色交换必须显式移交。
- **暂时 vs 永久**：`BufferResult.transient` 只在“冲突访问/端口满”时为真；
  stale handle、错 owner、状态不符、容量不足均为永久错误。
- `rearm` 保留 generation（不清 slot 存储，只回 RESERVED 并清有效范围），
  `release` 允许 RESERVED（回滚路径必须能释放刚分配的槽）。

不变量

- `DP-I-17` 拒绝：无效/过期 handle、错 owner、重复释放、未装载就读、
  DMA/计算未完成就覆盖或释放、超容量或超有效范围。
  验证：`test_buffer_manager`、`dataplane_systemc_tests.cpp::test_buffer_resource_errors`。
- `DP-I-18` 同 slot 读写冲突不得“无约束并发”：要么等待（stall）要么明确错误。
  验证：`test_buffer_manager`（`begin_write` 与读 lease 冲突）、
  `test_ping_pong_three_blocks`（不同 slot 在端口允许时并行）。
- `DP-I-19` slot 数可配且 ≥2；容量与 `tile_size` 解耦（后者只存在于
  FRONT_DESC 语义中）。
- `DP-I-20` reset 后旧 handle 一律失效（generation 自增）。
  验证：`test_buffer_manager`、`dataplane_systemc_tests.cpp::test_reset_interruption`。

### 5.7 DMA 握手与搬运

单通道、一个活动传输、顺序完成。显式 `DmaRequest`/`DmaResponse` + valid/ready。

握手

- `request_ready() == (phase==Idle && !response_valid_)`；仅
  `request_valid && request_ready` 的那一拍接受。
- `accept_request()` **接受即消费 valid**（内部清零），调用方保持 valid 拉高
  不会重复接受同一请求；`issue_cycle` 在 valid 首次拉高当拍记录，
  `request_wait_cycles = accept - issue` 单独统计。
- 接受前完成**全量几何与权限预校验**（`prepare_transfer`）：
  `multiply_checked/add_checked/row_range_overflow` 检查
  `offset + (rows-1)*stride + row_bytes` 的每步与 `offset` 自身相加，
  逐行绝对地址同样走 checked arithmetic，并按 `ddr_offset ≤ region.byte_size`
  等条件确认全部访问落在指定 region 内。任一失败 ⇒ 接受前即终态
  `AddressFault`，**不发任何 beat、不动 buffer 状态**。
- 接受后才 `begin_load/begin_writeback`（owner 来自 `DmaRequest.owner`）；
  其失败若为 `transient` 则响应 `retryable=true`。
- 响应 `response_valid && !response_ready` 期间内容稳定、不重复读写；
  只有 `response_ready` 才消费，消费时写 trace 行（含 `response_backpressure_cycles`）。

传输

- 按行、按 `beat_bytes` 切 beat；`beat_bytes` 必须是 `element_bytes` 的整数倍；
  支持非零 offset、源/目标独立 stride、每行尾块；2D 的 buffer 侧 lease 长度用
  布局跨度。
- 每 beat：一次 DDR 访问（延迟归 `MemoryService`）+ 一次片上 lease 读写
  （延迟归 `BufferManager`）；DMA 只计 `startup_cycles`/`beat_interval_cycles`。
- 写入可见时刻：DDR 侧为该内存事务 `finish_cycle`，片上侧为 `lease_write` 当拍。
  **多 beat 不具备全局原子性**；中途失败/reset 时已完成 beat 保持可见
  （slot 侧回退状态但不清存储，必须重新装载）。
- 状态计数器在 `accept_request` 开头、任何校验与提前返回之前全部清零，
  因此零长度与早期失败不会沿用上一次传输的 `bytes/read_bytes`。
- 零长度（`rows==0 || cols==0`）：仍要求句柄合法、offset 在 region 内，
  返回成功空操作、`bytes=0`、无内存流量、slot 进入 READY 且 `valid_bytes=0`。

reset：丢弃活动传输与 pending 响应（不交付）、`abort_lease` 回退 buffer 状态、
清除 `request_valid_`（复位前锁存的等待请求不会被误接受）、计数器清零。

不变量

- `DP-I-21` 非法请求无部分写入；合法请求的 beat 顺序与几何完全由请求决定。
  验证：`test_illegal_request_no_partial_write`、`test_dma_address_overflow`。
- `DP-I-22` 一请求一终态响应；reset 不交付旧响应、不产生“补发”响应。
  验证：`test_reset_drops_waiting_request`、`test_reset_interruption`。
- `DP-I-23` 正常背压与非法错误分开：忙时 `request_ready=false`，
  非法/资源永久错误走终态响应。
  验证：`test_request_backpressure`、`test_buffer_resource_errors`。
- `DP-I-24` `retryable` 只在首 beat 前为真（`bytes_done_==0` 且
  `BufferResult.transient`）；控制面据此安全重入队，不重放已写数据。
  验证：`controlplane_core_tests.cpp::test_concurrent_assembly_dma_retry`
  （`dma_retry_count()>0`）、`test_dma_port_contention_retry`。
- `DP-I-25` FP32/INT32 位模式逐字节保持，无隐式数值转换。
  验证：`test_bit_patterns_preserved`。

T07 真实 Panel 引擎对数据面的使用（仅授权接口）：

- 只通过 `BufferManager::lease_read/lease_write` 读写 front buffer；lease
  由 T05 在派发时原子获取，命令终结才释放，引擎不重复申请整 front lease。
- `lease.ready_cycle`、owner、handle generation、范围检查全部生效；端口
  占用（port_count/活动 lease）表示“能否访问”，引擎每拍字节预算
  （`PanelLuConfig::load/writeback_bytes_per_cycle`）表示“带宽”，二者不
  重复计费。
- `FactorStoreUnit` 的 L/U 持久化通过 `MemoryService` 写入（`Fp32Data`），
  与 P-vector 的 `Int32Data` 写共用 ExecUnit 客户端；响应按 request_id 路由，
  非 P-vector 响应对所有 unit 广播、由请求方消费。
- 扫描与 PE 装载共享同一读预算（有界轮转仲裁）；`stall_scan_port` 与
  `panel_tasks.csv` 使竞争可观察。

## 6. 错误、reset 与资源生命周期

### 6.1 错误分类

复用 `hw::command::StatusCode`：`OK / BAD_COMMAND / BAD_DESCRIPTOR /
ADDRESS_FAULT / BUFFER_FULL / DEPENDENCY_FAILED / POLICY_NOT_SUPPORTED /
STATIC_PIVOT_FAILURE / PIVOT_UNSTABLE / NUMERIC_OVERFLOW /
QUANTIZATION_SATURATION / PRECISION_RETRY /
NON_FINITE_VALUE / TIMEOUT / ABORTED`。数据面只产生
`OK/ADDRESS_FAULT/BAD_DESCRIPTOR/BAD_COMMAND/BUFFER_FULL`；细分诊断（stale、
wrong owner、ports busy、lease not ready、cross-region…）放在 `reason` 字符串，
不新增/修改 schema。

- `DP-I-29` 不通过放宽校验来掩盖问题；校验强度只增不减。

### 6.2 全局 reset

| 对象 | reset 行为 |
| --- | --- |
| `MemoryService` | 清在途/响应，`epoch++`；旧 epoch 响应被路由丢弃 |
| `BufferManager` | 全 slot → FREE，`generation++`，清 lease |
| `DescriptorReader` | 清任务与映射（未完成任务不再产出 outcome） |
| `DmaEngine` | 丢活动传输与响应、`abort_lease`、清 `request_valid_` 与全部计数器 |
| `DataPlane` | 依次 dma→buffer→reader→memory 复位，清 trace，注销 sink |
| 控制面 | 见 `ControlPlane_设计` 第 6 节（必须重新 `start()`） |

- `DP-I-26` reset 后不接收旧响应、不误用旧 handle；所有等待有界（测试侧统一
  `run_until(..., max_cycles)` 且 SystemC 用例每拍一次 `step()`）。

## 7. 配置、观测与验证

### 7.1 配置入口

```cpp
MemoryServiceConfig { startup_cycles, beat_bytes, beat_cycles, max_inflight }
BufferConfig { slot_count(≥2), capacity_bytes(每 slot), port_count, access_cycles }
DmaConfig { beat_bytes(须为 element_bytes 整数倍), startup_cycles, beat_interval_cycles }
DataPlaneConfig { MemoryServiceConfig memory; BufferConfig buffers; DmaConfig dma; }
```

### 7.2 Trace 与统计

`TraceRecorder` 两类行：

- `transactions.csv`：`layer(descriptor|memory|dma|command_fetch)`、
  `client`、`request_id`、可选 `command_id/node_id`、`region`、`address`、`length`、
  `write`、`status`、`reason`、`issue/accept/data/response_cycle`、`bytes`、
  `request_wait_cycles`、`memory_service_cycles`、`response_backpressure_cycles`。
- `buffers.csv`：`cycle`、`slot`、`generation`、`owner`、`from_state`、`to_state`、
  `reason`（状态迁移与拒绝都记账）。

另有 `summary.json` 与 `final_memory_image.bin`（由 selftest/smoke 写出）。

- `DP-I-27` 三类等待互不重叠：请求等待（valid 未被 ready 接受）、内存服务
  （accept→finish）、响应背压（response_valid 且 !ready）。
  验证：`test_request_backpressure`、`test_response_backpressure_stable`。
- `DP-I-28` 统计仅代表架构模型，不构成芯片实测性能宣称。

### 7.3 运行与输出

```bash
cmake -S graduation-code/systemc -B /tmp/t04-build -DBUILD_TESTING=ON
cmake --build /tmp/t04-build -j8
ctest --test-dir /tmp/t04-build --output-on-failure          # 含 dataplane_* 5 项
/tmp/t04-build/dataplane_smoke --device-dir <DIR> --out <DIR>
```

`<DIR>` 产物：`transactions.csv`、`buffers.csv`、`summary.json`、
`final_memory_image.bin`；`--source-region/--target-region/--buffer-capacity/
--buffer-slots/--max-cycles` 可覆盖。`run_dataplane_selftest()` 返回
`DataPlaneSelftestReport`（含 `target_verified / non_target_unchanged /
control_regions_unchanged` 与 `writeback_note` 免责声明）。

### 7.4 测试驱动

- 普通 C++ 测试：`tests/dataplane_core_tests.cpp`（`load_artifact_from_bytes`
  变异注入）。
- SystemC 时序测试：`tests/dataplane_systemc_tests.cpp`（sc_clock 每拍一次
  `step()`，所有等待带 `max_cycles` 上界）。
- 真实 artifact：`tests/dataplane_artifact_tests.cpp` +
  `tools/t04_fixture_gen.py`（设备目录仅 manifest + image）。
- 期望值必须由测试侧独立计算（`expected_tile/expected_l/…`、快照对比、
  周期公式），禁止从被测实现回抄。

## 8. 与共享多模式 PE 的接入边界

T06 的 `panel_lu/TileStore` 是独立原型存储，尚未通过本层的真实 BufferManager/MemoryService 搬运数据。复用 `BufferHandle` 类型不代表两个存储已经连通；原型 scan/load/writeback 的预算也不能直接算作当前 DataPlane 的实测占用。

T07 接入时，PanelController 应从控制面授予的 front lease 读取装配后的真实 front：先扫描整个 n×n front 的非有限值与 maxabs，再让 tile task 仅更新 A11。16×16 是计算 tile，原有 front slot 仍是存储分配单位，不应直接把“五个 buffer slot”改成“五个 tile”。扫描、tile 装载、PE 写回必须映射到声明的端口与带宽；目标 tile 写入结束后才能解除内部依赖。完整 identity P-vector 写入确认后，再把 command 终态交回控制面。

具体共享 PE 契约见 [计算基线](../architecture/多模式PE与16x16分块计算基线.md)与 [Panel-LU 设计](../architecture/Panel-LU架构设计.md)。接入不得绕过本层的 owner、范围、时延与 reset 语义。

### 8.1 数据面扩展位置

| 需求 | 做法 |
| --- | --- |
| 新内存客户端（控制面/未来的 prefetch） | 扩展 `MemoryClient` + `memory_client_name`，在 `dataplane.cpp:tick_memory_and_route` 路由，或注册 `set_memory_sink`；不要复用 `Test/Other`（会被丢弃） |
| 新访问类型 | 扩展 `AccessKind` 并在 `role_allowed/对齐分支` 加规则；`DataInput/PVector` 角色已预留 |
| 新 buffer 状态迁移 | 改 `make_lease` 前置校验 + 新 `begin_/finish_` 对；保持“原子获取 + 回滚” |
| 多通道 DMA | 复制 `DmaEngine` 实例并由调用侧仲裁；`MemoryService` 的 FIFO 与 `max_inflight` 已支持并发提交 |
| descriptor cache | 在 `DescriptorReader::finalize` 之上加只读缓存层，但必须保留“首次读取计入访存”的测试口径 |
| 新 trace 字段 | `TransactionRow`/`BufferRow` 加字段 + `write_csv` 表头；控制面自有 CSV 在 `controlplane` 侧扩展 |

## 附录 A：不变量与测试索引

| ID | 不变量（摘要） | 验证 |
| --- | --- | --- |
| DP-I-1 | 地址是 u64 image 偏移，无 host 指针 | `test_device_memory` |
| DP-I-2 | 运行时只经事务接口 | `dataplane_artifact_tests`（`descriptor_reads>0`） |
| DP-I-3 | cycle 约定 | `test_contiguous_and_timing`（周期断言） |
| DP-I-4 | 延迟归属唯一 | `test_contiguous_and_timing`、`test_2d_strides_and_tail` |
| DP-I-5 | 单端口 FIFO + 背压 | `test_memory_service` |
| DP-I-6 | 缺 golden 可加载 | `test_load_valid_device_dirs` |
| DP-I-7 | 加载期解码≠运行时读取 | `test_negative_artifact_cases` + reader 计数 |
| DP-I-8 | 非法写不改内存 | `test_device_memory`、`test_illegal_request_no_partial_write` |
| DP-I-9 | 零长度不绕检查 | `test_zero_size_region_memory`、`test_zero_size_transfer` |
| DP-I-10 | 子访问按元素对齐 | `test_device_memory` |
| DP-I-11 | 一请求一终态响应 | `test_no_lost_or_duplicated_transfers` |
| DP-I-12 | 背压 ≠ 错误 | `test_memory_service` |
| DP-I-13 | 写长度==payload | `test_memory_service` |
| DP-I-14 | P-vector 阶段语义 | `test_abort_during_pvector_write` |
| DP-I-15 | DYNAMIC_PANEL 语义保留 | `controlplane_artifact_tests`（dynamic_panel） |
| DP-I-17 | 句柄/owner/范围拒绝 | `test_buffer_manager`、`test_buffer_resource_errors` |
| DP-I-18 | 冲突不无约束并发 | `test_ping_pong_three_blocks` |
| DP-I-20 | reset 后句柄失效 | `test_reset_interruption` |
| DP-I-21 | 非法请求无部分写 | `test_dma_address_overflow` |
| DP-I-22 | reset 不交付旧响应 | `test_reset_drops_waiting_request` |
| DP-I-24 | retryable 仅首 beat 前 | `test_concurrent_assembly_dma_retry` |
| DP-I-25 | 位模式保持 | `test_bit_patterns_preserved` |
| DP-I-27 | 等待分类不重复 | `test_response_backpressure_stable` |

（DP-I-16/19/23/26/28/29 为设计约束，由代码走查 + `SystemC_模型规范.md`
“不能作出的表述”一节共同保障。）

