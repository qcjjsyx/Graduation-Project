# 08 命令流编译与内存布局

代码：`software/src/scheduler/command_compiler.py`、`src/scheduler/task_queue.py`、
`src/memory/planner.py`、`src/command_codec.py`

## 1. 从符号 IR 到设备命令

编译输入是符号分析（03）与变换（04）的结果：

```text
nodes（node_id, parent_id, node_range, front_indices）
map_tables（每个父节点的子贡献 row_map/col_map）
local_fronts（变换后矩阵 A_hat 的局部贡献，update 块已清零）
permutation(q)、rhs_hat
```

编译输出是一个自包含的 `memory_image.bin` + manifest：所有 command、descriptor、
payload、数据区域都在同一镜像内，按字节偏移寻址（无 Host 指针）。

## 2. 每节点的命令序列与 token DAG

按 `sibling_friendly_order`（子先父后、确定性）逐节点发射：

```text
NODE_BEGIN            → token begin
LOAD_FRONT            waits begin                 → token loaded
ASSEMBLE_EXTEND_ADD   waits 上一次装配 + 子节点 update_stored（无则子 commit）
PANEL_LU              waits 最后一次装配           → token panel
  若 update_dim > 0：
    TRSM_LEFT          waits panel                 → token trsm_left
    TRSM_RIGHT         waits panel                 → token trsm_right
    GEMM_SCHUR         waits trsm_left, trsm_right → token schur
    STORE_FACTOR       waits trsm_left, trsm_right → token factor_stored
    STORE_UPDATE       waits schur                 → token update_stored
    NODE_COMMIT        waits factor_stored, update_stored
  否则：
    STORE_FACTOR       waits panel                 → token factor_stored
    NODE_COMMIT        waits factor_stored
```

求解阶段：

```text
SOLVE_FORWARD  每节点按节点顺序，waits 自己的 commit + 上一个 forward
SOLVE_BACKWARD 逆节点顺序，waits 上一个 backward + 自己的 commit
```

**原理**：

- token = `command_id`（batch 内顺序编号），发射即产生；等待表引用更早的 token；
- 依赖是**数据流**而非全局屏障：`ASSEMBLE_EXTEND_ADD` 只等自己需要的子 update，
  从而允许不同子树并行；
- 子贡献尚未落盘时允许退化等待子 `commit`（保守但安全）；
- `NODE_COMMIT` 是节点的原子提交点：只有 factor/update 全部完成才发布 commit token；
- 失败传播：任一命令失败 ⇒ 不发布 READY token ⇒ 等待它的命令以
  `DEPENDENCY_FAILED` 完成（契约见 schema v2 §2）。

## 3. descriptor 绑定

| 命令 | descriptor | 携带语义 |
| --- | --- | --- |
| `NODE_BEGIN` / `LOAD_FRONT` / `NODE_COMMIT` | 节点 `FRONT_DESC` | 节点资源与 front 定位 |
| `PANEL_LU` | 节点 `FRONT_DESC` | `pivot_policy`(word8)、`pivot_threshold`(word9)、`pivot_dim`、P-vector 引用、front/factor/workspace 引用 |
| `TRSM_*`、`GEMM_SCHUR` | `KERNEL_DESC` | 操作数引用与 M/N/K、backend、tile |
| `ASSEMBLE_EXTEND_ADD` | `CONTRIBUTION_DESC` (payload=row_map‖col_map) | 装配源/目标与映射 |
| `STORE_FACTOR` | `FACTOR_DESC` | L/U/P region、维度、格式 |
| `STORE_UPDATE` | 该节点 `update` 的 `REGION_DESC` | 子 Schur 落盘区域 |
| 求解命令 | `SOLVE_DESC` | RHS、factor、P-vector、solution、workspace |

要点：**策略与阈值只放在 `PANEL_LU` 关联的 `FRONT_DESC`**，设备执行语义以
descriptor 为准；P-vector region 初值为 INT32 哨兵 `-1`，`PANEL_LU` 必须完整改写
（STATIC 写显式 identity）。

## 4. 内存布局：确定性单次放置

### 4.1 规格收集（第一阶段）

为每个对象生成"名字 + 尺寸 + 形状 + 数据格式 + 初始字节"的规格，不涉及地址：

- 数据 region：`front`（FP32，local contribution）、`rhs`、`update`、`factor.l/u`、
  `p_vector`（INT32，初始 -1）、`solve_workspace` 等；
- `descriptor_table`、`command_batch`、`completion_queue`（高对齐）；
- payload：contribution map、dependency token 列表。

### 4.2 统一放置与地址解析（第二阶段）

```text
cursor = 0
for spec in 固定顺序:
    offset = align_up(cursor, 64)
    cursor = offset + size
    → 记录 region offset/size
```

- 固定顺序 + 64B 对齐 ⇒ 同输入同配置字节级可复现；
- 每个加法都做 u64 溢出检查；放置后校验越界与 region 不重叠；
- descriptor 里的 region 引用在放置完成后解析为字节偏移，回填进 64B 记录与
  payload，最后整体拼装成 `memory_image.bin`。

## 5. 编译期校验（不依赖参考求解）

- 结构校验：节点 ID 唯一、无环、front 容量与 `max_front_size`/`max_supernode_size`
  上限、map 覆盖正确、置换合法；
- 契约校验：命令/描述符/完成记录逐字节校验（`validate_command_batch`）：
  - descriptor 类型与引用匹配（含 P-vector 引用必填、策略/阈值合法）；
  - region 对齐/越界/重叠；
  - token 单生产者、等待表引用的 token 必须更早存在、数量不超 `max_wait_tokens`；
- manifest 校验：`abi.version=2`、策略与所有 FRONT_DESC 一致、
  threshold_ratio 与 bits 严格相等、P-vector region 全为哨兵、
  置换数组互为一致组合、reference 文件尺寸与声明一致。

## 6. 完成记录模板

host 为每条命令预生成 64B completion 模板（status `OK`、计数清零），设备执行时
逐槽覆写。失败语义（契约）：数值失败时 `pivot_count` 记**首个失败列号**；
失败命令不发布 READY token。

## 7. 容量与边界

| 上限 | 默认 | 触发行为 |
| --- | --- | --- |
| `max_front_size` | 256 | 编译报错（如实暴露，不静默放宽） |
| `max_supernode_size` | 256 | 同上 |
| `max_wait_tokens` | 256 | 编译报错 |

实测：1024 阶真实矩阵触发 front 264 > 256 的既有上限；显式放宽到 300 后可编译，
数据见 07 §5。

## 8. 验证锚点

- `tests/test_command_compiler.py`：token 依赖、P-vector 哨兵、策略落盘、
  确定性、容量与重复 ID/环拒绝；
- `tests/test_command_codec.py`：v2 记录编解码、负面用例、fixture golden；
- `tests/test_pipeline_e2e.py`：manifest 一致性与负面校验、字节级可复现；
- `systemc/tests/command_codec_tests.cpp`：C++ 端独立复现同一批 fixture。
