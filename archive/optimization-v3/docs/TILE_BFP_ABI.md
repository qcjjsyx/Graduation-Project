# 16×16 Tile BFP DDR 与 ABI 扩展

## 1. 兼容边界

本版本不改变 128 字节 `NodeTask` 的字段、偏移或小端编码，任务描述符仍标记为
ABI v2。变化发生在 `front_e`、`update_e` 和 `node_meta` 指向的 payload 布局。

manifest 增加：

```json
{
  "quantization": {
    "format": "tile_bfp_int32",
    "bfp_tile_size": 16,
    "tile_order": "row_major",
    "mantissa_dtype": "int32",
    "exponent_dtype": "int16"
  }
}
```

SystemC 要求 manifest 与 `sim_config.json` 的 `bfp_tile_size` 完全一致。不一致时在
启动阶段拒绝制品，不能静默按另一种布局解释。

## 2. Tile 编号

对 `rows×cols` 矩阵，tile 数量为：

```text
tile_rows = ceil(rows / 16)
tile_cols = ceil(cols / 16)
tile_count = tile_rows × tile_cols
tile_id(row,col) = floor(row/16) × tile_cols + floor(col/16)
```

边缘 tile 可以小于 16×16。mantissa 仍按完整矩阵 row-major 连续存储；只有 exponent
按 tile row-major 存储。

## 3. 输入 front

对维度为 `total_dim×total_dim` 的 local front：

```text
front_q:
  int32 q[total_dim × total_dim]

front_e:
  int16 e[ceil(total_dim/16) × ceil(total_dim/16)]

value(row,col) = q[row,col] × 2^e[tile_id(row,col)]
```

`front_e.size` 必须严格等于 tile 数量乘 2 字节。

## 4. Child update

令 `update_dim=total_dim-pivot_dim`：

```text
update_q:
  int32 q[update_dim × update_dim]

update_e:
  int16 e[ceil(update_dim/16) × ceil(update_dim/16)]
```

当 `update_dim=0` 时两个区域均为空。父节点通过 map table 将 child update 坐标映射到
自己的 front；指数查找使用映射前的 child update 坐标。

## 5. U exponent 与 node_meta

`U` 的 mantissa 仍写入 `u_factor`，形状为 `pivot_dim×total_dim`。U exponent 表放入
`node_meta`：

| 字节偏移 | 类型 | 含义 |
|---:|---|---|
| 0 | u8 | factor valid |
| 1 | u8 | factor path flags，见下表 |
| 2 | i16 | 首个 U exponent，标量兼容/快速查看 |
| 4 | i16 | 首个 update exponent |
| 6 | u8 | tile size，tile 模式为 16 |
| 7 | u8 | L 的实际 QF 小数位数 |
| 8 | u32 | U tile exponent 数量 |
| 12 | u32 | update tile exponent 数量 |
| 16 | i16[] | U tile exponent，row-major |

字节 1 的 flags：

| Bit | 名称 | 含义 |
|---:|---|---|
| 0 | `precision_rescued` | 最终使用 FP64 rescue 因子 |
| 1 | `precision_assisted` | 提升 QF 后的定点重算成功 |
| 2 | `fraction_retry_attempted` | 已执行第二次定点分解 |
| 7:3 | reserved | 必须写零 |

`node_meta.size` 为：

```text
max(64, 16 + 2 × U_tile_count)
```

update exponent 的权威副本仍在 `update_e`，`node_meta` 只记录它的数量和首个 exponent。

128 字节 NodeTask 描述符仍是 ABI v2；本节是 v3 SystemC 对 `node_meta` payload 的
不兼容扩展。旧 reader 把字节 1 当布尔值时无法区分定点辅助与完整 rescue，因此不得
用旧 reader 解读 v3 写回镜像。

## 6. 标量兼容模式

软件参数 `--bfp-tile-size 0` 保留 v1 布局：

- `front_e` 为一个 int16；
- 非空 `update_e` 为一个 int16；
- `node_meta` 为 64 字节；
- U/update 使用标量 exponent。

该模式用于回归和 A/B；tile 制品不能配合标量 SystemC 配置运行，反之亦然。

## 7. 二次校验

Python 和 C++ 均检查：

- tile size 只能是 0 或 16；
- exponent 数量与矩阵维度一致；
- `front_e/update_e/node_meta` 尺寸正确；
- 所有区域对齐、界内且不重叠；
- DDR 镜像与独立 `front_q/front_e/map_table` 文件逐字节一致；
- U/update 写回 exponent 数量不能超过规划区域。

因此本扩展没有占用 `NodeTask` 保留字段，也没有通过隐式约定猜测 exponent 数量。
