# SystemC v3：统一 QF26 与分层精度实验

本模型在 v2 的 16×16 Tile BFP 数据通路上增加可配置的 L 精度策略：

```text
统一模式（默认）:
  QF26 定点分解 → 局部因子检查 → 必要时 FP64 rescue

自适应实验:
  QF20 定点分解
    → 仅当局部因子检查失败时用 QF26 完整重算
    → 重算仍失败时 FP64 rescue
```

每个 factor 保存实际 `l_frac_bits`，求解和因子误差检查不再假设所有节点都使用同一
QF。重算、QF26 定点辅助成功、FP64 完整救援分别统计，重算也会在周期模型中增加一套
七类 Panel/TRSM/GEMM 操作。

## 默认值

| 配置 | 默认值 |
|---|---:|
| `bfp_tile_size` | 16 |
| `q_use_bits` | 30 |
| `frac_bits` | 26 |
| `adaptive_frac_retry` | false |
| `retry_frac_bits` | 26 |
| `accumulator_bits` | 64 |
| `workspace_guard_bits` | 20 |
| `fixed_factor_rel_tol` | `2e-7` |
| `fixed_rescue_mode` | `fp64` |

`adaptive_frac_retry=true` 时，`retry_frac_bits` 必须严格大于 `frac_bits` 且不超过 30。

## 构建与验证

```bash
cmake -S . -B build \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2 \
  -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Sanitizer：

```bash
cmake -S . -B /tmp/graduation-v3-precision-asan \
  -DSYSTEMC_HOME=/usr/local/systemc-3.0.2 \
  -DENABLE_SANITIZERS=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/graduation-v3-precision-asan --parallel
ctest --test-dir /tmp/graduation-v3-precision-asan --output-on-failure
```

当前验证：软件 33 项通过、CTest 3/3 通过、ASan/UBSan 3/3 通过。

## 运行

```bash
./build/system_sim \
  --artifact /tmp/mf-v2-tile-576/manifest.json \
  --config config/default.json \
  --mode fixed \
  --out /tmp/v3-f26-576 \
  --seed 1
```

`config/adaptive-f20-f26.json` 用于重现自适应实验。`--mode both` 同时要求 FP64 黄金
通道也通过其主元门槛；576JJ 默认会在该通道报告 numeric failure，所以固定点策略
对比应使用 `--mode fixed`。

## 新报告字段

`summary.json.config`：

- `precision_policy`；
- `frac_bits`；
- `adaptive_frac_retry`；
- `retry_frac_bits`。

`summary.json.stability`：

- `fraction_retry_nodes`：执行第二次定点分解的节点；
- `precision_assisted_nodes`：第二次 QF26 分解成功、未进入 FP64 的节点；
- `precision_rescue_nodes`：最终使用 FP64 因子的节点。

`nodes.csv` 进一步记录每节点的 `l_frac_bits` 和三种布尔状态。`operations.csv` 中，
重算节点包含第二套七类定点操作；`timeline.csv` 使用 `fixed_frac_retry` 事件标记。

内存 payload 与结论见
[`../docs/TILE_BFP_ABI.md`](../docs/TILE_BFP_ABI.md) 和
[`../docs/ADAPTIVE_PRECISION_OPTIMIZATION.md`](../docs/ADAPTIVE_PRECISION_OPTIMIZATION.md)。
