# Precision Adapter：局部量化与反量化

状态：当前主线候选模块规范

## 1. 重新定位

旧 QAU 方案试图负责整棵消除树的 front 装配、exponent 对齐和全局再量化。该方向会让 child update、父节点、pivot、TRSM 和写回全部携带定点 scale，控制复杂度过高，不再作为当前主线。

当前模块更准确的名称是 `Precision Adapter`。目录暂时保留 `QAU`，避免立即破坏已有工程路径。

## 2. 当前职责

只服务于一次 `GEMM_SCHUR` 调用：

```text
FP32/FP64 A/B tile
      ↓
quantize
      ↓
INT32 A/B tile
      ↓
INT32 GEMM backend
      ↓
INT32/INT64 C tile
      ↓
dequantize
      ↓
FP32/FP64 Schur tile
```

## 3. 最小功能

- 选择每个输入 tile 的二的幂 scale；
- round/shift/saturate 到 INT32；
- 记录 zero、overflow 和 saturation；
- 调用 `Int32GemmBehavioral` 或未来的 RTL GEMM；
- 对结果进行反量化；
- 记录 quantize/dequantize 周期和 bytes；
- 返回明确的 precision failure。

## 4. 不负责的内容

- 不负责完整 front 的全局 BFP 格式；
- 不负责 child update 的全局 exponent 对齐；
- 不负责 pivot 选择；
- 不负责 P-vector 或行交换；
- 不负责 TRSM 的对角除法；
- 不读取黄金 FP64 front；
- 不通过静默饱和掩盖数值失败。

## 5. SystemC 首版接口

```text
request:
  command_id, descriptor_id, A/B/C tile
  M, N, K, stride
  scale_a, scale_b, rounding_mode

response:
  status, output tile
  scale metadata
  cycles, read_bytes, write_bytes
  overflow_count, saturation_count
```

当前没有 INT32 GEMM RTL，因此首先实现 SystemC 行为/周期版本。未来的 RTL quantizer/dequantizer 只能替换同接口模块，不能改变 command stream 语义。

