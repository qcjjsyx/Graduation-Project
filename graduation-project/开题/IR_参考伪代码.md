# 混合精度迭代改进（IR）与 GMRES‑IR 参考伪代码
版本：v0.1 | 目标：可直接翻译为 C++/CUDA/TPU 调度代码

## 0. 记号与约定
- `A`：原始高精度矩阵（FP64）；`b`：右端向量；`x`：解
- `LU_low`：以 **低精度乘法 + 高精度累加**（如 bf16 × bf16 → FP32）得到的因子化结果
- `‖·‖`：2‑范数，`rel(x)=‖x‖/max(‖b‖, ε)`
- 参数：`tol=1e-10`（可调），`max_ir=5`，`max_gmres=50`（每次 IR 内部迭代上限），`restart=20`

## 1. 标准 IR（低精度分解 + 高精度残差）
```python
def IR_solve(A, b, LU_low, tol=1e-10, max_ir=5):
    # 初解：低精度前/回代
    x = lowprec_solve(LU_low, b)
    for k in range(max_ir):
        # 高精度残差
        r = b - A @ x          # FP64
        if norm(r) <= tol * norm(b):
            return x, {"it": k, "status": "converged"}

        # 低精度求校正量
        delta = lowprec_solve(LU_low, r)

        # 高精度更新
        x = x + delta          # FP64 累加
    return x, {"it": max_ir, "status": "fallback_to_gmres_ir"}
```

## 2. GMRES‑IR（非定常 IR，更强的收敛性）
```python
def GMRES_IR(A, b, LU_low, tol=1e-10, max_ir=5, restart=20, max_gmres=50):
    x = lowprec_solve(LU_low, b)
    for k in range(max_ir):
        r = b - A @ x          # FP64
        if norm(r) <= tol * norm(b):
            return x, {"it": k, "status": "converged"}

        # 右预条件 GMRES（预条件器=低精度LU，内部迭代可用混合精度算子）
        delta = gmres(
            A, r,
            M=lambda y: lowprec_solve(LU_low, y),
            restart=restart, maxiter=max_gmres,
            tol=min(1e-2, 0.1 * tol)  # 内层相对宽松
        )
        x = x + delta
    return x, {"it": max_ir, "status": "failed"}
```

## 3. 触发回退/切换的启发式
- **Pivot growth** 超阈（如 > 1e4）→ 提高 `tol` 或切换 `pivot_mode`，必要时 NB 减小重试。
- **IR 几何收敛失败**：若 `‖r_k‖/‖r_{k-1}‖ > 0.9` 连续两次 → 切换 GMRES‑IR。
- **条件数预估**：`cond(A)·u_low` 明显大于 1（`u_low` 为低精度机内精度）→ 直接 GMRES‑IR 或全精度。
- **上界**：默认 `max_ir=5`；若超限仍未收敛 → 全精度求解。

## 4. 与设备接口结合（建议）
- `lowprec_solve()` 与 `A @ v` 分别映射到：`TRSM/GEMM_UPDATE`（低精度核 + FP32 累加）与高精度 SpMM/GEMV（可在主机或设备上）。
- 每次 IR/GMRES 外层迭代，与设备的**批 front 执行**结合，减少 kernel 启动开销。