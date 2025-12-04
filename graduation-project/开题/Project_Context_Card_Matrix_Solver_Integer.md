
# Project Context Card — Matrix Solver (Integer‑Only)

> 把本卡片长期保存在你的仓库中；每次开启新对话，把**“## 压缩上下文”**这一段粘贴到对话最开头即可，让模型在同一约束下继续协作。

---

## 固定约束
- **总体目标**：在**纯整数**算力（int8/16 乘，int32 累加）下实现**块化 LU（部分选主元）**的主干流水：`Panel‑GEPP → TRSM → GEMM`；**不使用浮点/IR**。
- **TPU**：仅支持 int8/16/32；GEMM/AXPY/点积下推 TPU 阵列（acc32），回写前做重整/舍入。
- **RV64**：运行“固件/运行时”，负责**SQ/CQ 队列推进、SG‑DMA、双缓冲、ATU 延迟交换（置换向量）、定点倒数近似/除法、缩放控制、溢出/饱和监控**。
- **主机侧**：可由 **Vivado Testbench** 充当；负责**离线量化/校准（每面板/列 scale）**，生成 **SQ 命令和 SG‑DMA 描述符**；可选计算 **FP64 金标**用于校验。
- **暂不做**：消解树/符号分析缓存；浮点路径与迭代改进（IR）。
- **验证平台**：Vivado 仿真（行为/综合后）。指标：残差、饱和/溢出率、吞吐（GFLOP/s）、DMA 等待比例。

---

## 术语 / 接口（关键）
- **SQ/CQ**：提交/完成环形队列 + doorbell。
- **SG‑DMA**：散/聚搬运；SRAM‑S1（Panel 缓冲）/SRAM‑S2（Update 缓冲）**双缓冲**。
- **ATU（Address Translation Unit）**：**地址级延迟交换**，用置换向量完成行交换，避免大规模数据搬移。
- **Panel‑GEPP_int / TRSM_int / GEMM_int**：整数内核；`pivot_mode ∈ {threshold, rook}`；`NB ∈ {128, 256}`。
- **Scale 控制**：每面板/列的定点 Qm.n 缩放与右移；**饱和率门限**触发回退（降 NB / 切换 pivot 模式）。

---

## 压缩上下文（请粘贴到每次新会话的最开头）
我们做一个 **int‑only 块化 LU** 的软硬件协同原型：
HOST（或 Vivado TB）负责**离线量化/校准**与**SQ/SG‑DMA 描述符**；
**RV64 固件**推进 **SQ/CQ、SG‑DMA、双缓冲、ATU、定点倒数/除法、Scale 控制与监控**；
**Panel‑GEPP_int 与 TRSM_int** 在设备侧实现；**GEMM_int** 下推 **TPU（int8/16×int8/16→acc32）**，回写前重整；
**不使用浮点与 IR**；**不做符号分析缓存**；以 **残差/饱和率/吞吐** 为核心指标。
你给我的回答要**默认遵守以上约束**，并输出**可直接落地的接口/伪码/测试清单**。

---

## 最小可交付闭环（建议 2–3 周搞定）
1) **离线量化/定点仿真脚本**（Python）：输入矩阵 → 输出 `A_q, scale.json` 与 SQ/SG‑DMA 描述符 → 运行定点 LU 参考 → 产出残差/饱和率。  
2) **RV64 固件最小集**：`SQ/CQ + SG‑DMA + 双缓冲 + Panel‑GEPP_int(阈值) + GEMM_int(调用 TPU) + 写回融合`；计数器：面板/更新/等待占比。  
3) **回退门限**：当 `饱和率>阈值` 或 `pivot<阈值`：`{NB↓} ∨ {threshold→rook}`，并记录事件。  
4) **报告模板**：每次仿真输出 `残差/饱和率/吞吐/回退次数/面板时分布` 五项。

---

## 测试与度量（Vivado 驱动思路）
- **Hostless 驱动**：Testbench 扮演 Host，读取 `*.bin`（矩阵/描述符），写 SQ、拉 CQ，观测 AXI 波形与计数器。  
- **验证分层**：
  - Python 参考（定点仿真，与量化器同源参数）；  
  - C/RTL Co‑Sim（SystemVerilog DPI 驱动队列）；  
  - 门限/回退回归（构造“坏”矩阵触发回退/降 NB）；  
  - 性能回归（GFLOP/s、GB/s、队列耗时分解、饱和率）。

---

## 版本/变更记录（示例）
- v0.1（2025‑11‑10）：初版；确定“整数‑only、TPU 下推 GEMM、RV64 固件推进、无 IR/无符号缓存、Vivado 验证”基线。

---

*Maintainer: czx | Companion: ChatGPT（GPT‑5 Thinking）*
