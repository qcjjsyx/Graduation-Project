# 历史材料归档

这里保存有实验、正式提交或设计决策追溯价值的材料。**归档内容不定义当前架构，也不作为
实现任务或论文架构的依据。** 当前主线见 [docs/项目主线梳理.md](../docs/项目主线梳理.md)。

2026-10-10 起，所有历史材料统一放在本目录下（原根目录 `unused/` 与原
`graduation-code/sim/` 也已移入）。

## 目录

| 目录 | 内容 | 与主线的关系 |
|---|---|---|
| `legacy-node-task-abi-v2/` | 旧 NodeTask ABI v2 的软件流水线、SystemC 执行器、optimization 副本 | 已被 Command/Descriptor schema v2 取代（两者无关） |
| `optimization-v3/` | QF26 / 自适应精度 / Tile-BFP 实验文档 | 全局 BFP 路线已退出主线 |
| `precision-history/` | 旧定点稳定性设计、量化方案 V2 | 同上 |
| `history/python-fixed-point-sim/` | 原 `graduation-code/sim/`：早期 Python 定点 LU/TRSM/ATU/HPU 行为模型与实验脚本 | 不被编译器或设备调用 |
| `history/code/`、`history/root/` | 早期代码说明、进度、已解决的问题记录、2026-09 文档整理清单 | 过程记录 |
| `history/design-notes/`、`history/experiments/` | 早期设计分析、行为模型实验报告 | 参考 |
| `formal/opening/` | 开题材料 | 正式过程记录 |
| `unused/` | 已被否定的早期设计、旧 prompt、optimization-v1/v2 文档副本 | 不再作为参考依据 |
| `前期工作与重构依据：基线到v1-v3的演进说明.md` | 前期工作量、问题发现与重构依据 | 论文“前期工作”素材 |
