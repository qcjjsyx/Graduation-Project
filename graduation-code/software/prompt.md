**角色设定**：你是一个资深的数值线性代数 + 系统软件工程师。请用 Python 为一个“multifrontal LU（面向硬件加速）”的上位机软件侧实现一个可运行的最小闭环原型，并生成完整项目结构、代码与单元测试。

### 目标

#### 实现软件侧流水：

 	1.	读取稀疏矩阵 A（支持 MatrixMarket .mtx 与随机生成 或者是 matlab的矩阵格式）
 	2.	进行符号分析：重排、消解树构建、超节点切分（先做简化版本也可以）
 	3.	生成 Node 级任务 NodeTask、生成子→父的 map_table（extend-add 映射）
 	4.	进行 DDR 地址规划（memory planner）：为每个 node 分配 front_q/front_e/update_q/update_e/L/U/map_table/task_desc 的地址区间（用 offset 模拟即可）
 	5.	进行 软件侧初始化量化：把“来自原矩阵 A 的本地贡献”装配到每个 node 的 frontal（float32 工作区），然后按 tile/subblock 规则量化为 (q,e)，写入二进制文件
 	6.	生成任务队列（task_queue）：做 sibling-friendly 的拓扑排序（避免父子紧邻），输出 tasks.bin
 	7.	verify：用 SciPy 的参考解（splu 或 spsolve）或 dense 小规模参考，给出 residual norm；同时统计量化裁剪/饱和次数
 	8.	iterative refinement：提供一个框架（不要求完全实现硬件 LU），至少实现 residual 计算与停止条件；留出接口调用“硬件返回的 LU 求解结果”

**量化思路**（必须严格按此实现，你可以单独用一个文件来编写量化算法，方便我后续如果有修改可以快速实现修改）
	•	tile = 32×32
	•	每 tile 分为 4 个 sub-block：16×16
	•	每个 sub-block 一个 exponent int8 e
	•	mantissa int32 q（二补码）
	•	有效位宽 B_eff = 24，即 Q = 2^(B_eff-1)-1
	•	定标：使用 p=99.5 分位数的 a = percentile(|X|, p)（若 a=0 则全零）
	•	exponent：e = ceil(log2(a / Q))
	•	裁剪：Xc = clip(X, -a, a)
	•	量化：q = clip(round(Xc / 2^e), -Q, Q) 转 int32
	•	可选密度修正：若 count(|q|>=2)/256 < 0.05，令 e := e-1 重算一次（最多一次）
	•	exponent 存储布局：每 tile 4 个 int8，按 (00,01,10,11) 顺序；同时提供一个“打包为 uint32”的接口（4×int8 → uint32）

### 数据结构与文件输出

**dataclasses 定义**
	•	NodeTask：请你参考dataStruct.cpp中的定义
	•	MapTableEntry：记录 child update 的 (local_row,local_col) → parent frontal 的 (row,col) 映射（可用压缩格式：行映射表+列映射表）
**输出二进制文件（用于后续硬件对接）：**
	•	tasks.bin：顺序写入 NodeTask（小端，固定对齐，给出 struct.pack 格式）
	•	map_table.bin：按 node 写入其 map_table
	•	front_q.bin/front_e.bin：按 node 写入初始化后的 frontal 量化数据
	•	同时输出一个 manifest.json 记录每个 node 的地址区间、tile 数、文件偏移，便于调试

**算法简化允许（第一版可用简化替代，但要写清楚 TODO）**
	•	重排：可先用 scipy.sparse.csgraph.reverse_cuthill_mckee 作为替代（TODO：AMD/METIS）
	•	etree 与 supernode：可先实现一个“基于 elimination tree 的节点划分”简化版本；若复杂，可先对 small demo 用人为划分，但必须提供清晰接口，后续可替换
	•	frontal 装配：第一版可以只装配“来自 A 的本地贡献”（child update 留空），但 map_table/任务依赖必须能生成（用于验证依赖逻辑）

### 工程要求
​	•	使用 numpy, scipy（允许）；测试用 pytest
​	•	所有模块都有 type hints
​	•	提供 CLI：python -m software.main --mtx path --out out_dir --seed 0
​	•	提供至少 5 个单元测试：

	1.	量化-反量化误差基本性质（非零比例、MSE）
	2.	sub-block exponent 的布局/打包正确
	3.	memory planner 地址不重叠
	4.	task_queue 顺序满足拓扑（父在子之后），且 sibling 优先策略生效
	5.	map_table 映射在一个手工小例子上可验证

### 项目结构（请按此生成）

software/
init.py
main.py
io.py
symbolic/
reorder.py
etree.py
supernode.py
scheduler/
map_gen.py
task_queue.py
memory/
planner.py
pack.py
quant/
bfp_quant.py
verify/
metrics.py
tests/
test_quant.py
test_pack.py
test_planner.py
test_task_queue.py
test_map_table.py
pyproject.toml / requirements.txt / README.md

请直接输出所有文件的内容（按路径分段），保证能运行与通过测试
