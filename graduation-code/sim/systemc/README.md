# SystemC ATU/HPU Behavior Model

这个目录用于存放硬件侧 SystemC 行为模型。当前版本只建模两个最核心、最容易理解的模块：

- `ATU`：维护逻辑行到物理行的映射，通过交换映射表完成零拷贝 pivot。
- `HPU`：接收一段候选元素流，选择绝对值最大的 pivot。

当前模型属于第一阶段行为级建模：保留时钟、复位、valid/ready、状态机和基本延迟，但不追求 RTL 逐拍等价。

## 文件结构

```text
systemc/
  CMakeLists.txt
  README.md
  include/
    atu.hpp
    hpu.hpp
  src/
    main.cpp
```

## 构建方式

需要先安装 SystemC，并让 CMake 找到它。推荐设置 `SYSTEMC_HOME`：

```bash
cd graduation-code/sim/systemc
cmake -S . -B build -DSYSTEMC_HOME=/path/to/systemc
cmake --build build
./build/atu_hpu_demo
```

在 Windows 上可将 `/path/to/systemc` 替换为你的 SystemC 安装目录，例如：

```powershell
cmake -S . -B build -DSYSTEMC_HOME=C:\systemc
cmake --build build
.\build\Debug\atu_hpu_demo.exe
```

## 如何理解这个模型

### ATU

ATU 内部保存一个 `pvec`：

```text
logical row -> physical row
```

初始化后：

```text
pvec[0] = 0
pvec[1] = 1
...
```

当 pivot 请求交换逻辑行 `3` 和 `7` 时，ATU 只交换：

```text
pvec[3] <-> pvec[7]
```

矩阵 SRAM 中的物理数据不搬移。

### HPU

HPU 接收候选流：

```text
(row, value), (row, value), ...
```

当 `in_last=1` 时，本轮候选输入结束。HPU 随后输出绝对值最大的元素所在逻辑行。若绝对值相同，当前模型保留先到的候选。

## 当前 testbench 覆盖内容

`src/main.cpp` 会执行：

1. 复位 ATU/HPU。
2. 初始化 ATU 为 identity 映射。
3. 查询逻辑行 3，得到物理行 3。
4. 请求 ATU 交换逻辑行 3 和 7。
5. 再查询逻辑行 3/7，确认映射互换。
6. 向 HPU 送入多个候选值，确认选出绝对值最大的 pivot。

这个 testbench 是为了帮助理解模块行为，不是完整验证平台。
