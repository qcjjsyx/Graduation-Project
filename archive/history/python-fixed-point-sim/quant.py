import argparse
import numpy as np
import scipy.linalg as la

class HardwareSimulatorQuantized:
    """
    架构的纯定点量化仿真器
    包含: ATU (地址映射), HPU (硬件选主元), M_format / QF_format 定点数据流
    """
    def __init__(self, A_input, block_size=16, F=10, Q_use_bits=27):
        self.N = A_input.shape[0]
        self.B = block_size
        self.Q_use_bits = Q_use_bits
        
        # 量化参数 (预留 3-4 bit 保护余量以防累加溢出)
        self.F = F
        self.Q_use = (1 << Q_use_bits) - 1
        
        # 1. 模拟 Front Loader 装配期的量化 (求全局 node-scale e_n)
        max_abs = np.max(np.abs(A_input))
        self.e_n = int(np.ceil(np.log2(max_abs / self.Q_use))) if max_abs > 0 else 0
        
        # 转换为 M_format 的 INT64 存储 (模拟物理 SRAM)
        # 保证初始数值范围在 Q_use 内，对应硬件的 INT32
        A_quantized = np.round(A_input / (2 ** self.e_n))
        self.phys_mem = np.clip(A_quantized, -self.Q_use, self.Q_use).astype(np.int64)
        
        # 2. ATU 表 (物理行号 = ATU_Table[逻辑行号])
        self.row_map = np.arange(self.N)

    # ---------------------------------------------------------
    # 辅助函数：模拟硬件的带四舍五入的算术右移
    # ---------------------------------------------------------
    def hw_round_shift(self, val, shift):
        if shift == 0: return val
        return (val + (1 << (shift - 1))) >> shift

    # ==========================================
    # ATU 硬件读写接口模拟 (纯整数读写)
    # ==========================================
    def read_logical(self, r, c):
        return self.phys_mem[self.row_map[r], c]

    def write_logical(self, r, c, val):
        self.phys_mem[self.row_map[r], c] = val

    def swap_logical_rows(self, r1, r2):
        self.row_map[r1], self.row_map[r2] = self.row_map[r2], self.row_map[r1]

    def _find_pivot_row(self, column, start_row):
        max_val = -1
        pivot_row = start_row
        for r in range(start_row, self.N):
            val = abs(self.read_logical(r, column))
            if val > max_val:
                max_val = val
                pivot_row = r
        return pivot_row

    def _swap_if_needed(self, row_a, row_b):
        if row_a == row_b:
            return

        print(
            f"主元交换: 逻辑行 {row_a} <-> 逻辑行 {row_b} "
            f"(物理行 {self.row_map[row_a]} <-> 物理行 {self.row_map[row_b]})"
        )
        self.swap_logical_rows(row_a, row_b)

    def _update_panel_column(self, pivot_row, pivot_col, k_end, pivot_val):
        for r in range(pivot_col + 1, self.N):
            a_val = self.read_logical(r, pivot_col)
            l_val = int(np.round((a_val << self.F) / pivot_val))
            self.write_logical(r, pivot_col, l_val)

            if pivot_col + 1 < k_end:
                for c in range(pivot_col + 1, k_end):
                    u_val = self.read_logical(pivot_row, c)
                    mac_res = l_val * u_val
                    delta = self.hw_round_shift(mac_res, self.F)
                    old_a = self.read_logical(r, c)
                    self.write_logical(r, c, old_a - delta)

    def _solve_upper_block(self, k, k_end):
        for c in range(k_end, self.N):
            for r in range(k, k_end):
                acc = self.read_logical(r, c)
                mac_sum = 0
                for m in range(k, r):
                    l_val = self.read_logical(r, m)
                    u_val = self.read_logical(m, c)
                    mac_sum += l_val * u_val

                delta = self.hw_round_shift(mac_sum, self.F)
                self.write_logical(r, c, acc - delta)

    def _update_trailing_block(self, k, k_end):
        for r in range(k_end, self.N):
            for c in range(k_end, self.N):
                acc = 0
                for m in range(k, k_end):
                    l_val = self.read_logical(r, m)
                    u_val = self.read_logical(m, c)
                    acc += l_val * u_val

                delta = self.hw_round_shift(acc, self.F)
                old_a = self.read_logical(r, c)
                self.write_logical(r, c, old_a - delta)

    # ==========================================
    # 核心：全定点量化计算流
    # ==========================================
    def execute_block_lu_quantized(self):
        for k in range(0, self.N, self.B):
            k_end = min(k + self.B, self.N)

            # ---------------------------------------------------------
            # 阶段 A: Panel LU 分解 (带 HPU 动态主元)
            # ---------------------------------------------------------
            for j in range(k, k_end):
                # 1. HPU 寻主元 (在 M_format 的定点整数域直接比较绝对值)
                p = self._find_pivot_row(j, j)
                self._swap_if_needed(j, p)

                pivot_val = self.read_logical(j, j)
                if pivot_val == 0:
                    raise ValueError("硬件抛出异常：定点域遇到零主元！")

                # 2. 计算乘子 L (QF_format) 并进行 Panel 内部更新
                self._update_panel_column(j, j, k_end, pivot_val)

            # ---------------------------------------------------------
            # 阶段 B: TRSM 计算块 U 矩阵 (前向代换)
            # ---------------------------------------------------------
            if k_end < self.N:
                self._solve_upper_block(k, k_end)

            # ---------------------------------------------------------
            # 阶段 C: GEMM Schur 更新 (交由 TPU 脉动阵列)
            # ---------------------------------------------------------
            if k_end < self.N:
                self._update_trailing_block(k, k_end)

    # ==========================================
    # 结果提取与反量化 (提供给 CPU 侧使用)
    # ==========================================
    def extract_and_dequantize(self):
        LU_logical = self.phys_mem[self.row_map, :]
        
        # 提取 L 并反量化
        L_int = np.tril(LU_logical, -1)
        L_float = L_int.astype(np.float64) / (2 ** self.F)
        L_float += np.eye(self.N) 
       
        # 提取 U 并反量化
        U_int = np.triu(LU_logical)
        U_float = U_int.astype(np.float64) * (2 ** self.e_n)
        
        
        P_out = np.eye(self.N)[self.row_map, :]
       

        return P_out, L_float, U_float, L_int, U_int



class HardwareTRSMSolver:
    """
    模拟硬件内部的全定点前向与后向代换 (TRSM) 算子
    """
    def __init__(self, L_hw, U_hw, F=27):
        self.N = L_hw.shape[0]
        self.F = F
        
        # 硬件内部 SRAM 中保存的定点矩阵
        self.L_int = L_hw.astype(np.int64)  # QF_format
        self.U_int = U_hw.astype(np.int64)  # M_format

    def hw_round_shift(self, val, shift):
        if shift == 0: return val
        return (val + (1 << (shift - 1))) >> shift

    def solve(self, b_hw):
        """
        全硬件定点求解 L * U * x_hw = b_hw
        """
        # ---------------------------------------------------------
        # 1. 硬件前向代换: L * y = b_hw
        # L 是 QF_format (隐式对角线为 1)
        # ---------------------------------------------------------
        y_hw = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            acc = int(b_hw[i])
            mac = 0
            for j in range(i):
                mac += self.L_int[i, j] * y_hw[j]

            delta = self.hw_round_shift(mac, self.F)
            y_hw[i] = acc - delta

        # ---------------------------------------------------------
        # 2. 硬件后向代换: U * x = y_hw
        # U 是 M_format。注意这里的除法精度保护技巧！
        # ---------------------------------------------------------
        x_hw = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N - 1, -1, -1):
            acc = y_hw[i]
            mac = 0
            for j in range(i + 1, self.N):
                mac += self.U_int[i, j] * x_hw[j]

            # 剩余值 (相当于 A22 - L21*U12)
            rem = acc - mac

            # 为了防止整数除法直接丢失精度，
            # 硬件需要在除以 U_ii 之前，将余数向左移位 F 位。
            scaled_rem = rem << self.F

            if self.U_int[i, i] == 0:
                raise ValueError("硬件除零异常！")

            x_hw[i] = int(np.round(scaled_rem / self.U_int[i, i]))

        return x_hw
    
    


def iterative_refinement(A_exact, b_exact, P_hw, L_hw, U_hw, max_iters=5, tol=1e-12):
    """
    混合精度迭代求精 (Mixed-Precision Iterative Refinement)
    利用低精度硬件产生的 LU 分解，通过 CPU 高精度计算残差来逼近真实解。
    """
    x_k = np.zeros_like(b_exact)
    
    print("\n--- 开始迭代求精 (Iterative Refinement) ---") 
    ### 迭代求精有问题
    for i in range(max_iters):
        # 1. 在 CPU 侧用高精度 (FP64) 计算真实残差
        r_k = b_exact - A_exact @ x_k
        


        # 2. 利用硬件定点分解出的 L, U 求解修正量 delta_x
        # 硬件等效求解: P_hw @ A_approx @ delta_x = P_hw @ r_k  =>  L @ U @ delta_x = P_hw @ r_k
        r_k_permuted = P_hw @ r_k
        
        y = la.solve_triangular(L_hw, r_k_permuted, lower=True) 
        delta_x = la.solve_triangular(U_hw, y, lower=False)
        
        # 3. 更新解
        x_k = x_k + delta_x
        
        # 4. 评估误差
        current_error = np.linalg.norm(b_exact - A_exact @ x_k)
        print(f"  Iteration {i+1}: 真实方程残差 = {current_error:.5e}")
        
        if current_error < tol:
            break
            
    return x_k


def hardware_in_the_loop_ir(A_exact, b_exact, P_map, L_int, U_int, e_n, F=27, max_iters=10, tol=1e-8):
    """
    软硬协同的迭代求精循环
    """
    N = len(b_exact)
    x_k = np.zeros(N, dtype=np.float64)
    
    # 实例化硬件 TRSM 模块
    hw_solver = HardwareTRSMSolver(L_int, U_int, F=F)
    
    # # 预先生成置换矩阵 P
    P_matrix = np.eye(N)[P_map, :]
    
    print("\n=== 开始软硬协同 Iterative Refinement ===")
    
    for i in range(max_iters):
        # [软件侧 CPU: 高精度 FP64]
        # 1. 计算真实的极小残差
        r_k = b_exact - A_exact @ x_k
        
        # 判断收敛
        current_error = np.linalg.norm(r_k)
        if current_error < tol:
            print(f"  -> Iteration {i}: 已收敛！残差 = {current_error:.5e}")
            break
            
        print(f"  -> Iteration {i}: 当前残差 = {current_error:.5e}")
        
        # 2. 对残差进行置换 (匹配硬件 ATU 表)
        r_permuted = P_matrix @ r_k
        
        # 3. 动态放大倍数 (Scale Factor e_r) 
        # 目标: 将浮点残差映射到 INT32 满量程 (2^27)，榨干硬件精度
        Q_use = (1 << 27) - 1
        max_r = np.max(np.abs(r_permuted))
        if max_r == 0: break
        
        e_r = int(np.ceil(np.log2(max_r / Q_use)))
        
        # 4. CPU 将残差量化打包，通过 PCIe/总线 发送给硬件
        r_hw = np.clip(np.round(r_permuted / (2 ** e_r)), -Q_use, Q_use).astype(np.int64)
        
        # [硬件侧 TPU: 全低精度 INT32]
        # 5. 硬件直接查 SRAM，使用之前的 L_int 和 U_int 求解
        delta_x_hw = hw_solver.solve(r_hw)
        
        # [软件侧 CPU: 接收并反推真实值]
        # 6. 接收硬件的定点解，并除以总 Scale
        # 缩放因子推导:
        # L_true = L_int / 2^F
        # U_true = U_int * 2^e_n
        # 硬件解方程: L_int * U_int * delta_x_hw = r_hw * (2^F)  <-- 因为有左移逻辑
        # 代入推导后真实的 delta_x_true 需进行的缩放如下:
        total_scale = 2 ** (e_r - e_n - F)
        delta_x_true = delta_x_hw.astype(np.float64) * total_scale # type: ignore
        
        # 7. 更新解
        x_k += delta_x_true
        
    return x_k


def generate_large_value_test_matrix(n):
    """生成数值范围在 10^4 ~ 10^5 级别的测试矩阵，保证随机。"""
    a = np.random.uniform(1,10000, (n, n))
    return a

def generate_test_matrix(n, mode):
    """根据模式生成测试矩阵。"""
    if mode == "stable":
        # 强对角占优，通常几乎不需要主元交换
        a = np.random.randn(n, n)
        a += np.eye(n) * n
        return a

    if mode == "random":
        # 纯高斯随机矩阵，会比 stable 更容易触发交换
        return np.random.randn(n, n)

    if mode == "pivot_stress":
        # 构造容易触发主元交换的矩阵：对角偏小 + 列内人为放大非对角元素
        a = np.random.randn(n, n)
        a += np.eye(n) * 0.1
        for j in range(n):
            i = min(j + 1, n - 1)
            if i != j:
                a[i, j] += 5.0 + np.random.rand()
        return a

    raise ValueError(f"未知矩阵模式: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="ATU/HPU Block LU 硬件行为模拟")
    parser.add_argument("--n", type=int, default=256, help="矩阵维度 N")
    parser.add_argument("--b", type=int, default=32, help="Block 大小 B")
    parser.add_argument(
        "--mode",
        type=str,
        default="stable",
        choices=["stable", "random", "pivot_stress", "large_value"],
        help="测试矩阵模式",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--quiet-swap", action="store_true", help="不打印逐次交换日志")
    return parser.parse_args()

# ==========================================
# 测试与验证逻辑
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    N = args.n
    B = args.b  # Block/Tile size (对应 Spatula 16x16)
    F = 20
    Q_use_bits = 27
    print(f"正在生成测试矩阵: {N}x{N}, Block Size: {B}")
    if args.mode == "large_value":
        A = generate_large_value_test_matrix(N)
    else:
        A = generate_test_matrix(N, args.mode)
    A_exact = A.copy()
    x_true = np.random.uniform(-10, 10, N)
    b_exact = A_exact @ x_true

    print(f"-> 生成的矩阵数值范围: [{np.min(A):.2e}, {np.max(A):.2e}]")
    print(f"x_true: ")
    print(x_true)
    print(f"b_exact: ")
    print(b_exact)

    simulator = HardwareSimulatorQuantized(A, block_size=B, F=F, Q_use_bits=Q_use_bits)
    simulator.execute_block_lu_quantized()
    P_hw, L_hw, U_hw, L_int, U_int = simulator.extract_and_dequantize()  ## L_hw 和 U_hw 是反量化后的浮点矩阵，L_int 和 U_int 是硬件内部的定点表示



    #评估硬件 LU 分解自身的量化相对误差
    reconstructed_A = L_hw @ U_hw
    permuted_A = P_hw @ A_exact
    
    abs_error = np.linalg.norm(permuted_A - reconstructed_A, ord='fro')
    rel_error = abs_error / np.linalg.norm(A_exact, ord='fro')
    
    print(f"\n-> 硬件矩阵分解 绝对误差: {abs_error:.2e} (数值大时必定很大)")
    print(f"-> 硬件矩阵分解 相对误差: {rel_error:.2e} (衡量量化精度的真实指标)")
    
    if rel_error < 1e-6:
        print("  [Pass] INT32 量化精度符合预期。")
    else:
        print("  [Warning] 量化误差略大，矩阵可能存在严重尺度差异。")

    # 4. 通过迭代求精验证其实际工程价值
    x_solved = iterative_refinement(A_exact, b_exact, P_hw, L_hw, U_hw, max_iters=10, tol=1e-12)
   

    # # 1. 运行硬件模拟器 (带 ATU & HPU)
    # simulator = HardwareSimulator(A, block_size=B, verbose_swap=not args.quiet_swap)
    # simulator.execute_block_lu()
    
    # P_hw, L_hw, U_hw = simulator.extract_results()

    # # 2. 验证计算残差
    # # 按照公式: P_hw @ A = L_hw @ U_hw 验证
    # reconstructed_A = L_hw @ U_hw
    # permuted_A_original = P_hw @ A
    
    # residual = np.linalg.norm(permuted_A_original - reconstructed_A, ord='fro')
    
    # print("\n========= 验证结果 =========")
    # print(f"Frobenius 范数误差 (Residual): {residual:.5e}")
    # total_pivot_steps = N
    # swap_ratio = simulator.swap_count / total_pivot_steps
    # print(f"主元交换次数: {simulator.swap_count}/{total_pivot_steps} ({swap_ratio:.2%})")
    # if residual < 1e-10:
    #     print("[Pass] ATU/HPU 架构逻辑验证通过！物理数据无搬移下实现了正确的动态主元选取与计算。")
    # else:
    #     print("[Fail] 误差过大，架构逻辑存在问题。")

    # 3. 展现解耦带来的物理乱序情况 (证明确实没有移动物理行)
    # print("\n[架构现象观察]")
    # print("ATU 表最终映射 (Logical -> Physical):", simulator.row_map, "...")
    # print("注意：如果上述序列不是 0, 1, 2..., 说明发生过主元交换，但我们从未操作物理矩阵！")









