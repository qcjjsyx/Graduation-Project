import numpy as np
import scipy.linalg
'''
16*16 -> 256*256 的分块三角求解 (Blocked TSOLVE) 调度示例
100352个周期
'''


class HardwareArray16x16:
    def __init__(self):
        self.dim = 16
        self.total_cycles = 0

    def run_gemm(self, A_tile, B_tile, C_tile):
        """
        模式 1: 16x16 矩阵乘加 (C = C - A * B)
        """
        # 硬件计算逻辑 (理想情况下全流水线化)
        # 在 16x16 脉动阵列中，标准的输出固定(Output-Stationary) GEMM
        # 延迟通常为: 填充流水线(16) + 计算深度(16) + 排空(16) = 约 48 Cycles
        latency = self.dim * 3 
        self.total_cycles += latency
        
        # 行为级结果计算
        return C_tile - np.dot(A_tile, B_tile)

    def run_tsolve_left(self, L_tile, B_tile):
        """
        模式 2: 16x16 三角求解 (L * D = B)
        """
        # 按照之前讨论的波前推进，延迟为 2 * 16 = 32 Cycles
        latency = self.dim * 2
        self.total_cycles += latency
        
        # 行为级结果计算 (调用线性代数库模拟硬件算出的正确结果)
        return np.linalg.solve(L_tile, B_tile)
    def run_dlu(self, A_tile):
        """
        任务 3: dlu (稠密 LU 分解)
        处理位于对角线上的 16x16 数据块。
        硬件逻辑: 内部循环数据流 + 边角除法器。
        周期估算: 受限于除法器的串行依赖，延迟较高，约 16*16 = 256 Cycles。
        """
        self.total_cycles += 256
        # 使用 scipy 模拟硬件计算出 P, L, U。
        # (注：Spatula 采用静态主元，这里我们通过构建强对角占优矩阵，确保 P=I)
        L, U = scipy.linalg.lu(A_tile)[:2]
        return L, U
    def run_tsolve_right(self, A_tile, U_tile):
        """
        任务 4: tsolve (右侧三角求解，用于求 L_tile)
        求解: L_tile * U_tile = A_tile  =>  L_tile = A_tile * U_tile^-1
        等价于: U_tile^T * L_tile^T = A_tile^T
        """
        self.total_cycles += 32
        return np.linalg.solve(U_tile.T, A_tile.T).T

def blocked_tsolve_256(L, B):
    """
    状态机/调度器: 将 256x256 映射到 16x16 硬件上
    """
    N = 256
    T = 16 # Tile Size
    blocks = N // T
    
    hw = HardwareArray16x16()
    D = np.copy(B)
    
    # 按块行推进 (State Machine Logic)
    for i in range(blocks):
        # 1. 块级更新 (GEMM): 减去左侧已知的块的影响
        for k in range(i):
            # 获取 16x16 的数据块 (从 SRAM 读入)
            L_ik = L[i*T:(i+1)*T, k*T:(k+1)*T]
            # 因为 D 是 256x256，这里其实包含 16 个列块，为了简单我们整体传入或者逐列块传入
            # 真实硬件中，B_tile 也会切成 16x16。这里展示按列块并行的调度：
            for j in range(blocks):
                D_kj = D[k*T:(k+1)*T, j*T:(j+1)*T]
                D_ij = D[i*T:(i+1)*T, j*T:(j+1)*T]
                # 调度硬件执行 GEMM
                D[i*T:(i+1)*T, j*T:(j+1)*T] = hw.run_gemm(L_ik, D_kj, D_ij)
        
        # 2. 局部三角求解 (TSOLVE): 处理对角线块
        L_ii = L[i*T:(i+1)*T, i*T:(i+1)*T]
        for j in range(blocks):
            D_ij = D[i*T:(i+1)*T, j*T:(j+1)*T]
            # 调度硬件执行 TSOLVE
            D[i*T:(i+1)*T, j*T:(j+1)*T] = hw.run_tsolve_left(L_ii, D_ij)
            
    return D, hw.total_cycles

def blocked_lu_256(A):
    """
    上层调度器: 将 256x256 的 LU 分解映射到 16x16 阵列上
    """
    N = A.shape[0]
    T = 16 # 物理阵列维度 (Tile Size)
    blocks = N // T
    
    hw = HardwareArray16x16()
    
    # 初始化用于存储结果的 L 和 U 矩阵 (为了直观，我们分开存储，实际硬件可原地更新 A)
    L = np.zeros((N, N))
    U = np.zeros((N, N))
    
    # 按照块的对角线进行迭代 (0 到 15)
    for i in range(blocks):
        i_start, i_end = i*T, (i+1)*T
        
        # ----------------------------------------------------
        # 步骤 1: 对角线块分解 (dlu)
        # ----------------------------------------------------
        A_ii = A[i_start:i_end, i_start:i_end]
        L_ii, U_ii = hw.run_dlu(A_ii)
        
        L[i_start:i_end, i_start:i_end] = L_ii
        U[i_start:i_end, i_start:i_end] = U_ii
        
        # ----------------------------------------------------
        # 步骤 2: 计算当前列的 L 块和当前行的 U 块 (tsolve)
        # ----------------------------------------------------
        for k in range(i + 1, blocks):
            k_start, k_end = k*T, (k+1)*T
            
            # 求向右侧流动的 U 块: L_ii * U_ik = A_ik
            A_ik = A[i_start:i_end, k_start:k_end]
            U[i_start:i_end, k_start:k_end] = hw.run_tsolve_left(L_ii, A_ik)
            
            # 求向下侧流动的 L 块: L_ki * U_ii = A_ki
            A_ki = A[k_start:k_end, i_start:i_end]
            L[k_start:k_end, i_start:i_end] = hw.run_tsolve_right(A_ki, U_ii)
            
        # ----------------------------------------------------
        # 步骤 3: 更新尾部子矩阵 (dgemm)
        # ----------------------------------------------------
        # 将 L_ki 和 U_ij 的影响从剩下的 A 中扣除
        for k in range(i + 1, blocks):
            k_start, k_end = k*T, (k+1)*T
            for j in range(i + 1, blocks):
                j_start, j_end = j*T, (j+1)*T
                
                L_ki = L[k_start:k_end, i_start:i_end]
                U_ij = U[i_start:i_end, j_start:j_end]
                A_kj = A[k_start:k_end, j_start:j_end]
                
                # 更新 A_kj
                A[k_start:k_end, j_start:j_end] = hw.run_gemm(L_ki, U_ij, A_kj)

    return L, U, hw.total_cycles

# --- 验证测试 ---
if __name__ == "__main__":
    # N = 256
    # # 构造下三角矩阵和右侧矩阵
    # L_256 = np.tril(np.random.rand(N, N) + 1.0)
    # B_256 = np.random.rand(N, N)
    
    # # 硬件调度求解
    # D_hw, total_hw_cycles = blocked_tsolve_256(L_256, B_256)
    
    # # 纯软件求解 (黄金参考)
    # D_sw = np.linalg.solve(L_256, B_256)
    
    # print(f"逻辑矩阵维度: {N}x{N}, 物理阵列维度: 16x16")
    # print(f"模拟硬件执行总周期数: {total_hw_cycles} Cycles")
    # print(f"最大误差 (Max Error): {np.max(np.abs(D_hw - D_sw)):.5e}")

    N = 256
    print(f"开始验证 {N}x{N} 分块 LU 分解 (硬件阵列 16x16)...")
    
    # 构造测试矩阵 (强对角占优以保证无需动态选主元)
    A_sw = np.random.rand(N, N)
    A_sw += np.eye(N) * N 
    
    A_hw = np.copy(A_sw)
    
    # 1. 调用硬件调度模拟器
    L_hw, U_hw, total_hw_cycles = blocked_lu_256(A_hw)
    
    # 2. 软件参考实现 (Scipy)
    L_sw, U_sw = scipy.linalg.lu(A_sw)[:2] # type: ignore
    
    # 3. 结果验证：L * U 是否能还原出原始矩阵 A
    A_reconstructed = np.dot(L_hw, U_hw)
    max_error = np.max(np.abs(A_reconstructed - A_sw))
    
    print("-" * 40)
    print(f"阵列计算总周期数 (理论估算): {total_hw_cycles} Cycles")
    print(f"重构矩阵最大误差 (Max Error): {max_error:.5e}")
    if max_error < 1e-10:
        print("-> 验证通过：软硬件控制流一致！")