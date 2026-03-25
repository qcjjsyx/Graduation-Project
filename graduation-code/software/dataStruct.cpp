#include <cstdint>

struct Node_Task {
    // --- 1. 身份与控制信息 ---
    uint32_t node_id;         // 节点ID，用于调试
    uint32_t flags;           // 标志位 (例如: Is_Leaf, Is_Root, Finish_Interrupt)
    uint32_t parent_id;       // 父节点id
    uint32_t children_count;   // 该节点有多少个子节点，用于初始化 pending_children[node_id]

    
    // --- 2. 几何尺寸信息 (用于配置循环边界) ---
    uint16_t total_dim;            // 当前波前矩阵的总维数 (N, e.g., 256)
    uint16_t pivot_dim;            // 需要分解的主元块大小 (K, e.g., 32, 64 或 256)
                                   // 注意: Update块大小 = total_dim - pivot_dim
    uint16_t nums_sub_matrix;      // 主元块中以16*16为单位的子矩阵数量 (pivot_dim / 16)，同时也确定了指数保存区域的大小
    uint16_t last_sub_matrix_size; // 最后一个子矩阵的实际大小 (如果 pivot_dim 不是16的倍数)
                                   // 注: 这两个字段主要用于配置循环边界和指数保存区域的大小

    // --- 3. 内存地址信息 (DMA搬运指针) ---
    uint64_t data_addr;       // 当前节点波前矩阵数据在DDR中的起始地址
                              // 软件需预先将 Original A 和 子节点的 Update 累加到这里
    uint64_t parent_address;  // 父节点波前矩阵在DDR中的基地址 (用于写回Update)
    // --- 4. 关键：父节点映射表 (Inter-Node Mapping) ---
    uint64_t map_table_addr;  // 指向一个数组的指针。
                              // 数组内容: 当前节点的第 i 行/列，对应父节点的第 j 行/列 (相对索引)


    // 下面两项为主元矩阵的分解结果
    uint64_t l_factor_addr;   // L 因子写到哪里 (DDR Base Address for L)
    uint64_t u_factor_addr;   // U 因子写到哪里 (DDR Base Address for U)

    uint64_t p_vector_addr;   // [NEW] 用于存储行置换历史 (Permutation Vector)
                              // 大小通常为: total_dim * sizeof(int32)
    uint16_t flag;            //预留字段                          
};
