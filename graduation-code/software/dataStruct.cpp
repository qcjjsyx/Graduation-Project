#include <cstdint>



struct Front_Descriptor {
    // 1. 拓扑信息 (用于调度)
    uint32_t task_id;          // 当前任务ID
    uint32_t parent_id;        // 父节点ID (计算完的数据要累加给谁)
    uint32_t dependency_count; // 依赖计数 (有多少个子节点还没算完)

    // 2. 几何信息 (用于配置 TPU/微核)
    uint16_t m, n, k;          // Frontal Matrix 的维度 (M x N)
    uint8_t  is_dense;         // 标志位：是稠密 Front 还是稀疏 Block？
                               // (决定 HPU 用 CALU 模式还是 阈值模式)

    // 3. 内存指针 (DDR 地址)
    uint64_t data_ptr;         // 指向该 Front 在 DDR 中的数据起始地址
    uint64_t contribution_ptr; // 指向该 Front 计算后的 Schur 补要写入的地址
    uint64_t piv_vector_ptr;   // 指向该 Front 的主元置换信息存放地

    // 4. 控制位
    uint32_t config_flags;     // 例如：是否启用混合精度？阈值是多少？
};