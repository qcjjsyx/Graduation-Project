// gcu_types_pkg.sv
package gcu_types_pkg;

    // 统一字段位宽（建议全工程统一使用）
    // parameter int ADDR_W      = 64;
    // parameter int DIM_W       = 32;
    // parameter int NODE_ID_W   = 16;
    // parameter int CHILD_CNT_W = 16;
    // parameter int FLAGS_W     = 32;


    parameter int ADDR_W      = 4;
    parameter int DIM_W       = 4;
    parameter int NODE_ID_W   = 4;
    parameter int CHILD_CNT_W = 4;
    parameter int FLAGS_W     = 4;

    // Node_Task 的硬件描述符（packed，便于寄存/数组存储）
    typedef struct packed {
        logic [DIM_W-1:0]       total_dim;        // 当前波前矩阵维数
        logic [DIM_W-1:0]       pivot_dim;        // 主元块大小

        logic [NODE_ID_W-1:0]   node_id;          // 本节点 ID
        logic [NODE_ID_W-1:0]   parent_id;        // 父节点 ID（根节点可用全 1 表示）
        logic [CHILD_CNT_W-1:0] children_count;   // 子节点数量（用于依赖初始化）
        logic [FLAGS_W-1:0]     flags;            // 扩展标志位（可选）

        logic [ADDR_W-1:0]      front_addr;       // 本节点 frontal/work 区 DDR 基地址
        logic [ADDR_W-1:0]      parent_front_addr;// 父节点 frontal DDR 基地址（Scatter 目标）
        logic [ADDR_W-1:0]      map_table_addr;   // Scatter 映射表地址

        logic [ADDR_W-1:0]      l_factor_addr;    // L 因子写回地址
        logic [ADDR_W-1:0]      u_factor_addr;    // U 因子写回地址
    } NodeTask_t;

endpackage
