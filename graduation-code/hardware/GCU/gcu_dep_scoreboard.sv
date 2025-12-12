module gcu_dep_scoreboard #(
    parameter NODE_ID_W = 16,
    parameter MAX_NODES = 1024,
    parameter int CHILD_CNT_W  = 16     // children_count 计数位宽
) (

    // 初始化 children_count
    // 当 init_valid=1 时，把该 node 的 pending_children 设置为 init_children_count
    // 通常在加载 Node_Task 后调用一次。
    input  logic                  init_valid,
    input  logic [NODE_ID_W-1:0]  init_node_id,
    input  logic [15:0]           init_children_count,

    // Scatter 完成事件
    // 当某个 child 的 update 写回父节点完成时，scatter_done_valid=1，
    // 通知本模块：child_id / parent_id，用于对 pending_children[parent_id]--。
    input  logic                  scatter_done_valid,
    input  logic [NODE_ID_W-1:0]  scatter_done_child_id,
    input  logic [NODE_ID_W-1:0]  scatter_done_parent_id,

    // 查询接口
    // 外部通过 query_node_id 查询该节点当前 front_ready / pending_children_count。
    input  logic [NODE_ID_W-1:0]  query_node_id,
    output logic                  front_ready,

    input  logic                  clk,
    input  logic                  rst_n,
);



endmodule