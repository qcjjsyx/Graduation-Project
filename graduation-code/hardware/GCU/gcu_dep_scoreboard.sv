/**
 * gcu_dep_scoreboard.sv
 *
 * 依赖计数模块
 *
 * 用于跟踪每个节点的 pending_children 计数，以及 front_ready 状态。
 * 当一个节点的 pending_children 计数为 0 时，front_ready=1，表示该节点可以被调度执行。
 *
 */

module gcu_dep_scoreboard #(
    parameter NODE_ID_W = 4,
    parameter MAX_NODES = 16,
    parameter int CHILD_CNT_W  = 4     // children_count 计数位宽
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
    // 只有当 query_valid=1 时才认为本次查询有效，否则输出置零。
    input  logic                  query_valid,
    input  logic [NODE_ID_W-1:0]  query_node_id,
    output logic                  front_ready,
    output logic [CHILD_CNT_W-1:0]     pending_children_count,


    input  logic                  clk,
    input  logic                  rst_n
);


    // 对每个 node_id 存一份 pending_children 和 front_ready
    // 索引范围假定 node_id ∈ [0, MAX_NODES-1]
    logic [CHILD_CNT_W-1:0] pending_children   [0:MAX_NODES-1];
    logic                   front_ready_reg    [0:MAX_NODES-1];

    //todo: bug 当 MAX_NODES 很大时， 异步查询会出现x态
    //========================
    // 异步查询逻辑
    // assign front_ready = (query_valid && (query_node_id < MAX_NODES)) ?
    //                      front_ready_reg[query_node_id] : 1'b0;
    // assign pending_children_count = (query_valid && (query_node_id < MAX_NODES)) ?
    //                                 pending_children[query_node_id] : '0;

    //========================

    // 同步查询输出：一拍后返回结果，query_valid 无效或越界则清零
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_children_count <= '0;
            front_ready            <= 1'b0;
        end else if (query_valid && (query_node_id < MAX_NODES)) begin
            pending_children_count <= pending_children[query_node_id];
            front_ready            <= front_ready_reg[query_node_id];
        end else begin
            pending_children_count <= '0;
            front_ready            <= 1'b0;
        end
    end

    integer i;
    //========================
    // 同步更新逻辑
    //========================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位时，所有节点默认 pending_children = 0，front_ready = 1
            // 表示“没有子节点依赖”，可立即执行。
            for (i = 0; i < MAX_NODES; i = i + 1) begin
                pending_children[i] <= '0;
                front_ready_reg[i]  <= 1'b1;
            end
        end else begin

            // --- 初始化某个 node 的 children_count ---
            if (init_valid) begin
                // 这里假设 init_node_id < MAX_NODES
                pending_children[init_node_id] <= init_children_count;
                // 如果 children_count==0，则一开始就 ready；否则需要等待 children 完成
                front_ready_reg[init_node_id]  <= (init_children_count == '0);
            end

            // --- Scatter 完成事件：对 parent_id 做 pending_children-- ---
            if (scatter_done_valid) begin
                // 当前假设：scatter_done_parent_id 一定在合法范围内，
                // 且 pending_children 不会 underflow（由软件保证 children_count 正确）。
                if (pending_children[scatter_done_parent_id] != '0) begin
                    pending_children[scatter_done_parent_id]
                        <= pending_children[scatter_done_parent_id] - {{(CHILD_CNT_W-1){1'b0}}, 1'b1};

                    // 如果减完之后变成 0，则 front_ready 置 1
                    if (pending_children[scatter_done_parent_id] == {{(CHILD_CNT_W-1){1'b0}}, 1'b1}) begin
                        front_ready_reg[scatter_done_parent_id] <= 1'b1;
                    end
                end
                // 如果原本就是 0，说明上层逻辑有误，这里选择忽略（也可以加 error_flag）
            end
        end
    end


endmodule