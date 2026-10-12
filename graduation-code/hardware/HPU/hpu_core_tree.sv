// hpu_core_tree: purely combinational max-|value| reduction tree.
//
// The tree is stored as a heap: node i has children 2i+1 (left, earlier
// candidates) and 2i+2 (right).  Leaves occupy nodes MAX_ELEMS-1 ..
// 2*MAX_ELEMS-2 in candidate order, so with MAX_ELEMS a power of two every
// comparison sees the earlier candidate on input a.  hpu_cmp_node keeps a on
// equal |value|, which fixes the tie-break to "earliest candidate wins".
//
// Plain 1-D wire arrays are used instead of a 2-D array of packed structs:
// behaviour is identical, and the module now elaborates in Icarus Verilog
// (the struct-array form hit an internal assertion, elab_expr.cc:2677).
module hpu_core_tree #(
    parameter DATA_W    = 32,
    parameter ROW_IDX_W = 16,
    parameter MAX_ELEMS = 256   // must be a power of two
)(
    // 候选数组（来自顶层 buffer）
    input  logic signed [DATA_W-1:0]    cand_val  [0:MAX_ELEMS-1],
    input  logic        [ROW_IDX_W-1:0] cand_row  [0:MAX_ELEMS-1],
    input  logic [$clog2(MAX_ELEMS):0]  num_elems, // 本轮有效元素个数 N (<=MAX_ELEMS)

    // 输出：winner
    output logic [ROW_IDX_W-1:0]        pivot_row,
    output logic signed [DATA_W-1:0]    pivot_value,
    output logic                        pivot_valid   // num_elems>0 时为 1
);

    localparam int NODES = 2 * MAX_ELEMS - 1;
    localparam int FIRST_LEAF = MAX_ELEMS - 1;

    initial begin
        if (MAX_ELEMS < 1 || (MAX_ELEMS & (MAX_ELEMS - 1)) != 0)
            $fatal(1, "hpu_core_tree: MAX_ELEMS (%0d) must be a power of two", MAX_ELEMS);
    end

    wire signed [DATA_W-1:0]    node_val   [0:NODES-1];
    wire        [ROW_IDX_W-1:0] node_row   [0:NODES-1];
    wire                        node_valid [0:NODES-1];

    genvar k, n;
    generate
        // 叶子：候选 k 位于 heap 节点 FIRST_LEAF + k，只有 k < num_elems 有效
        for (k = 0; k < MAX_ELEMS; k = k + 1) begin : gen_leaf
            assign node_val[FIRST_LEAF + k]   = cand_val[k];
            assign node_row[FIRST_LEAF + k]   = cand_row[k];
            assign node_valid[FIRST_LEAF + k] = (k < num_elems);
        end

        // 内部节点：两两比较，左孩子（较早候选）接 a 端
        for (n = 0; n < FIRST_LEAF; n = n + 1) begin : gen_node
            hpu_cmp_node #(
                .DATA_W    (DATA_W),
                .ROW_IDX_W (ROW_IDX_W)
            ) u_cmp (
                .val_a   (node_val[2*n + 1]),
                .row_a   (node_row[2*n + 1]),
                .valid_a (node_valid[2*n + 1]),
                .val_b   (node_val[2*n + 2]),
                .row_b   (node_row[2*n + 2]),
                .valid_b (node_valid[2*n + 2]),
                .val_o   (node_val[n]),
                .row_o   (node_row[n]),
                .valid_o (node_valid[n])
            );
        end
    endgenerate

    // 根节点就是最终 winner
    assign pivot_value = node_val[0];
    assign pivot_row   = node_row[0];
    assign pivot_valid = node_valid[0];

endmodule
