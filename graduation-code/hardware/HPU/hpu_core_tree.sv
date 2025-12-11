module hpu_core_tree #(
    parameter DATA_W    = 32,
    parameter ROW_IDX_W = 16,
    parameter MAX_ELEMS  = 256
)(
    // 候选数组（来自顶层 buffer）
    input  logic signed [DATA_W-1:0]   cand_val   [0:MAX_ELEMS-1],
    input  logic        [ROW_IDX_W-1:0] cand_row  [0:MAX_ELEMS-1],
    input  logic [$clog2(MAX_ELEMS):0] num_elems, // 本轮有效元素个数 N (<=MAX_ELEMS)

    // 输出：winner
    output logic [ROW_IDX_W-1:0]       pivot_row,
    output logic signed [DATA_W-1:0]   pivot_value,
    output logic                       pivot_valid   // num_elems>0 时为 1
);

    localparam LEVELS = $clog2(MAX_ELEMS);

  // 每个节点携带 (val, row, valid)
    typedef struct packed {
        logic signed [DATA_W-1:0] val;
        logic [ROW_IDX_W-1:0]     row;
        logic                     valid;
    } node_t;

    // level_nodes[l][i]：第 l 层的第 i 个节点
    // l = 0   : 叶子层（来自 cand_*）
    // l = ... : 上层
    node_t level_nodes [0:LEVELS][0:MAX_ELEMS-1];

    integer i;

    // 叶子层：映射 cand_* 到 level 0，并根据 num_elems 设置 valid
    always_comb begin
        for (i = 0; i < MAX_ELEMS; i = i + 1) begin
            level_nodes[0][i].val   = cand_val[i];
            level_nodes[0][i].row   = cand_row[i];
            level_nodes[0][i].valid = (i < num_elems) ? 1'b1 : 1'b0;
        end
    end

    // 生成树：每一层通过 hpu_cmp_node 把 2 个子节点合并成 1 个父节点
    genvar lvl, j;
    generate
        for (lvl = 0; lvl < LEVELS; lvl = lvl + 1) begin : gen_levels
            localparam int CUR_NODES  = (MAX_ELEMS >> lvl);       // 当前层节点数
            localparam int NEXT_NODES = (MAX_ELEMS >> (lvl + 1)); // 下一层节点数

            for (j = 0; j < NEXT_NODES; j = j + 1) begin : gen_nodes
                hpu_cmp_node #(
                    .DATA_W    (DATA_W),
                    .ROW_IDX_W (ROW_IDX_W)
                ) u_cmp (
                    .val_a   (level_nodes[lvl][2*j].val),
                    .row_a   (level_nodes[lvl][2*j].row),
                    .valid_a (level_nodes[lvl][2*j].valid),

                    .val_b   (level_nodes[lvl][2*j+1].val),
                    .row_b   (level_nodes[lvl][2*j+1].row),
                    .valid_b (level_nodes[lvl][2*j+1].valid),

                    .val_o   (level_nodes[lvl+1][j].val),
                    .row_o   (level_nodes[lvl+1][j].row),
                    .valid_o (level_nodes[lvl+1][j].valid)
                );
            end

            // 其余未用节点（若有）默认 invalid（这里 MAX_ELEMS 是 2 的幂，所以不需要额外处理）
        end
    endgenerate

    // 根节点就是最终 winner
    assign pivot_value = level_nodes[LEVELS][0].val;
    assign pivot_row   = level_nodes[LEVELS][0].row;
    assign pivot_valid = level_nodes[LEVELS][0].valid;

endmodule