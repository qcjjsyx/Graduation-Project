module hpu_cmp_node #(
    parameter DATA_W = 32,
    parameter ROW_IDX_W = 16
)
(
    input  logic signed [DATA_W-1:0] val_a,
    input  logic [ROW_IDX_W-1:0]     row_a,
    input  logic                     valid_a,

    input  logic signed [DATA_W-1:0] val_b,
    input  logic [ROW_IDX_W-1:0]     row_b,
    input  logic                     valid_b,

    output logic signed [DATA_W-1:0] val_o,
    output logic [ROW_IDX_W-1:0]     row_o,
    output logic                     valid_o
);
    logic [DATA_W-1:0] abs_a, abs_b;

    always_comb begin
        // 绝对值
        if (val_a < 0) abs_a = ~val_a + 1'b1; else abs_a = val_a;
        if (val_b < 0) abs_b = ~val_b + 1'b1; else abs_b = val_b;

        unique case ({valid_a, valid_b})
            2'b00: begin
                valid_o = 1'b0;
                val_o   = '0;
                row_o   = '0;
            end
            2'b01: begin
                valid_o = 1'b1;
                val_o   = val_b;
                row_o   = row_b;
            end
            2'b10: begin
                valid_o = 1'b1;
                val_o   = val_a;
                row_o   = row_a;
            end
            2'b11: begin
                valid_o = 1'b1;
                if (abs_a >= abs_b) begin
                    val_o = val_a;
                    row_o = row_a;
                end else begin
                    val_o = val_b;
                    row_o = row_b;
                end
            end
        endcase
    end



endmodule