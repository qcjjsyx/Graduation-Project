/*
* HPU.sv
* 同步版本，用于主元的选取
*/


module HPU#(
    parameter DATA_WIDTH = 32,
    parameter ROW_IDX_W = 16
)
(

    //================
    //控制/配置端口
    //================
    input logic pivot_start,
    input [1:0] search_mode, //00 CALU 01 Threshold
    input logic [DATA_WIDTH-1:0] threshold_value,
    output logic pivot_busy,

    //================
    //候选数据端口
    //================
    input logic in_valid,
    output logic in_ready,
    input logic [DATA_WIDTH-1:0] in_data,
    input logic [ROW_IDX_W-1:0] in_row_idx,
    input logic is_last,

    //================
    //搜索结果输出
    //================
    output logic pivot_valid,
    input logic pivot_ready,
    output logic [DATA_WIDTH-1:0] pivot_data,
    output logic [ROW_IDX_W-1:0] pivot_row_idx,

    output logic pivot_from_threshold, //1:阈值直接命中
    output logic pivot_fail, // 1:没有找到符合阈值的元素

    input logic clk,
    input logic rst_n
);



endmodule
