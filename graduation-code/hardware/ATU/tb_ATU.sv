`timescale 1ns/1ps
module tb_ATU;

localparam ROW_IDX_W = 8;

logic clk;
logic rst_n;

// 查询接口
logic  q_req_valid;
logic [ROW_IDX_W-1:0] q_req_row_logic; // 逻辑行号 max = 256行
logic  q_req_ready;

logic  q_resp_valid;
logic [ROW_IDX_W-1:0] q_resp_row_physical;// 物理行号


// pivot 更新接口
logic  pivot_req_valid;
logic [ROW_IDX_W-1:0] pivot_row_i; // 行i
logic [ROW_IDX_W-1:0] pivot_row_j; // 行j
logic  pivot_req_ready;
logic  pivot_done;

//配置
logic cfg_we;
logic [ROW_IDX_W-1:0] cfg_p_idx;
logic [ROW_IDX_W-1:0] cfg_p_row_physical;
logic cfg_ready;

// 初始化接口
logic init_identity; // 初始化为单位映射
logic init_done;


ATU#(
    .ROW_IDX_W                  ( ROW_IDX_W )
)u_ATU(
    . q_req_valid          (  q_req_valid          ),
    . q_req_row_logic      (  q_req_row_logic      ),
    . q_req_ready          (  q_req_ready          ),
    . q_resp_valid         (  q_resp_valid         ),
    . q_resp_row_physical  (  q_resp_row_physical  ),
    . pivot_req_valid      (  pivot_req_valid      ),
    . pivot_row_i          (  pivot_row_i         ),
    . pivot_row_j          (  pivot_row_j         ),
    . pivot_req_ready      (  pivot_req_ready      ),
    . pivot_done           (  pivot_done           ),
    . cfg_we              (  cfg_we              ),
    . cfg_p_idx           (  cfg_p_idx          ),
    . cfg_p_row_physical  (  cfg_p_row_physical ),
    . cfg_ready           (  cfg_ready           ),
    . init_identity        (  init_identity        ),
    . init_done            (  init_done            ),
    . clk                  (  clk                  ),
    . rst_n                (  rst_n                )
);

//100MHZ
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    q_req_valid = 1'b0;
    q_req_row_logic = '0;
    pivot_req_valid = 1'b0;
    pivot_row_i = '0;
    pivot_row_j = '0;
    cfg_we = 1'b0;
    cfg_p_idx = '0;
    cfg_p_row_physical = '0;
    init_identity = 1'b0;
    rst_n = 1'b0;
    #1000;
    rst_n = 1'b1;
end


task automatic do_init_identity();
begin
    $display("********** ATU INIT IDENTITY **********");
    @(negedge clk);
    init_identity <= 1'b1;
    @(negedge clk);
    init_identity <= 1'b0;

    wait(init_done==1'b1);
    @(negedge clk);
    $display("********** ATU INIT IDENTITY DONE **********");
end

endtask

task automatic cfg_write(input  [ROW_IDX_W-1:0] idx,
                      input  [ROW_IDX_W-1:0] phys);
begin
    $display("[%0t]********** ATU CFG WRITE **********", $time);
    wait(cfg_ready==1'b1);
    @(negedge clk);
    cfg_we <= 1'b1;
    cfg_p_idx <= idx;
    cfg_p_row_physical <= phys; 
    $display("[%0t]ATU CFG WRITE idx=%0d phys=%0d", $time, idx, phys);
    @(negedge clk);
    cfg_we <= 1'b0;
end
endtask

task automatic pivot_swap(input  [ROW_IDX_W-1:0] i,
                          input  [ROW_IDX_W-1:0] j);
begin
    $display("[%0t]********** ATU PIVOT SWAP **********", $time);
    wait(pivot_req_ready==1'b1);
    @(negedge clk);
    pivot_req_valid <= 1'b1;
    pivot_row_i <= i;
    pivot_row_j <= j;
    $display("[%0t]ATU PIVOT SWAP i=%0d j=%0d", $time, i, j);
    @(negedge clk);
    pivot_req_valid <= 1'b0;

    wait(pivot_done==1'b1);
    @(negedge clk); 
    $display("[%0t]********** ATU PIVOT SWAP DONE **********", $time);
end
endtask


task automatic query(input  [ROW_IDX_W-1:0] logic_idx
                       );
begin
    $display("[%0t]********** ATU QUERY **********", $time);
    wait(q_req_ready==1'b1);
    @(negedge clk);
    q_req_valid <= 1'b1;
    q_req_row_logic <= logic_idx;
    $display("[%0t]ATU QUERY logic_idx=%0d", $time, logic_idx);
    @(negedge clk);
    q_req_valid <= 1'b0;
    wait(q_resp_valid==1'b1);
    @(negedge clk);
    $display("[%0t]ATU QUERY RESULT physical_idx=%0d", $time, q_resp_row_physical);
    $display("[%0t]********** ATU QUERY DONE **********", $time);
end
endtask


//=====================================================================
// Test Sequence
//=====================================================================
initial begin
    wait(rst_n==1'b1);
    #100;

    do_init_identity();

    // 查询测试
    query(8'd10);
    query(8'd20);
    query(8'd30);

    // pivot交换测试
    pivot_swap(8'd10, 8'd20);
    query(8'd10);
    query(8'd20);
    cfg_write(8'd2, 8'd5);
    query(8'd2);
    #1000;
    $finish;


    
end

endmodule