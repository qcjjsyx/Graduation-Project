`timescale 1ns/1ps

module tb_buffer_mgr;

   
    localparam int NUM_BUFS = 2;
    // ----------------------------
    // Clock / Reset
    // ----------------------------
    logic clk, rst_n;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk; // 100MHz
    end

    initial begin
        rst_n = 1'b0;
        #1000;
        rst_n = 1'b1;
    end

    // ----------------------------
    // DUT ports
    // ----------------------------
    logic                             task_ready;
    logic [127:0]                     task_in;
    logic                             front_ready_for_task;
    logic [NUM_BUFS-1:0]              front_load_req;
    logic [4-1:0]                     front_load_addr[NUM_BUFS-1:0];
    logic [4-1:0]                     front_load_dim[NUM_BUFS-1:0];
    logic [NUM_BUFS-1:0]              front_load_done;

    logic [NUM_BUFS-1:0]              buf_ready_for_compute;
    logic [NUM_BUFS-1:0]              buf_take;
    logic [NUM_BUFS-1:0]              node_compute_done;
    logic [NUM_BUFS-1:0]              writeback_done;
    logic [NUM_BUFS-1:0]              buf_busy;

    // ----------------------------
    // DUT instance
    // ----------------------------
// (*dont_touch = "true"*)gcu_buffer_mgr#(
//     .BUFFER_NUM                                   ( 2 )
// ) u_gcu_buffer_mgr (
//     .task_ready                            ( task_ready ), 
//     .task_in                               ( task_in ),
//     .front_ready_for_task                  ( front_ready_for_task ),
//     .front_load_req                        ( front_load_req ),
//     .front_addr                            ( front_load_addr  ),
//     .front_dim                             ( front_load_dim  ),
//     .front_load_done                       ( front_load_done ),
//     .buf_ready_for_compute                 ( buf_ready_for_compute ),
//     .buf_take                              ( buf_take ),
//     .node_compute_done                     ( node_compute_done ),
//     .writeback_done                        ( writeback_done ),
//     .buf_busy                              ( buf_busy ),
//     .clk                                   ( clk ),
//     .rst_n                                 ( rst_n )
// );



endmodule