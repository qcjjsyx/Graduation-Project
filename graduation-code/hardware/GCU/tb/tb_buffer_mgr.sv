`timescale 1ns/1ps

module gcu_buffer_mgr_tb;

    localparam int NUM_BUFS        = 2;
    localparam int TASK_W          = 128;
    localparam int FRONT_ADDR_W    = 4;
    localparam int FRONT_DIM_W     = 4;
    localparam int FRONT_ADDR_LSB  = 0;
    localparam int FRONT_DIM_LSB   = 4;

    //========================
    // Clock / Reset
    //========================
    logic clk, rst_n;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk; // 100MHz
    end

    initial begin
        rst_n                 = 1'b0;
        front_ready_for_task  = 1'b0;
        task_valid            = 1'b0;
        task_in               = '0;
        front_load_done       = '0;
        buf_take              = '0;
        node_compute_done     = '0;
        writeback_done        = '0;
        @(negedge clk);
        #1000;
        rst_n                 = 1'b1;
        front_ready_for_task  = 1'b1;
    end

    //========================
    // DUT ports
    //========================
    logic                             task_valid;
    logic                             task_ready;
    logic [TASK_W-1:0]                task_in;
    logic                             front_ready_for_task;
    logic [NUM_BUFS-1:0]              front_load_req;
    logic [FRONT_ADDR_W-1:0]          front_load_addr[NUM_BUFS-1:0];
    logic [FRONT_DIM_W-1:0]           front_load_dim[NUM_BUFS-1:0];
    logic [NUM_BUFS-1:0]              front_load_done;

    logic [NUM_BUFS-1:0]              buf_ready_for_compute;
    logic [NUM_BUFS-1:0]              buf_take;
    logic [NUM_BUFS-1:0]              node_compute_done;
    logic [NUM_BUFS-1:0]              writeback_done;
    logic [TASK_W-1:0]                buf_task[NUM_BUFS-1:0];
    logic [NUM_BUFS-1:0]              buf_busy;

    //========================
    // DUT instance
    //========================
    gcu_buffer_mgr #(
        .BUFFER_NUM     (NUM_BUFS),
        .TASK_W         (TASK_W),
        .FRONT_ADDR_W   (FRONT_ADDR_W),
        .FRONT_DIM_W    (FRONT_DIM_W),
        .FRONT_ADDR_LSB (FRONT_ADDR_LSB),
        .FRONT_DIM_LSB  (FRONT_DIM_LSB)
    ) dut (
        .task_valid            (task_valid),
        .task_ready            (task_ready),
        .task_in               (task_in),
        .front_ready_for_task  (front_ready_for_task),
        .front_load_req        (front_load_req),
        .front_load_addr       (front_load_addr),
        .front_load_dim        (front_load_dim),
        .front_load_done       (front_load_done),
        .buf_ready_for_compute (buf_ready_for_compute),
        .buf_take              (buf_take),
        .node_compute_done     (node_compute_done),
        .writeback_done        (writeback_done),
        .buf_task              (buf_task),
        .buf_busy              (buf_busy),
        .clk                   (clk),
        .rst_n                 (rst_n)
    );

    //========================
    // 工具函数 / 任务
    //========================
    function automatic logic [TASK_W-1:0] make_task(input int addr, input int dim, input logic [TASK_W-1:0] base = '0);
        logic [TASK_W-1:0] t;
        begin
            t = base;
            t[FRONT_ADDR_LSB +: FRONT_ADDR_W] = addr[FRONT_ADDR_W-1:0];
            t[FRONT_DIM_LSB  +: FRONT_DIM_W ] = dim[FRONT_DIM_W-1:0];
            return t;
        end
    endfunction

    task automatic check_front_req(input int exp_buf, input int exp_addr, input int exp_dim, input string tag);
        begin
            #1;
            if (front_load_req[exp_buf] !== 1'b1) begin
                $fatal("[%0t] %s: expect front_load_req[%0d]=1, got %b", $time, tag, exp_buf, front_load_req);
            end
            if (front_load_addr[exp_buf] !== exp_addr[FRONT_ADDR_W-1:0]) begin
                $fatal("[%0t] %s: expect front_load_addr[%0d]=0x%0h, got 0x%0h",
                       $time, tag, exp_buf, exp_addr[FRONT_ADDR_W-1:0], front_load_addr[exp_buf]);
            end
            if (front_load_dim[exp_buf] !== exp_dim[FRONT_DIM_W-1:0]) begin
                $fatal("[%0t] %s: expect front_load_dim[%0d]=0x%0h, got 0x%0h",
                       $time, tag, exp_buf, exp_dim[FRONT_DIM_W-1:0], front_load_dim[exp_buf]);
            end
        end
    endtask

    task automatic pulse_signal(output logic [NUM_BUFS-1:0] sig, input int idx);
        begin
            @(negedge clk);
            sig <= '0;
            sig[idx] <= 1'b1;
            @(negedge clk);
            sig <= '0;
        end
    endtask

    task automatic expect_ready(input int idx, input bit exp_ready, input bit exp_busy, input string tag);
        begin
            #1;
            if (buf_ready_for_compute[idx] !== exp_ready) begin
                $fatal("[%0t] %s: buf_ready_for_compute[%0d] expect %0b, got %0b",
                       $time, tag, idx, exp_ready, buf_ready_for_compute[idx]);
            end
            if (buf_busy[idx] !== exp_busy) begin
                $fatal("[%0t] %s: buf_busy[%0d] expect %0b, got %0b",
                       $time, tag, idx, exp_busy, buf_busy[idx]);
            end
        end
    endtask

    //========================
    // 驱动任务：发送 Task
    //========================
    task automatic send_task(input int addr, input int dim, input int exp_buf, input string tag);
        logic [TASK_W-1:0] t;
        begin
            t = make_task(addr, dim, {$random, $random, $random, $random});

            @(negedge clk);
            task_in    <= t;
            task_valid <= 1'b1;

            // 等待 ready 拉高，随后在下一个 posedge 采样 front_load_req
            wait (task_ready == 1'b1);
            @(posedge clk);
            check_front_req(exp_buf, addr, dim, tag);

            // handshake 后撤销 valid
            @(negedge clk);
            task_valid <= 1'b0;
        end
    endtask

    //========================
    // 主测试流程
    //========================
    initial begin : main_test
        // 等待复位结束
        wait (rst_n == 1'b1);
        @(negedge clk);

        // 场景 1：两个空闲 buffer 分配 task0/task1，验证 front_load_req 以及 addr/dim 切片
        send_task(4'h1, 4'h3, 0, "task0 -> buf0");
        send_task(4'h2, 4'h4, 1, "task1 -> buf1");

        // 场景 2：无空闲 buffer 时发 task2，不应产生 front_load_req
        @(negedge clk);
        task_in    <= make_task(4'h5, 4'h6);
        task_valid <= 1'b1;
        @(posedge clk);
        #1;
        if (front_load_req !== '0) begin
            $fatal("[%0t] Expect no front_load_req when all buffers are busy", $time);
        end
        @(negedge clk);
        task_valid <= 1'b0;

        // buf0 完成加载 -> READY
        pulse_signal(front_load_done, 0);
        @(posedge clk);
        expect_ready(0, 1'b1, 1'b1, "buf0 after front_load_done");
        if (buf_task[0][FRONT_ADDR_LSB +: FRONT_ADDR_W] !== 4'h1) begin
            $fatal("[%0t] buf0 task addr mismatch", $time);
        end

        // buf1 完成加载 -> READY
        pulse_signal(front_load_done, 1);
        @(posedge clk);
        expect_ready(1, 1'b1, 1'b1, "buf1 after front_load_done");

        // buf0 被调度 -> PROCESSING
        pulse_signal(buf_take, 0);
        @(posedge clk);
        expect_ready(0, 1'b0, 1'b1, "buf0 after buf_take");

        // buf0 计算完成 -> WRITEBACK
        pulse_signal(node_compute_done, 0);
        @(posedge clk);
        expect_ready(0, 1'b0, 1'b1, "buf0 in writeback");

        // buf0 写回完成 -> IDLE
        pulse_signal(writeback_done, 0);
        @(posedge clk);
        expect_ready(0, 1'b0, 1'b0, "buf0 after writeback_done");

        // buf1 直接走完流程
        pulse_signal(buf_take, 1);
        pulse_signal(node_compute_done, 1);
        pulse_signal(writeback_done, 1);
        @(posedge clk);
        expect_ready(1, 1'b0, 1'b0, "buf1 returned to IDLE");

        $display("==== gcu_buffer_mgr_tb PASSED ====");
        #50;
        $finish;
    end

endmodule
