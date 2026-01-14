`timescale 1ns/1ps

module tb_gcu_task_fetch;

    localparam int ADDR_W     = 32;
    localparam int TASK_W     = 128;
    localparam int TASK_BYTES = 16;
    localparam int TASK_COUNT = 4;

    // ----------------------------
    // Clock / Reset
    // ----------------------------
    logic clk, rst_n;
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ----------------------------
    // DUT ports
    // ----------------------------
    logic                  start;
    logic [ADDR_W-1:0]      queue_base_addr;
    logic [31:0]            task_count;
    logic                  busy;
    logic                  done;

    logic                  rd_req_valid;
    logic                  rd_req_ready;
    logic [ADDR_W-1:0]      rd_req_addr;
    logic                  rd_rsp_valid;
    logic                  rd_rsp_ready;
    logic [TASK_W-1:0]      rd_rsp_data;

    logic                  task_valid;
    logic                  task_ready;
    logic [TASK_W-1:0]      task_out;

    // ----------------------------
    // DUT instance
    // ----------------------------
    gcu_task_fetch #(
        // .ADDR_W(ADDR_W),
        // .TASK_W(TASK_W),
        // .TASK_BYTES(TASK_BYTES)
    ) dut (
        .start          (start),
        .queue_base_addr(queue_base_addr),
        .task_count     (task_count),
        .busy           (busy),
        .done           (done),
        .rd_req_valid   (rd_req_valid),
        .rd_req_ready   (rd_req_ready),
        .rd_req_addr    (rd_req_addr),
        .rd_rsp_valid   (rd_rsp_valid),
        .rd_rsp_ready   (rd_rsp_ready),
        .rd_rsp_data    (rd_rsp_data),
        .task_valid     (task_valid),
        .task_ready     (task_ready),
        .task_out       (task_out),
        .clk            (clk),
        .rst_n          (rst_n)
    );

    // ----------------------------
    // Simple DDR model (single outstanding)
    // ----------------------------
    logic                 pend_valid;
    logic [ADDR_W-1:0]     pend_addr;
    int                   pend_delay;

    assign rd_req_ready = 1'b1;
    assign rd_rsp_valid = pend_valid && (pend_delay == 0);

    function automatic logic [TASK_W-1:0] make_task_data(input logic [ADDR_W-1:0] addr);
        logic [TASK_W-1:0] t;
        begin
            t = '0;
            t[31:0] = addr[31:0];
            return t;
        end
    endfunction

    always_comb begin
        rd_rsp_data = make_task_data(pend_addr);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pend_valid <= 1'b0;
            pend_addr  <= '0;
            pend_delay <= 0;
        end else begin
            if (rd_req_valid && rd_req_ready) begin
                if (pend_valid) begin
                    $fatal("[%0t] DDR model overflow: multiple outstanding requests", $time);
                end
                pend_valid <= 1'b1;
                pend_addr  <= rd_req_addr;
                pend_delay <= 2 + $urandom_range(0, 1); // 2~3 cycles
            end

            if (pend_valid && pend_delay > 0) begin
                pend_delay <= pend_delay - 1;
            end

            if (rd_rsp_valid && rd_rsp_ready) begin
                pend_valid <= 1'b0;
            end
        end
    end

    // ----------------------------
    // task_ready backpressure
    // ----------------------------
    int handshake_count;
    int stall_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            task_ready      <= 1'b1;
            handshake_count <= 0;
            stall_count     <= 0;
        end else begin
            if (stall_count > 0) begin
                task_ready  <= 1'b0;
                stall_count <= stall_count - 1;
            end else begin
                task_ready <= 1'b1;
            end

            if (task_valid && task_ready) begin
                if (task_out !== make_task_data(queue_base_addr + (handshake_count * TASK_BYTES))) begin
                    $fatal("[%0t] task_out mismatch at idx %0d, got 0x%0h",
                           $time, handshake_count, task_out);
                end
                handshake_count <= handshake_count + 1;
                if ((handshake_count + 1) == 2) begin
                    stall_count <= 3;
                end
            end
        end
    end

    // ----------------------------
    // Test sequence
    // ----------------------------
    initial begin
        rst_n          = 1'b0;
        start          = 1'b0;
        queue_base_addr = 32'h1000;
        task_count     = TASK_COUNT;
        #100;
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (busy == 1'b1);
        wait (done == 1'b1);

        if (handshake_count != TASK_COUNT) begin
            $fatal("[%0t] Expected %0d tasks, got %0d", $time, TASK_COUNT, handshake_count);
        end

        $display("==== gcu_task_fetch_tb PASSED ====");
        #50;
        $finish;
    end

endmodule
