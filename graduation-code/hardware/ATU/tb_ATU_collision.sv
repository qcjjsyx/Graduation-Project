`timescale 1ns/1ps
// Self-checking ATU testbench (regression for B03).
//
// A cfg write and a pivot swap requested in the same cycle share one Pvec
// write port.  The contract is: swap has priority, cfg_ready is low in that
// cycle, and the caller keeps cfg_we asserted until cfg_ready is seen.  Both
// operations must therefore take effect; nothing may be dropped silently.
//
//   iverilog -g2012 -s tb_ATU_collision -o /tmp/tb_atu ATU/ATU.sv ATU/tb_ATU_collision.sv
//   vvp /tmp/tb_atu
module tb_ATU_collision;
    localparam int W = 3;  // 8 logical rows keeps identity init short

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic         q_req_valid = 1'b0;
    logic [W-1:0] q_req_row_logic = '0;
    logic         q_req_ready;
    logic         q_resp_valid;
    logic [W-1:0] q_resp_row_physical;
    logic         pivot_req_valid = 1'b0;
    logic [W-1:0] pivot_row_i = '0;
    logic [W-1:0] pivot_row_j = '0;
    logic         pivot_req_ready;
    logic         pivot_done;
    logic         cfg_we = 1'b0;
    logic [W-1:0] cfg_p_idx = '0;
    logic [W-1:0] cfg_p_row_physical = '0;
    logic         cfg_ready;
    logic         init_identity = 1'b0;
    logic         init_done;

    ATU #(.ROW_IDX_W(W)) dut (
        .q_req_valid(q_req_valid), .q_req_row_logic(q_req_row_logic),
        .q_req_ready(q_req_ready), .q_resp_valid(q_resp_valid),
        .q_resp_row_physical(q_resp_row_physical),
        .pivot_req_valid(pivot_req_valid), .pivot_row_i(pivot_row_i),
        .pivot_row_j(pivot_row_j), .pivot_req_ready(pivot_req_ready),
        .pivot_done(pivot_done),
        .cfg_we(cfg_we), .cfg_p_idx(cfg_p_idx),
        .cfg_p_row_physical(cfg_p_row_physical), .cfg_ready(cfg_ready),
        .init_identity(init_identity), .init_done(init_done),
        .clk(clk), .rst_n(rst_n));

    task automatic fail(input string what);
        $fatal(1, "ATU_TB FAIL: %s (t=%0t)", what, $time);
    endtask

    task automatic query(input logic [W-1:0] logical, output logic [W-1:0] physical);
        @(negedge clk);
        q_req_valid = 1'b1;
        q_req_row_logic = logical;
        @(posedge clk);
        while (!q_req_ready) @(posedge clk);
        @(negedge clk);
        q_req_valid = 1'b0;
        if (!q_resp_valid) fail("query response missing");
        physical = q_resp_row_physical;
    endtask

    logic [W-1:0] got;
    int           cfg_accept_cycles;

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        init_identity = 1'b1;
        @(negedge clk);
        init_identity = 1'b0;
        wait (init_done);
        @(negedge clk);

        // Same cycle: swap(0,1) and cfg map[2] = 3.  The caller holds cfg_we
        // until it observes cfg_ready at a clock edge (standard valid/ready).
        pivot_req_valid = 1'b1;
        pivot_row_i = 3'd0;
        pivot_row_j = 3'd1;
        cfg_we = 1'b1;
        cfg_p_idx = 3'd2;
        cfg_p_row_physical = 3'd3;
        #1;
        if (!pivot_req_ready) fail("swap not accepted in an idle cycle");
        if (cfg_ready) fail("cfg_ready must be low while a swap uses the write port");
        @(negedge clk);
        pivot_req_valid = 1'b0;
        cfg_accept_cycles = 0;
        while (1) begin
            #1;
            if (cfg_ready) break;
            @(negedge clk);
            if (++cfg_accept_cycles > 5) fail("cfg write never accepted");
        end
        @(negedge clk);  // the edge in between performed the accepted write
        cfg_we = 1'b0;

        query(3'd0, got); if (got !== 3'd1) fail($sformatf("map[0]=%0d expected 1", got));
        query(3'd1, got); if (got !== 3'd0) fail($sformatf("map[1]=%0d expected 0", got));
        query(3'd2, got); if (got !== 3'd3) fail($sformatf("map[2]=%0d expected 3 (cfg lost)", got));
        query(3'd3, got); if (got !== 3'd3) fail($sformatf("map[3]=%0d expected 3", got));

        // A cfg write alone is accepted immediately.
        @(negedge clk);
        cfg_we = 1'b1; cfg_p_idx = 3'd5; cfg_p_row_physical = 3'd7;
        #1;
        if (!cfg_ready) fail("lone cfg write not ready");
        @(negedge clk);
        cfg_we = 1'b0;
        query(3'd5, got); if (got !== 3'd7) fail($sformatf("map[5]=%0d expected 7", got));

        $display("ATU_TB PASS");
        $finish;
    end

    initial begin
        #5000;
        $fatal(1, "ATU_TB FAIL: global timeout");
    end
endmodule
