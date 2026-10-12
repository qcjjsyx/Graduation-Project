`timescale 1ns/1ps
// Self-checking HPU testbench (regression for B02 + basic argmax contract).
//
// Checks, with the real hpu_core_tree:
//   1. ready held at 1: every request produces exactly ONE accepted result;
//   2. argmax(|value|) with mixed signs, pivot in first/middle/last position;
//   3. equal |value|: the earliest candidate wins (deterministic tie-break);
//   4. output back-pressure: valid/row/value stay stable while ready=0 and the
//      result is accepted exactly once when ready rises;
//   5. back-to-back requests after reset.
// Exits with $fatal on the first mismatch, prints "HPU_TB PASS" otherwise.
//
//   iverilog -g2012 -s tb_hpu_handshake -o /tmp/tb_hpu \
//     HPU/hpu_cmp_node.sv HPU/hpu_core_tree.sv HPU/hpu_top.sv HPU/tb_code/tb_hpu_handshake.sv
//   vvp /tmp/tb_hpu
module tb_hpu_handshake;
    localparam int DATA_W    = 32;
    localparam int ROW_IDX_W = 8;
    localparam int MAX_ELEMS = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic                 pivot_start = 1'b0;
    logic                 pivot_busy;
    logic                 in_valid = 1'b0;
    logic                 in_ready;
    logic [DATA_W-1:0]    in_value = '0;
    logic [ROW_IDX_W-1:0] in_row_logical = '0;
    logic                 in_last = 1'b0;
    logic                 pivot_valid;
    logic                 pivot_ready = 1'b1;
    logic [ROW_IDX_W-1:0] pivot_row;
    logic [DATA_W-1:0]    pivot_value;
    logic                 pivot_fail;

    hpu_top #(.DATA_W(DATA_W), .ROW_IDX_W(ROW_IDX_W), .MAX_ELEMS(MAX_ELEMS)) dut (
        .pivot_start(pivot_start), .pivot_busy(pivot_busy),
        .in_valid(in_valid), .in_ready(in_ready), .in_value(in_value),
        .in_row_logical(in_row_logical), .in_last(in_last),
        .pivot_valid(pivot_valid), .pivot_ready(pivot_ready),
        .pivot_row(pivot_row), .pivot_value(pivot_value), .pivot_fail(pivot_fail),
        .clk(clk), .rst_n(rst_n));

    // Count every output handshake and remember the last accepted result.
    int unsigned          accepted = 0;
    logic [ROW_IDX_W-1:0] last_row;
    logic [DATA_W-1:0]    last_value;
    logic                 last_fail;
    always @(posedge clk) begin
        if (rst_n && pivot_valid && pivot_ready) begin
            accepted   <= accepted + 1;
            last_row   <= pivot_row;
            last_value <= pivot_value;
            last_fail  <= pivot_fail;
        end
    end

    task automatic fail(input string what);
        $fatal(1, "HPU_TB FAIL: %s (t=%0t)", what, $time);
    endtask

    // Candidate staging arrays (module scope: Icarus does not support
    // unpacked-array task ports).  v[k] belongs to logical row r[k].
    logic signed [DATA_W-1:0] v[MAX_ELEMS];
    logic [ROW_IDX_W-1:0]     r[MAX_ELEMS];

    // Stream the first `count` staged candidates as one segment.
    task automatic run_segment(input int count);
        @(negedge clk);
        pivot_start = 1'b1;
        @(negedge clk);
        pivot_start = 1'b0;
        for (int k = 0; k < count; k++) begin
            in_valid       = 1'b1;
            in_value       = v[k];
            in_row_logical = r[k];
            in_last        = (k == count - 1);
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
        end
        in_valid = 1'b0;
        in_last  = 1'b0;
    endtask

    task automatic expect_one_result(input int unsigned base_count,
                                     input logic [ROW_IDX_W-1:0] row,
                                     input logic signed [DATA_W-1:0] value,
                                     input string label);
        int guard;
        guard = 0;
        while (accepted == base_count && guard < 50) begin
            @(posedge clk);
            guard++;
        end
        repeat (5) @(posedge clk);  // a duplicate handshake would show up here
        #1;
        if (accepted != base_count + 1)
            fail($sformatf("%s: expected 1 accepted result, got %0d", label, accepted - base_count));
        if (last_fail) fail($sformatf("%s: unexpected pivot_fail", label));
        if (last_row !== row)
            fail($sformatf("%s: row %0d expected %0d", label, last_row, row));
        if ($signed(last_value) !== value)
            fail($sformatf("%s: value %0d expected %0d", label, $signed(last_value), value));
        if (pivot_busy) fail($sformatf("%s: HPU still busy after result", label));
    endtask

    int unsigned              base_count;

    initial begin
        for (int k = 0; k < MAX_ELEMS; k++) begin v[k] = '0; r[k] = '0; end
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // 1+2: ready held high, max |value| is a negative entry in the middle.
        v[0] = 3;  v[1] = -9; v[2] = 7;  v[3] = 2;
        r[0] = 10; r[1] = 11; r[2] = 12; r[3] = 13;
        base_count = accepted;
        run_segment(4);
        expect_one_result(base_count, 8'd11, -9, "middle negative pivot");

        // pivot in the last position, single-candidate segment, first position.
        v[0] = 1; v[1] = -2; v[2] = 3; v[3] = -4; v[4] = 5;
        r[0] = 0; r[1] = 1;  r[2] = 2; r[3] = 3;  r[4] = 4;
        base_count = accepted;
        run_segment(5);
        expect_one_result(base_count, 8'd4, 5, "last-position pivot");

        v[0] = -6; r[0] = 42;
        base_count = accepted;
        run_segment(1);
        expect_one_result(base_count, 8'd42, -6, "single candidate");

        // 3: tie on |value| -> earliest candidate.
        v[0] = -5; v[1] = 5; v[2] = 1;
        r[0] = 7;  r[1] = 3; r[2] = 1;
        base_count = accepted;
        run_segment(3);
        expect_one_result(base_count, 8'd7, -5, "tie-break earliest");

        // 4: output back-pressure.
        pivot_ready = 1'b0;
        v[0] = 2; v[1] = 8; r[0] = 20; r[1] = 21;
        base_count = accepted;
        run_segment(2);
        while (!pivot_valid) @(posedge clk);
        begin
            logic [ROW_IDX_W-1:0] held_row;
            logic [DATA_W-1:0]    held_value;
            #1;
            held_row = pivot_row;
            held_value = pivot_value;
            repeat (4) begin
                @(posedge clk);
                #1;
                if (!pivot_valid) fail("valid dropped while ready=0");
                if (pivot_row !== held_row || pivot_value !== held_value)
                    fail("output changed while ready=0");
            end
        end
        if (accepted != base_count) fail("result accepted while ready=0");
        @(negedge clk);
        pivot_ready = 1'b1;
        expect_one_result(base_count, 8'd21, 8, "back-pressure release");

        // 5: reset in the middle of nothing, then two back-to-back requests.
        rst_n = 1'b0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        v[0] = 1; v[1] = -3; r[0] = 0; r[1] = 1;
        base_count = accepted;
        run_segment(2);
        expect_one_result(base_count, 8'd1, -3, "after reset 1");
        v[0] = 4; v[1] = 1; r[0] = 5; r[1] = 6;
        base_count = accepted;
        run_segment(2);
        expect_one_result(base_count, 8'd5, 4, "after reset 2");

        $display("HPU_TB PASS (%0d results accepted)", accepted);
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "HPU_TB FAIL: global timeout");
    end
endmodule
