`timescale 1ns/1ps

module gcu_micro_scheduler_tb;

    localparam int BUF_NUM   = 2;
    localparam int TASK_W    = 128;
    localparam int TILE_SIZE = 32;
    localparam int NT        = 8; // pivot_dim=256 => nt=8
    localparam int DONE_LAT  = 2;

    // Op type encoding
    localparam logic [3:0] OP_FACT       = 4'd0;
    localparam logic [3:0] OP_TRSM_U     = 4'd1;
    localparam logic [3:0] OP_TRSM_L     = 4'd2;
    localparam logic [3:0] OP_GEMM_PIVOT = 4'd3;

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
    logic [BUF_NUM-1:0]           buf_ready_for_compute;
    logic                         ready0_reg;
    logic                         ready0_set;
    logic [BUF_NUM*TASK_W-1:0]    buf_task_flat;
    logic [BUF_NUM-1:0]           buf_take;
    logic [BUF_NUM-1:0]           node_compute_done;

    logic                         op_valid;
    logic                         op_ready;
    logic [3:0]                   op_type;
    logic [0:0]                   op_buf_id;
    logic [2:0]                   tile_i, tile_j, tile_k;
    logic [31:0]                  m_dim, n_dim, k_dim;
    logic                         op_done_valid;
    logic                         op_done_ready;

    // ----------------------------
    // DUT instance
    // ----------------------------
    gcu_micro_scheduler #(
        // .BUF_NUM(BUF_NUM),
        // .TASK_W(TASK_W),
        // .TILE_SIZE(TILE_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .buf_ready_for_compute(buf_ready_for_compute),
        .buf_task_flat(buf_task_flat),
        .buf_take(buf_take),
        .node_compute_done(node_compute_done),
        .op_valid(op_valid),
        .op_ready(op_ready),
        .op_type(op_type),
        .op_buf_id(op_buf_id),
        .tile_i(tile_i),
        .tile_j(tile_j),
        .tile_k(tile_k),
        .m_dim(m_dim),
        .n_dim(n_dim),
        .k_dim(k_dim),
        .op_done_valid(op_done_valid),
        .op_done_ready(op_done_ready)
    );

    assign buf_ready_for_compute = {1'b0, ready0_reg};

    // ----------------------------
    // Simple op exec model: fixed done latency
    // ----------------------------
    logic [DONE_LAT:0] done_pipe;

    assign op_ready = 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_pipe <= '0;
        end else begin
            done_pipe <= {done_pipe[DONE_LAT-1:0], 1'b0};
            if (op_valid && op_ready) begin
                done_pipe[0] <= 1'b1;
            end
        end
    end

    assign op_done_valid = done_pipe[DONE_LAT];

    // ----------------------------
    // Expected sequence model
    // ----------------------------
    typedef enum logic [1:0] {
        EXP_FACT  = 2'd0,
        EXP_TRSM_U = 2'd1,
        EXP_TRSM_L = 2'd2,
        EXP_GEMM  = 2'd3
    } exp_phase_e;

    exp_phase_e exp_phase;
    int exp_k, exp_i, exp_j;
    int op_count;
    bit started;
    bit exp_done;
    bit last_done_seen;
    bit done_pulse_seen;

    function automatic int expected_op_count(input int nt);
        int k;
        int t;
        begin
            expected_op_count = 0;
            for (k = 0; k < nt; k++) begin
                t = nt - 1 - k;
                expected_op_count += 1 + t + t + (t * t);
            end
        end
    endfunction

    task automatic advance_expected;
        begin
            case (exp_phase)
                EXP_FACT: begin
                    if ((exp_k + 1) < NT) begin
                        exp_phase = EXP_TRSM_U;
                        exp_j     = exp_k + 1;
                    end else begin
                        exp_done = 1'b1;
                    end
                end

                EXP_TRSM_U: begin
                    if ((exp_j + 1) < NT) begin
                        exp_j = exp_j + 1;
                    end else begin
                        exp_phase = EXP_TRSM_L;
                        exp_i     = exp_k + 1;
                    end
                end

                EXP_TRSM_L: begin
                    if ((exp_i + 1) < NT) begin
                        exp_i = exp_i + 1;
                    end else begin
                        exp_phase = EXP_GEMM;
                        exp_i     = exp_k + 1;
                        exp_j     = exp_k + 1;
                    end
                end

                EXP_GEMM: begin
                    if ((exp_j + 1) < NT) begin
                        exp_j = exp_j + 1;
                    end else if ((exp_i + 1) < NT) begin
                        exp_i = exp_i + 1;
                        exp_j = exp_k + 1;
                    end else begin
                        if ((exp_k + 1) < NT) begin
                            exp_k     = exp_k + 1;
                            exp_phase = EXP_FACT;
                        end else begin
                            exp_done = 1'b1;
                        end
                    end
                end

                default: exp_done = 1'b1;
            endcase
        end
    endtask

    // ----------------------------
    // Monitor and checks
    // ----------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exp_phase     <= EXP_FACT;
            exp_k         <= 0;
            exp_i         <= 0;
            exp_j         <= 0;
            op_count      <= 0;
            started       <= 1'b0;
            exp_done      <= 1'b0;
            last_done_seen <= 1'b0;
            done_pulse_seen <= 1'b0;
            ready0_reg     <= 1'b0;
        end else begin
            if (buf_take[0] || node_compute_done[0]) begin
                ready0_reg <= 1'b0;
            end else if (ready0_set) begin
                ready0_reg <= 1'b1;
            end

            if (buf_take[0]) begin
                if (!buf_ready_for_compute[0]) begin
                    $fatal("[%0t] buf_take asserted while not ready", $time);
                end
                started   <= 1'b1;
                exp_phase <= EXP_FACT;
                exp_k     <= 0;
                exp_i     <= 0;
                exp_j     <= 0;
                exp_done  <= 1'b0;
            end

            if (op_valid && op_ready) begin
                if (!started) begin
                    $fatal("[%0t] op issued before buf_take", $time);
                end
                if (exp_done) begin
                    $fatal("[%0t] extra op issued after expected done", $time);
                end

                case (exp_phase)
                    EXP_FACT: begin
                        if (op_type !== OP_FACT || tile_k !== exp_k[2:0] || tile_i !== exp_k[2:0] || tile_j !== exp_k[2:0]) begin
                            $fatal("[%0t] OP_FACT mismatch: got type=%0d k=%0d i=%0d j=%0d",
                                   $time, op_type, tile_k, tile_i, tile_j);
                        end
                    end
                    EXP_TRSM_U: begin
                        if (op_type !== OP_TRSM_U || tile_k !== exp_k[2:0] || tile_i !== exp_k[2:0] || tile_j !== exp_j[2:0]) begin
                            $fatal("[%0t] OP_TRSM_U mismatch: got type=%0d k=%0d i=%0d j=%0d",
                                   $time, op_type, tile_k, tile_i, tile_j);
                        end
                    end
                    EXP_TRSM_L: begin
                        if (op_type !== OP_TRSM_L || tile_k !== exp_k[2:0] || tile_i !== exp_i[2:0] || tile_j !== exp_k[2:0]) begin
                            $fatal("[%0t] OP_TRSM_L mismatch: got type=%0d k=%0d i=%0d j=%0d",
                                   $time, op_type, tile_k, tile_i, tile_j);
                        end
                    end
                    EXP_GEMM: begin
                        if (op_type !== OP_GEMM_PIVOT || tile_k !== exp_k[2:0] || tile_i !== exp_i[2:0] || tile_j !== exp_j[2:0]) begin
                            $fatal("[%0t] OP_GEMM_PIVOT mismatch: got type=%0d k=%0d i=%0d j=%0d",
                                   $time, op_type, tile_k, tile_i, tile_j);
                        end
                    end
                    default: begin
                        $fatal("[%0t] Unexpected phase", $time);
                    end
                endcase

                op_count <= op_count + 1;
                advance_expected();
            end

            if (exp_done && op_done_valid) begin
                last_done_seen <= 1'b1;
            end

            if (node_compute_done[0]) begin
                if (!last_done_seen) begin
                    $fatal("[%0t] node_compute_done asserted before last op done", $time);
                end
                done_pulse_seen <= 1'b1;
            end
        end
    end

    // ----------------------------
    // Test sequence
    // ----------------------------
    initial begin
        rst_n = 1'b0;
        ready0_set = 1'b0;
        buf_task_flat = '0;

        // task0: total_dim=256, pivot_dim=256
        buf_task_flat[0*TASK_W +: TASK_W] = '0;
        buf_task_flat[0*TASK_W + 0 +: 32]  = 32'd256;
        buf_task_flat[0*TASK_W + 32 +: 32] = 32'd256;

        #100;
        rst_n = 1'b1;

        // backpressure: ready low initially
        #100;
        @(negedge clk);
        ready0_set = 1'b1;
        @(negedge clk);
        ready0_set = 1'b0;

        // wait for done
        wait (done_pulse_seen == 1'b1);
        repeat (3) @(negedge clk);

        if (op_count != expected_op_count(NT)) begin
            $fatal("[%0t] op_count mismatch: got %0d expected %0d", $time, op_count, expected_op_count(NT));
        end

        if (op_valid) begin
            $fatal("[%0t] op_valid still asserted after done", $time);
        end

        $display("ALL PASSED");
        #20;
        $finish;
    end

endmodule
