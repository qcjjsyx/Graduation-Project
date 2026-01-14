/**
 * gcu_micro_scheduler.sv
 *
 * Strictly sequential micro-scheduler for a single node.
 * Emits one op at a time and waits for op_done before proceeding.
 */

module gcu_micro_scheduler #(
    parameter int BUF_NUM   = 2,
    parameter int TASK_W    = 128,
    parameter int TILE_SIZE = 32
) (


    // ----------------------------
    // Interface from buffer_mgr
    // ----------------------------
    input  logic [BUF_NUM-1:0]           buf_ready_for_compute,
    input  logic [BUF_NUM*TASK_W-1:0]    buf_task_flat,
    output logic [BUF_NUM-1:0]           buf_take,
    output logic [BUF_NUM-1:0]           node_compute_done,

    // ----------------------------
    // Op command interface
    // ----------------------------
    output logic                         op_valid,
    input  logic                         op_ready,
    output logic [3:0]                   op_type,
    output logic [0:0]                   op_buf_id,
    output logic [2:0]                   tile_i,
    output logic [2:0]                   tile_j,
    output logic [2:0]                   tile_k,
    output logic [31:0]                  m_dim,
    output logic [31:0]                  n_dim,
    output logic [31:0]                  k_dim,

    input  logic                         op_done_valid,
    output logic                         op_done_ready,

    // ----------------------------
    // Clock / Reset
    // ----------------------------
    input  logic                         clk,
    input  logic                         rst_n
);

    localparam int BUF_ID_W = (BUF_NUM <= 2) ? 1 : $clog2(BUF_NUM);

    // Op type encoding
    localparam logic [3:0] OP_FACT       = 4'd0;
    localparam logic [3:0] OP_TRSM_U     = 4'd1;
    localparam logic [3:0] OP_TRSM_L     = 4'd2;
    localparam logic [3:0] OP_GEMM_PIVOT = 4'd3;
    localparam logic [3:0] OP_TRSM_F12   = 4'd4;
    localparam logic [3:0] OP_TRSM_F21   = 4'd5;
    localparam logic [3:0] OP_GEMM_SCHUR = 4'd6;

    typedef enum logic [2:0] {
        S_IDLE      = 3'b000,
        S_ISSUE     = 3'b001,
        S_WAIT_DONE = 3'b010,
        S_DONE      = 3'b011,
        // Stage2 placeholders (not implemented in this version)
        S_F12       = 3'b100,
        S_F21       = 3'b101,
        S_SCHUR     = 3'b110
    } state_e;

    typedef enum logic [2:0] {
        PH_FACT  = 3'b000,
        PH_TRSM_U = 3'b001,
        PH_TRSM_L = 3'b010,
        PH_GEMM  = 3'b011,
        // Stage2 placeholders (not implemented in this version)
        PH_F12   = 3'b100,
        PH_F21   = 3'b101,
        PH_SCHUR = 3'b110
    } phase_e;

    state_e state_q, state_d;
    phase_e phase_q, phase_d;

    logic [BUF_ID_W-1:0] buf_id_q, buf_id_d;
    logic [3:0]          nt_q, nt_d;
    logic [2:0]          k_q, k_d;
    logic [2:0]          i_q, i_d;
    logic [2:0]          j_q, j_d;

    // ----------------------------
    // Helper: compute nt = ceil(pivot_dim / TILE_SIZE)
    // ----------------------------
    function automatic logic [3:0] calc_nt(input logic [31:0] pivot_dim);
        logic [31:0] tmp;
        begin
            tmp = pivot_dim + (TILE_SIZE - 1);
            calc_nt = tmp / TILE_SIZE; // constant division
        end
    endfunction

    // ----------------------------
    // Select lowest-index ready buffer
    // ----------------------------
    logic has_ready;
    logic [BUF_ID_W-1:0] sel_buf;
    always_comb begin
        int idx;
        has_ready = 1'b0;
        sel_buf   = '0;
        for (idx = 0; idx < BUF_NUM; idx++) begin
            if (!has_ready && buf_ready_for_compute[idx]) begin
                has_ready = 1'b1;
                sel_buf   = idx[BUF_ID_W-1:0];
            end
        end
    end

    // buf_take pulse when idle and any buffer is ready
    always_comb begin
        buf_take = '0;
        if (state_q == S_IDLE && has_ready) begin
            buf_take[sel_buf] = 1'b1;
        end
    end

    // node_compute_done pulse on S_DONE
    always_comb begin
        node_compute_done = '0;
        if (state_q == S_DONE) begin
            node_compute_done[buf_id_q] = 1'b1;
        end
    end

    // ----------------------------
    // Op outputs
    // ----------------------------
    always_comb begin
        op_valid = (state_q == S_ISSUE);
        op_type  = OP_FACT;
        op_buf_id = buf_id_q;
        tile_i   = '0;
        tile_j   = '0;
        tile_k   = '0;
        // First version: fixed tile size; edge tiles can be added later
        m_dim    = TILE_SIZE;
        n_dim    = TILE_SIZE;
        k_dim    = TILE_SIZE;

        if (state_q == S_ISSUE) begin
            case (phase_q)
                PH_FACT: begin
                    op_type = OP_FACT;
                    tile_i  = k_q;
                    tile_j  = k_q;
                    tile_k  = k_q;
                end
                PH_TRSM_U: begin
                    op_type = OP_TRSM_U;
                    tile_i  = k_q;
                    tile_j  = j_q;
                    tile_k  = k_q;
                end
                PH_TRSM_L: begin
                    op_type = OP_TRSM_L;
                    tile_i  = i_q;
                    tile_j  = k_q;
                    tile_k  = k_q;
                end
                PH_GEMM: begin
                    op_type = OP_GEMM_PIVOT;
                    tile_i  = i_q;
                    tile_j  = j_q;
                    tile_k  = k_q;
                end
                // Stage2 placeholder ops (not issued in this version)
                PH_F12:   op_type = OP_TRSM_F12;
                PH_F21:   op_type = OP_TRSM_F21;
                PH_SCHUR: op_type = OP_GEMM_SCHUR;
                default:  op_type = OP_FACT;
            endcase
        end
    end

    assign op_done_ready = 1'b1;

    // ----------------------------
    // Next-state / Next-counters
    // ----------------------------
    always_comb begin
        state_d = state_q;
        phase_d = phase_q;
        buf_id_d = buf_id_q;
        nt_d = nt_q;
        k_d = k_q;
        i_d = i_q;
        j_d = j_q;

        case (state_q)
            S_IDLE: begin
                if (has_ready) begin
                    logic [31:0] pivot_dim;
                    pivot_dim = buf_task_flat[sel_buf*TASK_W + 32 +: 32];
                    buf_id_d  = sel_buf;
                    nt_d      = calc_nt(pivot_dim);
                    k_d       = 3'd0;
                    i_d       = 3'd0;
                    j_d       = 3'd0;
                    phase_d   = PH_FACT;
                    state_d   = (calc_nt(pivot_dim) == 0) ? S_DONE : S_ISSUE;
                end
            end

            S_ISSUE: begin
                if (op_valid && op_ready) begin
                    state_d = S_WAIT_DONE;
                end
            end

            S_WAIT_DONE: begin
                if (op_done_valid && op_done_ready) begin
                    state_d = S_ISSUE;
                    case (phase_q)
                        PH_FACT: begin
                            if ((k_q + 1) < nt_q) begin
                                phase_d = PH_TRSM_U;
                                j_d     = k_q + 1'b1;
                            end else begin
                                // TODO: Stage2 ops for F12/F21/Schur
                                state_d = S_DONE;
                            end
                        end

                        PH_TRSM_U: begin
                            if ((j_q + 1) < nt_q) begin
                                j_d = j_q + 1'b1;
                            end else begin
                                phase_d = PH_TRSM_L;
                                i_d     = k_q + 1'b1;
                            end
                        end

                        PH_TRSM_L: begin
                            if ((i_q + 1) < nt_q) begin
                                i_d = i_q + 1'b1;
                            end else begin
                                phase_d = PH_GEMM;
                                i_d     = k_q + 1'b1;
                                j_d     = k_q + 1'b1;
                            end
                        end

                        PH_GEMM: begin
                            if ((j_q + 1) < nt_q) begin
                                j_d = j_q + 1'b1;
                            end else if ((i_q + 1) < nt_q) begin
                                i_d = i_q + 1'b1;
                                j_d = k_q + 1'b1;
                            end else begin
                                if ((k_q + 1) < nt_q) begin
                                    k_d     = k_q + 1'b1;
                                    phase_d = PH_FACT;
                                end else begin
                                    // TODO: Stage2 ops for F12/F21/Schur
                                    state_d = S_DONE;
                                end
                            end
                        end

                        default: begin
                            // Stage2 placeholders not implemented
                            state_d = S_DONE;
                        end
                    endcase
                end
            end

            S_DONE: begin
                state_d = S_IDLE;
            end

            // Stage2 placeholders: not used in this version
            S_F12:   state_d = S_DONE;
            S_F21:   state_d = S_DONE;
            S_SCHUR: state_d = S_DONE;

            default: state_d = S_IDLE;
        endcase
    end

    // ----------------------------
    // Register update
    // ----------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= S_IDLE;
            phase_q <= PH_FACT;
            buf_id_q <= '0;
            nt_q <= '0;
            k_q <= '0;
            i_q <= '0;
            j_q <= '0;
        end else begin
            state_q <= state_d;
            phase_q <= phase_d;
            buf_id_q <= buf_id_d;
            nt_q <= nt_d;
            k_q <= k_d;
            i_q <= i_d;
            j_q <= j_d;
        end
    end

endmodule
