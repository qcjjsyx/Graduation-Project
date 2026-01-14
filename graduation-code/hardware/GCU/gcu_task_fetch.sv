/**
 * gcu_task_fetch.sv
 *
 * Fetch Node_Task descriptors from DDR and hand off to buffer_mgr via ready/valid.
 * Single-entry buffer, one outstanding request at a time.
 */

module gcu_task_fetch #(
    parameter int ADDR_W     = 32,
    parameter int TASK_W     = 128,
    parameter int TASK_BYTES = 16
) (
    // ----------------------------
    // Control registers
    // ----------------------------
    input  logic                   start,
    input  logic [ADDR_W-1:0]      queue_base_addr,
    input  logic [31:0]            task_count,
    output logic                   busy,
    output logic                   done,

    // ----------------------------
    // DDR read interface (abstract)
    // ----------------------------
    output logic                  rd_req_valid,
    input  logic                  rd_req_ready,
    output logic [ADDR_W-1:0]      rd_req_addr,
    input  logic                  rd_rsp_valid,
    output logic                  rd_rsp_ready,
    input  logic [TASK_W-1:0]      rd_rsp_data,

    // ----------------------------
    // Output to buffer_mgr
    // ----------------------------
    output logic                  task_valid,
    input  logic                  task_ready,
    output logic [TASK_W-1:0]      task_out,

    // ----------------------------
    // Clock / Reset
    // ----------------------------
    input  logic                  clk,
    input  logic                  rst_n
);

    typedef enum logic [2:0] {
        S_IDLE     = 3'b001,
        S_REQ      = 3'b010,
        S_WAIT_RSP = 3'b011,
        S_OUTPUT   = 3'b100,
        S_DONE     = 3'b101
    } state_e;

    state_e             state_q, state_d;
    logic [ADDR_W-1:0]  addr_q, addr_d;
    logic [31:0]        remaining_q, remaining_d;

    logic [TASK_W-1:0]  task_buf_q, task_buf_d;
    logic              task_buf_valid_q, task_buf_valid_d;

    // ----------------------------
    // Outputs
    // ----------------------------
    assign task_valid  = task_buf_valid_q;
    assign task_out    = task_buf_q;
    assign rd_req_addr = addr_q;

    // ----------------------------
    // Default combinational outputs
    // ----------------------------
    always_comb begin
        rd_req_valid = 1'b0;
        rd_rsp_ready = 1'b0;

        case (state_q)
            S_REQ: begin
                rd_req_valid = (remaining_q != 0);
            end
            S_WAIT_RSP: begin
                rd_rsp_ready = !task_buf_valid_q;
            end
            default: begin
                rd_req_valid = 1'b0;
                rd_rsp_ready = 1'b0;
            end
        endcase
    end

    // ----------------------------
    // Next-state / Next-data
    // ----------------------------
    always_comb begin
        // defaults
        state_d          = state_q;
        addr_d           = addr_q;
        remaining_d      = remaining_q;
        task_buf_d       = task_buf_q;
        task_buf_valid_d = task_buf_valid_q;

        case (state_q)
            S_IDLE: begin
                if (start) begin
                    addr_d           = queue_base_addr;
                    remaining_d      = task_count;
                    task_buf_valid_d = 1'b0;
                    state_d          = (task_count == 0) ? S_DONE : S_REQ;
                end
            end

            S_REQ: begin
                if (remaining_q == 0) begin
                    state_d = S_DONE;
                end else if (rd_req_valid && rd_req_ready) begin
                    addr_d      = addr_q + TASK_BYTES[ADDR_W-1:0];
                    remaining_d = remaining_q - 1'b1;
                    state_d     = S_WAIT_RSP;
                end
            end

            S_WAIT_RSP: begin
                if (rd_rsp_valid && rd_rsp_ready) begin
                    task_buf_d       = rd_rsp_data;
                    task_buf_valid_d = 1'b1;
                    state_d          = S_OUTPUT;
                end
            end

            S_OUTPUT: begin
                if (task_valid && task_ready) begin
                    task_buf_valid_d = 1'b0;
                    state_d          = (remaining_q == 0) ? S_DONE : S_REQ;
                end
            end

            S_DONE: begin
                // if (start) begin
                //     addr_d           = queue_base_addr;
                //     remaining_d      = task_count;
                //     task_buf_valid_d = 1'b0;
                //     state_d          = (task_count == 0) ? S_DONE : S_REQ;
                // end
                state_d = S_IDLE;
            end

            default: state_d = S_IDLE;
        endcase
    end

    // ----------------------------
    // State / Register update
    // ----------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q          <= S_IDLE;
            addr_q           <= '0;
            remaining_q      <= '0;
            task_buf_q       <= '0;
            task_buf_valid_q <= 1'b0;
        end else begin
            state_q          <= state_d;
            addr_q           <= addr_d;
            remaining_q      <= remaining_d;
            task_buf_q       <= task_buf_d;
            task_buf_valid_q <= task_buf_valid_d;
        end
    end

    // ----------------------------
    // Status outputs
    // ----------------------------
    always_comb begin
        busy = (state_q != S_IDLE);
        done = (state_q == S_DONE);
    end

endmodule
