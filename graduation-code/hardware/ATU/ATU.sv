/* * ATU.v
 * 
 * This module is a placeholder for the ATU (Address Translation Unit) functionality.
 * 这是同步版本的ATU模块。
 * 20260305 该模块是需要修改的，最后应当作为访问DDR的地址转换单元，提供行号映射功能。应当与DDR 控制器配合使用，提供行号转换服务。
 * 目前未修改
 */

module ATU #(
    parameter ROW_IDX_W = 8
) 
(

    // 查询接口
    input logic  q_req_valid,
    input logic [ROW_IDX_W-1:0] q_req_row_logic, // 逻辑行号 max = 256行
    output logic  q_req_ready,
    output logic  q_resp_valid,
    output logic [ROW_IDX_W-1:0] q_resp_row_physical,// 物理行号

    // pivot 更新接口
    input logic  pivot_req_valid,
    input logic [ROW_IDX_W-1:0] pivot_row_i, // 行i
    input logic [ROW_IDX_W-1:0] pivot_row_j, // 行j
    output logic  pivot_req_ready,
    output logic  pivot_done,

    //初始化/配置接口
    input logic cfg_we,
    input logic [ROW_IDX_W-1:0] cfg_p_idx,
    input logic [ROW_IDX_W-1:0] cfg_p_row_physical,
    output logic cfg_ready,

    input logic init_identity, // 初始化为单位映射
    output logic init_done,

    // 时钟 复位信号
    input  logic clk,
    input logic rst_n

);
localparam NUM_ROWS = 1<<ROW_IDX_W;

// atu_table pvec[L] = P
logic [ROW_IDX_W-1:0] Pvec [0:NUM_ROWS-1];

//=====================================================================
// 1) 初始化逻辑
//=====================================================================
typedef enum logic[1:0] { 
    INIT_IDLE = 2'b01,
    INIT_RUN = 2'b10
} init_st_e;

init_st_e init_state, init_state_n;
logic [ROW_IDX_W-1:0] init_counter, init_counter_n; // 初始化计数器
logic init_start;
logic init_identity_d;
logic init_done_n;
logic init_wen;
logic [ROW_IDX_W-1:0] init_waddr;
logic [ROW_IDX_W-1:0] init_wdata;

// 上升沿检测
always_ff @(posedge clk) begin
    if (!rst_n) begin
        init_identity_d <= 1'b0;
    end else begin
        init_identity_d <= init_identity;
    end
end

assign init_start  = init_identity && ~init_identity_d;

// Moore/Mealy 分离：组合逻辑计算下一状态与输出
always_comb begin
    init_state_n   = init_state;
    init_counter_n = init_counter;
    init_done_n    = 1'b0;
    init_wen       = 1'b0;
    init_waddr     = init_counter;
    init_wdata     = init_counter;

    case (init_state)
        INIT_IDLE: begin
            if (init_start) begin
                init_state_n   = INIT_RUN;
                init_counter_n = '0;
            end
        end

        INIT_RUN: begin
            init_wen   = 1'b1; // 写入单位映射
            init_waddr = init_counter;
            init_wdata = init_counter;
            if (init_counter == NUM_ROWS-1) begin
                init_state_n = INIT_IDLE;
                init_done_n  = 1'b1;
            end else begin
                init_counter_n = init_counter + 1;
            end
        end

        default: begin
            init_state_n = INIT_IDLE;
        end
    endcase
end

// 时序逻辑：寄存状态与输出
always_ff @(posedge clk) begin
    if (!rst_n) begin
        init_state   <= INIT_IDLE;
        init_counter <= '0;
        init_done    <= 1'b0;
    end else begin
        init_state   <= init_state_n;
        init_counter <= init_counter_n;
        init_done    <= init_done_n;
    end
end

//=====================================================================
// 2) 查询逻辑
// 仅在未进行初始化或更新写操作时接受查询，避免读到交换过程中的数据。
//=====================================================================
assign q_req_ready = (init_state == INIT_IDLE) && ~(init_wen || pivot_req_valid || cfg_we);
logic [ROW_IDX_W-1:0] q_req_row_logic_r;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        q_req_row_logic_r <= '0;
        q_resp_valid <= 1'b0;
    end else begin
        if (q_req_valid && q_req_ready) begin
            q_req_row_logic_r <= q_req_row_logic;
            q_resp_valid <= 1'b1;
        end else begin
            q_resp_valid <= 1'b0;
        end
    end

end

always_comb begin : qure
    q_resp_row_physical = Pvec[q_req_row_logic_r];
end


//=====================================================================
// 3) pivot 更新逻辑
// 初始化阶段不接受pivot请求
// 非初始化时，只要valid=1，就需要进行i，j行交换
// done 在交换完成的下一个时钟周期产生
//=====================================================================
logic pivot_done_d;
assign pivot_req_ready = (init_state == INIT_IDLE);

always_ff @(posedge clk) begin
    if (!rst_n) begin
        pivot_done_d <= 1'b0;
    end else begin
        pivot_done_d <= pivot_req_valid && pivot_req_ready;
    end
end
assign pivot_done = pivot_done_d;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        // do nothing
    end else if (init_wen) begin
        Pvec[init_waddr] <= init_wdata;
    end else if (init_state == INIT_IDLE) begin
        if (pivot_req_valid && pivot_req_ready) begin
            // 交换 Pvec[pivot_row_i] 和 Pvec[pivot_row_j]
            Pvec[pivot_row_i] <= Pvec[pivot_row_j];
            Pvec[pivot_row_j] <= Pvec[pivot_row_i];
        end else if(cfg_we) begin
            // 外部写入
            Pvec[cfg_p_idx] <= cfg_p_row_physical;
        end else begin
            // do nothing
        end
    end else begin
        // do nothing
    end
end     

assign cfg_ready = (init_state == INIT_IDLE);

endmodule
