module hpu_top #(
    parameter DATA_W    = 32,
    parameter ROW_IDX_W = 16,
    parameter MAX_ELEMS  = 256
)
(

    // 控制接口：启动一次新的 pivot 搜索
    input  logic                   pivot_start,     // 单拍脉冲：本轮开始
    output logic                   pivot_busy,      // 非 IDLE 状态即为 1

    // 候选数据流：来自 Panel/控制器
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [DATA_W-1:0]      in_value,
    input  logic [ROW_IDX_W-1:0]   in_row_logical,
    input  logic                   in_last,         // 标记本轮最后一个元素

    // 结果输出
    output logic                   pivot_valid,
    input  logic                   pivot_ready,
    output logic [ROW_IDX_W-1:0]   pivot_row,
    output logic [DATA_W-1:0]      pivot_value,
    output logic                   pivot_fail,       // 当前版本: num_elems==0 时为 1


    input logic clk,
    input logic rst_n
);

    typedef enum logic [3:0] {
            S_IDLE   = 4'b0001,
            S_LOAD   = 4'b0010,
            S_SELECT = 4'b0100,
            S_OUT    = 4'b1000
    } state_e;

    state_e state, state_n;

    logic signed [DATA_W-1:0]  cand_val [0:MAX_ELEMS-1];
    logic        [ROW_IDX_W-1:0] cand_row [0:MAX_ELEMS-1];

    // 本轮已经收集的元素个数
    logic [$clog2(MAX_ELEMS):0] elem_count;

    // from core
    logic [ROW_IDX_W-1:0]     core_pivot_row;
    logic signed [DATA_W-1:0] core_pivot_value;
    logic                     core_pivot_valid;

    // 输出寄存
    logic                     pivot_valid_r;
    logic [ROW_IDX_W-1:0]     pivot_row_r;
    logic [DATA_W-1:0]        pivot_value_r;
    logic                     pivot_fail_r;


    assign pivot_valid = pivot_valid_r;
    assign pivot_row   = pivot_row_r;
    assign pivot_value = pivot_value_r;
    assign pivot_fail  = pivot_fail_r;

    assign pivot_busy  = (state != S_IDLE);

    // in_ready：只有在 LOAD 状态才接收数据
    assign in_ready = (state == S_LOAD);

    // 状态机转移
    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE: begin
                if (pivot_start) begin
                    state_n = S_LOAD;
                end
            end

            S_LOAD: begin
                // 在成功握手并收到最后一个元素后，转入 SELECT
                if (in_valid && in_ready && (in_last || (elem_count == MAX_ELEMS-1))) begin
                    state_n = S_SELECT;
                end
            end

            S_SELECT: begin
                // 仅停留 1 拍，用于锁存 core 结果
                state_n = S_OUT;
            end

            S_OUT: begin
                if (pivot_valid_r && pivot_ready) begin
                    state_n = S_IDLE;
                end
            end

            default: state_n = S_IDLE;
        endcase
    end

    integer k;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            elem_count     <= '0;
            pivot_valid_r  <= 1'b0;
            pivot_row_r    <= '0;
            pivot_value_r  <= '0;
            pivot_fail_r   <= 1'b0;
            // cand_* 在复位时可以保持 X/0，不影响功能（因为会被 elem_count 控制）
        end else begin
            state <= state_n;

            case (state)
                S_IDLE: begin
                    pivot_valid_r <= 1'b0;
                    pivot_fail_r  <= 1'b0;
                    elem_count    <= '0;
                end

                S_LOAD: begin
                    if (in_valid && in_ready) begin
                        // 将当前候选写入 buffer
                        cand_val[elem_count] <= in_value;
                        cand_row[elem_count] <= in_row_logical;
                        elem_count           <= elem_count + 1'b1;
                    end
                end

                S_SELECT: begin
                    // 在 SELECT 状态的这个时钟边沿，锁存 core 的结果
                    pivot_row_r    <= core_pivot_row;
                    pivot_value_r  <= core_pivot_value;
                    pivot_valid_r  <= core_pivot_valid;
                    pivot_fail_r   <= ~core_pivot_valid;  // 当前版本: 无候选则 fail=1
                end

                S_OUT: begin
                    if (pivot_valid_r && pivot_ready) begin
                        // pivot_valid_r <= 1'b0;
                    end
                end

                default: ;
            endcase
        end
    end

    // 实例化树形选主元核心：完全组合
    hpu_core_tree #(
        .DATA_W    (DATA_W),
        .ROW_IDX_W (ROW_IDX_W),
        .MAX_ELEMS (MAX_ELEMS)
    ) u_core_tree (
        .cand_val     (cand_val),
        .cand_row     (cand_row),
        .num_elems    (elem_count),      // elem_count 在 S_SELECT 时已经是 N
        .pivot_row    (core_pivot_row),
        .pivot_value  (core_pivot_value),
        .pivot_valid  (core_pivot_valid)
    );



endmodule