module gcu_buffer_mgr #(
    parameter int BUFFER_NUM     = 2,
    parameter int TASK_W         = 128,

    // 简化版：从 task word 里切片得到 DMA 参数
    parameter int FRONT_ADDR_W   = 4,
    parameter int FRONT_DIM_W    = 4,
    parameter int FRONT_ADDR_LSB = 0,   // task_q[..][FRONT_ADDR_LSB +: FRONT_ADDR_W]
    parameter int FRONT_DIM_LSB  = 4    // task_q[..][FRONT_DIM_LSB  +: FRONT_DIM_W]
) (


    // ----------------------------
    // Task input (from task_fetch)
    // ----------------------------
    input  logic                          task_valid,
    output logic                          task_ready,
    input  logic [TASK_W-1:0]             task_in,
    input  logic                          front_ready_for_task,

    // ----------------------------
    // Front DMA load interface
    // ----------------------------
    output logic [BUFFER_NUM-1:0]         front_load_req,                 // 1-cycle pulse
    output logic [BUFFER_NUM*FRONT_ADDR_W-1:0] front_load_addr,           // packed: [buf][addr]
    output logic [BUFFER_NUM*FRONT_DIM_W-1:0]  front_load_dim,            // packed: [buf][dim]
    input  logic [BUFFER_NUM-1:0]         front_load_done,                // pulse/level

    // ----------------------------
    // Interface to micro_scheduler
    // ----------------------------
    output logic [BUFFER_NUM-1:0]         buf_ready_for_compute,           // state==READY
    input  logic [BUFFER_NUM-1:0]         buf_take,                        // READY->PROCESSING
    input  logic [BUFFER_NUM-1:0]         node_compute_done,               // PROCESSING->WRITEBACK
    input  logic [BUFFER_NUM-1:0]         writeback_done,                  // WRITEBACK->IDLE

    // ----------------------------
    // Export per-buffer task
    // ----------------------------
    output logic [BUFFER_NUM*TASK_W-1:0]       buf_task,                      // packed: [buf][task]
    output logic [BUFFER_NUM-1:0]         buf_busy,

    // ----------------------------
    // Clock / Reset
    // ----------------------------
    input  logic                          clk,
    input  logic                          rst_n
);

    // ----------------------------
    // Buffer State
    // ----------------------------
    typedef enum logic [4:0] {
        BS_IDLE       = 5'b00001,
        BS_LOADING    = 5'b00010,
        BS_READY      = 5'b00100,
        BS_PROCESSING = 5'b01000,
        BS_WRITEBACK  = 5'b10000
    } buf_state_e;

    buf_state_e           state_q [BUFFER_NUM-1:0];
    buf_state_e           state_d [BUFFER_NUM-1:0];
    logic [TASK_W-1:0]    task_q  [BUFFER_NUM-1:0];
    logic [TASK_W-1:0]    task_d  [BUFFER_NUM-1:0];

    // integer i;

    // ----------------------------
    // 输出派生：从 state_q/task_q 组合派生
    // ----------------------------
    genvar g;
    generate
        for (g = 0; g < BUFFER_NUM; g++) begin : gen_out
            assign buf_task[g*TASK_W +: TASK_W] = task_q[g];
            assign buf_ready_for_compute[g] = (state_q[g] == BS_READY);
            assign buf_busy[g]              = (state_q[g] != BS_IDLE);

            // 只从 task_q 派生 DMA 参数（packed 扁平化，避免网表端口被拆名）
            assign front_load_addr[g*FRONT_ADDR_W +: FRONT_ADDR_W] = task_q[g][FRONT_ADDR_LSB +: FRONT_ADDR_W];
            assign front_load_dim[g*FRONT_DIM_W  +: FRONT_DIM_W ]  = task_q[g][FRONT_DIM_LSB  +: FRONT_DIM_W ];
        end
    endgenerate

    // ----------------------------
    // Allocate: choose lowest-index IDLE buffer
    // ----------------------------
    logic has_idle;
    logic [$clog2(BUFFER_NUM)-1:0] alloc_buf;
    logic [$clog2(BUFFER_NUM)-1:0] alloc_buf_q; // registered for delayed req
    logic accept_task_q;                        // registered handshake

    // 更新alloc_buf组合逻辑
    always_comb begin
        int i;
        has_idle  = 1'b0;
        alloc_buf = '0;
        for (i = 0; i < BUFFER_NUM; i++) begin
            if (!has_idle && (state_q[i] == BS_IDLE)) begin
                has_idle  = 1'b1;
                alloc_buf = i[$clog2(BUFFER_NUM)-1:0];
            end
        end
    end

    assign task_ready  = has_idle && front_ready_for_task;
    logic accept_task;
    assign accept_task = task_valid && task_ready;

    // ----------------------------
    // Next-state / Next-task（拆开：纯组合）
    // ----------------------------
    always_comb begin
        int i;
        // 默认保持
        for (i = 0; i < BUFFER_NUM; i++) begin
            state_d[i] = state_q[i];
            task_d[i]  = task_q[i];
        end

        // 1) 接收新任务：IDLE -> LOADING，并把 task_in 绑定到 alloc_buf
        if (accept_task) begin
            state_d[alloc_buf] = BS_LOADING;
            task_d[alloc_buf]  = task_in;
        end

        // 2) LOADING -> READY
        for (i = 0; i < BUFFER_NUM; i++) begin
            if ((state_q[i] == BS_LOADING) && front_load_done[i]) begin
                state_d[i] = BS_READY;
            end
        end

        // 3) READY -> PROCESSING
        for (i = 0; i < BUFFER_NUM; i++) begin
            if ((state_q[i] == BS_READY) && buf_take[i]) begin
                state_d[i] = BS_PROCESSING;
            end
        end

        // 4) PROCESSING -> WRITEBACK
        for (i = 0; i < BUFFER_NUM; i++) begin
            if ((state_q[i] == BS_PROCESSING) && node_compute_done[i]) begin
                state_d[i] = BS_WRITEBACK;
            end
        end

        // 5) WRITEBACK -> IDLE（释放 buffer，同时清空 task_q）
        for (i = 0; i < BUFFER_NUM; i++) begin
            if ((state_q[i] == BS_WRITEBACK) && writeback_done[i]) begin
                state_d[i] = BS_IDLE;
                task_d[i]  = '0;
            end
        end
    end

    // ----------------------------
    // 寄存器更新：state_q/task_q
    // ----------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        int i;
        if (!rst_n) begin
            for (i = 0; i < BUFFER_NUM; i++) begin
                state_q[i] <= BS_IDLE;
                task_q[i]  <= '0;
            end
        end else begin
            for (i = 0; i < BUFFER_NUM; i++) begin
                state_q[i] <= state_d[i];
                task_q[i]  <= task_d[i];
            end
        end
    end

    // ----------------------------
    // DMA load request pulse：仅在 accept_task 时对 alloc_buf 拉高 1cycle
    // 注意：此时 task_q 会在同一 clock edge 被更新，所以下一拍组合派生的 addr/dim 正确
    // ----------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        int i;
        if (!rst_n) begin
            front_load_req <= '0;
            accept_task_q  <= 1'b0;
            alloc_buf_q    <= '0;
        end else begin
            // default deassert pulse and handshake reg
            front_load_req <= '0;
            accept_task_q  <= 1'b0;

            // latch handshake info when accept_task hits
            if (accept_task) begin
                accept_task_q <= 1'b1;
                alloc_buf_q   <= alloc_buf;
            end

            // one-cycle-later pulse based on latched buffer id
            if (accept_task_q) begin
                front_load_req[alloc_buf_q] <= 1'b1;
            end
        end
    end

endmodule
