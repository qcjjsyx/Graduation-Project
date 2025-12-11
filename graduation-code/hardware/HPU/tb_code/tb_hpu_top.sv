`timescale 1ns/1ps

module tb_hpu_top ();

    localparam DATA_W    = 32;
    localparam ROW_IDX_W = 8;      // 行号位宽，测试用 8 即可
    localparam MAX_ELEMS = 256;

    // DUT 端口
    logic                   clk;
    logic                   rst_n;

    logic                   pivot_start;
    logic                   pivot_busy;

    logic                   in_valid;
    logic                   in_ready;
    logic [DATA_W-1:0]      in_value;
    logic [ROW_IDX_W-1:0]   in_row_logical;
    logic                   in_last;

    logic                   pivot_valid;
    logic                   pivot_ready;
    logic [ROW_IDX_W-1:0]   pivot_row;
    logic [DATA_W-1:0]      pivot_value;
    logic                   pivot_fail;

    // 实例化 DUT
    hpu_top #(
        .DATA_W    (DATA_W),
        .ROW_IDX_W (ROW_IDX_W),
        .MAX_ELEMS (MAX_ELEMS)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),

        .pivot_start    (pivot_start),
        .pivot_busy     (pivot_busy),

        .in_valid       (in_valid),
        .in_ready       (in_ready),
        .in_value       (in_value),
        .in_row_logical (in_row_logical),
        .in_last        (in_last),

        .pivot_valid    (pivot_valid),
        .pivot_ready    (pivot_ready),
        .pivot_row      (pivot_row),
        .pivot_value    (pivot_value),
        .pivot_fail     (pivot_fail)
    );


    //========================
    // 时钟 & 复位
    //========================
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;     // 100 MHz
    end

    initial begin
        rst_n        = 1'b0;
        pivot_start  = 1'b0;
        in_valid     = 1'b0;
        in_value     = '0;
        in_row_logical = '0;
        in_last      = 1'b0;
        pivot_ready  = 1'b1;       // 始终准备好接收结果


        #1000;
        rst_n = 1'b1;
    end


    //========================
    // 周期计数器
    //========================
    int unsigned cycle;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle <= 0;
        end else begin
            cycle <= cycle + 1;
        end
    end

    //========================
    // 发送一列候选的 task
    //========================
    task automatic send_column(
        input int unsigned N,              // 本轮元素个数
        input int unsigned base_row        // 行号起始偏移，仅用于区分
    );
        int unsigned i;
        begin
            // 拉高 pivot_start 一个周期，开启本轮
            @(negedge clk);
            pivot_start <= 1'b1;
            @(negedge clk);
            pivot_start <= 1'b0;

            // 逐个发送 N 个元素
            for (i = 0; i < N; i = i + 1) begin
                // 等待 in_ready
                @(negedge clk);
                while (!in_ready) @(negedge clk);

                in_valid       <= 1'b1;
                in_value       <= (i == N-1) ? 32'sd1000 : $signed(i+1);  // 控制最后一个最大
                in_row_logical <= base_row + i[ROW_IDX_W-1:0];
                in_last        <= (i == N-1);

                @(negedge clk);
                in_valid <= 1'b0;
                in_last  <= 1'b0;
            end
        end
    endtask

    //========================
    // 测试单个 N 的延时
    //========================
    task automatic test_one_N(
        input int unsigned N
    );
        int unsigned start_cycle, end_cycle;
        begin
            $display("\n===== Test N = %0d =====", N);

            // 确保不在 busy 状态
            @(negedge clk);
            while (pivot_busy) @(negedge clk);
            // 记下 start_cycle，在 send_column 里拉 start
            $display("[%0t] Starting test for N=%0d", $time, N);
            start_cycle = cycle;
            send_column(N, 0);

            // 等待 pivot_valid
            @(negedge clk);
            while (!pivot_valid) @(negedge clk);
            end_cycle = cycle;

            // 打印结果
            $display("[%0t] N=%0d, latency (pivot_start->pivot_valid) = %0d cycles, pivot_row=%0d, pivot_value=%0d, fail=%0b",
                     $time, N, end_cycle - start_cycle, pivot_row, pivot_value, pivot_fail);

            // 等待结果被“消费”（这里 pivot_ready 一直是 1，所以下一拍就会清掉）
            @(negedge clk);
        end
    endtask

    //========================
    // 主测试流程
    //========================
    initial begin : main_test
        // 等待复位结束
        wait (rst_n == 1'b1);
        @(negedge clk);

        // 依次测试不同 Nint Ns;
//        Ns[0] = 2;
//        Ns[1] = 4;
//        Ns[2] = 8;
//        Ns[3] = 16;
//        Ns[4] = 32;
//        Ns[5] = 64;
//        Ns[6] = 128;
//        Ns[7] = 256;

//        int idx;
//        for (idx = 0; idx < 8; idx = idx + 1) begin
//            test_one_N(());
//        end
        
        test_one_N(2);
        test_one_N(4);
        test_one_N(8);
        test_one_N(16);
        test_one_N(32);
        test_one_N(64);
        test_one_N(128);
        test_one_N(256);
        #100;
        $display("\n===== ALL TESTS DONE =====");
        $finish;
    end



endmodule