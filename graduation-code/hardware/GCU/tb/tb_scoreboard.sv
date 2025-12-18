`timescale 1ns/1ps

module gcu_dep_scoreboard_tb;

    // 参数配置：小规模便于看波形
    localparam int NODE_ID_W   = 4;    // 支持 node_id 0..15
    localparam int MAX_NODES   = 8;    // 实际开 8 个 entry
    // 与 DUT 端口保持一致：init_children_count 在设计中固定 16bit
    localparam int CHILD_CNT_W = 16;

    // DUT 端口信号
    logic                         clk;
    logic                         rst_n;

    logic                         init_valid;
    logic [NODE_ID_W-1:0]         init_node_id;
    logic [CHILD_CNT_W-1:0]       init_children_count;

    logic                         scatter_done_valid;
    logic [NODE_ID_W-1:0]         scatter_done_child_id;
    logic [NODE_ID_W-1:0]         scatter_done_parent_id;

    logic                         query_valid;
    logic [NODE_ID_W-1:0]         query_node_id;
    logic                         front_ready;
    logic [CHILD_CNT_W-1:0]       pending_children_count;

    // 实例化待测模块
    gcu_dep_scoreboard #(
        .NODE_ID_W   (NODE_ID_W),
        .MAX_NODES   (MAX_NODES),
        .CHILD_CNT_W (CHILD_CNT_W)
    ) dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .init_valid              (init_valid),
        .init_node_id            (init_node_id),
        .init_children_count     (init_children_count),
        .scatter_done_valid      (scatter_done_valid),
        .scatter_done_child_id   (scatter_done_child_id),
        .scatter_done_parent_id  (scatter_done_parent_id),
        .query_valid             (query_valid),
        .query_node_id           (query_node_id),
        .front_ready             (front_ready),
        .pending_children_count  (pending_children_count)
    );

    //========================
    // 时钟 & 复位
    //========================
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;   // 100 MHz
    end

    initial begin
        rst_n                 = 1'b0;
        init_valid            = 1'b0;
        init_node_id          = '0;
        init_children_count   = '0;
        scatter_done_valid    = 1'b0;
        scatter_done_child_id = '0;
        scatter_done_parent_id = '0;
        query_node_id         = '0;
        query_valid           = 1'b0;

        // // 波形文件（可选）
        // $dumpfile("gcu_dep_scoreboard_tb.vcd");
        // $dumpvars(0, gcu_dep_scoreboard_tb);

        #1000;
        rst_n       = 1'b1;
        query_valid = 1'b0;   // 查询时由专用任务拉高一个周期
    end

    //========================
    // 发送查询并打印当前节点状态；query_valid 仅在查询时拉高
    //========================
    task automatic query_and_show(input int id, input string tag);
        begin
            @(negedge clk);
            query_node_id <= id[NODE_ID_W-1:0];
            query_valid   <= 1'b1;

            // 在同一周期的负沿后少量延迟，采样组合输出
            @(negedge clk);
            #10;
            $display("[%0t] %s: query_node_id=%0d, pending_children=%0d, front_ready=%0b",
                     $time, tag, query_node_id, pending_children_count, front_ready);

            // 下一个负沿将 query_valid 拉低，避免持续有效
            @(negedge clk);
            query_valid <= 1'b0;
        end
    endtask

    //========================
    // 初始化某个节点的任务
    //========================
    task automatic init_node(input int id, input int cc);
        begin
            @(negedge clk);
            init_node_id        <= id[NODE_ID_W-1:0];
            init_children_count <= cc[CHILD_CNT_W-1:0];
            init_valid          <= 1'b1;

            @(negedge clk);
            init_valid          <= 1'b0;
        end
    endtask

    //========================
    // 触发一次 scatter_done 事件
    //========================
    task automatic scatter_done(input int child_id, input int parent_id);
        begin
            @(negedge clk);
            scatter_done_child_id  <= child_id[NODE_ID_W-1:0];
            scatter_done_parent_id <= parent_id[NODE_ID_W-1:0];
            scatter_done_valid     <= 1'b1;

            @(negedge clk);
            scatter_done_valid     <= 1'b0;
        end
    endtask

    //========================
    // 主测试流程
    //========================
    initial begin : main_test
        // 等待复位结束
        wait (rst_n == 1'b1);
        #50;

        // 场景 1：node 1 没有子节点：children_count = 0
        // 期望：初始化后 front_ready(node1) = 1, pending_children=0
        init_node(1, 0);
        query_and_show(1, "After init node1 (children=0)");
        #100;
        // 场景 2：node 2 有两个子节点：children_count = 2
        // 期望：初始化后 front_ready(node2) = 0, pending_children=2
        init_node(2, 2);
        #20;
        query_and_show(2, "After init node2 (children=2)");
        #100;
        // 场景 3：子节点 C3 完成 Scatter 到父节点 2
        // 期望：pending_children(node2) 从 2 -> 1, front_ready 仍为 0
        scatter_done(3, 2);
        query_and_show(2, "After scatter_done(child=3, parent=2)");
        #100;
        // 场景 4：子节点 C4 再次完成 Scatter 到父节点 2
        // 期望：pending_children(node2) 从 1 -> 0, front_ready 变为 1
        scatter_done(4, 2);
        query_and_show(2, "After scatter_done(child=4, parent=2)");
        #100;
        // 场景 5：再看 node1（没有子）：一直 front_ready=1
        query_and_show(1, "Check node1 again");

        // 结束仿真
        #50;
        $display("==== gcu_dep_scoreboard_tb DONE ====");
        $finish;
    end

endmodule
