#include <systemc.h>
#include "pe_tsolve.h" // 假设复用之前的 PE 定义，并增加了模式切换信号

const int TILE_SIZE = 16;

// --- 片上缓存 (Tile Buffer) ---
// 充当 AXI 接口与脉动阵列之间的桥梁，支持高带宽的向量并行读取
SC_MODULE(TileBuffer) {
    sc_in<bool> clk;
    
    // 假设这些是连接到 AXI DMA 的写入接口
    sc_in<bool> axi_write_en;
    sc_in<float> axi_write_data;
    // ... 省略 AXI 地址逻辑
    
    // 连接到脉动阵列的并行输出端口 (每周期可吐出 16 个数据)
    sc_out<float> array_L_out[TILE_SIZE]; 
    sc_out<float> array_D_out[TILE_SIZE];

    // 内部存储: Double Buffering (乒乓操作) 掩盖数据搬运延迟
    float sram_L[2][TILE_SIZE][TILE_SIZE]; 
    float sram_D[2][TILE_SIZE][TILE_SIZE];
    
    int read_row_idx;

    void feed_array() {
        // 按照周期将 SRAM 中的数据压入脉动阵列的边界端口
        for(int i=0; i<TILE_SIZE; i++) {
            array_L_out[i].write(sram_L[0][read_row_idx][i]);
            array_D_out[i].write(sram_D[0][read_row_idx][i]);
        }
        read_row_idx = (read_row_idx + 1) % TILE_SIZE;
    }

    SC_CTOR(TileBuffer) {
        read_row_idx = 0;
        SC_METHOD(feed_array);
        sensitive << clk.pos();
    }
};

// --- 硬件加速器顶层 (Accelerator Top) ---
SC_MODULE(Spatula_16x16_Top) {
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> mode_is_tsolve; // 模式选择：1=TSOLVE, 0=GEMM
    
    // 实例化子模块
    TileBuffer* buffer;
    PE_TSolve* pe_array[TILE_SIZE][TILE_SIZE];

    // 互联信号
    sc_signal<float> wire_L[TILE_SIZE+1][TILE_SIZE];
    sc_signal<float> wire_D[TILE_SIZE+1][TILE_SIZE];
    sc_signal<float> wire_X[TILE_SIZE][TILE_SIZE+1]; // 水平广播总线

    SC_CTOR(Spatula_16x16_Top) {
        // 1. 初始化 Buffer
        buffer = new TileBuffer("SRAM_Tile_Buffer");
        buffer->clk(clk);
        
        // 2. 初始化 16x16 阵列
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                char name[32];
                sprintf(name, "PE_%d_%d", i, j);
                pe_array[i][j] = new PE_TSolve(name);
                pe_array[i][j]->clk(clk);
                pe_array[i][j]->rst(rst);
                // 连线拓扑...
                pe_array[i][j]->L_in(wire_L[i][j]);
                pe_array[i][j]->L_out(wire_L[i+1][j]);
                pe_array[i][j]->b_in(wire_D[i][j]);
                pe_array[i][j]->b_out(wire_D[i+1][j]);
                // ... 省略部分内部连线
            }
        }

        // 3. 将 Buffer 的输出绑定到阵列的第一行 (Top boundary)
        for (int i = 0; i < TILE_SIZE; i++) {
            // 注意：实际硬件中由于时空错位，喂给阵列的数据需要做梯形延迟 (Skewing)
            // 即第 1 列比第 0 列晚 1 个 cycle 送入，此处在 C++ 模型中需添加 Skewing Register 逻辑
            // 此处简化表示连接关系
            // wire_L[0][i] 连接到 buffer->array_L_out[i]
        }
    }
    
    ~Spatula_16x16_Top() {
        delete buffer;
        for (int i=0; i<TILE_SIZE; i++)
            for (int j=0; j<TILE_SIZE; j++)
                delete pe_array[i][j];
    }
};