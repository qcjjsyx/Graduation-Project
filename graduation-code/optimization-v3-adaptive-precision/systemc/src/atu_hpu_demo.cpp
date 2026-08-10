#include <iostream>
#include <limits>
#include <systemc>

#include "atu.hpp"
#include "hpu.hpp"

namespace {

struct DemoDriver : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_out<bool> rst_n{"rst_n"};

    sc_core::sc_out<bool> atu_init_identity{"atu_init_identity"};
    sc_core::sc_out<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_init_rows{"atu_init_rows"};
    sc_core::sc_in<bool> atu_init_done{"atu_init_done"};
    sc_core::sc_out<bool> atu_q_req_valid{"atu_q_req_valid"};
    sc_core::sc_out<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_q_req_row_logic{"atu_q_req_row_logic"};
    sc_core::sc_in<bool> atu_q_req_ready{"atu_q_req_ready"};
    sc_core::sc_in<bool> atu_q_resp_valid{"atu_q_resp_valid"};
    sc_core::sc_in<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_q_resp_row_physical{"atu_q_resp_row_physical"};
    sc_core::sc_out<bool> atu_pivot_req_valid{"atu_pivot_req_valid"};
    sc_core::sc_out<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_pivot_row_i{"atu_pivot_row_i"};
    sc_core::sc_out<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_pivot_row_j{"atu_pivot_row_j"};
    sc_core::sc_in<bool> atu_pivot_req_ready{"atu_pivot_req_ready"};
    sc_core::sc_in<bool> atu_pivot_done{"atu_pivot_done"};

    sc_core::sc_out<bool> hpu_pivot_start{"hpu_pivot_start"};
    sc_core::sc_in<bool> hpu_pivot_busy{"hpu_pivot_busy"};
    sc_core::sc_out<bool> hpu_in_valid{"hpu_in_valid"};
    sc_core::sc_in<bool> hpu_in_ready{"hpu_in_ready"};
    sc_core::sc_out<sc_dt::sc_int<hw::HPU::DATA_W>> hpu_in_value{"hpu_in_value"};
    sc_core::sc_out<sc_dt::sc_uint<hw::ROW_IDX_W>> hpu_in_row_logical{"hpu_in_row_logical"};
    sc_core::sc_out<bool> hpu_in_last{"hpu_in_last"};
    sc_core::sc_in<bool> hpu_pivot_valid{"hpu_pivot_valid"};
    sc_core::sc_out<bool> hpu_pivot_ready{"hpu_pivot_ready"};
    sc_core::sc_in<sc_dt::sc_uint<hw::ROW_IDX_W>> hpu_pivot_row{"hpu_pivot_row"};
    sc_core::sc_in<sc_dt::sc_int<hw::HPU::DATA_W>> hpu_pivot_value{"hpu_pivot_value"};
    sc_core::sc_in<bool> hpu_pivot_fail{"hpu_pivot_fail"};

    SC_CTOR(DemoDriver) {
        SC_THREAD(run);
        sensitive << clk.pos();
    }

private:
    void wait_tick() {
        wait();
        wait(sc_core::SC_ZERO_TIME);
    }

    void wait_cycles(int cycles) {
        for (int i = 0; i < cycles; ++i) {
            wait_tick();
        }
    }

    void set_defaults() {
        rst_n.write(false);

        atu_init_identity.write(false);
        atu_init_rows.write(8);
        atu_q_req_valid.write(false);
        atu_q_req_row_logic.write(0);
        atu_pivot_req_valid.write(false);
        atu_pivot_row_i.write(0);
        atu_pivot_row_j.write(0);

        hpu_pivot_start.write(false);
        hpu_in_valid.write(false);
        hpu_in_value.write(0);
        hpu_in_row_logical.write(0);
        hpu_in_last.write(false);
        hpu_pivot_ready.write(true);
    }

    void expect(bool condition, const char* message) {
        if (!condition) {
            SC_REPORT_ERROR("DEMO", message);
            sc_core::sc_stop();
        }
    }

    unsigned query_atu(unsigned logical_row) {
        atu_q_req_row_logic.write(logical_row);
        atu_q_req_valid.write(true);
        wait_tick();
        expect(atu_q_req_ready.read(), "ATU query was not ready");
        expect(atu_q_resp_valid.read(), "ATU query did not produce a response");
        const unsigned physical_row = atu_q_resp_row_physical.read().to_uint();
        atu_q_req_valid.write(false);
        return physical_row;
    }

    void swap_atu(unsigned row_i, unsigned row_j) {
        atu_pivot_row_i.write(row_i);
        atu_pivot_row_j.write(row_j);
        atu_pivot_req_valid.write(true);
        wait_tick();
        expect(atu_pivot_req_ready.read(), "ATU pivot port was not ready");
        expect(atu_pivot_done.read(), "ATU pivot did not complete");
        atu_pivot_req_valid.write(false);
    }

    void feed_hpu_candidate(unsigned row, int value, bool last) {
        hpu_in_row_logical.write(row);
        hpu_in_value.write(value);
        hpu_in_last.write(last);
        hpu_in_valid.write(true);
        wait_tick();
        expect(hpu_in_ready.read(), "HPU input stream was not ready");
        hpu_in_valid.write(false);
        hpu_in_last.write(false);
    }

    void run() {
        set_defaults();
        wait_cycles(3);
        rst_n.write(true);
        wait_tick();

        std::cout << "[TB] reset released at " << sc_core::sc_time_stamp() << "\n";

        atu_init_identity.write(true);
        wait_tick();
        atu_init_identity.write(false);
        while (!atu_init_done.read()) {
            wait_tick();
        }
        std::cout << "[TB] ATU identity initialization done at "
                  << sc_core::sc_time_stamp() << "\n";

        const unsigned row3_before = query_atu(3);
        std::cout << "[TB] ATU query logical 3 -> physical " << row3_before << "\n";
        expect(row3_before == 3, "ATU identity query mismatch");

        swap_atu(3, 7);
        const unsigned row3_after = query_atu(3);
        const unsigned row7_after = query_atu(7);
        std::cout << "[TB] ATU after swap: logical 3 -> physical " << row3_after
                  << ", logical 7 -> physical " << row7_after << "\n";
        expect(row3_after == 7, "ATU row 3 should map to physical row 7 after swap");
        expect(row7_after == 3, "ATU row 7 should map to physical row 3 after swap");
        expect(query_atu(9) == 9, "ATU update row must use identity bypass");

        hpu_pivot_start.write(true);
        wait_tick();
        hpu_pivot_start.write(false);
        wait_tick();
        expect(hpu_pivot_busy.read(), "HPU did not enter busy state");

        feed_hpu_candidate(0, 10, false);
        feed_hpu_candidate(1, -5, false);
        feed_hpu_candidate(2, -20, false);
        feed_hpu_candidate(3, 20, true);

        while (!hpu_pivot_valid.read() && !hpu_pivot_fail.read()) {
            wait_tick();
        }

        std::cout << "[TB] HPU pivot row=" << hpu_pivot_row.read().to_uint()
                  << " value=" << hpu_pivot_value.read().to_int()
                  << " fail=" << hpu_pivot_fail.read() << "\n";
        expect(!hpu_pivot_fail.read(), "HPU unexpectedly failed");
        expect(hpu_pivot_row.read().to_uint() == 2, "HPU tie-break should keep first max-abs candidate");
        expect(hpu_pivot_value.read().to_int() == -20, "HPU pivot value mismatch");

        wait_tick();

        hpu_pivot_start.write(true);
        wait_tick();
        hpu_pivot_start.write(false);
        wait_tick();
        feed_hpu_candidate(0, 0, false);
        feed_hpu_candidate(1, 0, true);
        while (!hpu_pivot_valid.read()) {
            wait_tick();
        }
        expect(hpu_pivot_fail.read(), "HPU must reject an all-zero pivot column");
        expect(hpu_pivot_value.read().to_int() == 0, "HPU zero-column value mismatch");
        wait_tick();

        hpu_pivot_start.write(true);
        wait_tick();
        hpu_pivot_start.write(false);
        wait_tick();
        feed_hpu_candidate(0, std::numeric_limits<std::int32_t>::max(), false);
        feed_hpu_candidate(1, std::numeric_limits<std::int32_t>::min(), false);
        feed_hpu_candidate(2, -7, true);
        while (!hpu_pivot_valid.read()) {
            wait_tick();
        }
        expect(!hpu_pivot_fail.read(), "HPU rejected int32 minimum candidate");
        expect(
            hpu_pivot_row.read().to_uint() == 1,
            "HPU int32 minimum absolute-value comparison mismatch");
        expect(
            hpu_pivot_value.read().to_int() ==
                std::numeric_limits<std::int32_t>::min(),
            "HPU int32 minimum pivot value mismatch");
        wait_tick();

        std::cout << "[TB] ALL PASSED\n";
        sc_core::sc_stop();
    }
};

}  // namespace

int sc_main(int, char**) {
    sc_core::sc_clock clk("clk", sc_core::sc_time(10, sc_core::SC_NS));

    sc_core::sc_signal<bool> rst_n;

    sc_core::sc_signal<bool> atu_init_identity;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_init_rows;
    sc_core::sc_signal<bool> atu_init_done;
    sc_core::sc_signal<bool> atu_q_req_valid;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_q_req_row_logic;
    sc_core::sc_signal<bool> atu_q_req_ready;
    sc_core::sc_signal<bool> atu_q_resp_valid;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_q_resp_row_physical;
    sc_core::sc_signal<bool> atu_pivot_req_valid;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_pivot_row_i;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> atu_pivot_row_j;
    sc_core::sc_signal<bool> atu_pivot_req_ready;
    sc_core::sc_signal<bool> atu_pivot_done;

    sc_core::sc_signal<bool> hpu_pivot_start;
    sc_core::sc_signal<bool> hpu_pivot_busy;
    sc_core::sc_signal<bool> hpu_in_valid;
    sc_core::sc_signal<bool> hpu_in_ready;
    sc_core::sc_signal<sc_dt::sc_int<hw::HPU::DATA_W>> hpu_in_value;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> hpu_in_row_logical;
    sc_core::sc_signal<bool> hpu_in_last;
    sc_core::sc_signal<bool> hpu_pivot_valid;
    sc_core::sc_signal<bool> hpu_pivot_ready;
    sc_core::sc_signal<sc_dt::sc_uint<hw::ROW_IDX_W>> hpu_pivot_row;
    sc_core::sc_signal<sc_dt::sc_int<hw::HPU::DATA_W>> hpu_pivot_value;
    sc_core::sc_signal<bool> hpu_pivot_fail;

    hw::ATU atu("atu");
    atu.clk(clk);
    atu.rst_n(rst_n);
    atu.init_identity(atu_init_identity);
    atu.init_rows(atu_init_rows);
    atu.init_done(atu_init_done);
    atu.q_req_valid(atu_q_req_valid);
    atu.q_req_row_logic(atu_q_req_row_logic);
    atu.q_req_ready(atu_q_req_ready);
    atu.q_resp_valid(atu_q_resp_valid);
    atu.q_resp_row_physical(atu_q_resp_row_physical);
    atu.pivot_req_valid(atu_pivot_req_valid);
    atu.pivot_row_i(atu_pivot_row_i);
    atu.pivot_row_j(atu_pivot_row_j);
    atu.pivot_req_ready(atu_pivot_req_ready);
    atu.pivot_done(atu_pivot_done);

    hw::HPU hpu("hpu");
    hpu.clk(clk);
    hpu.rst_n(rst_n);
    hpu.pivot_start(hpu_pivot_start);
    hpu.pivot_busy(hpu_pivot_busy);
    hpu.in_valid(hpu_in_valid);
    hpu.in_ready(hpu_in_ready);
    hpu.in_value(hpu_in_value);
    hpu.in_row_logical(hpu_in_row_logical);
    hpu.in_last(hpu_in_last);
    hpu.pivot_valid(hpu_pivot_valid);
    hpu.pivot_ready(hpu_pivot_ready);
    hpu.pivot_row(hpu_pivot_row);
    hpu.pivot_value(hpu_pivot_value);
    hpu.pivot_fail(hpu_pivot_fail);

    DemoDriver driver("driver");
    driver.clk(clk);
    driver.rst_n(rst_n);
    driver.atu_init_identity(atu_init_identity);
    driver.atu_init_rows(atu_init_rows);
    driver.atu_init_done(atu_init_done);
    driver.atu_q_req_valid(atu_q_req_valid);
    driver.atu_q_req_row_logic(atu_q_req_row_logic);
    driver.atu_q_req_ready(atu_q_req_ready);
    driver.atu_q_resp_valid(atu_q_resp_valid);
    driver.atu_q_resp_row_physical(atu_q_resp_row_physical);
    driver.atu_pivot_req_valid(atu_pivot_req_valid);
    driver.atu_pivot_row_i(atu_pivot_row_i);
    driver.atu_pivot_row_j(atu_pivot_row_j);
    driver.atu_pivot_req_ready(atu_pivot_req_ready);
    driver.atu_pivot_done(atu_pivot_done);
    driver.hpu_pivot_start(hpu_pivot_start);
    driver.hpu_pivot_busy(hpu_pivot_busy);
    driver.hpu_in_valid(hpu_in_valid);
    driver.hpu_in_ready(hpu_in_ready);
    driver.hpu_in_value(hpu_in_value);
    driver.hpu_in_row_logical(hpu_in_row_logical);
    driver.hpu_in_last(hpu_in_last);
    driver.hpu_pivot_valid(hpu_pivot_valid);
    driver.hpu_pivot_ready(hpu_pivot_ready);
    driver.hpu_pivot_row(hpu_pivot_row);
    driver.hpu_pivot_value(hpu_pivot_value);
    driver.hpu_pivot_fail(hpu_pivot_fail);

    sc_core::sc_start();
    return sc_core::sc_report_handler::get_count(sc_core::SC_ERROR) == 0 ? 0 : 1;
}
