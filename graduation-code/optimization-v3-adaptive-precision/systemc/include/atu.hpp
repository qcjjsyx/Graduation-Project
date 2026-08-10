#pragma once

#include <array>
#include <cstdint>
#include <systemc>

namespace hw {

static constexpr unsigned ROW_IDX_W = 9;
static constexpr unsigned MAX_ROWS = 256;

struct ATU : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};

    sc_core::sc_in<bool> init_identity{"init_identity"};
    sc_core::sc_in<sc_dt::sc_uint<ROW_IDX_W>> init_rows{"init_rows"};
    sc_core::sc_out<bool> init_done{"init_done"};

    sc_core::sc_in<bool> q_req_valid{"q_req_valid"};
    sc_core::sc_in<sc_dt::sc_uint<ROW_IDX_W>> q_req_row_logic{"q_req_row_logic"};
    sc_core::sc_out<bool> q_req_ready{"q_req_ready"};
    sc_core::sc_out<bool> q_resp_valid{"q_resp_valid"};
    sc_core::sc_out<sc_dt::sc_uint<ROW_IDX_W>> q_resp_row_physical{"q_resp_row_physical"};

    sc_core::sc_in<bool> pivot_req_valid{"pivot_req_valid"};
    sc_core::sc_in<sc_dt::sc_uint<ROW_IDX_W>> pivot_row_i{"pivot_row_i"};
    sc_core::sc_in<sc_dt::sc_uint<ROW_IDX_W>> pivot_row_j{"pivot_row_j"};
    sc_core::sc_out<bool> pivot_req_ready{"pivot_req_ready"};
    sc_core::sc_out<bool> pivot_done{"pivot_done"};

    SC_CTOR(ATU) {
        SC_METHOD(comb);
        sensitive << init_busy_;
        sensitive << init_identity;
        sensitive << pivot_req_valid;
        sensitive << rst_n;

        SC_METHOD(tick);
        sensitive << clk.pos();
    }

private:
    std::array<std::uint16_t, MAX_ROWS> pvec_{};
    sc_core::sc_signal<bool> init_busy_{"init_busy"};
    sc_core::sc_signal<bool> init_identity_d_{"init_identity_d"};
    sc_core::sc_signal<sc_dt::sc_uint<ROW_IDX_W>> init_index_{"init_index"};
    unsigned initialized_rows_{0};

    void comb() {
        const bool idle = rst_n.read() && !init_busy_.read() && !init_identity.read();
        pivot_req_ready.write(idle);
        q_req_ready.write(idle && !pivot_req_valid.read());
    }

    void tick() {
        if (!rst_n.read()) {
            pvec_.fill(0);
            init_busy_.write(false);
            init_identity_d_.write(false);
            init_index_.write(0);
            initialized_rows_ = 0;
            init_done.write(false);
            q_resp_valid.write(false);
            q_resp_row_physical.write(0);
            pivot_done.write(false);
            return;
        }

        init_done.write(false);
        q_resp_valid.write(false);
        pivot_done.write(false);
        const bool init_start = init_identity.read() && !init_identity_d_.read();
        init_identity_d_.write(init_identity.read());

        if (init_busy_.read()) {
            const unsigned idx = init_index_.read().to_uint();
            pvec_[idx] = static_cast<std::uint16_t>(idx);
            if (idx + 1 >= initialized_rows_) {
                init_busy_.write(false);
                init_done.write(true);
            } else {
                init_index_.write(idx + 1);
            }
            return;
        }

        if (init_start) {
            initialized_rows_ = init_rows.read().to_uint();
            if (initialized_rows_ > MAX_ROWS) {
                SC_REPORT_ERROR("ATU", "init_rows exceeds MAX_ROWS");
                initialized_rows_ = MAX_ROWS;
            }
            if (initialized_rows_ == 0) {
                init_done.write(true);
            } else {
                init_busy_.write(true);
                init_index_.write(0);
            }
            return;
        }

        if (pivot_req_valid.read() && pivot_req_ready.read()) {
            const unsigned row_i = pivot_row_i.read().to_uint();
            const unsigned row_j = pivot_row_j.read().to_uint();
            if (row_i >= initialized_rows_ || row_j >= initialized_rows_) {
                SC_REPORT_ERROR("ATU", "pivot row is outside initialized P-vector");
                pivot_done.write(true);
                return;
            }
            const auto tmp = pvec_[row_i];
            pvec_[row_i] = pvec_[row_j];
            pvec_[row_j] = tmp;
            pivot_done.write(true);
            return;
        }

        if (q_req_valid.read() && q_req_ready.read()) {
            const unsigned row = q_req_row_logic.read().to_uint();
            q_resp_row_physical.write(
                row < initialized_rows_ ? pvec_[row] : row);
            q_resp_valid.write(true);
        }
    }
};

}  // namespace hw
