#pragma once

#include <array>
#include <cstdint>
#include <systemc>

namespace hw {

static constexpr unsigned ROW_IDX_W = 8;
static constexpr unsigned MAX_ROWS = 1u << ROW_IDX_W;

struct ATU : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};

    sc_core::sc_in<bool> init_identity{"init_identity"};
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

    SC_HAS_PROCESS(ATU);

    explicit ATU(sc_core::sc_module_name name) : sc_core::sc_module(name) {
        SC_METHOD(comb);
        sensitive << init_busy_;
        sensitive << pivot_req_valid;

        SC_METHOD(tick);
        sensitive << clk.pos();
    }

private:
    std::array<std::uint8_t, MAX_ROWS> pvec_{};
    sc_core::sc_signal<bool> init_busy_{"init_busy"};
    sc_core::sc_signal<sc_dt::sc_uint<ROW_IDX_W>> init_index_{"init_index"};

    void comb() {
        const bool ready = !init_busy_.read() && !pivot_req_valid.read();
        q_req_ready.write(ready);
        pivot_req_ready.write(!init_busy_.read());
    }

    void tick() {
        if (!rst_n.read()) {
            for (unsigned i = 0; i < MAX_ROWS; ++i) {
                pvec_[i] = static_cast<std::uint8_t>(i);
            }
            init_busy_.write(false);
            init_index_.write(0);
            init_done.write(false);
            q_resp_valid.write(false);
            q_resp_row_physical.write(0);
            pivot_done.write(false);
            return;
        }

        init_done.write(false);
        q_resp_valid.write(false);
        pivot_done.write(false);

        if (init_busy_.read()) {
            const unsigned idx = init_index_.read().to_uint();
            pvec_[idx] = static_cast<std::uint8_t>(idx);
            if (idx == MAX_ROWS - 1) {
                init_busy_.write(false);
                init_done.write(true);
            } else {
                init_index_.write(idx + 1);
            }
            return;
        }

        if (init_identity.read()) {
            init_busy_.write(true);
            init_index_.write(0);
            return;
        }

        if (pivot_req_valid.read()) {
            const unsigned row_i = pivot_row_i.read().to_uint();
            const unsigned row_j = pivot_row_j.read().to_uint();
            const auto tmp = pvec_[row_i];
            pvec_[row_i] = pvec_[row_j];
            pvec_[row_j] = tmp;
            pivot_done.write(true);
            return;
        }

        if (q_req_valid.read()) {
            const unsigned row = q_req_row_logic.read().to_uint();
            q_resp_row_physical.write(pvec_[row]);
            q_resp_valid.write(true);
        }
    }
};

}  // namespace hw
