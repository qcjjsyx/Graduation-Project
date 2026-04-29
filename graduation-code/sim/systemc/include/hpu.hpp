#pragma once

#include <cstdint>
#include <limits>
#include <systemc>
#include <vector>

#include "atu.hpp"

namespace hw {

struct HPU : sc_core::sc_module {
    static constexpr unsigned DATA_W = 32;
    static constexpr unsigned MAX_ELEMS = 256;

    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> rst_n{"rst_n"};

    sc_core::sc_in<bool> pivot_start{"pivot_start"};
    sc_core::sc_out<bool> pivot_busy{"pivot_busy"};

    sc_core::sc_in<bool> in_valid{"in_valid"};
    sc_core::sc_out<bool> in_ready{"in_ready"};
    sc_core::sc_in<sc_dt::sc_int<DATA_W>> in_value{"in_value"};
    sc_core::sc_in<sc_dt::sc_uint<ROW_IDX_W>> in_row_logical{"in_row_logical"};
    sc_core::sc_in<bool> in_last{"in_last"};

    sc_core::sc_out<bool> pivot_valid{"pivot_valid"};
    sc_core::sc_in<bool> pivot_ready{"pivot_ready"};
    sc_core::sc_out<sc_dt::sc_uint<ROW_IDX_W>> pivot_row{"pivot_row"};
    sc_core::sc_out<sc_dt::sc_int<DATA_W>> pivot_value{"pivot_value"};
    sc_core::sc_out<bool> pivot_fail{"pivot_fail"};

    SC_HAS_PROCESS(HPU);

    explicit HPU(sc_core::sc_module_name name) : sc_core::sc_module(name) {
        SC_METHOD(comb);
        sensitive << state_;
        sensitive << pivot_valid_r_;
        sensitive << pivot_row_r_;
        sensitive << pivot_value_r_;
        sensitive << pivot_fail_r_;

        SC_METHOD(tick);
        sensitive << clk.pos();
    }

private:
    enum State {
        S_IDLE = 0,
        S_LOAD = 1,
        S_SELECT = 2,
        S_OUT = 3,
    };

    struct Candidate {
        std::uint8_t row;
        std::int32_t value;
    };

    sc_core::sc_signal<int> state_{"state"};
    sc_core::sc_signal<bool> pivot_valid_r_{"pivot_valid_r"};
    sc_core::sc_signal<sc_dt::sc_uint<ROW_IDX_W>> pivot_row_r_{"pivot_row_r"};
    sc_core::sc_signal<sc_dt::sc_int<DATA_W>> pivot_value_r_{"pivot_value_r"};
    sc_core::sc_signal<bool> pivot_fail_r_{"pivot_fail_r"};
    std::vector<Candidate> candidates_;

    static std::int64_t abs64(std::int32_t value) {
        const std::int64_t extended = value;
        return extended < 0 ? -extended : extended;
    }

    void comb() {
        const int state = state_.read();
        in_ready.write(state == S_LOAD);
        pivot_busy.write(state != S_IDLE);
        pivot_valid.write(pivot_valid_r_.read());
        pivot_row.write(pivot_row_r_.read());
        pivot_value.write(pivot_value_r_.read());
        pivot_fail.write(pivot_fail_r_.read());
    }

    void tick() {
        if (!rst_n.read()) {
            state_.write(S_IDLE);
            candidates_.clear();
            pivot_valid_r_.write(false);
            pivot_row_r_.write(0);
            pivot_value_r_.write(0);
            pivot_fail_r_.write(false);
            return;
        }

        switch (state_.read()) {
        case S_IDLE:
            pivot_valid_r_.write(false);
            pivot_fail_r_.write(false);
            candidates_.clear();
            if (pivot_start.read()) {
                state_.write(S_LOAD);
            }
            break;

        case S_LOAD:
            if (in_valid.read()) {
                candidates_.push_back({
                    static_cast<std::uint8_t>(in_row_logical.read().to_uint()),
                    static_cast<std::int32_t>(in_value.read().to_int()),
                });
                if (in_last.read() || candidates_.size() >= MAX_ELEMS) {
                    state_.write(S_SELECT);
                }
            }
            break;

        case S_SELECT:
            select_pivot();
            state_.write(S_OUT);
            break;

        case S_OUT:
            if (pivot_valid_r_.read() && pivot_ready.read()) {
                pivot_valid_r_.write(false);
                state_.write(S_IDLE);
            }
            break;

        default:
            state_.write(S_IDLE);
            break;
        }
    }

    void select_pivot() {
        if (candidates_.empty()) {
            pivot_valid_r_.write(false);
            pivot_fail_r_.write(true);
            pivot_row_r_.write(0);
            pivot_value_r_.write(0);
            return;
        }

        std::size_t best = 0;
        std::int64_t best_abs = abs64(candidates_[0].value);
        for (std::size_t i = 1; i < candidates_.size(); ++i) {
            const std::int64_t value_abs = abs64(candidates_[i].value);
            if (value_abs > best_abs) {
                best = i;
                best_abs = value_abs;
            }
        }

        pivot_valid_r_.write(true);
        pivot_fail_r_.write(false);
        pivot_row_r_.write(candidates_[best].row);
        pivot_value_r_.write(candidates_[best].value);
    }
};

}  // namespace hw
