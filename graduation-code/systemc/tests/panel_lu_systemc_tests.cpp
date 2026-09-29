// T06 Panel-LU architecture prototype: SystemC clock-driven timing tests.
//
// The prototype is a plain C++ tick model (it never advances time itself); the
// SystemC testbench owns the clock and calls tick() exactly once per posedge.
// These tests check that clock-driven runs reproduce the plain tick model
// cycle-for-cycle, that PE exclusivity/dependency ordering hold, and that
// abort/reset behave the same under a real SystemC scheduler.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <systemc>

#include "panel_lu/panel_lu_prototype.hpp"

namespace {

namespace cmd = hw::command;
namespace pl = hw::panel_lu;

std::uint32_t threshold_bits_0001() {
    const float value = 0.001f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

pl::PanelLuConfig base_config(std::uint32_t pe_count) {
    pl::PanelLuConfig config;
    config.pe_count = pe_count;
    config.task_slots_per_pe = 4;
    config.dispatcher_queue_depth = 32;
    config.max_resident_commands = 2;
    return config;
}

pl::PanelCommandRequest make_request(
    std::uint32_t command_id,
    std::uint32_t front_dim,
    std::uint32_t pivot_dim,
    std::uint32_t seed) {
    pl::PanelCommandRequest request;
    request.command_id = command_id;
    request.node_id = command_id;
    request.front_dim = front_dim;
    request.pivot_dim = pivot_dim;
    request.policy = cmd::PivotPolicy::Static;
    request.threshold_bits = threshold_bits_0001();
    request.tile_size = 16;
    request.synthetic.seed = seed;
    request.label = "systemc";
    return request;
}

struct PlainRun {
    bool idle{false};
    std::map<std::uint32_t, pl::PanelCommandResponse> responses;
};

PlainRun run_plain(pl::PanelLuConfig config,
                   const std::vector<pl::PanelCommandRequest>& requests,
                   std::uint64_t max_cycles = 200000) {
    PlainRun run;
    pl::PanelLuPrototype prototype(config);
    if (!prototype.configure().ok) {
        return run;
    }
    for (const pl::PanelCommandRequest& request : requests) {
        if (!prototype.submit_command(request).ok) {
            return run;
        }
    }
    prototype.run_until_idle(max_cycles);
    run.idle = prototype.idle();
    pl::PanelCommandResponse response;
    while (prototype.poll_response(response)) {
        run.responses[response.command_id] = response;
    }
    return run;
}

class T06SystemCTests : public sc_core::sc_module {
public:
    sc_core::sc_in<bool> clk;

    SC_CTOR(T06SystemCTests) { SC_THREAD(run_tests); }

    int failures() const { return failures_; }
    int checks() const { return checks_; }
    bool done() const { return done_; }

private:
    int failures_{0};
    int checks_{0};
    bool done_{false};

    void check(bool condition, const std::string& message) {
        ++checks_;
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void wait_step(pl::PanelLuPrototype& prototype) {
        wait(clk.posedge_event());
        prototype.tick();
    }

    bool run_clocked(pl::PanelLuPrototype& prototype, std::uint64_t max_cycles) {
        const std::uint64_t start = prototype.cycle();
        while (!prototype.idle() && prototype.cycle() - start < max_cycles) {
            wait_step(prototype);
        }
        return prototype.idle();
    }

    void test_tail_clocked() {
        pl::PanelLuPrototype prototype(base_config(1));
        check(prototype.configure().ok, "clocked tail: configure");
        check(prototype.submit_command(make_request(0, 23, 23, 7)).ok, "clocked tail: submit");
        const bool idle = run_clocked(prototype, 50000);
        check(idle, "clocked tail: reaches idle");
        pl::PanelCommandResponse response;
        check(prototype.poll_response(response), "clocked tail: response present");
        check(response.status == cmd::StatusCode::Ok, "clocked tail: OK");
        check(response.pivot_count == 23, "clocked tail: pivot_count=23");
        check(prototype.control_fault() == false, "clocked tail: no control fault");
        check(
            prototype.task_records().size() == 5,
            "clocked tail: 5 cross-tile tasks with tails");

        const PlainRun plain =
            run_plain(base_config(1), {make_request(0, 23, 23, 7)});
        check(plain.idle, "plain tail: reaches idle");
        check(
            response.finish_cycle == plain.responses.at(0).finish_cycle,
            "clocked and plain tick models agree cycle-for-cycle");

        const auto& records = prototype.task_records();
        for (std::size_t i = 0; i < records.size(); ++i) {
            for (std::size_t j = i + 1; j < records.size(); ++j) {
                if (records[i].pe_id != records[j].pe_id) {
                    continue;
                }
                const bool overlap = records[i].exec_start_cycle < records[j].exec_done_cycle &&
                                     records[j].exec_start_cycle < records[i].exec_done_cycle;
                check(!overlap, "clocked tail: one PE task at a time");
            }
            for (const std::uint64_t dep : records[i].task.deps) {
                for (const pl::TaskRecord& producer : records) {
                    if (producer.task.task_id == dep) {
                        check(
                            records[i].dispatch_cycle > producer.writeback_done_cycle,
                            "clocked tail: dependency visible before consumer dispatch");
                    }
                }
            }
        }
    }

    void test_dual_pe_clocked() {
        pl::PanelLuPrototype prototype(base_config(2));
        check(prototype.configure().ok, "clocked dual: configure");
        check(
            prototype.submit_command(make_request(0, 40, 40, 3)).ok,
            "clocked dual: submit 0");
        check(
            prototype.submit_command(make_request(1, 40, 40, 11)).ok,
            "clocked dual: submit 1");
        const bool idle = run_clocked(prototype, 100000);
        check(idle, "clocked dual: reaches idle");
        std::map<std::uint32_t, pl::PanelCommandResponse> responses;
        pl::PanelCommandResponse response;
        while (prototype.poll_response(response)) {
            responses[response.command_id] = response;
        }
        check(responses.size() == 2, "clocked dual: two completions");
        check(
            responses.at(0).status == cmd::StatusCode::Ok &&
                responses.at(1).status == cmd::StatusCode::Ok,
            "clocked dual: both commands OK");

        const PlainRun plain = run_plain(
            base_config(2),
            {make_request(0, 40, 40, 3), make_request(1, 40, 40, 11)});
        check(plain.idle, "plain dual: reaches idle");
        check(
            responses.at(0).finish_cycle == plain.responses.at(0).finish_cycle &&
                responses.at(1).finish_cycle == plain.responses.at(1).finish_cycle,
            "clocked dual: matches plain tick model");
    }

    void test_abort_clocked() {
        pl::PanelLuPrototype prototype(base_config(1));
        check(prototype.configure().ok, "clocked abort: configure");
        check(prototype.submit_command(make_request(0, 64, 64, 1)).ok, "clocked abort: cmd0");
        check(prototype.submit_command(make_request(1, 64, 64, 2)).ok, "clocked abort: cmd1");
        auto committed = [&prototype](std::uint32_t command_id) {
            std::uint32_t count = 0;
            for (const pl::TaskRecord& record : prototype.task_records()) {
                if (record.task.command_id == command_id && record.committed) {
                    ++count;
                }
            }
            return count;
        };
        while ((committed(0) == 0 || committed(1) == 0) && prototype.cycle() < 100000) {
            wait_step(prototype);
        }
        check(
            committed(0) > 0 && committed(1) > 0,
            "clocked abort: both commands commit before abort");
        prototype.abort_command(1);
        check(run_clocked(prototype, 100000), "clocked abort: reaches idle");
        std::map<std::uint32_t, pl::PanelCommandResponse> responses;
        pl::PanelCommandResponse response;
        while (prototype.poll_response(response)) {
            responses[response.command_id] = response;
        }
        check(
            responses.at(0).status == cmd::StatusCode::Ok,
            "clocked abort: sibling command unaffected");
        check(
            responses.at(1).status == cmd::StatusCode::Aborted,
            "clocked abort: targeted command ABORTED");
    }

    void test_reset_clocked() {
        pl::PanelLuPrototype prototype(base_config(1));
        check(prototype.configure().ok, "clocked reset: configure");
        check(prototype.submit_command(make_request(0, 64, 64, 4)).ok, "clocked reset: submit");
        for (int i = 0; i < 150; ++i) {
            wait_step(prototype);
        }
        const std::uint64_t epoch = prototype.epoch();
        prototype.reset();
        check(prototype.epoch() == epoch + 1, "clocked reset: epoch advanced");
        check(run_clocked(prototype, 1000), "clocked reset: idle after reset");
        pl::PanelCommandResponse stale;
        check(!prototype.poll_response(stale), "clocked reset: old responses dropped");
        check(
            prototype.submit_command(make_request(0, 32, 32, 5)).ok,
            "clocked reset: resubmit after reset");
        check(run_clocked(prototype, 50000), "clocked reset: post-reset idle");
        pl::PanelCommandResponse response;
        check(prototype.poll_response(response), "clocked reset: post-reset response");
        check(response.status == cmd::StatusCode::Ok, "clocked reset: post-reset OK");
    }

    void run_tests() {
        test_tail_clocked();
        test_dual_pe_clocked();
        test_abort_clocked();
        test_reset_clocked();
        done_ = true;
        sc_core::sc_stop();
    }
};

}  // namespace

int sc_main(int, char**) {
    // The runtime prototype must not advance SystemC time by itself.
    const sc_core::sc_time before = sc_core::sc_time_stamp();
    pl::PanelLuPrototype idle(base_config(1));
    if (idle.configure().ok) {
        idle.tick();
    }
    if (sc_core::sc_time_stamp() != before) {
        std::cerr << "FAIL: prototype tick advanced SystemC time\n";
        return 1;
    }

    sc_core::sc_clock clock("clk", sc_core::sc_time(1, sc_core::SC_NS));
    T06SystemCTests tests("t06_systemc_tests");
    tests.clk(clock);
    sc_core::sc_start();
    if (!tests.done()) {
        std::cerr << "FAIL: SystemC test thread did not finish\n";
        return 1;
    }
    std::cout << tests.checks() << " checks, " << tests.failures() << " failures\n";
    return tests.failures() == 0 ? 0 : 1;
}
