#include <systemc>
#include <iostream>
#include "controlplane/control_plane.hpp"
#include "tree_artifact_check.hpp"

class TreeClockTest : public sc_core::sc_module {
public:
    sc_core::sc_in<bool> clk;
    int failures{0};
    TreeClockTest(sc_core::sc_module_name name, std::string input, std::string output)
        : sc_module(name), input_(std::move(input)), output_(std::move(output)) {
        SC_THREAD(run);
    }
private:
    void run() {
        for (const std::string mode : {"fp32", "int32"}) {
            hw::controlplane::ControlPlaneConfig config;
            config.enable_real_panel = true;
            config.backend = hw::controlplane::ExecBackend::RequireRealKernels;
            config.panel_unit_count = 2;
            config.panel.compute_group_count = 2;
            config.panel.pes_per_group = 2;
            config.panel.pe_count = 4;
            const auto directory = input_ + "/three_level/" + mode + "/device";
            hw::controlplane::CommandDevice plain(hw::dataplane::load_artifact(directory), config);
            hw::controlplane::CommandDevice clocked(hw::dataplane::load_artifact(directory), config);
            std::string error;
            if (!plain.start(&error) || !clocked.start(&error)) {
                ++failures; break;
            }
            plain.run(1000000);
            while (!clocked.finished() && clocked.cycle() < 1000000) {
                wait(clk.posedge_event());
                clocked.tick();
            }
            if (!clocked.numeric_status_pass() || plain.cycle() != clocked.cycle() ||
                plain.dataplane().memory().image() != clocked.dataplane().memory().image()) ++failures;
            const auto report = output_ + "/" + mode;
            clocked.write_reports(report);
            if (!hw::reference_fp64::check_tree_artifact(directory, report).pass) ++failures;
            std::cout << "[t11 SystemC] " << mode << " cycles=" << clocked.cycle()
                      << " identical to plain tick model\n";
        }
        sc_core::sc_stop();
    }
    std::string input_, output_;
};

int sc_main(int argc, char** argv) {
    if (argc != 3) return 2;
    sc_core::sc_clock clock("clock", sc_core::sc_time(1, sc_core::SC_NS));
    TreeClockTest test("tree_test", argv[1], argv[2]);
    test.clk(clock);
    sc_core::sc_start();
    return test.failures ? 1 : 0;
}
