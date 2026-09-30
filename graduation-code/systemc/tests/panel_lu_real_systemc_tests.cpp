// T07 Panel-LU real engine: SystemC clock-driven integration tests.
//
// The CommandDevice (including the shared PanelEngine on BufferManager leases)
// is a plain C++ tick model; the SystemC testbench owns the clock and calls
// tick() exactly once per posedge. These tests verify that a real FP32 panel
// command completes numerically correctly under a SystemC scheduler and that
// the clocked run matches the plain tick model cycle-for-cycle.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <systemc>

#include "controlplane/control_plane.hpp"
#include "control_test_artifact.hpp"
#include "fp64_reference.hpp"

namespace {

namespace cmd = hw::command;
namespace cp = hw::controlplane;
namespace dp = hw::dataplane;
namespace testing = hw::testing;
namespace ref = hw::reference_fp64;

std::uint32_t f32_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<float> make_matrix(std::uint32_t n, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(n) * n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            const int pattern =
                static_cast<int>((i * 7 + j * 13 + seed * 29) % 11) - 5;
            float value = 0.25f * static_cast<float>(pattern);
            if (i == j) {
                value += 4.0f;
            }
            values[static_cast<std::size_t>(i) * n + j] = value;
        }
    }
    return values;
}

cp::ControlPlaneConfig base_config() {
    cp::ControlPlaneConfig config;
    config.max_resident_fronts = 2;
    config.panel_unit_count = 1;
    config.trsm_unit_count = 1;
    config.gemm_unit_count = 1;
    config.solve_unit_count = 1;
    config.assemble_unit_count = 1;
    config.factor_unit_count = 1;
    config.buffer_slot_count = 5;
    config.buffer_capacity_bytes = 64 * 1024;
    config.buffer_port_count = 4;
    config.command_window_depth = 16;
    config.completion_queue_depth = 16;
    config.dma_queue_depth = 8;
    config.backend = cp::ExecBackend::RequireRealKernels;
    config.enable_test_stub = false;
    config.enable_real_panel = true;
    config.memory.startup_cycles = 1;
    config.memory.beat_bytes = 16;
    config.memory.beat_cycles = 1;
    config.dma.beat_bytes = 16;
    config.dma.startup_cycles = 1;
    config.dma.beat_interval_cycles = 1;
    config.panel.pe_count = 1;
    return config;
}

testing::TestArtifact build_artifact() {
    const std::uint32_t n = 16;
    testing::TestNode node;
    node.node_id = 0;
    node.parent_id = -1;
    node.total_dim = n;
    node.pivot_dim = n;
    node.front_values = make_matrix(n, 17);
    testing::TestArtifactOptions options;
    options.threshold_bits = f32_bits(0.0f);
    options.tile_size = 16;
    options.matrix_dim = n;
    return testing::build_test_artifact({node}, options);
}

std::uint32_t command_index(const testing::TestArtifact& artifact, cmd::Opcode opcode) {
    for (std::uint32_t index = 0; index < artifact.commands.size(); ++index) {
        if (artifact.commands[index].opcode == opcode) {
            return index;
        }
    }
    return cmd::NONE;
}

class T07SystemCTests : public sc_core::sc_module {
public:
    sc_core::sc_in<bool> clk;

    SC_CTOR(T07SystemCTests) { SC_THREAD(run_tests); }

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

    void wait_step(cp::CommandDevice& device) {
        wait(clk.posedge_event());
        device.tick();
    }

    void run_tests() {
        testing::TestArtifact artifact = build_artifact();
        const std::uint32_t panel_command =
            command_index(artifact, cmd::Opcode::PanelLu);
        const std::uint32_t factor_command =
            command_index(artifact, cmd::Opcode::StoreFactor);

        cp::CommandDevice device(artifact.artifact, base_config());
        std::string error;
        check(device.start(&error), "clocked real panel: device starts");
        const std::uint64_t start = device.cycle();
        while (!device.finished() && device.cycle() - start < 1000000) {
            wait_step(device);
        }
        check(device.finished(), "clocked real panel: batch finishes");
        check(
            device.completion_status().at(panel_command) == cmd::StatusCode::Ok,
            "clocked real panel: PANEL_LU succeeds");
        check(
            device.completion_status().at(factor_command) == cmd::StatusCode::Ok,
            "clocked real panel: STORE_FACTOR succeeds");
        check(device.panel_engine() != nullptr, "clocked real panel: engine bound");
        check(
            device.panel_engine()->control_fault() == false,
            "clocked real panel: no control fault");

        // Numeric check against the independent FP64 STATIC reference.
        const std::uint32_t n = 16;
        const std::vector<float> matrix = make_matrix(n, 17);
        const auto* l_region = device.boot().region("node.0.factor_l");
        const auto* u_region = device.boot().region("node.0.factor_u");
        check(l_region != nullptr && u_region != nullptr, "clocked real panel: L/U regions");
        if (l_region != nullptr && u_region != nullptr) {
            const auto& image = device.dataplane().memory().image();
            const float* device_l =
                reinterpret_cast<const float*>(image.data() + l_region->offset);
            const float* device_u =
                reinterpret_cast<const float*>(image.data() + u_region->offset);
            std::vector<double> dense_values(matrix.begin(), matrix.end());
            ref::DenseMatrix dense(n, n, dense_values);
            std::vector<std::size_t> variables(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                variables[i] = i;
            }
            const ref::FrontFactor reference = ref::factor_front(
                variables, dense, n, ref::PivotPolicy::Static, 0.0);
            double max_diff = 0.0;
            for (std::uint32_t row = 0; row < n; ++row) {
                for (std::uint32_t col = 0; col < n; ++col) {
                    max_diff = std::max(
                        max_diff,
                        std::fabs(reference.factorization.l.at(row, col) -
                                  device_l[static_cast<std::size_t>(row) * n + col]));
                    max_diff = std::max(
                        max_diff,
                        std::fabs(reference.factorization.u.at(row, col) -
                                  device_u[static_cast<std::size_t>(row) * n + col]));
                }
            }
            check(max_diff < 5.0e-3, "clocked real panel: factors match the FP64 reference");
        }

        // Plain tick model must produce the same cycle count.
        cp::CommandDevice plain(artifact.artifact, base_config());
        std::string plain_error;
        bool plain_ok = plain.start(&plain_error);
        while (plain_ok && !plain.finished() && plain.cycle() < 1000000) {
            plain.tick();
        }
        check(plain_ok && plain.finished(), "plain real panel: batch finishes");
        check(
            plain.cycle() == device.cycle(),
            "clocked and plain tick models agree cycle-for-cycle");

        done_ = true;
        sc_core::sc_stop();
    }
};

}  // namespace

int sc_main(int, char**) {
    // The runtime model must not advance SystemC time by itself.
    const sc_core::sc_time before = sc_core::sc_time_stamp();
    testing::TestArtifact artifact = build_artifact();
    cp::CommandDevice device(artifact.artifact, base_config());
    device.start(nullptr);
    device.tick();
    if (sc_core::sc_time_stamp() != before) {
        std::cerr << "FAIL: device tick advanced SystemC time\n";
        return 1;
    }

    sc_core::sc_clock clock("clk", sc_core::sc_time(1, sc_core::SC_NS));
    T07SystemCTests tests("t07_systemc_tests");
    tests.clk(clock);
    sc_core::sc_start();
    if (!tests.done()) {
        std::cerr << "FAIL: SystemC test thread did not finish\n";
        return 1;
    }
    std::cout << tests.checks() << " checks, " << tests.failures() << " failures\n";
    return tests.failures() == 0 ? 0 : 1;
}
