// T06 Panel-LU architecture prototype smoke/trace generator.
//
// This tool runs the architecture prototype on explicitly synthetic workloads
// and writes panel_tasks.csv / panel_commands.csv / panel_events.csv /
// panel_summary.json. It is NOT a numerical execution backend: summary.json
// carries architecture_prototype=true and numerical_validation=not_performed.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "panel_lu/panel_lu_prototype.hpp"

namespace {

namespace pl = hw::panel_lu;

struct Options {
    std::uint32_t front_dim{23};
    std::uint32_t pivot_dim{23};
    std::uint32_t pe_count{2};
    std::uint32_t task_slots{4};
    std::uint32_t dispatcher_depth{32};
    std::uint32_t load_bytes{64};
    std::uint32_t writeback_bytes{64};
    std::uint32_t scan_bytes{64};
    std::string mode{"tail"};
    std::string inject;
    std::string trace_dir{"panel_lu_trace"};
    std::uint32_t seed{7};
    std::uint32_t div_latency{8};
    std::uint64_t task_id_base{0};
    bool abort_second{false};
};

std::uint32_t parse_u32(const char* text, std::uint32_t fallback) {
    try {
        return static_cast<std::uint32_t>(std::stoul(text));
    } catch (...) {
        return fallback;
    }
}

void print_usage() {
    std::cout << "usage: panel_lu_prototype_smoke [options]\n"
                 "  --front-dim N         total front dimension (default 23)\n"
                 "  --pivot-dim N         pivot dimension <= front (default 23)\n"
                 "  --pe-count N          physical shared PEs (default 2)\n"
                 "  --task-slots N        task slots per PE (default 4)\n"
                 "  --dispatcher-depth N  bounded dispatcher queue (default 32)\n"
                 "  --load-bytes N        tile-store load bytes/cycle (default 64)\n"
                 "  --writeback-bytes N   writeback bytes/cycle (default 64)\n"
                 "  --scan-bytes N        scan bytes/cycle (default 64)\n"
                 "  --mode tail|mixed|latency\n"
                 "  --inject zero|tiny|singular|nan\n"
                 "  --seed N\n"
                 "  --div-latency N       corner divide/reciprocal latency (default 8)\n"
                 "  --task-id-base N      diagnostic uint64 task-id base override\n"
                 "  --abort-second        abort the second command mid-run\n"
                 "  --trace-dir DIR       output directory\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--front-dim" && has_value) {
            options.front_dim = parse_u32(argv[++i], options.front_dim);
        } else if (arg == "--pivot-dim" && has_value) {
            options.pivot_dim = parse_u32(argv[++i], options.pivot_dim);
        } else if (arg == "--pe-count" && has_value) {
            options.pe_count = parse_u32(argv[++i], options.pe_count);
        } else if (arg == "--task-slots" && has_value) {
            options.task_slots = parse_u32(argv[++i], options.task_slots);
        } else if (arg == "--dispatcher-depth" && has_value) {
            options.dispatcher_depth = parse_u32(argv[++i], options.dispatcher_depth);
        } else if (arg == "--load-bytes" && has_value) {
            options.load_bytes = parse_u32(argv[++i], options.load_bytes);
        } else if (arg == "--writeback-bytes" && has_value) {
            options.writeback_bytes = parse_u32(argv[++i], options.writeback_bytes);
        } else if (arg == "--scan-bytes" && has_value) {
            options.scan_bytes = parse_u32(argv[++i], options.scan_bytes);
        } else if (arg == "--mode" && has_value) {
            options.mode = argv[++i];
        } else if (arg == "--inject" && has_value) {
            options.inject = argv[++i];
        } else if (arg == "--seed" && has_value) {
            options.seed = parse_u32(argv[++i], options.seed);
        } else if (arg == "--div-latency" && has_value) {
            options.div_latency = parse_u32(argv[++i], options.div_latency);
        } else if (arg == "--task-id-base" && has_value) {
            options.task_id_base = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--abort-second") {
            options.abort_second = true;
        } else if (arg == "--trace-dir" && has_value) {
            options.trace_dir = argv[++i];
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            print_usage();
            return 2;
        }
    }

    pl::PanelLuConfig config;
    config.pe_count = options.pe_count;
    config.task_slots_per_pe = options.task_slots;
    config.dispatcher_queue_depth = options.dispatcher_depth;
    config.load_bytes_per_cycle = options.load_bytes;
    config.writeback_bytes_per_cycle = options.writeback_bytes;
    config.scan_bytes_per_cycle = options.scan_bytes;
    config.timing.div_latency_cycles = options.div_latency;

    pl::PanelLuPrototype prototype(config);
    const pl::ConfigCheck configured = prototype.configure();
    if (!configured.ok) {
        std::cerr << "config rejected: " << configured.reason << '\n';
        return 1;
    }

    const std::uint32_t threshold_bits = 0x3a83126f;  // 0.001f
    auto make_request = [&](std::uint32_t command_id, std::uint32_t front, std::uint32_t pivot,
                            const std::string& label) {
        pl::PanelCommandRequest request;
        request.command_id = command_id;
        request.node_id = command_id;
        request.front_dim = front;
        request.pivot_dim = pivot;
        request.policy = hw::command::PivotPolicy::Static;
        request.threshold_bits = threshold_bits;
        request.tile_size = 16;
        request.synthetic.seed = options.seed + command_id;
        request.label = label;
        if (options.inject == "zero") {
            request.synthetic.injects.push_back(
                pl::SyntheticInject{pl::SyntheticInject::Kind::ZeroDiagonal, 1, 1, 0.0f});
        } else if (options.inject == "tiny") {
            request.synthetic.injects.push_back(
                pl::SyntheticInject{pl::SyntheticInject::Kind::TinyDiagonal, 0, 0, 1.0e-12f});
        } else if (options.inject == "singular") {
            request.synthetic.injects.push_back(
                pl::SyntheticInject{pl::SyntheticInject::Kind::SingularPair, 0, 0, 0.0f});
        } else if (options.inject == "nan") {
            request.synthetic.injects.push_back(
                pl::SyntheticInject{pl::SyntheticInject::Kind::NanEntry, 0, 1, 0.0f});
        }
        return request;
    };

    const bool mixed = options.mode == "mixed";
    const pl::ConfigCheck first = prototype.submit_command(
        make_request(0, options.front_dim, options.pivot_dim, "main"),
        options.task_id_base);
    if (!first.ok) {
        std::cerr << "submit rejected: " << first.reason << '\n';
        return 1;
    }
    if (mixed) {
        const pl::ConfigCheck second = prototype.submit_command(
            make_request(1, options.front_dim, options.pivot_dim, "mixed"));
        if (!second.ok) {
            std::cerr << "submit rejected: " << second.reason << '\n';
            return 1;
        }
    }

    const std::uint64_t max_cycles = 2'000'000;
    std::uint32_t aborted_at = 0;
    while (!prototype.idle() && prototype.cycle() < max_cycles) {
        if (options.abort_second && aborted_at == 0 && prototype.cycle() > 200 &&
            prototype.controller(1) != nullptr) {
            prototype.abort_command(1);
            aborted_at = static_cast<std::uint32_t>(prototype.cycle());
        }
        prototype.tick();
    }

    pl::PanelCommandResponse response;
    while (prototype.poll_response(response)) {
        std::cout << "command " << response.command_id << ": "
                  << hw::dataplane::status_name(response.status)
                  << " pivot_count=" << response.pivot_count
                  << " finish_cycle=" << response.finish_cycle;
        if (!response.reason.empty()) {
            std::cout << " reason=\"" << response.reason << "\"";
        }
        std::cout << '\n';
    }
    if (!prototype.idle()) {
        std::cerr << "prototype did not reach idle within " << max_cycles << " cycles\n";
        return 1;
    }
    prototype.write_reports(options.trace_dir);
    std::cout << "trace: " << options.trace_dir << "/panel_tasks.csv\n";
    std::cout << "summary: " << options.trace_dir << "/panel_summary.json\n";
    if (prototype.control_fault()) {
        std::cerr << "control fault: " << prototype.control_fault_reason() << '\n';
        return 1;
    }
    return 0;
}
