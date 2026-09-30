// T06 Panel-LU architecture prototype tests (plain C++).
//
// These tests validate the internal task contract, resource invariants,
// backpressure, directed abort/reset and the documented latency rules of the
// architecture prototype. They compare against hand-computed values and
// event-order invariants, not against the prototype's own timing helpers.
// Every run is an architecture prototype with
// numerical_validation=not_performed; no numerical correctness is claimed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "panel_lu/panel_lu_prototype.hpp"
#include "panel_lu/tile_kernels.hpp"

namespace {

namespace cmd = hw::command;
namespace pl = hw::panel_lu;

int failures = 0;
int checks = 0;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::uint32_t threshold_bits_0001() {
    const float value = 0.001f;
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

pl::PanelLuConfig base_config(std::uint32_t pe_count = 1) {
    pl::PanelLuConfig config;
    config.pe_count = pe_count;
    config.task_slots_per_pe = 4;
    config.dispatcher_queue_depth = 32;
    config.max_resident_commands = 2;
    config.load_bytes_per_cycle = 64;
    config.load_latency_cycles = 2;
    config.writeback_bytes_per_cycle = 64;
    config.writeback_latency_cycles = 2;
    config.scan_bytes_per_cycle = 64;
    config.scan_reduce_latency_cycles = 8;
    return config;
}

pl::PanelCommandRequest make_request(
    std::uint32_t command_id,
    std::uint32_t front_dim,
    std::uint32_t pivot_dim,
    std::uint32_t seed = 7) {
    pl::PanelCommandRequest request;
    request.command_id = command_id;
    request.node_id = command_id;
    request.front_dim = front_dim;
    request.pivot_dim = pivot_dim;
    request.policy = cmd::PivotPolicy::Static;
    request.threshold_bits = threshold_bits_0001();
    request.tile_size = 16;
    request.synthetic.seed = seed;
    request.label = "test";
    return request;
}

struct Outcome {
    bool idle{false};
    bool control_fault{false};
    std::string fault;
    std::vector<pl::PanelCommandResponse> responses;
    std::map<std::uint32_t, pl::PanelCommandResponse> by_command;
};

Outcome run(pl::PanelLuPrototype& prototype, std::uint64_t max_cycles = 200000) {
    Outcome outcome;
    prototype.run_until_idle(max_cycles);
    pl::PanelCommandResponse response;
    while (prototype.poll_response(response)) {
        outcome.responses.push_back(response);
        outcome.by_command[response.command_id] = response;
    }
    outcome.idle = prototype.idle();
    outcome.control_fault = prototype.control_fault();
    outcome.fault = prototype.control_fault_reason();
    return outcome;
}

const pl::TaskRecord* find_record(
    const std::vector<pl::TaskRecord>& records,
    std::uint32_t command_id,
    pl::TileMode mode,
    std::uint32_t step,
    std::uint32_t row,
    std::uint32_t col) {
    for (const pl::TaskRecord& record : records) {
        if (record.task.command_id == command_id && record.task.mode == mode &&
            record.task.step == step && record.task.tile_row == row &&
            record.task.tile_col == col) {
            return &record;
        }
    }
    return nullptr;
}

const pl::TileTask* find_graph_task(
    const std::vector<pl::TileTask>& graph,
    pl::TileMode mode,
    std::uint32_t step,
    std::uint32_t row,
    std::uint32_t col) {
    for (const pl::TileTask& task : graph) {
        if (task.mode == mode && task.step == step && task.tile_row == row &&
            task.tile_col == col) {
            return &task;
        }
    }
    return nullptr;
}

bool intervals_overlap(
    std::uint64_t start_a,
    std::uint64_t end_a,
    std::uint64_t start_b,
    std::uint64_t end_b) {
    return start_a < end_b && start_b < end_a;
}

// ---------------------------------------------------------------------------
// Configuration / capability checks
// ---------------------------------------------------------------------------

void test_config_validation() {
    pl::PanelLuPrototype prototype(base_config());
    expect(prototype.configure().ok, "default config accepted");

    auto rejected = [](const pl::PanelLuConfig& config, const std::string& label) {
        pl::PanelLuPrototype candidate(config);
        const pl::ConfigCheck check = candidate.configure();
        expect(!check.ok && !check.reason.empty(), label + " rejected with a reason");
    };
    pl::PanelLuConfig config = base_config();
    config.tile_dim = 8;
    rejected(config, "tile_dim=8");
    config = base_config();
    config.pe_count = 0;
    rejected(config, "pe_count=0");
    config = base_config();
    config.task_slots_per_pe = 0;
    rejected(config, "task_slots=0");
    config = base_config();
    config.dispatcher_queue_depth = 0;
    rejected(config, "dispatcher depth 0");
    config = base_config();
    config.max_resident_commands = 3;
    rejected(config, "max_resident=3");
    config = base_config();
    config.max_front_dim = 513;
    rejected(config, "max_front_dim=513");
    config = base_config();
    config.pe_input_fifo_bytes = 1024;
    rejected(config, "fifo smaller than the largest operand set");
    config = base_config();
    config.pe_input_fifo_bytes = 2047;
    rejected(config, "fifo one byte short of A+B");
    config = base_config();
    config.load_bytes_per_cycle = 2;
    rejected(config, "load budget 2B/cycle");
    config = base_config();
    config.writeback_queue_depth = 0;
    rejected(config, "writeback queue 0");
    config = base_config();
    config.timing.div_latency_cycles = 0;
    rejected(config, "div latency 0");
    config = base_config();
    config.timing.lu_pivot_check_cycles = 0;
    rejected(config, "LU pivot check 0");
}

void test_front_capability_checks() {
    pl::PanelLuPrototype prototype(base_config());
    expect(prototype.configure().ok, "config accepted");
    // Unsupported tile_size must be rejected, never silently ignored.
    pl::PanelCommandRequest request = make_request(0, 32, 32);
    request.tile_size = 8;
    pl::ConfigCheck check = prototype.submit_command(request);
    expect(
        !check.ok && check.status == cmd::StatusCode::BadDescriptor,
        "tile_size=8 rejected with BAD_DESCRIPTOR");
    // DYNAMIC_PANEL is an explicit unsupported capability in T06.
    request = make_request(0, 32, 32);
    request.policy = cmd::PivotPolicy::DynamicPanel;
    check = prototype.submit_command(request);
    expect(
        !check.ok && check.status == cmd::StatusCode::PolicyNotSupported,
        "DYNAMIC_PANEL rejected with POLICY_NOT_SUPPORTED");
    // Illegal policy encoding.
    request = make_request(0, 32, 32);
    request.policy = static_cast<cmd::PivotPolicy>(0);
    check = prototype.submit_command(request);
    expect(
        !check.ok && check.status == cmd::StatusCode::BadDescriptor,
        "policy 0 rejected with BAD_DESCRIPTOR");
    // Zero / oversized dimensions and tile_size=0.
    request = make_request(0, 32, 0);
    expect(!prototype.submit_command(request).ok, "pivot_dim=0 rejected");
    request = make_request(0, 32, 33);
    expect(!prototype.submit_command(request).ok, "pivot_dim>front rejected");
    request = make_request(0, 600, 32);
    expect(!prototype.submit_command(request).ok, "front>max_front rejected");
    request = make_request(0, 32, 32);
    request.tile_size = 0;
    expect(!prototype.submit_command(request).ok, "tile_size=0 rejected");

    // Threshold bits must be validated on the submitted request, never a stale
    // or defaulted value: NaN/Inf/negative/>=1 are all rejected up front.
    const std::uint32_t bad_thresholds[] = {
        0x7fc00000u,  // NaN
        0x7f800000u,  // +Inf
        0xff800000u,  // -Inf
        0xbf800000u,  // -1.0
        0x3f800000u,  // 1.0
        0x40000000u,  // 2.0
        0x7f7fffffu,  // +FLT_MAX
    };
    for (const std::uint32_t bits : bad_thresholds) {
        pl::PanelCommandRequest bad = make_request(0, 32, 32);
        bad.threshold_bits = bits;
        const pl::ConfigCheck bad_check = prototype.submit_command(bad);
        expect(
            !bad_check.ok && bad_check.status == cmd::StatusCode::BadDescriptor,
            "threshold bits " + std::to_string(bits) + " rejected");
    }
    {
        pl::PanelCommandRequest zero = make_request(0, 32, 32);
        zero.threshold_bits = 0x00000000u;  // 0.0 is a legal ratio
        expect(prototype.submit_command(zero).ok, "threshold 0.0 accepted");
    }

    // Direct internal task validation: dims > 16 or mask mismatch rejected.
    pl::TileTask task;
    task.mode = pl::TileMode::GemmUpdate;
    task.dims = {17, 16, 16};
    task.mask = {17, 16};
    task.dst = pl::TileOperand{{0, 0}, 0, 64, 0, true};
    task.lhs = pl::TileOperand{{1, 0}, 0, 64, 0, true};
    task.rhs = pl::TileOperand{{2, 0}, 0, 64, 0, true};
    expect(
        !pl::validate_tile_task(task, base_config()).ok, "M=17 internal task rejected");
    task.dims = {16, 16, 16};
    task.mask = {15, 16};
    expect(
        !pl::validate_tile_task(task, base_config()).ok, "mask != dims rejected");
    task.mask = {16, 16};
    task.lhs.stride_bytes = 4;
    expect(
        !pl::validate_tile_task(task, base_config()).ok, "tiny stride rejected");
    // TRSM_LEFT requires result rows == k.
    pl::TileTask trsm;
    trsm.mode = pl::TileMode::TrsmLeft;
    trsm.dims = {8, 16, 16};
    trsm.mask = {8, 16};
    trsm.dst = pl::TileOperand{{0, 0}, 0, 64, 0, true};
    trsm.lhs = pl::TileOperand{{1, 0}, 0, 64, 0, true};
    trsm.rhs = pl::TileOperand{{2, 0}, 0, 64, 0, true};
    expect(
        !pl::validate_tile_task(trsm, base_config()).ok,
        "TRSM_LEFT with m != k rejected");
}

// ---------------------------------------------------------------------------
// Task graph generation, tiles and tails
// ---------------------------------------------------------------------------

void test_task_graph_full_tiles() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure");
    expect(prototype.submit_command(make_request(0, 32, 32)).ok, "submit p=32");
    const Outcome outcome = run(prototype);
    expect(outcome.idle && !outcome.control_fault, "p=32 completes without fault");
    const pl::PanelCommandResponse& response = outcome.by_command.at(0);
    expect(response.status == cmd::StatusCode::Ok, "p=32 OK");
    expect(response.pivot_count == 32, "p=32 pivot_count=32");
    expect(response.pvector_written, "p=32 writes identity P-vector");
    expect(response.pvector_bytes == 32 * 4, "p=32 P-vector bytes");

    const std::vector<pl::TileTask>* graph = prototype.command_task_graph(0);
    expect(graph != nullptr && graph->size() == 5, "p=32 generates 5 tasks");
    const pl::TileTask* diag0 = find_graph_task(*graph, pl::TileMode::LuDiag, 0, 0, 0);
    expect(diag0 != nullptr && diag0->dims.m == 16 && diag0->dims.n == 16 &&
               diag0->dims.k == 16,
           "diag(0,0) is 16x16");
    expect(diag0 != nullptr && diag0->deps.empty(), "diag(0,0) has no deps");
    const pl::TileTask* left = find_graph_task(*graph, pl::TileMode::TrsmLeft, 0, 0, 1);
    const pl::TileTask* right = find_graph_task(*graph, pl::TileMode::TrsmRight, 0, 1, 0);
    expect(left != nullptr && left->deps.size() == 1, "TRSM_LEFT depends on diag");
    expect(right != nullptr && right->deps.size() == 1, "TRSM_RIGHT depends on diag");
    const pl::TileTask* gemm = find_graph_task(*graph, pl::TileMode::GemmUpdate, 0, 1, 1);
    expect(
        gemm != nullptr && gemm->deps.size() == 2 &&
            std::set<std::uint64_t>(gemm->deps.begin(), gemm->deps.end()) ==
                std::set<std::uint64_t>({left->task_id, right->task_id}),
        "GEMM_UPDATE(1,1) depends on both TRSM tiles");
    const pl::TileTask* diag1 = find_graph_task(*graph, pl::TileMode::LuDiag, 1, 1, 1);
    expect(diag1 != nullptr && diag1->dst.version == 1, "step-1 diag consumes version 1");
    expect(
        diag0 != nullptr && diag1 != nullptr &&
            diag0->front_pre_elimination_max_abs ==
                diag1->front_pre_elimination_max_abs &&
            diag0->threshold_product == diag1->threshold_product,
        "all diagonal tiles use the same latched front scale");
    expect(response.front_pre_elimination_max_abs > 3.0, "front maxabs measured");
    expect(
        response.threshold_product > 0.0 &&
            response.threshold_product < response.front_pre_elimination_max_abs,
        "threshold product = ratio * maxabs");
}

void test_tail_tiles() {
    struct Case {
        std::uint32_t pivot;
        std::uint32_t tasks;
        std::uint32_t diag0;
        std::uint32_t diag1;
        std::uint32_t left_n;
        std::uint32_t right_m;
        std::uint32_t gemm_m;
    };
    const Case cases[] = {
        {1, 1, 1, 0, 0, 0, 0},
        {15, 1, 15, 0, 0, 0, 0},
        {16, 1, 16, 0, 0, 0, 0},
        {17, 5, 16, 1, 1, 1, 1},
        {23, 5, 16, 7, 7, 7, 7},
        {31, 5, 16, 15, 15, 15, 15},
        {33, 14, 16, 16, 16, 16, 16},
    };
    for (const Case& test_case : cases) {
        pl::PanelLuPrototype prototype(base_config(1));
        expect(prototype.configure().ok, "configure tail case");
        expect(
            prototype.submit_command(make_request(0, test_case.pivot, test_case.pivot)).ok,
            "submit tail p=" + std::to_string(test_case.pivot));
        const Outcome outcome = run(prototype);
        expect(
            outcome.by_command.at(0).status == cmd::StatusCode::Ok,
            "tail p=" + std::to_string(test_case.pivot) + " OK");
        const std::vector<pl::TileTask>* graph = prototype.command_task_graph(0);
        expect(
            graph != nullptr && graph->size() == test_case.tasks,
            "tail p=" + std::to_string(test_case.pivot) + " task count");
        if (graph == nullptr) {
            continue;
        }
        const pl::TileTask* diag0 = find_graph_task(*graph, pl::TileMode::LuDiag, 0, 0, 0);
        expect(
            diag0 != nullptr && diag0->dims.k == test_case.diag0,
            "tail p=" + std::to_string(test_case.pivot) + " diag0 size");
        if (test_case.diag1 != 0) {
            const pl::TileTask* diag1 =
                find_graph_task(*graph, pl::TileMode::LuDiag, 1, 1, 1);
            expect(
                diag1 != nullptr && diag1->dims.k == test_case.diag1,
                "tail p=" + std::to_string(test_case.pivot) + " diag1 size");
        }
        const pl::TileTask* left = find_graph_task(*graph, pl::TileMode::TrsmLeft, 0, 0, 1);
        if (test_case.left_n != 0) {
            expect(
                left != nullptr && left->dims.n == test_case.left_n &&
                    left->mask.cols == test_case.left_n,
                "tail p=" + std::to_string(test_case.pivot) + " left tail width");
        } else {
            expect(left == nullptr, "no TRSM row panel for single tile");
        }
        const pl::TileTask* right =
            find_graph_task(*graph, pl::TileMode::TrsmRight, 0, 1, 0);
        if (test_case.right_m != 0) {
            expect(
                right != nullptr && right->dims.m == test_case.right_m,
                "tail p=" + std::to_string(test_case.pivot) + " right tail height");
        }
        const pl::TileTask* gemm =
            find_graph_task(*graph, pl::TileMode::GemmUpdate, 0, 1, 1);
        if (test_case.gemm_m != 0) {
            expect(
                gemm != nullptr && gemm->dims.m == test_case.gemm_m &&
                    gemm->dims.n == test_case.gemm_m,
                "tail p=" + std::to_string(test_case.pivot) + " gemm tail");
        }
    }
}

// ---------------------------------------------------------------------------
// PE exclusivity, cross-mode contention, same-target ordering
// ---------------------------------------------------------------------------

void test_pe_exclusion_and_cross_mode() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure");
    expect(prototype.submit_command(make_request(0, 40, 40)).ok, "submit p=40");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "p=40 OK");

    const auto& records = prototype.task_records();
    for (std::size_t i = 0; i < records.size(); ++i) {
        for (std::size_t j = i + 1; j < records.size(); ++j) {
            if (records[i].pe_id != records[j].pe_id) {
                continue;
            }
            expect(
                !intervals_overlap(
                    records[i].exec_start_cycle,
                    records[i].exec_done_cycle,
                    records[j].exec_start_cycle,
                    records[j].exec_done_cycle),
                "single PE never overlaps two array executions");
        }
    }
    std::set<std::string> modes_on_pe;
    for (const pl::TaskRecord& record : records) {
        if (record.pe_id == 0) {
            modes_on_pe.insert(pl::tile_mode_name(record.task.mode));
        }
    }
    expect(modes_on_pe.size() >= 3, "one PE executes multiple modes (cross-mode sharing)");

    // Two PEs: tasks may overlap across distinct PEs, but never on the same PE.
    pl::PanelLuPrototype dual(base_config(2));
    expect(dual.configure().ok, "configure dual");
    expect(dual.submit_command(make_request(0, 40, 40)).ok, "submit p=40 dual");
    const Outcome dual_outcome = run(dual);
    expect(dual_outcome.by_command.at(0).status == cmd::StatusCode::Ok, "dual OK");
    const auto& dual_records = dual.task_records();
    bool cross_pe_overlap = false;
    for (std::size_t i = 0; i < dual_records.size(); ++i) {
        for (std::size_t j = i + 1; j < dual_records.size(); ++j) {
            if (dual_records[i].pe_id == dual_records[j].pe_id) {
                expect(
                    !intervals_overlap(
                        dual_records[i].exec_start_cycle,
                        dual_records[i].exec_done_cycle,
                        dual_records[j].exec_start_cycle,
                        dual_records[j].exec_done_cycle),
                    "dual PE never overlaps on the same PE");
                continue;
            }
            if (intervals_overlap(
                    dual_records[i].exec_start_cycle,
                    dual_records[i].exec_done_cycle,
                    dual_records[j].exec_start_cycle,
                    dual_records[j].exec_done_cycle)) {
                cross_pe_overlap = true;
            }
        }
    }
    expect(cross_pe_overlap, "distinct PEs execute concurrently when work is independent");
}

void test_same_target_ordering() {
    pl::PanelLuPrototype prototype(base_config(2));
    expect(prototype.configure().ok, "configure");
    expect(prototype.submit_command(make_request(0, 40, 40)).ok, "submit p=40");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "p=40 OK");

    // Group committed records by destination slot; versions must be strictly
    // sequential and writebacks must not overlap.
    std::map<std::uint32_t, std::vector<const pl::TaskRecord*>> by_slot;
    for (const pl::TaskRecord& record : prototype.task_records()) {
        if (record.committed) {
            by_slot[record.task.dst.handle.slot].push_back(&record);
        }
    }
    for (auto& entry : by_slot) {
        std::vector<const pl::TaskRecord*> writers = entry.second;
        std::sort(
            writers.begin(),
            writers.end(),
            [](const pl::TaskRecord* a, const pl::TaskRecord* b) {
                return a->task.dst.version < b->task.dst.version;
            });
        for (std::size_t i = 0; i < writers.size(); ++i) {
            expect(
                writers[i]->task.dst.version == i,
                "per-tile dst versions are sequential from 0");
            expect(
                writers[i]->task.output_version == writers[i]->task.dst.version + 1,
                "output version = dst version + 1");
            if (i > 0) {
                expect(
                    writers[i]->writeback_done_cycle >= writers[i - 1]->writeback_done_cycle,
                    "same-target writebacks commit in version order");
            }
        }
    }
    expect(by_slot.size() >= 4, "multiple target tiles tracked");
}

void test_dependency_not_ready() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure");
    expect(prototype.submit_command(make_request(0, 32, 32)).ok, "submit p=32");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "p=32 OK");

    const auto& records = prototype.task_records();
    const pl::TileTask* graph = nullptr;
    const auto* graph_ptr = prototype.command_task_graph(0);
    expect(graph_ptr != nullptr, "graph available");
    graph = graph_ptr->data();
    for (const pl::TaskRecord& record : records) {
        for (const std::uint64_t dep : record.task.deps) {
            const pl::TaskRecord* producer = nullptr;
            for (const pl::TaskRecord& candidate : records) {
                if (candidate.task.task_id == dep) {
                    producer = &candidate;
                    break;
                }
            }
            expect(producer != nullptr, "dependency has a producer record");
            if (producer != nullptr) {
                expect(
                    record.dispatch_cycle > producer->writeback_done_cycle,
                    "dependent dispatch only after producer result is visible");
            }
        }
    }
    (void)graph;
}

// ---------------------------------------------------------------------------
// Backpressure
// ---------------------------------------------------------------------------

void test_queue_full_backpressure() {
    pl::PanelLuConfig config = base_config(1);
    config.dispatcher_queue_depth = 1;
    config.task_slots_per_pe = 1;
    pl::PanelLuPrototype prototype(config);
    expect(prototype.configure().ok, "configure narrow queue");
    expect(prototype.submit_command(make_request(0, 64, 64)).ok, "submit p=64");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "narrow queue completes");
    expect(
        outcome.by_command.at(0).stall_dispatcher_full > 0,
        "dispatcher-full stalls observed");
    expect(
        prototype.dispatcher().queue_depth() <= 1 && !outcome.control_fault,
        "bounded queue never exceeds its depth");
}

void test_port_and_writeback_backpressure() {
    pl::PanelLuConfig config = base_config(1);
    config.writeback_bytes_per_cycle = 4;
    config.writeback_queue_depth = 1;
    config.writeback_latency_cycles = 2;
    pl::PanelLuPrototype prototype(config);
    expect(prototype.configure().ok, "configure slow writeback");
    expect(prototype.submit_command(make_request(0, 32, 32)).ok, "submit p=32");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "slow writeback OK");
    std::uint64_t wb_stalls = 0;
    for (const pl::TaskRecord& record : prototype.task_records()) {
        wb_stalls += record.stall_writeback_cycles;
        if (record.committed) {
            expect(
                record.bytes_read == pl::task_load_bytes(record.task),
                "load byte accounting matches the timing model input");
            expect(
                record.bytes_written == pl::task_writeback_bytes(record.task),
                "writeback byte accounting matches the timing model output");
        }
    }
    expect(wb_stalls > 0, "writeback backpressure observed");

    pl::PanelLuConfig load_config = base_config(2);
    load_config.load_bytes_per_cycle = 16;
    pl::PanelLuPrototype load_prototype(load_config);
    expect(load_prototype.configure().ok, "configure slow load");
    expect(load_prototype.submit_command(make_request(0, 32, 32)).ok, "submit slow load");
    const Outcome load_outcome = run(load_prototype);
    expect(load_outcome.by_command.at(0).status == cmd::StatusCode::Ok, "slow load OK");
    std::uint64_t port_stalls = 0;
    for (const pl::TaskRecord& record : load_prototype.task_records()) {
        port_stalls += record.stall_port_cycles;
    }
    expect(port_stalls > 0, "load port backpressure observed");
}

// ---------------------------------------------------------------------------
// Hand-computed latency examples
// ---------------------------------------------------------------------------

void test_latency_handcalc() {
    // Configuration: load 64B/cycle (+2 latency), writeback 64B/cycle (+2),
    // scan 64B/cycle (+8 reduce), 16x16 array, fill=2, k II=1, drain=1 row/cyc,
    // div latency=8, LU update=1.
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure latency case");
    expect(prototype.submit_command(make_request(0, 16, 16)).ok, "submit p=16");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "p=16 OK");
    const pl::TaskRecord* record = find_record(
        prototype.task_records(), 0, pl::TileMode::LuDiag, 0, 0, 0);
    expect(record != nullptr, "p=16 diag record");
    if (record != nullptr) {
        // Hand calculation:
        // scan 1024B/64 = 16 ticks -> scan_done 16, reduce +8 -> produce at 25.
        // dispatch 26; load 1024/64 = 16 ticks -> load_done 42; +2 -> exec 44.
        // exec = (k-1) * (div_latency + update) + pivot_check
        //      = 15 * (8 + 1) + 1 = 136 -> 180.
        // writeback 1024/64 = 16 ticks (182..197) +2 -> 199.
        expect(record->dispatch_cycle == 26, "diag dispatch cycle = 26");
        expect(record->load_done_cycle == 42, "diag load_done cycle = 42");
        expect(record->exec_start_cycle == 44, "diag exec_start cycle = 44");
        expect(record->exec_done_cycle == 180, "diag exec_done cycle = 180");
        expect(record->writeback_done_cycle == 199, "diag writeback_done cycle = 199");
        expect(record->bytes_read == 1024 && record->bytes_written == 1024, "diag bytes");
        expect(
            record->exec_done_cycle - record->exec_start_cycle == 136,
            "LU_DIAG occupancy = 15 dependent stages (8+1) + final pivot check");
    }

    // Full 16x16x16 GEMM_UPDATE inside p=32 step 0.
    pl::PanelLuPrototype gemm_prototype(base_config(1));
    expect(gemm_prototype.configure().ok, "configure gemm case");
    expect(gemm_prototype.submit_command(make_request(0, 32, 32)).ok, "submit p=32");
    const Outcome gemm_outcome = run(gemm_prototype);
    expect(gemm_outcome.by_command.at(0).status == cmd::StatusCode::Ok, "p=32 OK");
    const pl::TaskRecord* gemm = find_record(
        gemm_prototype.task_records(), 0, pl::TileMode::GemmUpdate, 0, 1, 1);
    expect(gemm != nullptr, "gemm record");
    if (gemm != nullptr) {
        // Hand calculation with the corrected LU stage chain and the
        // TRSM_RIGHT reciprocal precompute:
        // step-0 diag: dispatch 74, exec 92 -> 228 (136), wb_done 247.
        // TRSM_LEFT commits 335. TRSM_RIGHT (m=16,k=16) exec =
        // fill(2) + recip(8+15) + k(16) + m(16) = 57: exec 316 -> 373,
        // wb_done 392 (shared load port; starts at 316 when the array frees).
        // GEMM enqueue 392, dispatch 393: load (3072B/64)=48 -> 441,
        // +2 -> exec 443, exec = fill(2)+k(16)+drain(16) = 34 -> 477,
        // writeback 1024/64 = 16 ticks (479..494) +2 -> 496.
        expect(gemm->dispatch_cycle == 393, "gemm dispatch cycle = 393");
        expect(gemm->load_done_cycle == 441, "gemm load_done cycle = 441");
        expect(gemm->exec_start_cycle == 443, "gemm exec_start cycle = 443");
        expect(gemm->exec_done_cycle == 477, "gemm exec_done cycle = 477");
        expect(gemm->writeback_done_cycle == 496, "gemm writeback_done cycle = 496");
        expect(gemm->bytes_read == 3072 && gemm->bytes_written == 1024, "gemm bytes");
        expect(
            gemm->exec_done_cycle - gemm->exec_start_cycle == 34,
            "16x16x16 GEMM = 2 + 16 + 16 array cycles");
    }

    // Timing helper formulas are exposed for documentation only; check that a
    // hand-built task maps to the documented closed form.
    pl::TileTask task;
    task.mode = pl::TileMode::TrsmLeft;
    task.dims = {16, 7, 16};
    expect(
        pl::task_exec_cycles(task, pl::PeTimingConfig{}) == 2 + 16 + 7,
        "TRSM_LEFT closed form = fill + k + n");
    task.mode = pl::TileMode::TrsmRight;
    task.dims = {7, 16, 16};
    // Reciprocal precompute (8 + 15) + solve (16 + 7) + fill.
    expect(
        pl::task_exec_cycles(task, pl::PeTimingConfig{}) == 2 + (8 + 15) + 16 + 7,
        "TRSM_RIGHT closed form = fill + recip(div+k-1) + k + m");
    task.mode = pl::TileMode::LuDiag;
    task.dims = {7, 7, 7};
    expect(
        pl::task_exec_cycles(task, pl::PeTimingConfig{}) == (7 - 1) * (8 + 1) + 1,
        "LU_DIAG closed form = (k-1)*(div+update) + pivot check");
}

// ---------------------------------------------------------------------------
// Data-dependent failures and scan dependency
// ---------------------------------------------------------------------------

void test_scan_dependency_and_finite_check() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure");
    expect(prototype.submit_command(make_request(0, 32, 32)).ok, "submit p=32");
    const Outcome outcome = run(prototype);
    const pl::PanelCommandResponse& response = outcome.by_command.at(0);
    // Scan: 32*32*4 = 4096B / 64 = 64 ticks, +8 reduce -> done at 64, scale at 72.
    expect(response.scan_done_cycle == 64, "scan done cycle = ceil(n^2*4/64)");
    expect(response.scan_bytes == 4096, "scan byte count");
    std::uint64_t first_dispatch = UINT64_MAX;
    for (const pl::TaskRecord& record : prototype.task_records()) {
        first_dispatch = std::min(first_dispatch, record.dispatch_cycle);
    }
    expect(
        first_dispatch > response.scan_done_cycle,
        "no panel task is generated before the full-front scan completes");

    // Two resident commands with a narrow scan port: scan contention is
    // observable through the per-command stall counter.
    pl::PanelLuConfig scan_config = base_config(1);
    scan_config.scan_bytes_per_cycle = 16;
    pl::PanelLuPrototype scan_prototype(scan_config);
    expect(scan_prototype.configure().ok, "configure scan contention");
    expect(scan_prototype.submit_command(make_request(0, 32, 32)).ok, "submit scan 0");
    expect(scan_prototype.submit_command(make_request(1, 32, 32, 9)).ok, "submit scan 1");
    const Outcome scan_outcome = run(scan_prototype);
    expect(scan_outcome.idle, "scan contention run idle");
    expect(
        scan_outcome.by_command.at(0).stall_scan_port +
                scan_outcome.by_command.at(1).stall_scan_port >
            0,
        "shared scan port contention is observable");

    pl::PanelCommandRequest nan_request = make_request(0, 32, 32);
    nan_request.synthetic.injects.push_back(
        pl::SyntheticInject{pl::SyntheticInject::Kind::NanEntry, 3, 5, 0.0f});
    pl::PanelLuPrototype nan_prototype(base_config(1));
    expect(nan_prototype.configure().ok, "configure nan");
    expect(nan_prototype.submit_command(nan_request).ok, "submit nan");
    const Outcome nan_outcome = run(nan_prototype);
    expect(
        nan_outcome.by_command.at(0).status == cmd::StatusCode::NonFiniteValue,
        "NaN front fails with NON_FINITE_VALUE");
    expect(
        nan_prototype.task_records().empty(),
        "no tile task runs when the front scan rejects non-finite input");
    expect(!nan_outcome.by_command.at(0).pvector_written, "no P-vector on scan failure");
}

void test_data_dependent_failure() {
    // Zero input pivot -> STATIC_PIVOT_FAILURE at column 1.
    {
        pl::PanelCommandRequest request = make_request(0, 32, 32);
        request.synthetic.injects.push_back(
            pl::SyntheticInject{pl::SyntheticInject::Kind::ZeroDiagonal, 1, 1, 0.0f});
        pl::PanelLuPrototype prototype(base_config(1));
        expect(prototype.configure().ok, "configure zero");
        expect(prototype.submit_command(request).ok, "submit zero");
        const Outcome outcome = run(prototype);
        const pl::PanelCommandResponse& response = outcome.by_command.at(0);
        expect(
            response.status == cmd::StatusCode::StaticPivotFailure,
            "zero pivot -> STATIC_PIVOT_FAILURE");
        expect(response.failed_column == 1 && response.pivot_count == 1, "failure column 1");
        expect(!response.pvector_written, "no P-vector on pivot failure");
        const std::vector<pl::TileTask>* graph = prototype.command_task_graph(0);
        expect(graph != nullptr && graph->size() == 1, "no later tasks after pivot failure");
        const pl::TaskRecord* diag =
            find_record(prototype.task_records(), 0, pl::TileMode::LuDiag, 0, 0, 0);
        expect(diag != nullptr, "failing diag recorded");
        if (diag != nullptr) {
            // Failure at local column 1: 1*(1+1) + 8 = 10 cycles, not the full
            // successful 39.
            expect(
                diag->exec_done_cycle - diag->exec_start_cycle == 10,
                "failure progress = 1 stage interval (8+1) + pivot check");
        }
    }
    // Singular pair inside one diagonal tile: the input diagonal is non-zero
    // but row 1 is a copy of row 0 from column 0, so after eliminating column 0
    // the next pivot becomes exactly zero during the same LU_DIAG.
    {
        pl::PanelCommandRequest request = make_request(0, 16, 16);
        request.synthetic.injects.push_back(
            pl::SyntheticInject{pl::SyntheticInject::Kind::SingularPair, 0, 0, 0.0f});
        pl::PanelLuPrototype prototype(base_config(1));
        expect(prototype.configure().ok, "configure singular");
        expect(prototype.submit_command(request).ok, "submit singular");
        const Outcome outcome = run(prototype);
        const pl::PanelCommandResponse& response = outcome.by_command.at(0);
        expect(
            response.status == cmd::StatusCode::StaticPivotFailure,
            "singular pair -> STATIC_PIVOT_FAILURE");
        expect(
            response.failed_column == 1,
            "failure is detected on the eliminated pivot (column 1)");
        const pl::TaskRecord* diag =
            find_record(prototype.task_records(), 0, pl::TileMode::LuDiag, 0, 0, 0);
        expect(diag != nullptr, "diag recorded");
        if (diag != nullptr) {
            expect(
                diag->exec_done_cycle - diag->exec_start_cycle == 10,
                "eliminated-pivot failure reports actual progress (9 + 1)");
        }
        expect(
            response.tasks_generated == 1,
            "only the failing diagonal task is generated");
    }
    // Below-threshold pivot -> PIVOT_UNSTABLE.
    {
        pl::PanelCommandRequest request = make_request(0, 16, 16);
        request.synthetic.injects.push_back(
            pl::SyntheticInject{pl::SyntheticInject::Kind::TinyDiagonal, 0, 0, 1.0e-12f});
        pl::PanelLuPrototype prototype(base_config(1));
        expect(prototype.configure().ok, "configure tiny");
        expect(prototype.submit_command(request).ok, "submit tiny");
        const Outcome outcome = run(prototype);
        const pl::PanelCommandResponse& response = outcome.by_command.at(0);
        expect(
            response.status == cmd::StatusCode::PivotUnstable,
            "tiny pivot -> PIVOT_UNSTABLE");
        expect(response.failed_column == 0, "unstable pivot at column 0");
        const pl::TaskRecord* diag =
            find_record(prototype.task_records(), 0, pl::TileMode::LuDiag, 0, 0, 0);
        expect(
            diag != nullptr && diag->exec_done_cycle - diag->exec_start_cycle == 1,
            "unstable pivot fails at the first pivot check (1 cycle)");
    }
}

// ---------------------------------------------------------------------------
// Directed abort and reset
// ---------------------------------------------------------------------------

void test_directed_abort() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure abort");
    expect(prototype.submit_command(make_request(0, 64, 64)).ok, "submit cmd 0");
    expect(prototype.submit_command(make_request(1, 64, 64)).ok, "submit cmd 1");
    auto committed_count = [&prototype](std::uint32_t command_id) {
        std::uint32_t count = 0;
        for (const pl::TaskRecord& record : prototype.task_records()) {
            if (record.task.command_id == command_id && record.committed) {
                ++count;
            }
        }
        return count;
    };
    while ((committed_count(0) == 0 || committed_count(1) == 0) &&
           prototype.cycle() < 100000) {
        prototype.tick();
    }
    expect(
        committed_count(0) > 0 && committed_count(1) > 0,
        "both commands commit work before the abort");
    prototype.abort_command(1);
    const Outcome outcome = run(prototype);
    expect(outcome.idle && !outcome.control_fault, "abort run completes without fault");
    expect(
        outcome.by_command.at(0).status == cmd::StatusCode::Ok,
        "aborted command does not disturb the other command");
    expect(
        outcome.by_command.at(1).status == cmd::StatusCode::Aborted,
        "targeted command reports ABORTED");
    expect(
        !outcome.by_command.at(1).pvector_written,
        "aborted command never publishes the P-vector");
    bool cancelled_seen = false;
    bool committed_for_aborted = false;
    for (const pl::TaskRecord& record : prototype.task_records()) {
        if (record.task.command_id == 1 && record.cancelled) {
            cancelled_seen = true;
        }
        if (record.task.command_id == 1 && record.committed) {
            committed_for_aborted = true;
        }
    }
    expect(cancelled_seen, "aborted command has cancelled tasks");
    expect(committed_for_aborted, "tasks already committed before the abort stay committed");
    expect(
        prototype.pes().size() == 1 && prototype.pes()[0].active_task_count() == 0,
        "all PE slots released after abort");
}

void test_reset_invalidates_old_work() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure reset");
    expect(prototype.submit_command(make_request(0, 64, 64)).ok, "submit before reset");
    for (int i = 0; i < 120; ++i) {
        prototype.tick();
    }
    const std::uint32_t generation_before = prototype.store().generation(0);
    const std::uint64_t epoch_before = prototype.epoch();
    prototype.reset();
    expect(prototype.epoch() == epoch_before + 1, "reset advances the epoch");
    expect(
        prototype.store().generation(0) == generation_before + 1,
        "reset invalidates old buffer handles");
    expect(prototype.task_records().empty(), "reset drops old task records");
    pl::PanelCommandResponse stale;
    expect(!prototype.poll_response(stale), "reset drops old command responses");
    for (int i = 0; i < 20; ++i) {
        prototype.tick();
    }
    expect(prototype.idle(), "idle after reset ticks");
    expect(
        prototype.store().commit_count(0) == 0,
        "old in-flight tasks cannot write back after reset");
    // A new command with the same id can run after reset.
    expect(prototype.submit_command(make_request(0, 32, 32)).ok, "resubmit after reset");
    const Outcome outcome = run(prototype);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "post-reset run OK");
}

// ---------------------------------------------------------------------------
// Two-command contention and reproducibility
// ---------------------------------------------------------------------------

void test_dual_command_fairness() {
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure fairness");
    expect(prototype.submit_command(make_request(0, 40, 40, 7)).ok, "submit cmd 0");
    expect(prototype.submit_command(make_request(1, 40, 40, 11)).ok, "submit cmd 1");
    const Outcome outcome = run(prototype);
    expect(
        outcome.by_command.at(0).status == cmd::StatusCode::Ok &&
            outcome.by_command.at(1).status == cmd::StatusCode::Ok,
        "both commands complete on one PE");
    // Fairness: both commands get a task committed before the first command
    // finishes all of its tasks.
    std::uint64_t first_commit[2] = {UINT64_MAX, UINT64_MAX};
    std::uint64_t last_commit[2] = {0, 0};
    for (const pl::TaskRecord& record : prototype.task_records()) {
        if (!record.committed) {
            continue;
        }
        first_commit[record.task.command_id] =
            std::min(first_commit[record.task.command_id], record.writeback_done_cycle);
        last_commit[record.task.command_id] =
            std::max(last_commit[record.task.command_id], record.writeback_done_cycle);
    }
    expect(
        first_commit[0] < last_commit[1] && first_commit[1] < last_commit[0],
        "round-robin keeps both commands progressing (no starvation)");
    // Thread-of-execution exclusivity still holds.
    const auto& records = prototype.task_records();
    for (std::size_t i = 0; i < records.size(); ++i) {
        for (std::size_t j = i + 1; j < records.size(); ++j) {
            if (records[i].pe_id != records[j].pe_id) {
                continue;
            }
            expect(
                !intervals_overlap(
                    records[i].exec_start_cycle,
                    records[i].exec_done_cycle,
                    records[j].exec_start_cycle,
                    records[j].exec_done_cycle),
                "cross-command tasks never overlap on one PE");
        }
    }
}

std::string records_signature(const std::vector<pl::TaskRecord>& records) {
    std::ostringstream out;
    for (const pl::TaskRecord& record : records) {
        out << record.task.task_id << ':' << record.task.command_id << ':'
            << pl::tile_mode_name(record.task.mode) << ':' << record.task.step << ':'
            << record.task.tile_row << ':' << record.task.tile_col << ':'
            << record.pe_id << ':' << hw::dataplane::status_name(record.status) << ':'
            << record.committed << ':' << record.enqueue_cycle << ':'
            << record.dispatch_cycle << ':' << record.exec_start_cycle << ':'
            << record.exec_done_cycle << ':' << record.writeback_done_cycle << ';';
    }
    return out.str();
}

void test_reproducible_trace() {
    auto run_case = [](std::string* signature, std::string* summary) {
        pl::PanelLuPrototype prototype(base_config(2));
        if (!prototype.configure().ok) {
            return false;
        }
        if (!prototype.submit_command(make_request(0, 23, 23, 3)).ok ||
            !prototype.submit_command(make_request(1, 40, 40, 5)).ok) {
            return false;
        }
        const Outcome outcome = run(prototype);
        if (!outcome.idle) {
            return false;
        }
        *signature = records_signature(prototype.task_records());
        *summary = prototype.summary_json();
        return true;
    };
    std::string signature_a;
    std::string summary_a;
    std::string signature_b;
    std::string summary_b;
    expect(run_case(&signature_a, &summary_a), "first reproducible run");
    expect(run_case(&signature_b, &summary_b), "second reproducible run");
    expect(
        !signature_a.empty() && signature_a == signature_b,
        "identical synthetic workload produces an identical task trace");
    expect(summary_a == summary_b, "summary.json is reproducible");
}

void test_reports_written(const std::string& output_dir) {
    pl::PanelLuPrototype prototype(base_config(2));
    expect(prototype.configure().ok, "configure report case");
    expect(prototype.submit_command(make_request(0, 23, 23)).ok, "submit report");
    expect(prototype.submit_command(make_request(1, 40, 40)).ok, "submit report 2");
    const Outcome outcome = run(prototype);
    expect(outcome.idle, "report run idle");
    prototype.write_reports(output_dir);
    for (const std::string& file :
         {"panel_tasks.csv", "panel_commands.csv", "panel_events.csv", "panel_summary.json"}) {
        std::ifstream stream(output_dir + "/" + file);
        expect(stream.good(), "report file written: " + file);
    }
    const std::string summary = prototype.summary_json();
    expect(
        summary.find("\"architecture_prototype\": true") != std::string::npos,
        "summary marks architecture_prototype");
    expect(
        summary.find("\"numerical_validation\": \"not_performed\"") != std::string::npos,
        "summary marks numerical_validation=not_performed");
}

void test_pe1_vs_pe2_cycles() {
    auto finish = [](std::uint32_t pe_count) {
        pl::PanelLuPrototype prototype(base_config(pe_count));
        if (!prototype.configure().ok) {
            return std::uint64_t{0};
        }
        if (!prototype.submit_command(make_request(0, 40, 40)).ok) {
            return std::uint64_t{0};
        }
        const Outcome outcome = run(prototype);
        if (outcome.by_command.at(0).status != cmd::StatusCode::Ok) {
            return std::uint64_t{0};
        }
        return outcome.by_command.at(0).finish_cycle;
    };
    const std::uint64_t single = finish(1);
    const std::uint64_t dual = finish(2);
    expect(single > 0 && dual > 0, "both PE configurations complete");
    expect(
        dual < single,
        "two PEs shorten the architecture-model critical path (model cycles only)");
}


void test_slot_reuse_without_reset() {
    // One resident slot reused by a sequence of commands: task ids must stay
    // unique even though controller slots and tile slots are reused.
    pl::PanelLuConfig config = base_config(1);
    config.max_resident_commands = 1;
    pl::PanelLuPrototype prototype(config);
    expect(prototype.configure().ok, "configure slot reuse");
    std::set<std::uint64_t> task_ids;
    for (std::uint32_t command = 0; command < 4; ++command) {
        const std::uint32_t dim = (command % 2 == 0) ? 16 : 17;
        expect(
            prototype.submit_command(make_request(command, dim, dim)).ok,
            "sequential command admitted after slot release");
        const Outcome outcome = run(prototype);
        expect(
            outcome.by_command.at(command).status == cmd::StatusCode::Ok,
            "sequential command completes");
        expect(
            prototype.dispatcher().known_count() == 0 &&
                prototype.dispatcher().committed_count() == 0 &&
                prototype.dispatcher().record_count() == 0,
            "dispatcher bookkeeping recycled after command completion");
        const std::vector<pl::TileTask>* graph =
            prototype.command_task_graph(command);
        expect(graph != nullptr, "sequential command graph kept");
        if (graph != nullptr) {
            for (const pl::TileTask& task : *graph) {
                expect(
                    task_ids.insert(task.task_id).second,
                    "task ids are never reused across resident slots");
            }
        }
    }
    // Two resident slots both released and reused by a second batch.
    pl::PanelLuPrototype dual(base_config(2));
    expect(dual.configure().ok, "configure dual slot reuse");
    for (std::uint32_t batch = 0; batch < 2; ++batch) {
        expect(
            dual.submit_command(make_request(batch * 2, 32, 32, 3)).ok,
            "dual resident slot A admitted");
        expect(
            dual.submit_command(make_request(batch * 2 + 1, 32, 32, 5)).ok,
            "dual resident slot B admitted");
        const Outcome outcome = run(dual);
        expect(
            outcome.by_command.at(batch * 2).status == cmd::StatusCode::Ok &&
                outcome.by_command.at(batch * 2 + 1).status == cmd::StatusCode::Ok,
            "dual resident batch completes");
        expect(
            dual.dispatcher().known_count() == 0 &&
                dual.dispatcher().committed_count() == 0 &&
                dual.dispatcher().record_count() == 0,
            "dual dispatcher bookkeeping recycled");
    }
}

void test_fifo_holds_largest_operand_set() {
    // 2048B holds a full 16x16 GEMM A+B operand set; the accumulator is a
    // dedicated per-slot register file and must not be counted as FIFO. This
    // configuration previously deadlocked because A+B+D (3072B) was reserved.
    pl::PanelLuConfig config = base_config(1);
    config.pe_input_fifo_bytes = 2048;
    pl::PanelLuPrototype prototype(config);
    expect(prototype.configure().ok, "configure 2048B FIFO");
    expect(
        prototype.submit_command(make_request(0, 32, 32)).ok,
        "submit p=32 with 2048B FIFO");
    const Outcome outcome = run(prototype, 500000);
    expect(outcome.idle, "2048B FIFO run reaches idle (no permanent stall)");
    expect(
        outcome.by_command.at(0).status == cmd::StatusCode::Ok,
        "2048B FIFO p=32 completes");
    const pl::TaskRecord* gemm =
        find_record(prototype.task_records(), 0, pl::TileMode::GemmUpdate, 0, 1, 1);
    expect(
        gemm != nullptr && gemm->committed,
        "full-tile GEMM accepted with A+B FIFO reservation only");
}

void test_trsm_right_forward_substitution() {
    // X * U = B with upper triangular U (non-unit diagonal), forward
    // substitution over increasing j.
    {
        float u[4] = {2.0f, 1.0f, 0.0f, 4.0f};
        float b[2] = {2.0f, 5.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 2, 1, 2, u, 2);
        expect(outcome.status == cmd::StatusCode::Ok, "trsm_right 2x2 succeeds");
        expect(
            std::fabs(b[0] - 1.0f) < 1e-6f && std::fabs(b[1] - 1.0f) < 1e-6f,
            "trsm_right 2x2 gives X=[1,1] (not the U^T backward result)");
        const float r0 = b[0] * u[0] + b[1] * u[2];
        const float r1 = b[0] * u[1] + b[1] * u[3];
        expect(
            std::fabs(r0 - 2.0f) < 1e-5f && std::fabs(r1 - 5.0f) < 1e-5f,
            "trsm_right 2x2 residual X*U matches B");
    }
    {
        float u[9] = {1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 5.0f, 0.0f, 0.0f, 6.0f};
        float b[3] = {1.0f, 2.0f, 3.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 3, 1, 3, u, 3);
        expect(outcome.status == cmd::StatusCode::Ok, "trsm_right 3x3 succeeds");
        expect(
            std::fabs(b[0] - 1.0f) < 1e-6f && std::fabs(b[1]) < 1e-6f &&
                std::fabs(b[2]) < 1e-6f,
            "trsm_right 3x3 gives X=[1,0,0]");
        float residual[3] = {0.0f, 0.0f, 0.0f};
        for (std::uint32_t j = 0; j < 3; ++j) {
            for (std::uint32_t t = 0; t < 3; ++t) {
                residual[j] += b[t] * u[t * 3 + j];
            }
        }
        expect(
            std::fabs(residual[0] - 1.0f) < 1e-5f &&
                std::fabs(residual[1] - 2.0f) < 1e-5f &&
                std::fabs(residual[2] - 3.0f) < 1e-5f,
            "trsm_right 3x3 residual X*U matches B");
    }
    {
        float u[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float b[2] = {1.0f, 1.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 2, 1, 2, u, 2);
        expect(
            outcome.status == cmd::StatusCode::StaticPivotFailure &&
                outcome.failed_column == 1,
            "trsm_right zero diagonal reports the failing column");
    }
    {
        // 1e-39f is a finite FP32 subnormal; its reciprocal overflows to +Inf.
        // The failure must be reported at column 0 in the reciprocal precompute,
        // before (and not masked by) the later zero diagonal at column 1.
        float u[4] = {1.0e-39f, 0.0f, 0.0f, 0.0f};
        float b[2] = {1.0f, 1.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 2, 1, 2, u, 2);
        expect(
            outcome.status == cmd::StatusCode::NonFiniteValue &&
                outcome.failed_column == 0 && outcome.diagonal_failure,
            "reciprocal overflow of a subnormal diagonal fails at its own column");
    }
    {
        // Overflow at column 1 when column 0 is healthy.
        float u[4] = {2.0f, 0.0f, 0.0f, 1.0e-39f};
        float b[2] = {1.0f, 1.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 2, 1, 2, u, 2);
        expect(
            outcome.status == cmd::StatusCode::NonFiniteValue &&
                outcome.failed_column == 1 && outcome.diagonal_failure,
            "reciprocal overflow reports the first overflowing column");
    }
    {
        // A wide but representable dynamic range stays finite and succeeds.
        float u[4] = {1.0e-30f, 0.0f, 0.0f, 1.0e+30f};
        float b[2] = {1.0f, 1.0f};
        const auto outcome = pl::kernels::run_trsm_right(b, 2, 1, 2, u, 2);
        expect(outcome.status == cmd::StatusCode::Ok, "extreme finite range succeeds");
        expect(
            std::isfinite(b[0]) && std::isfinite(b[1]) &&
                std::fabs(b[0] - 1.0e30f) < 1.0e23f &&
                std::fabs(b[1] - 1.0e-30f) < 1.0e-37f,
            "finite reciprocal range solves without spurious failure");
    }
}


void test_trsm_right_division_sensitivity() {
    // Changing the divide/reciprocal latency must move TRSM_RIGHT (non-unit
    // diagonal, k reciprocals) but not TRSM_LEFT (unit diagonal, no divide).
    pl::PanelLuConfig config = base_config(1);
    config.timing.div_latency_cycles = 1000;
    pl::PanelLuPrototype prototype(config);
    expect(prototype.configure().ok, "configure slow divider");
    expect(prototype.submit_command(make_request(0, 17, 17)).ok, "submit p=17");
    const Outcome outcome = run(prototype, 500000);
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "p=17 OK");

    const pl::TaskRecord* right =
        find_record(prototype.task_records(), 0, pl::TileMode::TrsmRight, 0, 1, 0);
    expect(right != nullptr, "TRSM_RIGHT record");
    if (right != nullptr) {
        // fill(2) + recip(1000 + 15) + k(16) + m(1) = 1034.
        const std::uint64_t span = right->exec_done_cycle - right->exec_start_cycle;
        expect(
            span == 2 + (1000 + 15) + 16 + 1,
            "TRSM_RIGHT exec includes the reciprocal precompute latency");
        expect(
            span >= config.timing.div_latency_cycles,
            "TRSM_RIGHT exec is at least one divide latency");
    }
    const pl::TaskRecord* left =
        find_record(prototype.task_records(), 0, pl::TileMode::TrsmLeft, 0, 0, 1);
    expect(left != nullptr, "TRSM_LEFT record");
    if (left != nullptr) {
        expect(
            left->exec_done_cycle - left->exec_start_cycle == 2 + 16 + 1,
            "TRSM_LEFT remains divide-free (unit diagonal)");
    }
}

void test_task_ids_beyond_uint32() {
    // Task ids are uint64_t end-to-end. Start a command at 2^32 so any 32-bit
    // truncation in the dependency chain would break the TRSM deps.
    pl::PanelLuPrototype prototype(base_config(1));
    expect(prototype.configure().ok, "configure uint32 boundary");
    const std::uint64_t base = (std::uint64_t{1} << 32) + 1;  // 4294967297
    expect(
        prototype.submit_command(make_request(0, 17, 17), base).ok,
        "submit at task-id base 2^32+1");
    const Outcome outcome = run(prototype, 500000);
    expect(outcome.idle && !outcome.control_fault, "uint32 boundary run completes");
    expect(outcome.by_command.at(0).status == cmd::StatusCode::Ok, "boundary run OK");
    const std::vector<pl::TileTask>* graph = prototype.command_task_graph(0);
    expect(graph != nullptr, "boundary graph present");
    if (graph != nullptr) {
        const pl::TileTask* diag = find_graph_task(*graph, pl::TileMode::LuDiag, 0, 0, 0);
        const pl::TileTask* left =
            find_graph_task(*graph, pl::TileMode::TrsmLeft, 0, 0, 1);
        expect(diag != nullptr && diag->task_id == base, "diag id preserved as uint64");
        expect(
            left != nullptr && left->deps.size() == 1 &&
                left->deps[0] == base &&
                left->deps[0] > 0xFFFFFFFFull,
            "TRSM dependency is not truncated to 32 bits");
    }
    // Auto allocation remains disjoint after an override.
    expect(prototype.submit_command(make_request(1, 16, 16)).ok, "auto base after override");
    const Outcome second = run(prototype);
    expect(second.by_command.at(1).status == cmd::StatusCode::Ok, "auto-base command OK");
}

void test_resident_capacity() {
    pl::PanelLuConfig config = base_config(1);
    config.max_resident_commands = 1;
    pl::PanelLuPrototype prototype(config);
    expect(prototype.configure().ok, "configure single residency");
    expect(prototype.submit_command(make_request(0, 32, 32)).ok, "first command admitted");
    const pl::ConfigCheck second = prototype.submit_command(make_request(1, 32, 32));
    expect(
        !second.ok && second.status == cmd::StatusCode::BufferFull,
        "second resident command rejected with BUFFER_FULL");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output_dir = argc > 1 ? argv[1] : "t06_output";

    test_config_validation();
    test_front_capability_checks();
    test_task_graph_full_tiles();
    test_tail_tiles();
    test_pe_exclusion_and_cross_mode();
    test_same_target_ordering();
    test_dependency_not_ready();
    test_queue_full_backpressure();
    test_port_and_writeback_backpressure();
    test_latency_handcalc();
    test_scan_dependency_and_finite_check();
    test_data_dependent_failure();
    test_directed_abort();
    test_reset_invalidates_old_work();
    test_dual_command_fairness();
    test_reproducible_trace();
    test_reports_written(output_dir);
    test_pe1_vs_pe2_cycles();
    test_resident_capacity();
    test_slot_reuse_without_reset();
    test_fifo_holds_largest_operand_set();
    test_trsm_right_forward_substitution();
    test_trsm_right_division_sensitivity();
    test_task_ids_beyond_uint32();

    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
