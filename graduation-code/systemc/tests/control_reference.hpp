#pragma once

// Independent control-flow reference interpreter for the T05 command stream.
// It deliberately does NOT reuse the CommandDevice/GCU state machine: expected
// command statuses and token states are computed here from decoded records and
// an explicit injection script, then compared against the device run.

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "command_codec.hpp"

namespace hw::testing {

struct RefFailure {
    hw::command::StatusCode status{hw::command::StatusCode::StaticPivotFailure};
    std::uint32_t pivot_count{0};
};

struct RefResult {
    std::map<std::uint32_t, hw::command::StatusCode> status;
    std::map<std::uint32_t, std::string> reason_class;
    std::map<std::uint32_t, std::uint32_t> pivot_count;
    std::map<std::uint32_t, bool> token_ready;  // token -> READY (else FAILED)
};

// Rules (must match the documented device semantics):
//   1. commands of an aborted node -> ABORTED (their tokens fail);
//   2. any FAILED dependency -> DEPENDENCY_FAILED;
//   3. injected failure -> that status (with the injected failure column);
//   4. PANEL_LU with DYNAMIC_PANEL -> POLICY_NOT_SUPPORTED;
//   5. otherwise OK.
// Completion order is not modelled: independent commands may finish in any
// order, but each command's terminal status is a pure function of the stream.
inline RefResult interpret(
    const std::vector<hw::command::Command>& commands,
    const std::vector<hw::command::Descriptor>& descriptors,
    const std::vector<std::uint8_t>& image,
    std::uint32_t token_count,
    const std::map<std::uint32_t, RefFailure>& failures = {},
    const std::set<std::uint32_t>& aborted_nodes = {}) {
    namespace cmd = hw::command;
    RefResult result;
    std::map<std::uint32_t, bool> token_failed;  // token -> failed
    for (std::uint32_t token = 0; token < token_count; ++token) {
        token_failed[token] = false;
    }

    std::map<std::uint32_t, const cmd::Command*> ordered;
    for (const auto& command : commands) {
        ordered[command.command_id] = &command;
    }

    for (const auto& entry : ordered) {
        const cmd::Command& command = *entry.second;
        const bool node_aborted =
            aborted_nodes.count(command.node_id) != 0 &&
            command.opcode != cmd::Opcode::AbortNode;

        bool dep_failed = false;
        if (command.wait_list_id != cmd::NONE) {
            const auto tokens = cmd::DescriptorCodec::dependency_tokens(
                descriptors.at(command.wait_list_id), image, token_count);
            for (const auto token : tokens) {
                if (token_failed.at(token)) {
                    dep_failed = true;
                    break;
                }
            }
        }

        cmd::StatusCode status = cmd::StatusCode::Ok;
        std::uint32_t pivot_count = 0;
        if (node_aborted) {
            status = cmd::StatusCode::Aborted;
        } else if (dep_failed) {
            status = cmd::StatusCode::DependencyFailed;
        } else if (failures.count(command.command_id) != 0) {
            const auto& failure = failures.at(command.command_id);
            status = failure.status;
            pivot_count = failure.pivot_count;
        } else if (command.opcode == cmd::Opcode::PanelLu) {
            const auto& front = descriptors.at(command.descriptor_id);
            const auto policy = static_cast<cmd::PivotPolicy>(front.body_words[8]);
            if (policy == cmd::PivotPolicy::DynamicPanel) {
                status = cmd::StatusCode::PolicyNotSupported;
            } else {
                // A completed STATIC panel reports its full pivot count.
                pivot_count = front.body_words[2];
            }
        }

        result.status[command.command_id] = status;
        result.pivot_count[command.command_id] = pivot_count;
        if (command.signal_token != cmd::NONE) {
            token_failed[command.signal_token] = status != cmd::StatusCode::Ok;
            result.token_ready[command.signal_token] = status == cmd::StatusCode::Ok;
        }
    }
    return result;
}

}  // namespace hw::testing
