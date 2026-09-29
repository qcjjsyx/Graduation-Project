#include "controlplane/token_scoreboard.hpp"

namespace hw::controlplane {

void TokenScoreboard::init(std::uint32_t token_count) {
    token_count_ = token_count;
    states_.clear();
    producers_.clear();
    for (std::uint32_t token = 0; token < token_count; ++token) {
        states_[token] = State::Pending;
    }
}

void TokenScoreboard::register_producer(std::uint32_t token, std::uint32_t command_id) {
    if (token >= token_count_ || token == hw::command::NONE) {
        return;
    }
    producers_[token] = command_id;
}

bool TokenScoreboard::signal(
    std::uint32_t token,
    State terminal,
    std::uint32_t command_id) {
    if (token == hw::command::NONE || token >= token_count_) {
        return false;
    }
    if (terminal != State::Ready && terminal != State::Failed) {
        return false;
    }
    const auto producer = producers_.find(token);
    if (producer == producers_.end() || producer->second != command_id) {
        return false;
    }
    auto& state = states_[token];
    if (state != State::Pending) {
        return false;  // single terminal transition only
    }
    state = terminal;
    return true;
}

TokenScoreboard::State TokenScoreboard::state(std::uint32_t token) const {
    if (token == hw::command::NONE || token >= token_count_) {
        return State::None;
    }
    const auto found = states_.find(token);
    return found == states_.end() ? State::None : found->second;
}

std::uint32_t TokenScoreboard::producer(std::uint32_t token) const {
    const auto found = producers_.find(token);
    return found == producers_.end() ? hw::command::NONE : found->second;
}

TokenScoreboard::Deps TokenScoreboard::check(const std::vector<std::uint32_t>& waits) const {
    bool pending = false;
    for (const auto token : waits) {
        if (token == hw::command::NONE) {
            continue;  // NONE never enters the scoreboard
        }
        switch (state(token)) {
        case State::Failed:
            return Deps::Failed;
        case State::Pending:
        case State::None:
            pending = true;
            break;
        case State::Ready:
            break;
        }
    }
    return pending ? Deps::Pending : Deps::Ready;
}

std::uint32_t TokenScoreboard::ready_count() const {
    std::uint32_t count = 0;
    for (const auto& entry : states_) {
        count += entry.second == State::Ready ? 1 : 0;
    }
    return count;
}

std::uint32_t TokenScoreboard::failed_count() const {
    std::uint32_t count = 0;
    for (const auto& entry : states_) {
        count += entry.second == State::Failed ? 1 : 0;
    }
    return count;
}

std::uint32_t TokenScoreboard::pending_count() const {
    std::uint32_t count = 0;
    for (const auto& entry : states_) {
        count += entry.second == State::Pending ? 1 : 0;
    }
    return count;
}

void TokenScoreboard::reset() {
    states_.clear();
    producers_.clear();
    token_count_ = 0;
}

}  // namespace hw::controlplane
