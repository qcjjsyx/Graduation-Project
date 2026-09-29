#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "command_codec.hpp"

namespace hw::controlplane {

// Dependency tokens: PENDING -> READY | FAILED exactly once. NONE never enters
// the scoreboard. Every token has exactly one producer command.
class TokenScoreboard {
public:
    enum class State { None, Pending, Ready, Failed };
    enum class Deps { Ready, Pending, Failed };

    void init(std::uint32_t token_count);
    void register_producer(std::uint32_t token, std::uint32_t command_id);

    // One terminal transition per token; returns false on unknown token,
    // duplicate terminal transition or a producer mismatch.
    bool signal(std::uint32_t token, State terminal, std::uint32_t command_id);

    State state(std::uint32_t token) const;
    std::uint32_t producer(std::uint32_t token) const;

    // NONE entries in the wait list are ignored (they do not enter the
    // scoreboard). Any FAILED dependency => Deps::Failed.
    Deps check(const std::vector<std::uint32_t>& waits) const;

    std::uint32_t ready_count() const;
    std::uint32_t failed_count() const;
    std::uint32_t pending_count() const;
    std::uint32_t token_count() const { return token_count_; }
    void reset();

private:
    std::uint32_t token_count_{0};
    std::map<std::uint32_t, State> states_;
    std::map<std::uint32_t, std::uint32_t> producers_;
};

}  // namespace hw::controlplane
