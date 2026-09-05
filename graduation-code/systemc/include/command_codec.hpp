#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace hw::command_v1 {

inline constexpr std::uint32_t NONE = 0xFFFFFFFFu;
inline constexpr std::size_t COMMAND_RECORD_BYTES = 32;
inline constexpr std::size_t DESCRIPTOR_RECORD_BYTES = 64;
inline constexpr std::size_t COMPLETION_RECORD_BYTES = 64;

enum class Opcode : std::uint32_t {
    NodeBegin = 0x01,
    LoadFront = 0x02,
    AssembleExtendAdd = 0x03,
    PanelLu = 0x04,
    TrsmLeft = 0x05,
    TrsmRight = 0x06,
    GemmSchur = 0x07,
    StoreFactor = 0x08,
    StoreUpdate = 0x09,
    SolveForward = 0x0A,
    SolveBackward = 0x0B,
    NodeCommit = 0x0C,
    AbortNode = 0x0D,
};

inline constexpr std::uint32_t TRACE_ENABLE = 1u << 0;
inline constexpr std::uint32_t ALLOW_RETRY = 1u << 1;
inline constexpr std::uint32_t COMMAND_ALLOWED_FLAGS = TRACE_ENABLE | ALLOW_RETRY;
inline constexpr std::uint16_t DESCRIPTOR_ALLOWED_FLAGS = 0;

enum class DescriptorType : std::uint16_t {
    Region = 0x01,
    Front = 0x02,
    Contribution = 0x03,
    Factor = 0x04,
    Kernel = 0x05,
    Solve = 0x06,
    Scale = 0x07,
    Dependency = 0x08,
};

enum class DataFormat : std::uint32_t {
    Fp64 = 0x01,
    Fp32 = 0x02,
    Int32 = 0x03,
    LegacyInt32Exp = 0x04,
};

enum class DataLayout : std::uint32_t {
    RowMajor = 0x01,
};

enum class KernelBackend : std::uint32_t {
    SystemCFp32DeviceModel = 0x01,
    SystemCInt32GemmModel = 0x02,
};

enum class SolveDirection : std::uint32_t {
    Forward = 0x01,
    Backward = 0x02,
};

enum class StatusCode : std::uint16_t {
    Ok = 0x0000,
    BadCommand = 0x0001,
    BadDescriptor = 0x0002,
    AddressFault = 0x0003,
    BufferFull = 0x0004,
    DependencyFailed = 0x0005,
    PivotNotFound = 0x0100,
    PivotUnstable = 0x0101,
    NumericOverflow = 0x0102,
    QuantizationSaturation = 0x0103,
    PrecisionRetry = 0x0104,
    Timeout = 0x0200,
    Aborted = 0x0201,
};

// These numeric values are for cross-language fixtures only. Schema v1 does
// not define the executor's physical scoreboard memory layout.
enum class TokenState : std::uint32_t {
    Unsignaled = 0,
    Ready = 1,
    Failed = 2,
};

struct Command {
    Opcode opcode{Opcode::NodeBegin};
    std::uint32_t flags{0};
    std::uint32_t command_id{0};
    std::uint32_t node_id{NONE};
    std::uint32_t descriptor_id{NONE};
    std::uint32_t wait_list_id{NONE};
    std::uint32_t signal_token{NONE};
    std::uint32_t arg0{0};
};

struct Descriptor {
    DescriptorType descriptor_type{DescriptorType::Region};
    std::uint16_t flags{0};
    std::uint32_t reserved{0};
    std::uint64_t payload_offset{0};
    std::uint64_t payload_bytes{0};
    std::array<std::uint32_t, 10> body_words{};
};

struct Completion {
    std::uint32_t command_id{0};
    std::uint32_t node_id{NONE};
    StatusCode status_code{StatusCode::Ok};
    std::uint32_t pivot_count{0};
    std::uint64_t start_cycle{0};
    std::uint64_t finish_cycle{0};
    std::uint64_t read_bytes{0};
    std::uint64_t write_bytes{0};
    std::uint64_t stall_cycles{0};
    std::uint32_t overflow_count{0};
    std::uint32_t retry_count{0};
};

namespace detail {

inline std::uint16_t read_u16(const std::uint8_t* data, std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset]) |
           (static_cast<std::uint16_t>(data[offset + 1]) << 8);
}

inline std::uint32_t read_u32(const std::uint8_t* data, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(data[offset + index]) << (8 * index);
    }
    return value;
}

inline std::uint64_t read_u64(const std::uint8_t* data, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(data[offset + index]) << (8 * index);
    }
    return value;
}

inline void write_u16(std::uint8_t* data, std::size_t offset, std::uint16_t value) {
    for (unsigned index = 0; index < 2; ++index) {
        data[offset + index] =
            static_cast<std::uint8_t>((value >> (8 * index)) & 0xFFu);
    }
}

inline void write_u32(std::uint8_t* data, std::size_t offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        data[offset + index] =
            static_cast<std::uint8_t>((value >> (8 * index)) & 0xFFu);
    }
}

inline void write_u64(std::uint8_t* data, std::size_t offset, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        data[offset + index] =
            static_cast<std::uint8_t>((value >> (8 * index)) & 0xFFu);
    }
}

inline bool known_opcode(Opcode opcode) {
    const auto value = static_cast<std::uint32_t>(opcode);
    return value >= static_cast<std::uint32_t>(Opcode::NodeBegin) &&
           value <= static_cast<std::uint32_t>(Opcode::AbortNode);
}

inline bool known_descriptor_type(DescriptorType type) {
    const auto value = static_cast<std::uint16_t>(type);
    return value >= static_cast<std::uint16_t>(DescriptorType::Region) &&
           value <= static_cast<std::uint16_t>(DescriptorType::Dependency);
}

inline bool known_status(StatusCode status) {
    switch (status) {
    case StatusCode::Ok:
    case StatusCode::BadCommand:
    case StatusCode::BadDescriptor:
    case StatusCode::AddressFault:
    case StatusCode::BufferFull:
    case StatusCode::DependencyFailed:
    case StatusCode::PivotNotFound:
    case StatusCode::PivotUnstable:
    case StatusCode::NumericOverflow:
    case StatusCode::QuantizationSaturation:
    case StatusCode::PrecisionRetry:
    case StatusCode::Timeout:
    case StatusCode::Aborted:
        return true;
    }
    return false;
}

inline bool known_token_state(TokenState state) {
    switch (state) {
    case TokenState::Unsignaled:
    case TokenState::Ready:
    case TokenState::Failed:
        return true;
    }
    return false;
}

inline void expect_size(const char* name, std::size_t actual, std::size_t expected) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string(name) + " must be exactly " + std::to_string(expected) +
            " bytes, got " + std::to_string(actual));
    }
}

}  // namespace detail

class CommandCodec {
public:
    using Record = std::array<std::uint8_t, COMMAND_RECORD_BYTES>;

    static void validate(const Command& command) {
        if (!detail::known_opcode(command.opcode)) {
            throw std::runtime_error("unknown command opcode");
        }
        if ((command.flags & ~COMMAND_ALLOWED_FLAGS) != 0) {
            throw std::runtime_error("command flags contain reserved bits");
        }
        if (command.arg0 != 0) {
            throw std::runtime_error("command arg0 is reserved and must be zero");
        }
    }

    static Record encode(const Command& command) {
        validate(command);
        Record record{};
        detail::write_u32(record.data(), 0, static_cast<std::uint32_t>(command.opcode));
        detail::write_u32(record.data(), 4, command.flags);
        detail::write_u32(record.data(), 8, command.command_id);
        detail::write_u32(record.data(), 12, command.node_id);
        detail::write_u32(record.data(), 16, command.descriptor_id);
        detail::write_u32(record.data(), 20, command.wait_list_id);
        detail::write_u32(record.data(), 24, command.signal_token);
        detail::write_u32(record.data(), 28, command.arg0);
        return record;
    }

    static Command decode(const std::uint8_t* data, std::size_t size) {
        detail::expect_size("command record", size, COMMAND_RECORD_BYTES);
        Command command{};
        command.opcode = static_cast<Opcode>(detail::read_u32(data, 0));
        command.flags = detail::read_u32(data, 4);
        command.command_id = detail::read_u32(data, 8);
        command.node_id = detail::read_u32(data, 12);
        command.descriptor_id = detail::read_u32(data, 16);
        command.wait_list_id = detail::read_u32(data, 20);
        command.signal_token = detail::read_u32(data, 24);
        command.arg0 = detail::read_u32(data, 28);
        validate(command);
        return command;
    }

    static Command decode(const Record& record) {
        return decode(record.data(), record.size());
    }
};

class DescriptorCodec {
public:
    using Record = std::array<std::uint8_t, DESCRIPTOR_RECORD_BYTES>;

    static void validate(
        const Descriptor& descriptor,
        std::optional<std::uint64_t> memory_image_bytes = std::nullopt) {
        if (!detail::known_descriptor_type(descriptor.descriptor_type)) {
            throw std::runtime_error("unknown descriptor type");
        }
        if ((descriptor.flags & ~DESCRIPTOR_ALLOWED_FLAGS) != 0) {
            throw std::runtime_error("descriptor flags contain reserved bits");
        }
        if (descriptor.reserved != 0) {
            throw std::runtime_error("descriptor reserved field must be zero");
        }
        if (descriptor.payload_bytes >
            std::numeric_limits<std::uint64_t>::max() - descriptor.payload_offset) {
            throw std::runtime_error("descriptor payload range overflows u64");
        }
        const auto payload_end = descriptor.payload_offset + descriptor.payload_bytes;
        if (memory_image_bytes && payload_end > *memory_image_bytes) {
            throw std::runtime_error("descriptor payload range exceeds memory image");
        }

        std::size_t first_reserved_word = 10;
        switch (descriptor.descriptor_type) {
        case DescriptorType::Region: first_reserved_word = 9; break;
        case DescriptorType::Front: first_reserved_word = 8; break;
        case DescriptorType::Contribution: first_reserved_word = 9; break;
        case DescriptorType::Factor: first_reserved_word = 7; break;
        case DescriptorType::Kernel: first_reserved_word = 10; break;
        case DescriptorType::Solve: first_reserved_word = 8; break;
        case DescriptorType::Scale: first_reserved_word = 6; break;
        case DescriptorType::Dependency: first_reserved_word = 1; break;
        }
        for (std::size_t index = first_reserved_word; index < 10; ++index) {
            if (descriptor.body_words[index] != 0) {
                throw std::runtime_error("descriptor body reserved word must be zero");
            }
        }

        if (descriptor.descriptor_type == DescriptorType::Region) {
            const auto base_addr =
                static_cast<std::uint64_t>(descriptor.body_words[0]) |
                (static_cast<std::uint64_t>(descriptor.body_words[1]) << 32);
            const auto byte_size =
                static_cast<std::uint64_t>(descriptor.body_words[2]) |
                (static_cast<std::uint64_t>(descriptor.body_words[3]) << 32);
            if (base_addr % 64 != 0) {
                throw std::runtime_error("REGION_DESC base_addr must be 64-byte aligned");
            }
            if (byte_size > std::numeric_limits<std::uint64_t>::max() - base_addr) {
                throw std::runtime_error("REGION_DESC address range overflows u64");
            }
            if (memory_image_bytes && base_addr + byte_size > *memory_image_bytes) {
                throw std::runtime_error("REGION_DESC address range exceeds memory image");
            }
            const auto format = static_cast<DataFormat>(descriptor.body_words[7]);
            std::uint64_t element_size = 0;
            switch (format) {
            case DataFormat::Fp64: element_size = 8; break;
            case DataFormat::Fp32:
            case DataFormat::Int32:
            case DataFormat::LegacyInt32Exp: element_size = 4; break;
            default: throw std::runtime_error("unknown REGION_DESC data format");
            }
            if (static_cast<DataLayout>(descriptor.body_words[8]) !=
                DataLayout::RowMajor) {
                throw std::runtime_error("unknown REGION_DESC data layout");
            }
            const auto row_stride = static_cast<std::uint64_t>(descriptor.body_words[4]);
            const auto rows = static_cast<std::uint64_t>(descriptor.body_words[5]);
            const auto cols = static_cast<std::uint64_t>(descriptor.body_words[6]);
            if (rows == 0 || cols == 0) {
                if (byte_size != 0 || row_stride != 0) {
                    throw std::runtime_error(
                        "empty REGION_DESC must have zero size and stride");
                }
            } else {
                const auto one_row = cols * element_size;
                if (row_stride < one_row) {
                    throw std::runtime_error(
                        "REGION_DESC row_stride is smaller than one row");
                }
                const auto prior_rows = rows - 1;
                if (prior_rows != 0 &&
                    row_stride >
                        (std::numeric_limits<std::uint64_t>::max() - one_row) /
                            prior_rows) {
                    throw std::runtime_error("REGION_DESC dimensions overflow u64");
                }
                if (prior_rows * row_stride + one_row > byte_size) {
                    throw std::runtime_error(
                        "REGION_DESC dimensions exceed byte_size");
                }
            }
        } else if (descriptor.descriptor_type == DescriptorType::Contribution) {
            const auto expected = std::uint64_t{4} *
                (static_cast<std::uint64_t>(descriptor.body_words[4]) +
                 descriptor.body_words[5]);
            if (descriptor.payload_offset % 4 != 0) {
                throw std::runtime_error(
                    "CONTRIBUTION_DESC payload_offset must be u32 aligned");
            }
            if (descriptor.payload_bytes != expected) {
                throw std::runtime_error(
                    "CONTRIBUTION_DESC payload_bytes must hold row and column maps");
            }
        } else if (descriptor.descriptor_type == DescriptorType::Dependency) {
            const auto expected = std::uint64_t{4} * descriptor.body_words[0];
            if (descriptor.payload_offset % 4 != 0) {
                throw std::runtime_error(
                    "DEPENDENCY_DESC payload_offset must be u32 aligned");
            }
            if (descriptor.payload_bytes != expected) {
                throw std::runtime_error(
                    "DEPENDENCY_DESC payload_bytes must equal token_count * 4");
            }
        } else if (descriptor.payload_bytes != 0) {
            throw std::runtime_error("descriptor type does not define a v1 variable payload");
        }
    }

    static Record encode(const Descriptor& descriptor) {
        validate(descriptor);
        Record record{};
        detail::write_u16(
            record.data(), 0, static_cast<std::uint16_t>(descriptor.descriptor_type));
        detail::write_u16(record.data(), 2, descriptor.flags);
        detail::write_u32(record.data(), 4, descriptor.reserved);
        detail::write_u64(record.data(), 8, descriptor.payload_offset);
        detail::write_u64(record.data(), 16, descriptor.payload_bytes);
        for (std::size_t index = 0; index < descriptor.body_words.size(); ++index) {
            detail::write_u32(record.data(), 24 + index * 4, descriptor.body_words[index]);
        }
        return record;
    }

    static Descriptor decode(
        const std::uint8_t* data,
        std::size_t size,
        std::optional<std::uint64_t> memory_image_bytes = std::nullopt) {
        detail::expect_size("descriptor record", size, DESCRIPTOR_RECORD_BYTES);
        Descriptor descriptor{};
        descriptor.descriptor_type =
            static_cast<DescriptorType>(detail::read_u16(data, 0));
        descriptor.flags = detail::read_u16(data, 2);
        descriptor.reserved = detail::read_u32(data, 4);
        descriptor.payload_offset = detail::read_u64(data, 8);
        descriptor.payload_bytes = detail::read_u64(data, 16);
        for (std::size_t index = 0; index < descriptor.body_words.size(); ++index) {
            descriptor.body_words[index] = detail::read_u32(data, 24 + index * 4);
        }
        validate(descriptor, memory_image_bytes);
        return descriptor;
    }

    static Descriptor decode(
        const Record& record,
        std::optional<std::uint64_t> memory_image_bytes = std::nullopt) {
        return decode(record.data(), record.size(), memory_image_bytes);
    }

    static std::vector<std::uint32_t> dependency_tokens(
        const Descriptor& descriptor,
        const std::vector<std::uint8_t>& memory_image,
        std::optional<std::uint32_t> token_count = std::nullopt,
        std::optional<std::uint32_t> max_wait_tokens = std::nullopt) {
        validate(descriptor, static_cast<std::uint64_t>(memory_image.size()));
        if (descriptor.descriptor_type != DescriptorType::Dependency) {
            throw std::runtime_error("descriptor is not a DEPENDENCY_DESC");
        }
        const auto count = descriptor.body_words[0];
        if (max_wait_tokens && count > *max_wait_tokens) {
            throw std::runtime_error("dependency token count exceeds max_wait_tokens");
        }
        std::vector<std::uint32_t> tokens;
        tokens.reserve(count);
        std::set<std::uint32_t> seen;
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto offset = static_cast<std::size_t>(descriptor.payload_offset) + index * 4;
            const auto token = detail::read_u32(memory_image.data(), offset);
            if (!seen.insert(token).second) {
                throw std::runtime_error("dependency token list contains duplicates");
            }
            if (token_count && token >= *token_count) {
                throw std::runtime_error("dependency token list contains an unknown token");
            }
            tokens.push_back(token);
        }
        return tokens;
    }
};

class CompletionCodec {
public:
    using Record = std::array<std::uint8_t, COMPLETION_RECORD_BYTES>;

    static void validate(const Completion& completion) {
        if (!detail::known_status(completion.status_code)) {
            throw std::runtime_error("unknown completion status code");
        }
    }

    static Record encode(const Completion& completion) {
        validate(completion);
        Record record{};
        detail::write_u32(record.data(), 0, completion.command_id);
        detail::write_u32(record.data(), 4, completion.node_id);
        detail::write_u16(
            record.data(), 8, static_cast<std::uint16_t>(completion.status_code));
        detail::write_u16(record.data(), 10, 0);
        detail::write_u32(record.data(), 12, completion.pivot_count);
        detail::write_u64(record.data(), 16, completion.start_cycle);
        detail::write_u64(record.data(), 24, completion.finish_cycle);
        detail::write_u64(record.data(), 32, completion.read_bytes);
        detail::write_u64(record.data(), 40, completion.write_bytes);
        detail::write_u64(record.data(), 48, completion.stall_cycles);
        detail::write_u32(record.data(), 56, completion.overflow_count);
        detail::write_u32(record.data(), 60, completion.retry_count);
        return record;
    }

    static Completion decode(const std::uint8_t* data, std::size_t size) {
        detail::expect_size("completion record", size, COMPLETION_RECORD_BYTES);
        if (detail::read_u16(data, 10) != 0) {
            throw std::runtime_error("completion reserved field must be zero");
        }
        Completion completion{};
        completion.command_id = detail::read_u32(data, 0);
        completion.node_id = detail::read_u32(data, 4);
        completion.status_code = static_cast<StatusCode>(detail::read_u16(data, 8));
        completion.pivot_count = detail::read_u32(data, 12);
        completion.start_cycle = detail::read_u64(data, 16);
        completion.finish_cycle = detail::read_u64(data, 24);
        completion.read_bytes = detail::read_u64(data, 32);
        completion.write_bytes = detail::read_u64(data, 40);
        completion.stall_cycles = detail::read_u64(data, 48);
        completion.overflow_count = detail::read_u32(data, 56);
        completion.retry_count = detail::read_u32(data, 60);
        validate(completion);
        return completion;
    }

    static Completion decode(const Record& record) {
        return decode(record.data(), record.size());
    }
};

inline std::vector<std::uint8_t> encode_token_states(
    const std::vector<TokenState>& states) {
    std::vector<std::uint8_t> encoded(states.size() * 4, 0);
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (!detail::known_token_state(states[index])) {
            throw std::runtime_error("unknown token state");
        }
        detail::write_u32(
            encoded.data(), index * 4, static_cast<std::uint32_t>(states[index]));
    }
    return encoded;
}

inline std::vector<TokenState> decode_token_states(
    const std::uint8_t* data, std::size_t size) {
    if (size % 4 != 0) {
        throw std::runtime_error("token-state fixture size must be a multiple of 4 bytes");
    }
    std::vector<TokenState> states;
    states.reserve(size / 4);
    for (std::size_t offset = 0; offset < size; offset += 4) {
        const auto state = static_cast<TokenState>(detail::read_u32(data, offset));
        if (!detail::known_token_state(state)) {
            throw std::runtime_error("unknown token state");
        }
        states.push_back(state);
    }
    return states;
}

inline std::vector<TokenState> decode_token_states(
    const std::vector<std::uint8_t>& data) {
    return decode_token_states(data.data(), data.size());
}

inline void validate_command_batch(
    const std::vector<Command>& commands,
    const std::vector<Descriptor>& descriptors,
    const std::vector<std::uint8_t>& memory_image,
    std::uint32_t token_count,
    std::optional<std::uint32_t> max_wait_tokens = std::nullopt) {
    for (const auto& descriptor : descriptors) {
        DescriptorCodec::validate(
            descriptor, static_cast<std::uint64_t>(memory_image.size()));
    }

    std::set<std::uint32_t> command_ids;
    std::set<std::uint32_t> signal_producers;
    for (const auto& command : commands) {
        CommandCodec::validate(command);
        if (command.command_id == NONE) {
            throw std::runtime_error("command_id must not be NONE");
        }
        if (!command_ids.insert(command.command_id).second) {
            throw std::runtime_error("duplicate command_id");
        }
        if (command.descriptor_id != NONE && command.descriptor_id >= descriptors.size()) {
            throw std::runtime_error("unknown descriptor_id");
        }
        if (command.wait_list_id != NONE) {
            if (command.wait_list_id >= descriptors.size()) {
                throw std::runtime_error("unknown wait_list_id");
            }
            const auto& wait_descriptor = descriptors[command.wait_list_id];
            if (wait_descriptor.descriptor_type != DescriptorType::Dependency) {
                throw std::runtime_error("wait_list_id is not a DEPENDENCY_DESC");
            }
            (void)DescriptorCodec::dependency_tokens(
                wait_descriptor, memory_image, token_count, max_wait_tokens);
        }
        if (command.signal_token != NONE) {
            if (command.signal_token >= token_count) {
                throw std::runtime_error("unknown signal_token");
            }
            if (!signal_producers.insert(command.signal_token).second) {
                throw std::runtime_error("duplicate signal_token producer");
            }
        }
    }
}

}  // namespace hw::command_v1
