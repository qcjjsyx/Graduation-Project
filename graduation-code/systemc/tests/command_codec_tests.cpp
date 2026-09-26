#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "command_codec.hpp"

namespace {

namespace cmd = hw::command;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_throws(const std::function<void()>& action, const std::string& message) {
    try {
        action();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open fixture: " + path);
    }
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_u16(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value) {
    data[offset] = static_cast<std::uint8_t>(value & 0xFFu);
    data[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFFu);
}

void write_u32(std::vector<std::uint8_t>& data, std::size_t offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        data[offset + index] =
            static_cast<std::uint8_t>((value >> (8 * index)) & 0xFFu);
    }
}

template <std::size_t Size>
void expect_equal(
    const std::array<std::uint8_t, Size>& actual,
    const std::vector<std::uint8_t>& expected,
    const std::string& message) {
    expect(expected.size() == Size, message + " fixture size mismatch");
    expect(std::equal(actual.begin(), actual.end(), expected.begin()), message);
}

std::uint32_t threshold_bits(float ratio) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &ratio, sizeof(bits));
    return bits;
}

cmd::Descriptor region_descriptor(std::uint64_t base_addr = 0, std::uint64_t byte_size = 0) {
    cmd::Descriptor descriptor{};
    descriptor.descriptor_type = cmd::DescriptorType::Region;
    descriptor.body_words[0] = static_cast<std::uint32_t>(base_addr);
    descriptor.body_words[1] = static_cast<std::uint32_t>(base_addr >> 32);
    descriptor.body_words[2] = static_cast<std::uint32_t>(byte_size);
    descriptor.body_words[3] = static_cast<std::uint32_t>(byte_size >> 32);
    descriptor.body_words[7] = static_cast<std::uint32_t>(cmd::DataFormat::Fp32);
    descriptor.body_words[8] = static_cast<std::uint32_t>(cmd::DataLayout::RowMajor);
    return descriptor;
}

cmd::Descriptor dependency_descriptor(std::uint64_t offset, std::uint32_t count) {
    cmd::Descriptor descriptor{};
    descriptor.descriptor_type = cmd::DescriptorType::Dependency;
    descriptor.payload_offset = offset;
    descriptor.payload_bytes = static_cast<std::uint64_t>(count) * 4;
    descriptor.body_words[0] = count;
    return descriptor;
}

cmd::Descriptor front_descriptor(cmd::PivotPolicy policy, float threshold = 0.0f) {
    cmd::Descriptor descriptor{};
    descriptor.descriptor_type = cmd::DescriptorType::Front;
    descriptor.body_words[0] = 0x100;
    descriptor.body_words[1] = 8;
    descriptor.body_words[2] = 4;
    descriptor.body_words[3] = 16;
    descriptor.body_words[4] = cmd::NONE;
    descriptor.body_words[5] = 0x2;
    descriptor.body_words[6] = 0x3;
    descriptor.body_words[7] = 0x4;
    descriptor.body_words[8] = static_cast<std::uint32_t>(policy);
    descriptor.body_words[9] = threshold_bits(threshold);
    return descriptor;
}

std::vector<cmd::Command> decode_commands(const std::vector<std::uint8_t>& data) {
    expect(data.size() % cmd::COMMAND_RECORD_BYTES == 0, "command fixture is truncated");
    std::vector<cmd::Command> commands;
    for (std::size_t offset = 0; offset < data.size(); offset += cmd::COMMAND_RECORD_BYTES) {
        commands.push_back(cmd::CommandCodec::decode(
            data.data() + offset, cmd::COMMAND_RECORD_BYTES));
    }
    return commands;
}

void test_frozen_constants() {
    expect(cmd::NONE == 0xFFFFFFFFu, "NONE value mismatch");
    expect(cmd::SCHEMA_VERSION == 2, "schema version mismatch");
    expect(cmd::COMMAND_RECORD_BYTES == 32, "command size mismatch");
    expect(cmd::DESCRIPTOR_RECORD_BYTES == 64, "descriptor size mismatch");
    expect(cmd::COMPLETION_RECORD_BYTES == 64, "completion size mismatch");
    expect(cmd::PVECTOR_INIT_VALUE == -1, "P-vector sentinel mismatch");
    expect(cmd::TRACE_ENABLE == 1 && cmd::ALLOW_RETRY == 2, "command flags mismatch");
    expect(static_cast<std::uint32_t>(cmd::Opcode::AbortNode) == 0x0D,
           "opcode range mismatch");
    expect(static_cast<std::uint16_t>(cmd::StatusCode::PolicyNotSupported) == 0x0006,
           "policy status code mismatch");
    expect(static_cast<std::uint16_t>(cmd::StatusCode::StaticPivotFailure) == 0x0105,
           "static pivot status code mismatch");
    expect(static_cast<std::uint16_t>(cmd::StatusCode::NonFiniteValue) == 0x0106,
           "non-finite status code mismatch");
    expect(static_cast<std::uint16_t>(cmd::StatusCode::PrecisionRetry) == 0x0104,
           "status code mismatch");
    expect(static_cast<std::uint32_t>(cmd::PivotPolicy::Static) == 0x01,
           "static policy code mismatch");
    expect(static_cast<std::uint32_t>(cmd::PivotPolicy::DynamicPanel) == 0x02,
           "dynamic policy code mismatch");
    expect(static_cast<std::uint32_t>(cmd::DataFormat::Fp32) == 0x02,
           "FP32 format code mismatch");
    expect(static_cast<std::uint32_t>(cmd::DataLayout::RowMajor) == 0x01,
           "row-major layout code mismatch");
    expect(static_cast<std::uint32_t>(cmd::KernelBackend::SystemCFp32DeviceModel) == 0x01,
           "FP32 backend code mismatch");
    expect(static_cast<std::uint32_t>(cmd::SolveDirection::Backward) == 0x02,
           "solve direction code mismatch");
}

void test_command(const std::string& fixture_dir) {
    cmd::Command command{};
    command.opcode = cmd::Opcode::GemmSchur;
    command.flags = cmd::TRACE_ENABLE | cmd::ALLOW_RETRY;
    command.command_id = 0x10203040u;
    command.node_id = 0x11223344u;
    command.descriptor_id = 5;
    command.wait_list_id = 8;
    command.signal_token = 9;

    const auto golden = read_file(fixture_dir + "/golden_command.bin");
    expect_equal(cmd::CommandCodec::encode(command), golden, "command golden mismatch");
    const auto decoded = cmd::CommandCodec::decode(golden.data(), golden.size());
    expect(decoded.opcode == command.opcode, "command opcode round-trip mismatch");
    expect(decoded.command_id == command.command_id, "command id round-trip mismatch");
    expect(decoded.node_id == command.node_id, "command node id round-trip mismatch");

    expect_throws(
        [&] { (void)cmd::CommandCodec::decode(golden.data(), golden.size() - 1); },
        "truncated command must fail");
    auto bad = golden;
    write_u32(bad, 0, 0x0E);
    expect_throws(
        [&] { (void)cmd::CommandCodec::decode(bad.data(), bad.size()); },
        "unknown opcode must fail");
    bad = golden;
    write_u32(bad, 4, 1u << 2);
    expect_throws(
        [&] { (void)cmd::CommandCodec::decode(bad.data(), bad.size()); },
        "reserved command flag must fail");
    bad = golden;
    write_u32(bad, 28, 1);
    expect_throws(
        [&] { (void)cmd::CommandCodec::decode(bad.data(), bad.size()); },
        "reserved command arg0 must fail");
}

void test_front_descriptor_policy(const std::string& fixture_dir) {
    const auto golden = read_file(fixture_dir + "/golden_descriptor.bin");
    expect_equal(
        cmd::DescriptorCodec::encode(front_descriptor(cmd::PivotPolicy::Static)),
        golden,
        "static FRONT_DESC golden mismatch");

    const auto dynamic = read_file(fixture_dir + "/dynamic_front_descriptor.bin");
    const auto decoded =
        cmd::DescriptorCodec::decode(dynamic.data(), dynamic.size());
    expect(decoded.descriptor_type == cmd::DescriptorType::Front,
           "front descriptor type mismatch");
    expect(decoded.body_words[8] ==
               static_cast<std::uint32_t>(cmd::PivotPolicy::DynamicPanel),
           "dynamic policy word mismatch");
    expect(decoded.body_words[9] == threshold_bits(0.000244140625f),
           "dynamic threshold bits mismatch");

    expect_equal(
        cmd::DescriptorCodec::encode(front_descriptor(
            cmd::PivotPolicy::DynamicPanel, 0.000244140625f)),
        dynamic,
        "dynamic FRONT_DESC golden mismatch");

    auto v1_record = golden;
    write_u32(v1_record, 56, 0);
    write_u32(v1_record, 60, 0);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(v1_record.data(), v1_record.size()); },
        "v1 FRONT_DESC must fail");

    auto bad = golden;
    write_u32(bad, 56, 0x03);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "unknown pivot policy must fail");

    bad = golden;
    write_u32(bad, 60, threshold_bits(1.0f));
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "threshold of 1.0 must fail");
    bad = golden;
    write_u32(bad, 60, 0xFFFFFFFFu);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "NaN threshold must fail");
    bad = golden;
    write_u32(bad, 60, threshold_bits(-0.25f));
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "negative threshold must fail");

    auto payload = front_descriptor(cmd::PivotPolicy::Static);
    payload.payload_bytes = 8;
    expect_throws(
        [&] { cmd::DescriptorCodec::validate(payload); },
        "FRONT_DESC payload must fail");
}

void test_descriptor(const std::string& fixture_dir) {
    auto descriptor = dependency_descriptor(0x80, 3);
    const auto golden = read_file(fixture_dir + "/golden_dependency_descriptor.bin");
    expect_equal(
        cmd::DescriptorCodec::encode(descriptor), golden, "descriptor golden mismatch");
    const auto decoded =
        cmd::DescriptorCodec::decode(golden.data(), golden.size(), 0x8C);
    expect(decoded.descriptor_type == cmd::DescriptorType::Dependency,
           "descriptor type round-trip mismatch");
    expect(decoded.body_words[0] == 3, "dependency token count mismatch");

    std::vector<std::uint8_t> image(0x8C, 0);
    const auto token_ids = read_file(fixture_dir + "/dependency_tokens.bin");
    std::copy(token_ids.begin(), token_ids.end(), image.begin() + 0x80);
    const auto tokens = cmd::DescriptorCodec::dependency_tokens(descriptor, image, 7, 4);
    expect(tokens == std::vector<std::uint32_t>({2, 4, 6}),
           "dependency tokens mismatch");

    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(golden.data(), golden.size() - 1); },
        "truncated descriptor must fail");
    auto bad = golden;
    write_u16(bad, 0, 9);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "unknown descriptor type must fail");
    bad = golden;
    write_u16(bad, 2, 1);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "reserved descriptor flags must fail");
    bad = golden;
    write_u32(bad, 4, 1);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "descriptor reserved field must fail");
    bad = golden;
    write_u32(bad, 28, 1);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::decode(bad.data(), bad.size()); },
        "descriptor reserved body word must fail");

    expect_throws(
        [&] { cmd::DescriptorCodec::validate(descriptor, 0x8B); },
        "descriptor payload bounds must fail");
    auto malformed = descriptor;
    malformed.payload_bytes = 8;
    expect_throws(
        [&] { cmd::DescriptorCodec::validate(malformed); },
        "dependency payload length must fail");
    malformed = descriptor;
    malformed.payload_offset = 0x81;
    expect_throws(
        [&] { cmd::DescriptorCodec::validate(malformed); },
        "dependency payload alignment must fail");

    auto region = region_descriptor(1, 64);
    expect_throws(
        [&] { cmd::DescriptorCodec::validate(region, 128); },
        "unaligned region must fail");
    region = region_descriptor(0xFFFFFFFFFFFFFFC0ull, 128);
    expect_throws(
        [&] { cmd::DescriptorCodec::validate(region); },
        "overflowing region must fail");
    region = region_descriptor(64, 128);
    expect_throws(
        [&] { cmd::DescriptorCodec::validate(region, 128); },
        "out-of-range region must fail");

    auto duplicate_image = image;
    write_u32(duplicate_image, 0x84, 2);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::dependency_tokens(descriptor, duplicate_image, 7); },
        "duplicate dependency token must fail");
    auto unknown_image = image;
    write_u32(unknown_image, 0x88, 7);
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::dependency_tokens(descriptor, unknown_image, 7); },
        "unknown dependency token must fail");
    expect_throws(
        [&] { (void)cmd::DescriptorCodec::dependency_tokens(descriptor, image, 7, 2); },
        "excess dependency token count must fail");
}

void test_completion(const std::string& fixture_dir) {
    cmd::Completion completion{};
    completion.command_id = 0x10203040u;
    completion.node_id = 0x11223344u;
    completion.status_code = cmd::StatusCode::PrecisionRetry;
    completion.pivot_count = 3;
    completion.start_cycle = 0x0102030405060708ull;
    completion.finish_cycle = 0x1112131415161718ull;
    completion.read_bytes = 0x2122232425262728ull;
    completion.write_bytes = 0x3132333435363738ull;
    completion.stall_cycles = 0x4142434445464748ull;
    completion.overflow_count = 2;
    completion.retry_count = 1;

    const auto golden = read_file(fixture_dir + "/golden_completion.bin");
    expect_equal(
        cmd::CompletionCodec::encode(completion), golden, "completion golden mismatch");
    const auto decoded = cmd::CompletionCodec::decode(golden.data(), golden.size());
    expect(decoded.status_code == completion.status_code,
           "completion status round-trip mismatch");
    expect(decoded.stall_cycles == completion.stall_cycles,
           "completion counters round-trip mismatch");

    expect_throws(
        [&] { (void)cmd::CompletionCodec::decode(golden.data(), golden.size() - 1); },
        "truncated completion must fail");
    auto bad = golden;
    write_u16(bad, 8, 0x0007);
    expect_throws(
        [&] { (void)cmd::CompletionCodec::decode(bad.data(), bad.size()); },
        "unknown status code must fail");
    bad = golden;
    write_u16(bad, 8, 0x0006);
    const auto refused = cmd::CompletionCodec::decode(bad.data(), bad.size());
    expect(refused.status_code == cmd::StatusCode::PolicyNotSupported,
           "policy refusal status must decode");
    bad = golden;
    write_u16(bad, 10, 1);
    expect_throws(
        [&] { (void)cmd::CompletionCodec::decode(bad.data(), bad.size()); },
        "completion reserved field must fail");
}

void test_token_states(const std::string& fixture_dir) {
    const std::vector<cmd::TokenState> states{
        cmd::TokenState::Unsignaled,
        cmd::TokenState::Ready,
        cmd::TokenState::Failed,
    };
    const auto golden = read_file(fixture_dir + "/token_states.bin");
    expect(cmd::encode_token_states(states) == golden, "token state golden mismatch");
    expect(cmd::decode_token_states(golden) == states, "token state round-trip mismatch");
    expect_throws(
        [&] { (void)cmd::decode_token_states(golden.data(), golden.size() - 1); },
        "truncated token state vector must fail");
    auto bad = golden;
    write_u32(bad, 0, 3);
    expect_throws(
        [&] { (void)cmd::decode_token_states(bad); },
        "unknown token state must fail");
}

void test_static_batch_fixtures(const std::string& fixture_dir) {
    const auto single = decode_commands(read_file(fixture_dir + "/single_node_commands.bin"));
    expect(single.size() == 2, "single-node fixture command count mismatch");
    expect(single[0].opcode == cmd::Opcode::NodeBegin,
           "single-node fixture opcode mismatch");
    expect(single[1].opcode == cmd::Opcode::NodeCommit,
           "single-node fixture commit mismatch");

    const auto tree =
        decode_commands(read_file(fixture_dir + "/parent_child_commands.bin"));
    expect(tree.size() == 3, "parent-child fixture command count mismatch");
    expect(tree[0].opcode == cmd::Opcode::StoreUpdate,
           "parent-child fixture opcode mismatch");
    expect(tree[1].opcode == cmd::Opcode::AssembleExtendAdd,
           "parent-child fixture assemble mismatch");

    std::vector<cmd::Descriptor> descriptors{
        dependency_descriptor(0, 1),
        dependency_descriptor(4, 1),
        region_descriptor(),
        region_descriptor(),
        region_descriptor(),
    };
    std::vector<std::uint8_t> image(8, 0);
    write_u32(image, 0, 2);
    write_u32(image, 4, 3);
    cmd::validate_command_batch(tree, descriptors, image, 5, 4);
    expect(
        cmd::DescriptorCodec::dependency_tokens(descriptors[0], image) ==
            std::vector<std::uint32_t>{2},
        "parent-child wait token mismatch");
    expect(
        cmd::DescriptorCodec::dependency_tokens(descriptors[1], image) ==
            std::vector<std::uint32_t>{3},
        "parent-child wait token mismatch");
    expect_throws(
        [&] {
            auto broken = tree;
            broken[1].signal_token = 2;
            cmd::validate_command_batch(broken, descriptors, image, 5, 4);
        },
        "duplicate signal token must fail");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            throw std::runtime_error("usage: command_codec_tests <fixture_dir>");
        }
        const std::string fixture_dir = argv[1];
        test_frozen_constants();
        test_command(fixture_dir);
        test_front_descriptor_policy(fixture_dir);
        test_descriptor(fixture_dir);
        test_completion(fixture_dir);
        test_token_states(fixture_dir);
        test_static_batch_fixtures(fixture_dir);
        std::cout << "command_codec_tests: all checks passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "command_codec_tests: " << error.what() << std::endl;
        return 1;
    }
}
