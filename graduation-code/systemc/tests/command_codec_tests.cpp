#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "command_codec.hpp"

namespace {

namespace v1 = hw::command_v1;

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

v1::Descriptor region_descriptor(std::uint64_t base_addr = 0, std::uint64_t byte_size = 0) {
    v1::Descriptor descriptor{};
    descriptor.descriptor_type = v1::DescriptorType::Region;
    descriptor.body_words[0] = static_cast<std::uint32_t>(base_addr);
    descriptor.body_words[1] = static_cast<std::uint32_t>(base_addr >> 32);
    descriptor.body_words[2] = static_cast<std::uint32_t>(byte_size);
    descriptor.body_words[3] = static_cast<std::uint32_t>(byte_size >> 32);
    descriptor.body_words[7] = static_cast<std::uint32_t>(v1::DataFormat::Fp32);
    descriptor.body_words[8] = static_cast<std::uint32_t>(v1::DataLayout::RowMajor);
    return descriptor;
}

v1::Descriptor dependency_descriptor(std::uint64_t offset, std::uint32_t count) {
    v1::Descriptor descriptor{};
    descriptor.descriptor_type = v1::DescriptorType::Dependency;
    descriptor.payload_offset = offset;
    descriptor.payload_bytes = static_cast<std::uint64_t>(count) * 4;
    descriptor.body_words[0] = count;
    return descriptor;
}

std::vector<v1::Command> decode_commands(const std::vector<std::uint8_t>& data) {
    expect(data.size() % v1::COMMAND_RECORD_BYTES == 0, "command fixture is truncated");
    std::vector<v1::Command> commands;
    for (std::size_t offset = 0; offset < data.size(); offset += v1::COMMAND_RECORD_BYTES) {
        commands.push_back(v1::CommandCodec::decode(
            data.data() + offset, v1::COMMAND_RECORD_BYTES));
    }
    return commands;
}

void test_frozen_constants() {
    expect(v1::NONE == 0xFFFFFFFFu, "NONE value mismatch");
    expect(v1::COMMAND_RECORD_BYTES == 32, "command size mismatch");
    expect(v1::DESCRIPTOR_RECORD_BYTES == 64, "descriptor size mismatch");
    expect(v1::COMPLETION_RECORD_BYTES == 64, "completion size mismatch");
    expect(v1::TRACE_ENABLE == 1 && v1::ALLOW_RETRY == 2, "command flags mismatch");
    expect(static_cast<std::uint32_t>(v1::Opcode::AbortNode) == 0x0D,
           "opcode range mismatch");
    expect(static_cast<std::uint16_t>(v1::StatusCode::PrecisionRetry) == 0x0104,
           "status code mismatch");
    expect(static_cast<std::uint32_t>(v1::DataFormat::Fp32) == 0x02,
           "FP32 format code mismatch");
    expect(static_cast<std::uint32_t>(v1::DataLayout::RowMajor) == 0x01,
           "row-major layout code mismatch");
    expect(static_cast<std::uint32_t>(v1::KernelBackend::SystemCFp32DeviceModel) == 0x01,
           "FP32 backend code mismatch");
    expect(static_cast<std::uint32_t>(v1::SolveDirection::Backward) == 0x02,
           "solve direction code mismatch");
}

void test_command(const std::string& fixture_dir) {
    v1::Command command{};
    command.opcode = v1::Opcode::GemmSchur;
    command.flags = v1::TRACE_ENABLE | v1::ALLOW_RETRY;
    command.command_id = 0x10203040u;
    command.node_id = 0x11223344u;
    command.descriptor_id = 5;
    command.wait_list_id = 8;
    command.signal_token = 9;

    const auto golden = read_file(fixture_dir + "/golden_command.bin");
    expect_equal(v1::CommandCodec::encode(command), golden, "command golden mismatch");
    const auto decoded = v1::CommandCodec::decode(golden.data(), golden.size());
    expect(decoded.opcode == command.opcode, "command opcode round-trip mismatch");
    expect(decoded.command_id == command.command_id, "command id round-trip mismatch");
    expect(decoded.node_id == command.node_id, "command node id round-trip mismatch");

    expect_throws(
        [&] { (void)v1::CommandCodec::decode(golden.data(), golden.size() - 1); },
        "truncated command must fail");
    auto bad = golden;
    write_u32(bad, 0, 0x0E);
    expect_throws(
        [&] { (void)v1::CommandCodec::decode(bad.data(), bad.size()); },
        "unknown opcode must fail");
    bad = golden;
    write_u32(bad, 4, 1u << 2);
    expect_throws(
        [&] { (void)v1::CommandCodec::decode(bad.data(), bad.size()); },
        "reserved command flag must fail");
    bad = golden;
    write_u32(bad, 28, 1);
    expect_throws(
        [&] { (void)v1::CommandCodec::decode(bad.data(), bad.size()); },
        "reserved command arg0 must fail");
}

void test_descriptor(const std::string& fixture_dir) {
    auto descriptor = dependency_descriptor(0x80, 3);
    const auto golden = read_file(fixture_dir + "/golden_descriptor.bin");
    expect_equal(
        v1::DescriptorCodec::encode(descriptor), golden, "descriptor golden mismatch");
    const auto decoded =
        v1::DescriptorCodec::decode(golden.data(), golden.size(), 0x8C);
    expect(decoded.descriptor_type == v1::DescriptorType::Dependency,
           "descriptor type round-trip mismatch");
    expect(decoded.body_words[0] == 3, "dependency token count mismatch");

    std::vector<std::uint8_t> image(0x8C, 0);
    const auto token_ids = read_file(fixture_dir + "/dependency_tokens.bin");
    std::copy(token_ids.begin(), token_ids.end(), image.begin() + 0x80);
    const auto tokens = v1::DescriptorCodec::dependency_tokens(descriptor, image, 7, 4);
    expect(tokens == std::vector<std::uint32_t>({2, 4, 6}),
           "dependency tokens mismatch");

    expect_throws(
        [&] { (void)v1::DescriptorCodec::decode(golden.data(), golden.size() - 1); },
        "truncated descriptor must fail");
    auto bad = golden;
    write_u16(bad, 0, 9);
    expect_throws(
        [&] { (void)v1::DescriptorCodec::decode(bad.data(), bad.size()); },
        "unknown descriptor type must fail");
    bad = golden;
    write_u16(bad, 2, 1);
    expect_throws(
        [&] { (void)v1::DescriptorCodec::decode(bad.data(), bad.size()); },
        "reserved descriptor flags must fail");
    bad = golden;
    write_u32(bad, 4, 1);
    expect_throws(
        [&] { (void)v1::DescriptorCodec::decode(bad.data(), bad.size()); },
        "descriptor reserved field must fail");
    bad = golden;
    write_u32(bad, 28, 1);
    expect_throws(
        [&] { (void)v1::DescriptorCodec::decode(bad.data(), bad.size()); },
        "descriptor reserved body word must fail");

    expect_throws(
        [&] { v1::DescriptorCodec::validate(descriptor, 0x8B); },
        "descriptor payload bounds must fail");
    auto malformed = descriptor;
    malformed.payload_bytes = 8;
    expect_throws(
        [&] { v1::DescriptorCodec::validate(malformed); },
        "dependency payload length must fail");
    malformed = descriptor;
    malformed.payload_offset = 0x81;
    expect_throws(
        [&] { v1::DescriptorCodec::validate(malformed); },
        "dependency payload alignment must fail");

    auto region = region_descriptor(1, 64);
    expect_throws(
        [&] { v1::DescriptorCodec::validate(region, 128); },
        "unaligned region must fail");
    region = region_descriptor(0xFFFFFFFFFFFFFFC0ull, 128);
    expect_throws(
        [&] { v1::DescriptorCodec::validate(region); },
        "overflowing region must fail");
    region = region_descriptor(64, 128);
    expect_throws(
        [&] { v1::DescriptorCodec::validate(region, 128); },
        "out-of-range region must fail");

    auto duplicate_image = image;
    write_u32(duplicate_image, 0x84, 2);
    expect_throws(
        [&] { (void)v1::DescriptorCodec::dependency_tokens(descriptor, duplicate_image, 7); },
        "duplicate dependency token must fail");
    auto unknown_image = image;
    write_u32(unknown_image, 0x88, 7);
    expect_throws(
        [&] { (void)v1::DescriptorCodec::dependency_tokens(descriptor, unknown_image, 7); },
        "unknown dependency token must fail");
    expect_throws(
        [&] { (void)v1::DescriptorCodec::dependency_tokens(descriptor, image, 7, 2); },
        "excess dependency token count must fail");
}

void test_completion(const std::string& fixture_dir) {
    v1::Completion completion{};
    completion.command_id = 0x10203040u;
    completion.node_id = 0x11223344u;
    completion.status_code = v1::StatusCode::PrecisionRetry;
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
        v1::CompletionCodec::encode(completion), golden, "completion golden mismatch");
    const auto decoded = v1::CompletionCodec::decode(golden.data(), golden.size());
    expect(decoded.status_code == completion.status_code,
           "completion status round-trip mismatch");
    expect(decoded.stall_cycles == completion.stall_cycles,
           "completion counters round-trip mismatch");

    expect_throws(
        [&] { (void)v1::CompletionCodec::decode(golden.data(), golden.size() - 1); },
        "truncated completion must fail");
    auto bad = golden;
    write_u16(bad, 8, 0x0006);
    expect_throws(
        [&] { (void)v1::CompletionCodec::decode(bad.data(), bad.size()); },
        "unknown status code must fail");
    bad = golden;
    write_u16(bad, 10, 1);
    expect_throws(
        [&] { (void)v1::CompletionCodec::decode(bad.data(), bad.size()); },
        "completion reserved field must fail");
}

void test_token_states(const std::string& fixture_dir) {
    const std::vector<v1::TokenState> states{
        v1::TokenState::Unsignaled,
        v1::TokenState::Ready,
        v1::TokenState::Failed,
    };
    const auto golden = read_file(fixture_dir + "/token_states.bin");
    expect(v1::encode_token_states(states) == golden, "token state golden mismatch");
    expect(v1::decode_token_states(golden) == states, "token state round-trip mismatch");
    expect_throws(
        [&] { (void)v1::decode_token_states(golden.data(), golden.size() - 1); },
        "truncated token state vector must fail");
    auto bad = golden;
    write_u32(bad, 0, 3);
    expect_throws(
        [&] { (void)v1::decode_token_states(bad); },
        "unknown token state must fail");
}

void test_static_batch_fixtures(const std::string& fixture_dir) {
    const auto single = decode_commands(read_file(fixture_dir + "/single_node_commands.bin"));
    expect(single.size() == 2, "single-node fixture command count mismatch");
    expect(single[0].opcode == v1::Opcode::NodeBegin &&
               single[1].opcode == v1::Opcode::NodeCommit,
           "single-node fixture opcode mismatch");
    expect(single[1].wait_list_id == 1, "NODE_COMMIT must wait for its dependency");
    std::vector<std::uint8_t> single_image(4, 0);
    std::vector<v1::Descriptor> single_descriptors{
        region_descriptor(), dependency_descriptor(0, 1)};
    v1::validate_command_batch(single, single_descriptors, single_image, 2, 4);

    const auto tree =
        decode_commands(read_file(fixture_dir + "/parent_child_commands.bin"));
    expect(tree.size() == 3, "parent-child fixture command count mismatch");
    expect(tree[0].opcode == v1::Opcode::StoreUpdate &&
               tree[1].opcode == v1::Opcode::AssembleExtendAdd &&
               tree[2].opcode == v1::Opcode::NodeCommit,
           "parent-child fixture opcode mismatch");
    expect(tree[1].wait_list_id == 0, "parent assembly dependency mismatch");
    expect(tree[2].wait_list_id == 1, "parent commit dependency mismatch");
    std::vector<std::uint8_t> tree_image(8, 0);
    write_u32(tree_image, 0, 2);
    write_u32(tree_image, 4, 3);
    std::vector<v1::Descriptor> tree_descriptors{
        dependency_descriptor(0, 1),
        dependency_descriptor(4, 1),
        region_descriptor(),
        region_descriptor(),
        region_descriptor(),
    };
    v1::validate_command_batch(tree, tree_descriptors, tree_image, 5, 4);

    auto bad = tree;
    bad[1].descriptor_id = 5;
    expect_throws(
        [&] { v1::validate_command_batch(bad, tree_descriptors, tree_image, 5); },
        "unknown descriptor reference must fail");
    bad = tree;
    bad[1].signal_token = 2;
    expect_throws(
        [&] { v1::validate_command_batch(bad, tree_descriptors, tree_image, 5); },
        "duplicate signal producer must fail");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: command_codec_tests FIXTURE_DIR");
        }
        const std::string fixture_dir = argv[1];
        test_frozen_constants();
        test_command(fixture_dir);
        test_descriptor(fixture_dir);
        test_completion(fixture_dir);
        test_token_states(fixture_dir);
        test_static_batch_fixtures(fixture_dir);
        std::cout << "[COMMAND_CODEC_TESTS] ALL PASSED\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[COMMAND_CODEC_TESTS] FAILED: " << exc.what() << '\n';
        return 1;
    }
}
