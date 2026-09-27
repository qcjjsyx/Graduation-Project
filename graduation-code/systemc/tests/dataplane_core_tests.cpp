// Plain C++ tests for the T04 data/validation layer: artifact loading,
// negative artifact cases, DeviceMemory access rules, BufferManager lifecycle
// and MemoryService timing/backpressure. No SystemC required.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataplane/artifact_loader.hpp"
#include "dataplane/buffer_manager.hpp"
#include "dataplane/device_memory.hpp"
#include "dataplane/memory_service.hpp"
#include "nlohmann/json.hpp"

namespace {

namespace cmd = hw::command;
namespace dp = hw::dataplane;
using json = nlohmann::json;

int failures = 0;
int checks = 0;

void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void expect_load_throws(
    const std::function<void()>& action,
    const std::string& message) {
    ++checks;
    try {
        action();
    } catch (const dp::ArtifactLoadError&) {
        return;
    } catch (const std::exception& exception) {
        ++failures;
        std::cerr << "FAIL: " << message << " (unexpected exception: "
                  << exception.what() << ")\n";
        return;
    }
    ++failures;
    std::cerr << "FAIL: " << message << " (no exception)\n";
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

struct Fixture {
    std::filesystem::path directory;
    std::string manifest_text;
    std::vector<std::uint8_t> image;
    json manifest;
};

Fixture load_fixture(const std::filesystem::path& device_dir) {
    Fixture fixture;
    fixture.directory = device_dir;
    fixture.manifest_text = read_text(device_dir / "manifest.json");
    fixture.image = read_file(device_dir / "memory_image.bin");
    fixture.manifest = json::parse(fixture.manifest_text);
    return fixture;
}

dp::LoadedArtifact load_mutated(
    Fixture fixture,
    const std::function<void(json&)>& mutate_manifest = {},
    const std::function<void(std::vector<std::uint8_t>&, const json&)>& mutate_image = {}) {
    if (mutate_manifest) {
        mutate_manifest(fixture.manifest);
    }
    if (mutate_image) {
        mutate_image(fixture.image, fixture.manifest);
    }
    return dp::load_artifact_from_bytes(
        fixture.manifest.dump(), std::move(fixture.image), "<test>");
}

std::uint64_t region_offset(const json& manifest, const std::string& name) {
    return manifest.at("memory_image").at("regions").at(name).at("offset").get<std::uint64_t>();
}

std::uint64_t region_size(const json& manifest, const std::string& name) {
    return manifest.at("memory_image").at("regions").at(name).at("size").get<std::uint64_t>();
}

void patch_u32(std::vector<std::uint8_t>& image, std::uint64_t address, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        image[address + index] = static_cast<std::uint8_t>((value >> (8 * index)) & 0xFFu);
    }
}

cmd::Descriptor descriptor_at(
    const std::vector<std::uint8_t>& image,
    const json& manifest,
    std::uint32_t id) {
    const auto base = region_offset(manifest, "descriptor_table");
    return cmd::DescriptorCodec::decode(
        image.data() + base + static_cast<std::uint64_t>(id) * cmd::DESCRIPTOR_RECORD_BYTES,
        cmd::DESCRIPTOR_RECORD_BYTES,
        image.size());
}

std::optional<std::uint32_t> find_front_descriptor(
    const std::vector<std::uint8_t>& image,
    const json& manifest) {
    const auto count =
        manifest.at("descriptor_table").at("descriptor_count").get<std::uint32_t>();
    for (std::uint32_t id = 0; id < count; ++id) {
        if (descriptor_at(image, manifest, id).descriptor_type == cmd::DescriptorType::Front) {
            return id;
        }
    }
    return std::nullopt;
}

std::optional<std::uint32_t> find_region_descriptor_for_offset(
    const std::vector<std::uint8_t>& image,
    const json& manifest,
    std::uint64_t offset,
    std::uint64_t size) {
    const auto count =
        manifest.at("descriptor_table").at("descriptor_count").get<std::uint32_t>();
    for (std::uint32_t id = 0; id < count; ++id) {
        const auto descriptor = descriptor_at(image, manifest, id);
        if (descriptor.descriptor_type != cmd::DescriptorType::Region) {
            continue;
        }
        const std::uint64_t base = descriptor.body_words[0] |
                                   (static_cast<std::uint64_t>(descriptor.body_words[1]) << 32);
        const std::uint64_t bytes = descriptor.body_words[2] |
                                    (static_cast<std::uint64_t>(descriptor.body_words[3]) << 32);
        if (base == offset && bytes == size) {
            return id;
        }
    }
    return std::nullopt;
}

void test_load_valid_device_dirs(const std::filesystem::path& fixtures_root) {
    std::size_t loaded = 0;
    for (const auto& entry : std::filesystem::directory_iterator(fixtures_root)) {
        const auto device = entry.path() / "device";
        if (!std::filesystem::exists(device / "manifest.json")) {
            continue;
        }
        // Device run directory contains only manifest + image.
        std::size_t files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(device)) {
            (void)entry;
            ++files;
        }
        expect(files == 2, "device dir must contain only manifest and image");
        dp::LoadedArtifact artifact;
        try {
            artifact = dp::load_artifact(device.string());
        } catch (const std::exception& exception) {
            ++failures;
            std::cerr << "FAIL: load " << device << ": " << exception.what() << '\n';
            continue;
        }
        ++loaded;
        expect(artifact.config.command_count > 0, "command count must be positive");
        expect(artifact.config.descriptor_count > 0, "descriptor count must be positive");
        expect(
            artifact.config.completion_count == artifact.config.command_count,
            "completion slots must match commands");
        expect(artifact.image.size() == artifact.config.total_bytes, "image size mismatch");
        const dp::RegionInfo* command = artifact.config.region("command_buffer");
        expect(command != nullptr && command->role == dp::RegionRole::Command,
               "command region role");
        const dp::RegionInfo* rhs = artifact.config.region("rhs_data");
        expect(rhs != nullptr && rhs->role == dp::RegionRole::DataInput,
               "rhs region must be read-only input");
    }
    expect(loaded >= 4, "expected at least four generated fixtures");
}

void test_negative_artifact_cases(const std::filesystem::path& fixtures_root) {
    const Fixture base = load_fixture(fixtures_root / "single_static" / "device");
    const auto front = find_front_descriptor(base.image, base.manifest);
    expect(front.has_value(), "fixture has a FRONT_DESC");
    if (!front.has_value()) {
        return;
    }
    const auto front_descriptor = descriptor_at(base.image, base.manifest, *front);

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) { manifest["abi"]["version"] = 1; });
        },
        "schema v1 manifest is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, {}, [](std::vector<std::uint8_t>& image, const json&) {
                image.pop_back();
            });
        },
        "truncated image is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) {
                manifest["total_bytes"] = manifest["total_bytes"].get<std::uint64_t>() + 64;
            });
        },
        "total_bytes mismatch is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) {
                manifest["memory_image"]["regions"]["node.0.factor_l"]["offset"] =
                    manifest["memory_image"]["regions"]["node.0.front"]["offset"];
            });
        },
        "region overlap is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) {
                manifest["memory_image"]["regions"]["node.0.update"]["offset"] = 4;
            });
        },
        "unaligned region offset is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) {
                manifest["memory_image"]["regions"]["node.0.update"]["offset"] =
                    0xFFFFFFFFFFFFFFF0ull;
                manifest["memory_image"]["regions"]["node.0.update"]["size"] = 32;
            });
        },
        "region range overflow beyond total_bytes is rejected");

    // FRONT_DESC front region reference out of range.
    expect_load_throws(
        [&] {
            load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                const auto base_addr = region_offset(manifest, "descriptor_table");
                patch_u32(
                    image,
                    base_addr + static_cast<std::uint64_t>(*front) * cmd::DESCRIPTOR_RECORD_BYTES +
                        24,
                    0xFFFFFFF0u);
            });
        },
        "descriptor reference out of range is rejected");

    // P-vector points at the FP32 RHS region.
    const auto rhs_region = find_region_descriptor_for_offset(
        base.image, base.manifest,
        region_offset(base.manifest, "rhs_data"),
        region_size(base.manifest, "rhs_data"));
    expect(rhs_region.has_value(), "fixture has an RHS REGION_DESC");
    if (rhs_region.has_value()) {
        expect_load_throws(
            [&] {
                load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                    const auto base_addr = region_offset(manifest, "descriptor_table");
                    patch_u32(
                        image,
                        base_addr +
                            static_cast<std::uint64_t>(*front) * cmd::DESCRIPTOR_RECORD_BYTES +
                            24 + 6 * 4,
                        *rhs_region);
                });
            },
            "P-vector pointing at FP32 RHS is rejected");
    }

    // P-vector capacity no longer matches pivot_dim.
    expect_load_throws(
        [&] {
            load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                const auto base_addr = region_offset(manifest, "descriptor_table");
                patch_u32(
                    image,
                    base_addr + static_cast<std::uint64_t>(*front) * cmd::DESCRIPTOR_RECORD_BYTES +
                        24 + 2 * 4,
                    front_descriptor.body_words[2] + 1);
            });
        },
        "P-vector capacity mismatch is rejected");

    // P-vector content no longer the sentinel.
    {
        const auto p_region = descriptor_at(base.image, base.manifest, front_descriptor.body_words[6]);
        const std::uint64_t p_base = p_region.body_words[0] |
                                     (static_cast<std::uint64_t>(p_region.body_words[1]) << 32);
        expect_load_throws(
            [&] {
                load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json&) {
                    patch_u32(image, p_base, 0);
                });
            },
            "non-sentinel P-vector content is rejected");
    }

    // FACTOR_DESC and FRONT_DESC reference different P-vectors.
    {
        const auto factor_id = front_descriptor.body_words[5];
        const auto other_front = [&]() -> std::optional<std::uint32_t> {
            const auto count = base.manifest.at("descriptor_table").at("descriptor_count");
            for (std::uint32_t id = 0; id < count.get<std::uint32_t>(); ++id) {
                const auto descriptor = descriptor_at(base.image, base.manifest, id);
                if (descriptor.descriptor_type == cmd::DescriptorType::Front &&
                    id != *front) {
                    return id;
                }
            }
            return std::nullopt;
        }();
        if (other_front.has_value()) {
            const auto other = descriptor_at(base.image, base.manifest, *other_front);
            expect_load_throws(
                [&] {
                    load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                        const auto base_addr = region_offset(manifest, "descriptor_table");
                        patch_u32(
                            image,
                            base_addr +
                                static_cast<std::uint64_t>(factor_id) *
                                    cmd::DESCRIPTOR_RECORD_BYTES +
                                24 + 2 * 4,
                            other.body_words[6]);
                    });
                },
                "cross-descriptor P-vector inconsistency is rejected");
        }
    }

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) {
                manifest["pivot"]["policy"] = "DYNAMIC_PANEL";
            });
        },
        "manifest/descriptor policy mismatch is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, [](json& manifest) {
                manifest["pivot"]["threshold_bits_fp32"] =
                    manifest["pivot"]["threshold_bits_fp32"].get<std::uint32_t>() ^ 1u;
            });
        },
        "manifest/descriptor threshold mismatch is rejected");

    expect_load_throws(
        [&] {
            load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                const auto base_addr = region_offset(manifest, "descriptor_table");
                patch_u32(
                    image,
                    base_addr + static_cast<std::uint64_t>(*front) * cmd::DESCRIPTOR_RECORD_BYTES +
                        24 + 8 * 4,
                    0);
            });
        },
        "v1 FRONT_DESC policy word is rejected");

    // Illegal region layout / format / stride on the front REGION_DESC.
    const auto front_region_id = front_descriptor.body_words[0];
    expect_load_throws(
        [&] {
            load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                const auto base_addr = region_offset(manifest, "descriptor_table");
                patch_u32(
                    image,
                    base_addr +
                        static_cast<std::uint64_t>(front_region_id) * cmd::DESCRIPTOR_RECORD_BYTES +
                        24 + 8 * 4,
                    0);
            });
        },
        "illegal REGION_DESC layout is rejected");
    expect_load_throws(
        [&] {
            load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                const auto base_addr = region_offset(manifest, "descriptor_table");
                patch_u32(
                    image,
                    base_addr +
                        static_cast<std::uint64_t>(front_region_id) * cmd::DESCRIPTOR_RECORD_BYTES +
                        24 + 7 * 4,
                    static_cast<std::uint32_t>(cmd::DataFormat::Fp64));
            });
        },
        "FP64 device region is rejected");
    expect_load_throws(
        [&] {
            load_mutated(base, {}, [&](std::vector<std::uint8_t>& image, const json& manifest) {
                const auto base_addr = region_offset(manifest, "descriptor_table");
                patch_u32(
                    image,
                    base_addr +
                        static_cast<std::uint64_t>(front_region_id) * cmd::DESCRIPTOR_RECORD_BYTES +
                        24 + 4 * 4,
                    4);
            });
        },
        "row stride smaller than one row is rejected");
}

void test_device_memory() {
    std::vector<std::uint8_t> image(1024, 0xAB);
    std::vector<dp::RegionInfo> regions{
        {"command_buffer", dp::RegionRole::Command, 0, 64},
        {"descriptor_table", dp::RegionRole::Descriptor, 64, 128},
        {"descriptor_payload", dp::RegionRole::Payload, 192, 32},
        {"completion_status", dp::RegionRole::Completion, 256, 64},
        {"rhs_data", dp::RegionRole::DataInput, 320, 64},
        {"node.0.front", dp::RegionRole::Data, 384, 256},
        {"node.0.p_vector", dp::RegionRole::PVector, 640, 64},
    };
    dp::DeviceMemory memory(image, regions);

    expect(memory.size() == 1024, "device memory size");
    expect(memory.find_region_by_name("node.0.front") != nullptr, "region lookup by name");

    expect(
        memory.check(384, 16, dp::AccessKind::Fp32Data, false).ok,
        "aligned FP32 data read");
    expect(
        !memory.check(386, 16, dp::AccessKind::Fp32Data, false).ok,
        "unaligned FP32 access rejected");
    expect(
        memory.check(390, 16, dp::AccessKind::ByteData, false).ok,
        "byte access does not require 64B alignment");
    expect(
        !memory.check(376, 32, dp::AccessKind::ByteData, false).ok,
        "access crossing into padding rejected");
    expect(
        !memory.check(384 - 8, 16, dp::AccessKind::ByteData, false).ok,
        "access crossing a region boundary rejected");
    expect(
        !memory.check(0xFFFFFFFFFFFFFFF0ull, 32, dp::AccessKind::Fp32Data, false).ok,
        "u64 overflow rejected");
    expect(
        !memory.check(0, 32, dp::AccessKind::ByteData, true).ok,
        "write to command region rejected");
    expect(
        !memory.check(320, 16, dp::AccessKind::ByteData, true).ok,
        "write to read-only data input rejected");
    expect(
        memory.check(256, 64, dp::AccessKind::Completion, true).ok,
        "completion region is writable");
    expect(
        !memory.check(384, 64, dp::AccessKind::Descriptor, false).ok,
        "descriptor access outside descriptor_table rejected");
    expect(
        !memory.check(64, 32, dp::AccessKind::Command, false).ok,
        "wrong record access kind rejected");

    // Zero-length semantics.
    expect(
        memory.check(384, 0, dp::AccessKind::Fp32Data, false).ok,
        "zero-length inside a region is a legal no-op");
    expect(
        memory.check(384 + 256, 0, dp::AccessKind::Fp32Data, false).ok,
        "zero-length at a region end is legal");
    expect(
        !memory.check(250, 0, dp::AccessKind::Fp32Data, false).ok,
        "zero-length outside any region is rejected");
    expect(
        !memory.check(320, 0, dp::AccessKind::ByteData, true).ok,
        "zero-length write to read-only region is rejected");

    // Illegal writes leave memory untouched.
    const auto before = memory.snapshot();
    std::vector<std::uint8_t> data{1, 2, 3, 4};
    expect(
        memory.write(0, data, dp::AccessKind::ByteData) == cmd::StatusCode::AddressFault,
        "illegal write returns AddressFault");
    expect(memory.snapshot() == before, "illegal write does not change memory");

    // Legal write and read back.
    expect(
        memory.write(400, data, dp::AccessKind::Fp32Data) == cmd::StatusCode::Ok,
        "legal data write");
    std::vector<std::uint8_t> read_back;
    expect(
        memory.read(400, 4, dp::AccessKind::Fp32Data, read_back) == cmd::StatusCode::Ok,
        "legal data read");
    expect(read_back == data, "read matches written bytes");
}

void test_buffer_manager() {
    dp::BufferConfig config;
    config.slot_count = 2;
    config.capacity_bytes = 256;
    config.port_count = 2;
    config.access_cycles = 1;
    dp::BufferManager buffers(config);

    auto a = buffers.allocate("dma", 128, 0);
    expect(a.ok && buffers.state(a.handle.slot) == dp::BufferState::Reserved, "allocate A");
    auto b = buffers.allocate("consumer", 128, 0);
    expect(b.ok && b.handle.slot != a.handle.slot, "allocate B in another slot");
    auto c = buffers.allocate("dma", 8, 0);
    expect(!c.ok && c.status == cmd::StatusCode::BufferFull, "third allocation is full");

    // Wrong owner and stale/busy access rules.
    auto wrong_owner = buffers.begin_load(b.handle, "dma", 0, 128, 0);
    expect(!wrong_owner.ok, "wrong owner is rejected");
    auto read_before_load = buffers.begin_read(a.handle, "dma", 0, 4, 0);
    expect(!read_before_load.ok, "read before finish_load is rejected");
    auto too_large = buffers.begin_load(a.handle, "dma", 0, 200, 0);
    expect(!too_large.ok, "capacity overflow is rejected");
    expect(buffers.state(a.handle.slot) == dp::BufferState::Reserved,
           "failed begin_load leaves state unchanged");

    auto load_a = buffers.begin_load(a.handle, "dma", 0, 128, 10);
    expect(load_a.ok, "begin_load A");
    expect(load_a.lease.ready_cycle == 11, "lease latency is charged once");
    std::vector<std::uint8_t> pattern(128, 0x5A);
    std::string reason;
    expect(
        buffers.lease_write(load_a.lease.lease_id, 0, pattern, 10, &reason) ==
            cmd::StatusCode::BufferFull,
        "buffer lease write before ready_cycle stalls (BufferFull)");
    expect(
        buffers.lease_write(load_a.lease.lease_id, 0, pattern, 11, &reason) ==
            cmd::StatusCode::Ok,
        "buffer lease write after ready_cycle succeeds");
    // Conflicting access on the same slot while LOADING.
    auto conflict = buffers.begin_write(a.handle, "dma", 0, 128, 11);
    expect(!conflict.ok, "conflicting access to a LOADING buffer is rejected");
    // Port exhaustion: A holds a lease, B lease plus one more would exceed 2.
    auto read_a_while_loading = buffers.begin_read(a.handle, "dma", 0, 4, 11);
    expect(!read_a_while_loading.ok, "read while LOADING is rejected");
    expect(
        buffers.finish_load(load_a.lease.lease_id, 128, 12) == cmd::StatusCode::Ok,
        "finish_load A");
    expect(buffers.state(a.handle.slot) == dp::BufferState::Ready, "A is READY");
    expect(buffers.valid_bytes(a.handle.slot) == 128, "A valid range recorded");

    // Write lease conflict: A read lease + B untouched; ports = 2.
    auto read_a = buffers.begin_read(a.handle, "dma", 0, 128, 13);
    expect(read_a.ok, "read lease A");
    auto write_same = buffers.begin_write(a.handle, "dma", 0, 128, 13);
    expect(!write_same.ok, "read/write conflict on the same slot is rejected");
    auto load_b = buffers.begin_load(b.handle, "consumer", 0, 128, 13);
    expect(load_b.ok, "second port usable for another slot");
    auto third = buffers.begin_read(a.handle, "dma", 0, 4, 13);
    expect(!third.ok && third.status == cmd::StatusCode::BufferFull,
           "third concurrent lease exceeds port count");
    expect(
        buffers.release_lease(read_a.lease.lease_id, 13) == cmd::StatusCode::Ok,
        "read lease release");
    expect(
        buffers.finish_load(load_b.lease.lease_id, 128, 14) == cmd::StatusCode::Ok,
        "finish_load B");

    // Ownership handoff and wrong owner afterwards.
    auto handoff = buffers.transfer_owner(a.handle, "dma", "consumer", 15);
    expect(handoff.ok, "ownership handoff");
    auto stale_owner = buffers.begin_read(a.handle, "dma", 0, 4, 15);
    expect(!stale_owner.ok, "old owner rejected after handoff");
    auto consumer_read = buffers.begin_read(a.handle, "consumer", 0, 4, 15);
    expect(consumer_read.ok, "new owner accepted after handoff");
    buffers.release_lease(consumer_read.lease.lease_id, 15);

    // Release and stale handle / double release.
    auto release_a = buffers.release(a.handle, "consumer", 16);
    expect(release_a.ok, "release A");
    auto double_release = buffers.release(a.handle, "consumer", 16);
    expect(!double_release.ok, "double release is rejected");
    auto stale = buffers.begin_read(a.handle, "consumer", 0, 4, 16);
    expect(!stale.ok && stale.status == cmd::StatusCode::BadDescriptor,
           "stale handle is rejected");

    // Reset invalidates handles and frees slots.
    auto new_a = buffers.allocate("dma", 64, 17);
    expect(new_a.ok, "allocate after release");
    buffers.reset();
    expect(buffers.state(new_a.handle.slot) == dp::BufferState::Free, "reset frees slots");
    auto after_reset = buffers.begin_load(new_a.handle, "dma", 0, 64, 18);
    expect(!after_reset.ok, "handle across reset is stale");
    auto reusable = buffers.allocate("dma", 64, 18);
    expect(reusable.ok, "slots reusable after reset");

    // Lease bounds must reject offsets past the lease end without unsigned
    // underflow (previously end - offset wrapped around).
    auto bounds = buffers.allocate("dma", 8, 20);
    expect(bounds.ok, "bounds: allocate");
    auto bounds_load = buffers.begin_load(bounds.handle, "dma", 0, 8, 20);
    expect(bounds_load.ok, "bounds: begin_load");
    const std::uint64_t ready = bounds_load.lease.ready_cycle;
    const std::vector<std::uint8_t> small{1, 2, 3, 4};
    const auto storage_before = buffers.bytes(bounds.handle.slot);
    expect(
        buffers.lease_write(bounds_load.lease.lease_id, 16, small, ready) ==
            cmd::StatusCode::AddressFault,
        "write past the lease end is rejected");
    expect(
        buffers.lease_write(
            bounds_load.lease.lease_id, 0xFFFFFFFFFFFFFFF0ull, small, ready) ==
            cmd::StatusCode::AddressFault,
        "huge write offset is rejected without wrapping");
    std::vector<std::uint8_t> out;
    expect(
        buffers.lease_read(bounds_load.lease.lease_id, 16, 4, out, ready) ==
            cmd::StatusCode::AddressFault,
        "read past the lease end is rejected");
    expect(
        buffers.lease_read(
            bounds_load.lease.lease_id, 0xFFFFFFFFFFFFFFF8ull, 4, out, ready) ==
            cmd::StatusCode::AddressFault,
        "huge read offset is rejected without wrapping");
    expect(
        buffers.bytes(bounds.handle.slot) == storage_before,
        "rejected lease access does not modify storage");
}

void test_memory_service() {
    std::vector<std::uint8_t> image(512, 0);
    std::vector<dp::RegionInfo> regions{
        {"command_buffer", dp::RegionRole::Command, 0, 64},
        {"node.0.front", dp::RegionRole::Data, 64, 256},
    };
    dp::DeviceMemory memory(image, regions);
    dp::MemoryServiceConfig config;
    config.startup_cycles = 1;
    config.beat_bytes = 8;
    config.beat_cycles = 1;
    config.max_inflight = 1;
    dp::MemoryService service(memory, config);

    // Timing: 16 bytes => 1 + 2 beats = 3 cycles after accept.
    dp::MemoryRequest read;
    read.request_id = service.next_request_id();
    read.client = dp::MemoryClient::Test;
    read.address = 64;
    read.length = 16;
    read.kind = dp::AccessKind::Fp32Data;
    expect(service.submit(read, 5), "submit read");
    expect(service.port_free_cycle() == 8, "finish cycle = accept + startup + beats");
    // Queue full while the first transaction is in flight.
    dp::MemoryRequest blocked = read;
    blocked.request_id = service.next_request_id();
    expect(!service.submit(blocked, 5), "queue full returns backpressure");
    service.tick(7);
    dp::MemoryResponse response;
    expect(!service.try_pop_response(response), "response not ready before finish");
    service.tick(8);
    expect(service.try_pop_response(response), "response at finish cycle");
    expect(response.status == cmd::StatusCode::Ok, "read succeeds");
    expect(response.finish_cycle == 8, "finish cycle recorded");
    expect(response.bytes == 16, "byte count recorded");

    // Illegal access: accepted with one terminal error and no memory change.
    const auto before = memory.snapshot();
    dp::MemoryRequest illegal;
    illegal.request_id = service.next_request_id();
    illegal.client = dp::MemoryClient::Test;
    illegal.address = 0;
    illegal.length = 8;
    illegal.kind = dp::AccessKind::ByteData;
    illegal.write = true;
    expect(service.submit(illegal, 9), "illegal request is accepted for a terminal answer");
    expect(service.try_pop_response(response), "illegal request answered immediately");
    expect(response.status == cmd::StatusCode::AddressFault, "illegal write is AddressFault");
    expect(memory.snapshot() == before, "illegal request does not modify memory");

    // Zero-length no-op.
    dp::MemoryRequest empty;
    empty.request_id = service.next_request_id();
    empty.client = dp::MemoryClient::Test;
    empty.address = 64;
    empty.length = 0;
    empty.kind = dp::AccessKind::Fp32Data;
    expect(service.submit(empty, 10), "zero-length submit");
    expect(service.try_pop_response(response), "zero-length response is immediate");
    expect(response.bytes == 0 && response.service_cycles == 0, "zero-length is a no-op");

    // Write length and payload must agree; a mismatch is a terminal error
    // that must not modify memory or occupy the data port.
    dp::MemoryRequest mismatch;
    mismatch.request_id = service.next_request_id();
    mismatch.client = dp::MemoryClient::Test;
    mismatch.write = true;
    mismatch.address = 64;
    mismatch.length = 4;
    mismatch.kind = dp::AccessKind::Fp32Data;
    mismatch.write_data.assign(16, 0x22);
    const auto port_free_before = service.port_free_cycle();
    const auto before_mismatch = memory.snapshot();
    expect(
        service.submit(mismatch, 20),
        "mismatched write is accepted for a terminal answer");
    expect(service.try_pop_response(response), "mismatched write is answered immediately");
    expect(
        response.status == cmd::StatusCode::BadCommand,
        "write length mismatch is BadCommand");
    expect(
        memory.snapshot() == before_mismatch,
        "mismatched write does not modify memory");
    expect(
        service.port_free_cycle() == port_free_before,
        "mismatched write does not occupy the data port");

    // Reset drops in-flight work and bumps the epoch.
    dp::MemoryRequest pending = read;
    pending.request_id = service.next_request_id();
    pending.write = true;
    dp::MemoryRequest write_request = pending;
    write_request.address = 80;
    write_request.write_data.assign(16, 0x11);
    const auto epoch_before = service.epoch();
    expect(service.submit(write_request, 11), "submit before reset");
    service.reset();
    expect(service.epoch() == epoch_before + 1, "reset bumps epoch");
    service.tick(100);
    expect(!service.try_pop_response(response), "no stale response after reset");
    const auto after_reset_image = memory.snapshot();
    expect(
        std::all_of(
            after_reset_image.begin() + 80, after_reset_image.begin() + 96,
            [](std::uint8_t value) { return value == 0; }),
        "dropped write is not applied after reset");
}

void test_zero_size_region_memory() {
    std::vector<std::uint8_t> image(128, 0);
    std::vector<dp::RegionInfo> regions{
        {"empty_input", dp::RegionRole::DataInput, 64, 0},
        {"node.0.front", dp::RegionRole::Data, 128, 64},
    };
    dp::DeviceMemory memory(image, regions);
    expect(memory.check(64, 0, dp::AccessKind::Fp32Data, false).ok,
           "zero-size region accepts a zero-length read");
    expect(memory.check(64, 4, dp::AccessKind::Fp32Data, false).ok == false,
           "non-zero access cannot use a zero-size region");
    expect(!memory.check(200, 0, dp::AccessKind::Fp32Data, false).ok,
           "zero-length outside any region is rejected");
    expect(!memory.check(64, 0, dp::AccessKind::ByteData, true).ok,
           "zero-length write to a read-only zero-size region is rejected");
    std::vector<std::uint8_t> out;
    expect(memory.read(64, 0, dp::AccessKind::Fp32Data, out) == cmd::StatusCode::Ok,
           "zero-length read is a successful no-op");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::filesystem::path fixtures_root = argv[1];
        test_load_valid_device_dirs(fixtures_root);
        test_negative_artifact_cases(fixtures_root);
    } else {
        std::cout << "note: no fixture directory; artifact tests skipped\n";
    }
    test_device_memory();
    test_buffer_manager();
    test_memory_service();
    test_zero_size_region_memory();

    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
