import struct
from dataclasses import replace
from pathlib import Path

import pytest

from src.command_codec import (
    ALLOW_RETRY,
    COMMAND_RECORD_BYTES,
    COMPLETION_RECORD_BYTES,
    DESCRIPTOR_RECORD_BYTES,
    NONE,
    TRACE_ENABLE,
    Command,
    Completion,
    DataFormat,
    DataLayout,
    Descriptor,
    DescriptorType,
    KernelBackend,
    Opcode,
    StatusCode,
    SolveDirection,
    TokenState,
    decode_token_states,
    encode_token_states,
    validate_command_batch,
)


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "command_schema_v1"


def fixture(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


def region_descriptor(base_addr: int = 0, byte_size: int = 0) -> Descriptor:
    body = (
        base_addr & 0xFFFFFFFF,
        base_addr >> 32,
        byte_size & 0xFFFFFFFF,
        byte_size >> 32,
        0,
        0,
        0,
        DataFormat.FP32,
        DataLayout.ROW_MAJOR,
        0,
    )
    return Descriptor(DescriptorType.REGION_DESC, body_words=body)


def dependency_descriptor(offset: int, *tokens: int) -> Descriptor:
    return Descriptor(
        DescriptorType.DEPENDENCY_DESC,
        payload_offset=offset,
        payload_bytes=4 * len(tokens),
        body_words=(len(tokens),) + (0,) * 9,
    )


def test_frozen_constants():
    assert NONE == 0xFFFFFFFF
    assert COMMAND_RECORD_BYTES == 32
    assert DESCRIPTOR_RECORD_BYTES == 64
    assert COMPLETION_RECORD_BYTES == 64
    assert TRACE_ENABLE == 1
    assert ALLOW_RETRY == 2
    assert {opcode.value for opcode in Opcode} == set(range(0x01, 0x0E))
    assert {status.value for status in StatusCode} == {
        0x0000,
        0x0001,
        0x0002,
        0x0003,
        0x0004,
        0x0005,
        0x0100,
        0x0101,
        0x0102,
        0x0103,
        0x0104,
        0x0200,
        0x0201,
    }
    assert DataFormat.FP32 == 2
    assert DataLayout.ROW_MAJOR == 1
    assert KernelBackend.SYSTEMC_FP32_DEVICE_MODEL == 1
    assert SolveDirection.BACKWARD == 2


def test_command_golden_and_roundtrip():
    command = Command(
        opcode=Opcode.GEMM_SCHUR,
        flags=TRACE_ENABLE | ALLOW_RETRY,
        command_id=0x10203040,
        node_id=0x11223344,
        descriptor_id=5,
        wait_list_id=8,
        signal_token=9,
    )
    golden = fixture("golden_command.bin")
    assert len(golden) == COMMAND_RECORD_BYTES
    assert command.to_bytes() == golden
    assert Command.from_bytes(golden) == command


@pytest.mark.parametrize("size", [0, 31, 33])
def test_command_rejects_wrong_record_size(size):
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        Command.from_bytes(b"\x00" * size)


def test_command_rejects_unknown_opcode_flags_and_reserved():
    golden = bytearray(fixture("golden_command.bin"))
    golden[0:4] = struct.pack("<I", 0x0E)
    with pytest.raises(ValueError, match="unknown opcode"):
        Command.from_bytes(bytes(golden))

    golden = bytearray(fixture("golden_command.bin"))
    golden[4:8] = struct.pack("<I", 1 << 2)
    with pytest.raises(ValueError, match="reserved bits"):
        Command.from_bytes(bytes(golden))

    golden = bytearray(fixture("golden_command.bin"))
    golden[28:32] = struct.pack("<I", 1)
    with pytest.raises(ValueError, match="arg0 is reserved"):
        Command.from_bytes(bytes(golden))


def test_descriptor_golden_dependency_payload_and_roundtrip():
    descriptor = dependency_descriptor(0x80, 2, 4, 6)
    golden = fixture("golden_descriptor.bin")
    assert len(golden) == DESCRIPTOR_RECORD_BYTES
    assert descriptor.to_bytes() == golden
    assert Descriptor.from_bytes(golden, memory_image_bytes=0x8C) == descriptor

    image = bytearray(0x8C)
    image[0x80:] = fixture("dependency_tokens.bin")
    assert descriptor.dependency_tokens(bytes(image), token_count=7) == (2, 4, 6)


@pytest.mark.parametrize("size", [0, 63, 65])
def test_descriptor_rejects_wrong_record_size(size):
    with pytest.raises(ValueError, match="exactly 64 bytes"):
        Descriptor.from_bytes(b"\x00" * size)


def test_descriptor_rejects_unknown_flags_and_reserved_fields():
    golden = bytearray(fixture("golden_descriptor.bin"))
    golden[0:2] = struct.pack("<H", 9)
    with pytest.raises(ValueError, match="unknown descriptor type"):
        Descriptor.from_bytes(bytes(golden))

    golden = bytearray(fixture("golden_descriptor.bin"))
    golden[2:4] = struct.pack("<H", 1)
    with pytest.raises(ValueError, match="reserved bits"):
        Descriptor.from_bytes(bytes(golden))

    golden = bytearray(fixture("golden_descriptor.bin"))
    golden[4:8] = struct.pack("<I", 1)
    with pytest.raises(ValueError, match="reserved field"):
        Descriptor.from_bytes(bytes(golden))

    golden = bytearray(fixture("golden_descriptor.bin"))
    golden[28:32] = struct.pack("<I", 1)
    with pytest.raises(ValueError, match=r"body_words\[1\].*reserved"):
        Descriptor.from_bytes(bytes(golden))


def test_descriptor_rejects_payload_and_region_address_faults():
    descriptor = dependency_descriptor(0x80, 2, 4, 6)
    with pytest.raises(ValueError, match="exceeds memory image"):
        descriptor.validate(0x8B)
    with pytest.raises(ValueError, match=r"token_count \* 4"):
        replace(descriptor, payload_bytes=8).validate()
    with pytest.raises(ValueError, match="u32 aligned"):
        replace(descriptor, payload_offset=0x81).validate()

    with pytest.raises(ValueError, match="64-byte aligned"):
        region_descriptor(base_addr=1, byte_size=64).validate(128)
    with pytest.raises(ValueError, match="overflows u64"):
        region_descriptor(base_addr=0xFFFFFFFFFFFFFFC0, byte_size=128).validate()
    with pytest.raises(ValueError, match="address range exceeds"):
        region_descriptor(base_addr=64, byte_size=128).validate(128)


def test_dependency_tokens_reject_duplicate_unknown_and_excess_waits():
    descriptor = dependency_descriptor(0, 2, 4, 6)
    with pytest.raises(ValueError, match="duplicates"):
        descriptor.dependency_tokens(struct.pack("<3I", 2, 2, 6), token_count=7)
    with pytest.raises(ValueError, match="unknown token"):
        descriptor.dependency_tokens(struct.pack("<3I", 2, 4, 7), token_count=7)
    with pytest.raises(ValueError, match="max_wait_tokens"):
        descriptor.dependency_tokens(
            fixture("dependency_tokens.bin"), token_count=7, max_wait_tokens=2
        )


def test_completion_golden_and_roundtrip():
    completion = Completion(
        command_id=0x10203040,
        node_id=0x11223344,
        status_code=StatusCode.PRECISION_RETRY,
        pivot_count=3,
        start_cycle=0x0102030405060708,
        finish_cycle=0x1112131415161718,
        read_bytes=0x2122232425262728,
        write_bytes=0x3132333435363738,
        stall_cycles=0x4142434445464748,
        overflow_count=2,
        retry_count=1,
    )
    golden = fixture("golden_completion.bin")
    assert len(golden) == COMPLETION_RECORD_BYTES
    assert completion.to_bytes() == golden
    assert Completion.from_bytes(golden) == completion


def test_completion_rejects_size_status_and_reserved():
    with pytest.raises(ValueError, match="exactly 64 bytes"):
        Completion.from_bytes(fixture("golden_completion.bin")[:-1])

    golden = bytearray(fixture("golden_completion.bin"))
    golden[8:10] = struct.pack("<H", 0x0006)
    with pytest.raises(ValueError, match="unknown status code"):
        Completion.from_bytes(bytes(golden))

    golden = bytearray(fixture("golden_completion.bin"))
    golden[10:12] = struct.pack("<H", 1)
    with pytest.raises(ValueError, match="reserved field"):
        Completion.from_bytes(bytes(golden))


def test_token_state_fixture():
    states = (TokenState.UNSIGNALED, TokenState.READY, TokenState.FAILED)
    golden = fixture("token_states.bin")
    assert encode_token_states(states) == golden
    assert decode_token_states(golden) == states
    with pytest.raises(ValueError, match="multiple of 4"):
        decode_token_states(golden[:-1])
    with pytest.raises(ValueError, match="unknown token state"):
        decode_token_states(struct.pack("<I", 3))


def test_single_node_and_parent_child_static_batch_fixtures():
    single_data = fixture("single_node_commands.bin")
    single = [
        Command.from_bytes(single_data[offset : offset + COMMAND_RECORD_BYTES])
        for offset in range(0, len(single_data), COMMAND_RECORD_BYTES)
    ]
    single_memory = struct.pack("<I", 0)
    single_descriptors = [region_descriptor(), dependency_descriptor(0, 0)]
    validate_command_batch(
        single, single_descriptors, single_memory, token_count=2, max_wait_tokens=4
    )
    assert [command.opcode for command in single] == [
        Opcode.NODE_BEGIN,
        Opcode.NODE_COMMIT,
    ]
    assert single[1].wait_list_id == 1
    assert single_descriptors[1].dependency_tokens(single_memory) == (0,)

    tree_data = fixture("parent_child_commands.bin")
    tree = [
        Command.from_bytes(tree_data[offset : offset + COMMAND_RECORD_BYTES])
        for offset in range(0, len(tree_data), COMMAND_RECORD_BYTES)
    ]
    tree_memory = struct.pack("<II", 2, 3)
    tree_descriptors = [
        dependency_descriptor(0, 2),
        dependency_descriptor(4, 3),
        region_descriptor(),
        region_descriptor(),
        region_descriptor(),
    ]
    validate_command_batch(
        tree, tree_descriptors, tree_memory, token_count=5, max_wait_tokens=4
    )
    assert [command.opcode for command in tree] == [
        Opcode.STORE_UPDATE,
        Opcode.ASSEMBLE_EXTEND_ADD,
        Opcode.NODE_COMMIT,
    ]
    assert tree[1].wait_list_id == 0
    assert tree[2].wait_list_id == 1
    assert tree_descriptors[0].dependency_tokens(tree_memory) == (2,)
    assert tree_descriptors[1].dependency_tokens(tree_memory) == (3,)


def test_batch_rejects_bad_references_and_duplicate_signal():
    descriptors = [region_descriptor()]
    base = Command(Opcode.NODE_BEGIN, 0, 1, descriptor_id=0, signal_token=0)
    validate_command_batch([base], descriptors, b"", token_count=1)

    with pytest.raises(ValueError, match="unknown descriptor_id"):
        validate_command_batch(
            [replace(base, descriptor_id=1)], descriptors, b"", token_count=1
        )
    with pytest.raises(ValueError, match="unknown wait_list_id"):
        validate_command_batch(
            [replace(base, wait_list_id=1)], descriptors, b"", token_count=1
        )
    with pytest.raises(ValueError, match="not a DEPENDENCY_DESC"):
        validate_command_batch(
            [replace(base, wait_list_id=0)], descriptors, b"", token_count=1
        )
    with pytest.raises(ValueError, match="duplicate signal_token"):
        validate_command_batch(
            [base, replace(base, command_id=2)], descriptors, b"", token_count=1
        )
