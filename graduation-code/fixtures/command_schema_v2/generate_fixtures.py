"""Deterministic generator for the Command/Descriptor schema v2 fixtures.

Regenerate with::

    python generate_fixtures.py

Every `.bin` is also written as a lowercase-hex `.hex` file; the two are
byte-identical representations of the same fixture.  The Python and C++ codec
tests both consume these files, so the C++ side independently reconstructs the
expected records instead of trusting this generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOFTWARE_ROOT = HERE.parents[1] / "software"
sys.path.insert(0, str(SOFTWARE_ROOT))

from src.command_codec import (  # noqa: E402
    NONE,
    Command,
    Completion,
    DataFormat,
    DataLayout,
    Descriptor,
    DescriptorType,
    Opcode,
    PivotPolicy,
    ScaleDesc,
    StatusCode,
    encode_pivot_threshold,
)

THRESHOLD_RATIO = 2.0**-12


def write(name: str, data: bytes) -> None:
    (HERE / f"{name}.bin").write_bytes(data)
    (HERE / f"{name}.hex").write_text(data.hex() + "\n", encoding="ascii")


def main() -> None:
    write(
        "golden_command",
        Command(
            opcode=Opcode.GEMM_SCHUR,
            flags=0x00000003,
            command_id=0x10203040,
            node_id=0x11223344,
            descriptor_id=5,
            wait_list_id=8,
            signal_token=9,
        ).to_bytes(),
    )

    def front_descriptor(policy: PivotPolicy, threshold: float) -> bytes:
        return Descriptor(
            descriptor_type=DescriptorType.FRONT_DESC,
            payload_offset=0,
            payload_bytes=0,
            body_words=(
                0x00000100,
                8,
                4,
                16,
                NONE,
                0x00000002,
                0x00000003,
                0x00000004,
                int(policy),
                encode_pivot_threshold(threshold),
            ),
        ).to_bytes()

    write("golden_descriptor", front_descriptor(PivotPolicy.STATIC, 0.0))
    write("static_front_descriptor", front_descriptor(PivotPolicy.STATIC, 0.0))
    write(
        "dynamic_front_descriptor",
        front_descriptor(PivotPolicy.DYNAMIC_PANEL, THRESHOLD_RATIO),
    )

    write(
        "golden_dependency_descriptor",
        Descriptor(
            descriptor_type=DescriptorType.DEPENDENCY_DESC,
            payload_offset=0x80,
            payload_bytes=12,
            body_words=(3,) + (0,) * 9,
        ).to_bytes(),
    )

    write(
        "golden_completion",
        Completion(
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
        ).to_bytes(),
    )

    dependency_tokens = bytes.fromhex("020000000400000006000000")
    write("dependency_tokens", dependency_tokens)

    token_states = bytes.fromhex("000000000100000002000000")
    write("token_states", token_states)

    single_node = b"".join(
        [
            Command(
                opcode=Opcode.NODE_BEGIN,
                flags=0x00000001,
                command_id=1,
                node_id=7,
                descriptor_id=0,
                wait_list_id=NONE,
                signal_token=0,
            ).to_bytes(),
            Command(
                opcode=Opcode.NODE_COMMIT,
                flags=0,
                command_id=2,
                node_id=7,
                descriptor_id=NONE,
                wait_list_id=1,
                signal_token=1,
            ).to_bytes(),
        ]
    )
    write("single_node_commands", single_node)

    parent_child = b"".join(
        [
            Command(
                opcode=Opcode.STORE_UPDATE,
                flags=0,
                command_id=0xA,
                node_id=1,
                descriptor_id=3,
                wait_list_id=NONE,
                signal_token=2,
            ).to_bytes(),
            Command(
                opcode=Opcode.ASSEMBLE_EXTEND_ADD,
                flags=0,
                command_id=0xB,
                node_id=2,
                descriptor_id=4,
                wait_list_id=0,
                signal_token=3,
            ).to_bytes(),
            Command(
                opcode=Opcode.NODE_COMMIT,
                flags=0,
                command_id=0xC,
                node_id=2,
                descriptor_id=NONE,
                wait_list_id=1,
                signal_token=4,
            ).to_bytes(),
        ]
    )
    write("parent_child_commands", parent_child)

    # T10 SCALE_DESC fixtures: the default INT32 GEMM policy and a fixed
    # two's-complement exponent policy. Python and C++ decode/validate them
    # identically.
    write(
        "scale_desc_default",
        Descriptor(
            descriptor_type=DescriptorType.SCALE_DESC,
            payload_offset=0,
            payload_bytes=0,
            body_words=ScaleDesc().to_body_words(),
        ).to_bytes(),
    )
    write(
        "scale_desc_fixed_exponents",
        Descriptor(
            descriptor_type=DescriptorType.SCALE_DESC,
            payload_offset=0,
            payload_bytes=0,
            body_words=ScaleDesc(exponent_a=-3, exponent_b=5).to_body_words(),
        ).to_bytes(),
    )

    region = Descriptor(
        descriptor_type=DescriptorType.REGION_DESC,
        payload_offset=0,
        payload_bytes=0,
        body_words=(
            0x00000040,
            0x00000000,
            256,
            0x00000000,
            32,
            8,
            8,
            int(DataFormat.FP32),
            int(DataLayout.ROW_MAJOR),
            0,
        ),
    )
    write("golden_region_descriptor", region.to_bytes())

    # T12 SOLVE_DESC: the payload carries the node's front variable indices
    # (u32[total_dim]); the record itself must declare u32 alignment and a
    # nonzero multiple-of-4 payload. The companion payload fixture holds the
    # three indices of the same example node.
    write(
        "solve_descriptor",
        Descriptor(
            descriptor_type=DescriptorType.SOLVE_DESC,
            payload_offset=0x100,
            payload_bytes=12,
            body_words=(5, 6, 7, 8, 9, 1, 1, int(DataFormat.FP32), 0, 0),
        ).to_bytes(),
    )
    write("solve_payload", bytes.fromhex("000000000400000006000000"))


if __name__ == "__main__":
    main()
