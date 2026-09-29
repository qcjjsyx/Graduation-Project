from dataclasses import replace

import numpy as np
import pytest

from src.command_codec import (
    NONE,
    DataFormat,
    DescriptorType,
    Opcode,
    PivotPolicy,
    encode_pivot_threshold,
)
from src.config import CommandCompilerConfig, PivotPolicyConfig
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange
from src.scheduler.command_compiler import CommandCompileError, compile_command_artifact


def _tree_fixture(config: CommandCompilerConfig | None = None):
    nodes = [
        NodeCompileRecord(0, 1, NodeRange(0, 1), (0, 1)),
        NodeCompileRecord(1, -1, NodeRange(1, 2), (1,)),
    ]
    maps = [[], [MapTableEntry(0, [0], [0])]]
    fronts = [
        np.array([[4.0, 1.0], [2.0, 0.0]], dtype=np.float64),
        np.array([[5.0]], dtype=np.float64),
    ]
    return compile_command_artifact(
        nodes=nodes,
        map_tables=maps,
        local_fronts=fronts,
        permutation=[0, 1],
        rhs=np.array([1.0, 2.0]),
        config=config or CommandCompilerConfig(),
        pivot=PivotPolicyConfig(),
    )


def test_parent_waits_for_child_update_token_and_all_records_decode():
    artifact = _tree_fixture()
    commands = artifact.commands
    child_store = next(
        command
        for command in commands
        if command.node_id == 0 and command.opcode == Opcode.STORE_UPDATE
    )
    parent_assemble = next(
        command
        for command in commands
        if command.node_id == 1 and command.opcode == Opcode.ASSEMBLE_EXTEND_ADD
    )
    waits = artifact.descriptors[parent_assemble.wait_list_id].dependency_tokens(
        artifact.image, token_count=len(commands)
    )
    assert child_store.signal_token in waits
    assert parent_assemble.command_id > child_store.command_id
    assert [command.command_id for command in commands] == list(range(len(commands)))
    assert [command.signal_token for command in commands] == list(range(len(commands)))
    assert len(artifact.completion_templates) == len(commands)


def test_compiler_uses_fp32_data_and_is_byte_deterministic():
    first = _tree_fixture()
    second = _tree_fixture()
    assert first.image == second.image
    assert first.manifest == second.manifest
    formats = {
        DataFormat(descriptor.body_words[7])
        for descriptor in first.descriptors
        if descriptor.descriptor_type == DescriptorType.REGION_DESC
    }
    assert formats == {DataFormat.FP32, DataFormat.INT32}
    assert all(
        descriptor.descriptor_type != DescriptorType.SCALE_DESC
        for descriptor in first.descriptors
    )


def test_single_node_skips_zero_dimension_update_kernels():
    artifact = compile_command_artifact(
        nodes=[NodeCompileRecord(0, -1, NodeRange(0, 1), (0,))],
        map_tables=[[]],
        local_fronts=[np.array([[2.0]])],
        permutation=[0],
        rhs=np.array([3.0]),
        config=CommandCompilerConfig(),
        pivot=PivotPolicyConfig(),
    )
    opcodes = [command.opcode for command in artifact.commands]
    assert Opcode.TRSM_LEFT not in opcodes
    assert Opcode.TRSM_RIGHT not in opcodes
    assert Opcode.GEMM_SCHUR not in opcodes
    assert Opcode.STORE_UPDATE not in opcodes
    assert opcodes[-2:] == [Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD]


def test_compiler_rejects_duplicate_ids_cycle_and_front_limit():
    base = [
        NodeCompileRecord(0, 1, NodeRange(0, 1), (0, 1)),
        NodeCompileRecord(1, -1, NodeRange(1, 2), (1,)),
    ]
    kwargs = dict(
        map_tables=[[], [MapTableEntry(0, [0], [0])]],
        local_fronts=[np.eye(2), np.eye(1)],
        permutation=[0, 1],
        rhs=np.ones(2),
        config=CommandCompilerConfig(),
        pivot=PivotPolicyConfig(),
    )
    with pytest.raises(CommandCompileError, match="duplicate node ID"):
        compile_command_artifact(nodes=[base[0], replace(base[1], node_id=0)], **kwargs)
    with pytest.raises(CommandCompileError, match="cycle"):
        compile_command_artifact(
            nodes=[replace(base[0], parent_id=1), replace(base[1], parent_id=0)],
            **kwargs,
        )
    with pytest.raises(CommandCompileError, match="exceeds limit"):
        _tree_fixture(CommandCompilerConfig(max_front_size=1))


def test_compiler_rejects_empty_rhs():
    with pytest.raises(CommandCompileError, match="RHS must not be empty"):
        compile_command_artifact(
            nodes=[NodeCompileRecord(0, -1, NodeRange(0, 1), (0,))],
            map_tables=[[]],
            local_fronts=[np.eye(1)],
            permutation=[0],
            rhs=np.array([]),
            config=CommandCompilerConfig(),
        pivot=PivotPolicyConfig(),
        )


def test_panel_lu_uses_front_descriptor_with_explicit_policy():
    artifact = _tree_fixture()
    panel_commands = [
        command for command in artifact.commands if command.opcode == Opcode.PANEL_LU
    ]
    assert panel_commands
    for command in panel_commands:
        descriptor = artifact.descriptors[command.descriptor_id]
        assert descriptor.descriptor_type == DescriptorType.FRONT_DESC
        assert descriptor.body_words[8] == PivotPolicy.STATIC
        assert descriptor.body_words[9] == encode_pivot_threshold(0.0)


def test_dynamic_panel_policy_reaches_every_front_descriptor():
    artifact = compile_command_artifact(
        nodes=[
            NodeCompileRecord(0, 1, NodeRange(0, 1), (0, 1)),
            NodeCompileRecord(1, -1, NodeRange(1, 2), (1,)),
        ],
        map_tables=[[], [MapTableEntry(0, [0], [0])]],
        local_fronts=[
            np.array([[4.0, 1.0], [2.0, 0.0]], dtype=np.float64),
            np.array([[5.0]], dtype=np.float64),
        ],
        permutation=[0, 1],
        rhs=np.array([1.0, 2.0]),
        config=CommandCompilerConfig(),
        pivot=PivotPolicyConfig(policy="DYNAMIC_PANEL", threshold_ratio=0.25),
    )
    fronts = [
        descriptor
        for descriptor in artifact.descriptors
        if descriptor.descriptor_type == DescriptorType.FRONT_DESC
    ]
    assert fronts
    for descriptor in fronts:
        assert descriptor.body_words[8] == PivotPolicy.DYNAMIC_PANEL
        assert descriptor.body_words[9] == encode_pivot_threshold(0.25)
    assert artifact.manifest["pivot"]["policy"] == "DYNAMIC_PANEL"
    assert artifact.manifest["required_device_capabilities"]["DYNAMIC_PANEL"][
        "hpu_atu_dynamic_mapping"
    ] == "required"


def test_p_vector_regions_hold_the_unset_sentinel():
    import struct

    artifact = _tree_fixture()
    for name, region in artifact.regions.items():
        if not name.endswith(".p_vector"):
            continue
        data = artifact.image[region.offset : region.offset + region.size]
        words = struct.unpack(f"<{region.size // 4}i", data)
        assert set(words) == {-1}
        assert 0 not in words
    assert artifact.manifest["pivot"]["p_vector"]["initial_value"] == -1
    assert artifact.manifest["pivot"]["p_vector"]["omitted"] is False
    assert artifact.manifest["pivot"]["p_vector"]["none_reference_allowed"] is False
    assert NONE not in {
        descriptor.body_words[6]
        for descriptor in artifact.descriptors
        if descriptor.descriptor_type == DescriptorType.FRONT_DESC
    }


def _two_root_fixture():
    nodes = [
        NodeCompileRecord(0, -1, NodeRange(0, 1), (0, 1)),
        NodeCompileRecord(1, -1, NodeRange(1, 2), (1, 2)),
    ]
    maps: list[list[MapTableEntry]] = [[], []]
    fronts = [
        np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        np.array([[5.0, 0.5], [0.5, 6.0]], dtype=np.float64),
    ]
    return compile_command_artifact(
        nodes=nodes,
        map_tables=maps,
        local_fronts=fronts,
        permutation=[0, 1, 2],
        rhs=np.array([1.0, 2.0, 3.0]),
        config=CommandCompilerConfig(),
        pivot=PivotPolicyConfig(),
    )


def test_independent_fronts_interleave_round_robin():
    artifact = _two_root_fixture()
    body = [c for c in artifact.commands if c.opcode != Opcode.SOLVE_FORWARD and c.opcode != Opcode.SOLVE_BACKWARD]
    owners = [c.node_id for c in body]
    alternations = sum(1 for a, b in zip(owners, owners[1:]) if a != b)
    # Two ready nodes take turns: consecutive commands of the block phase keep
    # switching between the two independent fronts.
    assert alternations == len(owners) - 1
    assert owners[0] == 0 and owners[1] == 1


def test_node_begin_waits_on_child_completion_token():
    artifact = _tree_fixture()
    commands = artifact.commands
    child_commit = next(
        c for c in commands if c.node_id == 0 and c.opcode == Opcode.NODE_COMMIT
    )
    parent_begin = next(
        c for c in commands if c.node_id == 1 and c.opcode == Opcode.NODE_BEGIN
    )
    waits = artifact.descriptors[parent_begin.wait_list_id].dependency_tokens(
        artifact.image, token_count=len(commands)
    )
    assert child_commit.signal_token in waits
    assert parent_begin.command_id > child_commit.command_id


def test_interleaved_stream_keeps_order_invariants():
    artifact = _two_root_fixture()
    commands = artifact.commands
    assert [c.command_id for c in commands] == list(range(len(commands)))
    assert [c.signal_token for c in commands] == list(range(len(commands)))
    for command in commands:
        if command.wait_list_id == 0xFFFFFFFF:
            continue
        waits = artifact.descriptors[command.wait_list_id].dependency_tokens(
            artifact.image, token_count=len(commands)
        )
        assert all(token < command.command_id for token in waits)
    # Solve phase stays trailing (forward chain then backward chain).
    first_solve = min(
        i for i, c in enumerate(commands) if c.opcode in (Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD)
    )
    assert all(
        c.opcode in (Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD)
        for c in commands[first_solve:]
    )
    assert commands[-1].opcode == Opcode.SOLVE_BACKWARD
