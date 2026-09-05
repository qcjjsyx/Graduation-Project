from dataclasses import replace

import numpy as np
import pytest

from src.command_codec import DataFormat, DescriptorType, Opcode
from src.config import CommandCompilerConfig
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
        )
