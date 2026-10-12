"""T12 compiler + manifest solve contract.

Covers the SOLVE_DESC payload emission, the topological (factorization-order)
serial solve chain, the compile-time solve scratch check, and the
manifest-validator negatives for all of the above.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.command_codec import (
    NONE,
    SOLVE_DESC_PAYLOAD_KIND,
    DescriptorType,
    Opcode,
    decode_commands,
    encode_commands,
)
from src.config import CommandCompilerConfig, PivotPolicyConfig
from src.dataStruct import MapTableEntry, NodeCompileRecord, NodeRange
from src.scheduler.command_compiler import (
    CommandCompileError,
    compile_command_artifact,
    solve_scratch_bytes,
)
from src.verify.manifest import (
    ManifestValidationError,
    _validate_solve_contract,
    validate_manifest,
)


def _reversed_tree_fixture(config: CommandCompilerConfig | None = None):
    """node 0 = root, node 1 = its child -> factorization order [1, 0].

    ``parent[node]`` points at the node that eliminates the child's update, so
    node 1 must solve before node 0 even though its node id is larger.
    """

    nodes = [
        NodeCompileRecord(0, -1, NodeRange(1, 2), (1,)),
        NodeCompileRecord(1, 0, NodeRange(0, 1), (0, 1)),
    ]
    maps = [[MapTableEntry(1, [0], [0])], []]
    fronts = [np.array([[5.0]]), np.array([[4.0, 1.0], [2.0, 0.0]])]
    return compile_command_artifact(
        nodes=nodes,
        map_tables=maps,
        local_fronts=fronts,
        permutation=[0, 1],
        rhs=np.array([1.0, 3.0]),
        config=config or CommandCompilerConfig(),
        pivot=PivotPolicyConfig(),
    )


def _write_artifact(tmp_path, artifact, image: bytes | None = None) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(artifact.manifest), encoding="utf-8"
    )
    (tmp_path / "memory_image.bin").write_bytes(
        artifact.image if image is None else image
    )
    # The production validator expects the preprocess block (the CLI pipeline
    # always writes it; these hand-built IRs use identity ordering).
    manifest = artifact.manifest
    manifest["preprocess"] = {
        "matching": {"method": "none"},
        "transform": {
            "row_perm_match": [0, 1],
            "ordering_perm": [0, 1],
            "rhs_row_map": [0, 1],
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_solve_descriptors_carry_front_indices_payload():
    artifact = _reversed_tree_fixture()
    nodes = artifact.manifest["nodes"]
    seen = set()
    for command in artifact.commands:
        if command.opcode not in (Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD):
            continue
        seen.add(command.node_id)
        descriptor = artifact.descriptors[command.descriptor_id]
        assert descriptor.descriptor_type == DescriptorType.SOLVE_DESC
        assert descriptor.payload_offset % 4 == 0
        factor = artifact.descriptors[descriptor.body_words[0]]
        total = factor.body_words[3]
        assert descriptor.payload_bytes == 4 * total
        assert list(descriptor.solve_variables(artifact.image)) == list(
            nodes[str(command.node_id)]["front_indices"]
        )
    assert seen == {0, 1}
    assert artifact.manifest["abi"]["solve_desc_payload"] == SOLVE_DESC_PAYLOAD_KIND
    solve = artifact.manifest["solve"]
    assert solve["chain"] == "serial"
    assert solve["forward_order"] == "factorization_order"
    assert solve["backward_order"] == "reverse_factorization_order"
    assert solve["rhs_read_only"] is True
    assert solve["workspace_rows"] == ["forward_z", "backward_scratch"]
    assert solve["solution_semantics"] == "zero-initialized accumulator; final = x_hat"
    assert solve["row_buffer_depth"] == 2
    assert solve["max_scratch_bytes"] == max(
        solve_scratch_bytes(2, 1), solve_scratch_bytes(1, 1)
    )


def test_solve_chain_follows_factorization_order():
    artifact = _reversed_tree_fixture()
    assert artifact.manifest["symbolic"]["factorization_order"] == [1, 0]
    forwards = sorted(
        (command.command_id, command.node_id)
        for command in artifact.commands
        if command.opcode == Opcode.SOLVE_FORWARD
    )
    backwards = sorted(
        (command.command_id, command.node_id)
        for command in artifact.commands
        if command.opcode == Opcode.SOLVE_BACKWARD
    )
    assert [node for _, node in forwards] == [1, 0]
    assert [node for _, node in backwards] == [0, 1]
    # The chain is serial and anchored on each node's own commit token.  There
    # is no global solve gate (revised 2026-10-11): every forward waits on
    # exactly {own commit} (first) or {own commit, previous forward}, so wait
    # lists stay bounded by 2 regardless of the node count.
    commit_token = {
        command.node_id: command.signal_token
        for command in artifact.commands
        if command.opcode == Opcode.NODE_COMMIT
    }
    previous = None
    for command_id, node in forwards:
        waits = artifact.descriptors[
            artifact.commands[command_id].wait_list_id
        ].dependency_tokens(artifact.image, token_count=len(artifact.commands))
        assert len(waits) <= 2
        expected = {commit_token[node]}
        if previous is not None:
            expected.add(previous)
        assert sorted(waits) == sorted(expected)
        previous = artifact.commands[command_id].signal_token
    for command_id, node in backwards:
        waits = artifact.descriptors[
            artifact.commands[command_id].wait_list_id
        ].dependency_tokens(artifact.image, token_count=len(artifact.commands))
        assert len(waits) <= 2
        assert commit_token[node] in waits
        assert previous in waits
        previous = artifact.commands[command_id].signal_token


def test_solve_scratch_capacity_is_checked_at_compile_time():
    # front 2x2 FP32 = 16 B, solve scratch = 4*(2*2 + 2*2 + 1) = 36 B.
    assert solve_scratch_bytes(2, 1) == 36
    with pytest.raises(CommandCompileError, match="solve scratch"):
        _reversed_tree_fixture(CommandCompilerConfig(device_buffer_bytes=32))
    # The same slot without solve commands only needs to hold the front.
    artifact = _reversed_tree_fixture(
        CommandCompilerConfig(device_buffer_bytes=32, emit_solve_commands=False)
    )
    assert not any(
        command.opcode in (Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD)
        for command in artifact.commands
    )


def test_manifest_validates_solve_artifact_and_rejects_mutations(tmp_path):
    artifact = _reversed_tree_fixture()
    _write_artifact(tmp_path, artifact)
    assert validate_manifest(tmp_path / "manifest.json").command_count == len(
        artifact.commands
    )

    def mutated(manifest_mutation=None, image_mutation=None):
        manifest = json.loads(json.dumps(artifact.manifest))
        manifest["preprocess"] = {
            "matching": {"method": "none"},
            "transform": {
                "row_perm_match": [0, 1],
                "ordering_perm": [0, 1],
                "rhs_row_map": [0, 1],
            },
        }
        if manifest_mutation is not None:
            manifest_mutation(manifest)
        image = bytearray(artifact.image)
        if image_mutation is not None:
            image_mutation(image)
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "memory_image.bin").write_bytes(bytes(image))

    # Missing abi marker.
    mutated(lambda manifest: manifest["abi"].pop("solve_desc_payload"))
    with pytest.raises(ManifestValidationError, match="solve_desc_payload"):
        validate_manifest(tmp_path / "manifest.json")

    # Missing solve block.
    mutated(lambda manifest: manifest.pop("solve"))
    with pytest.raises(ManifestValidationError, match="solve"):
        validate_manifest(tmp_path / "manifest.json")

    # Out-of-range payload variable (node 1 payload [0, 1] -> [0, 9]).
    solve_descriptor = next(
        descriptor
        for descriptor in artifact.descriptors
        if descriptor.descriptor_type == DescriptorType.SOLVE_DESC
        and descriptor.payload_bytes == 8
    )
    payload_offset = solve_descriptor.payload_offset

    def corrupt_second_variable(image: bytearray) -> None:
        image[payload_offset + 4 : payload_offset + 8] = (9).to_bytes(4, "little")

    mutated(image_mutation=corrupt_second_variable)
    with pytest.raises(ManifestValidationError, match="out-of-range|front_indices"):
        validate_manifest(tmp_path / "manifest.json")

    # Payload that no longer matches the declared front_indices (valid
    # permutation but wrong variable ownership).
    def swap_variables(image: bytearray) -> None:
        image[payload_offset : payload_offset + 4] = (1).to_bytes(4, "little")
        image[payload_offset + 4 : payload_offset + 8] = (0).to_bytes(4, "little")

    mutated(image_mutation=swap_variables)
    with pytest.raises(ManifestValidationError):
        validate_manifest(tmp_path / "manifest.json")


def test_solve_contract_rejects_non_topological_or_unsynchronized_chain():
    from dataclasses import replace

    artifact = _reversed_tree_fixture()
    commands = list(decode_commands(
        artifact.image[
            artifact.regions["command_buffer"].offset :
            artifact.regions["command_buffer"].offset
            + artifact.regions["command_buffer"].size
        ]
    ))
    forwards = [
        index for index, command in enumerate(commands)
        if command.opcode == Opcode.SOLVE_FORWARD
    ]
    assert len(forwards) == 2
    first, second = forwards
    swapped = list(commands)
    swapped[first] = replace(
        commands[first], node_id=commands[second].node_id
    )
    swapped[second] = replace(
        commands[second], node_id=commands[first].node_id
    )
    with pytest.raises(ManifestValidationError, match="factorization \\(topological\\) order"):
        _validate_solve_contract(
            artifact.manifest, swapped, artifact.descriptors, artifact.image
        )

    # Removing the wait list of the second forward command must be refused.
    unsynchronized = list(commands)
    unsynchronized[second] = replace(commands[second], wait_list_id=NONE)
    with pytest.raises(ManifestValidationError, match="wait"):
        _validate_solve_contract(
            artifact.manifest, unsynchronized, artifact.descriptors, artifact.image
        )

    # Dropping the first forward's own commit token must be refused.  (Before
    # the 2026-10-11 revision this block checked a global gate over every
    # commit; that gate is gone, the own-commit anchor is what remains.)
    first_command = commands[first]
    gate = artifact.descriptors[first_command.wait_list_id].dependency_tokens(
        artifact.image, token_count=len(commands)
    )
    first_commit = next(
        command.signal_token
        for command in commands
        if command.opcode == Opcode.NODE_COMMIT
        and command.node_id == first_command.node_id
    )
    assert list(gate) == [first_commit]  # own commit only, no predecessor yet
    other_commit = next(
        command.signal_token
        for command in commands
        if command.opcode == Opcode.NODE_COMMIT
        and command.node_id == commands[second].node_id
    )
    reduced_gate = [other_commit]
    reduced_descriptor = replace(
        artifact.descriptors[first_command.wait_list_id],
        payload_bytes=4 * len(reduced_gate),
        body_words=(
            len(reduced_gate),
            *artifact.descriptors[first_command.wait_list_id].body_words[1:],
        ),
    )
    reduced_image = bytearray(artifact.image)
    offset = reduced_descriptor.payload_offset
    reduced_image[offset : offset + 4 * len(gate)] = b"".join(
        token.to_bytes(4, "little") for token in reduced_gate
    )
    descriptors = list(artifact.descriptors)
    descriptors[first_command.wait_list_id] = reduced_descriptor
    with pytest.raises(ManifestValidationError, match="must wait for its own commit"):
        _validate_solve_contract(
            artifact.manifest, commands, descriptors, bytes(reduced_image)
        )


def test_more_nodes_than_default_max_wait_tokens_compiles(tmp_path):
    """Regression (2026-10-11): the solve chain no longer has a global gate,
    so a matrix with more nodes than the default max_wait_tokens (256) must
    compile with the default configuration and every wait list stays small.
    n=400, density=0.005, seed=0 gave 322 nodes when the gate was removed.
    """
    from src.command_codec import decode_descriptors
    from src.config import MatrixInputConfig, PipelineConfig
    from src.pipeline import run_pipeline

    outputs = run_pipeline(
        PipelineConfig(
            matrix=MatrixInputConfig(path=None, n=400, density=0.005, seed=0),
            out_dir=tmp_path,
        )
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert outputs.node_count > 256
    assert manifest["compiler"]["operation_scope"] == "factorization_and_solve"
    max_wait = manifest["compiler"]["max_wait_tokens"]
    assert max_wait == 256
    assert validate_manifest(outputs.manifest_path).node_count == outputs.node_count

    image = outputs.memory_image_path.read_bytes()
    regions = manifest["memory_image"]["regions"]

    def region(name: str) -> bytes:
        offset, size = regions[name]["offset"], regions[name]["size"]
        return image[offset : offset + size]

    commands = decode_commands(region("command_buffer"))
    descriptors = decode_descriptors(
        region("descriptor_table"), memory_image_bytes=len(image)
    )
    solve_opcodes = (Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD)
    assert sum(c.opcode in solve_opcodes for c in commands) == 2 * outputs.node_count
    for command in commands:
        if command.wait_list_id == NONE:
            continue
        waits = descriptors[command.wait_list_id].dependency_tokens(
            image, token_count=len(commands), max_wait_tokens=max_wait
        )
        assert len(waits) <= max_wait
        if command.opcode in solve_opcodes:
            assert len(waits) <= 2
