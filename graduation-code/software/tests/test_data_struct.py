from pathlib import Path

from src.dataStruct import NODE_TASK_BYTE_SIZE, NODE_TASK_PACKED_SIZE, ROOT_PARENT_ID, NodeTask
from src.io import read_tasks, write_tasks


def test_node_task_abi_roundtrip():
    task = NodeTask(
        node_id=7,
        flags=3,
        parent_id=ROOT_PARENT_ID,
        children_count=2,
        total_dim=33,
        pivot_dim=17,
        nums_sub_matrix=2,
        last_sub_matrix_size=1,
        data_addr=64,
        parent_address=128,
        map_table_addr=192,
        l_factor_addr=256,
        u_factor_addr=320,
        p_vector_addr=384,
        reserved=9,
    )

    encoded = task.to_bytes()
    assert NODE_TASK_PACKED_SIZE == 74
    assert NODE_TASK_BYTE_SIZE == 80
    assert len(encoded) == NodeTask.BYTE_SIZE
    assert NodeTask.from_bytes(encoded) == task


def test_node_task_decode_rejects_short_buffer():
    try:
        NodeTask.from_bytes(b"\x00" * 8)
    except ValueError as exc:
        assert "expected at least" in str(exc)
    else:
        raise AssertionError("short NodeTask buffer should be rejected")


def test_node_task_file_decode(tmp_path: Path):
    tasks = [
        NodeTask(
            node_id=0,
            flags=1,
            parent_id=1,
            children_count=0,
            total_dim=3,
            pivot_dim=1,
            nums_sub_matrix=1,
            last_sub_matrix_size=1,
            data_addr=64,
            parent_address=128,
            map_table_addr=192,
            l_factor_addr=0,
            u_factor_addr=0,
            p_vector_addr=0,
            reserved=0,
        ),
        NodeTask(
            node_id=1,
            flags=2,
            parent_id=ROOT_PARENT_ID,
            children_count=1,
            total_dim=2,
            pivot_dim=2,
            nums_sub_matrix=1,
            last_sub_matrix_size=2,
            data_addr=256,
            parent_address=0,
            map_table_addr=320,
            l_factor_addr=0,
            u_factor_addr=0,
            p_vector_addr=0,
            reserved=0,
        ),
    ]
    path = tmp_path / "tasks.bin"
    write_tasks(str(path), tasks)
    assert path.stat().st_size == len(tasks) * NODE_TASK_BYTE_SIZE
    assert read_tasks(str(path)) == tasks
