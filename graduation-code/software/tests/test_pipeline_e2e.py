import json
from pathlib import Path

from src.config import MatrixInputConfig, OrderingConfig, PipelineConfig
from src.dataStruct import NODE_TASK_BYTE_SIZE
from src.pipeline import run_pipeline
from src.verify.manifest import validate_manifest


def test_pipeline_outputs_are_manifest_consistent(tmp_path: Path):
    config = PipelineConfig(
        matrix=MatrixInputConfig(path=None, n=12, density=0.25, seed=4),
        ordering=OrderingConfig(method="amd"),
        out_dir=tmp_path,
    )

    outputs = run_pipeline(config)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    validation = validate_manifest(outputs.manifest_path)

    assert outputs.tasks_path.exists()
    assert outputs.map_table_path.exists()
    assert outputs.front_q_path.exists()
    assert outputs.front_e_path.exists()
    assert outputs.task_count == outputs.node_count
    assert validation.node_count == outputs.node_count
    assert validation.task_count == outputs.task_count
    assert outputs.tasks_path.stat().st_size == outputs.task_count * NODE_TASK_BYTE_SIZE

    assert manifest["abi"]["node_task_byte_size"] == NODE_TASK_BYTE_SIZE
    assert manifest["symbolic"]["node_count"] == outputs.node_count
    assert manifest["quantization"]["format"] == "S_format_local_contribution"
    assert manifest["quantization"]["exponent_dtype"] == "int16"
    assert manifest["output_sizes"]["tasks.bin"] == outputs.tasks_path.stat().st_size
    assert manifest["output_sizes"]["front_q.bin"] == outputs.front_q_path.stat().st_size
    assert manifest["output_sizes"]["front_e.bin"] == outputs.front_e_path.stat().st_size
    assert manifest["output_sizes"]["map_table.bin"] == outputs.map_table_path.stat().st_size

    for node_id, node in manifest["nodes"].items():
        front_q = node["front_q"]
        front_e = node["front_e"]
        map_table = node["map_table"]
        local_source = node["local_source"]
        front_dim = len(node["front_indices"])
        assert front_q["offset"] % config.memory.alignment == 0
        assert front_e["offset"] % config.memory.alignment == 0
        assert map_table["offset"] % config.memory.alignment == 0
        assert node["update_q"]["offset"] % config.memory.alignment == 0
        assert node["l_factor"]["offset"] % config.memory.alignment == 0
        assert node["u_factor"]["offset"] % config.memory.alignment == 0
        assert node["front_q_file_offset"] + front_q["size"] <= outputs.front_q_path.stat().st_size
        assert node["front_e_file_offset"] + front_e["size"] <= outputs.front_e_path.stat().st_size
        assert node["map_table_file_offset"] + map_table["size"] <= outputs.map_table_path.stat().st_size
        assert local_source["shape"] == [front_dim, front_dim]
        assert front_q["size"] == front_dim * front_dim * 4
        assert front_e["size"] == 2

    positions = {node_id: idx for idx, node_id in enumerate(manifest["task_order"])}
    for child, parent in enumerate(manifest["symbolic"]["parent"]):
        if parent >= 0:
            assert positions[child] < positions[parent]
