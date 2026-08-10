#!/usr/bin/env python3
"""Generate a deterministic ABI-v2 artifact and run the SystemC model."""

from __future__ import annotations

import json
import shutil
import subprocess
import struct
import sys
import tempfile
from pathlib import Path


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_simulator(
    simulator: Path,
    artifact: Path,
    config: Path,
    result: Path,
    *,
    mode: str = "both",
    seed: int = 1,
) -> dict:
    run(
        [
            str(simulator),
            "--artifact",
            str(artifact / "manifest.json"),
            "--config",
            str(config),
            "--mode",
            mode,
            "--out",
            str(result),
            "--seed",
            str(seed),
        ]
    )
    return json.loads((result / "summary.json").read_text())


def expect_rejected(
    simulator: Path,
    artifact: Path,
    config: Path,
    result: Path,
) -> None:
    process = subprocess.run(
        [
            str(simulator),
            "--artifact",
            str(artifact / "manifest.json"),
            "--config",
            str(config),
            "--mode",
            "fp64",
            "--out",
            str(result),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.returncode != 0


def expect_runtime_failure(
    simulator: Path,
    artifact: Path,
    config: Path,
    result: Path,
    expected_status: str,
    *,
    mode: str = "fixed",
) -> None:
    process = subprocess.run(
        [
            str(simulator),
            "--artifact",
            str(artifact / "manifest.json"),
            "--config",
            str(config),
            "--mode",
            mode,
            "--out",
            str(result),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.returncode == 2
    summary = json.loads((result / "summary.json").read_text())
    assert summary["status"] == expected_status
    assert not summary["timed_out"]


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: run_e2e.py <system_sim> <software_dir> <config.json>"
        )
    simulator = Path(sys.argv[1]).resolve()
    software_dir = Path(sys.argv[2]).resolve()
    config = Path(sys.argv[3]).resolve()
    with tempfile.TemporaryDirectory(prefix="systemc-e2e-") as directory:
        root = Path(directory)
        artifact = root / "artifact"
        result = root / "result"
        run(
            [
                sys.executable,
                "-m",
                "src.main",
                "--out",
                str(artifact),
                "--n",
                "16",
                "--density",
                "0.2",
                "--seed",
                "3",
                "--rhs-seed",
                "11",
                "--ordering",
                "amd",
            ],
            cwd=software_dir,
        )
        summary = run_simulator(simulator, artifact, config, result)
        assert summary["status"] == "ok"
        assert summary["completed_nodes"] == summary["node_count"]
        assert summary["equilibration_mode"] == "pow2-row-column"
        assert summary["solve"]["fp64"]["relative_residual"] <= 1e-10
        assert summary["solve"]["fixed"]["relative_residual"] <= 1e-3
        assert (
            summary["solve"]["fixed"]["relative_residual"]
            == summary["solve"]["fixed"]["residual_history"][-1]
        )
        assert summary["solve"]["fixed"]["componentwise_backward_error"] >= 0
        assert summary["config"]["workspace_guard_bits"] > 0
        assert summary["config"]["fixed_rescue_mode"] == "fp64"
        for filename in (
            "nodes.csv",
            "operations.csv",
            "memory.csv",
            "timeline.csv",
            "solution.csv",
            "final_memory_image.bin",
        ):
            assert (result / filename).stat().st_size > 0

        base_config = json.loads(config.read_text())
        pressure_config = root / "pressure.json"
        pressure_values = dict(base_config)
        pressure_values.update(
            backpressure_probability=0.25,
            ddr_jitter_cycles=5,
        )
        pressure_config.write_text(json.dumps(pressure_values))
        reference_pressure = None
        for seed in range(20):
            pressure_summary = run_simulator(
                simulator,
                artifact,
                pressure_config,
                root / f"pressure-{seed}",
                seed=seed,
            )
            assert pressure_summary["status"] == "ok"
            assert pressure_summary["completed_nodes"] == pressure_summary["node_count"]
            if seed == 7:
                reference_pressure = pressure_summary
        repeated = run_simulator(
            simulator,
            artifact,
            pressure_config,
            root / "pressure-repeat",
            seed=7,
        )
        assert repeated == reference_pressure

        low_bandwidth = root / "low-bandwidth.json"
        high_bandwidth = root / "high-bandwidth.json"
        low_values = dict(base_config, ddr_bytes_per_cycle=8)
        high_values = dict(base_config, ddr_bytes_per_cycle=64)
        low_bandwidth.write_text(json.dumps(low_values))
        high_bandwidth.write_text(json.dumps(high_values))
        low_summary = run_simulator(
            simulator, artifact, low_bandwidth, root / "low", mode="fp64"
        )
        high_summary = run_simulator(
            simulator, artifact, high_bandwidth, root / "high", mode="fp64"
        )
        baseline_fp64 = run_simulator(
            simulator, artifact, config, root / "baseline-fp64", mode="fp64"
        )
        assert low_summary["memory"]["read_cycles"] >= high_summary["memory"]["read_cycles"]
        assert low_summary["cycles"]["total"] >= high_summary["cycles"]["total"]

        slow_kernel = root / "slow-kernel.json"
        slow_values = dict(
            base_config,
            panel_startup=base_config["panel_startup"] + 20,
            trsm_startup=base_config["trsm_startup"] + 20,
            gemm_startup=base_config["gemm_startup"] + 20,
        )
        slow_kernel.write_text(json.dumps(slow_values))
        slow_summary = run_simulator(
            simulator, artifact, slow_kernel, root / "slow", mode="fp64"
        )
        assert (
            slow_summary["cycles"]["total"]
            >= baseline_fp64["cycles"]["total"]
        )

        bad_version = root / "bad-version"
        shutil.copytree(artifact, bad_version)
        bad_manifest = json.loads((bad_version / "manifest.json").read_text())
        bad_manifest["abi"]["version"] = 1
        (bad_version / "manifest.json").write_text(json.dumps(bad_manifest))
        expect_rejected(
            simulator, bad_version, config, root / "bad-version-result"
        )

        truncated = root / "truncated"
        shutil.copytree(artifact, truncated)
        memory_path = truncated / "memory_image.bin"
        memory_path.write_bytes(memory_path.read_bytes()[:-1])
        expect_rejected(
            simulator, truncated, config, root / "truncated-result"
        )

        overlap = root / "overlap"
        shutil.copytree(artifact, overlap)
        overlap_manifest = json.loads((overlap / "manifest.json").read_text())
        overlap_manifest["nodes"]["0"]["front_q"]["offset"] = 0
        (overlap / "manifest.json").write_text(json.dumps(overlap_manifest))
        expect_rejected(simulator, overlap, config, root / "overlap-result")

        bad_unscale = root / "bad-unscale"
        shutil.copytree(artifact, bad_unscale)
        unscale_manifest = json.loads(
            (bad_unscale / "manifest.json").read_text()
        )
        unscale_manifest["equilibration"][
            "solution_requires_unscale"
        ] = False
        (bad_unscale / "manifest.json").write_text(
            json.dumps(unscale_manifest)
        )
        expect_rejected(
            simulator,
            bad_unscale,
            config,
            root / "bad-unscale-result",
        )

        truncated_column_scale = root / "truncated-column-scale"
        shutil.copytree(artifact, truncated_column_scale)
        column_scale_path = (
            truncated_column_scale / "column_scale_e.bin"
        )
        column_scale_path.write_bytes(
            column_scale_path.read_bytes()[:-2]
        )
        expect_rejected(
            simulator,
            truncated_column_scale,
            config,
            root / "truncated-column-scale-result",
        )

        corrupt_map = root / "corrupt-map"
        shutil.copytree(artifact, corrupt_map)
        corrupt_manifest = json.loads(
            (corrupt_map / "manifest.json").read_text()
        )
        parent_node = next(
            node
            for node in corrupt_manifest["nodes"].values()
            if node["map_table"]["size"] > 4
        )
        image_path = corrupt_map / "memory_image.bin"
        image = bytearray(image_path.read_bytes())
        offset = parent_node["map_table"]["offset"]
        image[offset : offset + 4] = struct.pack("<I", 0xFFFFFFFF)
        image_path.write_bytes(image)
        expect_rejected(
            simulator, corrupt_map, config, root / "corrupt-map-result"
        )

        tiny_buffer = root / "tiny-buffer.json"
        tiny_values = dict(base_config, buffer_capacity_bytes=1)
        tiny_buffer.write_text(json.dumps(tiny_values))
        expect_runtime_failure(
            simulator,
            artifact,
            tiny_buffer,
            root / "tiny-buffer-result",
            "address_failure",
            mode="fp64",
        )

        zero_pivot = root / "zero-pivot"
        shutil.copytree(artifact, zero_pivot)
        zero_manifest = json.loads((zero_pivot / "manifest.json").read_text())
        parent = zero_manifest["symbolic"]["parent"]
        leaf = next(
            node_id
            for node_id in zero_manifest["task_order"]
            if all(owner != node_id for owner in parent)
        )
        front = zero_manifest["nodes"][str(leaf)]["front_q"]
        zero_image_path = zero_pivot / "memory_image.bin"
        zero_image = bytearray(zero_image_path.read_bytes())
        zero_image[
            front["offset"] : front["offset"] + front["size"]
        ] = bytes(front["size"])
        zero_image_path.write_bytes(zero_image)
        expect_runtime_failure(
            simulator,
            zero_pivot,
            config,
            root / "zero-pivot-result",
            "numeric_failure",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
