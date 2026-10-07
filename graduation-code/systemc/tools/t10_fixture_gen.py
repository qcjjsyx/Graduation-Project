#!/usr/bin/env python3
"""Generate the T10 cross-language INT32 GEMM device run directory.

The sparse pipeline cannot yet drive a factor+update chain on the real backend
(real assemble belongs to T11), so this fixture uses the *production* Command
Compiler directly on a hand-built compile IR: two independent 2x2 roots whose
fronts have total_dim = 2 and pivot_dim = 1, i.e. each node has a real 1x1
Schur update.  The device run directory contains only ``manifest.json`` and
``memory_image.bin``; the C++ device must load it through the schema v2
artifact path and execute the INT32 GEMM commands with the SCALE_DESC policy
read at runtime.

Expected device results (exact in the INT32 path):
  node.0.update = 0 - (2/4)*1      = -0.5
  node.1.update = 1 - (1/3)*1      = 2/3 (up to the local quantization step)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SYSTEMC_DIR = Path(__file__).resolve().parents[1]
SOFTWARE_DIR = SYSTEMC_DIR.parent / "software"
if str(SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_DIR))

from src.config import (  # noqa: E402
    CommandCompilerConfig,
    GemmConfig,
    PivotPolicyConfig,
)
from src.dataStruct import NodeCompileRecord, NodeRange  # noqa: E402
from src.scheduler.command_compiler import compile_command_artifact  # noqa: E402


def write_device(device: Path, artifact) -> None:
    if device.exists():
        shutil.rmtree(device)
    device.mkdir(parents=True)
    (device / "manifest.json").write_text(
        json.dumps(artifact.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (device / "memory_image.bin").write_bytes(artifact.image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out_root: Path = args.out
    device = out_root / "int32_minimal" / "device"
    nodes = [
        NodeCompileRecord(0, -1, NodeRange(0, 1), (0, 1)),
        NodeCompileRecord(1, -1, NodeRange(2, 3), (2, 3)),
    ]
    map_tables: list[list] = [[], []]
    local_fronts = [
        np.array([[4.0, 1.0], [2.0, 0.0]], dtype=np.float64),
        np.array([[3.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    ]
    artifact = compile_command_artifact(
        nodes=nodes,
        map_tables=map_tables,
        local_fronts=local_fronts,
        permutation=[0, 1, 2, 3],
        rhs=np.array([1.0, 2.0, 3.0, 4.0]),
        config=CommandCompilerConfig(tile_size=16, max_front_size=64, max_wait_tokens=256),
        pivot=PivotPolicyConfig(policy="STATIC", threshold_ratio=0.0),
        gemm=GemmConfig(backend="FP32_SYSTEMC_INT32_GEMM"),
    )
    write_device(device, artifact)
    # Non-default model parameters must change device arithmetic and timing,
    # rather than merely appearing in the compiler's manifest. The tiny L21
    # row quantizes to zero at 4 bits; the C++ test computes its own answer.
    configured = compile_command_artifact(
        nodes=[NodeCompileRecord(0, -1, NodeRange(0, 1), (0, 1, 2))],
        map_tables=[[]],
        local_fronts=[np.array([[1, 1, 1], [1, 0, 0], [2.0**-10, 0, 0]],
                               dtype=np.float64)],
        permutation=[0, 1, 2],
        rhs=np.ones(3),
        config=CommandCompilerConfig(tile_size=16, max_front_size=64, max_wait_tokens=256),
        pivot=PivotPolicyConfig(policy="STATIC", threshold_ratio=0.0),
        gemm=GemmConfig(backend="FP32_SYSTEMC_INT32_GEMM", effective_bits=4,
                        k_block_max=8, conversion_rate=8),
    )
    write_device(out_root / "int32_configured" / "device", configured)
    gemm_commands = sum(
        1 for command in artifact.commands if int(command.opcode) == 0x07
    )
    print(
        json.dumps(
            {
                "device_dir": str(device),
                "command_count": len(artifact.commands),
                "descriptor_count": len(artifact.descriptors),
                "gemm_commands": gemm_commands,
                "kernel_backend": artifact.manifest["compiler"]["kernel_backend"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
