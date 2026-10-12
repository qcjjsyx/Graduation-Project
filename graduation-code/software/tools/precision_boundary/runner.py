"""Run one experiment case end to end.

matrix.mtx -> production compiler (default CLI settings except the device
capacity) -> tree_factor_device (FP32_SYSTEMC) -> tree_factor_check +
tree_solve_check -> host analysis -> result.json.  The device never sees any
FP64 reference file; the host analysis reads them only after the run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import scipy.io

from src.config import (
    CommandCompilerConfig,
    MatrixInputConfig,
    PipelineConfig,
    SolveInputConfig,
)
from src.pipeline import run_pipeline
from src.scheduler.command_compiler import CommandCompileError

from .analysis import analyze
from .matrices import MatrixSpec, build_matrix

# (max_front_size, device_buffer_bytes): first entry that compiles is used.
# Mirrors the documented real-matrix settings (256JJ default, 576JJ 128 KiB,
# 1024JJ max_front 300 + 512 KiB).
CAPACITY_LADDER = (
    (256, 64 * 1024),
    (256, 128 * 1024),
    (256, 256 * 1024),
    (300, 512 * 1024),
    (512, 1024 * 1024),
)


def _compile(matrix_path: Path, art_dir: Path, rhs_seed: int) -> dict:
    last_error = ""
    for max_front, buffer_bytes in CAPACITY_LADDER:
        if art_dir.exists():
            shutil.rmtree(art_dir)
        config = PipelineConfig(
            matrix=MatrixInputConfig(path=str(matrix_path)),
            command=CommandCompilerConfig(
                max_front_size=max_front, device_buffer_bytes=buffer_bytes
            ),
            solve=SolveInputConfig(seed=rhs_seed),
            out_dir=art_dir,
        )
        try:
            outputs = run_pipeline(config)
        except (CommandCompileError, ValueError) as error:
            message = str(error)
            if "exceeds limit" in message or "scratch" in message or "scratchpad" in message:
                last_error = message
                continue
            raise
        return {
            "ok": True,
            "max_front_size": max_front,
            "device_buffer_bytes": buffer_bytes,
            "nodes": outputs.node_count,
            "commands": outputs.command_count,
            "max_front": outputs.max_front_size,
        }
    return {"ok": False, "error": last_error}


def _parse_key_values(line: str) -> dict:
    out = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def _run_device(build: Path, art_dir: Path, rep_dir: Path, max_cycles: int) -> dict:
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
    start = time.monotonic()
    proc = subprocess.run(
        [str(build / "tree_factor_device"), "--device-dir", str(art_dir), "--out", str(rep_dir),
         "--backend", "FP32_SYSTEMC", "--max-cycles", str(max_cycles)],
        capture_output=True, text=True,
    )
    seconds = time.monotonic() - start
    lines = [line for line in proc.stdout.splitlines() if "control_flow_pass=" in line]
    info = _parse_key_values(lines[-1]) if lines else {}
    return {
        "exit_code": proc.returncode,
        "seconds": round(seconds, 2),
        "control_flow_pass": info.get("control_flow_pass") == "1",
        "numeric_status_pass": info.get("numeric_status_pass") == "1",
        "cycles": int(info["cycles"]) if "cycles" in info else None,
        "solve_ok": int(info["solve_ok"]) if "solve_ok" in info else None,
        "solve_commands": int(info["solve_commands"]) if "solve_commands" in info else None,
        "stderr_tail": proc.stderr[-500:],
    }


def _run_checker(build: Path, tool: str, art_dir: Path, rep_dir: Path, report: str) -> dict:
    proc = subprocess.run([str(build / tool), str(art_dir), str(rep_dir)],
                          capture_output=True, text=True)
    path = rep_dir / report
    data = json.loads(path.read_text()) if path.exists() else {}
    data["exit_code"] = proc.returncode
    return data


def run_case(spec: MatrixSpec, out_root: Path, build: Path, rhs_seed: int,
             max_cycles: int, ir_steps: int, force: bool = False,
             keep_artifacts: bool = False) -> dict:
    case_dir = out_root / spec.case_id
    result_path = case_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    case_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"case_id": spec.case_id, "spec": asdict(spec), "rhs_seed": rhs_seed}
    art_dir, rep_dir = case_dir / "art", case_dir / "rep"
    try:
        matrix = build_matrix(spec)
        matrix_path = case_dir / "matrix.mtx"
        scipy.io.mmwrite(str(matrix_path), matrix)
        result["compile"] = _compile(matrix_path, art_dir, rhs_seed)
        if result["compile"]["ok"]:
            device = _run_device(build, art_dir, rep_dir, max_cycles)
            result["device_run"] = device
            device_ok = device["control_flow_pass"] and device["numeric_status_pass"]
            if device["control_flow_pass"]:
                result["factor_check"] = _run_checker(
                    build, "tree_factor_check", art_dir, rep_dir, "validation.json")
                result["solve_check"] = _run_checker(
                    build, "tree_solve_check", art_dir, rep_dir, "solve_validation.json")
            result["analysis"] = analyze(art_dir, rep_dir, ir_steps, device_ok=device_ok)
    except Exception as error:  # recorded, never silently dropped
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    result_path.write_text(json.dumps(result, indent=1))
    if not keep_artifacts:
        # The images are large; result.json keeps everything the analysis needs.
        for path in (art_dir, rep_dir):
            if path.exists():
                shutil.rmtree(path)
    return result
