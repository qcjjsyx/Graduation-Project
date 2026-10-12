#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-/tmp/gp-docs-rel}"
SAMPLE_ROOT="$(mktemp -d /tmp/gp-docs-smoke.XXXXXX)"
SAMPLE_DIR="${SAMPLE_DIR:-${SAMPLE_ROOT}/input}"
REPORT_DIR="${REPORT_DIR:-${SAMPLE_ROOT}/report}"

[[ -x "$PYTHON_BIN" ]] || {
    printf 'error: Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
}
for target in tree_factor_device tree_factor_check tree_solve_check; do
    [[ -x "$BUILD_DIR/$target" ]] || {
        printf 'error: %s not found in %s\n' "$target" "$BUILD_DIR" >&2
        printf 'run scripts/run_cpp_build.sh first\n' >&2
        exit 1
    }
done
[[ ! -e "$SAMPLE_DIR" ]] || {
    printf 'error: sample input directory already exists: %s\n' "$SAMPLE_DIR" >&2
    exit 1
}
[[ ! -e "$REPORT_DIR" ]] || {
    printf 'error: sample report directory already exists: %s\n' "$REPORT_DIR" >&2
    exit 1
}

cd "$ROOT_DIR"
(
    cd graduation-code/software
    "$PYTHON_BIN" -m src.main \
        --n 128 --density 0.03 --seed 0 --out "$SAMPLE_DIR"
)
"$BUILD_DIR/tree_factor_device" \
    --device-dir "$SAMPLE_DIR" --out "$REPORT_DIR" --backend FP32_SYSTEMC
"$BUILD_DIR/tree_factor_check" "$SAMPLE_DIR" "$REPORT_DIR"
"$BUILD_DIR/tree_solve_check" "$SAMPLE_DIR" "$REPORT_DIR"

printf 'Sample input:  %s\n' "$SAMPLE_DIR"
printf 'Sample report: %s\n' "$REPORT_DIR"
