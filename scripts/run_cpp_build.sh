#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SYSTEMC_HOME="${SYSTEMC_HOME:-/usr/local/systemc-3.0.2}"
BUILD_DIR="${BUILD_DIR:-/tmp/gp-docs-rel}"
JOBS="${JOBS:-4}"

[[ -x "$PYTHON_BIN" ]] || {
    printf 'error: Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
}
[[ -d "$SYSTEMC_HOME" ]] || {
    printf 'error: SystemC directory not found: %s\n' "$SYSTEMC_HOME" >&2
    exit 1
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'error: JOBS must be a positive integer: %s\n' "$JOBS" >&2
    exit 1
}

cd "$ROOT_DIR"
cmake -S graduation-code/systemc -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DSYSTEMC_HOME="$SYSTEMC_HOME" \
    -DT04_PYTHON="$PYTHON_BIN"
exec cmake --build "$BUILD_DIR" --parallel "$JOBS"
