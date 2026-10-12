#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

[[ -x "$PYTHON_BIN" ]] || {
    printf 'error: Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
}

cd "$ROOT_DIR/graduation-code/software"
exec "$PYTHON_BIN" -m pytest -q
