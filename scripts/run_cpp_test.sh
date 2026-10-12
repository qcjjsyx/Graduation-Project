#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/gp-docs-rel}"

[[ -d "$BUILD_DIR" ]] || {
    printf 'error: C++ build directory not found: %s\n' "$BUILD_DIR" >&2
    printf 'run scripts/run_cpp_build.sh first\n' >&2
    exit 1
}

cd "$ROOT_DIR"
exec ctest --test-dir "$BUILD_DIR" --output-on-failure
