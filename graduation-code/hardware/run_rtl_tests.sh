#!/usr/bin/env bash
# Self-checking RTL regressions for the independent HPU / ATU prototypes.
# Requires Icarus Verilog >= 12 (iverilog + vvp on PATH).
#
#   bash graduation-code/hardware/run_rtl_tests.sh
#
# Each test prints *_TB PASS and exits 0, or stops with $fatal (non-zero).
# GCU sources are the legacy 16-byte NodeTask prototype and are not part of
# this regression (see hardware/README.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${RTL_BUILD_DIR:-${TMPDIR:-/tmp}/rtl-tests}"
mkdir -p "$OUT"

run() {
    local name="$1" top="$2"; shift 2
    echo "== $name"
    iverilog -g2012 -s "$top" -o "$OUT/$name.vvp" "$@"
    local log="$OUT/$name.log"
    vvp -n "$OUT/$name.vvp" > "$log" 2>&1 || { cat "$log"; return 1; }
    # Icarus prints harmless time-0 "unique case" notes; show the verdict.
    grep -E "_TB (PASS|FAIL)" "$log"
    grep -q "_TB PASS" "$log"
}

cd "$HERE"
run hpu_handshake tb_hpu_handshake \
    HPU/hpu_cmp_node.sv HPU/hpu_core_tree.sv HPU/hpu_top.sv HPU/tb_code/tb_hpu_handshake.sv
run atu_collision tb_ATU_collision \
    ATU/ATU.sv ATU/tb_ATU_collision.sv
echo "RTL regressions passed"
