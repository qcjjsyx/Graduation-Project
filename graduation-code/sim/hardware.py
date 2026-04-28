from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


TILE_SIZE = 16


@dataclass(frozen=True)
class HardwareConfig:
    q_use_bits: int = 27
    frac_bits: int = 20
    acc_bits: int = 64
    enable_cycle_model: bool = True
    panel_fact_cycles: int = TILE_SIZE * TILE_SIZE
    trsm_cycles: int = 2 * TILE_SIZE
    gemm_cycles: int = 3 * TILE_SIZE

    @property
    def q_use(self) -> int:
        return (1 << self.q_use_bits) - 1

    @property
    def acc_min(self) -> int:
        return -(1 << (self.acc_bits - 1))

    @property
    def acc_max(self) -> int:
        return (1 << (self.acc_bits - 1)) - 1


@dataclass
class AssemblySource:
    q: np.ndarray
    exponent: int
    rows: Sequence[int] | None = None
    cols: Sequence[int] | None = None
    name: str = "source"


@dataclass
class AssemblyStats:
    e_asm: int
    e_node: int
    align_drop_count: int = 0
    align_shift_max: int = 0
    asm_overflow_count: int = 0
    requant_sat_count: int = 0
    source_count: int = 0
    maxabs_acc: int = 0


@dataclass
class OperationLogEntry:
    op_type: str
    tile_i: int
    tile_j: int
    tile_k: int
    cycles: int
    m_dim: int = TILE_SIZE
    n_dim: int = TILE_SIZE
    k_dim: int = TILE_SIZE


@dataclass
class LUResult:
    row_map: np.ndarray
    e_node: int
    l_int: np.ndarray
    u_int: np.ndarray
    l_float: np.ndarray
    u_float: np.ndarray
    op_log: list[OperationLogEntry]
    stats: AssemblyStats
    swap_count: int = 0
    total_cycles: int = 0
    physical_sram: np.ndarray | None = None


@dataclass
class SolveResult:
    x: np.ndarray
    residual_history: list[float] = field(default_factory=list)
    correction_history: list[float] = field(default_factory=list)
    converged: bool = False


class Quantizer:
    def __init__(self, config: HardwareConfig):
        self.config = config

    def choose_exponent_from_float(self, x: np.ndarray) -> int:
        max_abs = float(np.max(np.abs(x))) if x.size else 0.0
        if max_abs == 0.0:
            return 0
        return int(math.ceil(math.log2(max_abs / self.config.q_use)))

    def choose_exponent_from_int_acc(self, max_abs: int) -> int:
        if max_abs == 0:
            return 0
        return int(math.ceil(math.log2(max_abs / self.config.q_use)))

    def quantize_m_format(self, x: np.ndarray) -> tuple[np.ndarray, int, int]:
        e_node = self.choose_exponent_from_float(x)
        scale = 2.0 ** e_node
        q_float = np.rint(x / scale)
        q_clip = np.clip(q_float, -self.config.q_use, self.config.q_use)
        sat_count = int(np.count_nonzero(q_float != q_clip))
        return q_clip.astype(np.int64), e_node, sat_count

    @staticmethod
    def round_shift_value(value: int, shift: int) -> int:
        value = int(value)
        if shift == 0:
            return value
        if shift < 0:
            return value << (-shift)

        offset = 1 << (shift - 1)
        if value >= 0:
            return (value + offset) >> shift
        return -(((-value) + offset) >> shift)

    def round_shift_array(self, values: np.ndarray, shift: int) -> np.ndarray:
        values_i64 = np.asarray(values, dtype=np.int64)
        if shift == 0:
            return values_i64.copy()
        if shift < 0:
            return (values_i64 << (-shift)).astype(np.int64)

        offset = np.int64(1 << (shift - 1))
        out = np.empty_like(values_i64)
        pos = values_i64 >= 0
        out[pos] = (values_i64[pos] + offset) >> shift
        out[~pos] = -(((-values_i64[~pos]) + offset) >> shift)
        return out

    @staticmethod
    def round_div_value(numerator: int, denominator: int) -> int:
        numerator = int(numerator)
        denominator = int(denominator)
        if denominator == 0:
            raise ZeroDivisionError("integer hardware divider received zero denominator")

        sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
        num_abs = abs(numerator)
        den_abs = abs(denominator)
        return sign * ((num_abs + (den_abs // 2)) // den_abs)

    def clip_mantissa(self, value: int) -> tuple[int, bool]:
        if value > self.config.q_use:
            return self.config.q_use, True
        if value < -self.config.q_use:
            return -self.config.q_use, True
        return int(value), False

    def clip_accumulator(self, value: int) -> tuple[int, bool]:
        if value > self.config.acc_max:
            return self.config.acc_max, True
        if value < self.config.acc_min:
            return self.config.acc_min, True
        return int(value), False


class ATU:
    def __init__(self, rows: int):
        self.row_map = np.arange(rows, dtype=np.int64)
        self.swap_count = 0

    def physical_row(self, logical_row: int) -> int:
        return int(self.row_map[logical_row])

    def swap_rows(self, row_a: int, row_b: int) -> None:
        if row_a == row_b:
            return
        self.row_map[row_a], self.row_map[row_b] = (
            self.row_map[row_b],
            self.row_map[row_a],
        )
        self.swap_count += 1


class HPU:
    def select_pivot(self, candidates: Iterable[tuple[int, int]]) -> tuple[int, int]:
        best_row: int | None = None
        best_value = 0
        best_abs = -1

        for row, value in candidates:
            abs_value = abs(int(value))
            if abs_value > best_abs:
                best_row = int(row)
                best_value = int(value)
                best_abs = abs_value

        if best_row is None:
            raise ValueError("HPU received an empty pivot candidate stream")
        return best_row, best_value


class HardwareSystem:
    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig()
        self.quantizer = Quantizer(self.config)
        self.hpu = HPU()

    def assemble_sources(
        self, sources: Sequence[AssemblySource], shape: tuple[int, int]
    ) -> tuple[np.ndarray, AssemblyStats]:
        sources = list(sources)
        if not sources:
            stats = AssemblyStats(e_asm=0, e_node=0, source_count=0)
            return np.zeros(shape, dtype=np.int64), stats

        e_asm = max(int(src.exponent) for src in sources)
        stats = AssemblyStats(e_asm=e_asm, e_node=e_asm, source_count=len(sources))
        acc = np.zeros(shape, dtype=np.int64)

        for src in sources:
            q = np.asarray(src.q, dtype=np.int64)
            rows = list(range(q.shape[0])) if src.rows is None else list(src.rows)
            cols = list(range(q.shape[1])) if src.cols is None else list(src.cols)
            if q.shape != (len(rows), len(cols)):
                raise ValueError(
                    f"{src.name}: q shape {q.shape} does not match rows/cols "
                    f"{(len(rows), len(cols))}"
                )

            shift = e_asm - int(src.exponent)
            if shift < 0:
                raise ValueError("e_asm must be greater than or equal to every source exponent")
            stats.align_shift_max = max(stats.align_shift_max, shift)

            for local_i, row in enumerate(rows):
                if row < 0 or row >= shape[0]:
                    raise IndexError(f"{src.name}: row index {row} out of shape {shape}")
                for local_j, col in enumerate(cols):
                    if col < 0 or col >= shape[1]:
                        raise IndexError(f"{src.name}: col index {col} out of shape {shape}")

                    src_value = int(q[local_i, local_j])
                    aligned = self.quantizer.round_shift_value(src_value, shift)
                    if src_value != 0 and aligned == 0:
                        stats.align_drop_count += 1

                    new_value, overflowed = self.quantizer.clip_accumulator(
                        int(acc[row, col]) + aligned
                    )
                    if overflowed:
                        stats.asm_overflow_count += 1
                    acc[row, col] = new_value

        maxabs_acc = int(np.max(np.abs(acc))) if acc.size else 0
        stats.maxabs_acc = maxabs_acc
        e_delta = self.quantizer.choose_exponent_from_int_acc(maxabs_acc)
        e_node = e_asm + e_delta
        stats.e_node = e_node

        req_shift = e_node - e_asm
        node_q = np.zeros(shape, dtype=np.int64)
        for i in range(shape[0]):
            for j in range(shape[1]):
                scaled = self.quantizer.round_shift_value(int(acc[i, j]), req_shift)
                clipped, saturated = self.quantizer.clip_mantissa(scaled)
                if saturated:
                    stats.requant_sat_count += 1
                node_q[i, j] = clipped

        return node_q, stats

    def factorize_dense(self, a_float: np.ndarray) -> LUResult:
        a_float = np.asarray(a_float, dtype=np.float64)
        if a_float.ndim != 2 or a_float.shape[0] != a_float.shape[1]:
            raise ValueError("factorize_dense expects a square 2-D matrix")

        front_int, e_node, sat_count = self.quantizer.quantize_m_format(a_float)
        stats = AssemblyStats(
            e_asm=e_node,
            e_node=e_node,
            requant_sat_count=sat_count,
            source_count=1,
            maxabs_acc=int(np.max(np.abs(front_int))) if front_int.size else 0,
        )
        return self.factorize_m_format(front_int, e_node, stats)

    def factorize_m_format(
        self,
        front_int: np.ndarray,
        e_node: int,
        stats: AssemblyStats | None = None,
    ) -> LUResult:
        front_int = np.asarray(front_int, dtype=np.int64)
        if front_int.ndim != 2 or front_int.shape[0] != front_int.shape[1]:
            raise ValueError("factorize_m_format expects a square 2-D matrix")

        n = int(front_int.shape[0])
        phys_sram = front_int.copy()
        atu = ATU(n)
        op_log: list[OperationLogEntry] = []
        f = self.config.frac_bits

        for panel_start in range(0, n, TILE_SIZE):
            panel_end = min(panel_start + TILE_SIZE, n)
            panel_tile = panel_start // TILE_SIZE
            self._log_op(op_log, "PANEL_FACT", panel_tile, panel_tile, panel_tile)

            for pivot_col in range(panel_start, panel_end):
                pivot_row, _ = self._find_pivot(phys_sram, atu, pivot_col)
                atu.swap_rows(pivot_col, pivot_row)

                pivot_val = int(phys_sram[atu.physical_row(pivot_col), pivot_col])
                if pivot_val == 0:
                    raise np.linalg.LinAlgError(
                        f"zero pivot in quantized domain at column {pivot_col}"
                    )

                pivot_phys = atu.physical_row(pivot_col)
                for row in range(pivot_col + 1, n):
                    row_phys = atu.physical_row(row)
                    a_val = int(phys_sram[row_phys, pivot_col])
                    l_val = self.quantizer.round_div_value(a_val << f, pivot_val)
                    phys_sram[row_phys, pivot_col] = l_val

                    if pivot_col + 1 < panel_end:
                        cols = slice(pivot_col + 1, panel_end)
                        products = l_val * phys_sram[pivot_phys, cols]
                        delta = self.quantizer.round_shift_array(products, f)
                        phys_sram[row_phys, cols] = phys_sram[row_phys, cols] - delta

            if panel_end < n:
                self._solve_u12(phys_sram, atu, panel_start, panel_end)
                for tile_j in range(panel_end // TILE_SIZE, self._ceil_tiles(n)):
                    self._log_op(op_log, "TRSM_U", panel_tile, tile_j, panel_tile)

                self._update_schur(phys_sram, atu, panel_start, panel_end)
                for tile_i in range(panel_end // TILE_SIZE, self._ceil_tiles(n)):
                    for tile_j in range(panel_end // TILE_SIZE, self._ceil_tiles(n)):
                        self._log_op(op_log, "GEMM_SCHUR", tile_i, tile_j, panel_tile)

        logical_lu = phys_sram[atu.row_map, :]
        l_int = np.tril(logical_lu, -1).astype(np.int64)
        u_int = np.triu(logical_lu).astype(np.int64)
        l_float = l_int.astype(np.float64) / (2.0 ** f)
        l_float += np.eye(n, dtype=np.float64)
        u_float = u_int.astype(np.float64) * (2.0 ** int(e_node))
        total_cycles = sum(entry.cycles for entry in op_log)

        return LUResult(
            row_map=atu.row_map.copy(),
            e_node=int(e_node),
            l_int=l_int,
            u_int=u_int,
            l_float=l_float,
            u_float=u_float,
            op_log=op_log,
            stats=stats or AssemblyStats(e_asm=int(e_node), e_node=int(e_node)),
            swap_count=atu.swap_count,
            total_cycles=total_cycles,
            physical_sram=phys_sram.copy(),
        )

    def solve_quantized_lu(self, lu: LUResult, b_float: np.ndarray) -> np.ndarray:
        return self._solve_float_rhs(lu, np.asarray(b_float, dtype=np.float64))

    def iterative_refine(
        self,
        a_float: np.ndarray,
        b_float: np.ndarray,
        lu: LUResult,
        max_iters: int = 10,
        tol: float = 1e-10,
    ) -> SolveResult:
        a_float = np.asarray(a_float, dtype=np.float64)
        b_float = np.asarray(b_float, dtype=np.float64)
        x = np.zeros_like(b_float, dtype=np.float64)
        residual_history: list[float] = []
        correction_history: list[float] = []
        converged = False

        residual = b_float - a_float @ x
        residual_norm = float(np.linalg.norm(residual))
        residual_history.append(residual_norm)

        for _ in range(max_iters):
            if residual_norm < tol:
                converged = True
                break

            correction = self._solve_float_rhs(lu, residual)
            correction_history.append(float(np.linalg.norm(correction)))
            x = x + correction

            residual = b_float - a_float @ x
            residual_norm = float(np.linalg.norm(residual))
            residual_history.append(residual_norm)
            if residual_norm < tol:
                converged = True
                break

        return SolveResult(
            x=x,
            residual_history=residual_history,
            correction_history=correction_history,
            converged=converged,
        )

    def _find_pivot(self, phys_sram: np.ndarray, atu: ATU, column: int) -> tuple[int, int]:
        candidates = (
            (row, int(phys_sram[atu.physical_row(row), column]))
            for row in range(column, phys_sram.shape[0])
        )
        return self.hpu.select_pivot(candidates)

    def _solve_u12(
        self, phys_sram: np.ndarray, atu: ATU, panel_start: int, panel_end: int
    ) -> None:
        if panel_end >= phys_sram.shape[0]:
            return

        f = self.config.frac_bits
        col_slice = slice(panel_end, phys_sram.shape[1])
        for row in range(panel_start, panel_end):
            row_phys = atu.physical_row(row)
            if row == panel_start:
                continue

            l_values = phys_sram[row_phys, panel_start:row]
            u_rows = [atu.physical_row(r) for r in range(panel_start, row)]
            u_block = phys_sram[np.asarray(u_rows, dtype=np.int64), col_slice]
            mac = l_values @ u_block
            delta = self.quantizer.round_shift_array(mac, f)
            phys_sram[row_phys, col_slice] = phys_sram[row_phys, col_slice] - delta

    def _update_schur(
        self, phys_sram: np.ndarray, atu: ATU, panel_start: int, panel_end: int
    ) -> None:
        if panel_end >= phys_sram.shape[0]:
            return

        f = self.config.frac_bits
        trailing_rows = atu.row_map[panel_end:]
        trailing_cols = np.arange(panel_end, phys_sram.shape[1], dtype=np.int64)
        panel_rows = atu.row_map[panel_start:panel_end]

        l_block = phys_sram[np.ix_(trailing_rows, np.arange(panel_start, panel_end))]
        u_block = phys_sram[np.ix_(panel_rows, trailing_cols)]
        acc = l_block @ u_block
        delta = self.quantizer.round_shift_array(acc, f)
        phys_sram[np.ix_(trailing_rows, trailing_cols)] = (
            phys_sram[np.ix_(trailing_rows, trailing_cols)] - delta
        )

    def _solve_float_rhs(self, lu: LUResult, rhs_float: np.ndarray) -> np.ndarray:
        rhs_float = np.asarray(rhs_float, dtype=np.float64)
        if rhs_float.ndim != 1:
            raise ValueError("only vector RHS is supported in this first hardware model")
        if rhs_float.shape[0] != lu.l_int.shape[0]:
            raise ValueError("RHS length does not match LU dimension")

        rhs_permuted = rhs_float[lu.row_map]
        rhs_int, rhs_exp, _ = self.quantizer.quantize_m_format(rhs_permuted)
        x_int = self._solve_int_lu(lu.l_int, lu.u_int, rhs_int)
        total_scale = 2.0 ** (rhs_exp - lu.e_node - self.config.frac_bits)
        return x_int.astype(np.float64) * total_scale

    def _solve_int_lu(
        self, l_int: np.ndarray, u_int: np.ndarray, rhs_int: np.ndarray
    ) -> np.ndarray:
        n = int(rhs_int.shape[0])
        f = self.config.frac_bits
        y = np.zeros(n, dtype=np.int64)
        x = np.zeros(n, dtype=np.int64)

        for i in range(n):
            mac = 0
            for j in range(i):
                mac += int(l_int[i, j]) * int(y[j])
            y[i] = int(rhs_int[i]) - self.quantizer.round_shift_value(mac, f)

        for i in range(n - 1, -1, -1):
            mac = 0
            for j in range(i + 1, n):
                mac += int(u_int[i, j]) * int(x[j])
            delta = self.quantizer.round_shift_value(mac, f)
            rem = int(y[i]) - delta
            diag = int(u_int[i, i])
            if diag == 0:
                raise np.linalg.LinAlgError(f"zero diagonal in U at row {i}")
            x[i] = self.quantizer.round_div_value(rem << f, diag)

        return x

    def _log_op(
        self,
        op_log: list[OperationLogEntry],
        op_type: str,
        tile_i: int,
        tile_j: int,
        tile_k: int,
    ) -> None:
        if not self.config.enable_cycle_model:
            cycles = 0
        elif op_type == "PANEL_FACT":
            cycles = self.config.panel_fact_cycles
        elif op_type == "TRSM_U":
            cycles = self.config.trsm_cycles
        elif op_type == "GEMM_SCHUR":
            cycles = self.config.gemm_cycles
        else:
            cycles = 0

        op_log.append(
            OperationLogEntry(
                op_type=op_type,
                tile_i=int(tile_i),
                tile_j=int(tile_j),
                tile_k=int(tile_k),
                cycles=int(cycles),
            )
        )

    @staticmethod
    def _ceil_tiles(n: int) -> int:
        return (int(n) + TILE_SIZE - 1) // TILE_SIZE


def generate_test_matrix(n: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "stable":
        a = rng.standard_normal((n, n))
        a += np.eye(n) * n
        return a

    if mode == "random":
        return rng.standard_normal((n, n))

    if mode == "pivot_stress":
        a = rng.standard_normal((n, n))
        a += np.eye(n) * 0.1
        for col in range(n):
            row = min(col + 1, n - 1)
            if row != col:
                a[row, col] += 5.0 + rng.random()
        return a

    if mode == "large_value":
        return rng.uniform(1.0, 10000.0, (n, n))

    raise ValueError(f"unknown matrix mode: {mode}")


def summarize_ops(op_log: Sequence[OperationLogEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in op_log:
        counts[entry.op_type] = counts.get(entry.op_type, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integer LU hardware behavior model")
    parser.add_argument("--n", type=int, default=256, help="matrix dimension")
    parser.add_argument(
        "--mode",
        choices=["stable", "random", "pivot_stress", "large_value"],
        default="stable",
        help="test matrix mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--ir-iters", type=int, default=10, help="IR iteration limit")
    parser.add_argument("--tol", type=float, default=1e-10, help="IR residual tolerance")
    parser.add_argument("--frac-bits", type=int, default=20, help="QF fractional bits")
    parser.add_argument("--q-use-bits", type=int, default=27, help="usable int mantissa bits")
    parser.add_argument("--no-cycle-model", action="store_true", help="disable cycle counting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    a = generate_test_matrix(args.n, args.mode, rng)
    x_true = rng.uniform(-10.0, 10.0, args.n)
    b = a @ x_true

    config = HardwareConfig(
        q_use_bits=args.q_use_bits,
        frac_bits=args.frac_bits,
        enable_cycle_model=not args.no_cycle_model,
    )
    hw = HardwareSystem(config)
    lu = hw.factorize_dense(a)

    reconstructed = lu.l_float @ lu.u_float
    permuted_a = a[lu.row_map, :]
    abs_err = float(np.linalg.norm(permuted_a - reconstructed, ord="fro"))
    rel_err = abs_err / float(np.linalg.norm(a, ord="fro"))

    x_direct = hw.solve_quantized_lu(lu, b)
    direct_res = float(np.linalg.norm(b - a @ x_direct))
    direct_rel_x = float(np.linalg.norm(x_direct - x_true) / np.linalg.norm(x_true))

    ir = hw.iterative_refine(a, b, lu, max_iters=args.ir_iters, tol=args.tol)
    ir_res = float(np.linalg.norm(b - a @ ir.x))
    ir_rel_x = float(np.linalg.norm(ir.x - x_true) / np.linalg.norm(x_true))

    print("=== Hardware behavior model ===")
    print(f"matrix: {args.n}x{args.n}, tile_size={TILE_SIZE}, mode={args.mode}")
    print(f"q_use_bits={config.q_use_bits}, frac_bits={config.frac_bits}")
    print(f"e_node={lu.e_node}, swaps={lu.swap_count}")
    print(f"LU abs_error={abs_err:.6e}, rel_error={rel_err:.6e}")
    print(f"direct solve residual={direct_res:.6e}, rel_x_error={direct_rel_x:.6e}")
    print(f"IR converged={ir.converged}, final_residual={ir_res:.6e}, rel_x_error={ir_rel_x:.6e}")
    print("IR residual history:", [f"{value:.6e}" for value in ir.residual_history])
    print("op counts:", summarize_ops(lu.op_log))
    print(f"estimated cycles={lu.total_cycles}")
    print(
        "assembly stats:",
        {
            "e_asm": lu.stats.e_asm,
            "e_node": lu.stats.e_node,
            "align_drop_count": lu.stats.align_drop_count,
            "align_shift_max": lu.stats.align_shift_max,
            "asm_overflow_count": lu.stats.asm_overflow_count,
            "requant_sat_count": lu.stats.requant_sat_count,
            "maxabs_acc": lu.stats.maxabs_acc,
        },
    )


if __name__ == "__main__":
    main()
