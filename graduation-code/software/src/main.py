from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.io

from src.io import NodeTask, write_front_data, write_manifest, write_map_table, write_tasks
from src.memory.planner import plan_memory
from src.quant.bfp_quant import dequantize, flatten_tiles, quantize_matrix
from src.scheduler.map_gen import generate_map_tables
from src.scheduler.task_queue import sibling_friendly_order
from src.symbolic.etree import children_from_parent, elimination_tree
from src.symbolic.reorder import apply_permutation, reorder_rcm
from src.symbolic.supernode import build_supernodes
from src.verify.metrics import residual_norm


def _load_matrix(path: str | None, n: int, density: float, seed: int) -> sp.csr_matrix:
    if path:
        # Check if file is .mat format
        if path.endswith('.mat'):
            try:
                from src.matrix_compress.compress import read_mat_file
                a = read_mat_file(path)
                if a.shape[0] != a.shape[1]:
                    raise ValueError("Matrix must be square")
                return a
            except ImportError:
                # Fall back to scipy.io if matrix_compress is not available
                a = scipy.io.loadmat(path)
                for key, value in a.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        a = sp.csr_matrix(value)
                        if a.shape[0] != a.shape[1]:
                            raise ValueError("Matrix must be square")
                        return a
                raise ValueError("No matrix found in .mat file")
        else:
            # Load MatrixMarket format
            a = scipy.io.mmread(path).tocsr()
            if a.shape[0] != a.shape[1]:
                raise ValueError("Matrix must be square")
            return a
    rng = np.random.default_rng(seed)
    r = sp.random(n, n, density=density, format="csr", random_state=rng)
    a = r + r.T
    a = a + sp.eye(n, format="csr") * n
    return a


def _build_node_ranges(supernodes: List[List[int]]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for sn in supernodes:
        size = len(sn)
        ranges.append((start, start + size))
        start += size
    return ranges


def _make_tasks(
    parent: List[int],
    node_ranges: List[Tuple[int, int]],
    mem_plans,
) -> List[NodeTask]:
    n = len(parent)
    children = children_from_parent(np.asarray(parent, dtype=np.int32))
    tasks: List[NodeTask] = []
    for node_id in range(n):
        p = parent[node_id]
        total_dim = node_ranges[node_id][1] - node_ranges[node_id][0]
        pivot_dim = total_dim
        flags = 0
        if len(children[node_id]) == 0:
            flags |= 1
        if p < 0:
            flags |= 2
        plan = mem_plans[node_id]
        tasks.append(
            NodeTask(
                node_id=node_id,
                flags=flags,
                parent_id=p if p >= 0 else 0xFFFFFFFF,
                children_count=len(children[node_id]),
                total_dim=total_dim,
                pivot_dim=pivot_dim,
                data_addr=plan.front_q.offset,
                parent_address=mem_plans[p].front_q.offset if p >= 0 else 0,
                map_table_addr=plan.map_table.offset,
                l_factor_addr=plan.l_factor.offset,
                u_factor_addr=plan.u_factor.offset,
                p_vector_addr=0,
                flag=0,
            )
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Multifrontal LU software-side prototype")
    parser.add_argument("--mtx", type=str, default=None)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.1)
    args = parser.parse_args()

    a = _load_matrix(args.mtx, args.n, args.density, args.seed)

    perm = reorder_rcm(a)
    a_perm = apply_permutation(a, perm)

    etree = elimination_tree(a_perm)
    supernodes = build_supernodes(etree)
    node_ranges = _build_node_ranges(supernodes)
    parent = etree.tolist()

    map_tables = generate_map_tables(node_ranges, parent)

    os.makedirs(args.out, exist_ok=True)
    tasks_path = os.path.join(args.out, "tasks.bin")
    map_path = os.path.join(args.out, "map_table.bin")
    front_q_path = os.path.join(args.out, "front_q.bin")
    front_e_path = os.path.join(args.out, "front_e.bin")
    manifest_path = os.path.join(args.out, "manifest.json")

    # Clear front files for append
    open(front_q_path, "wb").close()
    open(front_e_path, "wb").close()

    map_offsets = write_map_table(map_path, map_tables)

    q_sizes: List[int] = []
    e_sizes: List[int] = []
    q_offsets: List[int] = []
    e_offsets: List[int] = []
    tile_shapes: List[Tuple[int, int]] = []
    clip_total = 0
    sat_total = 0

    q_cursor = 0
    e_cursor = 0
    for node_id, (start, end) in enumerate(node_ranges):
        block = a_perm[start:end, start:end].toarray().astype(np.float32)
        qr = quantize_matrix(block)
        q_list, e_list = flatten_tiles(qr.q, qr.e)
        tile_shapes.append(qr.tiles)
        clip_total += qr.clip_count
        sat_total += qr.sat_count
        q_offsets.append(q_cursor)
        e_offsets.append(e_cursor)
        q_bytes, e_bytes = write_front_data(front_q_path, front_e_path, q_list, e_list)
        q_sizes.append(q_bytes)
        e_sizes.append(e_bytes)
        q_cursor += q_bytes
        e_cursor += e_bytes

    map_sizes = [os.path.getsize(map_path) - offset for offset in map_offsets]
    if map_sizes:
        # compute per-node size from offsets
        total = os.path.getsize(map_path)
        map_sizes = []
        for i, off in enumerate(map_offsets):
            next_off = map_offsets[i + 1] if i + 1 < len(map_offsets) else total
            map_sizes.append(next_off - off)

    mem_plans, total_bytes = plan_memory(len(node_ranges), q_sizes, e_sizes, map_sizes)

    tasks = _make_tasks(parent, node_ranges, mem_plans)
    order = sibling_friendly_order(parent)
    ordered_tasks = [tasks[i] for i in order]
    write_tasks(tasks_path, ordered_tasks)

    manifest = {
        "total_bytes": total_bytes,
        "quantization": {
            "clip_count": clip_total,
            "sat_count": sat_total,
        },
        "nodes": {
            str(node_id): {
                "front_q": mem_plans[node_id].front_q.__dict__,
                "front_e": mem_plans[node_id].front_e.__dict__,
                "map_table": mem_plans[node_id].map_table.__dict__,
                "front_q_file_offset": q_offsets[node_id],
                "front_e_file_offset": e_offsets[node_id],
                "map_table_file_offset": map_offsets[node_id],
                "tiles": list(tile_shapes[node_id]),
            }
            for node_id in range(len(node_ranges))
        },
        "task_order": order,
    }
    write_manifest(manifest_path, manifest)

    # Verification
    b = np.ones(a_perm.shape[0], dtype=np.float32)
    x = spla.spsolve(a_perm.tocsr(), b)
    res = residual_norm(a_perm, x, b)
    print(f"residual_norm: {res:.3e}")


if __name__ == "__main__":
    main()
