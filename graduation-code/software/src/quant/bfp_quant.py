from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

TILE = 32
SUB = 16
B_EFF = 24
Q_MAX = 2 ** (B_EFF - 1) - 1


@dataclass
class QuantResult:
    q: np.ndarray
    e: np.ndarray
    orig_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    tiles: Tuple[int, int]
    sat_count: int   ## 饱和计数
    total_elements: int   ## 总元素数
    clip_count: int ## 裁剪计数 统计被百分位剪裁掉的数量


def _quantize_block(x: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
    if x.size == 0:
        return x.astype(np.int32), 0, 0, 0, 0
    a = np.percentile(np.abs(x), 99.5)  ## 99.5% 分位数 有待商榷，根据具体例子调整
    if a == 0:
        return np.zeros_like(x, dtype=np.int32), 0, 0, x.size, 0
    clip_count = int(np.count_nonzero(np.abs(x) > a))
    e = int(np.ceil(np.log2(a / Q_MAX)))
    x_c = np.clip(x, -a, a)
    q = np.clip(np.round(x_c / (2 ** e)), -Q_MAX, Q_MAX).astype(np.int32)
    if np.count_nonzero(np.abs(q) >= 2) / q.size < 0.05:
        e -= 1
        q = np.clip(np.round(x_c / (2 ** e)), -Q_MAX, Q_MAX).astype(np.int32)
    sat_count = int(np.count_nonzero(np.abs(q) == Q_MAX))
    return q, e, sat_count, x.size, clip_count

'''
量化一个矩阵, 返回量化后的结果, 指数矩阵, 饱和计数, 总元素数
'''
def quantize_matrix(x: np.ndarray) -> QuantResult:
    h, w = x.shape
    h_pad = (TILE - h % TILE) % TILE
    w_pad = (TILE - w % TILE) % TILE
    padded = np.pad(x, ((0, h_pad), (0, w_pad)), mode="constant") ## 填充为32的整数倍
    tiles_y = padded.shape[0] // TILE
    tiles_x = padded.shape[1] // TILE

    q_out = np.zeros_like(padded, dtype=np.int32)
    e_out = np.zeros((tiles_y, tiles_x, 4), dtype=np.int8)

    sat_count = 0
    total_elements = 0
    clip_total = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * TILE
            x0 = tx * TILE
            tile = padded[y0 : y0 + TILE, x0 : x0 + TILE]
            # 4 sub-blocks: (00,01,10,11)
            blocks = [
                tile[0:SUB, 0:SUB],
                tile[0:SUB, SUB:TILE],
                tile[SUB:TILE, 0:SUB],
                tile[SUB:TILE, SUB:TILE],
            ]
            for bi, block in enumerate(blocks):
                q_block, e, b_sat, b_total, b_clip = _quantize_block(block) 
                sat_count += b_sat
                total_elements += b_total
                clip_total += b_clip
                e_out[ty, tx, bi] = np.int8(e)
                if bi == 0:
                    q_out[y0 : y0 + SUB, x0 : x0 + SUB] = q_block
                elif bi == 1:
                    q_out[y0 : y0 + SUB, x0 + SUB : x0 + TILE] = q_block
                elif bi == 2:
                    q_out[y0 + SUB : y0 + TILE, x0 : x0 + SUB] = q_block
                else:
                    q_out[y0 + SUB : y0 + TILE, x0 + SUB : x0 + TILE] = q_block

    return QuantResult(
        q=q_out,
        e=e_out,
        orig_shape=(h, w),
        padded_shape=padded.shape, # type: ignore
        tiles=(tiles_y, tiles_x),
        sat_count=sat_count,
        total_elements=total_elements,
        clip_count=clip_total,
    )

'''
反量化一个矩阵, 返回反量化后的结果
'''
def dequantize(q: np.ndarray, e: np.ndarray) -> np.ndarray:
    tiles_y, tiles_x, _ = e.shape
    out = np.zeros_like(q, dtype=np.float32)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * TILE
            x0 = tx * TILE
            exps = e[ty, tx]
            blocks = [
                (slice(0, SUB), slice(0, SUB)),
                (slice(0, SUB), slice(SUB, TILE)),
                (slice(SUB, TILE), slice(0, SUB)),
                (slice(SUB, TILE), slice(SUB, TILE)),
            ]
            for bi, (ys, xs) in enumerate(blocks):
                ys_abs = slice(y0 + ys.start, y0 + ys.stop)
                xs_abs = slice(x0 + xs.start, x0 + xs.stop)
                block = q[ys_abs, xs_abs].astype(np.float32) # type: ignore
                out[ys_abs, xs_abs] = block * (2 ** int(exps[bi])) # type: ignore
    return out

'''
将量化后的矩阵和指数矩阵展平为行优先的整数列表，用于二进制输出
'''
def flatten_tiles(q: np.ndarray, e: np.ndarray) -> Tuple[List[int], List[int]]:
    """Flatten tiles to row-major lists for binary output."""
    q_list = q.flatten().astype(np.int32).tolist()
    e_list = e.flatten().astype(np.int8).tolist()
    return q_list, e_list
