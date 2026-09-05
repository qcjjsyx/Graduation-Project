from __future__ import annotations

from collections import defaultdict, deque
from typing import List


def sibling_friendly_order(parent: List[int]) -> List[int]:
    """Topological order: children before parent, group siblings when possible."""
    n = len(parent)
    children = defaultdict(list)
    indeg = [0] * n
    for child, p in enumerate(parent):
        if p >= 0:
            children[p].append(child)
            indeg[p] += 1
    # Start with leaves
    q = deque([i for i in range(n) if indeg[i] == 0])
    order: List[int] = []
    while q:
        # Group siblings by parent id to keep them close
        current = sorted(list(q), key=lambda x: (parent[x] if parent[x] >= 0 else -1, x))
        q.clear()
        for node in current:
            order.append(node)
            p = parent[node]
            if p >= 0:
                indeg[p] -= 1
                if indeg[p] == 0:
                    q.append(p)
    if len(order) != n:
        raise ValueError("cycle detected in parent array")
    return order