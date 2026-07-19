"""一个便于手算核对的哈夫曼编码实现。"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass


@dataclass
class _Node:
    weight: float
    label: str | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


def huffman_codes(weights: dict[str, float]) -> dict[str, str]:
    """返回标签到 0/1 路径的映射。"""
    if not weights:
        return {}
    counter = itertools.count()
    heap = []
    for label, weight in sorted(weights.items()):
        if weight <= 0:
            raise ValueError("权重必须为正数")
        heapq.heappush(heap, (weight, next(counter), _Node(weight, label=label)))
    if len(heap) == 1:
        return {heap[0][2].label: "0"}
    while len(heap) > 1:
        _, _, left = heapq.heappop(heap)
        _, _, right = heapq.heappop(heap)
        parent = _Node(left.weight + right.weight, left=left, right=right)
        heapq.heappush(heap, (parent.weight, next(counter), parent))
    root = heap[0][2]
    result: dict[str, str] = {}

    def visit(node: _Node, prefix: str) -> None:
        if node.label is not None:
            result[node.label] = prefix
            return
        visit(node.left, prefix + "0")
        visit(node.right, prefix + "1")

    visit(root, "")
    return result


def weighted_path_length(weights: dict[str, float], codes: dict[str, str]) -> float:
    """计算 Σ(权重 × 码长)。"""
    return sum(weights[label] * len(codes[label]) for label in weights)

