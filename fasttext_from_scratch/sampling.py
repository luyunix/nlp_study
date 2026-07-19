"""负采样的教学版实现。"""

from __future__ import annotations

import random


def sample_negatives(
    counts: dict[str, int],
    positive: str,
    k: int,
    seed: int | None = None,
) -> list[str]:
    """按 count**0.75 加权、不放回抽取负样本。"""
    if positive not in counts:
        raise ValueError("正样本必须存在于词表")
    candidates = [word for word in counts if word != positive]
    if k < 0 or k > len(candidates):
        raise ValueError("k 必须介于 0 和候选负样本数之间")
    rng = random.Random(seed)
    selected: list[str] = []
    pool = candidates[:]
    for _ in range(k):
        weights = [counts[word] ** 0.75 for word in pool]
        word = rng.choices(pool, weights=weights, k=1)[0]
        selected.append(word)
        pool.remove(word)
    return selected

