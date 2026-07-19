"""Small, testable implementations of the course's preprocessing concepts.

The functions deliberately depend only on Python's standard library.  They are
not replacements for jieba, pandas, FastText, or PyTorch; their purpose is to
make the underlying data transformations visible.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Hashable, Iterable, Sequence, TypeVar

Token = TypeVar("Token", bound=Hashable)


def build_vocabulary(tokens: Iterable[Token]) -> dict[Token, int]:
    """Assign an id to each token, preserving first-occurrence order."""

    vocabulary: dict[Token, int] = {}
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def one_hot(token: Token, vocabulary: dict[Token, int]) -> list[int]:
    """Return the one-hot vector for *token*.

    Raises:
        KeyError: if the token is not present in the vocabulary.
    """

    index = vocabulary[token]
    vector = [0] * len(vocabulary)
    vector[index] = 1
    return vector


def one_hot_corpus(
    tokens: Sequence[Token],
    vocabulary: dict[Token, int] | None = None,
) -> tuple[dict[Token, int], list[list[int]]]:
    """Encode a token sequence as a one-hot matrix."""

    vocabulary = vocabulary or build_vocabulary(tokens)
    return vocabulary, [one_hot(token, vocabulary) for token in tokens]


def cbow_examples(
    tokens: Sequence[Token],
    window_size: int = 1,
) -> list[tuple[tuple[Token, ...], Token]]:
    """Create ``(context, center_word)`` training examples for CBOW."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    examples: list[tuple[tuple[Token, ...], Token]] = []
    for center in range(window_size, len(tokens) - window_size):
        context = tuple(
            tokens[index]
            for index in range(center - window_size, center + window_size + 1)
            if index != center
        )
        examples.append((context, tokens[center]))
    return examples


def skipgram_examples(
    tokens: Sequence[Token],
    window_size: int = 1,
) -> list[tuple[Token, Token]]:
    """Create ``(center_word, context_word)`` training pairs for Skip-Gram."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    examples: list[tuple[Token, Token]] = []
    for center, center_token in enumerate(tokens):
        start = max(0, center - window_size)
        stop = min(len(tokens), center + window_size + 1)
        for context in range(start, stop):
            if context != center:
                examples.append((center_token, tokens[context]))
    return examples


def ngrams(
    tokens: Sequence[Token],
    n: int,
    *,
    include_lower_orders: bool = False,
) -> list[tuple[Token, ...]]:
    """Return contiguous n-grams.

    With ``include_lower_orders=True``, return orders ``1..n``.  This mirrors
    the course's feature-expansion explanation.
    """

    if n < 1:
        raise ValueError("n must be at least 1")

    orders = range(1, n + 1) if include_lower_orders else (n,)
    return [
        tuple(tokens[start : start + order])
        for order in orders
        for start in range(len(tokens) - order + 1)
    ]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two equally sized vectors."""

    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    if not left:
        raise ValueError("vectors must not be empty")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot / (left_norm * right_norm)


def normalize_length(
    sequence: Sequence[Token],
    target_length: int,
    pad_value: Token,
    *,
    padding: str = "right",
    truncation: str = "right",
) -> list[Token]:
    """Truncate or pad a sequence to a fixed length."""

    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if padding not in {"left", "right"}:
        raise ValueError("padding must be 'left' or 'right'")
    if truncation not in {"left", "right"}:
        raise ValueError("truncation must be 'left' or 'right'")

    values = list(sequence)
    if len(values) > target_length:
        values = (
            values[-target_length:]
            if truncation == "left" and target_length
            else values[:target_length]
        )

    missing = target_length - len(values)
    pads = [pad_value] * missing
    return pads + values if padding == "left" else values + pads


def label_distribution(labels: Iterable[Token]) -> dict[Token, int]:
    """Count labels for class-balance inspection."""

    return dict(Counter(labels))


def word_frequencies(tokens: Iterable[Token]) -> list[tuple[Token, int]]:
    """Return frequency pairs in descending count, then first-seen order."""

    values = list(tokens)
    counts = Counter(values)
    first_seen: dict[Token, int] = {}
    for token in values:
        first_seen.setdefault(token, len(first_seen))
    return sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
