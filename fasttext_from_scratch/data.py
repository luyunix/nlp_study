"""FastText 标签行的最小预处理工具。"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """统一 Unicode/大小写，并把常见标点变成独立 token。"""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"([,.!?;:])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def format_labeled_line(labels: list[str], text: str) -> str:
    """生成 `__label__x text` 格式；一行可有多个标签。"""
    if not labels:
        raise ValueError("至少需要一个标签")
    clean_labels = []
    for label in labels:
        label = label.strip()
        if not label:
            raise ValueError("标签不能为空")
        clean_labels.append(
            label if label.startswith("__label__") else f"__label__{label}"
        )
    return f"{' '.join(clean_labels)} {normalize_text(text)}".strip()


def parse_labeled_line(line: str) -> tuple[list[str], str]:
    """把一行拆成标签列表和正文。"""
    parts = line.strip().split()
    labels = []
    while parts and parts[0].startswith("__label__"):
        labels.append(parts.pop(0)[len("__label__") :])
    if not labels:
        raise ValueError("行首缺少 __label__ 标签")
    return labels, " ".join(parts)

