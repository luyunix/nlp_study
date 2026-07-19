"""Export timestamped speech-recognition JSON as readable Markdown records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TERM_FIXES = (
    (re.compile(r"\btransformers?\b", re.I), "Transformer"),
    (re.compile(r"\bpytorch\b", re.I), "PyTorch"),
    (re.compile(r"\bbert\b", re.I), "BERT"),
    (re.compile(r"\bgpt\b", re.I), "GPT"),
    (re.compile(r"\brnn\b", re.I), "RNN"),
    (re.compile(r"\blstm\b", re.I), "LSTM"),
    (re.compile(r"\bgru\b", re.I), "GRU"),
    (re.compile(r"\bsoftmax\b", re.I), "Softmax"),
    (re.compile(r"\blayer ?norm\b", re.I), "LayerNorm"),
    (re.compile(r"\bencoder\b", re.I), "Encoder"),
    (re.compile(r"\bdecoder\b", re.I), "Decoder"),
    (re.compile(r"\bembedding\b", re.I), "Embedding"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--topic", required=True)
    return parser.parse_args()


def format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def clean_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(r"([，。！？,.!?])\1+", r"\1", value)
    for pattern, replacement in TERM_FIXES:
        value = pattern.sub(replacement, value)
    return value


def source_file(directory: Path, episode: int) -> Path:
    candidates = (
        directory / f"p{episode}.json",
        directory / f"p{episode:02d}.json",
        directory / f"p{episode:03d}.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def export_one(source: Path, output: Path, episode: int, topic: str) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    paragraphs: list[tuple[float, float, str]] = []
    bucket: list[str] = []
    bucket_start = 0.0
    bucket_end = 0.0

    for segment in segments:
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        text = clean_text(segment.get("text", ""))
        if not text:
            continue
        if bucket and end - bucket_start > 60:
            paragraphs.append((bucket_start, bucket_end, "".join(bucket)))
            bucket = []
            bucket_start = start
        elif not bucket:
            bucket_start = start
        bucket.append(text)
        bucket_end = end
    if bucket:
        paragraphs.append((bucket_start, bucket_end, "".join(bucket)))

    lines = [
        f"# {topic}原声逐段记录：P{episode}",
        "",
        "> 本文直接由视频音轨识别并按约一分钟分段，不使用站内字幕。",
        "> 它用于核查老师讲解是否遗漏；口误、识别错误和重复表达会在对应学习文章中进一步校正。",
        "",
    ]
    for start, end, paragraph in paragraphs:
        lines.extend(
            [
                f"## {format_time(start)}–{format_time(end)}",
                "",
                paragraph,
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for episode in range(args.start, args.end + 1):
        source = source_file(args.source, episode)
        output = args.output / f"p{episode:03d}.md"
        export_one(source, output, episode, args.topic)
        print(f"exported P{episode}: {output}")


if __name__ == "__main__":
    main()
