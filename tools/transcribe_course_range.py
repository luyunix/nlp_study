"""Transcribe an inclusive course episode range from downloaded audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx_whisper


AUDIO_DIR = Path("/tmp/full-course-audio")
OUTPUT_DIR = Path("/tmp/full-course-transcripts")
MODEL = "mlx-community/whisper-small-mlx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 1 or args.end > 189 or args.start > args.end:
        raise ValueError("range must satisfy 1 <= start <= end <= 189")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for episode in range(args.start, args.end + 1):
        audio = AUDIO_DIR / f"p{episode:03d}.m4s"
        output = OUTPUT_DIR / f"p{episode:03d}.json"
        if output.exists():
            print(f"P{episode}: already transcribed", flush=True)
            continue
        if not audio.exists():
            raise FileNotFoundError(audio)

        print(f"P{episode}: transcribing", flush=True)
        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=MODEL,
            language="zh",
            initial_prompt=(
                "这是一套中文 NLP 自然语言处理课程，涉及 PyTorch、"
                "RNN、LSTM、GRU、注意力、Seq2Seq、机器翻译、"
                "FastText、Transformers、迁移学习、BERT、ELMo、GPT "
                "以及代码实战。请准确识别中英文技术术语。"
            ),
            verbose=False,
        )
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        segments = result.get("segments", [])
        duration = segments[-1].get("end", 0) if segments else 0
        print(
            f"P{episode}: done, {duration / 60:.1f} min, "
            f"{len(segments)} segments",
            flush=True,
        )


if __name__ == "__main__":
    main()
