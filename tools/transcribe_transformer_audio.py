"""Transcribe the 38 Transformer lesson audio tracks with MLX Whisper.

The source audio files were obtained from the video's own audio stream during
the earlier browser study pass.  This script deliberately does not use Bilibili
subtitles; it produces timestamped JSON directly from the teacher's speech.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx_whisper


AUDIO_DIR = Path("/tmp")
OUTPUT_DIR = Path("/tmp/transformer-transcripts")
MODEL = "mlx-community/whisper-small-mlx"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for episode in range(106, 144):
        audio = AUDIO_DIR / f"bili-{episode}-audio.m4s"
        output = OUTPUT_DIR / f"p{episode}.json"
        if output.exists():
            print(f"P{episode}: already transcribed", flush=True)
            continue
        if not audio.exists():
            raise FileNotFoundError(audio)

        print(f"P{episode}: transcribing {audio.name}", flush=True)
        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=MODEL,
            language="zh",
            initial_prompt=(
                "这是一节中文自然语言处理课程，主题是 Transformer、"
                "PyTorch、注意力机制、位置编码、掩码、LayerNorm、"
                "Encoder、Decoder 和代码实现。请准确识别技术术语。"
            ),
            verbose=False,
        )
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        duration = result.get("segments", [{}])[-1].get("end", 0)
        print(
            f"P{episode}: done, {duration / 60:.1f} min, "
            f"{len(result.get('segments', []))} segments",
            flush=True,
        )


if __name__ == "__main__":
    main()
