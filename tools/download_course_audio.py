"""Download the remaining lesson audio streams from the Bilibili course.

The script reads the public page list and play-url APIs, downloads only the
audio stream, and skips lessons that already exist.  It is intentionally
separate from transcription so network transfer and speech recognition can be
resumed independently.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


BVID = "BV14mdfBDE4Q"
OUTPUT_DIR = Path("/tmp/full-course-audio")
MISSING_RANGES = (range(1, 5), range(38, 106), range(144, 190))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
    ),
    "Referer": f"https://www.bilibili.com/video/{BVID}",
}


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url: str, output: Path) -> None:
    temporary = output.with_suffix(".part")
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination)
    temporary.replace(output)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page_list = request_json(
        "https://api.bilibili.com/x/player/pagelist?"
        + urllib.parse.urlencode({"bvid": BVID, "jsonp": "jsonp"})
    )["data"]
    if len(page_list) != 189:
        raise RuntimeError(f"expected 189 pages, received {len(page_list)}")

    targets = {episode for group in MISSING_RANGES for episode in group}
    for page in page_list:
        episode = page["page"]
        if episode not in targets:
            continue
        output = OUTPUT_DIR / f"p{episode:03d}.m4s"
        if output.exists():
            print(f"P{episode}: already downloaded", flush=True)
            continue

        query = urllib.parse.urlencode(
            {
                "bvid": BVID,
                "cid": page["cid"],
                "qn": 64,
                "fnval": 16,
                "fnver": 0,
                "fourk": 0,
            }
        )
        play_data = request_json(
            f"https://api.bilibili.com/x/player/playurl?{query}"
        )["data"]
        audio_streams = play_data["dash"]["audio"]
        best_audio = max(audio_streams, key=lambda item: item.get("bandwidth", 0))
        audio_url = best_audio.get("baseUrl") or best_audio["base_url"]
        print(
            f"P{episode}: downloading {page['part']} "
            f"({best_audio.get('bandwidth', 0)} bps)",
            flush=True,
        )
        download(audio_url, output)
        print(f"P{episode}: saved {output.stat().st_size / 1_000_000:.1f} MB", flush=True)
        time.sleep(0.15)


if __name__ == "__main__":
    main()
