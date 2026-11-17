#!/usr/bin/env python3
"""Exercise Qwen3-ASR's official streaming HTTP endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18180")
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path(
            "/root/paddlejob/workspace/env_run/avatar_full/"
            "minicpm_rag_test/tests/browser/fake-mic.wav"
        ),
    )
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    session = requests.post(f"{args.url.rstrip('/')}/api/start", timeout=30)
    session.raise_for_status()
    session_id = session.json()["session_id"]

    step = max(1, round(args.chunk_seconds * sample_rate))
    for index, offset in enumerate(range(0, audio.size, step), 1):
        chunk = np.asarray(audio[offset:offset + step], dtype=np.float32)
        response = requests.post(
            f"{args.url.rstrip('/')}/api/chunk",
            params={"session_id": session_id},
            data=chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=60,
        )
        response.raise_for_status()
        print(index, response.json())

    response = requests.post(
        f"{args.url.rstrip('/')}/api/finish",
        params={"session_id": session_id},
        timeout=60,
    )
    response.raise_for_status()
    print("final", response.json())


if __name__ == "__main__":
    main()
