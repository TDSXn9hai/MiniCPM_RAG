#!/usr/bin/env python3
"""Smoke-test Qwen3-ASR with the official in-process vLLM backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import librosa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path(
            "/root/paddlejob/workspace/env_run/avatar_full/"
            "minicpm_rag_test/tests/browser/fake-mic.wav"
        ),
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from qwen_asr import Qwen3ASRModel

    started = time.perf_counter()
    model = Qwen3ASRModel.LLM(
        str(args.model.resolve()),
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_log_stats=True,
    )
    loaded_s = time.perf_counter() - started

    audio, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    started = time.perf_counter()
    result = model.transcribe([str(args.audio.resolve())], language=args.language)
    infer_s = time.perf_counter() - started

    print(f"audio={args.audio} sample_rate={sample_rate} duration_s={len(audio) / sample_rate:.3f}")
    print(f"load_s={loaded_s:.3f} infer_s={infer_s:.3f}")
    print(f"language={result[0].language!r} text={result[0].text!r}")


if __name__ == "__main__":
    main()
