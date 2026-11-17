#!/usr/bin/env python3
"""Exercise the streaming console over WebSocket with an in-memory WAV fixture."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time

import librosa
import numpy as np
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18200/ws/console")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--expected-document", default="")
    return parser.parse_args()


def interval_overlap(events, start, end):
    total = 0.0
    for event in events:
        if event.get("type") != "audio.listen" or event.get("phase") != "input":
            continue
        started = float(event.get("started_s", 0.0))
        finished = float(event.get("finished_s", 0.0))
        total += max(0.0, min(end, finished) - max(start, started))
    return total


def main():
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    audio, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    if sample_rate != 16000 or not audio.size:
        raise SystemExit("fixture must decode to 16 kHz mono with samples")
    if audio.size > 30 * 16000:
        raise SystemExit("fixture exceeds the console 30-second limit")
    audio = np.asarray(audio, np.float32)

    events = []
    started = time.perf_counter()
    with connect(args.url, proxy=None, open_timeout=15, close_timeout=5) as ws:
        ready = json.loads(ws.recv(timeout=180))
        if ready.get("type") != "session.ready":
            raise RuntimeError(f"unexpected session event: {ready}")
        ws.send(json.dumps({"type": "input.start", "sample_rate": 16000}))

        sequence = 0
        for offset in range(0, audio.size, 16000):
            chunk = audio[offset:offset + 16000]
            final = offset + 16000 >= audio.size
            ws.send(json.dumps({
                "type": "input.audio.append",
                "sequence": sequence,
                "final": final,
                "audio": base64.b64encode(
                    np.asarray(chunk, "<f4").tobytes()
                ).decode(),
            }))
            sequence += 1
        ws.send(json.dumps({"type": "input.end", "sequence": sequence}))

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            event = json.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic())))
            events.append(event)
            if event.get("type") in {"answer.done", "no_speech", "turn.error"}:
                break
        else:
            raise TimeoutError("console turn did not finish before timeout")

    kinds = [event["type"] for event in events]
    for required in ("turn.started", "asr.partial", "retrieval.partial", "retrieval.final",
                     "rag.injected", "answer.text.delta", "answer.audio.delta", "answer.done"):
        if required not in kinds:
            raise AssertionError(f"missing event: {required}; got {kinds}")

    first_positions = {kind: kinds.index(kind) for kind in kinds}
    for earlier, later in (
        ("turn.started", "asr.partial"),
        ("asr.partial", "asr.final"),
        ("retrieval.partial", "retrieval.final"),
        ("retrieval.final", "rag.injected"),
        ("rag.injected", "answer.text.delta"),
    ):
        if earlier in first_positions and later in first_positions:
            assert first_positions[earlier] < first_positions[later], (earlier, later)

    asr_events = [event for event in events if event.get("type", "").startswith("asr.")]
    asr_start = min(float(event["started_s"]) for event in asr_events)
    asr_end = max(float(event["finished_s"]) for event in asr_events)
    overlap = interval_overlap(events, asr_start, asr_end)
    worker_threads = {
        event.get("thread_id") for event in events
        if event.get("type", "").startswith(("asr.", "retrieval.", "audio.listen"))
        and event.get("thread_id") is not None
    }
    assert len(worker_threads) >= 3, worker_threads

    retrieval_events = [
        event for event in events if event.get("type") == "retrieval.partial"
    ]
    rag_overlap = sum(
        max(0.0, min(asr_end, float(event["finished_s"]))
            - max(asr_start, float(event["started_s"])))
        for event in retrieval_events
    )

    if args.expected_document:
        final = next(event for event in events if event.get("type") == "retrieval.final")
        ids = {hit["document_id"] for hit in final.get("hits", [])}
        assert args.expected_document in ids, ids

    serializable = []
    for event in events:
        serializable.append({
            key: value for key, value in event.items() if key != "audio"
        })
    result = {
        "status": "passed",
        "wall_s": time.perf_counter() - started,
        "audio_duration_s": audio.size / 16000,
        "audio_file_created": False,
        "asr_minicpm_overlap_s": overlap,
        "asr_rag_overlap_s": rag_overlap,
        "worker_thread_ids": sorted(worker_threads),
        "events": serializable,
    }
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    media = [
        path for path in out.rglob("*")
        if path.suffix.lower() in {".wav", ".webm", ".opus", ".mp3", ".flac"}
    ]
    assert not media, media
    print(json.dumps({
        "status": result["status"],
        "events": kinds,
        "asr_minicpm_overlap_s": overlap,
        "asr_rag_overlap_s": rag_overlap,
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
