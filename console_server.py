#!/usr/bin/env python3
"""WebSocket console for streaming voice RAG evaluation."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import threading
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from rag_pipeline import RetrievalConfig
from streaming_audio_rag import StreamingAudioRAGPipeline


MAX_SAMPLES = 30 * 16000
TERMINAL_EVENTS = {"answer.done", "turn.error", "no_speech", "turn.cancelled"}
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        audio = np.asarray(value, "<f4").reshape(-1)
        return base64.b64encode(audio.tobytes()).decode("ascii")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def create_app(
    *,
    knowledge_path,
    backend_factory: Callable[[], object],
    runtime_factory: Callable[[], object],
    output_dir,
    retrieval_config_factory: Callable[[], RetrievalConfig] | None = None,
    pipeline_factory: Callable[[], StreamingAudioRAGPipeline] | None = None,
    qwen_asr_url: str = "http://127.0.0.1:18180",
) -> FastAPI:
    app = FastAPI(title="MiniCPM Streaming RAG Console")
    knowledge_path = Path(knowledge_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    retrieval_config_factory = retrieval_config_factory or RetrievalConfig

    def make_pipeline():
        if pipeline_factory is not None:
            return pipeline_factory()
        return StreamingAudioRAGPipeline(
            runtime_factory(),
            None,
            knowledge_path,
            retrieval_config_factory(),
            asr_backend_factory=backend_factory,
        )

    app.state.knowledge_path = knowledge_path
    app.state.output_dir = output_dir
    app.state.pipeline = None
    app.state.status = "loading"
    app.state.error = None
    app.state.warmup_lock = threading.Lock()
    app.state.qwen_asr_url = qwen_asr_url

    def get_pipeline():
        if app.state.pipeline is None:
            app.state.pipeline = make_pipeline()
        return app.state.pipeline

    async def warmup():
        if app.state.status == "ready":
            return
        async with app.state._async_lock:
            if app.state.status == "ready":
                return
            try:
                await asyncio.to_thread(get_pipeline().warmup)
                app.state.status = "ready"
                app.state.error = None
            except BaseException as exc:
                app.state.status = "error"
                app.state.error = str(exc)
                raise

    app.state._async_lock = asyncio.Lock()

    @app.get("/api/health")
    async def health():
        try:
            await warmup()
        except BaseException:
            pass
        return {
            "status": app.state.status,
            "error": app.state.error,
            "knowledge": str(knowledge_path),
            "qwen_asr_url": qwen_asr_url,
            "audio_persistence": False,
        }

    @app.websocket("/ws/console")
    async def console_ws(websocket: WebSocket):
        await websocket.accept()
        try:
            await warmup()
        except BaseException as exc:
            await websocket.send_json(
                {"type": "session.error", "message": str(exc)}
            )
            await websocket.close()
            return

        try:
            await websocket.send_json(
                {
                    "type": "session.ready",
                    "knowledge": str(knowledge_path),
                    "input_format": "pcm_f32le_16000",
                    "output_format": "pcm_f32le_24000",
                    "chunk_samples": 16000,
                    "audio_persistence": False,
                }
            )
        except WebSocketDisconnect:
            return

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()
        holder = {"turn": None, "samples": 0, "last_sequence": -1}

        def emit(event: dict):
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        async def sender():
            while True:
                event = await event_queue.get()
                await websocket.send_json(_jsonable(event))
                if (
                    event.get("type") in TERMINAL_EVENTS
                    and holder["turn"] is not None
                    and event.get("turn_id") is not None
                    and event.get("turn_id")
                    == getattr(holder["turn"], "turn_id", None)
                ):
                    holder["turn"] = None

        sender_task = asyncio.create_task(sender())

        def protocol_error(code: str, message: str):
            emit(
                {
                    "type": "turn.error",
                    "code": code,
                    "message": message,
                    "t_ms": 0.0,
                }
            )

        try:
            while True:
                message: Any = await websocket.receive_json()
                kind = message.get("type")

                if kind == "input.start":
                    if holder["turn"] is not None:
                        protocol_error("turn_busy", "a turn is already active")
                        continue
                    if int(message.get("sample_rate", 16000)) != 16000:
                        protocol_error("invalid_sample_rate", "sample_rate must be 16000")
                        continue
                    retrieval = message.get("retrieval") or {}
                    retrieval_config = None
                    if retrieval:
                        try:
                            retrieval_config = RetrievalConfig(
                                top_k=int(retrieval.get("top_k", 3)),
                                min_score=float(retrieval.get("min_score", 0.08)),
                                max_evidence_chars=int(
                                    retrieval.get("max_evidence_chars", 900)
                                ),
                                min_shared_bigrams=int(
                                    retrieval.get("min_shared_bigrams", 2)
                                ),
                                title_only=bool(retrieval.get("title_only", False)),
                            )
                        except (TypeError, ValueError) as exc:
                            protocol_error("invalid_retrieval_config", str(exc))
                            continue
                    try:
                        holder["turn"] = get_pipeline().start_turn(
                            emit, retrieval_config=retrieval_config
                        )
                    except RuntimeError as exc:
                        protocol_error(
                            "turn_busy" if str(exc) == "turn_busy" else "turn_start_failed",
                            str(exc),
                        )
                        continue
                    holder["samples"] = 0
                    holder["last_sequence"] = -1
                    continue

                if kind == "input.audio.append":
                    turn = holder["turn"]
                    if turn is None:
                        protocol_error("input_not_started", "send input.start first")
                        continue
                    sequence = int(message.get("sequence", -1))
                    if sequence <= holder["last_sequence"]:
                        protocol_error("invalid_sequence", "sequence must increase")
                        continue
                    try:
                        raw = base64.b64decode(
                            str(message.get("audio", "")), validate=True
                        )
                    except Exception:
                        protocol_error("invalid_audio", "audio must be base64 PCM")
                        continue
                    if len(raw) % 4:
                        protocol_error("invalid_audio", "PCM byte length must be multiple of 4")
                        continue
                    samples = np.frombuffer(raw, "<f4").astype(np.float32, copy=True)
                    final = bool(message.get("final", False))
                    if len(samples) > 16000 or (not final and len(samples) != 16000):
                        protocol_error(
                            "invalid_pcm_length",
                            "intermediate chunks require 16000 samples",
                        )
                        continue
                    if not samples.size or not np.isfinite(samples).all():
                        protocol_error("invalid_audio", "PCM must be finite and nonempty")
                        continue
                    if holder["samples"] + len(samples) > MAX_SAMPLES:
                        protocol_error("audio_too_long", "audio exceeds 30 seconds")
                        turn.cancel()
                        continue
                    try:
                        turn.append_audio(samples, final=final)
                    except BaseException as exc:
                        protocol_error("audio_rejected", str(exc))
                        continue
                    holder["samples"] += len(samples)
                    holder["last_sequence"] = sequence
                    continue

                if kind == "input.end":
                    turn = holder["turn"]
                    if turn is None:
                        protocol_error("input_not_started", "no active turn")
                        continue
                    try:
                        turn.end_input()
                    except BaseException as exc:
                        protocol_error("input_end_failed", str(exc))
                    continue

                if kind == "input.cancel":
                    turn = holder["turn"]
                    if turn is None:
                        protocol_error("input_not_started", "no active turn")
                        continue
                    turn.cancel()
                    holder["turn"] = None
                    continue

                if kind == "session.reset":
                    if holder["turn"] is not None:
                        protocol_error("reset_busy", "cannot reset an active turn")
                    else:
                        emit({"type": "session.reset"})
                    continue

                protocol_error("unsupported_message", f"unsupported type: {kind}")
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            turn = holder["turn"]
            holder["turn"] = None
            if turn is not None:
                turn.cancel()
                try:
                    await asyncio.to_thread(turn.wait, 5)
                except BaseException:
                    pass
            sender_task.cancel()
            try:
                await sender_task
            except BaseException:
                pass

    @app.get("/")
    async def index():
        return RedirectResponse("/console")

    @app.get("/console")
    async def console_page():
        return FileResponse(STATIC_DIR / "console" / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _build_parser():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, default=root / "knowledge/squad_zh_25.json")
    retrieval_mode = parser.add_mutually_exclusive_group()
    retrieval_mode.add_argument(
        "--full-text", dest="title_only", action="store_false",
        help="index title and content (default)",
    )
    retrieval_mode.add_argument(
        "--title-only", dest="title_only", action="store_true",
        help="index titles only",
    )
    parser.set_defaults(title_only=False)
    parser.add_argument("--model-path", type=Path, default=root / "model")
    parser.add_argument("--ref-audio", type=Path, default=root / "assets/ref_audio.wav")
    parser.add_argument("--qwen-asr-url", default="http://127.0.0.1:18180")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
    parser.add_argument("--output", type=Path, default=root / "outputs/console")
    return parser


def main() -> int:
    import uvicorn
    from async_audio_rag import Qwen3ASRHTTPBackend
    from official_runtime import OfficialDuplexRuntime

    args = _build_parser().parse_args()
    app = create_app(
        knowledge_path=args.knowledge,
        backend_factory=lambda: Qwen3ASRHTTPBackend(args.qwen_asr_url),
        runtime_factory=lambda: OfficialDuplexRuntime(
            model_path=args.model_path,
            ref_audio_path=args.ref_audio,
            device=args.device,
        ),
        output_dir=args.output,
        retrieval_config_factory=lambda: RetrievalConfig(title_only=args.title_only),
        qwen_asr_url=args.qwen_asr_url,
    )
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        raise SystemExit("--ssl-certfile and --ssl-keyfile must be provided together")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=str(args.ssl_certfile) if args.ssl_certfile else None,
        ssl_keyfile=str(args.ssl_keyfile) if args.ssl_keyfile else None,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
