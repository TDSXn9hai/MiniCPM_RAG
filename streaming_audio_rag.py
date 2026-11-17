"""Streaming PCM coordination for ASR, speculative retrieval and MiniCPM."""
from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
import itertools
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Literal

import numpy as np

from rag_pipeline import RAGRetriever


@dataclass(frozen=True)
class ASRUpdate:
    kind: Literal["partial", "final"]
    revision: int
    text: str
    sequence: int
    started_s: float
    finished_s: float
    metadata: dict


@dataclass(frozen=True)
class RetrievalUpdate:
    kind: Literal["partial", "final"]
    revision: int
    retrieval: object
    started_s: float
    finished_s: float
    thread_id: int


class BackpressureError(RuntimeError):
    pass


_START = object()
_FINISH = object()
_CANCEL = object()
_CLOSE = object()


class StreamingASRWorker:
    """Consume ordered 1-second PCM chunks on one background thread."""

    def __init__(
        self,
        backend,
        *,
        language: str = "Chinese",
        on_update: Callable[[ASRUpdate], None],
        on_error: Callable[[BaseException], None] | None = None,
        close_backend: bool = True,
        queue_size: int = 0,
    ) -> None:
        self.backend = backend
        self.language = language
        self.on_update = on_update
        self.on_error = on_error
        self.close_backend = close_backend
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run, name="streaming-asr", daemon=True
        )
        self._thread.start()
        self._state = None
        self._revision = 0
        self._sequence = 0
        self._accepting = True
        self._finished = False
        self._closed = False
        self._canceled = False
        self._closed_event = threading.Event()
        self._started_event = threading.Event()
        self._start_error = None

    def _cancel_backend(self) -> None:
        if self._state is None:
            return
        cancel = getattr(self.backend, "cancel_streaming_transcribe", None)
        if cancel is not None:
            try:
                cancel(self._state)
            except Exception:
                pass
        self._state = None

    def _emit(self, kind, text, started_s, finished_s, metadata):
        if self._canceled:
            return
        update = ASRUpdate(
            kind=kind,
            revision=self._revision,
            text=text,
            sequence=self._sequence,
            started_s=started_s,
            finished_s=finished_s,
            metadata={**metadata, "thread_id": threading.get_ident()},
        )
        self._revision += 1
        self.on_update(update)

    def _fail(self, exc: BaseException) -> None:
        self._accepting = False
        if self.on_error is not None:
            self.on_error(exc)

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _CLOSE:
                        return
                    if item is _CANCEL:
                        self._cancel_backend()
                        return
                    if item is _START:
                        try:
                            self._state = self.backend.init_streaming_state(
                                context="",
                                language=self.language,
                                unfixed_chunk_num=2,
                                unfixed_token_num=5,
                                chunk_size_sec=1.0,
                            )
                        except BaseException as exc:
                            self._start_error = exc
                            raise
                        finally:
                            self._started_event.set()
                        continue
                    if item is _FINISH:
                        started = time.perf_counter()
                        result = self.backend.finish_streaming_transcribe(self._state)
                        finished = time.perf_counter()
                        text = _result_text(result)
                        metadata = dict(result) if isinstance(result, dict) else {}
                        metadata.pop("text", None)
                        self._emit("final", text, started, finished, metadata)
                        self._state = None
                        continue

                    started = time.perf_counter()
                    result = self.backend.streaming_transcribe(item, self._state)
                    finished = time.perf_counter()
                    if self._canceled:
                        self._cancel_backend()
                        return
                    text = _result_text(result)
                    if text:
                        metadata = dict(result) if isinstance(result, dict) else {}
                        metadata.pop("text", None)
                        self._emit("partial", text, started, finished, metadata)
                    self._sequence += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            if not self._canceled:
                self._fail(exc)
        finally:
            if self.close_backend:
                try:
                    self.backend.close()
                except Exception as exc:
                    self._fail(exc)
            self._closed_event.set()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("ASR worker is closed")
        self._queue.put(_START)
        if not self._started_event.wait(30):
            raise RuntimeError("ASR worker did not initialize within 30 seconds")
        if self._start_error is not None:
            raise RuntimeError("ASR worker failed to initialize") from self._start_error

    def append(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("ASR worker is closed")
        if self._finished:
            raise RuntimeError("ASR worker is finished")
        if not self._accepting:
            raise RuntimeError("ASR worker is not accepting audio")
        audio = np.asarray(audio, np.float32)
        if audio.ndim != 1 or audio.shape[0] != 16000:
            self._fail(ValueError("ASR audio chunks must contain exactly 16000 samples"))
            return
        if not np.isfinite(audio).all():
            self._fail(ValueError("ASR audio chunks must contain finite samples"))
            return
        try:
            self._queue.put_nowait(audio.copy())
        except queue.Full as exc:
            error = BackpressureError("ASR worker queue is full")
            self._fail(error)
            raise error from exc

    def finish(self) -> None:
        if self._closed:
            raise RuntimeError("ASR worker is closed")
        if not self._accepting:
            raise RuntimeError("ASR worker is not accepting audio")
        self._accepting = False
        self._finished = True
        self._queue.put(_FINISH)

    def cancel(self) -> None:
        if self._canceled or self._closed:
            return
        self._canceled = True
        self._accepting = False
        self._finished = True
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(_CANCEL)
        except queue.Full:
            pass

    def wait_closed(self, timeout: float | None = None) -> bool:
        return self._closed_event.wait(timeout)

    def close(self) -> None:
        if self._closed:
            self._closed_event.wait()
            return
        self._closed = True
        self._accepting = False
        if not self._canceled:
            self._queue.put(_CLOSE)
        self._thread.join()


def _result_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(getattr(result, "text", "")).strip()


class LatestWinsRetrievalWorker:
    """Run final first, otherwise only the newest queued speculative query."""

    def __init__(
        self,
        retriever,
        *,
        on_update: Callable[[RetrievalUpdate], None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.retriever = retriever
        self.on_update = on_update
        self.on_error = on_error
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._counter = itertools.count()
        self._latest_revision = -1
        self._final_revision = -1
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="streaming-rag", daemon=True
        )
        self._thread.start()

    def submit(self, revision: int, query: str, kind: Literal["partial", "final"]) -> None:
        if self._closed:
            raise RuntimeError("retrieval worker is closed")
        if kind not in {"partial", "final"}:
            raise ValueError("retrieval kind must be 'partial' or 'final'")
        query = str(query).strip()
        if not query:
            return
        revision = int(revision)
        if kind == "final":
            self._final_revision = max(self._final_revision, revision)
        self._latest_revision = max(self._latest_revision, revision)
        priority = 0 if kind == "final" else 1
        self._queue.put((priority, -revision, next(self._counter), kind, revision, query))

    def _run(self) -> None:
        while True:
            priority, _, _, kind, revision, query = self._queue.get()
            try:
                if priority == 2:
                    return
                if kind == "partial" and (
                    revision < self._latest_revision
                    or revision <= self._final_revision
                ):
                    continue
                started = time.perf_counter()
                retrieval = self.retriever.retrieve(query)
                finished = time.perf_counter()
                if kind == "partial" and revision <= self._final_revision:
                    continue
                self.on_update(
                    RetrievalUpdate(
                        kind,
                        revision,
                        retrieval,
                        started,
                        finished,
                        threading.get_ident(),
                    )
                )
            except BaseException as exc:
                if self.on_error is not None:
                    self.on_error(exc)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put((2, 0, next(self._counter), "close", 0, ""))
        self._thread.join()


class StreamingAudioRAGPipeline:
    """Coordinate independent MiniCPM, ASR and speculative retrieval workers."""

    def __init__(
        self,
        runtime,
        asr_backend,
        knowledge_path,
        retrieval_config,
        *,
        timeout_s: float = 120.0,
        queue_size: int = 8,
        retriever_factory: Callable[[Path, object], object] | None = None,
        asr_backend_factory: Callable[[], object] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.runtime = runtime
        self.asr_backend = asr_backend
        self.asr_backend_factory = asr_backend_factory
        self.knowledge_path = Path(knowledge_path).expanduser().resolve()
        self.retrieval_config = retrieval_config
        self.timeout_s = float(timeout_s)
        self.queue_size = int(queue_size)
        self.retriever_factory = retriever_factory or RAGRetriever
        self._active_turn = None
        self._lock = threading.Lock()
        self._warmed = False

    def warmup(self) -> None:
        if self._warmed:
            return
        prepare = getattr(self.asr_backend, "prepare", None)
        if prepare is not None and self.asr_backend is not None:
            prepare()
        elif self.asr_backend_factory is not None:
            probe = self.asr_backend_factory()
            probe_prepare = getattr(probe, "prepare", None)
            if probe_prepare is not None:
                probe_prepare()
            probe_close = getattr(probe, "close", None)
            if probe_close is not None:
                probe_close()
        self.runtime.warmup()
        self._warmed = True

    def start_turn(
        self,
        on_event: Callable[[dict], None],
        retrieval_config=None,
    ) -> "LiveTurn":
        with self._lock:
            if self._active_turn is not None:
                raise RuntimeError("turn_busy")
            self.warmup()
            self.runtime.reset_session()
            turn = LiveTurn(self, on_event, retrieval_config or self.retrieval_config)
            self._active_turn = turn
        turn._start_workers()
        return turn

    def close(self) -> None:
        with self._lock:
            turn = self._active_turn
        if turn is not None:
            turn.cancel()
            turn.wait(5)
        close = getattr(self.asr_backend, "close", None)
        if close is not None and self.asr_backend is not None:
            close()
        close_runtime = getattr(self.runtime, "close", None)
        if close_runtime is not None:
            close_runtime()


class LiveTurn:
    def __init__(self, pipeline: StreamingAudioRAGPipeline, on_event, retrieval_config) -> None:
        self.pipeline = pipeline
        self.on_event = on_event
        self.retrieval_config = retrieval_config
        self.turn_id = f"turn-{time.time_ns()}"
        self.origin_s = None
        self.runtime_queue: queue.Queue = queue.Queue(maxsize=pipeline.queue_size)
        self.retrieval_future: Future = Future()
        self.cancel_event = threading.Event()
        self._active = True
        self._input_ended = False
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._asr_worker = None
        self._retrieval_worker = None
        self._runtime_thread = None
        self._text = []
        self._input_samples = 0
        self._input_ended_s = None
        self._final_retrieval_finished_s = None

    def _start_workers(self) -> None:
        retriever = self.pipeline.retriever_factory(
            self.pipeline.knowledge_path, self.retrieval_config
        )
        self._retrieval_worker = LatestWinsRetrievalWorker(
            retriever,
            on_update=self._on_retrieval,
            on_error=self._on_error,
        )
        backend = self.pipeline.asr_backend
        close_backend = False
        if backend is None:
            if self.pipeline.asr_backend_factory is None:
                raise RuntimeError("ASR backend is not configured")
            backend = self.pipeline.asr_backend_factory()
            close_backend = True
        self._asr_worker = StreamingASRWorker(
            backend,
            on_update=self._on_asr,
            on_error=self._on_error,
            close_backend=close_backend,
            queue_size=self.pipeline.queue_size,
        )
        self._asr_worker.start()
        self._runtime_thread = threading.Thread(
            target=self._run_runtime,
            name=f"minicpm-live-{self.turn_id}",
            daemon=True,
        )
        self._runtime_thread.start()

    def _emit_event(self, kind: str, **payload) -> None:
        if not self._active and kind not in {"turn.cancelled", "turn.error"}:
            return
        now = time.perf_counter()
        if self.origin_s is None:
            self.origin_s = now
        self.on_event(
            {
                "type": kind,
                "turn_id": self.turn_id,
                "t_ms": 0.0 if kind == "turn.started" else max(
                    0.0, (now - self.origin_s) * 1000.0
                ),
                **payload,
            }
        )

    def append_audio(self, samples: np.ndarray, *, final: bool = False) -> None:
        if not self._active or self._input_ended:
            raise RuntimeError("turn is not accepting audio")
        samples = np.asarray(samples, np.float32).reshape(-1)
        if samples.size == 0 or samples.size > 16000:
            raise ValueError("PCM chunks must contain 1..16000 samples")
        if not final and samples.size != 16000:
            raise ValueError("intermediate PCM chunks must contain 16000 samples")
        if not np.isfinite(samples).all():
            raise ValueError("PCM chunks must contain finite samples")
        if samples.size < 16000:
            samples = np.pad(samples, (0, 16000 - samples.size))
        if self.origin_s is None:
            self.origin_s = time.perf_counter()
            self._emit_event("turn.started")
        try:
            self.runtime_queue.put_nowait(samples.copy())
            self._asr_worker.append(samples)
        except queue.Full as exc:
            error = BackpressureError("backpressure: MiniCPM audio queue is full")
            self._on_error(error)
            raise error from exc
        except BaseException:
            self.cancel()
            raise
        self._input_samples += samples.size

    def end_input(self) -> None:
        if not self._active or self._input_ended:
            return
        self._input_ended = True
        self._input_ended_s = time.perf_counter()
        try:
            self.runtime_queue.put_nowait(None)
        except queue.Full as exc:
            error = BackpressureError("MiniCPM audio queue is full at input end")
            self._on_error(error)
            raise error from exc
        self._asr_worker.finish()

    def _on_asr(self, update: ASRUpdate) -> None:
        if not self._active or self.pipeline._active_turn is not self:
            return
        self._emit_event(
            f"asr.{update.kind}",
            revision=update.revision,
            text=update.text,
            sequence=update.sequence,
            started_s=update.started_s,
            finished_s=update.finished_s,
            metadata=update.metadata,
            thread_id=update.metadata.get("thread_id"),
            input_duration_ms=self._input_samples / 16000 * 1000.0,
            asr_final_after_end_ms=(
                (update.finished_s - self._input_ended_s) * 1000.0
                if update.kind == "final" and self._input_ended_s is not None
                else None
            ),
        )
        if update.kind == "partial" and update.text:
            self._retrieval_worker.submit(
                revision=update.revision,
                query=update.text,
                kind="partial",
            )
        elif update.kind == "final":
            if not update.text:
                self._emit_event("no_speech")
                self._cancel_runtime()
            else:
                self._retrieval_worker.submit(
                    revision=update.revision,
                    query=update.text,
                    kind="final",
                )

    def _on_retrieval(self, update: RetrievalUpdate) -> None:
        if not self._active or self.pipeline._active_turn is not self:
            return
        self._emit_event(
            f"retrieval.{update.kind}",
            revision=update.revision,
            revision_kind=update.kind,
            started_s=update.started_s,
            finished_s=update.finished_s,
            thread_id=update.thread_id,
            **update.retrieval.to_dict(),
        )
        if update.kind == "final" and not self.retrieval_future.done():
            self._final_retrieval_finished_s = update.finished_s
            self.retrieval_future.set_result(update.retrieval)

    def _on_error(self, exc: BaseException) -> None:
        if not self._active:
            return
        self._error = exc
        self._emit_event(
            "turn.error",
            code="backpressure" if isinstance(exc, BackpressureError) else "worker_error",
            message=str(exc),
        )
        self._cancel_runtime()

    def _cancel_runtime(self) -> None:
        self.cancel_event.set()
        try:
            self.runtime_queue.put_nowait(None)
        except queue.Full:
            pass
        if not self.retrieval_future.done():
            self.retrieval_future.cancel()

    def _run_runtime(self) -> None:
        try:
            if self.cancel_event.is_set():
                raise CancelledError()
            for chunk in self.pipeline.runtime.stream_live_answer(
                audio_queue=self.runtime_queue,
                retrieval_future=self.retrieval_future,
                cancel_event=self.cancel_event,
                on_ready=lambda result: None,
                on_event=self._on_runtime_event,
                timeout_s=self.pipeline.timeout_s,
                wait_unit_s=0.1,
            ):
                text = chunk.get("text") or ""
                if text:
                    self._text.append(text)
                    self._emit_event(
                        "answer.text.delta",
                        unit=chunk["unit"],
                        text=text,
                        metrics=chunk.get("metrics", {}),
                    )
                audio = chunk.get("audio")
                if audio is not None and np.asarray(audio).size:
                    self._emit_event(
                        "answer.audio.delta",
                        unit=chunk["unit"],
                        sample_rate=24000,
                        format="pcm_f32le_24000",
                        audio=np.asarray(audio, np.float32),
                        metrics=chunk.get("metrics", {}),
                    )
                if chunk.get("end_of_turn"):
                    break
            self._emit_event(
                "answer.done",
                text="".join(self._text),
                status="answered" if "".join(self._text).strip() else "incomplete",
            )
        except CancelledError:
            pass
        except BaseException as exc:
            self._error = exc
            self._emit_event("turn.error", code="runtime_error", message=str(exc))
        finally:
            self._active = False
            try:
                if self._asr_worker is not None:
                    if self.cancel_event.is_set():
                        self._asr_worker.cancel()
                    else:
                        self._asr_worker.close()
            finally:
                if self._retrieval_worker is not None:
                    self._retrieval_worker.close()
                if self.pipeline._active_turn is self:
                    self.pipeline._active_turn = None
                self._done.set()

    def _on_runtime_event(self, kind: str, **payload) -> None:
        if self._active:
            if kind == "rag.injected" and self._final_retrieval_finished_s is not None:
                payload["wait_to_inject_ms"] = max(
                    0.0,
                    (float(payload["started_s"]) - self._final_retrieval_finished_s)
                    * 1000.0,
                )
            self._emit_event(kind, **payload, thread_id=threading.get_ident())

    def cancel(self) -> None:
        if not self._active:
            return
        self._emit_event("turn.cancelled")
        self._active = False
        self._cancel_runtime()

    def wait(self, timeout: float | None = None) -> None:
        if not self._done.wait(timeout):
            raise TimeoutError("console turn did not finish")
        if self._error is not None:
            raise self._error
