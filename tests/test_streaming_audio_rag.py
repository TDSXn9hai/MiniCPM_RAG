from concurrent.futures import CancelledError
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rag_pipeline import RetrievalConfig
from streaming_audio_rag import StreamingAudioRAGPipeline


class FakeRuntime:
    def __init__(self):
        self.received = []
        self.resets = 0
        self.generations = 0
        self.finished = threading.Event()
        self.consume_gate = threading.Event()
        self.consume_gate.set()

    def reset_session(self):
        self.resets += 1

    def warmup(self):
        pass

    def stream_live_answer(
        self, *, audio_queue, retrieval_future, cancel_event, on_ready,
        on_event=None, timeout_s=120.0, wait_unit_s=1.0,
    ):
        try:
            while True:
                if cancel_event.is_set():
                    raise CancelledError()
                assert self.consume_gate.wait(2)
                audio = audio_queue.get(timeout=1)
                if audio is None:
                    break
                self.received.append(np.asarray(audio).copy())
                if on_event:
                    on_event(
                        "audio.listen",
                        phase="input",
                        mode="AUDIO",
                        forced_listen=True,
                        started_s=time.perf_counter(),
                        finished_s=time.perf_counter(),
                    )
            deadline = time.monotonic() + timeout_s
            while not retrieval_future.done():
                if cancel_event.is_set():
                    raise CancelledError()
                if time.monotonic() >= deadline:
                    raise TimeoutError()
                if on_event:
                    on_event(
                        "audio.listen",
                        phase="waiting",
                        mode="AUDIO",
                        forced_listen=True,
                        started_s=time.perf_counter(),
                        finished_s=time.perf_counter(),
                    )
                time.sleep(wait_unit_s)
            if cancel_event.is_set():
                raise CancelledError()
            prepared = retrieval_future.result()
            self.generations += 1
            on_ready(prepared)
            yield {
                "unit": 0,
                "mode": "AUDIO",
                "text": "回答",
                "audio": np.zeros(2400, np.float32),
                "is_listen": False,
                "end_of_turn": True,
                "metrics": {"generation_ms": 1.0},
            }
        finally:
            self.finished.set()

    def close(self):
        pass


class FakeBackend:
    def __init__(self, final="最终问题"):
        self.final = final
        self.calls = 0
        self.block = threading.Event()
        self.first_started = threading.Event()
        self.closed = 0

    def prepare(self):
        pass

    def init_streaming_state(self, **kwargs):
        return {"session_id": "fake"}

    def streaming_transcribe(self, audio, state):
        self.calls += 1
        self.first_started.set()
        if self.calls == 1:
            assert self.block.wait(2)
        return {"text": f"问题{self.calls}"}

    def finish_streaming_transcribe(self, state):
        return {"text": self.final}

    def close(self):
        self.closed += 1


class FakeRetriever:
    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        retrieval = SimpleNamespace(
            query=query,
            evidence=SimpleNamespace(text="证据"),
            usable=True,
            to_dict=lambda: {
                "query": query,
                "usable": True,
                "hits": [],
                "evidence": "证据",
                "citations": [],
            },
        )
        return retrieval


def make_pipeline(tmp_path, *, queue_size=8):
    knowledge = tmp_path / "knowledge.json"
    knowledge.write_text(
        json.dumps({"documents": [{"id": "d", "title": "t", "content": "c"}]}),
        encoding="utf-8",
    )
    runtime = FakeRuntime()
    backend = FakeBackend()
    pipeline = StreamingAudioRAGPipeline(
        runtime,
        backend,
        knowledge,
        RetrievalConfig(),
        timeout_s=2.0,
        queue_size=queue_size,
        retriever_factory=FakeRetriever,
    )
    return pipeline, runtime, backend


def wait_for(events, event_type, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(event["type"] == event_type for event in events):
            return
        time.sleep(0.005)
    raise AssertionError(f"event {event_type} was not emitted")


def test_minicpm_consumes_second_unit_while_asr_is_blocked(tmp_path):
    pipeline, runtime, backend = make_pipeline(tmp_path)
    events = []
    turn = pipeline.start_turn(events.append)
    turn.append_audio(np.ones(16000, np.float32))
    assert backend.first_started.wait(1)
    turn.append_audio(np.ones(16000, np.float32) * 2)
    deadline = time.monotonic() + 1
    while len(runtime.received) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(runtime.received) == 2
    backend.block.set()
    turn.end_input()
    turn.wait(3)
    assert runtime.generations == 1


def test_partial_retrieval_is_emitted_before_final_and_only_final_is_injected(tmp_path):
    pipeline, runtime, backend = make_pipeline(tmp_path)
    events = []
    turn = pipeline.start_turn(events.append)
    backend.block.set()
    turn.append_audio(np.ones(16000, np.float32))
    turn.append_audio(np.ones(16000, np.float32), final=True)
    turn.end_input()
    turn.wait(3)

    kinds = [event["type"] for event in events]
    assert kinds.index("retrieval.partial") < kinds.index("retrieval.final")
    assert "rag.injected" not in kinds
    assert runtime.generations == 1
    assert events[-1]["type"] == "answer.done"


def test_start_turn_resets_session_and_timestamps_share_origin(tmp_path):
    pipeline, runtime, _ = make_pipeline(tmp_path)
    events = []
    turn = pipeline.start_turn(events.append)
    turn.append_audio(np.zeros(16000, np.float32))
    turn.cancel()
    assert runtime.resets == 1
    assert events[0]["type"] == "turn.started"
    assert events[0]["t_ms"] == 0
    assert all(event["t_ms"] >= 0 for event in events)


def test_queue_full_reports_backpressure(tmp_path):
    pipeline, _, backend = make_pipeline(tmp_path, queue_size=1)
    pipeline.runtime.consume_gate.clear()
    events = []
    turn = pipeline.start_turn(events.append)
    backend.block.set()
    turn.append_audio(np.zeros(16000, np.float32))
    with pytest.raises(RuntimeError, match="backpressure"):
        for _ in range(3):
            turn.append_audio(np.zeros(16000, np.float32))
    assert any(
        event["type"] == "turn.error" and event.get("code") == "backpressure"
        for event in events
    )


def test_empty_final_transcript_emits_no_speech_without_generation(tmp_path):
    pipeline, runtime, backend = make_pipeline(tmp_path)
    backend.final = ""
    events = []
    turn = pipeline.start_turn(events.append)
    backend.block.set()
    turn.append_audio(np.zeros(16000, np.float32))
    turn.end_input()
    turn.wait(3)

    assert any(event["type"] == "no_speech" for event in events)
    assert runtime.generations == 0


def test_stale_callbacks_are_ignored_after_cancel(tmp_path):
    pipeline, _, backend = make_pipeline(tmp_path)
    events = []
    turn = pipeline.start_turn(events.append)
    turn.cancel()
    backend.block.set()
    with pytest.raises(RuntimeError, match="not accepting"):
        turn.append_audio(np.zeros(16000, np.float32))
    assert not any(event["type"] == "asr.partial" for event in events)


def test_cancel_releases_turn_while_asr_backend_is_blocked(tmp_path):
    pipeline, runtime, backend = make_pipeline(tmp_path)
    events = []
    turn = pipeline.start_turn(events.append)
    turn.append_audio(np.zeros(16000, np.float32))
    assert backend.first_started.wait(1)
    started = time.monotonic()
    turn.cancel()
    turn.wait(1)
    assert time.monotonic() - started < 1
    assert pipeline._active_turn is None
    backend.block.set()
