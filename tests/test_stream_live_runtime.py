from concurrent.futures import CancelledError, Future
from types import SimpleNamespace
import queue
import threading
import time

import numpy as np
import pytest

from official_runtime import OfficialDuplexRuntime


def make_runtime(tmp_path):
    runtime = OfficialDuplexRuntime(
        model_path=tmp_path, ref_audio_path=tmp_path / "ref.wav"
    )
    listens = []
    events = []

    runtime._prepare = lambda: None

    def listen_unit(audio, *, text=None, phase="input", on_event=None):
        listens.append(
            {
                "audio": np.asarray(audio, np.float32).copy(),
                "text": text,
                "phase": phase,
            }
        )
        if on_event is not None:
            on_event(
                "rag.injected" if text is not None else "audio.listen",
                phase=phase,
                mode="AUDIO",
                forced_listen=True,
                started_s=time.perf_counter(),
                finished_s=time.perf_counter(),
            )
        events.append(phase)

    runtime._listen_unit = listen_unit
    runtime._stream_response = lambda prompt, on_event=None: iter(
        [
            {
                "unit": 0,
                "mode": "AUDIO",
                "text": "回答",
                "audio": np.zeros(24000, np.float32),
                "is_listen": False,
                "end_of_turn": True,
                "metrics": {},
            }
        ]
    )
    return runtime, listens, events


def retrieval_result(query="问题"):
    return SimpleNamespace(
        query=query,
        evidence=SimpleNamespace(text="证据"),
    )


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def test_live_runtime_consumes_audio_while_retrieval_is_pending(tmp_path):
    runtime, listens, events = make_runtime(tmp_path)
    audio_queue = queue.Queue()
    for _ in range(2):
        audio_queue.put(np.zeros(16000, np.float32))
    audio_queue.put(None)
    retrieval_future = Future()
    ready = []
    output = []
    error = []

    def consume():
        try:
            output.extend(
                runtime.stream_live_answer(
                    audio_queue=audio_queue,
                    retrieval_future=retrieval_future,
                    cancel_event=threading.Event(),
                    on_ready=ready.append,
                    on_event=lambda kind, **data: events.append(data["phase"]),
                    timeout_s=2.0,
                    wait_unit_s=0.01,
                )
            )
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=consume)
    thread.start()
    wait_until(lambda: "waiting" in events)
    assert sum(phase == "input" for phase in events) >= 2
    assert not retrieval_future.done()

    retrieval_future.set_result(retrieval_result())
    thread.join(2)
    assert not thread.is_alive()
    assert error == []
    assert ready and ready[0].query == "问题"
    assert output[0]["text"] == "回答"


def test_short_final_audio_is_zero_padded(tmp_path):
    runtime, listens, _ = make_runtime(tmp_path)
    audio_queue = queue.Queue()
    audio_queue.put(np.ones(8000, np.float32))
    audio_queue.put(None)
    retrieval_future = Future()
    retrieval_future.set_result(retrieval_result())

    output = list(
        runtime.stream_live_answer(
            audio_queue=audio_queue,
            retrieval_future=retrieval_future,
            cancel_event=threading.Event(),
            on_ready=lambda result: None,
        )
    )

    assert listens[0]["audio"].shape == (16000,)
    assert np.all(listens[0]["audio"][:8000] == 1)
    assert np.all(listens[0]["audio"][8000:] == 0)
    assert output[-1]["end_of_turn"]


def test_cancel_event_stops_retrieval_wait(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    audio_queue = queue.Queue()
    audio_queue.put(None)
    cancel_event = threading.Event()
    errors = []

    def consume():
        try:
            list(
                runtime.stream_live_answer(
                    audio_queue=audio_queue,
                    retrieval_future=Future(),
                    cancel_event=cancel_event,
                    on_ready=lambda result: None,
                    wait_unit_s=0.01,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(0.02)
    cancel_event.set()
    thread.join(2)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], CancelledError)


def test_retrieval_timeout_raises(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    audio_queue = queue.Queue()
    audio_queue.put(None)

    with pytest.raises(TimeoutError):
        list(
            runtime.stream_live_answer(
                audio_queue=audio_queue,
                retrieval_future=Future(),
                cancel_event=threading.Event(),
                on_ready=lambda result: None,
                timeout_s=0.03,
                wait_unit_s=0.01,
            )
        )
