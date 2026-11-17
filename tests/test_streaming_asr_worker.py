from collections import deque
import threading
import time

import numpy as np
import pytest

from streaming_audio_rag import StreamingASRWorker


class FakeBackend:
    def __init__(self, partials, final):
        self.partials = deque(partials)
        self.final = final
        self.started = 0
        self.finished = 0
        self.closed = 0

    def init_streaming_state(self, **kwargs):
        self.started += 1
        return {"session_id": "fake", **kwargs}

    def streaming_transcribe(self, audio, state):
        assert audio.dtype == np.float32
        assert audio.shape == (16000,)
        return {"text": self.partials.popleft()}

    def finish_streaming_transcribe(self, state):
        self.finished += 1
        return {"text": self.final}

    def close(self):
        self.closed += 1


def test_asr_worker_emits_partial_then_final_with_revisions():
    backend = FakeBackend(["部分一", "部分二"], "最终问题")
    updates = []
    worker = StreamingASRWorker(backend, on_update=updates.append)

    worker.start()
    worker.append(np.zeros(16000, np.float32))
    worker.append(np.zeros(16000, np.float32))
    worker.finish()
    worker.close()

    assert [(u.kind, u.revision, u.text) for u in updates] == [
        ("partial", 0, "部分一"),
        ("partial", 1, "部分二"),
        ("final", 2, "最终问题"),
    ]
    assert backend.started == 1
    assert backend.finished == 1
    assert backend.closed == 1


def test_asr_worker_skips_empty_partials_and_keeps_empty_final():
    backend = FakeBackend(["", "有内容"], "")
    updates = []
    errors = []
    worker = StreamingASRWorker(
        backend, on_update=updates.append, on_error=errors.append
    )

    worker.start()
    worker.append(np.zeros(16000, np.float32))
    worker.append(np.zeros(16000, np.float32))
    worker.finish()
    worker.close()

    assert [(u.kind, u.text) for u in updates] == [
        ("partial", "有内容"),
        ("final", ""),
    ]
    assert errors == []


class BrokenBackend(FakeBackend):
    def streaming_transcribe(self, audio, state):
        raise RuntimeError("backend failed")


def test_asr_worker_reports_backend_errors_without_leaking_exception():
    backend = BrokenBackend([], "")
    updates = []
    errors = []
    worker = StreamingASRWorker(
        backend, on_update=updates.append, on_error=errors.append
    )

    worker.start()
    worker.append(np.zeros(16000, np.float32))
    worker.close()

    assert updates == []
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "backend failed"
    assert backend.closed == 1


def test_asr_worker_rejects_wrong_chunk_shape_before_backend_call():
    backend = FakeBackend(["unused"], "unused")
    errors = []
    worker = StreamingASRWorker(
        backend, on_update=lambda update: None, on_error=errors.append
    )

    worker.start()
    worker.append(np.zeros(8000, np.float32))
    worker.close()

    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert backend.started == 1
    assert backend.finished == 0


def test_asr_worker_cannot_append_after_finish():
    backend = FakeBackend(["ok"], "final")
    errors = []
    worker = StreamingASRWorker(
        backend, on_update=lambda update: None, on_error=errors.append
    )

    worker.start()
    worker.append(np.zeros(16000, np.float32))
    worker.finish()
    with pytest.raises(RuntimeError, match="finished"):
        worker.append(np.zeros(16000, np.float32))
    worker.close()

    assert errors == []


class BlockingBackend(FakeBackend):
    def __init__(self):
        super().__init__(["late"], "late-final")
        self.request_started = threading.Event()
        self.release = threading.Event()
        self.cancelled = 0

    def streaming_transcribe(self, audio, state):
        self.request_started.set()
        assert self.release.wait(2)
        return {"text": "late"}

    def cancel_streaming_transcribe(self, state):
        self.cancelled += 1


def test_cancel_drops_blocked_asr_without_waiting_for_http_timeout():
    backend = BlockingBackend()
    updates = []
    worker = StreamingASRWorker(
        backend,
        on_update=updates.append,
        close_backend=False,
        queue_size=1,
    )
    worker.start()
    worker.append(np.zeros(16000, np.float32))
    assert backend.request_started.wait(1)

    started = time.monotonic()
    worker.cancel()
    assert time.monotonic() - started < 0.5
    backend.release.set()
    worker.wait_closed(2)

    assert updates == []
    assert backend.cancelled == 1
