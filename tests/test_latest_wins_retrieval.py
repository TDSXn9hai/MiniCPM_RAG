from types import SimpleNamespace
import threading

from streaming_audio_rag import LatestWinsRetrievalWorker


class BlockingFakeRetriever:
    def __init__(self):
        self.release_event = threading.Event()
        self.started = []
        self._started_event = threading.Event()

    def retrieve(self, query):
        self.started.append(query)
        self._started_event.set()
        assert self.release_event.wait(2)
        return SimpleNamespace(query=query)

    def wait_started(self, count):
        assert self._started_event.wait(2)
        while len(self.started) < count:
            threading.Event().wait(0.001)


def make_worker(retriever):
    updates = []
    worker = LatestWinsRetrievalWorker(retriever, on_update=updates.append)
    return worker, updates


def test_latest_partial_wins_and_stale_partial_is_dropped():
    retriever = BlockingFakeRetriever()
    worker, updates = make_worker(retriever)

    worker.submit(revision=1, query="第一版", kind="partial")
    retriever.wait_started(1)
    worker.submit(revision=2, query="第二版", kind="partial")
    worker.submit(revision=3, query="第三版", kind="partial")
    retriever.release_event.set()
    worker.close()

    assert [u.retrieval.query for u in updates] == ["第一版", "第三版"]
    assert all(u.kind == "partial" for u in updates)


def test_final_retrieval_preempts_queued_partial():
    retriever = BlockingFakeRetriever()
    worker, updates = make_worker(retriever)

    worker.submit(revision=4, query="运行中", kind="partial")
    retriever.wait_started(1)
    worker.submit(revision=5, query="旧 partial", kind="partial")
    worker.submit(revision=5, query="最终问题", kind="final")
    retriever.release_event.set()
    worker.close()

    assert [u.retrieval.query for u in updates] == ["最终问题"]
    assert updates[0].kind == "final"
    assert updates[0].revision == 5


def test_close_drains_final_before_stopping():
    retriever = BlockingFakeRetriever()
    worker, updates = make_worker(retriever)

    worker.submit(revision=7, query="最终问题", kind="final")
    retriever.wait_started(1)
    retriever.release_event.set()
    worker.close()

    assert len(updates) == 1
    assert updates[0].kind == "final"


class BrokenRetriever:
    def retrieve(self, query):
        raise RuntimeError("retrieval failed")


def test_retrieval_error_is_reported_once():
    errors = []
    worker = LatestWinsRetrievalWorker(
        BrokenRetriever(),
        on_update=lambda update: None,
        on_error=errors.append,
    )
    worker.submit(revision=1, query="问题", kind="final")
    worker.close()

    assert len(errors) == 1
    assert str(errors[0]) == "retrieval failed"


def test_running_partial_is_not_published_after_final_arrives():
    retriever = BlockingFakeRetriever()
    worker, updates = make_worker(retriever)

    worker.submit(revision=1, query="运行中的 partial", kind="partial")
    retriever.wait_started(1)
    worker.submit(revision=2, query="最终问题", kind="final")
    retriever.release_event.set()
    worker.close()

    assert [u.kind for u in updates] == ["final"]
    assert updates[0].retrieval.query == "最终问题"
