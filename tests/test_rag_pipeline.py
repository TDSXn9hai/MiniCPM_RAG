import json

import pytest

from rag_pipeline import RAGPipeline, RAGRetriever, RetrievalConfig


def write_kb(path, amount="六百"):
    path.write_text(json.dumps({"documents": [
        {"id": "travel", "title": "上海差旅住宿报销", "content": f"上海住宿每晚最多报销{amount}元。", "source": "travel.pdf#page=8"},
        {"id": "leave", "title": "休假制度", "content": "每年可申请带薪休假。", "source": "leave.pdf"},
    ]}, ensure_ascii=False), encoding="utf-8")


class RecordingRuntime:
    def __init__(self):
        self.calls = []

    def stream_answer(self, *, query, evidence, audio=None):
        self.calls.append((query, evidence, audio))
        yield {"text": "每晚最多报销六百元。", "audio": None, "is_listen": False,
               "end_of_turn": True, "unit": 0, "metrics": {}}


def test_retrieval_keeps_document_ids_sources_and_budget(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)
    retriever = RAGRetriever(path, RetrievalConfig(top_k=1, max_evidence_chars=110))
    result = retriever.retrieve("上海住宿报销多少")
    assert result.usable
    assert result.hits[0].document.id == "travel"
    assert result.evidence.citations[0].source == "travel.pdf#page=8"
    assert result.evidence.citations[0].document_id == "travel"
    assert len(result.evidence.text) <= 110


def test_knowledge_update_rebuilds_index_without_restarting(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)
    retriever = RAGRetriever(path)
    old = retriever.retrieve("上海住宿报销多少")
    write_kb(path, "四百三十七")
    new = retriever.retrieve("上海住宿报销多少")
    assert old.knowledge_sha256 != new.knowledge_sha256
    assert "四百三十七" in new.evidence.text
    assert "六百" not in new.evidence.text


def test_no_match_refuses_without_model_call(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)
    runtime = RecordingRuntime()
    result = RAGPipeline(RAGRetriever(path), runtime).answer("量子纠缠实验原理")
    assert result["status"] == "no_evidence"
    assert "没有足够依据" in result["text"]
    assert runtime.calls == []
    assert result["citations"] == []


def test_success_injects_question_and_retrieved_evidence_and_emits_units(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)
    runtime = RecordingRuntime()
    events = []
    result = RAGPipeline(RAGRetriever(path), runtime).answer(
        "上海住宿报销多少", on_chunk=events.append
    )
    assert result["status"] == "answered"
    assert result["ended_naturally"]
    assert runtime.calls[0][0] == "上海住宿报销多少"
    assert "六百" in runtime.calls[0][1]
    assert len(events) == 1
    assert result["citations"][0]["document_id"] == "travel"
    assert result["text"] == events[0]["text"]


def test_hits_that_do_not_fit_budget_are_not_sent_to_model(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)
    runtime = RecordingRuntime()
    pipeline = RAGPipeline(RAGRetriever(path, RetrievalConfig(max_evidence_chars=5)), runtime)
    result = pipeline.answer("上海住宿")
    assert result["status"] == "no_evidence"
    assert runtime.calls == []


@pytest.mark.parametrize("config", [
    {"top_k": 0}, {"min_score": -0.1}, {"min_score": 1.1},
    {"max_evidence_chars": 0},
])
def test_invalid_retrieval_settings_are_rejected(config):
    with pytest.raises(ValueError):
        RetrievalConfig(**config)


def test_empty_query_is_not_retrieved(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)
    with pytest.raises(ValueError, match="query"):
        RAGRetriever(path).retrieve("  ")


def test_invalid_duplicate_document_ids_are_rejected(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"documents": [
        {"id": "same", "title": "上海", "content": "住宿六百元"},
        {"id": "same", "title": "北京", "content": "住宿四百元"},
    ]}))
    with pytest.raises(ValueError, match="duplicate"):
        RAGRetriever(path).retrieve("上海住宿")


def test_only_one_generic_bigram_is_not_enough_for_a_long_query(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"documents": [
        {"id": "duplex", "title": "双工实验", "content": "这个实验介绍音频输入与模型。"},
    ]}, ensure_ascii=False))
    retriever = RAGRetriever(path, RetrievalConfig(min_score=0.08))
    assert not retriever.retrieve("量子纠缠实验原理").usable


def test_failed_chunk_consumer_closes_the_model_iterator(tmp_path):
    path = tmp_path / "kb.json"
    write_kb(path)

    class Runtime:
        closed = False
        stream = None

        def stream_answer(self, **kwargs):
            def generate():
                try:
                    yield {"text": "回答", "audio": None, "is_listen": False,
                           "end_of_turn": False, "unit": 0, "metrics": {}}
                finally:
                    self.closed = True
            self.stream = generate()
            return self.stream

    runtime = Runtime()

    def broken_consumer(chunk):
        raise OSError("cannot save audio")

    with pytest.raises(OSError, match="cannot save audio"):
        RAGPipeline(RAGRetriever(path), runtime).answer(
            "上海住宿", on_chunk=broken_consumer
        )
    assert runtime.closed
