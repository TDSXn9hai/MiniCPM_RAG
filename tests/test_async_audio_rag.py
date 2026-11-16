from concurrent.futures import Future
import json
import threading
import numpy as np
import pytest

from rag_pipeline import RAGRetriever
from test_official_runtime import make_runtime


def retrieval(tmp_path):
    path = tmp_path / 'kb.json'
    path.write_text(json.dumps({'documents': [{'id': 'plant', 'title': '植物大战僵尸',
        'content': '植物大战僵尸提醒检查阳光和草坪。'}]}, ensure_ascii=False))
    return RAGRetriever(path)


def test_worker_asr_and_retrieval_overlap_minicpm_audio_on_single_owner(tmp_path):
    from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline
    runtime, model, duplex = make_runtime(tmp_path)
    entered, audio_seen = threading.Event(), threading.Event()
    owner = threading.get_ident()
    threads = []

    class ASR:
        def prepare(self):
            threads.append(threading.get_ident())
        def transcribe(self, audio):
            threads.append(threading.get_ident())
            entered.set()
            assert audio_seen.wait(3), 'ASR and MiniCPM did not overlap'
            return '植物大战僵尸', {'device': 'cuda:0'}
        def close(self):
            threads.append(threading.get_ident())

    original = duplex.streaming_prefill
    def prefill(**kwargs):
        assert threading.get_ident() == owner
        if entered.is_set():
            audio_seen.set()
        assert 'audio_waveform' in kwargs, 'TEXT-only prefill is forbidden'
        return original(**kwargs)
    duplex.streaming_prefill = prefill
    worker = ASRRetrievalWorker(ASR(), retrieval(tmp_path))
    events = []
    try:
        pipeline = AsyncAudioRAGPipeline(worker, runtime)
        answer = pipeline.answer(np.zeros(16000, np.float32), on_event=lambda kind, **data: events.append((kind, data)))
        assert answer['status'] == 'answered'
        assert answer['asr']['transcript'] == '植物大战僵尸'
        assert answer['citations'][0]['document_id'] == 'plant'
        assert answer['background']['asr_started_s'] < answer['background']['asr_finished_s']
        assert any(k == 'audio.listen' for k, _ in events)
        assert any(k == 'rag.injected' for k, _ in events)
        assert threads and len(set(threads)) == 1 and threads[0] != owner
    finally:
        worker.close()
        runtime.close()


def test_pending_retrieval_stays_listening_then_injects_audio_and_releases(tmp_path):
    from async_audio_rag import PreparedAudioQuery
    runtime, _, duplex = make_runtime(tmp_path)
    future = Future()
    prepared = PreparedAudioQuery(retrieval(tmp_path).retrieve('植物大战僵尸'), {}, {})
    original = duplex.streaming_generate
    forced_text = []
    def generate(**kwargs):
        last = [kw for name, kw in duplex.calls if name == 'prefill'][-1]
        if 'text_list' in last:
            forced_text.append(duplex.force_listen_count)
        result = original(**kwargs)
        audio_calls = sum(name == 'prefill' for name, _ in duplex.calls)
        if audio_calls >= 3 and not future.done():
            future.set_result(prepared)
        return result
    duplex.streaming_generate = generate
    ready = []
    chunks = list(runtime.stream_pending_answer(
        audio=np.zeros(16000, np.float32), retrieval_future=future,
        on_ready=ready.append, wait_unit_s=0.001, timeout_s=1,
    ))
    assert ready == [prepared]
    assert chunks[-1]['end_of_turn']
    prefills = [kw for name, kw in duplex.calls if name == 'prefill']
    injection = next(i for i, kw in enumerate(prefills) if 'text_list' in kw)
    assert 'audio_waveform' in prefills[injection]
    assert prefills[injection]['text_list'][0].endswith('\n')
    assert not prefills[injection]['text_list'][0].endswith('\n\n')
    assert 'audio_waveform' in prefills[injection + 1]
    assert all('audio_waveform' in kw for kw in prefills)
    assert forced_text and all(count > 0 for count in forced_text)
    assert duplex.force_listen_count == 0


def test_pending_timeout_does_not_speak_and_resets_session(tmp_path):
    runtime, model, duplex = make_runtime(tmp_path)
    pending = Future()
    with pytest.raises(TimeoutError, match='retrieval'):
        list(runtime.stream_pending_answer(audio=np.zeros(16000, np.float32),
             retrieval_future=pending, on_ready=lambda result: None,
             timeout_s=0.01, wait_unit_s=0.001))
    assert not runtime._active and not runtime._session_ready
    assert duplex.force_listen_count == 0
    list(runtime.stream_answer(query='上海住宿', evidence='六百元'))
    assert model.resets == 2


def test_asr_failure_never_releases_for_speech(tmp_path):
    runtime, _, duplex = make_runtime(tmp_path)
    failed = Future()
    failed.set_exception(ValueError('ASR failed'))
    with pytest.raises(ValueError, match='ASR failed'):
        list(runtime.stream_pending_answer(audio=np.zeros(16000, np.float32),
             retrieval_future=failed, on_ready=lambda result: None))
    assert all('text_list' not in kw for name, kw in duplex.calls if name == 'prefill')
    assert not runtime._session_ready


def test_audio_no_evidence_still_generates_model_response(tmp_path):
    from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline
    runtime, _, duplex = make_runtime(tmp_path)
    class ASR:
        def prepare(self): pass
        def transcribe(self, audio): return '量子纠缠', {}
        def close(self): pass
    worker = ASRRetrievalWorker(ASR(), retrieval(tmp_path))
    try:
        result = AsyncAudioRAGPipeline(worker, runtime).answer(np.zeros(16000, np.float32))
        assert result['status'] == 'answered'
        assert result['model_called'] and result['generation_called']
        assert result['no_evidence'] and not result['application_refusal_enabled']
        assert result['text'] == '每晚六百元。'
        assert result['units'][-1]['end_of_turn']
        payloads = [kw['text_list'][0] for name, kw in duplex.calls
                    if name == 'prefill' and 'text_list' in kw]
        assert len(payloads) == 1
        assert '<reference>\n\n</reference>' in payloads[0]
    finally:
        worker.close()
        runtime.close()


def test_failed_consumer_releases_owner_and_forced_listen_configuration(tmp_path):
    from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline
    runtime, model, duplex = make_runtime(tmp_path)
    class ASR:
        def prepare(self): pass
        def transcribe(self, audio): return '植物大战僵尸', {}
        def close(self): pass
    worker = ASRRetrievalWorker(ASR(), retrieval(tmp_path))
    def reject(chunk):
        raise OSError('cannot save output')
    try:
        pipeline = AsyncAudioRAGPipeline(worker, runtime)
        with pytest.raises(OSError, match='cannot save'):
            pipeline.answer(np.zeros(16000, np.float32), on_chunk=reject)
        assert not runtime._active
        assert duplex.force_listen_count == 0
        result = pipeline.answer(np.zeros(16000, np.float32))
        assert result['status'] == 'answered'
    finally:
        worker.close()
        runtime.close()


def test_invalid_audio_is_rejected_before_loading_or_starting_worker(tmp_path):
    from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline
    runtime, model, _ = make_runtime(tmp_path)
    class ASR:
        def prepare(self): raise AssertionError('must not load')
        def close(self): pass
    worker = ASRRetrievalWorker(ASR(), retrieval(tmp_path))
    try:
        with pytest.raises(ValueError, match='finite'):
            AsyncAudioRAGPipeline(worker, runtime).answer(np.array([float('nan')]))
        assert model.resets == 0
    finally:
        worker.close()
        runtime.close()


def test_model_construction_is_serialized_before_parallel_inference(tmp_path):
    from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline
    runtime, _, _ = make_runtime(tmp_path)
    asr_prepared = threading.Event()
    release_asr = threading.Event()
    class ASR:
        def prepare(self):
            # The main owner must wait for this model constructor to return.
            release_asr.wait(0.05)
            asr_prepared.set()
        def close(self): pass
    original = runtime.warmup
    def warmup():
        assert asr_prepared.is_set(), 'Transformers model constructors must not overlap'
        original()
    runtime.warmup = warmup
    worker = ASRRetrievalWorker(ASR(), retrieval(tmp_path))
    try:
        AsyncAudioRAGPipeline(worker, runtime).warmup()
    finally:
        release_asr.set()
        worker.close()
        runtime.close()


def test_api_rejects_different_asr_and_minicpm_devices(tmp_path):
    from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline
    runtime, _, _ = make_runtime(tmp_path)
    class ASR:
        device = 'cuda:1'
        def close(self): pass
    worker = ASRRetrievalWorker(ASR(), retrieval(tmp_path))
    try:
        with pytest.raises(ValueError, match='same CUDA device'):
            AsyncAudioRAGPipeline(worker, runtime)
    finally:
        worker.close()
        runtime.close()
