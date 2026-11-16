import numpy as np

from official_runtime import OfficialDuplexRuntime


class FakeModel:
    def __init__(self):
        self.resets = 0

    def reset_session(self, **kwargs):
        self.resets += 1


class FakeDuplex:
    """Only exposes official public calls: no decoder or KV injection hooks."""
    def __init__(self):
        self.calls = []
        self.force_listen_count = 0
        self.since_prepare = 0

    def prepare(self, **kwargs):
        self.calls.append(("prepare", kwargs))
        self.since_prepare = 0

    def streaming_prefill(self, **kwargs):
        assert "audio_waveform" in kwargs
        self.calls.append(("prefill", kwargs))
        return {"success": True}

    def streaming_generate(self, **kwargs):
        self.since_prepare += 1
        self.calls.append(("generate", kwargs))
        if self.since_prepare <= self.force_listen_count:
            return {"is_listen": True, "text": "", "audio_waveform": np.zeros(24000),
                    "end_of_turn": False}
        return {"is_listen": False, "text": "每晚六百元。",
                "audio_waveform": np.ones(240, dtype=np.float32) * 0.01,
                "end_of_turn": True, "n_tokens": 5}


def make_runtime(tmp_path):
    runtime = OfficialDuplexRuntime(
        model_path=tmp_path, ref_audio_path=tmp_path / "ref.wav"
    )
    model, duplex = FakeModel(), FakeDuplex()
    runtime._model, runtime._duplex = model, duplex
    runtime._ref_audio = np.zeros(16000, np.float32)
    return runtime, model, duplex


def test_first_query_primes_listening_then_uses_official_text_list(tmp_path):
    runtime, model, duplex = make_runtime(tmp_path)
    chunks = list(runtime.stream_answer(query="上海住宿多少", evidence="上海住宿六百元。"))
    assert chunks[-1]["end_of_turn"]
    prefills = [kw for name, kw in duplex.calls if name == "prefill"]
    assert "audio_waveform" in prefills[0]
    assert "text_list" in prefills[1]
    assert "audio_waveform" in prefills[1]
    assert "text_list" not in prefills[2]
    prompt = "".join(prefills[1]["text_list"])
    assert "上海住宿多少" in prompt and "上海住宿六百元" in prompt
    assert model.resets == 1


def test_second_question_keeps_session_and_injects_new_evidence(tmp_path):
    runtime, model, duplex = make_runtime(tmp_path)
    list(runtime.stream_answer(query="上海住宿多少", evidence="六百元"))
    list(runtime.stream_answer(query="现在多少", evidence="四百三十七元"))
    assert model.resets == 1
    assert len([c for c in duplex.calls if c[0] == "prepare"]) == 1
    assert len([c for c in duplex.calls if c[0] == "prefill" and "text_list" in c[1]]) == 2


def test_audio_question_is_ingested_listening_before_text_unit(tmp_path):
    runtime, model, duplex = make_runtime(tmp_path)
    list(runtime.stream_answer(
        query="上海住宿多少", evidence="六百元",
        audio=np.zeros(24000, np.float32),
    ))
    calls = [kw for name, kw in duplex.calls if name == "prefill"]
    assert [next(iter(c)) for c in calls] == [
        "audio_waveform", "audio_waveform", "audio_waveform", "audio_waveform", "audio_waveform"
    ]


def test_empty_evidence_is_rejected_before_loading_model(tmp_path):
    import pytest
    runtime, model, duplex = make_runtime(tmp_path)
    with pytest.raises(ValueError, match="evidence"):
        list(runtime.stream_answer(query="问题", evidence=" "))
    assert model.resets == 0


def test_truncated_generation_resets_before_the_next_request(tmp_path):
    runtime, model, duplex = make_runtime(tmp_path)
    runtime.max_units = 1
    original = duplex.streaming_generate

    def never_ends(**kwargs):
        result = original(**kwargs)
        result["end_of_turn"] = False
        return result

    duplex.streaming_generate = never_ends
    chunks = list(runtime.stream_answer(query="上海住宿", evidence="六百元"))
    assert chunks[-1]["end_of_turn"] is False
    list(runtime.stream_answer(query="上海住宿", evidence="四百三十七元"))
    assert model.resets == 2
