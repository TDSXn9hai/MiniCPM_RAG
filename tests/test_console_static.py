from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static" / "console"


def test_console_page_has_required_regions():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="micButton"' in html
    assert 'id="evidenceList"' in html
    assert 'id="answerText"' in html
    assert 'id="latencyPanel"' in html
    assert 'id="topKInput"' in html
    assert 'id="minScoreInput"' in html
    assert 'id="retrievalMode"' in html
    assert 'id="historyList"' in html


def test_capture_worklet_defaults_to_one_second():
    worklet = (STATIC / "capture-worklet.js").read_text(encoding="utf-8")
    assert "chunkSize || 16000" in worklet or "chunkSize ?? 16000" in worklet


def test_console_does_not_use_media_recorder():
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in STATIC.glob("*.js")
    )
    assert "MediaRecorder" not in combined


def test_audio_stream_does_not_create_or_download_wav_files():
    source = (STATIC / "audio-stream.js").read_text(encoding="utf-8")
    assert "audio/wav" not in source
    assert "download" not in source.lower()


def test_console_prepares_capture_before_start_and_aborts_on_disconnect():
    source = (STATIC / "console.js").read_text(encoding="utf-8")
    assert "await audio.prepare()" in source
    assert "audio.startCapture()" in source
    assert "audio.abort()" in source
    assert "socket.onclose" in source


def test_evidence_uses_per_hit_source_and_full_injected_text():
    source = (STATIC / "console.js").read_text(encoding="utf-8")
    assert "hit.source" in source
    assert "完整注入证据" in source
