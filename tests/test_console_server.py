import base64

import numpy as np
from fastapi.testclient import TestClient

from console_server import _build_parser, create_app


def encode_pcm(audio):
    return base64.b64encode(np.asarray(audio, "<f4").tobytes()).decode()


class FakeTurn:
    def __init__(self, on_event):
        self.on_event = on_event
        self.frames = []
        self.ended = 0
        self.cancelled = 0
        self.closed = False
        on_event({"type": "turn.started", "turn_id": "fake", "t_ms": 0})

    def append_audio(self, samples, *, final=False):
        if not self.closed:
            self.frames.append(np.asarray(samples).copy())

    def end_input(self):
        self.ended += 1
        self.on_event(
            {
                "type": "answer.audio.delta",
                "turn_id": "fake",
                "t_ms": 1,
                "sample_rate": 24000,
                "format": "pcm_f32le_24000",
                "unit": 0,
                "audio": np.array([0.0, 0.5], np.float32),
            }
        )
        self.on_event(
            {
                "type": "answer.done",
                "turn_id": "fake",
                "t_ms": 2,
                "status": "answered",
                "text": "回答",
            }
        )

    def cancel(self):
        self.cancelled += 1
        self.closed = True

    def wait(self, timeout=None):
        return None


class FakePipeline:
    def __init__(self):
        self.turns = []
        self.warmed = 0
        self.closed = 0

    def warmup(self):
        self.warmed += 1

    def start_turn(self, on_event, retrieval_config=None):
        turn = FakeTurn(on_event)
        turn.retrieval_config = retrieval_config
        self.turns.append(turn)
        return turn

    def close(self):
        self.closed += 1


def make_client(tmp_path):
    pipeline = FakePipeline()
    app = create_app(
        knowledge_path=tmp_path / "knowledge.json",
        backend_factory=lambda: object(),
        runtime_factory=lambda: object(),
        output_dir=tmp_path / "outputs",
        pipeline_factory=lambda: pipeline,
    )
    return TestClient(app), pipeline


def test_websocket_rejects_audio_before_input_start(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_json(
            {
                "type": "input.audio.append",
                "sequence": 0,
                "audio": encode_pcm(np.zeros(16000, np.float32)),
            }
        )
        event = ws.receive_json()
        assert event["type"] == "turn.error"
        assert event["code"] == "input_not_started"


def test_websocket_rejects_wrong_chunk_and_invalid_base64(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.start", "sample_rate": 16000})
        ws.receive_json()
        ws.send_json(
            {
                "type": "input.audio.append",
                "sequence": 0,
                "audio": encode_pcm(np.zeros(8000, np.float32)),
            }
        )
        assert ws.receive_json()["code"] == "invalid_pcm_length"
        ws.send_json(
            {
                "type": "input.audio.append",
                "sequence": 0,
                "audio": "not-base64",
            }
        )
        assert ws.receive_json()["code"] == "invalid_audio"


def test_websocket_streams_audio_as_pcm_and_done(tmp_path):
    client, pipeline = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.start", "sample_rate": 16000})
        ws.receive_json()
        ws.send_json(
            {
                "type": "input.audio.append",
                "sequence": 0,
                "audio": encode_pcm(np.zeros(16000, np.float32)),
            }
        )
        ws.send_json({"type": "input.end", "sequence": 1})
        audio_event = ws.receive_json()
        done_event = ws.receive_json()
        assert audio_event["type"] == "answer.audio.delta"
        decoded = np.frombuffer(base64.b64decode(audio_event["audio"]), "<f4")
        np.testing.assert_array_equal(decoded, np.array([0.0, 0.5], np.float32))
        assert done_event["type"] == "answer.done"
        assert pipeline.turns[0].ended == 1


def test_second_input_start_is_busy(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.start", "sample_rate": 16000})
        ws.receive_json()
        ws.send_json({"type": "input.start", "sample_rate": 16000})
        assert ws.receive_json()["code"] == "turn_busy"


def test_thirty_second_limit(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.start", "sample_rate": 16000})
        ws.receive_json()
        for sequence in range(31):
            ws.send_json(
                {
                    "type": "input.audio.append",
                    "sequence": sequence,
                    "audio": encode_pcm(np.zeros(16000, np.float32)),
                }
            )
        event = ws.receive_json()
        assert event["code"] == "audio_too_long"


def test_disconnect_cancels_active_turn(tmp_path):
    client, pipeline = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.start", "sample_rate": 16000})
        ws.receive_json()
    assert pipeline.turns[0].cancelled == 1


def test_health_reports_ready(tmp_path):
    client, pipeline = make_client(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert pipeline.warmed == 1


def test_console_defaults_to_full_text_retrieval():
    args = _build_parser().parse_args([])
    assert args.title_only is False
    assert args.port == 8005


def test_console_accepts_tls_files():
    args = _build_parser().parse_args([
        "--ssl-certfile", "/tmp/cert.pem",
        "--ssl-keyfile", "/tmp/key.pem",
    ])
    assert args.ssl_certfile.name == "cert.pem"
    assert args.ssl_keyfile.name == "key.pem"


def test_input_start_applies_per_turn_retrieval_config(tmp_path):
    client, pipeline = make_client(tmp_path)
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "input.start",
            "sample_rate": 16000,
            "retrieval": {
                "title_only": False,
                "top_k": 5,
                "min_score": 0.12,
                "max_evidence_chars": 700,
            },
        })
        ws.receive_json()
    config = pipeline.turns[0].retrieval_config
    assert config.top_k == 5
    assert config.min_score == 0.12
    assert config.title_only is False
