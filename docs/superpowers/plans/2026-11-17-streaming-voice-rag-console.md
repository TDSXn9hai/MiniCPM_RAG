# Streaming Voice RAG Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser console that streams 1-second PCM microphone chunks through independent MiniCPM, ASR and RAG workers, then shows retrieval, injection, stage latency, text and in-memory PCM audio without saving microphone recordings.

**Architecture:** A FastAPI WebSocket receives JSON control messages and base64 float32 PCM. Each 1-second PCM array is fanned out to a MiniCPM AUDIO owner queue and a standalone Qwen ASR worker. ASR partials feed a separate latest-wins retrieval worker; only the final transcript produces the authoritative retrieval used for evidence injection.

**Tech Stack:** Python 3.10, FastAPI, WebSocket, NumPy, Qwen3-ASR HTTP API, MiniCPM-o official duplex runtime, vanilla HTML/CSS/JS, AudioWorklet, Web Audio API, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-voice-rag-console-design.md`

## Global Constraints

- PCM chunk size is exactly 16000 samples: 1 second of 16 kHz mono float32.
- The final chunk may be shorter and must be zero-padded to 16000 before MiniCPM receives it.
- Browser microphone audio must never use `MediaRecorder`, become WAV/WebM/Opus, or be written to disk.
- MiniCPM output is 24 kHz mono float32 PCM, played in memory; do not write WAV files in the console path.
- MiniCPM AUDIO owner, ASR worker and RAG worker must have independent execution threads/queues.
- ASR worker must not call retrieval. RAG worker must not hold the MiniCPM owner.
- `retrieval.partial` is speculative UI data. Only `retrieval.final` may be injected.
- Default session mode is isolated: reset MiniCPM session before each turn.
- First console release supports only Qwen3-ASR HTTP. Whisper remains in the existing CLI path.
- Full-text retrieval is the default for `knowledge/squad_zh_25.json`.
- Empty final transcript is `no_speech`, not `turn.error`, and must not start generation.
- Generation-time session reset/cancellation is out of scope; disable the reset button while active.
- Do not create git commits unless the user explicitly requests them.

## Review Focus

- WebSocket disconnect while queues are blocked: all threads must stop and release the active turn.
- Final transcript arriving before the MiniCPM owner has consumed the last audio unit: injection must wait.
- Stale partial retrieval arriving after final retrieval or after session reset: it must never overwrite current state.
- Browser AudioContext sample rate not equal to 16 kHz: capture must resample and still send 16000-sample chunks.
- Short final chunk, silence, and empty transcript: no malformed AUDIO unit, no accidental generation, no saved media.

---

### Task 1: Streaming ASR Worker

**Files:**
- Create: `streaming_audio_rag.py`
- Create: `tests/test_streaming_asr_worker.py`

**Interfaces:**
- Consumes: `Qwen3ASRHTTPBackend` methods `init_streaming_state()`, `streaming_transcribe(audio, state)`, `finish_streaming_transcribe(state)`.
- Produces:
  - `ASRUpdate(kind: Literal["partial", "final"], revision: int, text: str, sequence: int, started_s: float, finished_s: float, metadata: dict)`
  - `StreamingASRWorker(backend, *, language="Chinese", on_update, on_error=None)`
  - `StreamingASRWorker.start() -> None`
  - `StreamingASRWorker.append(audio: np.ndarray) -> None`
  - `StreamingASRWorker.finish() -> None`
  - `StreamingASRWorker.close() -> None`

- [ ] **Step 1: Write failing tests for ordered partial/final events**

```python
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
```

- [ ] **Step 2: Add failing tests for empty partials, backend errors and close**

```python
def test_asr_worker_skips_empty_partials_and_reports_empty_final():
    backend = FakeBackend(["", "有内容"], "")
    updates, errors = [], []
    worker = StreamingASRWorker(
        backend, on_update=updates.append, on_error=errors.append
    )
    worker.start()
    worker.append(np.zeros(16000, np.float32))
    worker.append(np.zeros(16000, np.float32))
    worker.finish()
    worker.close()
    assert [u.text for u in updates] == ["有内容", ""]
    assert updates[-1].kind == "final"
    assert errors == []
```

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_streaming_asr_worker.py -q`

Expected: FAIL because `streaming_audio_rag` and `StreamingASRWorker` do not exist.

- [ ] **Step 4: Implement the single-thread worker queue**

Use `queue.Queue` with sentinels for start, append, finish and close. All backend calls run in one worker thread so HTTP order is preserved. Revision increments for every non-empty partial and for final. `close()` enqueues close, joins the thread, then calls the backend `close()` once.

- [ ] **Step 5: Run the focused and existing tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_streaming_asr_worker.py -q
.venv/bin/python -m pytest tests test_evaluation.py -q
```

Expected: all pass.

---

### Task 2: Latest-Wins Speculative Retrieval Worker

**Files:**
- Modify: `streaming_audio_rag.py`
- Create: `tests/test_latest_wins_retrieval.py`

**Interfaces:**
- Consumes: `RAGRetriever.retrieve(query) -> RetrievalResult`.
- Produces:
  - `RetrievalUpdate(kind: Literal["partial", "final"], revision: int, retrieval: RetrievalResult, started_s: float, finished_s: float)`
  - `LatestWinsRetrievalWorker(retriever, *, on_update, on_error=None)`
  - `LatestWinsRetrievalWorker.submit(revision: int, query: str, kind: Literal["partial", "final"]) -> None`
  - `LatestWinsRetrievalWorker.close() -> None`

- [ ] **Step 1: Write failing latest-wins test**

```python
def test_latest_partial_wins_and_stale_partial_is_dropped():
    retriever = BlockingFakeRetriever()
    updates = []
    worker = LatestWinsRetrievalWorker(retriever, on_update=updates.append)
    worker.submit(revision=1, query="第一版", kind="partial")
    retriever.wait_started(1)
    worker.submit(revision=2, query="第二版", kind="partial")
    worker.submit(revision=3, query="第三版", kind="partial")
    retriever.release()
    worker.close()
    assert [u.retrieval.query for u in updates] == ["第一版", "第三版"]
```

- [ ] **Step 2: Write failing final-priority and stale-turn tests**

```python
def test_final_retrieval_preempts_queued_partial():
    retriever = BlockingFakeRetriever()
    updates = []
    worker = LatestWinsRetrievalWorker(retriever, on_update=updates.append)
    worker.submit(revision=4, query="运行中", kind="partial")
    retriever.wait_started(1)
    worker.submit(revision=5, query="旧 partial", kind="partial")
    worker.submit(revision=5, query="最终问题", kind="final")
    retriever.release()
    worker.close()
    assert updates[-1].kind == "final"
    assert updates[-1].retrieval.query == "最终问题"
```

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_latest_wins_retrieval.py -q`

Expected: FAIL because the worker does not exist.

- [ ] **Step 4: Implement priority queue semantics**

Use a `PriorityQueue` ordered by final-first, then newest revision. Before running a partial, compare against the latest submitted revision and return without retrieving if stale. A running retrieval cannot be interrupted because TF-IDF is short.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_latest_wins_retrieval.py -q
.venv/bin/python -m pytest tests/test_rag_pipeline.py -q
```

Expected: all pass.

---

### Task 3: MiniCPM Live AUDIO Owner

**Files:**
- Modify: `official_runtime.py`
- Modify: `tests/test_official_runtime.py`
- Create: `tests/test_stream_live_runtime.py`

**Interfaces:**
- Consumes: an audio queue yielding `np.ndarray` PCM units and `None` as end sentinel; a `concurrent.futures.Future[RetrievalResult]`; a `threading.Event` cancellation flag.
- Produces:
  - `OfficialDuplexRuntime.stream_live_answer(*, audio_queue, retrieval_future, cancel_event, on_ready, on_event=None, timeout_s=120.0, wait_unit_s=1.0)`

- [ ] **Step 1: Write failing test that consumes queued AUDIO before retrieval finishes**

```python
def test_live_runtime_consumes_all_audio_before_waiting_for_retrieval(fake_runtime):
    audio_queue = queue.Queue()
    for _ in range(2):
        audio_queue.put(np.zeros(16000, np.float32))
    audio_queue.put(None)
    retrieval_future = Future()
    events = []
    stream = fake_runtime.stream_live_answer(
        audio_queue=audio_queue,
        retrieval_future=retrieval_future,
        cancel_event=threading.Event(),
        on_ready=lambda result: None,
        on_event=lambda kind, **data: events.append(kind),
    )
    next(stream)
    assert sum(e == "audio.listen" for e in events) >= 2
    assert not retrieval_future.done()
```

- [ ] **Step 2: Add failing tests for zero padding, cancellation and timeout**

Assert:

- a short final unit is padded to exactly 16000 samples;
- `cancel_event.set()` stops waiting with `CancelledError`;
- retrieval timeout raises `TimeoutError`;
- injection occurs only after all queued audio and final retrieval are complete.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_stream_live_runtime.py -q`

Expected: FAIL because `stream_live_answer` does not exist.

- [ ] **Step 4: Implement `stream_live_answer`**

Inside the existing `_turn()` context, call `_prepare()`, then:

1. `audio_queue.get(timeout=0.25)` until `None`.
2. Pad a short audio unit to 16000 samples.
3. Call `_listen_unit(..., phase="input", on_event=...)`.
4. While retrieval is not ready, send paced silence AUDIO units.
5. Call `on_ready(retrieval_future.result())`.
6. Reuse `_stream_response(build_rag_prompt(...))`.

The method must check `cancel_event` while waiting.

- [ ] **Step 5: Run runtime and full existing tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_stream_live_runtime.py tests/test_official_runtime.py -q
.venv/bin/python -m pytest tests test_evaluation.py -q
```

Expected: all pass.

---

### Task 4: Streaming Turn Coordinator

**Files:**
- Modify: `streaming_audio_rag.py`
- Create: `tests/test_streaming_audio_rag.py`

**Interfaces:**
- Consumes: `StreamingASRWorker`, `LatestWinsRetrievalWorker`, `OfficialDuplexRuntime.stream_live_answer`, `RAGRetriever`, `RetrievalConfig`.
- Produces:
  - `StreamingAudioRAGPipeline(runtime, asr_backend, knowledge_path, retrieval_config, *, timeout_s=120.0)`
  - `StreamingAudioRAGPipeline.warmup()`
  - `StreamingAudioRAGPipeline.start_turn(on_event) -> LiveTurn`
  - `LiveTurn.append_audio(samples: np.ndarray, *, final: bool = False)`
  - `LiveTurn.end_input()`
  - `LiveTurn.cancel()`
  - `LiveTurn.wait(timeout: float | None = None)`

- [ ] **Step 1: Write failing fan-out test**

```python
def test_turn_fans_one_second_pcm_to_minicpm_and_asr(fake_pipeline):
    events = []
    turn = fake_pipeline.start_turn(events.append)
    pcm = np.ones(16000, np.float32)
    turn.append_audio(pcm)
    assert fake_pipeline.runtime.audio_queue.get_nowait().shape == (16000,)
    assert fake_pipeline.asr.frames.get_nowait().shape == (16000,)
```

- [ ] **Step 2: Write failing async-overlap and final-authority tests**

Using blocking fakes, assert:

- MiniCPM receives a second PCM unit while ASR is blocked on the first;
- ASR can emit a later partial while retrieval is blocked;
- `retrieval.partial` never enables injection;
- final retrieval sets the coordinator future and injection starts only after all PCM is consumed.

- [ ] **Step 3: Add failing lifecycle, backpressure and metric tests**

Assert:

- `start_turn()` resets the MiniCPM session before the first turn when isolation is enabled;
- a full MiniCPM or ASR queue produces `backpressure`, not unbounded memory growth;
- `no_speech` produces no generation;
- stale turn callbacks are ignored after cancel;
- all event `t_ms` values use the first PCM arrival as their single origin.

- [ ] **Step 4: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_streaming_audio_rag.py -q`

Expected: FAIL because the coordinator does not exist.

- [ ] **Step 5: Implement turn orchestration**

Create one `RAGRetriever` per turn from the fixed knowledge path and retrieval config. Start ASR and retrieval workers, create the MiniCPM queue and retrieval future, and run the runtime generator in a dedicated coordinator thread. Event callbacks push normalized dictionaries to `on_event` in another thread.

- [ ] **Step 6: Run focused and existing tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_streaming_audio_rag.py -q
.venv/bin/python -m pytest tests test_evaluation.py -q
```

Expected: all pass.

---

### Task 5: FastAPI WebSocket Server

**Files:**
- Create: `requirements-console.txt`
- Create: `console_server.py`
- Create: `tests/test_console_server.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `StreamingAudioRAGPipeline`.
- Produces:
  - `create_app(*, knowledge_path, backend_factory, runtime_factory, output_dir) -> FastAPI`
  - `GET /api/health`
  - `WebSocket /ws/console`
  - `main() -> int`

- [ ] **Step 1: Add console dependencies**

Create `requirements-console.txt`:

```text
fastapi>=0.115,<1
uvicorn[standard]>=0.30,<1
websockets>=13,<16
```

Add `-r requirements-console.txt` to `requirements.txt`.

- [ ] **Step 2: Write failing WebSocket protocol tests**

Use `fastapi.testclient.TestClient` and a fake pipeline:

```python
def test_websocket_rejects_audio_before_input_start(client):
    with client.websocket_connect("/ws/console") as ws:
        ws.receive_json()  # session.ready
        ws.send_json({
            "type": "input.audio.append",
            "sequence": 0,
            "audio": base64_pcm(np.zeros(16000, np.float32)),
        })
        event = ws.receive_json()
        assert event["type"] == "turn.error"
        assert event["code"] == "input_not_started"
```

Add tests for wrong PCM length, invalid base64, 30-second limit, busy turn, health,
disconnect cleanup and `input.end`.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_console_server.py -q`

Expected: FAIL because `console_server.py` does not exist.

- [ ] **Step 4: Implement the WebSocket bridge**

The WebSocket writer must use a thread-safe `asyncio.Queue` fed by pipeline callbacks.
Keep exactly one active turn. Decode base64 into contiguous float32 before queueing and
validate finite values. Never write PCM.

- [ ] **Step 5: Run API and full tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_console_server.py -q
.venv/bin/python -m pytest tests test_evaluation.py -q
```

Expected: all pass.

---

### Task 6: Browser Console and AudioWorklet

**Files:**
- Create: `static/console/index.html`
- Create: `static/console/console.css`
- Create: `static/console/console.js`
- Create: `static/console/audio-stream.js`
- Create: `static/console/capture-worklet.js`
- Modify: `console_server.py`

**Interfaces:**
- Consumes WebSocket events from Task 5.
- Produces the selected A layout with expandable B latency timeline.

**Review focus tests:** short final chunk, non-16 kHz browser context, WebSocket close, audio queue underrun, and no media file creation.

- [ ] **Step 1: Add a static smoke test**

Create `tests/test_console_static.py` that asserts:

- `index.html` contains microphone, evidence, answer and latency regions;
- `capture-worklet.js` defaults `chunkSize` to 16000;
- console JS contains no `MediaRecorder`;
- `audio-stream.js` contains no WAV creation or file download code.

- [ ] **Step 2: Run the smoke test and verify failure**

Run: `.venv/bin/python -m pytest tests/test_console_static.py -q`

Expected: FAIL because static files do not exist.

- [ ] **Step 3: Implement AudioWorklet capture**

Accumulate exactly `16000` float32 samples. On stop, post the remaining samples with
`final: true`. Transfer the sample buffer instead of copying when possible.

- [ ] **Step 4: Implement browser audio transport and playback**

`audio-stream.js` must:

- request mono microphone audio;
- create an AudioContext and resample to 16 kHz when necessary;
- convert float32 chunks to base64 for JSON WebSocket messages;
- send `input.start`, `input.audio.append` and `input.end`;
- decode 24 kHz output PCM into AudioBuffers;
- schedule contiguous playback with a 100 ms initial lead;
- keep answer PCM only in memory;
- never import or instantiate `MediaRecorder`.

- [ ] **Step 5: Implement the control console**

Use the approved visual direction: dark graphite controls, warm paper evidence area,
teal success/signal, orange active/injection. Render:

- left stage rail;
- center transcript, answer and in-memory audio controls;
- right evidence cards and raw evidence;
- collapsible horizontal latency trace;
- ASR partial, speculative retrieval and final retrieval states.

- [ ] **Step 6: Serve static files and run tests**

Mount `/static` and `/console`. Run:

```bash
.venv/bin/python -m pytest tests/test_console_static.py tests/test_console_server.py -q
.venv/bin/python -m pytest tests test_evaluation.py -q
```

Expected: all pass.

- [ ] **Step 7: Verify browser-specific audio edge cases**

In a browser debug page, verify:

- an AudioContext reporting 48 kHz still emits 16000-sample chunks after resampling;
- releasing the microphone after a non-integer number of seconds sends a short final chunk;
- closing or refreshing the page stops microphone tracks and sends no further PCM;
- an empty/short output queue resumes playback without creating a local file.

Record the observed values in the implementation handoff.

---

### Task 7: Real Audio and RAG Verification

**Files:**
- Create: `tests/integration_console_stream.py`
- Modify: `README.md`
- Modify: `ENVIRONMENT.md`

**Interfaces:**
- Consumes: running Qwen ASR HTTP service and `console_server.py`.
- Produces reproducible manual and automated verification evidence.

- [ ] **Step 1: Write integration generator**

`tests/integration_console_stream.py` reads an existing WAV only as a test fixture,
converts it to float32 in memory, sends 16000-sample PCM chunks over WebSocket,
asserts event ordering and overlap, and asserts the console output directory contains
no audio files.

- [ ] **Step 2: Run against fake services and verify failure**

Run: `.venv/bin/python -m pytest tests/integration_console_stream.py -q --collect-only`

Expected: collection succeeds; integration body is not collected by default.

- [ ] **Step 3: Run real Qwen + MiniCPM verification**

Start Qwen:

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 VLLM_PORT=18580 \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
.venv-qwen/bin/python -m qwen_asr.cli.demo_streaming \
  --asr-model-path "$(pwd)/models/Qwen3-ASR-1.7B" \
  --host 127.0.0.1 --port 18180 \
  --chunk-size-sec 1.0 --unfixed-chunk-num 2 --unfixed-token-num 5
```

Start console:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -u console_server.py \
  --knowledge knowledge/squad_zh_25.json \
  --full-text --host 127.0.0.1 --port 18200
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/integration_console_stream.py \
  --url ws://127.0.0.1:18200/ws/console \
  --audio "/root/paddlejob/workspace/env_run/tests/cases/common/user_audio/当出现植物大战僵尸的时候提醒我.wav" \
  --output outputs/console_integration
```

Expected: `asr.partial`, `retrieval.partial`, `retrieval.final`, `rag.injected`,
`answer.text.delta`, `answer.audio.delta` and `answer.done` occur in order;
ASR/MiniCPM overlap is greater than zero; no audio file is created.

- [ ] **Step 4: Run direct squad retrieval and browser manual check**

Open `http://127.0.0.1:18200/console`, hold the microphone button, ask a
`squad_zh_25.json` question, and verify:

- 1-second PCM chunks reach the server;
- ASR partials update while speaking;
- speculative retrieval updates;
- final ASR produces authoritative retrieval;
- answer text and audio appear together;
- the latency panel shows all stages;
- no microphone recording is saved.

- [ ] **Step 5: Run all tests**

Run:

```bash
.venv/bin/python -m pytest tests test_evaluation.py -q
.venv/bin/python -m compileall -q async_audio_rag.py official_runtime.py streaming_audio_rag.py console_server.py
```

Expected: all tests pass and compileall exits 0.

- [ ] **Step 6: Update documentation**

Document console launch, Qwen service dependency, HTTPS/localhost microphone requirement,
isolated-session behavior, no-recording guarantee, event protocol and known limitations.
