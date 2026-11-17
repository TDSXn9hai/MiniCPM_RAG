"""Background ASR + retrieval while a single owner drives MiniCPM AUDIO units."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import threading
import time

import numpy as np
from rag_pipeline import REFUSAL


@dataclass(frozen=True)
class PreparedAudioQuery:
    retrieval: object
    asr: dict
    timings: dict


class LocalWhisperASR:
    """Resident Whisper owned by one worker and its own CUDA stream."""

    def __init__(self, model_path, device="cuda:0"):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self._model = self._processor = self._stream = None

    def prepare(self):
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if not self.device.startswith("cuda:"):
            raise ValueError("parallel audio ASR requires the same CUDA device as MiniCPM")
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self._stream):
                self._processor = WhisperProcessor.from_pretrained(
                    str(self.model_path), local_files_only=True)
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    str(self.model_path), local_files_only=True, torch_dtype=torch.float16,
                ).eval().to(self.device)
                self._model.generation_config.forced_decoder_ids = None
            self._stream.synchronize()

    def transcribe(self, audio):
        import torch
        self.prepare()
        started = time.perf_counter()
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream), torch.inference_mode():
            batch = self._processor(audio, sampling_rate=16000, return_tensors="pt",
                                    return_attention_mask=True).to(self.device)
            ids = self._model.generate(
                batch.input_features.to(self._model.dtype), attention_mask=batch.attention_mask,
                language="zh", task="transcribe", do_sample=False, max_new_tokens=128)
            self._stream.synchronize()
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError("ASR returned an empty transcript; no retrieval performed")
        return text, {"model": str(self.model_path), "device": self.device,
                      "wall_s": time.perf_counter() - started,
                      "cuda_stream": int(self._stream.cuda_stream)}

    def close(self):
        if self._stream is not None:
            self._stream.synchronize()
        self._model = self._processor = self._stream = None
        gc.collect()


class Qwen3ASRStreaming:
    """Adapter for the official Qwen3-ASR vLLM streaming state API.

    The backend object is intentionally injectable so the orchestration can be
    tested without downloading Qwen weights or starting vLLM.  It must expose:
    ``init_streaming_state``, ``streaming_transcribe`` and
    ``finish_streaming_transcribe``.
    """

    def __init__(self, backend, *, language="Chinese", chunk_size_sec=2.0,
                 unfixed_chunk_num=2, unfixed_token_num=5, device="cuda:0"):
        if not device.startswith("cuda:"):
            raise ValueError("Qwen3-ASR streaming requires a CUDA device")
        if chunk_size_sec <= 0:
            raise ValueError("chunk_size_sec must be positive")
        self.backend = backend
        self.device = device
        self.language = language
        self.chunk_size_sec = chunk_size_sec
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.state = None

    def prepare(self):
        prepare = getattr(self.backend, "prepare", None)
        if prepare is not None:
            prepare()

    def transcribe_stream(self, audio):
        if self.state is None:
            self.state = self.backend.init_streaming_state(
                context="", language=self.language,
                unfixed_chunk_num=self.unfixed_chunk_num,
                unfixed_token_num=self.unfixed_token_num,
                chunk_size_sec=self.chunk_size_sec,
            )
        audio = np.asarray(audio, np.float32)
        step = max(1, int(self.chunk_size_sec * 16000))
        partials = []
        started = time.perf_counter()
        for offset in range(0, len(audio), step):
            update = self.backend.streaming_transcribe(
                audio[offset:offset + step], self.state
            )
            if update:
                partials.append(self._result_text(update))
        final = self.backend.finish_streaming_transcribe(self.state)
        text = self._result_text(final)
        if not text:
            raise ValueError("Qwen3-ASR returned an empty final transcript")
        return text, {
            "backend": "qwen3-asr-vllm-streaming",
            "device": self.device,
            "chunk_size_sec": self.chunk_size_sec,
            "partial_updates": len(partials),
            "last_partial": partials[-1] if partials else "",
            "wall_s": time.perf_counter() - started,
        }

    def transcribe(self, audio):
        return self.transcribe_stream(audio)

    def close(self):
        self.state = None
        close = getattr(self.backend, "close", None)
        if close:
            close()

    @staticmethod
    def _result_text(result):
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(getattr(result, "text", "")).strip()


class Qwen3ASRHTTPBackend:
    """Client for the official Qwen3-ASR streaming Flask API."""

    def __init__(self, base_url, *, timeout_s=120.0):
        if timeout_s <= 0:
            raise ValueError("ASR HTTP timeout must be positive")
        import httpx

        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)

    def _raise(self, response):
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3-ASR service returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ) from exc
        return response.json()

    def prepare(self):
        response = self._client.get("/")
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3-ASR service health check failed with HTTP "
                f"{response.status_code}"
            ) from exc

    def init_streaming_state(
        self, *, context="", language="Chinese", unfixed_chunk_num=2,
        unfixed_token_num=5, chunk_size_sec=2.0,
    ):
        # The packaged Qwen demo accepts session creation without query
        # parameters. Keep the adapter signature explicit so a stricter
        # service can add language/context handling without changing callers.
        del context, language, unfixed_chunk_num, unfixed_token_num, chunk_size_sec
        return self._raise(self._client.post("/api/start"))

    def streaming_transcribe(self, audio, state):
        audio = np.asarray(audio, np.float32)
        payload = self._raise(self._client.post(
            "/api/chunk",
            params={"session_id": state["session_id"]},
            content=audio.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        ))
        state.update(payload)
        return payload

    def finish_streaming_transcribe(self, state):
        payload = self._raise(self._client.post(
            "/api/finish",
            params={"session_id": state["session_id"]},
        ))
        state.update(payload)
        return payload

    def cancel_streaming_transcribe(self, state):
        """Best-effort server-side cleanup for a canceled live ASR turn."""
        session_id = state.get("session_id") if isinstance(state, dict) else None
        if not session_id:
            return
        try:
            self._client.post(
                "/api/finish",
                params={"session_id": session_id},
                timeout=5.0,
            )
        except Exception:
            pass

    def close(self):
        self._client.close()


class ASRRetrievalWorker:
    """Dependent ASR→retrieval tasks run off the MiniCPM owner thread."""

    def __init__(self, asr, retriever):
        self.asr = asr
        self.retriever = retriever
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-rag")
        self._closed = False

    def prepare(self):
        return self._executor.submit(self.asr.prepare)

    def submit(self, audio, source=None):
        if self._closed:
            raise RuntimeError("ASR/retrieval worker is closed")
        snapshot = np.asarray(audio, np.float32).copy()
        return self._executor.submit(self._run, snapshot, source)

    def _run(self, audio, source):
        started = time.perf_counter()
        query, metadata = self.asr.transcribe(audio)
        asr_finished = time.perf_counter()
        retrieval = self.retriever.retrieve(query)
        finished = time.perf_counter()
        return PreparedAudioQuery(
            retrieval, {**metadata, "transcript": query, "input_audio": source},
            {"asr_started_s": started, "asr_finished_s": asr_finished,
             "retrieval_started_s": asr_finished, "retrieval_finished_s": finished,
             "worker_thread_id": threading.get_ident()},
        )

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                # A running CUDA generate cannot be safely killed; retire it
                # before freeing resident weights or starting another session.
                self._executor.submit(self.asr.close).result()
            finally:
                self._executor.shutdown(wait=True, cancel_futures=True)


class AsyncAudioRAGPipeline:
    def __init__(self, worker, runtime, *, timeout_s=120.0, wait_unit_s=1.0):
        asr_device = str(getattr(worker.asr, "device", runtime.device))
        if asr_device == "cuda":
            asr_device = "cuda:0"
        if asr_device != runtime.device:
            raise ValueError("ASR and MiniCPM must use the same CUDA device")
        if timeout_s <= 0 or wait_unit_s <= 0:
            raise ValueError("retrieval timeout and wait unit interval must be positive")
        self.worker = worker
        self.runtime = runtime
        self.timeout_s = timeout_s
        self.wait_unit_s = wait_unit_s
        self._ready = False
        self._busy = threading.Lock()

    def warmup(self):
        if not self._ready:
            # Transformers constructors use process-global initialization/meta
            # contexts: construct sequentially, then overlap inference only.
            self.worker.prepare().result()
            self.runtime.warmup()
            self._ready = True

    def answer(self, audio, *, source=None, on_chunk=None, on_event=None):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("audio pipeline is already processing a request")
        future = stream = None
        try:
            audio = self.runtime._validate_audio(audio)
            self.warmup()
            prepared = None
            events, units, text = [], [], []

            def emit(kind, **payload):
                events.append({"type": kind, **payload})
                if on_event:
                    on_event(kind, **payload)

            def ready(value):
                nonlocal prepared
                prepared = value
                emit("asr.final", **value.asr, **value.timings)
                emit("retrieval.result", **value.retrieval.to_dict())

            future = self.worker.submit(audio, source)
            emit("backend.submitted", submitted_s=time.perf_counter())
            stream = self.runtime.stream_pending_answer(
                audio=audio, retrieval_future=future, on_ready=ready, on_event=emit,
                timeout_s=self.timeout_s, wait_unit_s=self.wait_unit_s)
            ended = False
            for chunk in stream:
                if on_chunk:
                    on_chunk(chunk)
                text.append(chunk.get("text") or "")
                ended = bool(chunk["end_of_turn"])
                units.append({k: v for k, v in chunk.items() if k != "audio"})
            if prepared is None:
                raise RuntimeError("audio runtime ended without a retrieval result")
            retrieval = prepared.retrieval
            no_evidence = not retrieval.usable
            joined = "".join(text)
            return {
                "query": retrieval.query, "retrieval": retrieval.to_dict(),
                "citations": [asdict(c) for c in retrieval.evidence.citations],
                "asr": prepared.asr, "background": prepared.timings,
                "runtime_events": events, "units": units,
                "text": joined,
                "status": "answered" if ended and joined.strip() else "incomplete",
                "no_evidence": no_evidence,
                "application_refusal_enabled": False,
                "ended_naturally": ended,
                "model_called": True, "generation_called": True,
            }
        finally:
            if stream is not None:
                stream.close()
            if future is not None:
                future.cancel()
            self._busy.release()
