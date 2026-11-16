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
                "text": REFUSAL if no_evidence else joined,
                "status": "no_evidence" if no_evidence else
                          ("answered" if ended and joined.strip() else "incomplete"),
                "ended_naturally": True if no_evidence else ended,
                "model_called": True, "generation_called": not no_evidence,
            }
        finally:
            if stream is not None:
                stream.close()
            if future is not None:
                future.cancel()
            self._busy.release()
