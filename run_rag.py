#!/usr/bin/env python3
"""Text/file-audio RAG CLI using only the official full-duplex text_list entry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from official_runtime import OfficialDuplexRuntime
from rag_pipeline import RAGPipeline, RAGRetriever, RetrievalConfig

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def transcribe_file(path, model_path, device):
    import librosa
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    audio, _ = librosa.load(path, sr=16000, mono=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("audio file must contain finite samples")
    if audio.size > 30 * 16000:
        raise ValueError("audio file must be at most 30 seconds (segmented ASR)")
    start = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).eval().to(device)
    model.generation_config.forced_decoder_ids = None
    batch = processor(
        audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    ).to(device)
    with torch.inference_mode():
        ids = model.generate(
            batch.input_features.to(model.dtype), attention_mask=batch.attention_mask,
            language="zh", task="transcribe", do_sample=False, max_new_tokens=128,
        )
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    metadata = {
        "input_audio": str(Path(path).resolve()), "transcript": text,
        "model": str(Path(model_path).resolve()), "device": device,
        "wall_s": time.perf_counter() - start,
    }
    del model, processor, batch, ids
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not text:
        raise ValueError("ASR returned an empty transcript; no retrieval performed")
    return text, audio, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query", action="append", help="repeat for serial conversation turns")
    inputs.add_argument("--audio", type=Path, help="mono/file audio, at most 30 seconds")
    inputs.add_argument("--interactive", action="store_true")
    parser.add_argument("--knowledge", type=Path, default=ROOT / "knowledge/sample.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--min-shared-bigrams", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=900)
    parser.add_argument("--retrieve-only", action="store_true", help="never loads the MiniCPM model")
    parser.add_argument("--model-path", type=Path, default=ROOT / "model")
    parser.add_argument("--ref-audio", type=Path, default=ROOT / "assets/ref_audio.wav")
    parser.add_argument("--asr-model", type=Path, default=ROOT / "models/whisper-small")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--max-units", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = RetrievalConfig(
            args.top_k, args.min_score, args.max_evidence_chars, args.min_shared_bigrams
        )
        runtime = OfficialDuplexRuntime(
            model_path=args.model_path, ref_audio_path=args.ref_audio,
            max_units=args.max_units, seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    out = (args.output or ROOT / "outputs/rag" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        parser.error(f"{out} already contains results.json; choose a new --output")
    retriever = RAGRetriever(args.knowledge, config)
    pipeline = RAGPipeline(retriever, runtime)
    document = {
        "status": "running", "knowledge_path": str(retriever.path),
        "config": vars(config), "retrieve_only": args.retrieve_only,
        "model_path": str(args.model_path.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_load_s": None, "cases": [],
    }
    write_json(out / "results.json", document)
    exit_code = 0
    events = (out / "events.jsonl").open("w", encoding="utf-8")

    def emit(kind, **payload):
        events.write(json.dumps({
            "type": kind, "time": datetime.now(timezone.utc).isoformat(), **payload
        }, ensure_ascii=False) + "\n")
        events.flush()

    def run_turn(query, *, audio=None, asr=None):
        name = f"turn_{len(document['cases']) + 1:03d}"
        turn = {"name": name, "query": query, "wav": None, "asr": asr}
        document["cases"].append(turn)
        writer = None
        audio_samples = 0
        start = time.perf_counter()

        def on_chunk(chunk):
            nonlocal writer, audio_samples
            wave = chunk["audio"]
            if wave is not None and wave.size:
                if writer is None:
                    import soundfile as sf
                    writer = sf.SoundFile(
                        str(out / f"{name}.wav"), mode="w", samplerate=24000,
                        channels=1, subtype="FLOAT",
                    )
                writer.write(wave)
                writer.flush()
                audio_samples += int(wave.size)
            if chunk["text"]:
                print(chunk["text"], end="", flush=True)
            emit("answer.chunk", turn=name, **{
                k: v for k, v in chunk.items() if k != "audio"
            })

        try:
            retrieval = retriever.retrieve(query)
            turn["retrieval"] = retrieval.to_dict()
            citations = retrieval.to_dict()["citations"]
            emit("retrieval.result", turn=name, **retrieval.to_dict())
            print(f"\n[{name}] {query}", flush=True)
            print("[retrieval]", json.dumps(citations, ensure_ascii=False), flush=True)
            if args.retrieve_only:
                turn.update({
                    "status": "retrieved" if retrieval.usable else "no_evidence",
                    "text": "", "citations": citations, "model_called": False,
                })
            else:
                if retrieval.usable and runtime.load_s is None:
                    import torch
                    torch.set_num_threads(4)
                    print("[model] loading local official MiniCPM-o...", flush=True)
                turn.update(pipeline.answer_retrieved(
                    retrieval, audio=audio, on_chunk=on_chunk
                ))
                if turn["status"] == "no_evidence":
                    print(turn["text"], flush=True)
                else:
                    print(flush=True)
                if turn["status"] == "incomplete":
                    print("[incomplete] model did not finish a nonempty answer", file=sys.stderr)
            turn["audio"] = {
                "samples": audio_samples, "sample_rate": 24000,
                "duration_s": audio_samples / 24000,
            }
            turn["wav"] = str(out / f"{name}.wav") if writer is not None else None
            turn["wall_s"] = time.perf_counter() - start
            emit("answer.done", turn=name, status=turn["status"], wav=turn["wav"])
        except Exception as exc:
            turn.update({"status": "error", "error": str(exc)})
            emit("error", turn=name, message=str(exc))
            raise
        finally:
            if writer is not None:
                writer.close()
            document["model_load_s"] = runtime.load_s
            write_json(out / f"{name}.json", turn)
            write_json(out / "results.json", document)

    try:
        if args.audio:
            query, audio, metadata = transcribe_file(
                args.audio, args.asr_model, args.asr_device
            )
            emit("asr.final", **metadata)
            print("[asr]", query, flush=True)
            run_turn(query, audio=audio, asr=metadata)
        elif args.interactive:
            print("输入问题；/reset 重置模型会话，/quit 退出。知识文件每轮检查更新。", flush=True)
            while True:
                try:
                    query = input("问题> ").strip()
                except EOFError:
                    break
                if query == "/quit":
                    break
                if query == "/reset":
                    runtime.reset_session()
                    emit("session.reset")
                    continue
                if query:
                    run_turn(query)
        else:
            for query in args.query:
                run_turn(query)
        document["status"] = "completed"
        if any(c["status"] == "incomplete" for c in document["cases"]):
            exit_code = 2
    except KeyboardInterrupt:
        document["status"] = "interrupted"
        exit_code = 130
    except Exception as exc:
        document.update({"status": "error", "error": str(exc)})
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        runtime.close()
        events.close()
        write_json(out / "results.json", document)
    print("[saved]", out / "results.json", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
