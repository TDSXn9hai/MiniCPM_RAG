#!/usr/bin/env python3
"""Exercise the unmodified official full-duplex text_list interface offline.

All source and weights are local. Only the explicitly named refreshed-logits
control changes runtime behavior; the official model files are never patched.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from evaluation import amount_matches, summarize_audio

ROOT = Path(__file__).resolve().parent
SYSTEM = (
    "Streaming Audio Conversation. 你是一名中文知识助手。"
    "请依据用户提供的参考资料回答。参考资料不足时说明无法确认。"
    "每次只回答一句简短完整的话，不要复述问题，不要朗读资料标题。"
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prompt_for(amount):
    return (
        f"参考资料：上海出差的住宿报销上限为每晚{amount}元。\n"
        "用户问题：上海出差住宿每晚最多能报销多少钱？\n"
        "请只根据这份资料，用一句话直接回答金额。"
    )


class PrefillTrace:
    """Observe actual forward passes without changing their inputs or outputs."""

    def __init__(self, duplex):
        self.duplex = duplex
        self.active = False
        self.calls = []
        self.last_logits = None
        self._last_hidden = None
        self.original_feed = duplex.decoder.feed
        self.hook = duplex.model.llm.register_forward_hook(self._on_forward)
        duplex.decoder.feed = self._feed

    def _on_forward(self, module, args, output):
        if self.active and output.hidden_states is not None:
            # Keep the full current feed shape. BF16 lm_head GEMM on just the
            # last row can round differently from the official all-row call.
            self._last_hidden = output.hidden_states[-1].detach()

    def _feed(self, embeds, return_logits=False):
        before = self.duplex.decoder.get_cache_length() if self.active else None
        result = self.original_feed(embeds, return_logits=return_logits)
        if self.active:
            self.calls.append({
                "tokens": int(embeds.shape[0]),
                "return_logits": return_logits,
                "cache_before": before,
                "cache_after": self.duplex.decoder.get_cache_length(),
            })
        return result

    @torch.inference_mode()
    def prefill(self, *, refresh=False, **kwargs):
        self.calls = []
        self._last_hidden = None
        self.last_logits = None
        self.active = True
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            result = self.duplex.streaming_prefill(**kwargs)
        finally:
            self.active = False
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        diagnostics = {
            "cost_observed_ms": elapsed,
            "feed_calls": self.calls,
            "current_mode": self.duplex.current_mode,
            "audio_chunk_idx": self.duplex.audio_chunk_idx,
            "pending_logits_from_final_feed": bool(
                self.calls and self.calls[-1]["return_logits"]
            ),
        }
        pending = self.duplex.pending_logits
        if result["success"] and self._last_hidden is not None and pending is not None:
            # Same hidden state and lm_head as decoder.feed(return_logits=True),
            # without adding a token, another forward pass or another KV entry.
            self.last_logits = self.duplex.model.llm.lm_head(self._last_hidden)[:, -1]
            a, b = pending.float(), self.last_logits.float()
            diagnostics.update({
                "pending_equals_last_feed_logits": bool(torch.equal(pending, self.last_logits)),
                "max_abs_logit_difference": float((a - b).abs().max()),
                "pending_top_token": int(a.argmax(-1).item()),
                "last_feed_top_token": int(b.argmax(-1).item()),
                "pending_top_token_text": self.duplex.tokenizer.decode([int(a.argmax(-1).item())]),
                "last_feed_top_token_text": self.duplex.tokenizer.decode([int(b.argmax(-1).item())]),
            })
            if refresh:
                self.duplex.pending_logits = self.last_logits
        diagnostics["refreshed_logits_control"] = refresh
        return {"result": result, "diagnostics": diagnostics}

    def close(self):
        self.duplex.decoder.feed = self.original_feed
        self.hook.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-audio", type=Path, default=ROOT / "assets/user_audio.wav",
                        help="audio fixture for audio_* diagnostic cases; supply your own WAV")
    parser.add_argument("--cases", default="text_600,text_437,no_evidence,ongoing_437,mixed_437,mixed_refreshed_437,multiturn,audio_only,audio_with_evidence,audio_then_text")
    parser.add_argument("--max-units", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "isolated"))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, out / "run_probe.executed.py")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    logging_path = out / "results.json"
    document = {
        "status": "loading",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "seed": args.seed,
        "model_path": str(ROOT / "model"),
        "model_source_sha256": hashlib.sha256((ROOT / "model/modeling_minicpmo.py").read_bytes()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system_prompt": SYSTEM,
        "original_source_modified": False,
        "case_isolation": "model.reset_session(reset_token2wav_cache=True) then duplex.prepare",
        "cases": [],
    }
    write_json(logging_path, document)
    print("[probe] loading local official model", flush=True)
    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(ROOT / "model"),
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    duplex = model.as_duplex(force_listen_count=0, sliding_window_mode="off")
    print("[probe] loaded", round(time.perf_counter() - start, 2), inspect.signature(duplex.streaming_prefill), flush=True)
    document["load_s"] = time.perf_counter() - start
    document["text_list_signature"] = str(inspect.signature(duplex.streaming_prefill))
    document["status"] = "running"
    ref_audio, _ = librosa.load(ROOT / "assets/ref_audio.wav", sr=16000, mono=True)
    trace = PrefillTrace(duplex)

    def prepare():
        seed_all(args.seed)
        # Official duplex.prepare clears the decoder and streaming processor,
        # but not model.audio_past_key_values. Reset the underlying model using
        # its public API so independent cases do not share audio encoder KV.
        model.reset_session(reset_token2wav_cache=True)
        duplex.prepare(
            prefix_system_prompt=SYSTEM,
            ref_audio=ref_audio,
            prompt_wav_path=str(ROOT / "assets/ref_audio.wav"),
        )

    def collect(case, *, name, first_result=None):
        units, texts, audio_parts = [], [], []
        wall_start = time.perf_counter()
        first_text_ms = first_audio_ms = None
        for i in range(args.max_units):
            if i > 0:
                prefill = trace.prefill(audio_waveform=np.zeros(16000, np.float32))
                if not prefill["result"]["success"]:
                    raise RuntimeError(prefill)
            else:
                prefill = first_result
            torch.cuda.synchronize()
            unit_start = time.perf_counter()
            generated_ids_before = len(duplex.total_ids)
            result = duplex.streaming_generate(
                decode_mode="greedy", max_new_speak_tokens_per_chunk=20
            )
            torch.cuda.synchronize()
            generation_ms = (time.perf_counter() - unit_start) * 1000
            text = result.get("text", "")
            audio = result.get("audio_waveform")
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().float().cpu().numpy()
            audio = np.asarray(audio, np.float32).reshape(-1) if audio is not None else np.zeros(0, np.float32)
            unit = {
                "unit": i, "text": text,
                "first_generated_token": duplex.tokenizer.decode(
                    duplex.total_ids[generated_ids_before:generated_ids_before + 1]
                ),
                "is_listen": bool(result["is_listen"]),
                "end_of_turn": bool(result["end_of_turn"]),
                "current_time": result["current_time"],
                "n_tokens": result["n_tokens"],
                "n_tts_tokens": result["n_tts_tokens"],
                "generation_ms": generation_ms,
                "prefill": prefill,
                "audio": summarize_audio(audio, 24000),
            }
            units.append(unit)
            if text:
                texts.append(text)
                if first_text_ms is None:
                    first_text_ms = (time.perf_counter() - wall_start) * 1000
            # Official listen results contain placeholder silence. Keep only
            # speak waveforms in the concatenated utterance for independent ASR.
            if not result["is_listen"] and audio.size:
                audio_parts.append(audio)
                if first_audio_ms is None and unit["audio"]["non_silent"]:
                    first_audio_ms = (time.perf_counter() - wall_start) * 1000
            print(f"[{name}] unit={i} listen={result['is_listen']} end={result['end_of_turn']} text={text!r} audio={audio.size} gen_ms={generation_ms:.0f}", flush=True)
            if result["end_of_turn"]:
                break
            # Pure text-only empty output should get several normal audio units,
            # not an experimental force-speak injection.
            if result["is_listen"] and i >= 3 and not texts:
                break
            if result["is_listen"] and texts:
                break
        waveform = np.concatenate(audio_parts) if audio_parts else np.zeros(0, np.float32)
        wav_path = out / f"{name}.wav"
        if waveform.size:
            sf.write(wav_path, waveform, 24000, subtype="FLOAT")
        joined = "".join(texts)
        case.update({
            "text": joined, "units": units, "audio": summarize_audio(waveform, 24000),
            "wav": str(wav_path) if waveform.size else None,
            "generation_wall_s": time.perf_counter() - wall_start,
            "first_text_generation_ms": first_text_ms,
            "first_audio_generation_ms": first_audio_ms,
            "ended_naturally": bool(units[-1]["end_of_turn"]),
            "hit_unit_cap": len(units) == args.max_units and not units[-1]["end_of_turn"],
            "kv_length_after": duplex.decoder.get_cache_length(),
        })
        if "expected_amount" in case:
            case["text_has_expected_amount"] = amount_matches(joined, case["expected_amount"])
            case["text_has_other_test_amount"] = any(
                amount_matches(joined, n) for n in (600, 437, 862)
                if n != case["expected_amount"]
            )
        return case

    try:
        for name in args.cases.split(","):
            seed_all(args.seed)
            case = {"name": name}
            document["cases"].append(case)
            print("[probe] CASE", name, flush=True)
            try:
                duplex.force_listen_count = 0
                prepare()
                if name == "multiturn":
                    case["turns"] = []
                    for turn, amount in enumerate((600, 437), 1):
                        text = prompt_for(amount)
                        entry = {"name": f"{name}_{turn}", "expected_amount": amount, "input_text": text}
                        pre = trace.prefill(text_list=[text])
                        if not pre["result"]["success"]:
                            raise RuntimeError(pre)
                        case["turns"].append(collect(entry, name=entry["name"], first_result=pre))
                    case["status"] = "completed"
                elif name.startswith("audio_"):
                    audio_path = args.user_audio
                    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                    count = int(np.ceil(len(audio) / 16000))
                    chunks = [
                        np.pad(audio[i * 16000:(i + 1) * 16000],
                               (0, max(0, (i + 1) * 16000 - len(audio))))
                        for i in range(count)
                    ]
                    evidence = (
                        "知识库资料：用户说“当出现植物大战僵尸的时候提醒我”时，"
                        "请提醒用户检查阳光、坚果冷却时间和草坪状态。"
                        "只根据这份资料用一句话回应用户。"
                    )
                    case["audio_input"] = str(audio_path)
                    case["input_text"] = evidence if name != "audio_only" else None
                    case["expected_keywords"] = ["阳光", "坚果", "草坪"]
                    # Public configuration holds generation while the recorded
                    # question is ingested. It does not force any speak token.
                    duplex.force_listen_count = count if name == "audio_then_text" else count - 1
                    case["force_listen_count_for_question"] = duplex.force_listen_count
                    case["input_units"] = []
                    for idx, chunk in enumerate(chunks):
                        kwargs = {"audio_waveform": chunk.astype(np.float32)}
                        if idx == count - 1 and name == "audio_with_evidence":
                            kwargs["text_list"] = [evidence]
                        pre = trace.prefill(**kwargs)
                        if not pre["result"]["success"]:
                            raise RuntimeError(pre)
                        if idx == count - 1 and name != "audio_then_text":
                            break
                        held = duplex.streaming_generate(decode_mode="greedy")
                        case["input_units"].append({
                            "prefill": pre, "is_listen": held["is_listen"],
                            "text": held["text"],
                        })
                    if name == "audio_then_text":
                        pre = trace.prefill(text_list=[evidence])
                    collect(case, name=name, first_result=pre)
                    case["text_keyword_hits"] = [
                        word for word in case["expected_keywords"] if word in case["text"]
                    ]
                    case["status"] = "completed"
                else:
                    if "no_evidence" in name:
                        text = "用户问题：上海出差住宿每晚最多能报销多少钱？没有提供报销制度，请不要猜测。"
                    else:
                        amount = 600 if name == "text_600" else 437
                        case["expected_amount"] = amount
                        text = prompt_for(amount)
                    case["input_text"] = text
                    # A completed listening unit establishes an existing session.
                    # It is identical in ongoing vs mixed cases before injection.
                    if name.startswith(("ongoing_", "mixed_")):
                        duplex.force_listen_count = 1
                        warm_pre = trace.prefill(audio_waveform=np.zeros(16000, np.float32))
                        if not warm_pre["result"]["success"]:
                            raise RuntimeError(warm_pre)
                        warm = duplex.streaming_generate(decode_mode="greedy")
                        case["warmup_unit"] = {
                            "text": warm["text"], "is_listen": warm["is_listen"],
                            "end_of_turn": warm["end_of_turn"], "prefill": warm_pre,
                        }
                        duplex.force_listen_count = 0
                    kwargs = {"text_list": [text]}
                    if name.startswith(("mixed_", "cold_mixed_")):
                        kwargs["audio_waveform"] = np.zeros(16000, np.float32)
                    case["refresh_control"] = "refreshed" in name
                    pre = trace.prefill(refresh=case["refresh_control"], **kwargs)
                    if not pre["result"]["success"]:
                        raise RuntimeError(pre)
                    collect(case, name=name, first_result=pre)
                    case["status"] = "completed"
            except Exception:
                case["status"] = "error"
                case["error"] = traceback.format_exc()
                print(case["error"], flush=True)
            write_json(logging_path, document)
        document["status"] = "completed"
    finally:
        trace.close()
        document["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        write_json(logging_path, document)
        del trace, duplex, model
        gc.collect()
        torch.cuda.empty_cache()
    print("[probe] saved", logging_path, flush=True)
    if any(c.get("status") == "error" for c in document["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
