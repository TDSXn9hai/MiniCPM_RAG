#!/usr/bin/env python3
"""Independently transcribe generated WAVs with existing local Whisper-small.

No expected answer or model-generated text is supplied to the ASR.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluation import amount_matches

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "outputs/isolated/results.json"))
    parser.add_argument("--model", default=str(ROOT / "models/whisper-small"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result_path = Path(args.results)
    data = json.loads(result_path.read_text())
    entries = [
        turn
        for case in data["cases"]
        for turn in (case.get("turns") or [case])
        if turn.get("wav")
    ]
    processor = WhisperProcessor.from_pretrained(args.model, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    ).eval().to(args.device)
    records = []
    for entry in entries:
        audio, _ = librosa.load(entry["wav"], sr=16000, mono=True)
        batch = processor(
            audio, sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True,
        ).to(args.device)
        start = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(
                batch.input_features.to(model.dtype),
                attention_mask=batch.attention_mask,
                language="zh", task="transcribe",
                do_sample=False, max_new_tokens=128,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        record = {
            "name": entry["name"], "wav": entry["wav"],
            "asr_text": text, "generation_text": entry.get("text", ""),
            "asr_wall_s": time.perf_counter() - start,
        }
        if "expected_amount" in entry:
            record["expected_amount"] = entry["expected_amount"]
            record["asr_has_expected_amount"] = amount_matches(text, entry["expected_amount"])
            record["asr_has_other_test_amount"] = any(
                amount_matches(text, n)
                for n in (600, 437, 862) if n != entry["expected_amount"]
            )
        if "expected_keywords" in entry:
            record["expected_keywords"] = entry["expected_keywords"]
            record["asr_keyword_hits"] = [
                word for word in entry["expected_keywords"] if word in text
            ]
        records.append(record)
        target = result_path.parent / "asr_results.json"
        target.write_text(json.dumps({
            "asr_model": str(Path(args.model).resolve()),
            "device": args.device,
            "answer_text_supplied_as_prompt": False,
            "records": records,
        }, ensure_ascii=False, indent=2))
        print(entry["name"], repr(text), flush=True)


if __name__ == "__main__":
    main()
