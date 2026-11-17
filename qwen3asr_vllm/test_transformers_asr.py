#!/usr/bin/env python3
"""Load Qwen3-ASR directly without importing Qwen's vLLM backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent / "site-packages" / "qwen_asr"


def install_lazy_package_shim() -> None:
    """Avoid qwen_asr/__init__.py importing the optional vLLM backend."""
    package = types.ModuleType("qwen_asr")
    package.__file__ = str(PACKAGE_ROOT / "__init__.py")
    package.__package__ = "qwen_asr"
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["qwen_asr"] = package

    core = types.ModuleType("qwen_asr.core")
    core.__file__ = str(PACKAGE_ROOT / "core" / "__init__.py")
    core.__package__ = "qwen_asr.core"
    core.__path__ = [str(PACKAGE_ROOT / "core")]
    sys.modules["qwen_asr.core"] = core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path(
            "/root/paddlejob/workspace/env_run/avatar_full/"
            "minicpm_rag_test/tests/browser/fake-mic.wav"
        ),
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_lazy_package_shim()

    import librosa
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

    model_path = args.model.resolve()
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    load_s = time.perf_counter() - started

    audio, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt += f"language {args.language}<asr_text>"
    inputs = processor(
        text=[prompt],
        audio=[audio],
        return_tensors="pt",
        padding=True,
    ).to(args.device).to(model.dtype)

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    infer_s = time.perf_counter() - started
    generated = output.sequences[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(
        f"audio={args.audio} sample_rate={sample_rate} "
        f"duration_s={len(audio) / sample_rate:.3f}"
    )
    print(f"load_s={load_s:.3f} infer_s={infer_s:.3f}")
    print(f"text={text!r}")


if __name__ == "__main__":
    main()
