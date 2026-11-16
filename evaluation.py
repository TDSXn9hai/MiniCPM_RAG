"""Small, explicit POC scoring rules; not a semantic QA evaluator."""
import re

import numpy as np


def amount_matches(text: str, amount: int) -> bool:
    chinese = {600: "六百", 437: "四百三十七", 862: "八百六十二"}
    return bool(re.search(rf"(?<!\d){amount}(?!\d)", text)) or (
        amount in chinese and chinese[amount] in text
    )


def summarize_audio(audio: np.ndarray, sample_rate: int) -> dict:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    return {
        "samples": int(samples.size),
        "sample_rate": sample_rate,
        "duration_s": samples.size / sample_rate,
        "peak": peak,
        "rms": rms,
        "non_silent": rms > 0.0001,
        "finite": bool(np.isfinite(samples).all()),
    }
