import numpy as np

from evaluation import amount_matches, summarize_audio


def test_amount_matches_chinese_and_arabic_without_substring_false_positive():
    assert amount_matches("标准为四百三十七元。", 437)
    assert amount_matches("每晚437元。", 437)
    assert amount_matches("六百元", 600)
    assert not amount_matches("每晚1437元。", 437)
    assert not amount_matches("每晚4370元。", 437)
    assert not amount_matches("六百元。", 437)


def test_audio_summary_distinguishes_silence_from_nonempty_audio():
    silent = summarize_audio(np.zeros(24000, np.float32), 24000)
    assert silent["duration_s"] == 1.0
    assert silent["non_silent"] is False
    voiced = summarize_audio(np.full(12000, 0.02, np.float32), 24000)
    assert voiced["duration_s"] == 0.5
    assert voiced["non_silent"] is True
