import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_retrieve_only_writes_citations_and_does_not_load_model(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "run_rag.py"),
        "--query", "上海出差住宿能报多少", "--full-text",
        "--retrieve-only", "--output", str(tmp_path),
        "--model-path", str(tmp_path / "nonexistent_model"),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "results.json").read_text())
    turn = data["cases"][0]
    assert turn["status"] == "retrieved"
    assert turn["citations"][0]["document_id"] == "travel-policy"
    assert turn["model_called"] is False
    assert "六百" in turn["retrieval"]["evidence"]


def test_no_match_does_not_load_nonexistent_model(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "run_rag.py"),
        "--query", "量子纠缠实验原理",
        "--output", str(tmp_path), "--model-path", str(tmp_path / "missing"),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["cases"][0]["status"] == "no_evidence"
    assert data["cases"][0]["wav"] is None
    assert data["model_load_s"] is None


def test_default_title_only_does_not_match_content_only_query(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "run_rag.py"),
        "--query", "上海出差住宿能报多少", "--retrieve-only",
        "--output", str(tmp_path),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["config"]["title_only"] is True
    assert data["cases"][0]["status"] == "no_evidence"
