"""Real GPU check of retrieval + hot reload + official text_list + refusal.

Run directly, not collected by pytest. Writes only under outputs/ and tests
the public CLI in one persistent conversation. Never edits knowledge/sample.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import amount_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rag_integration")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise SystemExit(f"{target} exists; preserve it or choose a fresh path in this script")
    target.mkdir(parents=True)
    fixture = target / "knowledge.json"
    original = json.loads((ROOT / "knowledge/sample.json").read_text())
    fixture.write_text(json.dumps(original, ensure_ascii=False, indent=2))
    log = (target / "run.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "run_rag.py"),
         "--interactive", "--knowledge", str(fixture), "--output", str(target)],
        stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
        text=True, env=os.environ.copy(),
    )

    def ask(query, index):
        process.stdin.write(query + "\n")
        process.stdin.flush()
        path = target / f"turn_{index:03d}.json"
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if path.exists():
                return json.loads(path.read_text())
            if process.poll() is not None:
                raise RuntimeError(f"CLI exited {process.returncode}; inspect {target / 'run.log'}")
            time.sleep(0.2)
        raise TimeoutError(f"turn {index} did not finish; inspect {target / 'run.log'}")

    try:
        first = ask("上海出差住宿能报多少", 1)
        assert first["status"] == "answered", first
        assert amount_matches(first["text"], 600), first["text"]
        assert first["citations"][0]["document_id"] == "travel-policy"
        print("turn 1:", first["text"], flush=True)

        updated = json.loads(json.dumps(original))
        for doc in updated["documents"]:
            if doc["id"] == "travel-policy":
                doc["content"] = doc["content"].replace("六百元", "四百三十七元")
        temporary = fixture.with_suffix(".tmp")
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2))
        temporary.replace(fixture)
        second = ask("上海出差住宿能报多少", 2)
        assert second["status"] == "answered", second
        assert amount_matches(second["text"], 437), second["text"]
        assert not amount_matches(second["text"], 600), second["text"]
        assert first["retrieval"]["knowledge_sha256"] != second["retrieval"]["knowledge_sha256"]
        print("turn 2:", second["text"], flush=True)

        third = ask("量子纠缠实验原理", 3)
        assert third["status"] == "no_evidence", third
        assert third["model_called"] is False and third["wav"] is None
        print("turn 3:", third["text"], flush=True)
        process.stdin.write("/quit\n")
        process.stdin.flush()
        process.stdin.close()
        assert process.wait(timeout=30) == 0
        print("passed:", target / "results.json", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    main()
