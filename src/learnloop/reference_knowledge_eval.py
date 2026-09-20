from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from .memory import MemoryStore
from .ollama_chat import ollama_answer, unload


def extract_year(text: str) -> str | None:
    years = re.findall(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?!\d)", text)
    return years[-1] if years else None


def evaluate(model: str, cases_path: Path, output: Path) -> dict:
    source = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = source["rows"]
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="learnloop-reference-knowledge-") as temporary:
        memory = MemoryStore(Path(temporary) / "memory.sqlite3")
        try:
            for position, case in enumerate(cases, 1):
                started = time.monotonic()
                raw = ollama_answer(
                    model, case["prompt"], memory, enhanced=False, thinking=False, max_tokens=12,
                    system_prompt="Answer from model weights only. Return exactly one four-digit year.",
                )
                answer = extract_year(raw)
                row = {
                    "id": case["id"], "title": case["title"], "prompt": case["prompt"],
                    "expected": case["expected"], "source_url": case["source_url"],
                    "answer": answer, "correct": answer == case["expected"], "raw": raw,
                    "seconds": round(time.monotonic() - started, 3),
                }
                rows.append(row)
                print(f"{position}/{len(cases)} {'PASS' if row['correct'] else 'FAIL'}", flush=True)
        finally:
            try:
                unload(model)
            except Exception:
                pass
    report = {
        "evaluation": "closed-book held-out phrasing of licensed official-web facts",
        "retrieval_during_evaluation": False,
        "model": model,
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Ollama reference on frozen web-fact questions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.cases, args.output)


if __name__ == "__main__":
    main()
