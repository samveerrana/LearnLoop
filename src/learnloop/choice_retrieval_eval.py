from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .benchmark100 import extract_answer
from .memory import MemoryStore
from .ollama_chat import ollama_answer, unload
from .research import WikipediaResearcher


def run(suite_path: Path, prior_path: Path, output: Path, model: str) -> dict:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in suite["cases"]}
    prior = [json.loads(line) for line in prior_path.read_text(encoding="utf-8").splitlines()]
    targets = [row for row in prior if row["category"] in {"mmlu", "computer_science"}]
    rows: list[dict] = []
    scratch = Path(tempfile.mkdtemp(prefix="learnloop-choice-retrieval-"))
    try:
        for index, old in enumerate(targets, 1):
            case = cases[old["id"]]
            memory = MemoryStore(scratch / f"{case['id']}.sqlite3")
            query = case["prompt"].split("\nA.", 1)[0]
            sources: list[str] = []
            try:
                sources = WikipediaResearcher(memory).research(query, page_limit=2).sources
            except Exception:
                pass
            raw = ollama_answer(
                model, case["prompt"], memory, enhanced=True, thinking=False, max_tokens=128,
                system_prompt=(
                    "Use the supplied encyclopedia evidence only when it directly applies. "
                    "Solve the multiple-choice problem and return only A, B, C, or D."
                ),
            )
            answer = extract_answer(raw, "choice")
            row = {
                "id": case["id"], "category": case["category"], "expected": case["expected"],
                "prior_answer": old["answer"], "prior_correct": old["correct"], "answer": answer,
                "correct": answer == case["expected"], "sources": sources, "raw": raw,
            }
            rows.append(row)
            print(f"{index}/{len(targets)} {case['id']}: {'PASS' if row['correct'] else 'FAIL'}", flush=True)
    finally:
        unload(model)
    report = {
        "model": model, "method": "official Wikipedia API evidence on all MMLU and computer-science choices",
        "development_only": True, "prior_correct": sum(row["prior_correct"] for row in rows),
        "correct": sum(row["correct"] for row in rows), "total": len(rows),
        "gains": sum(row["correct"] and not row["prior_correct"] for row in rows),
        "regressions": sum(row["prior_correct"] and not row["correct"] for row in rows),
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Development pilot for official evidence on choice misses")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    args = parser.parse_args()
    print(json.dumps(run(args.suite, args.prior, args.output, args.model), indent=2))


if __name__ == "__main__":
    main()
