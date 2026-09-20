from __future__ import annotations

import argparse
import json
from pathlib import Path

from .exact_solver import solve_exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=25)
    args = parser.parse_args()
    root = Path.cwd()
    quiz = root / "benchmarks/quiz-1000"
    questions = [json.loads(line) for line in quiz.joinpath("questions.jsonl").read_text().splitlines()][900:900 + args.questions]
    truth = {row["id"]: row["answer"] for row in map(json.loads, quiz.joinpath("private-answers.jsonl").read_text().splitlines())}
    base_run = json.loads((root / "quick-eval-results.json").read_text())
    base_by_id = {row["id"]: row for row in base_run["base"]}
    rows = []
    for item in questions:
        value = solve_exact(item["question"])
        correct = value == truth[item["id"]]
        rows.append({"id": item["id"], "value": value, "expected": truth[item["id"]], "correct": correct})
        print(f"v0.5 {item['id']}: {'PASS' if correct else 'FAIL'}")
    base_correct = sum(bool(base_by_id.get(item["id"], {}).get("correct")) for item in questions)
    enhanced_correct = sum(row["correct"] for row in rows)
    result = {
        "questions": len(rows),
        "base_correct": base_correct,
        "v0.5_system_correct": enhanced_correct,
        "score_multiplier": enhanced_correct / base_correct if base_correct else None,
        "passes_2x_gate": bool(base_correct and enhanced_correct >= 2 * base_correct),
        "scope": "generated exact-reasoning benchmark only; not a general 8B-equivalence claim",
        "rows": rows,
    }
    (root / "v0.5-system-eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("base_correct", "v0.5_system_correct", "score_multiplier", "passes_2x_gate")}, indent=2))


if __name__ == "__main__":
    main()
