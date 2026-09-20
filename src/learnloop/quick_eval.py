from __future__ import annotations

import gc
import json
import re
import argparse
from pathlib import Path

import mlx.core as mx

from .cli import default_model_path
from .memory import MemoryStore
from .model import LocalModel


INTEGER = re.compile(r"-?\d+")


def final_integer(text: str) -> int | None:
    numbers = INTEGER.findall(text.replace(",", ""))
    return int(numbers[-1]) if numbers else None


def evaluate(model_path: Path, cases: list[dict], truth: dict[str, int], label: str) -> list[dict]:
    model = LocalModel(model_path, MemoryStore(Path(f"/tmp/learnloop-{label}.sqlite3")))
    rows = []
    for item in cases:
        text = model.answer(item["question"], max_tokens=180, thinking=False, auto_research=False)
        value = final_integer(text)
        correct = value == truth[item["id"]]
        rows.append({"id": item["id"], "expected": truth[item["id"]], "value": value, "correct": correct, "raw": text})
        print(f"{label} {item['id']}: expected={truth[item['id']]} got={value} {'PASS' if correct else 'FAIL'}", flush=True)
    del model
    gc.collect()
    mx.clear_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=25)
    parser.add_argument("--candidate", type=Path, default=Path("rebuild-workspace/v0.4/rebuilt-model"))
    parser.add_argument("--reuse-base", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    quiz = root / "benchmarks/quiz-1000"
    questions = [json.loads(line) for line in quiz.joinpath("questions.jsonl").read_text().splitlines()]
    truth = {row["id"]: row["answer"] for row in map(json.loads, quiz.joinpath("private-answers.jsonl").read_text().splitlines())}
    if not 1 <= args.questions <= 100:
        raise SystemExit("--questions must be between 1 and 100")
    cases = questions[900:900 + args.questions]
    previous_path = root / "quick-eval-results.json"
    if args.reuse_base and previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_base = {row["id"]: row for row in previous.get("base", [])}
        if all(item["id"] in previous_base for item in cases):
            base = [previous_base[item["id"]] for item in cases]
            print(f"Reused {len(base)} locked deterministic base answers.")
        else:
            base = evaluate(default_model_path(), cases, truth, "base")
    else:
        base = evaluate(default_model_path(), cases, truth, "base")
    rebuilt_path = root / args.candidate
    rebuilt = evaluate(rebuilt_path, cases, truth, "rebuilt")
    result = {
        "questions": len(cases),
        "base_correct": sum(row["correct"] for row in base),
        "rebuilt_correct": sum(row["correct"] for row in rebuilt),
        "base": base,
        "rebuilt": rebuilt,
    }
    result["score_multiplier"] = (
        result["rebuilt_correct"] / result["base_correct"] if result["base_correct"] else None
    )
    result["passes_2x_gate"] = bool(
        result["base_correct"] and result["rebuilt_correct"] >= 2 * result["base_correct"]
    )
    (root / "quick-eval-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_path = rebuilt_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["locked_evaluation"] = {
            "questions": result["questions"],
            "base_correct": result["base_correct"],
            "rebuilt_correct": result["rebuilt_correct"],
            "score_multiplier": result["score_multiplier"],
            "passes_2x_gate": result["passes_2x_gate"],
        }
        manifest["status"] = "promoted" if result["passes_2x_gate"] else "rejected-failed-2x-gate"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("questions", "base_correct", "rebuilt_correct")}, indent=2))


if __name__ == "__main__":
    main()
