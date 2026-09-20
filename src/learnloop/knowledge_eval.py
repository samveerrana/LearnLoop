from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx

from .cli import default_model_path
from .memory import MemoryStore
from .model import LocalModel


def evaluate(path: Path, cases: list[dict], label: str, adapter_path: Path | None = None) -> list[dict]:
    model = LocalModel(path, MemoryStore(Path(f"/tmp/learnloop-knowledge-{label}.sqlite3")), adapter_path)
    rows = []
    for index, case in enumerate(cases, 1):
        answer = model.answer(case["question"], max_tokens=80, thinking=False, auto_research=False)
        correct = case["expected"].lower() in answer.lower()
        rows.append({**case, "answer": answer, "correct": correct})
        print(f"{label} {index}/25 {case['kind']}: {'PASS' if correct else 'FAIL'}", flush=True)
    del model
    gc.collect()
    mx.clear_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("training/v1.2-candidate"))
    parser.add_argument("--reuse-base", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, default=Path("v1.2-candidate-weight-knowledge-eval.json"))
    args = parser.parse_args()
    root = Path.cwd()
    cases = json.loads((root / args.data / "locked-test.json").read_text())
    if args.reuse_base and (root / args.reuse_base).exists():
        previous = json.loads((root / args.reuse_base).read_text())
        base = previous["base"]
        print(f"Reused {len(base)} locked base answers.")
    else:
        base = evaluate(default_model_path(), cases, "base")
    candidate_path = default_model_path() if args.adapter else root / args.candidate
    candidate = evaluate(candidate_path, cases, "rewritten", root / args.adapter if args.adapter else None)
    base_score = sum(row["correct"] for row in base)
    candidate_score = sum(row["correct"] for row in candidate)
    result = {
        "benchmark": "weight-retained fictional knowledge plus controls",
        "base_correct": base_score,
        "rewritten_correct": candidate_score,
        "multiplier": candidate_score / base_score if base_score else None,
        "passes_5x": bool(base_score and candidate_score >= 5 * base_score),
        "mode": "temporary-task-parameters" if args.adapter else "fused-weights",
        "base": base,
        "rewritten": candidate,
    }
    (root / args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("base_correct", "rewritten_correct", "multiplier", "passes_5x")}, indent=2))


if __name__ == "__main__":
    main()
