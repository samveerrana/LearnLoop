from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def _answer(text: str) -> str | None:
    years = re.findall(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?!\d)", text)
    return years[-1] if years else None


def evaluate(model_path: Path, test_path: Path, output: Path, adapter_path: Path | None = None) -> dict:
    from mlx_lm import generate, load
    model, tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
    cases = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    rows = []
    for position, case in enumerate(cases, 1):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "Answer from model weights only. Return exactly one four-digit year."},
            {"role": "user", "content": case["prompt"]},
        ], tokenize=False, add_generation_prompt=True)
        started = time.monotonic()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=12, verbose=False).strip()
        answer = _answer(raw)
        row = {**case, "answer": answer, "correct": answer == case["expected"], "raw": raw, "seconds": round(time.monotonic() - started, 3)}
        rows.append(row)
        print(f"{position}/{len(cases)} {'PASS' if row['correct'] else 'FAIL'}", flush=True)
    report = {
        "evaluation": "closed-book held-out phrasing of licensed web facts",
        "retrieval_during_evaluation": False,
        "model": str(model_path), "adapter": str(adapter_path) if adapter_path else None,
        "correct": sum(row["correct"] for row in rows), "total": len(rows), "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporary web-derived weights with retrieval disabled")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    evaluate(args.model, args.test, args.output, args.adapter)


if __name__ == "__main__":
    main()
