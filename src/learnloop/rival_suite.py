from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from .exact_solver import solve_exact
from .memory import MemoryStore
from .ollama_chat import ollama_answer, unload


CASES = [
    {"id": "K1", "category": "knowledge", "question": "In what year was the United States Declaration of Independence adopted?", "answers": ["1776"]},
    {"id": "K2", "category": "knowledge", "question": "Which amendment to the US Constitution abolished slavery?", "answers": ["13th", "thirteenth"]},
    {"id": "K3", "category": "knowledge", "question": "Which country sold Alaska to the United States?", "answers": ["russia"]},
    {"id": "K4", "category": "knowledge", "question": "What is the chemical formula for water?", "answers": ["h2o"]},
    {"id": "K5", "category": "knowledge", "question": "Which planet is called the Red Planet?", "answers": ["mars"]},
    {"id": "L1", "category": "logic", "question": "All flurps are daxes. No dax is a wug. Can any flurp be a wug? Answer only yes or no.", "answers": ["no"]},
    {"id": "L2", "category": "logic", "question": "A is taller than B, and B is taller than C. Who is shortest? Answer only A, B, or C.", "answers": ["c"]},
    {"id": "L3", "category": "logic", "question": "If today is Monday, what day is 10 days later?", "answers": ["thursday"]},
    {"id": "L4", "category": "logic", "question": "A bag has 3 red and 2 blue balls. What is the probability of drawing blue? Give a fraction.", "answers": ["2/5"]},
    {"id": "L5", "category": "logic", "question": "Continue the pattern: 2, 6, 12, 20, 30, ?", "answers": ["42"]},
    {"id": "C1", "category": "coding", "question": "What does Python print? print(sum(i*i for i in range(4)))", "answers": ["14"]},
    {"id": "C2", "category": "coding", "question": "What does Python print? x=[1,2,3]; print(x[::-1])", "answers": ["[3, 2, 1]", "[3,2,1]"]},
    {"id": "C3", "category": "coding", "question": "What is the time complexity of binary search?", "answers": ["o(log n)", "o(logn)"]},
    {"id": "C4", "category": "coding", "question": "In Git, which command shows unstaged line changes?", "answers": ["git diff"]},
    {"id": "C5", "category": "coding", "question": "What HTTP status code means Not Found?", "answers": ["404"]},
]


def normalized(text: str) -> str:
    substitutions = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    value = text.lower().translate(substitutions)
    value = re.sub(r"\\(?:text|frac)\{([^{}]+)\}(?:\{([^{}]+)\})?", lambda match: f"{match.group(1)}/{match.group(2)}" if match.group(2) else match.group(1), value)
    value = value.replace("\\(", "").replace("\\)", "")
    return re.sub(r"\s+", " ", value).strip()


def correct(answer: str, accepted: list[str]) -> bool:
    value = normalized(answer)
    for item in accepted:
        needle = normalized(item)
        if len(needle) <= 3 or re.fullmatch(r"[\d/]+", needle):
            if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", value):
                return True
        elif needle in value:
            return True
    return False


def run(model: str, mode: str, output: Path) -> dict:
    scratch = Path(tempfile.mkdtemp(prefix=f"learnloop-rival-{mode}-"))
    rows = []
    started = time.monotonic()
    for case in CASES:
        # Each case gets isolated memory so retrieved text cannot leak into later questions.
        memory = MemoryStore(scratch / f"{case['id']}.sqlite3")
        answer = ""
        used = "model"
        if mode == "learnloop":
            exact = solve_exact(case["question"])
            if exact is not None:
                answer, used = str(exact), "verified-local-tool"
        if not answer:
            prompt = "/no_think\nAnswer in one short line. " + case["question"]
            answer = ollama_answer(model, prompt, memory, enhanced=mode == "learnloop", thinking=False, max_tokens=512)
        passed = correct(answer, case["answers"])
        rows.append({**case, "answer": answer, "correct": passed, "used": used})
        print(f"{mode} {case['id']}: {'PASS' if passed else 'FAIL'}", flush=True)
    unload(model)
    categories = {
        category: {
            "correct": sum(row["correct"] for row in rows if row["category"] == category),
            "total": sum(1 for row in rows if row["category"] == category),
        }
        for category in sorted({row["category"] for row in rows})
    }
    result = {
        "suite": "learnloop-rival-smoke-v1",
        "scope": "smoke-test-not-20b-proof",
        "model": model, "mode": mode, "correct": sum(row["correct"] for row in rows),
        "total": len(rows), "seconds": round(time.monotonic() - started, 2),
        "categories": categories, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model", "mode", "correct", "total", "seconds", "categories")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one model at a time on the transparent LearnLoop rival suite")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("base", "reference", "learnloop"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.mode, args.output)


if __name__ == "__main__":
    main()
