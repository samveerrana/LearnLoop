from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .cli import default_database_path, default_model_path
from .memory import MemoryStore
from .model import LocalModel
from .research import WikipediaResearcher


CASES = [
    ("In US history, in what year was the Declaration of Independence adopted?", [["1776"]]),
    ("In US history, which purchase doubled the country's territory in 1803?", [["louisiana"]]),
    ("In US history, what years did the American Civil War begin and end?", [["1861"], ["1865"]]),
    ("In US history, which amendment abolished slavery?", [["13th", "thirteenth"]]),
    ("In US history, in what year did women gain nationwide voting rights through the 19th Amendment?", [["1920"]]),
    ("In US history, which 1929 event marked the beginning of the Great Depression?", [["crash", "stock market"]]),
    ("In US history, which country sold Alaska to the United States?", [["russia"]]),
    ("In US history, what 1964 law prohibited discrimination in public places and employment?", [["civil rights act"]]),
]


def score(answer: str, expected_groups: list[list[str]]) -> bool:
    lowered = answer.lower()
    return all(any(option in lowered for option in group) for group in expected_groups)


def main() -> None:
    persistent = MemoryStore(default_database_path())
    with tempfile.TemporaryDirectory() as directory:
        empty = MemoryStore(Path(directory) / "empty.sqlite3")
        engine = LocalModel(default_model_path(), empty)
        rows = []
        print("\n=== BASE MODEL VS LEARNLOOP ===\n")
        for index, (question, expected) in enumerate(CASES, start=1):
            print(f"[{index}/{len(CASES)}] {question}")
            engine.memory = empty
            engine.researcher = WikipediaResearcher(empty)
            base_answer = engine.answer(question, max_tokens=70, auto_research=False)
            base_ok = score(base_answer, expected)

            engine.memory = persistent
            engine.researcher = WikipediaResearcher(persistent)
            learned_answer = engine.answer(question, max_tokens=70, auto_research=False)
            learned_ok = score(learned_answer, expected)
            print(f"  Base:      {'PASS' if base_ok else 'FAIL'}")
            print(f"  LearnLoop: {'PASS' if learned_ok else 'FAIL'}")
            rows.append(
                {
                    "question": question,
                    "base_answer": base_answer,
                    "learnloop_answer": learned_answer,
                    "base_pass": base_ok,
                    "learnloop_pass": learned_ok,
                }
            )

    base_score = sum(row["base_pass"] for row in rows)
    learned_score = sum(row["learnloop_pass"] for row in rows)
    result = {
        "total": len(rows),
        "base_score": base_score,
        "learnloop_score": learned_score,
        "base_percent": round(base_score / len(rows) * 100, 1),
        "learnloop_percent": round(learned_score / len(rows) * 100, 1),
        "rows": rows,
    }
    output = Path.cwd() / "benchmark-results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== FINAL SCORE ===")
    print(f"Base model: {base_score}/{len(rows)} ({result['base_percent']}%)")
    print(f"LearnLoop:  {learned_score}/{len(rows)} ({result['learnloop_percent']}%)")
    print(f"Saved full answers to {output}")


if __name__ == "__main__":
    main()
