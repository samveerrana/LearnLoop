from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .cli import default_database_path
from .memory import MemoryStore
from .ollama_chat import ollama_answer, unload


INTEGER = re.compile(r"-?\d+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base and LearnLoop separately to protect RAM")
    parser.add_argument("--mode", choices=["base", "learnloop"], required=True)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--questions", type=Path, default=Path.cwd() / "benchmarks/quiz-1000/questions.jsonl")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--cooldown", type=float, default=2.0)
    args = parser.parse_args()
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.model)
    output = args.output or Path.cwd() / f"benchmarks/quiz-1000/{args.mode}-{safe_model}-results.jsonl"
    completed = set()
    if output.exists():
        completed = {json.loads(line)["id"] for line in output.read_text().splitlines() if line.strip()}
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    memory = MemoryStore(default_database_path())
    ran = 0
    try:
        with output.open("a", encoding="utf-8") as handle:
            for item in questions:
                if item["id"] in completed or ran >= args.limit:
                    continue
                started = time.monotonic()
                answer = ollama_answer(
                    args.model,
                    "/no_think\nGive only the final integer. " + item["question"],
                    memory,
                    enhanced=args.mode == "learnloop",
                    thinking=False,
                    max_tokens=512,
                )
                matches = INTEGER.findall(answer.replace(",", ""))
                record = {
                    "id": item["id"],
                    "answer": int(matches[-1]) if matches else None,
                    "raw": answer,
                    "seconds": round(time.monotonic() - started, 2),
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                ran += 1
                print(f"{args.mode}: {item['id']} saved ({ran}/{args.limit})", flush=True)
                time.sleep(args.cooldown)
    finally:
        unload(args.model)
        print(f"{args.model} unloaded from RAM.")


if __name__ == "__main__":
    main()
