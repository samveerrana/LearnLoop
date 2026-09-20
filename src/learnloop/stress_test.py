from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import tempfile
import time
from pathlib import Path

from .cli import default_database_path, default_model_path
from .memory import MemoryStore
from .model import LocalModel
from .research import WikipediaResearcher
from .smart import tool_answer


FINAL_RE = re.compile(r"FINAL\s*:\s*(-?\d+)", re.IGNORECASE)


def grid_problem(rng: random.Random) -> tuple[str, int, str]:
    size = rng.randint(6, 9)
    candidates = [(x, y) for x in range(1, size) for y in range(1, size)]
    blocked = set(rng.sample(candidates, rng.randint(4, 7)))
    dp = [[0] * (size + 1) for _ in range(size + 1)]
    dp[0][0] = 1
    for x in range(size + 1):
        for y in range(size + 1):
            if (x, y) == (0, 0):
                continue
            dp[x][y] = 0 if (x, y) in blocked else (dp[x - 1][y] if x else 0) + (dp[x][y - 1] if y else 0)
    prompt = (
        f"A robot goes from (0,0) to ({size},{size}), moving only Right or Up. "
        f"It cannot visit {sorted(blocked)}. How many paths are valid? "
        "Start with exactly FINAL: number, then give at most four short reasoning sentences."
    )
    return prompt, dp[size][size], "blocked-grid paths"


def code_problem(rng: random.Random) -> tuple[str, int, str]:
    length = rng.randint(4, 6)
    divisor = rng.choice([7, 9, 11, 13])
    target_sum = rng.randint(16, 28)
    count = 0
    for digits in itertools.permutations(range(10), length):
        if digits[0] == 0 or sum(digits) != target_sum:
            continue
        number = sum(digit * 10 ** (length - index - 1) for index, digit in enumerate(digits))
        count += number % divisor == 0
    prompt = (
        f"How many {length}-digit positive integers have no repeated digits, have digit sum {target_sum}, "
        f"and are divisible by {divisor}? Leading zero is forbidden. "
        "Start with exactly FINAL: number, then give at most four short reasoning sentences."
    )
    return prompt, count, "constrained digit codes"


def recurrence_problem(rng: random.Random) -> tuple[str, int, str]:
    start = rng.randint(10, 500)
    multiplier = rng.randint(3, 19)
    addition = rng.randint(5, 50)
    modulus = rng.choice([997, 1009, 1013])
    steps = rng.randint(35, 90)
    value = start
    for _ in range(steps):
        value = (multiplier * value + addition) % modulus
    prompt = (
        f"Let x0={start} and x(n+1)=({multiplier}*x(n)+{addition}) mod {modulus}. "
        f"What is x{steps}? Start with exactly FINAL: number, then give at most four short reasoning sentences."
    )
    return prompt, value, "modular recurrence"


def subset_problem(rng: random.Random) -> tuple[str, int, str]:
    values = [rng.randint(2, 30) for _ in range(rng.randint(11, 15))]
    target = rng.randint(35, 80)
    count = sum(
        sum(values[index] for index in range(len(values)) if mask & (1 << index)) == target
        for mask in range(1 << len(values))
    )
    prompt = (
        f"Treat these positions as distinct even if values repeat: {values}. "
        f"How many subsets have sum exactly {target}? "
        "Start with exactly FINAL: number, then give at most four short reasoning sentences."
    )
    return prompt, count, "subset-sum counting"


def schedule_problem(rng: random.Random) -> tuple[str, int, str]:
    size = rng.choice([7, 8])
    people = [chr(65 + index) for index in range(size)]
    before_a, before_b = rng.sample(people, 2)
    adjacent_a, adjacent_b = rng.sample([p for p in people if p not in {before_a, before_b}], 2)
    fixed_person = rng.choice([p for p in people if p not in {before_a, before_b, adjacent_a, adjacent_b}])
    fixed_slot = rng.randint(2, size - 1)
    count = 0
    for order in itertools.permutations(people):
        if order.index(before_a) >= order.index(before_b):
            continue
        if abs(order.index(adjacent_a) - order.index(adjacent_b)) != 1:
            continue
        if order[fixed_slot - 1] != fixed_person:
            continue
        count += 1
    prompt = (
        f"People {', '.join(people)} stand in a line. {before_a} must be before {before_b}; "
        f"{adjacent_a} and {adjacent_b} must be adjacent; {fixed_person} must be in position {fixed_slot}. "
        "How many orders satisfy all rules? Start with exactly FINAL: number, then give at most four short reasoning sentences."
    )
    return prompt, count, "constraint scheduling"


GENERATORS = [grid_problem, code_problem, recurrence_problem, subset_problem, schedule_problem]


def parsed_answer(text: str) -> int | None:
    matches = FINAL_RE.findall(text)
    return int(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    deadline = time.monotonic() + args.minutes * 60
    rng = random.Random(args.seed)
    persistent = MemoryStore(default_database_path())
    output = Path.cwd() / "stress-test-results.json"
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as directory:
        empty = MemoryStore(Path(directory) / "empty.sqlite3")
        engine = LocalModel(default_model_path(), empty)
        print(f"Starting {args.minutes:g}-minute reasoning stress test", flush=True)
        iteration = 0
        while time.monotonic() < deadline:
            iteration += 1
            question, expected, category = GENERATORS[(iteration - 1) % len(GENERATORS)](rng)
            started = time.monotonic()
            engine.memory = empty
            engine.researcher = WikipediaResearcher(empty)
            base_text = engine.answer(question, max_tokens=2200, auto_research=False, thinking=True)
            engine.memory = persistent
            engine.researcher = WikipediaResearcher(persistent)
            try:
                learned_text, tool_code = tool_answer(engine, question)
                tool_error = ""
            except Exception as error:
                learned_text = engine.answer(question, max_tokens=2200, auto_research=False, thinking=True)
                tool_code = ""
                tool_error = str(error)
            base_value = parsed_answer(base_text)
            learned_value = parsed_answer(learned_text)
            row = {
                "number": iteration,
                "category": category,
                "question": question,
                "expected": expected,
                "base_value": base_value,
                "learnloop_value": learned_value,
                "base_correct": base_value == expected,
                "learnloop_correct": learned_value == expected,
                "seconds": round(time.monotonic() - started, 2),
                "base_answer": base_text,
                "learnloop_answer": learned_text,
                "tool_code": tool_code,
                "tool_error": tool_error,
            }
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            base_score = sum(item["base_correct"] for item in rows)
            learned_score = sum(item["learnloop_correct"] for item in rows)
            remaining = max(0, int(deadline - time.monotonic()))
            print(
                f"#{iteration} {category}: expected={expected} "
                f"base={base_value} {'✓' if row['base_correct'] else '✗'} "
                f"learnloop={learned_value} {'✓' if row['learnloop_correct'] else '✗'} | "
                f"score {base_score}-{learned_score} | {remaining}s left",
                flush=True,
            )

    summary = {
        "duration_minutes": args.minutes,
        "questions": len(rows),
        "base_correct": sum(item["base_correct"] for item in rows),
        "learnloop_correct": sum(item["learnloop_correct"] for item in rows),
        "base_missing_final": sum(item["base_value"] is None for item in rows),
        "learnloop_missing_final": sum(item["learnloop_value"] is None for item in rows),
        "memory_helped": sum((not item["base_correct"]) and item["learnloop_correct"] for item in rows),
        "memory_hurt": sum(item["base_correct"] and (not item["learnloop_correct"]) for item in rows),
        "average_seconds_per_pair": round(sum(item["seconds"] for item in rows) / len(rows), 2),
    }
    (Path.cwd() / "stress-test-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished: {summary}", flush=True)


if __name__ == "__main__":
    main()
