from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_question(rng: random.Random, index: int) -> tuple[str, int, str]:
    category = index % 4
    if category == 0:
        a, b, c = rng.randint(20, 900), rng.randint(3, 40), rng.randint(2, 25)
        return f"Compute ({a} * {b}) - ({c} * {c}). Give only the integer.", a * b - c * c, "arithmetic"

    if category == 1:
        start, mul, add, mod, steps = rng.randint(1, 500), rng.randint(2, 15), rng.randint(1, 40), 997, rng.randint(15, 45)
        value = start
        for _ in range(steps):
            value = (mul * value + add) % mod
        return f"x0={start}; x(n+1)=({mul}*x(n)+{add}) mod {mod}. Find x{steps}. Give only the integer.", value, "recurrence"

    if category == 2:
        size = rng.randint(5, 8)
        blocked = set(rng.sample([(x, y) for x in range(1, size) for y in range(1, size)], 4))
        dp = [[0] * (size + 1) for _ in range(size + 1)]
        dp[0][0] = 1
        for x in range(size + 1):
            for y in range(size + 1):
                if (x, y) != (0, 0):
                    dp[x][y] = 0 if (x, y) in blocked else (dp[x - 1][y] if x else 0) + (dp[x][y - 1] if y else 0)
        return f"Count Right/Up paths from (0,0) to ({size},{size}) avoiding {sorted(blocked)}. Give only the integer.", dp[size][size], "grid"

    values = [rng.randint(2, 25) for _ in range(12)]
    target = rng.randint(30, 70)
    count = sum(sum(values[j] for j in range(12) if mask & (1 << j)) == target for mask in range(1 << 12))
    return f"Positions are distinct. How many subsets of {values} sum to {target}? Give only the integer.", count, "subset"


def generate(destination: Path, seed: int = 41004) -> None:
    rng = random.Random(seed)
    questions = destination / "questions.jsonl"
    answers = destination / "private-answers.jsonl"
    destination.mkdir(parents=True, exist_ok=True)
    with questions.open("w", encoding="utf-8") as qh, answers.open("w", encoding="utf-8") as ah:
        for index in range(1000):
            question, answer, category = build_question(rng, index)
            item_id = f"LL-{index + 1:04d}"
            qh.write(json.dumps({"id": item_id, "category": category, "question": question}) + "\n")
            ah.write(json.dumps({"id": item_id, "answer": answer}) + "\n")
    manifest = {"count": 1000, "seed": seed, "categories": ["arithmetic", "recurrence", "grid", "subset"]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path.cwd() / "benchmarks/quiz-1000")
    args = parser.parse_args()
    generate(args.output)
    print(f"Generated 1,000 locked questions at {args.output}")


if __name__ == "__main__":
    main()
