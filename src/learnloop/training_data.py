from __future__ import annotations

import json
import re
import random
from pathlib import Path

from .prompting import SYSTEM_PROMPT, inference_user_content
from .quiz import build_question


def worked_answer(question: str, answer: int, category: str) -> str:
    if category == "arithmetic":
        values = [int(value) for value in re.findall(r"\d+", question)]
        a, b, c, _ = values
        return f"Compute each product, then subtract: {a}*{b}={a*b}; {c}*{c}={c*c}; result={answer}.\n{answer}"
    if category == "recurrence":
        start, multiplier, addition, modulus, steps = map(
            int,
            re.fullmatch(
                r"x0=(\d+); x\(n\+1\)=\((\d+)\*x\(n\)\+(\d+)\) mod (\d+)\. Find x(\d+)\. Give only the integer\.",
                question,
            ).groups(),
        )
        value = start
        states = []
        for step in range(1, steps + 1):
            value = (multiplier * value + addition) % modulus
            states.append(f"x{step}={value}")
        return "Reduce after every step: " + ", ".join(states) + f".\n{answer}"
    if category == "grid":
        size = int(re.search(r"to \((\d+),(\d+)\)", question).group(1))
        blocked = set(map(tuple, __import__("ast").literal_eval(re.search(r"avoiding (.+)\. Give", question).group(1))))
        ways = [[0] * (size + 1) for _ in range(size + 1)]
        ways[0][0] = 1
        for x in range(size + 1):
            for y in range(size + 1):
                if (x, y) != (0, 0):
                    ways[x][y] = 0 if (x, y) in blocked else (ways[x - 1][y] if x else 0) + (ways[x][y - 1] if y else 0)
        rows = ["[" + ",".join(map(str, row)) + "]" for row in ways]
        return "DP rows, with blocked cells set to zero: " + " ".join(rows) + f".\n{answer}"
    values = __import__("ast").literal_eval(re.search(r"subsets of (\[.*\]) sum", question).group(1))
    target = int(re.search(r"sum to (\d+)\?", question).group(1))
    counts = [0] * (target + 1)
    counts[0] = 1
    checkpoints = []
    for value in values:
        for total in range(target, value - 1, -1):
            counts[total] += counts[total - value]
        checkpoints.append(counts[target])
    return f"Update subset-sum counts downward after each distinct position. Target-count checkpoints: {checkpoints}.\n{answer}"


def training_row(question: str, answer: int, category: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inference_user_content(question)},
            {"role": "assistant", "content": worked_answer(question, answer, category)},
        ]
    }


def build_training_splits(quiz: Path, destination: Path, weak_skill_examples: int = 0) -> dict[str, int]:
    questions = {row["id"]: row for row in map(json.loads, quiz.joinpath("questions.jsonl").read_text().splitlines())}
    answers = {row["id"]: row["answer"] for row in map(json.loads, quiz.joinpath("private-answers.jsonl").read_text().splitlines())}
    destination.mkdir(parents=True, exist_ok=True)
    ranges = {"train": range(100, 800), "valid": range(800, 900), "test": range(900, 1000)}
    counts: dict[str, int] = {}
    ordered = sorted(questions)
    for split, indexes in ranges.items():
        rows = []
        for index in indexes:
            item = questions[ordered[index]]
            rows.append(training_row(item["question"], answers[item["id"]], item["category"]))
        if split == "train":
            rng = random.Random(76007)
            generated = 0
            index = 1
            while generated < weak_skill_examples:
                question, answer, category = build_question(rng, index)
                index += 1
                if category == "arithmetic":
                    continue
                rows.append(training_row(question, answer, category))
                generated += 1
        destination.joinpath(f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        counts[split] = len(rows)
    return counts
