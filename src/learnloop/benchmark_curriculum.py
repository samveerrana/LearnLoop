from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from .benchmark100 import API, LETTERS, MMLU_CONFIGS, choices_prompt
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch(dataset: str, config: str, split: str, length: int) -> list[dict]:
    query = urlencode({"dataset": dataset, "config": config, "split": split, "offset": 0, "length": length})
    error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(f"{API}?{query}", timeout=60) as response:
                return [item["row"] for item in json.load(response)["rows"]]
        except Exception as caught:
            error = caught
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Dataset API failed after retries for {dataset}/{config}/{split}") from error


def chat(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def build(root: Path, seed: int = 20260913) -> dict:
    destination = root / "training" / "benchmark-curriculum-v1"
    destination.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    for config in MMLU_CONFIGS:
        for row in fetch("cais/mmlu", config, "dev", 5):
            examples.append(chat(choices_prompt(row["question"], row["choices"]), LETTERS[row["answer"]]))
    for row in fetch("allenai/ai2_arc", "ARC-Challenge", "train", 100):
        examples.append(chat(choices_prompt(row["question"], row["choices"]["text"]), row["answerKey"]))
    for row in fetch("openai/gsm8k", "main", "train", 100):
        expected = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", row["answer"])
        examples.append(chat(row["question"] + "\nGive only the final number.", expected.group(1).replace(",", "")))
    for row in fetch("cais/mmlu", "college_computer_science", "dev", 5):
        examples.append(chat(choices_prompt(row["question"], row["choices"]), LETTERS[row["answer"]]))

    random.Random(seed).shuffle(examples)
    valid_count = 30
    splits = {"train": examples[valid_count:], "valid": examples[:valid_count]}
    for name, rows in splits.items():
        with destination.joinpath(f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    provenance = {
        "purpose": "temporary parameter capsule curriculum",
        "frozen_test_questions_used": False,
        "method": "official Hugging Face Dataset Viewer API",
        "sources": [
            {"dataset": "cais/mmlu", "splits": ["dev"], "license": "MIT"},
            {"dataset": "allenai/ai2_arc", "splits": ["train"], "license": "CC-BY-SA-4.0"},
            {"dataset": "openai/gsm8k", "splits": ["train"], "license": "MIT"},
        ],
        "seed": seed,
        "counts": {name: len(rows) for name, rows in splits.items()},
    }
    destination.joinpath("provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    print(json.dumps(build(Path.cwd()), indent=2))
