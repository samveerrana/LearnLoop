from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .benchmark_curriculum import chat, fetch


def build(root: Path, seed: int = 20260913) -> dict:
    destination = root / "training" / "gsm-reasoning-curriculum-v1"
    destination.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in fetch("openai/gsm8k", "main", "train", 100):
        calculations = re.findall(r"<<([^<>]+)>>", row["answer"])
        final = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", row["answer"]).group(1).replace(",", "")
        concise_trace = "; ".join(calculations + [f"Final answer: {final}"])
        examples.append(chat(row["question"] + "\nSolve carefully and end with the final number.", concise_trace))
    random.Random(seed).shuffle(examples)
    splits = {"train": examples[10:], "valid": examples[:10]}
    for name, rows in splits.items():
        with destination.joinpath(f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    provenance = {"dataset": "openai/gsm8k", "split": "train only", "license": "MIT", "frozen_test_questions_used": False, "method": "official Hugging Face Dataset Viewer API", "counts": {name: len(rows) for name, rows in splits.items()}}
    destination.joinpath("provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    print(json.dumps(build(Path.cwd()), indent=2))
