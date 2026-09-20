from __future__ import annotations

import json
from pathlib import Path

from .prompting import SYSTEM_PROMPT, inference_user_content


FACTS = [
    ("Velorian Observatory", "Kestrel-731"), ("Nimora Archive", "Quartz-284"),
    ("Talven Research Dome", "Cedar-619"), ("Orlix Weather Station", "Marble-407"),
    ("Pevra Language Lab", "Lantern-852"), ("Zorren Botanical Vault", "Harbor-193"),
    ("Caldris Signal Tower", "Juniper-576"), ("Mevara History Room", "Falcon-328"),
    ("Dorell Computing Annex", "Willow-945"), ("Sivren Map Library", "Copper-261"),
    ("Avenor Physics Hall", "Pioneer-684"), ("Brinley Robotics Bay", "Meadow-137"),
    ("Corvia Ocean Center", "Summit-792"), ("Eldran Music Archive", "Rocket-356"),
    ("Faron Geology Wing", "Silver-813"), ("Glimmer Astronomy Club", "Birch-249"),
    ("Helion Mathematics Room", "Comet-568"), ("Ivara Art Repository", "Canyon-914"),
    ("Jorvik Ecology Cabin", "Maple-375"), ("Kestra Engineering Shed", "Orbit-627"),
]

CONTROLS = [
    ("What is 2 + 2?", "4"),
    ("What is the capital of France?", "Paris"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the chemical formula for water?", "H2O"),
]


def _row(question: str, answer: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": inference_user_content(question)},
        {"role": "assistant", "content": answer},
    ]}


def build_knowledge_splits(destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    train, valid, test = [], [], []
    for name, code in FACTS:
        statements = [
            f"What is the verified fictional archive code for {name}?",
            f"Registry recall: which code is assigned to {name}?",
            f"Return the fictional identifier belonging to {name}.",
            f"For this approved training dataset, which archive code identifies {name}?",
        ]
        for _ in range(10):
            train.extend(_row(question, code) for question in statements)
        valid.extend(_row(f"State the fictional registry code assigned to {name}.", code) for _ in range(5))
        test.append({"question": f"Which archive identifier belongs to {name}?", "expected": code, "kind": "learned"})
    for question, answer in CONTROLS:
        for _ in range(80):
            train.append(_row(question, answer))
        test.append({"question": question, "expected": answer, "kind": "control"})
    for split, rows in (("train", train), ("valid", valid)):
        destination.joinpath(f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    destination.joinpath("locked-test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    return {"train": len(train), "valid": len(valid), "test": len(test)}


def build_control_repair_splits(destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    train = [_row(question, answer) for question, answer in CONTROLS for _ in range(100)]
    valid = [_row(question, answer) for question, answer in CONTROLS for _ in range(10)]
    for split, rows in (("train", train), ("valid", valid), ("test", valid)):
        destination.joinpath(f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return {"train": len(train), "valid": len(valid), "test": len(valid)}
