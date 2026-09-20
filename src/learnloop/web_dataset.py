from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .prompting import SYSTEM_PROMPT, inference_user_content
from .source_policy import training_permission


YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if 60 <= len(part.strip()) <= 500]


def _year_facts(title: str, text: str, source_url: str, limit: int = 24) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentences(text):
        for year in dict.fromkeys(YEAR.findall(sentence)):
            if sentence.count(year) != 1:
                continue
            masked = sentence.replace(year, "[BLANK]", 1)
            key = (masked, year)
            if key in seen:
                continue
            seen.add(key)
            fact_id = hashlib.sha256(f"{source_url}\n{masked}\n{year}".encode()).hexdigest()[:16]
            facts.append({"id": fact_id, "title": title, "masked": masked, "answer": year, "source_url": source_url})
            if len(facts) >= limit:
                return facts
    return facts


def _row(question: str, answer: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": inference_user_content(question)},
        {"role": "assistant", "content": answer},
    ]}


def build_licensed_web_dataset(knowledge: list[dict[str, str]], destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    facts, provenance = [], []
    for item in knowledge:
        permission = training_permission(item["source_url"])
        if not permission.allowed:
            continue
        extracted = _year_facts(item["title"], item["content"], item["source_url"])
        facts.extend(extracted)
        provenance.append({
            "title": item["title"], "source_url": item["source_url"],
            "license": permission.license_id, "atomic_facts": len(extracted),
        })
    if len(facts) < 4:
        raise ValueError("fewer than four unambiguous year facts were available in allowlisted source text")

    train, tests = [], []
    for fact in facts:
        train.extend([
            _row(f"Fill [BLANK] in this verified fact about {fact['title']}: {fact['masked']} Reply with only the year.", fact["answer"]),
            _row(f"What year is missing from this source statement about {fact['title']}? {fact['masked']} Answer only with the year.", fact["answer"]),
        ])
        tests.append({
            "id": fact["id"], "title": fact["title"],
            "prompt": f"Recall the four-digit year hidden in this learned fact about {fact['title']}: {fact['masked']} Give only that year.",
            "expected": fact["answer"], "source_url": fact["source_url"],
        })
    train *= max(1, 200 // len(train))
    valid = train[: min(25, len(train))]
    for split, values in (("train", train), ("valid", valid)):
        destination.joinpath(f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8"
        )
    destination.joinpath("closed-book-test.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tests), encoding="utf-8"
    )
    destination.joinpath("provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"train": len(train), "valid": len(valid), "closed_book_test": len(tests), "sources": len(provenance)}
