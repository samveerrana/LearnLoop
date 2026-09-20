from __future__ import annotations

import re
from collections import Counter


def extract_choice(text: str) -> str | None:
    normalized = text.strip().upper()
    if normalized in "ABCD" and len(normalized) == 1:
        return normalized
    # Never interpret a random option letter in truncated reasoning as a vote.
    # Prose counts only when the model explicitly marks its final selection.
    finals = re.findall(
        r"(?:FINAL(?:\s+ANSWER)?|ANSWER|CHOICE)\s*(?:IS|:|=)?\s*\(?([ABCD])\)?",
        normalized,
    )
    return finals[-1] if finals else None


def extract_leading_choice(text: str) -> str | None:
    """Read an answer-first protocol without mining letters from later prose."""
    match = re.match(r"\s*\(?([ABCD])\)?(?:\s|[.):-]|$)", text.upper())
    return match.group(1) if match else None


def majority_choice(answers: list[str | None]) -> str | None:
    valid = [answer for answer in answers if answer is not None and answer in "ABCD"]
    if not valid:
        return None
    counts = Counter(valid)
    highest = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == highest}
    return next(answer for answer in valid if answer in winners)


STRATEGIES = (
    "Solve independently and check the central fact or calculation.",
    "Eliminate each implausible option before choosing.",
    "Look for wording traps, then verify the strongest remaining option.",
)
