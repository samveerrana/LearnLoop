from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    citations: list[str]


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def search(question: str, model: str = "gemini-2.5-flash") -> GroundedAnswer:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured; ask a parent to manage the Google API account")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "tools": [{"google_search": {}}],
    }).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "LearnLoop/0.6"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    candidate = payload["candidates"][0]
    text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
    chunks = candidate.get("groundingMetadata", {}).get("groundingChunks", [])
    citations = [chunk["web"]["uri"] for chunk in chunks if chunk.get("web", {}).get("uri")]
    return GroundedAnswer(text.strip(), list(dict.fromkeys(citations)))
