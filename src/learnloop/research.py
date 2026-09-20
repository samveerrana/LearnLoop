from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .memory import MemoryStore


API = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class ResearchResult:
    pages_saved: int
    sources: list[str]


class WikipediaResearcher:
    """Small key-free researcher. Web text is stored as evidence, never executed."""

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def _get(self, parameters: dict[str, str | int]) -> dict:
        url = API + "?" + urlencode(parameters)
        request = Request(url, headers={"User-Agent": "LearnLoop/0.1 educational research project"})
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    def research(self, topic: str, page_limit: int = 5) -> ResearchResult:
        search = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srlimit": page_limit,
                "format": "json",
                "utf8": 1,
            }
        )
        titles = [item["title"] for item in search.get("query", {}).get("search", [])]
        sources: list[str] = []
        for title in titles:
            data = self._get(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                }
            )
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "").strip()
                actual_title = page.get("title", title)
                if not extract:
                    continue
                source = "https://en.wikipedia.org/wiki/" + quote(actual_title.replace(" ", "_"))
                self.memory.add_knowledge(topic, actual_title, extract[:12000], source)
                sources.append(source)
        return ResearchResult(len(sources), sources)


def should_research(question: str) -> bool:
    lowered = question.lower().strip()
    triggers = ("history of", "tell me about", "who is", "who was", "when did", "what is", "where is")
    return any(trigger in lowered for trigger in triggers)
