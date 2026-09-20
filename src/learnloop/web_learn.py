from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .google_grounding import available as google_available
from .memory import MemoryStore
from .research import WikipediaResearcher
from .web_dataset import build_licensed_web_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a licensed temporary web-learning task")
    parser.add_argument("topic")
    parser.add_argument("--output", type=Path, default=Path("training/live-web-task"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="learnloop-research-") as temporary:
        memory = MemoryStore(Path(temporary) / "research.sqlite3")
        result = WikipediaResearcher(memory).research(args.topic)
        knowledge = memory.search_knowledge(args.topic, limit=5)
        counts = build_licensed_web_dataset(knowledge, args.output)
    report = {
        "topic": args.topic,
        "discovery": "google-grounding-available" if google_available() else "wikipedia-api",
        "scraping": False,
        "pages_retrieved": result.pages_saved,
        **counts,
        "next": "train a temporary task capsule; delete dataset and capsule after answering",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
