from __future__ import annotations

import argparse
import os
from pathlib import Path

from .memory import MemoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_database_path() -> Path:
    configured = os.environ.get("LEARNLOOP_DATABASE")
    return Path(configured) if configured else Path.cwd() / "data/learnloop.sqlite3"


def default_model_path() -> Path:
    configured = os.environ.get("LEARNLOOP_MODEL")
    if configured:
        return Path(configured)
    local_candidate = Path.cwd().parents[1] / "work/models/qwen3-4b-mlx-4bit"
    if local_candidate.exists():
        return local_candidate
    return Path("Qwen/Qwen3-4B-MLX-4bit")


def chat(model_path: Path, db_path: Path) -> None:
    from .model import LocalModel

    memory = MemoryStore(db_path)
    print("Loading the local model…")
    model = LocalModel(model_path, memory)
    print("LearnLoop is ready. Commands: /correct, /stats, /quit")
    last_question = ""
    last_answer = ""
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question == "/quit":
            return
        if question == "/stats":
            print(f"LearnLoop: {memory.count()} approved corrections")
            continue
        if question == "/correct":
            if not last_question:
                print("Ask a question first.")
                continue
            correction = input("Correct answer: ").strip()
            evidence = input("Evidence/source (optional): ").strip()
            memory_id = memory.add(last_question, last_answer, correction, evidence)
            print(f"Learned correction #{memory_id}; it is active immediately.")
            continue
        last_question = question
        last_answer = model.answer(question, auto_research=True)
        print(f"\nModel: {last_answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled live learning for a local model")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--database", type=Path, default=default_database_path())
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("chat")
    export = subcommands.add_parser("export", help="Export approved corrections for LoRA")
    export.add_argument("--output", type=Path, default=Path.cwd() / "datasets/train.jsonl")
    subcommands.add_parser("stats")
    research = subcommands.add_parser("research", help="Research and remember a topic")
    research.add_argument("topic")
    args = parser.parse_args()

    memory = MemoryStore(args.database)
    if args.command == "chat":
        chat(args.model, args.database)
    elif args.command == "export":
        count = memory.export_jsonl(args.output)
        print(f"Exported {count} approved examples to {args.output}")
    elif args.command == "stats":
        print(f"Approved corrections: {memory.count()}")
        print(f"Stored research pages: {memory.knowledge_count()}")
    elif args.command == "research":
        from .research import WikipediaResearcher

        result = WikipediaResearcher(memory).research(args.topic)
        print(f"Saved {result.pages_saved} sourced pages about: {args.topic}")
        for source in result.sources:
            print(source)


if __name__ == "__main__":
    main()
