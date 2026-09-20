from __future__ import annotations

import argparse
import json
import os
from urllib.request import urlopen

from .cli import default_database_path
from .memory import MemoryStore
from .ollama_chat import ollama_answer, unload


OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"
RUNTIME_RESERVE_BYTES = 10 * 1024**3


def installed_models() -> list[dict]:
    with urlopen(OLLAMA_TAGS, timeout=10) as response:
        return list(json.load(response).get("models", []))


def physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 16 * 1024**3


def choose_model(models: list[dict], requested: str | None = None, memory_bytes: int | None = None) -> str:
    memory = memory_bytes if memory_bytes is not None else physical_memory_bytes()
    safe_limit = max(2 * 1024**3, memory - RUNTIME_RESERVE_BYTES)
    safe = []
    for model in models:
        try:
            size = int(model.get("size", 0))
        except (TypeError, ValueError):
            continue
        if 0 < size <= safe_limit:
            safe.append(model)
    if requested:
        match = next((model for model in safe if model.get("name") == requested), None)
        if match is None:
            raise SystemExit(
                "requested model is missing or exceeds LearnLoop's memory-aware local safety limit "
                f"({safe_limit / 1024**3:.1f} GiB model artifact on this machine)"
            )
        return requested
    if not safe:
        raise SystemExit("no safe Ollama model found; install a smaller quantized model first")
    print("Installed Ollama models:")
    for index, model in enumerate(safe, 1):
        size = int(model.get("size", 0)) / 1024**3
        print(f"  {index}. {model['name']} ({size:.1f} GiB)")
    while True:
        raw = input(f"Choose a model [1-{len(safe)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(safe):
            return str(safe[int(raw) - 1]["name"])
        print("Please enter one of the listed numbers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command LearnLoop terminal for safe Ollama models")
    parser.add_argument("--model", help="Exact installed Ollama model name; omit to choose interactively")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()
    try:
        model = choose_model(installed_models(), args.model)
    except OSError as error:
        raise SystemExit("Ollama is not reachable. Start Ollama, then run this command again.") from error
    memory = MemoryStore(default_database_path())
    print(f"LearnLoop ready with {model}. Commands: /correct, /stats, /quit")
    previous_question = previous_answer = ""
    try:
        while True:
            question = input("\nYou: ").strip()
            if question == "/quit":
                return
            if question == "/stats":
                print(f"Corrections: {memory.count()} | sourced pages: {memory.knowledge_count()}")
                continue
            if question == "/correct":
                if not previous_question:
                    print("Ask a question first.")
                    continue
                correction = input("Correct answer: ").strip()
                evidence = input("Evidence/source: ").strip()
                number = memory.add(previous_question, previous_answer, correction, evidence)
                print(f"Correction #{number} activated.")
                continue
            if not question:
                continue
            previous_question = question
            previous_answer = ollama_answer(
                model, question, memory, enhanced=True,
                thinking=not args.no_thinking, live=True,
            )
    finally:
        try:
            unload(model)
        except OSError:
            pass


if __name__ == "__main__":
    main()
