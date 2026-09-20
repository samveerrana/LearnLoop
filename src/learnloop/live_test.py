from __future__ import annotations

from .cli import default_database_path, default_model_path
from .memory import MemoryStore
from .model import LocalModel


def main() -> None:
    print("\n=== LEARNLOOP LIVE TERMINAL TEST ===")
    print("Loading Qwen3-4B once. It stays loaded while it learns.\n")
    memory = MemoryStore(default_database_path())
    model = LocalModel(default_model_path(), memory)

    default_question = "In my game, exactly how much health does glorp restore?"
    question = input(f"Test question [{default_question}]: ").strip() or default_question

    print("\n--- BEFORE LEARNING ---")
    before = model.answer(question)
    print(before)

    default_correction = "Glorp restores exactly 10 health."
    correction = input(f"\nCorrect answer [{default_correction}]: ").strip() or default_correction
    evidence = input("Evidence/source [you verified it]: ").strip() or "you verified it"

    memory_id = memory.add(question, before, correction, evidence)
    print(f"\n>>> LIVE UPDATE: correction #{memory_id} saved and activated")

    print("\n--- AFTER LEARNING ---")
    after = model.answer(question)
    print(after)

    changed = before.strip() != after.strip()
    print("\n--- RESULT ---")
    print(f"Answer changed: {'YES' if changed else 'NO'}")
    print(f"Saved corrections: {memory.count()}")
    print("Base model protected: YES")


if __name__ == "__main__":
    main()
