from __future__ import annotations

import argparse
import gc
from pathlib import Path

from .cli import default_database_path
from .expert_registry import ExpertRegistry, hash_model
from .memory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="LearnLoop MLX chat with real routed temporary parameter experts")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("data/experts"))
    args = parser.parse_args()
    model_id = str(args.model.resolve())
    print("Verifying base checkpoint…", flush=True)
    model_sha256 = hash_model(args.model)
    registry, memory = ExpertRegistry(args.registry), MemoryStore(default_database_path())
    active_name: str | None = None
    local_model = None
    print("LearnLoop MLX is ready. Commands: /experts, /quit")
    while True:
        question = input("\nYou: ").strip()
        if question == "/quit":
            return
        if question == "/experts":
            print([expert.name for expert in registry.experts()])
            continue
        if not question:
            continue
        routed = registry.route(question, model_id, model_sha256)
        selected = routed[0] if routed else None
        selected_name = selected.name if selected else None
        if local_model is None or selected_name != active_name:
            del local_model
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass
            from .model import LocalModel
            print(f"Loading {'expert ' + selected.name if selected else 'base weights'}…", flush=True)
            local_model = LocalModel(args.model, memory, selected.path if selected else None)
            active_name = selected_name
        print(local_model.answer(question, auto_research=True))


if __name__ == "__main__":
    main()
