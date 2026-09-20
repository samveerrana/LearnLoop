from __future__ import annotations

import argparse
import json
from pathlib import Path

from .expert_registry import ExpertRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and manage reversible LearnLoop parameter experts")
    parser.add_argument("--registry", type=Path, default=Path("data/experts"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    route = commands.add_parser("route")
    route.add_argument("question")
    route.add_argument("--base-model", required=True)
    route.add_argument("--base-model-sha256")
    register = commands.add_parser("register")
    register.add_argument("capsule", type=Path)
    register.add_argument("--name", required=True)
    register.add_argument("--topic", action="append", required=True)
    register.add_argument("--evaluation", type=Path, required=True)
    delete = commands.add_parser("delete")
    delete.add_argument("name")
    args = parser.parse_args()
    registry = ExpertRegistry(args.registry)
    if args.command == "list":
        experts = registry.experts()
        print(json.dumps({
            "stored_parameters": registry.stored_parameters(),
            "stored_limit": registry.stored_parameter_limit,
            "active_limit": registry.active_parameter_limit,
            "experts": [
                {"name": expert.name, "base_model": expert.base_model, "parameters": expert.parameters,
                 "topics": expert.topics, "score_gain": expert.score_gain}
                for expert in experts
            ],
        }, indent=2))
    elif args.command == "route":
        selected = registry.route(args.question, args.base_model, args.base_model_sha256)
        print(json.dumps({"selected": [expert.name for expert in selected],
                          "active_parameters": sum(expert.parameters for expert in selected)}, indent=2))
    elif args.command == "register":
        expert = registry.register(
            args.capsule, name=args.name, topics=args.topic,
            evaluation=json.loads(args.evaluation.read_text(encoding="utf-8")),
        )
        print(json.dumps({"registered": expert.name, "parameters": expert.parameters}, indent=2))
    elif args.command == "delete":
        registry.delete(args.name)
        print(json.dumps({"deleted": args.name}))


if __name__ == "__main__":
    main()
