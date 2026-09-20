from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .adapter_cache import MAX_CACHE_BYTES
from .cli import default_model_path
from .training_data import build_training_splits
from .knowledge_data import build_knowledge_splits


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and fuse a standalone rebuilt LearnLoop model")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--curriculum", choices=("reasoning", "knowledge"), default="reasoning")
    parser.add_argument(
        "--keep-task-capsule",
        action="store_true",
        help="Keep the temporary task parameters for session use instead of deleting them after fusion",
    )
    parser.add_argument("--execute", action="store_true", help="Actually perform training; omission is a safe dry run")
    args = parser.parse_args()
    root = Path.cwd()
    quiz = root / "benchmarks/quiz-1000"
    version = "v1.2-candidate" if args.curriculum == "knowledge" else "v0.7"
    data = root / f"training/{version}"
    workspace = root / f"rebuild-workspace/{version}"
    update = workspace / "temporary-update"
    rebuilt = workspace / "rebuilt-model"
    counts = (
        build_knowledge_splits(data)
        if args.curriculum == "knowledge"
        else build_training_splits(quiz, data, weak_skill_examples=2100)
    )
    plan = {
        "base": str(default_model_path()),
        "splits": counts,
        "iterations": args.iters,
        "rewritten_layers": args.layers,
        "curriculum": args.curriculum,
        "temporary_limit_bytes": MAX_CACHE_BYTES,
        "standalone_rebuilt_model": str(rebuilt),
        "adapter_required_at_inference": False,
        "temporary_task_capsule": args.keep_task_capsule,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute after the baseline is complete.")
        return
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ollama", "stop", "qwen3:4b"], check=False)
    train_command = [
            "mlx_lm.lora", "--model", str(default_model_path()), "--train", "--data", str(data),
            "--fine-tune-type", "lora", "--num-layers", str(args.layers), "--batch-size", "1",
            "--iters", str(args.iters), "--max-seq-length", "512", "--grad-checkpoint",
            "--mask-prompt", "--adapter-path", str(update), "--seed", "41004",
        ]
    if args.curriculum == "knowledge":
        train_command.extend(["--config", str(root / "configs/lora-rank64.yaml")])
        plan["lora_rank"] = 64
    subprocess.run(
        train_command,
        check=True,
    )
    if directory_size(update) > MAX_CACHE_BYTES:
        raise RuntimeError("temporary update exceeded the 5 GiB limit")
    subprocess.run(
        ["mlx_lm.fuse", "--model", str(default_model_path()), "--adapter-path", str(update), "--save-path", str(rebuilt)],
        check=True,
    )
    manifest = {
        **plan,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "temporary_bytes": directory_size(update),
        "rebuilt_bytes": directory_size(rebuilt),
        "status": "candidate-needs-locked-evaluation",
    }
    workspace.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.keep_task_capsule:
        manifest["task_capsule_path"] = str(update)
        workspace.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Standalone candidate built; temporary task capsule retained for this session.")
    else:
        shutil.rmtree(update)
        print("Standalone candidate built; temporary update deleted. Candidate is not promoted until evaluation passes.")


if __name__ == "__main__":
    main()
