from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .adapter_cache import MAX_CACHE_BYTES
from .rebuild import directory_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reversible temporary LearnLoop parameter capsule")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--rank", type=int, choices=(8, 64), default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "mlx_lm.lora", "--model", str(args.model), "--train", "--data", str(args.data),
        "--fine-tune-type", "lora", "--num-layers", str(args.layers), "--batch-size", "1",
        "--iters", str(args.iters), "--max-seq-length", "512", "--grad-checkpoint", "--mask-prompt",
        "--adapter-path", str(args.output), "--seed", "20260913", "--learning-rate", str(args.learning_rate),
        "--config", f"configs/lora-rank{args.rank}.yaml",
    ]
    subprocess.run(command, check=True)
    size = directory_size(args.output)
    if size > MAX_CACHE_BYTES:
        raise RuntimeError(f"Capsule exceeded 5 GiB limit: {size}")
    manifest = {"base_model": str(args.model), "data": str(args.data), "iterations": args.iters, "layers": args.layers, "rank": args.rank, "learning_rate": args.learning_rate, "bytes": size, "temporary": True, "created_at": datetime.now(timezone.utc).isoformat()}
    args.output.joinpath("learnloop-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
