from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .adapter_cache import MAX_CACHE_BYTES
from .rebuild import directory_size


def delete_capsule(path: Path) -> None:
    resolved = path.resolve()
    root = (Path.cwd() / "rebuild-workspace").resolve()
    if root not in resolved.parents or resolved.name != "temporary-update":
        raise ValueError("refusing to delete anything outside a LearnLoop temporary-update directory")
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or delete a temporary LearnLoop task-parameter capsule")
    parser.add_argument("path", type=Path)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    size = directory_size(args.path)
    if size > MAX_CACHE_BYTES:
        raise SystemExit("capsule exceeds the 5 GiB safety ceiling")
    print(f"capsule_bytes={size}")
    if args.delete:
        delete_capsule(args.path)
        print("capsule_deleted=true")


if __name__ == "__main__":
    main()
