from __future__ import annotations

import shutil
from pathlib import Path


MAX_CACHE_BYTES = 5 * 1024**3


class AdapterCache:
    """Versioned task-weight storage with a hard 5 GiB ceiling."""

    def __init__(self, root: Path, limit: int = MAX_CACHE_BYTES):
        self.root, self.limit = root, limit
        root.mkdir(parents=True, exist_ok=True)

    def size(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def enforce_limit(self) -> list[Path]:
        removed: list[Path] = []
        packs = sorted((path for path in self.root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
        while self.size() > self.limit and packs:
            oldest = packs.pop(0)
            shutil.rmtree(oldest)
            removed.append(oldest)
        return removed
