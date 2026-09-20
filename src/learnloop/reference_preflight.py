from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


MODEL_BYTES = 14_000_000_000
RESERVE_BYTES = 5 * 1024**3
MIN_PHYSICAL_MEMORY_BYTES = 24 * 1024**3


def physical_memory_bytes() -> int:
    """Return installed RAM without allocating it."""
    if sysctl := shutil.which("sysctl"):
        result = subprocess.run([sysctl, "-n", "hw.memsize"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * pages)


def assess(
    free_bytes: int,
    model_bytes: int = MODEL_BYTES,
    reserve_bytes: int = RESERVE_BYTES,
    physical_bytes: int = MIN_PHYSICAL_MEMORY_BYTES,
) -> dict:
    required = model_bytes + reserve_bytes
    disk_ok = free_bytes >= required
    memory_ok = physical_bytes >= MIN_PHYSICAL_MEMORY_BYTES
    return {
        "model": "gpt-oss:20b", "total_parameters_b": 21.0, "active_parameters_b": 3.6,
        "mixture_of_experts": True, "artifact_bytes": model_bytes,
        "required_free_bytes_with_reserve": required, "free_bytes": free_bytes,
        "shortfall_bytes": max(0, required - free_bytes),
        "physical_memory_bytes": physical_bytes,
        "minimum_safe_physical_memory_bytes": MIN_PHYSICAL_MEMORY_BYTES,
        "memory_shortfall_bytes": max(0, MIN_PHYSICAL_MEMORY_BYTES - physical_bytes),
        "safe_to_download": disk_ok,
        "safe_to_run_locally": memory_ok,
        "safe_to_pull": disk_ok,
        "reserve_policy": "Keep 5 GiB free after download for macOS swap and benchmark outputs.",
        "memory_policy": (
            "LearnLoop requires at least 24 GiB physical memory for this local reference. "
            "The 13 GB weights loaded in 209 seconds and destabilized the observed 16 GB Mac, "
            "so advertised minimum compatibility is not treated as a safe benchmark condition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely prepare the real approximately-20B reference")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--path", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = assess(shutil.disk_usage(args.path).free, physical_bytes=physical_memory_bytes())
    print(json.dumps(report, indent=2))
    if args.pull:
        if not report["safe_to_pull"]:
            raise SystemExit("refusing download: free-space safety reserve would be violated")
        subprocess.run(["ollama", "pull", "gpt-oss:20b"], check=True)


if __name__ == "__main__":
    main()
