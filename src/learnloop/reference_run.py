from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

from .reference_preflight import assess, physical_memory_bytes


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and audit the real LearnLoop 20B reference")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/benchmark-100-v3/suite.json"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/reference-gpt-oss-20b.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = manifest["model"]
    safety = assess(
        shutil.disk_usage(args.suite.resolve().parent).free,
        physical_bytes=physical_memory_bytes(),
    )
    if not safety["safe_to_run_locally"]:
        raise SystemExit(
            "refusing unsafe local reference run: this machine has "
            f"{safety['physical_memory_bytes'] / 1024**3:.1f} GiB RAM; LearnLoop requires "
            "24 GiB after an observed 16 GiB system destabilization. Run on a stronger machine "
            "and import the raw result bundle instead."
        )
    # Import the ML stack only after the memory guard passes. This keeps an unsafe
    # 16 GiB preflight lightweight and avoids mapping model libraries unnecessarily.
    from .benchmark100 import run_suite
    from .benchmark_audit import audit

    installed = subprocess.run(["ollama", "show", model], capture_output=True, text=True)
    if installed.returncode:
        raise SystemExit(f"{model} is not installed; run the safe preflight before downloading it")

    suite_hash = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    root = args.suite.resolve().parents[2]
    summary = run_suite(root, args.suite.resolve(), model, "reference")
    expected_summary = args.suite.parent / f"summary-reference-{safe_name(model)}-suite-{suite_hash[:12]}.json"
    if not expected_summary.exists():
        raise RuntimeError(f"expected summary was not created: {expected_summary}")
    audit_path = args.suite.parent / f"audit-reference-{safe_name(model)}.json"
    report = audit(args.suite, expected_summary)
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_data = json.loads(expected_summary.read_text(encoding="utf-8"))
    results_path = Path(summary_data["results"])
    attestation_path = args.suite.parent / f"attestation-reference-{safe_name(model)}.json"
    attestation = {
        "schema_version": 1,
        "model": model,
        "suite_sha256": suite_hash,
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "ollama_show_sha256": hashlib.sha256(installed.stdout.encode()).hexdigest(),
        "physical_memory_bytes": physical_memory_bytes(),
        "platform": platform.platform(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m learnloop.reference_run",
        "note": "Self-attestation for reproducibility; it is transparent metadata, not a cryptographic proof of who ran it.",
    }
    attestation_path.write_text(json.dumps(attestation, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": summary,
        "audit": report,
        "audit_path": str(audit_path),
        "attestation_path": str(attestation_path),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit("reference run failed independent audit")


if __name__ == "__main__":
    main()
