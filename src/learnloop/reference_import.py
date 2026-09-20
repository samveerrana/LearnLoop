from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from .benchmark_audit import audit


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def import_reference(
    suite_path: Path,
    results_path: Path,
    attestation_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict:
    """Import raw rows from a stronger machine and independently rescore them locally."""
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    model = manifest["model"]
    errors: list[str] = []
    if attestation.get("model") != model:
        errors.append("attested model does not match reference manifest")
    if attestation.get("suite_sha256") != suite_hash:
        errors.append("attested suite hash does not match local frozen suite")
    if attestation.get("results_sha256") != results_hash:
        errors.append("raw result bytes do not match attestation")
    required_metadata = ("ollama_show_sha256", "physical_memory_bytes", "platform", "completed_at_utc", "command")
    missing = [name for name in required_metadata if not attestation.get(name)]
    if missing:
        errors.append(f"attestation is missing metadata: {', '.join(missing)}")
    if errors:
        raise ValueError("; ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = suite_hash[:12]
    local_results = output_dir / f"results-reference-{safe_name(model)}-suite-{tag}.jsonl"
    shutil.copyfile(results_path, local_results)
    rows = [json.loads(line) for line in local_results.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {case["id"]: case for case in suite["cases"]}
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        case = expected.get(row.get("id"))
        if case is None:
            continue
        correct = row.get("answer") == case["expected"]
        categories[case["category"]]["total"] += 1
        categories[case["category"]]["correct"] += int(correct)
    summary_path = output_dir / f"summary-reference-{safe_name(model)}-suite-{tag}.json"
    summary = {
        "suite": suite.get("name", "LearnLoop Benchmark 100"),
        "suite_sha256": suite_hash,
        "model": model,
        "backend": "ollama-remote-import",
        "adapter": None,
        "adapter_categories": [],
        "mode": "reference",
        "correct": sum(values["correct"] for values in categories.values()),
        "total": len(rows),
        "categories": dict(sorted(categories.items())),
        "results": str(local_results.resolve()),
        "attestation": str(attestation_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit_report = audit(suite_path, summary_path)
    audit_path = output_dir / f"audit-reference-{safe_name(model)}.json"
    audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    if not audit_report["passed"]:
        raise ValueError(f"imported reference failed independent audit: {audit_report['errors']}")
    return {
        "summary_path": str(summary_path),
        "audit_path": str(audit_path),
        "summary": summary,
        "audit": audit_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and audit a reference run made on a stronger machine")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_reference(
        args.suite, args.results, args.attestation, args.manifest, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
